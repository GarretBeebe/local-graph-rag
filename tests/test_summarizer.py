"""Unit tests for community summarizer. No Ollama or Qdrant required."""

import pytest

from graph.store import GraphStore
from graph.summarizer import (
    _build_summary_prompt,
    _compute_member_hash,
    summarize_community,
)


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


def _add_entity_in_community(store: GraphStore, name: str, community: int) -> str:
    """Insert an entity and assign it to a community, bypassing detect_communities."""
    slug = store.upsert_entity(name, type="TYPE", description="test entity")
    store._conn.execute("UPDATE entities SET community = ? WHERE id = ?", (community, slug))
    store._conn.commit()
    return slug


_ZERO_EMBEDDING = b"\x00" * (768 * 4)


def test_member_hash_order_independent():
    entities = [
        {"id": "a", "type": "T", "description": "desc-a"},
        {"id": "b", "type": "T", "description": "desc-b"},
    ]
    relationships = [{"source_id": "a", "target_id": "b", "label": "uses"}]
    h1 = _compute_member_hash(entities, relationships)
    h2 = _compute_member_hash(list(reversed(entities)), list(reversed(relationships)))
    assert h1 == h2


def test_member_hash_changes_with_membership():
    base = [{"id": "a", "type": "T", "description": "desc-a"}]
    extra = base + [{"id": "b", "type": "T", "description": "desc-b"}]
    assert _compute_member_hash(base, []) != _compute_member_hash(extra, [])


def test_member_hash_changes_with_entity_metadata():
    e1 = [{"id": "a", "type": "T", "description": "old description"}]
    e2 = [{"id": "a", "type": "T", "description": "new, richer description"}]
    assert _compute_member_hash(e1, []) != _compute_member_hash(e2, [])


def test_member_hash_changes_with_relationship_topology():
    entities = [
        {"id": "a", "type": "T", "description": "desc-a"},
        {"id": "b", "type": "T", "description": "desc-b"},
    ]
    r1 = [{"source_id": "a", "target_id": "b", "label": "uses"}]
    r2 = [{"source_id": "a", "target_id": "b", "label": "extends"}]
    assert _compute_member_hash(entities, r1) != _compute_member_hash(entities, r2)
    assert _compute_member_hash(entities, r1) != _compute_member_hash(entities, [])


def test_summarize_community_skips_empty_community(store):
    assert summarize_community(99, store) is False


def test_summarize_community_skips_unchanged(store, monkeypatch):
    slug = _add_entity_in_community(store, "Alpha", 0)
    entities = store.get_entities_for_community(0)
    relationships = store.get_relationships_for_community(0)
    member_hash = _compute_member_hash(entities, relationships)
    store.upsert_community(0, "existing summary", [slug], member_hash, _ZERO_EMBEDDING)

    generate_calls = []

    def _fake_generate(*a, **kw):
        generate_calls.append(a)
        return "x"

    monkeypatch.setattr("api.ollama_client.generate", _fake_generate)
    monkeypatch.setattr("graph.summarizer.embed", lambda *a, **kw: [0.0] * 768)

    assert summarize_community(0, store) is False
    assert len(generate_calls) == 0


def test_summarize_community_regenerates_on_membership_change(store, monkeypatch):
    slug = _add_entity_in_community(store, "Beta", 0)
    store.upsert_community(0, "old summary", [slug], "stale_hash" * 4, _ZERO_EMBEDDING)

    monkeypatch.setattr("api.ollama_client.generate", lambda *a, **kw: "new summary")
    monkeypatch.setattr("graph.summarizer.embed", lambda *a, **kw: [0.1] * 768)

    assert summarize_community(0, store) is True
    communities = store.get_communities()
    assert len(communities) == 1
    assert communities[0]["summary"] == "new summary"


def test_delete_stale_communities_removes_old_rows(store):
    # Seed 3 communities, but only give entities to communities 1 and 3.
    for i, h in enumerate(["h1", "h2", "h3"], start=1):
        store.upsert_community(i, f"summary {i}", [f"e{i}"], h, _ZERO_EMBEDDING)
    _add_entity_in_community(store, "EntityOne", 1)
    _add_entity_in_community(store, "EntityThree", 3)

    store.delete_stale_communities()

    remaining = {c["id"] for c in store.get_communities()}
    assert remaining == {1, 3}


def test_build_summary_prompt_contains_entities_and_relationships():
    entities = [{"id": "alpha", "name": "Alpha", "type": "CLASS", "description": "does stuff"}]
    relationships = [{"source_id": "alpha", "target_id": "beta", "label": "calls", "weight": 1.0}]
    prompt = _build_summary_prompt(entities, relationships)
    assert "Alpha" in prompt
    assert "CLASS" in prompt
    assert "calls" in prompt
    assert "beta" in prompt
