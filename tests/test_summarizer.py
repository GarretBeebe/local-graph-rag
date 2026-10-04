"""Unit tests for community summarizer. No Ollama or Qdrant required."""

import pytest

from local_graph_rag.graph.store import GraphStore
from local_graph_rag.graph.summarizer import (
    _build_summary_prompt,
    _compute_member_hash,
    summarize_all_communities,
    summarize_community,
)
from tests.helpers import add_entity, add_relationship, patch_ollama_generate


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


def _add_entity_in_community(store: GraphStore, name: str, community: int) -> str:
    """Insert an entity and assign it to a community, bypassing detect_communities."""
    slug = add_entity(store, name, type="TYPE", description="test entity")
    store.conn.execute("UPDATE entities SET community = ? WHERE id = ?", (community, slug))
    store.conn.commit()
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

    patch_ollama_generate(monkeypatch, _fake_generate)
    monkeypatch.setattr("local_graph_rag.graph.summarizer.embed", lambda *a, **kw: [0.0] * 768)

    assert summarize_community(0, store) is False
    assert len(generate_calls) == 0


def test_summarize_community_regenerates_on_membership_change(store, monkeypatch):
    slug = _add_entity_in_community(store, "Beta", 0)
    store.upsert_community(0, "old summary", [slug], "stale_hash" * 4, _ZERO_EMBEDDING)

    patch_ollama_generate(monkeypatch, lambda *a, **kw: "new summary")
    monkeypatch.setattr("local_graph_rag.graph.summarizer.embed", lambda *a, **kw: [0.1] * 768)

    assert summarize_community(0, store) is True
    communities = store.get_communities()
    assert len(communities) == 1
    assert communities[0]["summary"] == "new summary"


def test_summarize_community_reuses_summary_after_renumbering(store, monkeypatch):
    """Louvain may give an unchanged community a new id; its summary must carry over."""
    slug = _add_entity_in_community(store, "Gamma", 5)
    member_hash = _compute_member_hash(
        store.get_entities_for_community(5), store.get_relationships_for_community(5)
    )
    store.upsert_community(0, "kept summary", [slug], member_hash, _ZERO_EMBEDDING)

    def _fail_generate(*a, **kw):
        raise AssertionError("an unchanged community must not be re-summarized")

    patch_ollama_generate(monkeypatch, _fail_generate)

    assert summarize_community(5, store) is False
    summaries = {c["id"]: c["summary"] for c in store.get_communities()}
    assert summaries[5] == "kept summary"


def test_summarize_all_second_run_on_unchanged_graph_makes_no_llm_calls(store, monkeypatch):
    for name in ("a", "b", "c", "d"):
        add_entity(store, name, type="T", description=f"entity {name}")
    add_relationship(store, "a", "b", "uses", "doc.py")
    add_relationship(store, "c", "d", "uses", "doc.py")
    store.upsert_community(99, "stale summary", ["gone"], "stale_hash", _ZERO_EMBEDDING)

    calls: list[int] = []

    def _fake_generate(*a, **kw):
        calls.append(1)
        return "summary"

    patch_ollama_generate(monkeypatch, _fake_generate)
    monkeypatch.setattr("local_graph_rag.graph.summarizer.embed", lambda *a, **kw: [0.1] * 768)

    first = summarize_all_communities(store)
    calls.clear()
    second = summarize_all_communities(store)

    assert first["summarized"] == 2
    assert second == {"summarized": 0, "skipped": 2, "failed": 0}
    assert calls == []
    assert 99 not in {c["id"] for c in store.get_communities()}


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
