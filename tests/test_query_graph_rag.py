"""Unit tests for the end-to-end query module. No Ollama or Qdrant required."""

from types import SimpleNamespace

import pytest

from local_graph_rag.rag.local_retrieval import LocalContext
from local_graph_rag.rag.query_graph_rag import _build_prompt, _format_local, _validate_mode


@pytest.fixture
def store(tmp_path):
    from local_graph_rag.graph.store import GraphStore

    s = GraphStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


def _empty_qdrant():
    """Qdrant stand-in whose query_points always returns zero points."""
    return SimpleNamespace(query_points=lambda *a, **kw: SimpleNamespace(points=[]))


# ---------------------------------------------------------------------------
# _format_local
# ---------------------------------------------------------------------------


def test_format_local_resolves_relationship_slugs_to_names():
    ctx = LocalContext(
        entities=[
            {
                "id": "graphstore", "name": "GraphStore",
                "type": "CLASS", "description": "Manages the graph",
            },
            {
                "id": "entity_extractor", "name": "Entity Extractor",
                "type": "MODULE", "description": "Extracts entities",
            },
        ],
        relationships=[
            {
                "source_id": "graphstore", "target_id": "entity_extractor",
                "label": "USES", "weight": 1.0,
            },
        ],
        chunk_texts=[],
    )
    prompt = _format_local(ctx, "what uses what?")
    assert "GraphStore --[USES]--> Entity Extractor" in prompt
    assert "graphstore --[USES]-->" not in prompt


def test_format_local_falls_back_to_slug_for_unresolvable_relationship_endpoint():
    # Relationship references an entity that didn't survive the cap/dedup into ctx.entities.
    ctx = LocalContext(
        entities=[
            {"id": "graphstore", "name": "GraphStore", "type": "CLASS", "description": ""},
        ],
        relationships=[
            {
                "source_id": "graphstore", "target_id": "missing_entity",
                "label": "USES", "weight": 1.0,
            },
        ],
        chunk_texts=[],
    )
    prompt = _format_local(ctx, "q")
    assert "GraphStore --[USES]--> missing_entity" in prompt


def test_format_local_caps_chunk_block_size():
    chunk = "x" * 6000
    ctx = LocalContext(entities=[], relationships=[], chunk_texts=[chunk, chunk, chunk])

    prompt = _format_local(ctx, "q")

    # 6000 + 6000 > 10_000, so only the first chunk is included.
    assert prompt.count(chunk) == 1


def test_format_local_chunk_block_always_includes_first_chunk_even_if_oversized():
    huge_chunk = "x" * 15000
    ctx = LocalContext(entities=[], relationships=[], chunk_texts=[huge_chunk, "small chunk"])

    prompt = _format_local(ctx, "q")

    assert huge_chunk in prompt
    assert "small chunk" not in prompt


# ---------------------------------------------------------------------------
# _validate_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["auto", "local", "global"])
def test_validate_mode_accepts_known_values(value):
    assert _validate_mode(value) == value


def test_validate_mode_rejects_unknown_value(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _validate_mode("bogus")
    assert exc_info.value.code != 0
    assert "bogus" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_local_mode_skips_get_communities(store, monkeypatch):
    monkeypatch.setattr("local_graph_rag.rag.local_retrieval.embed", lambda *a, **kw: [0.0] * 768)

    calls = []
    original = store.get_communities
    store.get_communities = lambda: calls.append(1) or original()  # type: ignore[method-assign]

    _build_prompt("question", "local", store, _empty_qdrant())

    assert calls == []


def test_build_prompt_global_mode_falls_back_to_local_when_context_empty(store, monkeypatch):
    # No communities seeded — global_retrieve returns an empty GlobalContext, and
    # _build_prompt should fall through to a local-mode prompt rather than handing
    # the LLM an empty "community summaries" block.
    monkeypatch.setattr("local_graph_rag.rag.local_retrieval.embed", lambda *a, **kw: [0.0] * 768)
    monkeypatch.setattr("local_graph_rag.rag.global_retrieval.embed", lambda *a, **kw: [0.0] * 768)

    prompt = _build_prompt("anything", "global", store, _empty_qdrant())

    assert "Use the knowledge graph context below to answer the question." in prompt
    assert "Use the community summaries below to answer the question." not in prompt
