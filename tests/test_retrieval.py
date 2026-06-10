"""Unit tests for local and global retrieval. No Ollama or Qdrant required."""

import numpy as np
import pytest

from local_graph_rag.graph.store import GraphStore
from local_graph_rag.rag.global_retrieval import GlobalContext, global_retrieve
from local_graph_rag.rag.local_retrieval import LocalContext, _extract_identifiers, local_retrieve


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Qdrant stub
# ---------------------------------------------------------------------------


class _FakePoint:
    def __init__(self, chunk_id: str, text: str = ""):
        self.id = chunk_id
        self.payload = {"text": text} if text else {}


class _FakeQdrant:
    def __init__(
        self,
        points: list[_FakePoint],
        scroll_results: dict[str, list[_FakePoint]] | None = None,
    ):
        self._points = points
        self._scroll_results = scroll_results or {}

    def query_points(self, *args, **kwargs):
        class _Result:
            pass

        r = _Result()
        r.points = self._points
        return r

    def scroll(self, *, scroll_filter, **kwargs):
        def_name = scroll_filter.must[0].match.value
        return self._scroll_results.get(def_name, []), None


# ---------------------------------------------------------------------------
# local_retrieve
# ---------------------------------------------------------------------------


def test_local_retrieve_chunk_texts_from_payload(store, monkeypatch):
    monkeypatch.setattr("local_graph_rag.rag.local_retrieval.embed", lambda *a, **kw: [0.0] * 768)
    client = _FakeQdrant([_FakePoint("c1", "hello world")])
    ctx = local_retrieve("q", store, client)
    assert ctx.chunk_texts == ["hello world"]


def test_local_retrieve_no_payload_text_excluded(store, monkeypatch):
    monkeypatch.setattr("local_graph_rag.rag.local_retrieval.embed", lambda *a, **kw: [0.0] * 768)
    client = _FakeQdrant([_FakePoint("c1")])  # no text in payload
    ctx = local_retrieve("q", store, client)
    assert ctx.chunk_texts == []


def test_local_retrieve_deduplicates_entities(store, monkeypatch):
    monkeypatch.setattr("local_graph_rag.rag.local_retrieval.embed", lambda *a, **kw: [0.0] * 768)

    # Two chunks both linked to the same entity
    slug = store.upsert_entity("Alpha", type="TYPE", description="desc")
    store.register_chunks([("c1", "doc.txt", 0), ("c2", "doc.txt", 1)])
    store.link_chunks([("c1", slug), ("c2", slug)])

    client = _FakeQdrant([_FakePoint("c1"), _FakePoint("c2")])
    ctx = local_retrieve("q", store, client, hops=1)

    entity_ids = [e["id"] for e in ctx.entities]
    assert entity_ids.count(slug) == 1


def test_local_retrieve_empty_when_no_chunk_links(store, monkeypatch):
    monkeypatch.setattr("local_graph_rag.rag.local_retrieval.embed", lambda *a, **kw: [0.0] * 768)
    # Qdrant returns a chunk that is not linked to any entity in the store
    client = _FakeQdrant([_FakePoint("c-unknown", "some text")])
    ctx = local_retrieve("q", store, client)
    assert ctx.entities == []
    assert ctx.relationships == []
    assert ctx.chunk_texts == ["some text"]


def test_local_retrieve_returns_empty_on_no_results(store, monkeypatch):
    monkeypatch.setattr("local_graph_rag.rag.local_retrieval.embed", lambda *a, **kw: [0.0] * 768)
    client = _FakeQdrant([])
    ctx = local_retrieve("q", store, client)
    assert ctx == LocalContext()


# ---------------------------------------------------------------------------
# _extract_identifiers
# ---------------------------------------------------------------------------


def test_extract_identifiers_leading_underscore():
    assert "_parse_extraction_response" in _extract_identifiers(
        "what does _parse_extraction_response do"
    )


def test_extract_identifiers_snake_case():
    assert "local_retrieve" in _extract_identifiers("explain local_retrieve please")


def test_extract_identifiers_camel_case():
    assert "GraphStore" in _extract_identifiers("what is GraphStore for")


def test_extract_identifiers_excludes_plain_english_words():
    assert _extract_identifiers("What does this function do") == []


# ---------------------------------------------------------------------------
# local_retrieve — def_name exact-match lookup
# ---------------------------------------------------------------------------


def test_local_retrieve_injects_def_name_exact_match(store, monkeypatch):
    monkeypatch.setattr("local_graph_rag.rag.local_retrieval.embed", lambda *a, **kw: [0.0] * 768)
    vector_hit = _FakePoint("c1", "from foo import bar")
    def_hit = _FakePoint("c2", "def _parse_extraction_response():\n    ...")
    client = _FakeQdrant(
        [vector_hit], scroll_results={"_parse_extraction_response": [def_hit]}
    )

    ctx = local_retrieve("what does _parse_extraction_response do", store, client)

    assert "from foo import bar" in ctx.chunk_texts
    assert "def _parse_extraction_response():\n    ..." in ctx.chunk_texts


def test_local_retrieve_def_name_match_deduped_with_vector_results(store, monkeypatch):
    monkeypatch.setattr("local_graph_rag.rag.local_retrieval.embed", lambda *a, **kw: [0.0] * 768)
    same_text = "def local_retrieve():\n    ..."
    client = _FakeQdrant(
        [_FakePoint("c1", same_text)],
        scroll_results={"local_retrieve": [_FakePoint("c2", same_text)]},
    )

    ctx = local_retrieve("explain local_retrieve", store, client)

    assert ctx.chunk_texts.count(same_text) == 1


def test_local_retrieve_no_identifiers_skips_def_name_lookup(store, monkeypatch):
    monkeypatch.setattr("local_graph_rag.rag.local_retrieval.embed", lambda *a, **kw: [0.0] * 768)
    client = _FakeQdrant([_FakePoint("c1", "hello world")])

    def _fail_scroll(*args, **kwargs):
        raise AssertionError("scroll should not be called when no identifiers are present")

    client.scroll = _fail_scroll

    ctx = local_retrieve("what is this about", store, client)
    assert ctx.chunk_texts == ["hello world"]


# ---------------------------------------------------------------------------
# global_retrieve
# ---------------------------------------------------------------------------


def _seed_community(store: GraphStore, community_id: int, vec: list[float], summary: str) -> None:
    embedding = np.array(vec, dtype=np.float32).tobytes()
    store.upsert_community(
        community_id, summary, [f"e{community_id}"], f"h{community_id}", embedding
    )


def test_global_retrieve_empty_communities(store, monkeypatch):
    monkeypatch.setattr(
        "local_graph_rag.rag.global_retrieval.embed",
        lambda *a, **kw: [0.0] * 3,
    )
    ctx = global_retrieve("q", store)
    assert ctx == GlobalContext()


def test_global_retrieve_top_n_by_cosine(store, monkeypatch):
    # Three orthogonal 3-dim communities
    _seed_community(store, 0, [1.0, 0.0, 0.0], "community zero")
    _seed_community(store, 1, [0.0, 1.0, 0.0], "community one")
    _seed_community(store, 2, [0.0, 0.0, 1.0], "community two")

    # Question embedding aligned with community 0
    monkeypatch.setattr(
        "local_graph_rag.rag.global_retrieval.embed",
        lambda *a, **kw: [1.0, 0.0, 0.0],
    )

    ctx = global_retrieve("q", store, n=2)
    assert len(ctx.community_summaries) == 2
    assert ctx.community_ids[0] == 0  # community 0 is best match


def test_global_retrieve_uses_pre_fetched_communities(store, monkeypatch):
    _seed_community(store, 0, [1.0, 0.0, 0.0], "only community")
    monkeypatch.setattr(
        "local_graph_rag.rag.global_retrieval.embed",
        lambda *a, **kw: [1.0, 0.0, 0.0],
    )

    communities = store.get_communities()
    # Passing pre-fetched communities — store.get_communities() must not be called again
    original_get = store.get_communities
    calls = []
    store.get_communities = lambda: calls.append(1) or original_get()  # type: ignore[method-assign]

    global_retrieve("q", store, communities=communities)
    assert calls == []  # no additional DB call


def test_global_retrieve_n_caps_results(store, monkeypatch):
    for i in range(5):
        vec = [0.0] * 5
        vec[i] = 1.0
        _seed_community(store, i, vec, f"summary {i}")

    monkeypatch.setattr(
        "local_graph_rag.rag.global_retrieval.embed",
        lambda *a, **kw: [1.0, 0.0, 0.0, 0.0, 0.0],
    )
    ctx = global_retrieve("q", store, n=3)
    assert len(ctx.community_summaries) == 3
