"""Unit tests for local and global retrieval. No Ollama or Qdrant required."""

import numpy as np
import pytest

from local_graph_rag.graph.store import GraphStore
from local_graph_rag.rag.global_retrieval import GlobalContext, global_retrieve
from local_graph_rag.rag.local_retrieval import LocalContext, _extract_identifiers, local_retrieve
from tests.helpers import patch_global_embed, patch_local_embed


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Qdrant stub
# ---------------------------------------------------------------------------


class _FakePoint:
    def __init__(
        self,
        chunk_id: str,
        text: str = "",
        def_name: str | None = None,
        chunk_index: int = 0,
    ):
        self.id = chunk_id
        payload: dict = {}
        if text:
            payload["text"] = text
        if def_name is not None:
            payload["def_name"] = def_name
        payload["chunk_index"] = chunk_index
        self.payload = payload


class _FakeQdrant:
    def __init__(
        self,
        points: list[_FakePoint],
        scroll_points: list[_FakePoint] | None = None,
    ):
        self._points = points
        self._scroll_points = scroll_points or []
        self.scroll_call_count = 0

    def query_points(self, *args, **kwargs):
        class _Result:
            pass

        r = _Result()
        r.points = self._points
        return r

    def scroll(self, *, scroll_filter, **kwargs):
        self.scroll_call_count += 1
        wanted = set(scroll_filter.must[0].match.any)
        matches = [p for p in self._scroll_points if p.payload.get("def_name") in wanted]
        return matches, None


# ---------------------------------------------------------------------------
# local_retrieve
# ---------------------------------------------------------------------------


def test_local_retrieve_chunk_texts_from_payload(store, monkeypatch):
    patch_local_embed(monkeypatch)
    client = _FakeQdrant([_FakePoint("c1", "hello world")])
    ctx = local_retrieve("q", store, client)
    assert ctx.chunk_texts == ["hello world"]


def test_local_retrieve_no_payload_text_excluded(store, monkeypatch):
    patch_local_embed(monkeypatch)
    client = _FakeQdrant([_FakePoint("c1")])  # no text in payload
    ctx = local_retrieve("q", store, client)
    assert ctx.chunk_texts == []


def test_local_retrieve_deduplicates_entities(store, monkeypatch):
    patch_local_embed(monkeypatch)

    # Two chunks both linked to the same entity
    slug = store.upsert_entity("Alpha", type="TYPE", description="desc")
    store.register_chunks([("c1", "doc.txt", 0), ("c2", "doc.txt", 1)])
    store.link_chunks([("c1", slug), ("c2", slug)])

    client = _FakeQdrant([_FakePoint("c1"), _FakePoint("c2")])
    ctx = local_retrieve("q", store, client, hops=1)

    entity_ids = [e["id"] for e in ctx.entities]
    assert entity_ids.count(slug) == 1


def test_local_retrieve_empty_when_no_chunk_links(store, monkeypatch):
    patch_local_embed(monkeypatch)
    # Qdrant returns a chunk that is not linked to any entity in the store
    client = _FakeQdrant([_FakePoint("c-unknown", "some text")])
    ctx = local_retrieve("q", store, client)
    assert ctx.entities == []
    assert ctx.relationships == []
    assert ctx.chunk_texts == ["some text"]


def test_local_retrieve_returns_empty_on_no_results(store, monkeypatch):
    patch_local_embed(monkeypatch)
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
    patch_local_embed(monkeypatch)
    vector_hit = _FakePoint("c1", "from foo import bar")
    def_hit = _FakePoint(
        "c2", "def _parse_extraction_response():\n    ...", def_name="_parse_extraction_response"
    )
    client = _FakeQdrant([vector_hit], scroll_points=[def_hit])

    ctx = local_retrieve("what does _parse_extraction_response do", store, client)

    assert "from foo import bar" in ctx.chunk_texts
    assert "def _parse_extraction_response():\n    ..." in ctx.chunk_texts


def test_local_retrieve_def_name_match_deduped_with_vector_results(store, monkeypatch):
    patch_local_embed(monkeypatch)
    same_text = "def local_retrieve():\n    ..."
    client = _FakeQdrant(
        [_FakePoint("c1", same_text)],
        scroll_points=[_FakePoint("c2", same_text, def_name="local_retrieve")],
    )

    ctx = local_retrieve("explain local_retrieve", store, client)

    assert ctx.chunk_texts.count(same_text) == 1


def test_local_retrieve_no_identifiers_skips_def_name_lookup(store, monkeypatch):
    patch_local_embed(monkeypatch)
    client = _FakeQdrant([_FakePoint("c1", "hello world")])

    def _fail_scroll(*args, **kwargs):
        raise AssertionError("scroll should not be called when no identifiers are present")

    client.scroll = _fail_scroll

    ctx = local_retrieve("what is this about", store, client)
    assert ctx.chunk_texts == ["hello world"]


def test_local_retrieve_multi_piece_def_ordered_by_chunk_index(store, monkeypatch):
    patch_local_embed(monkeypatch)
    piece_1 = _FakePoint("c2", "piece one", def_name="big_function", chunk_index=1)
    piece_0 = _FakePoint("c1", "piece zero", def_name="big_function", chunk_index=0)
    # Returned out of order — scroll doesn't guarantee ordering.
    client = _FakeQdrant([], scroll_points=[piece_1, piece_0])

    ctx = local_retrieve("explain big_function", store, client)

    assert ctx.chunk_texts == ["piece zero", "piece one"]


def test_local_retrieve_repeated_identifier_frees_dedup_slot(store, monkeypatch):
    patch_local_embed(monkeypatch)
    hit_a = _FakePoint("ca", "alpha body", def_name="alpha_func", chunk_index=0)
    hit_b = _FakePoint("cb", "beta body", def_name="beta_func", chunk_index=0)
    client = _FakeQdrant([], scroll_points=[hit_a, hit_b])

    question = "alpha_func alpha_func alpha_func alpha_func alpha_func beta_func"
    ctx = local_retrieve(question, store, client)

    assert "alpha body" in ctx.chunk_texts
    assert "beta body" in ctx.chunk_texts


def test_local_retrieve_single_scroll_call_for_multiple_identifiers(store, monkeypatch):
    patch_local_embed(monkeypatch)
    client = _FakeQdrant(
        [],
        scroll_points=[
            _FakePoint("c1", "alpha body", def_name="alpha_func", chunk_index=0),
            _FakePoint("c2", "beta body", def_name="beta_func", chunk_index=0),
        ],
    )

    local_retrieve("alpha_func and beta_func together", store, client)

    assert client.scroll_call_count == 1


# ---------------------------------------------------------------------------
# global_retrieve
# ---------------------------------------------------------------------------


def _seed_community(store: GraphStore, community_id: int, vec: list[float], summary: str) -> None:
    embedding = np.array(vec, dtype=np.float32).tobytes()
    store.upsert_community(
        community_id, summary, [f"e{community_id}"], f"h{community_id}", embedding
    )


def test_global_retrieve_empty_communities(store, monkeypatch):
    patch_global_embed(monkeypatch, [0.0] * 3)
    ctx = global_retrieve("q", store)
    assert ctx == GlobalContext()


def test_global_retrieve_top_n_by_cosine(store, monkeypatch):
    # Three orthogonal 3-dim communities
    _seed_community(store, 0, [1.0, 0.0, 0.0], "community zero")
    _seed_community(store, 1, [0.0, 1.0, 0.0], "community one")
    _seed_community(store, 2, [0.0, 0.0, 1.0], "community two")

    # Question embedding aligned with community 0
    patch_global_embed(monkeypatch, [1.0, 0.0, 0.0])

    ctx = global_retrieve("q", store, n=2)
    assert len(ctx.community_summaries) == 2
    assert ctx.community_ids[0] == 0  # community 0 is best match


def test_global_retrieve_uses_pre_fetched_communities(store, monkeypatch):
    _seed_community(store, 0, [1.0, 0.0, 0.0], "only community")
    patch_global_embed(monkeypatch, [1.0, 0.0, 0.0])

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

    patch_global_embed(monkeypatch, [1.0, 0.0, 0.0, 0.0, 0.0])
    ctx = global_retrieve("q", store, n=3)
    assert len(ctx.community_summaries) == 3
