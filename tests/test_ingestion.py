"""Unit tests for Phase 3 — fingerprint store methods and file hash utilities."""

import hashlib
import uuid
from pathlib import Path

import pytest
from qdrant_client.models import PayloadSchemaType, PointStruct

import local_graph_rag.ingest.index_documents as _idx_mod
from local_graph_rag.graph.store import GraphStore
from local_graph_rag.ingest.index_documents import (
    _collect_files,
    _compute_hash,
    _match_entity_chunks,
    _write_index_data,
    ensure_collection,
)

# ---------------------------------------------------------------------------
# GraphStore — fingerprints
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    return GraphStore(db_path=tmp_path / "test.db")


def test_get_hash_unknown(store: GraphStore):
    assert store.get_hash("/some/file.py") is None


def test_upsert_and_get_hash(store: GraphStore):
    store.upsert_hash("/docs/foo.md", "abc123")
    assert store.get_hash("/docs/foo.md") == "abc123"


def test_upsert_hash_overwrites(store: GraphStore):
    store.upsert_hash("/docs/foo.md", "old_hash")
    store.upsert_hash("/docs/foo.md", "new_hash")
    assert store.get_hash("/docs/foo.md") == "new_hash"


def test_delete_hash(store: GraphStore):
    store.upsert_hash("/docs/foo.md", "abc")
    store.delete_hash("/docs/foo.md")
    assert store.get_hash("/docs/foo.md") is None


def test_list_all_paths(store: GraphStore):
    store.upsert_hash("/a.md", "h1")
    store.upsert_hash("/b.md", "h2")
    assert set(store.list_all_paths()) == {"/a.md", "/b.md"}


# ---------------------------------------------------------------------------
# _compute_hash
# ---------------------------------------------------------------------------


def test_compute_hash_stable(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("hello world")
    assert _compute_hash(f) == _compute_hash(f)


def test_compute_hash_changes(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("content a")
    h1 = _compute_hash(f)
    f.write_text("content b")
    h2 = _compute_hash(f)
    assert h1 != h2


def test_compute_hash_matches_sha256(tmp_path: Path):
    f = tmp_path / "file.txt"
    content = b"known content"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert _compute_hash(f) == expected


# ---------------------------------------------------------------------------
# _match_entity_chunks
# ---------------------------------------------------------------------------


def test_match_entity_chunks_links_only_chunks_containing_name():
    chunks = ["Alpha appears here.", "Nothing relevant in this one."]
    pairs = _match_entity_chunks(chunks, ["c1", "c2"], [{"name": "Alpha"}], ["alpha"])
    assert pairs == [("c1", "alpha")]


def test_match_entity_chunks_uses_word_boundaries():
    chunks = ["The Catalog feature shipped today.", "The cat sat on the mat."]
    pairs = _match_entity_chunks(chunks, ["c1", "c2"], [{"name": "Cat"}], ["cat"])
    assert pairs == [("c2", "cat")]  # not chunk c1 — "Cat" has no word boundary inside "Catalog"


def test_match_entity_chunks_skips_short_names():
    pairs = _match_entity_chunks(["C is a programming language."], ["c1"], [{"name": "C"}], ["c"])
    assert pairs == []


# ---------------------------------------------------------------------------
# _collect_files — config load error propagation
# ---------------------------------------------------------------------------


def test_collect_files_raises_runtime_error_on_config_load_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    err = FileNotFoundError("config not found")
    monkeypatch.setattr(_idx_mod, "_CONFIG_LOAD_ERROR", err)
    monkeypatch.setattr(_idx_mod, "_INDEX_CONFIG", None)
    with pytest.raises(RuntimeError, match="Failed to load index config"):
        _collect_files()


# ---------------------------------------------------------------------------
# ensure_collection / _write_index_data — fake Qdrant client
# ---------------------------------------------------------------------------


class _FakeQdrant:
    def __init__(self, exists=False):
        self._exists = exists
        self.created_collection = False
        self.payload_index_calls: list[dict] = []
        self.upserts: list[dict] = []

    def collection_exists(self, name):
        return self._exists

    def create_collection(self, **kwargs):
        self.created_collection = True

    def create_payload_index(self, **kwargs):
        self.payload_index_calls.append(kwargs)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)


def test_ensure_collection_creates_payload_index(monkeypatch: pytest.MonkeyPatch):
    for exists in (False, True):
        monkeypatch.setattr(_idx_mod, "_collection_ensured", False)
        client = _FakeQdrant(exists=exists)

        ensure_collection(client)

        assert client.created_collection is (not exists)
        assert len(client.payload_index_calls) == 1
        call = client.payload_index_calls[0]
        assert call["field_name"] == "def_name"
        assert call["field_schema"] == PayloadSchemaType.KEYWORD


def _make_points(chunks: list[str], filepath: str) -> list[PointStruct]:
    return [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.0],
            payload={"text": c, "filepath": filepath, "chunk_index": i, "def_name": None},
        )
        for i, c in enumerate(chunks)
    ]


def test_write_index_data_skips_hash_on_extraction_failure(
    store: GraphStore, monkeypatch: pytest.MonkeyPatch
):
    """A failed extraction batch must leave the file's hash unset so the next
    run retries it (see extractor.ExtractionResult.had_failure).
    """
    monkeypatch.setattr("local_graph_rag.graph.extractor.EXTRACT_BATCH_TOKENS", 1)

    call_count = 0

    def _fake_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '{"entities": [], "relationships": []}'
        raise RuntimeError("boom")

    monkeypatch.setattr("local_graph_rag.rag.ollama_client.generate", _fake_generate)

    chunks = ["alpha entity chunk", "beta entity chunk"]
    points = _make_points(chunks, "foo.py")

    _write_index_data("foo.py", chunks, points, store, _FakeQdrant(), "hash123")

    assert store.get_hash("foo.py") is None


def test_write_index_data_sets_hash_on_full_success(
    store: GraphStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "local_graph_rag.rag.ollama_client.generate",
        lambda *a, **k: '{"entities": [], "relationships": []}',
    )

    chunks = ["alpha entity chunk"]
    points = _make_points(chunks, "foo.py")

    _write_index_data("foo.py", chunks, points, store, _FakeQdrant(), "hash123")

    assert store.get_hash("foo.py") == "hash123"
