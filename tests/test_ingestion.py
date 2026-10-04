"""Unit tests for Phase 3 — fingerprint store methods and file hash utilities."""

import hashlib
import json
import os
import sqlite3
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

import local_graph_rag.ingest.index_documents as _idx_mod
from local_graph_rag.graph.store import GraphStore
from local_graph_rag.ingest.doc_config import IndexConfig, IndexPath
from local_graph_rag.ingest.index_documents import (
    _collect_files,
    _compute_hash,
    _index_file,
    _match_entity_chunks,
    _write_index_data,
    ensure_collection,
)
from tests.helpers import EMPTY_EXTRACTION_JSON, patch_ollama_generate

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


def test_purge_file_deletes_hash(store: GraphStore):
    store.upsert_hash("/docs/foo.md", "abc")
    store.purge_file("/docs/foo.md")
    assert store.get_hash("/docs/foo.md") is None


def test_list_all_paths(store: GraphStore):
    store.upsert_hash("/a.md", "h1")
    store.upsert_hash("/b.md", "h2")
    assert set(store.list_all_paths()) == {"/a.md", "/b.md"}


def test_list_all_paths_includes_files_with_data_but_no_fingerprint(store: GraphStore):
    """A file whose indexing failed partway has no fingerprint but still has data."""
    store.upsert_hash("/a.md", "h1")
    store.register_chunks([("c1", "/b.md", 0)])
    store.cache_extraction("/c.md", 0, "sha", "{}")
    assert set(store.list_all_paths()) == {"/a.md", "/b.md", "/c.md"}


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
    def __init__(self, exists=False, vector_size: int = 768, distance=Distance.COSINE):
        self._exists = exists
        self._vector_size = vector_size
        self._distance = distance
        self.created_collection = False
        self.payload_index_calls: list[dict] = []
        self.upserts: list[dict] = []
        self.deleted: list[dict] = []

    def collection_exists(self, name):
        return self._exists

    def create_collection(self, **kwargs):
        self.created_collection = True

    def get_collection(self, **kwargs):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=VectorParams(size=self._vector_size, distance=self._distance)
                )
            )
        )

    def create_payload_index(self, **kwargs):
        self.payload_index_calls.append(kwargs)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def delete(self, **kwargs):
        self.deleted.append(kwargs)


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


def test_ensure_collection_rejects_mismatched_existing_vector_size(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(_idx_mod, "_collection_ensured", False)
    client = _FakeQdrant(exists=True, vector_size=123)

    with pytest.raises(RuntimeError, match="vector config"):
        ensure_collection(client)


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
            return EMPTY_EXTRACTION_JSON
        raise RuntimeError("boom")

    patch_ollama_generate(monkeypatch, _fake_generate)

    chunks = ["alpha entity chunk", "beta entity chunk"]
    points = _make_points(chunks, "foo.py")

    _write_index_data("foo.py", chunks, points, store, _FakeQdrant(), "hash123")

    assert store.get_hash("foo.py") is None


def test_write_index_data_sets_hash_on_full_success(
    store: GraphStore, monkeypatch: pytest.MonkeyPatch
):
    patch_ollama_generate(monkeypatch, lambda *a, **k: EMPTY_EXTRACTION_JSON)

    chunks = ["alpha entity chunk"]
    points = _make_points(chunks, "foo.py")

    _write_index_data("foo.py", chunks, points, store, _FakeQdrant(), "hash123")

    assert store.get_hash("foo.py") == "hash123"


# ---------------------------------------------------------------------------
# _collect_files / main — paths that could not be scanned keep their index entries
# ---------------------------------------------------------------------------


def _configure_index_paths(monkeypatch: pytest.MonkeyPatch, *index_paths: IndexPath) -> None:
    config = IndexConfig(
        index_paths=list(index_paths),
        allowed_extensions=frozenset({".md"}),
        ignore_patterns=[],
    )
    monkeypatch.setattr(_idx_mod, "_INDEX_CONFIG", config)
    monkeypatch.setattr(_idx_mod, "_CONFIG_LOAD_ERROR", None)
    monkeypatch.setattr(_idx_mod, "_DOCS_ROOTS", config.roots)
    monkeypatch.setattr(_idx_mod, "_ALLOWED", config.allowed_extensions)


def test_collect_files_reports_missing_and_empty_roots_as_unscanned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    present = tmp_path / "present"
    present.mkdir()
    (present / "a.md").write_text("hello")
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = tmp_path / "missing"
    _configure_index_paths(monkeypatch, IndexPath(present), IndexPath(empty), IndexPath(missing))

    files, unscanned = _collect_files()

    assert [f.name for f in files] == ["a.md"]
    assert set(unscanned) == {empty.resolve(), missing.resolve()}


def test_collect_files_reports_subdirectories_that_fail_to_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.md").write_text("hello")
    _configure_index_paths(monkeypatch, IndexPath(root))
    real_walk = os.walk

    def _walk_with_unreadable_subdir(top, topdown=True, onerror=None, followlinks=False):
        onerror(PermissionError(13, "Permission denied", str(root / "locked")))
        yield from real_walk(top, topdown=topdown, onerror=onerror, followlinks=followlinks)

    monkeypatch.setattr(_idx_mod.os, "walk", _walk_with_unreadable_subdir)

    files, unscanned = _collect_files()

    assert [f.name for f in files] == ["a.md"]
    assert unscanned == [(root / "locked").resolve()]


def test_main_keeps_index_entries_under_unscanned_paths(
    tmp_path: Path, store: GraphStore, monkeypatch: pytest.MonkeyPatch
):
    offline_root = (tmp_path / "offline").resolve()
    kept = str(offline_root / "note.md")
    deleted = str((tmp_path / "online" / "deleted.md").resolve())
    for path in (kept, deleted):
        store.register_chunks([(f"chunk-{path}", path, 0)])
        store.upsert_hash(path, "h")
    client = _FakeQdrant()
    monkeypatch.setattr(_idx_mod, "GraphStore", lambda: store)
    monkeypatch.setattr(_idx_mod, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(_idx_mod, "ensure_collection", lambda _client: None)
    monkeypatch.setattr(_idx_mod, "_collect_files", lambda: ([], [offline_root]))

    _idx_mod.main()

    assert store.list_all_paths() == [kept]
    assert len(client.deleted) == 1


def test_index_file_keeps_previous_version_when_embedding_fails(
    tmp_path: Path, store: GraphStore, monkeypatch: pytest.MonkeyPatch
):
    doc = tmp_path / "note.md"
    doc.write_text("# Title\n\nnew content")
    filepath = str(doc.resolve())
    store.register_chunks([("old-chunk", filepath, 0)])
    store.upsert_hash(filepath, "old-hash")
    _configure_index_paths(monkeypatch, IndexPath(tmp_path))

    def _ollama_down(*args, **kwargs):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(_idx_mod, "embed_batch", _ollama_down)
    client = _FakeQdrant()

    assert _index_file(doc, store, client) == "failed"
    assert store.get_chunks_for_file(filepath) == ["old-chunk"]
    assert client.deleted == []


def test_write_index_data_tolerates_fields_of_the_wrong_type(
    store: GraphStore, monkeypatch: pytest.MonkeyPatch
):
    """The raw reply is cached and replayed, so bad field types must not fail the file."""
    reply = json.dumps({
        "entities": [
            {"name": "Parser", "type": ["CLASS", "FUNCTION"], "description": {"text": "x"}},
            {"name": 5, "type": "CLASS"},
        ],
        "relationships": [{"source": "Parser", "target": ["Parser"], "label": "uses"}],
    })
    patch_ollama_generate(monkeypatch, lambda *a, **k: reply)
    chunks = ["Parser parses text"]

    result = _write_index_data(
        "f.py", chunks, _make_points(chunks, "f.py"), store, _FakeQdrant(), "h1"
    )

    assert [(e["name"], e["type"], e["description"]) for e in result.entities] == [
        ("Parser", None, "")
    ]
    assert result.relationships == []
    assert store.get_hash("f.py") == "h1"


def test_restoring_an_emptied_file_reindexes_it(
    tmp_path: Path, store: GraphStore, monkeypatch: pytest.MonkeyPatch
):
    """Regression: emptying a file kept its old fingerprint, so restoring it was skipped."""
    doc = tmp_path / "note.md"
    content = "# Title\n\nbody text"
    monkeypatch.setattr(_idx_mod, "embed_batch", lambda texts: [[0.0] * 768 for _ in texts])
    patch_ollama_generate(monkeypatch, lambda *a, **k: EMPTY_EXTRACTION_JSON)
    client = _FakeQdrant()

    doc.write_text(content)
    assert _index_file(doc, store, client) == "indexed"
    doc.write_text("")
    assert _index_file(doc, store, client) == "skipped"
    assert store.get_hash(str(doc.resolve())) == hashlib.sha256(b"").hexdigest()
    doc.write_text(content)
    assert _index_file(doc, store, client) == "indexed"


def test_collect_files_marks_unenterable_directories_unscanned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A directory listable but not enterable (read without execute) must not abort the run."""
    root = tmp_path / "root"
    locked = root / "locked"
    locked.mkdir(parents=True)
    (root / "a.md").write_text("hello")
    (locked / "b.md").write_text("unreachable")
    _configure_index_paths(monkeypatch, IndexPath(root))
    real_check = _idx_mod._is_safe_indexable_file

    def _check(path: Path) -> bool:
        if path.parent.name == "locked":
            raise PermissionError(13, "Permission denied", str(path))
        return real_check(path)

    monkeypatch.setattr(_idx_mod, "_is_safe_indexable_file", _check)

    files, unscanned = _collect_files()

    assert [f.name for f in files] == ["a.md"]
    assert unscanned == [locked.resolve()]


def test_index_file_counts_a_failed_fingerprint_write_as_failed(
    tmp_path: Path, store: GraphStore, monkeypatch: pytest.MonkeyPatch
):
    doc = tmp_path / "empty.md"
    doc.write_text("")

    def _locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "upsert_hash", _locked)

    assert _index_file(doc, store, _FakeQdrant()) == "failed"


def test_main_counts_a_file_whose_indexing_raises_as_failed(
    tmp_path: Path,
    store: GraphStore,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """One file's unexpected error (e.g. "database is locked" reading its fingerprint)
    must not abort the run."""
    good, bad = tmp_path / "good.md", tmp_path / "bad.md"
    monkeypatch.setattr(_idx_mod, "GraphStore", lambda: store)
    monkeypatch.setattr(_idx_mod, "get_qdrant_client", lambda: _FakeQdrant())
    monkeypatch.setattr(_idx_mod, "ensure_collection", lambda _client: None)
    monkeypatch.setattr(_idx_mod, "_collect_files", lambda: ([bad, good], []))

    def _index(path: Path, *_args: object) -> str:
        if path == bad:
            raise sqlite3.OperationalError("database is locked")
        return "indexed"

    monkeypatch.setattr(_idx_mod, "_index_file", _index)

    _idx_mod.main()

    assert "indexed: 1, skipped: 0, failed: 1" in capsys.readouterr().out
