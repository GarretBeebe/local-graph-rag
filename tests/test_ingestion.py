"""Unit tests for Phase 3 — fingerprint store methods and file hash utilities."""

import hashlib
from pathlib import Path

import pytest

from graph.store import GraphStore
from ingest.index_documents import _compute_hash

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
