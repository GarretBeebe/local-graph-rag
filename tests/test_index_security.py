"""Security hardening tests for ingest/index_documents.py."""

from pathlib import Path

import pytest

import ingest.index_documents as _mod
from ingest.index_documents import _is_safe_indexable_file, _is_within_size_limit


@pytest.fixture(autouse=True)
def patch_docs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point _DOCS_ROOT at a temp directory for each test."""
    monkeypatch.setattr(_mod, "_DOCS_ROOT", tmp_path.resolve())


# ---------------------------------------------------------------------------
# _is_safe_indexable_file
# ---------------------------------------------------------------------------


def test_normal_file_inside_root_is_accepted(tmp_path: Path):
    f = tmp_path / "notes.md"
    f.write_text("hello")
    assert _is_safe_indexable_file(f) is True


def test_symlink_to_file_outside_root_is_rejected(tmp_path: Path):
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)
    assert _is_safe_indexable_file(link) is False


def test_symlink_to_file_inside_root_is_rejected(tmp_path: Path):
    real = tmp_path / "real.md"
    real.write_text("data")
    link = tmp_path / "link.md"
    link.symlink_to(real)
    assert _is_safe_indexable_file(link) is False


def test_disallowed_extension_is_rejected(tmp_path: Path):
    f = tmp_path / "binary.exe"
    f.write_text("data")
    assert _is_safe_indexable_file(f) is False


def test_nonexistent_path_is_rejected(tmp_path: Path):
    assert _is_safe_indexable_file(tmp_path / "ghost.md") is False


# ---------------------------------------------------------------------------
# _is_within_size_limit
# ---------------------------------------------------------------------------


def test_file_within_limit_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_mod, "MAX_INDEX_FILE_BYTES", 100)
    f = tmp_path / "small.txt"
    f.write_bytes(b"x" * 100)
    assert _is_within_size_limit(f) is True


def test_file_over_limit_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_mod, "MAX_INDEX_FILE_BYTES", 100)
    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * 101)
    assert _is_within_size_limit(f) is False


