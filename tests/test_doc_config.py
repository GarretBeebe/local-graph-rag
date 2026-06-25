"""Tests for ingest/doc_config.py — config parsing and validation."""

from pathlib import Path

import pytest

from local_graph_rag.ingest.doc_config import load_index_config
from tests.helpers import patch_index_config_path


def _write_config(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "index_config.yaml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# load_index_config — returns None when INDEX_CONFIG_PATH is unset
# ---------------------------------------------------------------------------


def test_returns_none_when_not_configured(monkeypatch: pytest.MonkeyPatch):
    patch_index_config_path(monkeypatch, "")
    assert load_index_config() is None


# ---------------------------------------------------------------------------
# load_index_config — parsing
# ---------------------------------------------------------------------------


def test_parses_minimal_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = _write_config(
        tmp_path,
        """
index_paths:
  - path: /docs/notes
    recursive: true
""",
    )
    patch_index_config_path(monkeypatch, cfg_path)
    cfg = load_index_config()
    assert cfg is not None
    assert len(cfg.index_paths) == 1
    assert cfg.index_paths[0].path == Path("/docs/notes")
    assert cfg.index_paths[0].recursive is True
    assert cfg.index_paths[0].exclude_dirs == frozenset()


def test_parses_exclude_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = _write_config(
        tmp_path,
        """
index_paths:
  - path: /code
    recursive: true
    exclude_dirs: [.git, .venv, node_modules]
""",
    )
    patch_index_config_path(monkeypatch, cfg_path)
    cfg = load_index_config()
    assert cfg is not None
    assert cfg.index_paths[0].exclude_dirs == frozenset({".git", ".venv", "node_modules"})


def test_parses_allowed_extensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = _write_config(
        tmp_path,
        """
index_paths:
  - path: /docs
allowed_extensions: [.md, .txt, py]
""",
    )
    patch_index_config_path(monkeypatch, cfg_path)
    cfg = load_index_config()
    assert cfg is not None
    assert ".md" in cfg.allowed_extensions
    assert ".txt" in cfg.allowed_extensions
    assert ".py" in cfg.allowed_extensions  # normalized from "py"


def test_uses_default_extensions_when_omitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = _write_config(tmp_path, "index_paths:\n  - path: /docs\n")
    patch_index_config_path(monkeypatch, cfg_path)
    cfg = load_index_config()
    assert cfg is not None
    assert ".md" in cfg.allowed_extensions


def test_parses_ignore_patterns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = _write_config(
        tmp_path,
        """
index_paths:
  - path: /docs
ignore_patterns:
  - .git
  - "*.key"
""",
    )
    patch_index_config_path(monkeypatch, cfg_path)
    cfg = load_index_config()
    assert cfg is not None
    assert ".git" in cfg.ignore_patterns
    assert "*.key" in cfg.ignore_patterns


def test_real_config_excludes_claude_globally(monkeypatch: pytest.MonkeyPatch):
    """Regression test: .claude must be globally ignored, not just under one index_path.

    A prior config edit accidentally scoped .claude exclusion to only the Code
    path's exclude_dirs, leaving it unexcluded under the other 7 index paths.
    """
    real_config = Path(__file__).parent.parent / "config" / "index_config.yaml.example"
    patch_index_config_path(monkeypatch, real_config)
    cfg = load_index_config()
    assert cfg is not None
    assert ".claude" in cfg.ignore_patterns


def test_roots_resolves_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = _write_config(tmp_path, f"index_paths:\n  - path: {tmp_path}\n")
    patch_index_config_path(monkeypatch, cfg_path)
    cfg = load_index_config()
    assert cfg is not None
    assert tmp_path.resolve() in cfg.roots


# ---------------------------------------------------------------------------
# load_index_config — error cases
# ---------------------------------------------------------------------------


def test_raises_on_empty_index_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = _write_config(tmp_path, "index_paths: []\n")
    patch_index_config_path(monkeypatch, cfg_path)
    with pytest.raises(ValueError, match="no index_paths"):
        load_index_config()


def test_raises_on_invalid_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = tmp_path / "index_config.yaml"
    cfg_path.write_text("index_paths: [\nunclosed bracket")
    patch_index_config_path(monkeypatch, cfg_path)
    with pytest.raises(ValueError, match="invalid YAML"):
        load_index_config()


def test_raises_on_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    patch_index_config_path(monkeypatch, tmp_path / "nonexistent.yaml")
    with pytest.raises(FileNotFoundError):
        load_index_config()


def test_raises_on_quoted_recursive_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = _write_config(
        tmp_path,
        "index_paths:\n  - path: /docs\n    recursive: 'false'\n",
    )
    patch_index_config_path(monkeypatch, cfg_path)
    with pytest.raises(ValueError, match="quoted string"):
        load_index_config()


def test_bare_string_exclude_dirs_is_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = _write_config(
        tmp_path,
        "index_paths:\n  - path: /docs\n    exclude_dirs: .git\n",
    )
    patch_index_config_path(monkeypatch, cfg_path)
    cfg = load_index_config()
    assert cfg is not None
    assert cfg.index_paths[0].exclude_dirs == frozenset({".git"})


def test_path_stored_as_resolved_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = _write_config(tmp_path, f"index_paths:\n  - path: {tmp_path}\n")
    patch_index_config_path(monkeypatch, cfg_path)
    cfg = load_index_config()
    assert cfg is not None
    assert cfg.index_paths[0].path.is_absolute()
