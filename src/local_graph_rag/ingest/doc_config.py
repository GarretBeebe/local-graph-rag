"""Load and parse the document index configuration file."""

from dataclasses import dataclass, field
from pathlib import Path

from local_graph_rag.common.config import load_yaml_config
from local_graph_rag.common.paths import normalize_extensions
from local_graph_rag.settings import ALLOWED_EXTENSIONS, INDEX_CONFIG_PATH


@dataclass
class IndexPath:
    path: Path
    recursive: bool = True
    exclude_dirs: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        self.path = self.path.resolve()


@dataclass
class IndexConfig:
    index_paths: list[IndexPath]
    allowed_extensions: frozenset[str]
    ignore_patterns: list[str]

    @property
    def roots(self) -> list[Path]:
        # Paths are pre-resolved at construction time — no re-resolution needed.
        return [p.path for p in self.index_paths]


def load_index_config() -> IndexConfig | None:
    """Return a parsed IndexConfig if INDEX_CONFIG_PATH is set, else None (DOCS_PATH fallback)."""
    if not INDEX_CONFIG_PATH:
        return None
    raw = load_yaml_config(Path(INDEX_CONFIG_PATH))

    entries = raw.get("index_paths") or []
    if not entries:
        raise ValueError(f"index_config: no index_paths defined in {INDEX_CONFIG_PATH}")

    index_paths: list[IndexPath] = []
    for e in entries:
        d = e if isinstance(e, dict) else {}
        path_str = d.get("path") or (e if isinstance(e, str) else None)
        if not path_str:
            raise ValueError("index_config: each index_paths entry must have a 'path' key")

        recursive_raw = d.get("recursive", True)
        if isinstance(recursive_raw, str):
            raise ValueError(
                f"index_config: 'recursive' must be a YAML boolean (true/false), "
                f"not a quoted string: {recursive_raw!r}"
            )

        exclude_dirs_raw = d.get("exclude_dirs", [])
        if isinstance(exclude_dirs_raw, str):
            exclude_dirs_raw = [exclude_dirs_raw]

        index_paths.append(IndexPath(
            path=Path(path_str),
            recursive=bool(recursive_raw),
            exclude_dirs=frozenset(exclude_dirs_raw),
        ))

    raw_exts = raw.get("allowed_extensions")
    allowed_extensions = normalize_extensions(
        raw_exts if raw_exts is not None else ALLOWED_EXTENSIONS
    )

    return IndexConfig(
        index_paths=index_paths,
        allowed_extensions=allowed_extensions,
        ignore_patterns=raw.get("ignore_patterns") or [],
    )
