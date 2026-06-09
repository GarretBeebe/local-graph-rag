"""Shared path and file-filter helpers."""

from collections.abc import Iterable
from fnmatch import fnmatch
from pathlib import Path


def normalize_path(path: str | Path) -> str:
    """Return a normalized absolute path string."""
    return str(Path(path).expanduser().resolve())


def normalize_extensions(extensions: Iterable[str]) -> frozenset[str]:
    """Return a lowercase immutable extension set for repeated path checks."""
    return frozenset(ext.lower() for ext in extensions)


def has_allowed_extension(path: str | Path, allowed_extensions: Iterable[str]) -> bool:
    """Return True when the path suffix is in the allowed extension set."""
    normalized = (
        allowed_extensions
        if isinstance(allowed_extensions, frozenset)
        else normalize_extensions(allowed_extensions)
    )
    return Path(path).suffix.lower() in normalized


def _pattern_matches(
    pattern: str,
    parts: tuple[str, ...],
    path_text: str,
    normalized_path_text: str,
) -> bool:
    normalized_pattern = pattern.replace("\\", "/")
    if pattern in parts:
        return True
    if any(fnmatch(part, pattern) for part in parts):
        return True
    if fnmatch(path_text, pattern):
        return True
    return "/" in normalized_pattern and (
        fnmatch(normalized_path_text, normalized_pattern)
        or fnmatch(normalized_path_text, f"*/{normalized_pattern}")
        or fnmatch(normalized_path_text, f"*/{normalized_pattern}/*")
    )


def matches_ignore_pattern(path: str | Path, ignore_patterns: Iterable[str]) -> bool:
    """Return True when any ignore token or glob matches a path component."""
    path_obj = Path(path)
    parts = path_obj.parts
    path_text = str(path_obj)
    normalized_path_text = path_text.replace("\\", "/")
    return any(
        _pattern_matches(pattern, parts, path_text, normalized_path_text)
        for pattern in ignore_patterns
    )


def is_under_any_root(path: Path, roots: list[Path]) -> bool:
    """Return True if path falls under any of the given roots."""
    return any(path.is_relative_to(root) for root in roots)


def is_indexable_path(
    path: str | Path,
    allowed_extensions: Iterable[str],
    ignore_patterns: Iterable[str] = (),
) -> bool:
    """Return True when a path passes ignore and extension filters."""
    return not matches_ignore_pattern(path, ignore_patterns) and has_allowed_extension(
        path, allowed_extensions
    )
