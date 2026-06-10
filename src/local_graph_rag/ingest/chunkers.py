"""
Document chunking strategies for the ingest pipeline.

Splits raw document text into chunks suitable for embedding, dispatching
on file extension:
  - .py              — AST-based splitting at top-level function/class boundaries;
                       falls back to recursive character splitting if parsing fails
                       or the file contains no top-level definitions
  - .md / .markdown  — splits at Markdown header boundaries (H1–H6)
  - all others       — recursive character splitting with a 500-character window
                       and 100-character overlap

Public API: chunk_document(path, text) -> list[tuple[str, str | None]]

Each item is (chunk_text, def_name). For .py files, def_name is the name of the
top-level function/class the chunk belongs to (or None for module-level code,
docstrings, and gaps). For all other file types, def_name is always None.
"""

import ast
import re
from pathlib import Path

from local_graph_rag.settings import CHUNK_OVERLAP, CHUNK_SIZE, MAX_CHUNK_CHARS, MAX_MD_CHUNK

_SEPARATORS = ["\n\n", "\n", " ", ""]


def _tag_none(chunks: list[str]) -> list[tuple[str, str | None]]:
    return [(c, None) for c in chunks]


def _merge_splits(splits: list[str], separator: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    sep_len = len(separator)

    for split in splits:
        split_len = len(split)
        added_len = split_len + (sep_len if current else 0)
        if current and current_len + added_len > CHUNK_SIZE:
            chunks.append(separator.join(current))
            while current and current_len > CHUNK_OVERLAP:
                dropped = len(current[0]) + (sep_len if len(current) > 1 else 0)
                current_len -= dropped
                current.pop(0)
        current.append(split)
        current_len += split_len + (sep_len if len(current) > 1 else 0)

    if current:
        chunks.append(separator.join(current))

    return chunks


def _recursive_split(text: str, separators: list[str]) -> list[str]:
    separator = separators[-1]
    remaining: list[str] = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining = separators[i + 1 :]
            break

    parts = [s for s in text.split(separator) if s] if separator else list(text)
    good: list[str] = []
    result: list[str] = []

    for part in parts:
        if len(part) > CHUNK_SIZE:
            if good:
                result.extend(_merge_splits(good, separator))
                good = []
            result.extend(_recursive_split(part, remaining) if remaining else [part])
        else:
            good.append(part)

    if good:
        result.extend(_merge_splits(good, separator))

    return result


def chunk_text(text: str) -> list[str]:
    if not text.strip():
        return []
    return _recursive_split(text, _SEPARATORS)


# -------------------------
# Python code chunking
# -------------------------


def chunk_python(text: str) -> list[tuple[str, str | None]]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return _tag_none(chunk_text(text))

    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    def _span(start: int, end: int) -> str:
        """Return stripped text for 1-based inclusive line range [start, end]."""
        return "".join(lines[start - 1 : end]).strip()

    def _emit(segment: str, def_name: str | None, out: list[tuple[str, str | None]]) -> None:
        if segment:
            pieces = chunk_text(segment) if len(segment) > MAX_CHUNK_CHARS else [segment]
            out.extend((p, def_name) for p in pieces)

    chunks: list[tuple[str, str | None]] = []
    prev_end = 0

    for node in tree.body:
        # Emit any gap (imports, assignments, comments) before this node.
        if node.lineno > prev_end + 1:
            _emit(_span(prev_end + 1, node.lineno - 1), None, chunks)

        # Bare import statements carry ~no "what does X do" signal for any query —
        # skip emitting them as standalone chunks entirely.
        if isinstance(node, ast.Import | ast.ImportFrom):
            prev_end = node.end_lineno
            continue

        # Emit the node itself, tagged with its def/class name if applicable.
        segment = (
            ast.get_source_segment(text, node) or "".join(lines[node.lineno - 1 : node.end_lineno])
        ).strip()
        is_def = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        def_name = node.name if is_def else None
        _emit(segment, def_name, chunks)
        prev_end = node.end_lineno

    # Emit any trailing code after the last node.
    if prev_end < total_lines:
        _emit(_span(prev_end + 1, total_lines), None, chunks)

    return chunks if chunks else _tag_none(chunk_text(text))


# -------------------------
# Markdown chunking
# -------------------------

HEADER_PATTERN = re.compile(r"^#{1,6} ")


def _split_markdown_sections(text: str) -> list[str]:
    """Split a markdown document into sections at header boundaries."""
    lines = text.splitlines()
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        if HEADER_PATTERN.match(line) and current:
            sections.append("\n".join(current))
            current = []
        current.append(line)

    if current:
        sections.append("\n".join(current))

    return [s.strip() for s in sections if s.strip()]


def _split_oversized_markdown_section(section: str) -> list[str]:
    """Split a large markdown section into smaller chunks respecting MAX_MD_CHUNK."""
    if len(section) <= MAX_MD_CHUNK:
        return [section]

    paragraphs = section.split("\n\n")
    final_chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0  # length of "\n\n".join(buf)

    for p in paragraphs:
        # Adding a paragraph adds its length plus the separator ("\n\n") if buffer isn't empty.
        additional = len(p) + (2 if buf else 0)
        if buf and (buf_len + additional) > MAX_MD_CHUNK:
            final_chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0

        if len(p) > MAX_MD_CHUNK:
            final_chunks.extend(chunk_text(p))
        else:
            buf.append(p)
            buf_len += len(p) + (2 if buf_len else 0)

    if buf:
        final_chunks.append("\n\n".join(buf))

    return final_chunks


def chunk_markdown(text: str) -> list[str]:
    """Chunk markdown into sections and sub-sections that fit within MAX_MD_CHUNK."""
    sections = _split_markdown_sections(text)
    final_chunks: list[str] = []

    for section in sections:
        final_chunks.extend(_split_oversized_markdown_section(section))

    return final_chunks


# -------------------------
# Dispatcher
# -------------------------


def chunk_document(path: Path, text: str) -> list[tuple[str, str | None]]:

    suffix = path.suffix.lower()

    if suffix == ".py":
        return chunk_python(text)

    if suffix in {".md", ".markdown"}:
        return _tag_none(chunk_markdown(text))

    return _tag_none(chunk_text(text))
