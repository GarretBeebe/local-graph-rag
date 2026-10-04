"""Unit tests for chunkers.py — Python AST chunking, def_name tagging, and dispatch."""

from pathlib import Path

from local_graph_rag.ingest.chunkers import chunk_document, chunk_markdown, chunk_python
from local_graph_rag.settings import MAX_CHUNK_CHARS

# ---------------------------------------------------------------------------
# chunk_python — import skipping (Option 1)
# ---------------------------------------------------------------------------


def test_chunk_python_skips_plain_import():
    text = "import os\n\n\ndef foo():\n    return os.getcwd()\n"
    chunks = chunk_python(text)
    texts = [c for c, _ in chunks]
    assert not any("import os" in t for t in texts)
    assert any("def foo" in t for t in texts)


def test_chunk_python_skips_from_import():
    text = "from collections import OrderedDict\n\n\ndef foo():\n    return OrderedDict()\n"
    chunks = chunk_python(text)
    texts = [c for c, _ in chunks]
    assert not any("from collections" in t for t in texts)
    assert any("def foo" in t for t in texts)


def test_chunk_python_imports_only_falls_back_to_chunk_text():
    """If every top-level node is an import, AST chunking yields nothing and we
    fall back to recursive character splitting of the raw text (still tagged None)."""
    text = "import os\nimport sys\n"
    chunks = chunk_python(text)
    assert chunks  # fallback produces something rather than an empty list
    assert all(def_name is None for _, def_name in chunks)
    assert any("import os" in c for c, _ in chunks)


# ---------------------------------------------------------------------------
# chunk_python — def_name tagging (Option 2, revised)
# ---------------------------------------------------------------------------


def test_chunk_python_tags_function_def():
    text = "def foo():\n    return 1\n"
    chunks = chunk_python(text)
    assert ("def foo():\n    return 1", "foo") in chunks


def test_chunk_python_tags_async_function_def():
    text = "async def fetch():\n    return await thing()\n"
    chunks = chunk_python(text)
    matches = [c for c in chunks if c[1] == "fetch"]
    assert len(matches) == 1
    assert "async def fetch" in matches[0][0]


def test_chunk_python_tags_class_def():
    text = "class Foo:\n    def bar(self):\n        return 1\n"
    chunks = chunk_python(text)
    matches = [c for c in chunks if c[1] == "Foo"]
    assert len(matches) == 1
    assert "class Foo" in matches[0][0]


def test_chunk_python_module_docstring_untagged():
    text = '"""Module docstring."""\n\n\ndef foo():\n    return 1\n'
    chunks = chunk_python(text)
    docstring_chunks = [(c, name) for c, name in chunks if "Module docstring" in c]
    assert len(docstring_chunks) == 1
    assert docstring_chunks[0][1] is None


def test_chunk_python_module_level_assignment_untagged():
    text = "X = 1\n\n\ndef foo():\n    return X\n"
    chunks = chunk_python(text)
    for c, name in chunks:
        if c.startswith("X ="):
            assert name is None


# ---------------------------------------------------------------------------
# chunk_python — oversized definitions (Option 2: tag carries through splitting)
# ---------------------------------------------------------------------------


def test_chunk_python_oversized_def_all_pieces_share_def_name():
    body = "\n".join(f"    x{i} = {i}" for i in range(300))
    text = f"def big():\n{body}\n"
    assert len(text) > MAX_CHUNK_CHARS

    chunks = chunk_python(text)
    big_pieces = [c for c in chunks if c[1] == "big"]
    assert len(big_pieces) > 1
    assert all(name == "big" for _, name in big_pieces)


# ---------------------------------------------------------------------------
# chunk_python — fallback on syntax error
# ---------------------------------------------------------------------------


def test_chunk_python_syntax_error_falls_back_to_chunk_text():
    text = "def broken(:\n    pass\n"
    chunks = chunk_python(text)
    assert chunks
    assert all(name is None for _, name in chunks)
    assert any("def broken" in c for c, _ in chunks)


# ---------------------------------------------------------------------------
# chunk_document — dispatch shapes
# ---------------------------------------------------------------------------


def test_chunk_document_python_returns_def_name_tuples():
    text = "def foo():\n    return 1\n"
    chunks = chunk_document(Path("mod.py"), text)
    assert ("def foo():\n    return 1", "foo") in chunks


def test_chunk_document_markdown_returns_none_def_names():
    text = "# Title\n\nSome body text.\n"
    chunks = chunk_document(Path("doc.md"), text)
    assert chunks
    assert all(def_name is None for _, def_name in chunks)
    assert all(isinstance(c, str) and c for c, _ in chunks)


def test_chunk_document_other_extension_returns_none_def_names():
    text = "plain text content for a generic file\n" * 5
    chunks = chunk_document(Path("notes.txt"), text)
    assert chunks
    assert all(def_name is None for _, def_name in chunks)


# ---------------------------------------------------------------------------
# chunk_markdown — fenced code blocks
# ---------------------------------------------------------------------------


def test_chunk_markdown_does_not_split_on_comment_lines_inside_fences():
    text = "# Setup\n\nRun:\n\n```bash\n# install deps\nuv sync\n# start\nuv run app\n```\n"
    assert chunk_markdown(text) == [text.strip()]


def test_chunk_markdown_splits_on_headers_after_a_closed_fence():
    text = "# One\n\n~~~~\n# not a header\n~~~\nstill code\n~~~~\n\n# Two\n\nbody\n"
    chunks = chunk_markdown(text)
    assert len(chunks) == 2
    assert chunks[0].endswith("~~~~")
    assert chunks[1] == "# Two\n\nbody"


def test_chunk_markdown_fence_line_with_info_string_does_not_close_a_fence():
    text = "# A\n\n```\ncode\n```js\n# not a header\n```\n\n# B\nbody\n"
    chunks = chunk_markdown(text)
    assert len(chunks) == 2
    assert "# not a header" in chunks[0]


def test_chunk_markdown_closing_fence_must_be_at_least_as_long():
    closes = "# A\n\n```\ncode\n````\n# B\nbody\n"
    too_short = "# A\n\n````\ncode\n```\n# not a header\n````\n\n# B\nbody\n"
    assert len(chunk_markdown(closes)) == 2
    chunks = chunk_markdown(too_short)
    assert len(chunks) == 2
    assert "# not a header" in chunks[0]


def test_chunk_markdown_backtick_info_string_with_backticks_is_not_a_fence():
    text = "# A\n\n```x```\n\n# H\nbody\n"
    assert [c.split("\n", 1)[0] for c in chunk_markdown(text)] == ["# A", "# H"]


def test_chunk_markdown_tilde_fence_may_have_backticks_in_its_info_string():
    text = "# A\n\n~~~ `x`\n# not a header\n~~~\n\n# B\nbody\n"
    chunks = chunk_markdown(text)
    assert len(chunks) == 2
    assert "# not a header" in chunks[0]
