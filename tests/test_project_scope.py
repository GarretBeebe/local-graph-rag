"""Unit tests for rag/project_scope.py. No Ollama or Qdrant required."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import local_graph_rag.rag.project_scope as project_scope
from local_graph_rag.rag.project_scope import match_projects, project_paths

_NAMES = [
    "local-graph-rag",
    "local-research-assistant",
    "local-code-assistant",
    "rag-system",
    "vectorless-rag",
]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # A name written as an identifier always counts, whatever joins its words.
        ("In local-graph-rag, what is CHAT_MODELS?", ["local-graph-rag"]),
        ("what changed in local_graph_rag?", ["local-graph-rag"]),
        ("How does LocalGraphRag route questions?", ["local-graph-rag"]),
        ("compare rag-system with local-graph-rag", ["local-graph-rag", "rag-system"]),
        # Plain words and parts of a name need a cue word...
        ("tell me about recent changes to the graph rag project", ["local-graph-rag"]),
        ("what does the research assistant repo do?", ["local-research-assistant"]),
        ("the local code assistant codebase", ["local-code-assistant"]),
        # ...so general questions stay unscoped.
        ("what is a graph rag?", []),
        ("how does a rag system rank results?", []),
        # A part shared by several names, or a single word, names nothing.
        ("the local assistant project", []),
        ("the rag project", []),
        # A name inside a longer word doesn't count.
        ("see mylocal-graph-ragfork", []),
    ],
)
def test_match_projects(question: str, expected: list[str]):
    assert match_projects(question, _NAMES) == expected


def test_match_projects_one_word_name_needs_a_cue():
    assert match_projects("what is lakehouse?", ["lakehouse"]) == []
    assert match_projects("what is the lakehouse project?", ["lakehouse"]) == ["lakehouse"]


def test_project_paths_returns_the_named_projects_files_and_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    code, history = tmp_path / "code", tmp_path / "history"
    monkeypatch.setattr(project_scope, "_project_roots", lambda: (code,))
    monkeypatch.setattr(project_scope, "GIT_HISTORY_DIR", history)
    indexed = [
        str(code / "local-graph-rag" / "src" / "a.py"),
        str(code / "local-graph-rag" / "README.md"),
        str(code / "rag-system" / "b.py"),
        str(code / "top-level-file.md"),
        str(history / "local-graph-rag.md"),
        str(history / "rag-system.md"),
        str(tmp_path / "notes" / "local-graph-rag.md"),
    ]
    store = SimpleNamespace(list_all_paths=lambda: indexed)

    paths = project_paths("what changed in local-graph-rag?", store)

    assert paths == [indexed[0], indexed[1], indexed[4]]
    assert project_paths("what is a graph rag?", store) is None


def test_project_scoping_is_off_when_the_index_config_fails(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    def _broken_config():
        raise ValueError("bad yaml")

    monkeypatch.setattr(project_scope, "load_index_config", _broken_config)
    store = SimpleNamespace(list_all_paths=lambda: ["/code/local-graph-rag/a.py"])

    assert project_paths("what changed in local-graph-rag?", store) is None
    assert "Project scoping is off" in caplog.text
