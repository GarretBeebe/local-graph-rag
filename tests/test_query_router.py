"""Unit tests for the query router. No Ollama required."""

import pytest

from api.query_router import _heuristic, route_query


def test_route_returns_local_when_no_communities():
    assert route_query("what does GraphStore do?", communities_available=False) == "local"


def test_route_skips_llm_when_no_communities(monkeypatch):
    calls = []

    def _record(*a, **kw):
        calls.append(1)
        return "global"

    monkeypatch.setattr("api.query_router.ollama_client.generate", _record)
    route_query("what are the themes?", communities_available=False)
    assert calls == []


def test_route_uses_llm_global_response(monkeypatch):
    monkeypatch.setattr("api.query_router.ollama_client.generate", lambda *a, **kw: "global")
    assert route_query("what are the main themes?", communities_available=True) == "global"


def test_route_uses_llm_local_response(monkeypatch):
    monkeypatch.setattr("api.query_router.ollama_client.generate", lambda *a, **kw: "local")
    assert route_query("what does GraphStore do?", communities_available=True) == "local"


def test_route_falls_back_to_heuristic_on_llm_failure(monkeypatch):
    def _raise(*a, **kw):
        raise RuntimeError("ollama down")

    monkeypatch.setattr("api.query_router.ollama_client.generate", _raise)
    # "themes" is a global keyword → heuristic returns global
    result = route_query("what are the main themes?", communities_available=True)
    assert result == "global"


def test_route_falls_back_to_heuristic_on_unexpected_response(monkeypatch):
    monkeypatch.setattr("api.query_router.ollama_client.generate", lambda *a, **kw: "maybe")
    # "themes" → heuristic returns global
    result = route_query("what are the themes here?", communities_available=True)
    assert result == "global"


@pytest.mark.parametrize("question,expected", [
    ("what are the main themes?", "global"),
    ("give me an overview of the project", "global"),
    ("summarize the codebase", "global"),
    ("what does GraphStore do?", "local"),
    ("how does chunking work?", "local"),
    ("what is the relationship between extractor and store?", "local"),
])
def test_heuristic_classifies_correctly(question: str, expected: str):
    assert _heuristic(question) == expected
