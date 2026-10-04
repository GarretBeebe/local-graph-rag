"""Shared test helpers."""

from collections.abc import Callable
from typing import Any

from pytest import MonkeyPatch

from local_graph_rag.graph.store import GraphStore

ZERO_VECTOR_768 = [0.0] * 768
LOCAL_EMBED_TARGET = "local_graph_rag.rag.local_retrieval.embed"
GLOBAL_EMBED_TARGET = "local_graph_rag.rag.global_retrieval.embed"
INDEX_CONFIG_TARGET = "local_graph_rag.ingest.doc_config.INDEX_CONFIG_PATH"
OLLAMA_GENERATE_TARGET = "local_graph_rag.rag.ollama_client.generate"
ROUTER_GENERATE_TARGET = "local_graph_rag.rag.query_router.ollama_client.generate"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
AUTHORIZATION_HEADER = "Authorization"
EMPTY_EXTRACTION_JSON = '{"entities": [], "relationships": []}'


def patch_local_embed(
    monkeypatch: MonkeyPatch,
    vector: list[float] | None = None,
) -> None:
    monkeypatch.setattr(LOCAL_EMBED_TARGET, lambda *a, **kw: vector or ZERO_VECTOR_768)


def patch_global_embed(monkeypatch: MonkeyPatch, vector: list[float]) -> None:
    monkeypatch.setattr(GLOBAL_EMBED_TARGET, lambda *a, **kw: vector)


def patch_index_config_path(monkeypatch: MonkeyPatch, path: object) -> None:
    monkeypatch.setattr(INDEX_CONFIG_TARGET, str(path))


def patch_ollama_generate(monkeypatch: MonkeyPatch, fn: Callable[..., str]) -> None:
    monkeypatch.setattr(OLLAMA_GENERATE_TARGET, fn)


def patch_router_generate(monkeypatch: MonkeyPatch, fn: Callable[..., str]) -> None:
    monkeypatch.setattr(ROUTER_GENERATE_TARGET, fn)


def add_entity(
    store: GraphStore,
    name: str,
    *,
    type: str | None = None,
    description: str | None = None,
    doc: str = "doc.py",
) -> str:
    """Record one entity mention from `doc` and return its slug id."""
    return store.upsert_entities([{"name": name, "type": type, "description": description}], doc)[0]


def add_relationship(
    store: GraphStore, source: str, target: str, label: str, doc: str = "doc.py"
) -> None:
    store.upsert_relationships([(source, target, label, doc)])


def bearer_headers(token: str) -> dict[str, str]:
    return {AUTHORIZATION_HEADER: f"Bearer {token}"}


def chat_payload(
    content: str = "hi",
    *,
    role: str = "user",
    model: str = "test",
    **overrides: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": role, "content": content}],
    }
    payload.update(overrides)
    return payload
