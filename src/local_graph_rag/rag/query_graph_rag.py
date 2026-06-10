"""End-to-end Graph RAG query: route → retrieve → generate."""

import logging
import sys
import threading
from collections.abc import Iterator
from typing import Literal

from qdrant_client import QdrantClient

import local_graph_rag.rag.ollama_client as ollama_client
from local_graph_rag.graph.store import GraphStore
from local_graph_rag.rag.global_retrieval import GlobalContext, global_retrieve
from local_graph_rag.rag.local_retrieval import LocalContext, local_retrieve
from local_graph_rag.rag.query_router import route_query
from local_graph_rag.settings import GEN_MODEL

logger = logging.getLogger(__name__)

GraphMode = Literal["auto", "local", "global"]

_LOCAL_PROMPT = """\
Use the knowledge graph context below to answer the question. If the context is \
missing, incomplete, or not relevant, answer using your own knowledge instead.

Entities:
{entity_block}

Relationships:
{relationship_block}

Supporting text:
{chunk_block}

Question: {question}
Answer:"""

_GLOBAL_PROMPT = """\
Use the community summaries below to answer the question. If the summaries are \
missing, incomplete, or not relevant, answer using your own knowledge instead.

{summary_block}

Question: {question}
Answer:"""

# local_retrieve can return up to k + _MAX_IDENTIFIER_LOOKUPS chunks (vector hits
# plus def_name exact-matches), so cap the joined block to bound prompt size.
_MAX_CHUNK_BLOCK_CHARS = 10_000


def _format_local(ctx: LocalContext, question: str) -> str:
    names = {e["id"]: e["name"] for e in ctx.entities}
    entity_block = "\n".join(
        f"- {e['name']} ({e.get('type') or 'unknown'}): {e.get('description') or ''}"
        for e in ctx.entities
    ) or "(none)"
    rel_block = "\n".join(
        f"- {names.get(r['source_id'], r['source_id'])} --[{r['label']}]--> "
        f"{names.get(r['target_id'], r['target_id'])}"
        for r in ctx.relationships
    ) or "(none)"
    parts: list[str] = []
    total = 0
    for text in ctx.chunk_texts:
        if parts and total + len(text) > _MAX_CHUNK_BLOCK_CHARS:
            break
        parts.append(text)
        total += len(text)
    chunk_block = "\n\n".join(parts) or "(none)"
    return _LOCAL_PROMPT.format(
        entity_block=entity_block,
        relationship_block=rel_block,
        chunk_block=chunk_block,
        question=question,
    )


def _format_global(ctx: GlobalContext, question: str) -> str:
    summary_block = "\n\n".join(
        f"Community {cid}:\n{summary}"
        for cid, summary in zip(ctx.community_ids, ctx.community_summaries, strict=True)
    )
    return _GLOBAL_PROMPT.format(summary_block=summary_block, question=question)


def _build_prompt(
    question: str,
    graph_mode: GraphMode,
    store: GraphStore,
    client: QdrantClient,
    cancel: threading.Event | None = None,
) -> str:
    communities: list[dict] = store.get_communities() if graph_mode != "local" else []
    communities_available = any(
        c["embedding"] is not None and c["summary"] for c in communities
    )

    use_global = graph_mode == "global" or (
        graph_mode == "auto"
        and route_query(question, communities_available, cancel=cancel) == "global"
    )

    if use_global:
        ctx = global_retrieve(question, store, communities=communities)
        if not ctx.community_summaries:
            logger.warning("Global context empty — falling back to local retrieval")
            return _format_local(local_retrieve(question, store, client), question)
        return _format_global(ctx, question)

    return _format_local(local_retrieve(question, store, client), question)


def ask(
    question: str,
    model: str,
    graph_mode: GraphMode,
    store: GraphStore,
    client: QdrantClient,
    cancel: threading.Event | None = None,
) -> str:
    """Return a complete answer for the question."""
    prompt = _build_prompt(question, graph_mode, store, client, cancel=cancel)
    return ollama_client.generate(prompt, model, cancel=cancel)


def ask_stream_sync(
    question: str,
    model: str,
    graph_mode: GraphMode,
    store: GraphStore,
    client: QdrantClient,
    cancel: threading.Event | None = None,
) -> Iterator[str]:
    """Yield text chunks for a streaming answer."""
    prompt = _build_prompt(question, graph_mode, store, client, cancel=cancel)
    yield from ollama_client.stream_generate(prompt, model, cancel=cancel)


_VALID_MODES = ("auto", "local", "global")


def _validate_mode(value: str) -> GraphMode:
    """Validate a --mode CLI value, exiting with an error message if it's not recognized."""
    if value not in _VALID_MODES:
        print(f"Error: invalid --mode {value!r}; must be one of: {', '.join(_VALID_MODES)}")
        sys.exit(1)
    return value  # type: ignore[return-value]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    if len(sys.argv) < 2:
        print(
            "Usage: python -m local_graph_rag.rag.query_graph_rag "
            "<question> [--mode local|global|auto]"
        )
        sys.exit(1)

    question = sys.argv[1]
    graph_mode: GraphMode = "auto"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            graph_mode = _validate_mode(sys.argv[idx + 1])
        else:
            print("Error: --mode requires a value (auto, local, or global)")
            sys.exit(1)

    from local_graph_rag.common.qdrant import get_qdrant_client

    store = GraphStore()
    client = get_qdrant_client()
    try:
        for chunk in ask_stream_sync(question, GEN_MODEL, graph_mode, store, client):
            print(chunk, end="", flush=True)
        print()
    finally:
        store.close()


if __name__ == "__main__":
    main()
