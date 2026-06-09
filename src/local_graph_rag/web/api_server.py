"""
OpenAI-compatible chat completions endpoint backed by the local Graph RAG pipeline.

Run with:

    uvicorn local_graph_rag.web.api_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from local_graph_rag.common.qdrant import get_qdrant_client
from local_graph_rag.graph.store import GraphStore
from local_graph_rag.settings import (
    ALLOW_INSECURE_LOCALONLY,
    CORS_ORIGINS,
    GENERATION_CONCURRENCY_LIMIT,
    RAG_EXECUTOR_WORKERS,
)
from local_graph_rag.web import user_store
from local_graph_rag.web.middleware import security_headers_middleware, security_middleware
from local_graph_rag.web.rag_executor import init_rag_executor, init_stores, shutdown_rag_executor
from local_graph_rag.web.rate_limit import start_sweep_tasks
from local_graph_rag.web.routes import router

logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).parent
_STATIC_DIR = _WEB_DIR / "static"
_SESSION_PURGE_INTERVAL_SECONDS = 3600


async def _purge_sessions_periodically() -> None:
    while True:
        await asyncio.sleep(_SESSION_PURGE_INTERVAL_SECONDS)
        try:
            user_store.purge_expired_sessions()
        except Exception as exc:
            logger.warning("Periodic session purge failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    store = GraphStore()
    client = get_qdrant_client()
    init_rag_executor(RAG_EXECUTOR_WORKERS, GENERATION_CONCURRENCY_LIMIT)
    init_stores(store, client)

    user_store.init_db()
    try:
        user_store.purge_expired_sessions()
    except Exception as exc:
        logger.warning("Failed to purge expired sessions on startup: %s", exc)

    if ALLOW_INSECURE_LOCALONLY:
        logger.warning(
            "Authentication is DISABLED for local-only mode because ALLOW_INSECURE_LOCALONLY=true"
        )

    sweep_tasks = await start_sweep_tasks()
    sweep_tasks.append(asyncio.create_task(_purge_sessions_periodically()))
    try:
        yield
    finally:
        for t in sweep_tasks:
            t.cancel()
        for t in sweep_tasks:
            with suppress(asyncio.CancelledError):
                await t
        store.close()
        shutdown_rag_executor()


app = FastAPI(title="Graph RAG API", lifespan=lifespan)
app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")

# Middleware registration — LIFO: last registered = outermost layer.
app.middleware("http")(security_middleware)          # innermost — processes auth
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "User-Agent"],
)
app.middleware("http")(security_headers_middleware)  # outermost — always adds security headers

app.include_router(router)
