"""RAG pipeline executor: global state, thread pool management, and streaming bridge."""

import asyncio
import concurrent.futures
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any, TypeVar

from fastapi import HTTPException, Request
from qdrant_client import QdrantClient

import api.ollama_client as ollama_client
from api.query_graph_rag import ask, ask_stream_sync
from graph.store import GraphStore
from settings import RAG_EXECUTOR_WORKERS, RAG_REQUEST_TIMEOUT_SECONDS, STREAM_TIMEOUT_SECONDS
from web.openai_compat import make_stream_chunk

logger = logging.getLogger(__name__)

_DISCONNECT_POLL_SECONDS = 2.0
_RAG_CAPACITY_TIMEOUT_DETAIL = "RAG pipeline timed out waiting for capacity."

_RAG_EXECUTOR: ThreadPoolExecutor | None = None
_RAG_CONCURRENCY: asyncio.Semaphore | None = None
_store: GraphStore | None = None
_client: QdrantClient | None = None


def init_rag_executor(workers: int, concurrency_limit: int) -> None:
    global _RAG_EXECUTOR, _RAG_CONCURRENCY
    _RAG_EXECUTOR = ThreadPoolExecutor(max_workers=workers)
    effective = concurrency_limit
    if effective > 1:
        logger.warning(
            "GENERATION_CONCURRENCY_LIMIT=%d: GraphStore uses a shared SQLite connection "
            "unsafe for concurrent access; clamping to 1 until GraphStore is thread-hardened.",
            effective,
        )
        effective = 1
    _RAG_CONCURRENCY = asyncio.Semaphore(effective)


def init_stores(store: GraphStore, client: QdrantClient) -> None:
    global _store, _client
    _store = store
    _client = client


def shutdown_rag_executor() -> None:
    futs = [
        get_rag_executor().submit(ollama_client.close_session)
        for _ in range(RAG_EXECUTOR_WORKERS)
    ]
    for f in futs:
        with suppress(Exception):
            f.result(timeout=2)
    get_rag_executor().shutdown(wait=True)


_T = TypeVar("_T")


def _require_initialized(value: _T | None, name: str) -> _T:
    if value is None:
        raise RuntimeError(f"{name} has not been initialized — lifespan not started")
    return value


def get_rag_executor() -> ThreadPoolExecutor:
    return _require_initialized(_RAG_EXECUTOR, "RAG executor")


def _get_rag_concurrency() -> asyncio.Semaphore:
    return _require_initialized(_RAG_CONCURRENCY, "RAG concurrency limiter")


def _get_store() -> GraphStore:
    return _require_initialized(_store, "GraphStore")


def _get_client() -> QdrantClient:
    return _require_initialized(_client, "QdrantClient")


async def _wait_for_capacity(timeout: float) -> asyncio.Semaphore:
    semaphore = _get_rag_concurrency()
    await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
    return semaphore


def _submit_rag_job(
    semaphore: asyncio.Semaphore,
    fn: Callable[..., Any],
    *args: Any,
) -> asyncio.Future[Any]:
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(get_rag_executor(), fn, *args)
    except BaseException:
        semaphore.release()
        raise
    future.add_done_callback(lambda _f: semaphore.release())
    return future


async def _acquire_and_submit(
    fn: Callable[[], Any],
    timeout: float = RAG_REQUEST_TIMEOUT_SECONDS,
) -> tuple[asyncio.Future[Any], float]:
    started = time.monotonic()
    try:
        semaphore = await _wait_for_capacity(timeout)
    except TimeoutError:
        logger.warning("RAG pipeline timed out waiting for capacity after %.1fs", timeout)
        raise HTTPException(status_code=504, detail=_RAG_CAPACITY_TIMEOUT_DETAIL) from None
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        semaphore.release()
        raise HTTPException(status_code=504, detail=_RAG_CAPACITY_TIMEOUT_DETAIL)
    return _submit_rag_job(semaphore, fn), remaining


async def run_rag_with_timeout(
    question: str,
    model: str,
    graph_mode: str = "auto",
    timeout: float = RAG_REQUEST_TIMEOUT_SECONDS,
) -> str:
    cancel_event = threading.Event()
    store = _get_store()
    client = _get_client()
    future, remaining = await _acquire_and_submit(
        lambda: ask(question, model, graph_mode, store, client, cancel_event), timeout
    )
    try:
        answer = await asyncio.wait_for(asyncio.shield(future), timeout=remaining)
        return answer.strip()
    except TimeoutError:
        cancel_event.set()
        logger.warning("RAG pipeline timed out after %.1fs", timeout)
        raise HTTPException(
            status_code=504,
            detail="RAG pipeline timed out while generating an answer.",
        ) from None
    except Exception as e:
        cancel_event.set()
        logger.exception("RAG pipeline error")
        raise HTTPException(status_code=500, detail="RAG pipeline error") from e


async def _start_stream_worker(
    question: str,
    model: str,
    graph_mode: str,
) -> tuple[asyncio.Queue[str | Exception | None], threading.Event, asyncio.Future[Any]]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | Exception | None] = asyncio.Queue(maxsize=32)
    cancel_event = threading.Event()
    store = _get_store()
    client = _get_client()

    _put_timeout = STREAM_TIMEOUT_SECONDS
    _put_errors = (RuntimeError, concurrent.futures.TimeoutError, concurrent.futures.CancelledError)

    def _run() -> None:
        try:
            for text in ask_stream_sync(question, model, graph_mode, store, client, cancel_event):
                coro = queue.put(text)
                try:
                    asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=_put_timeout)
                except _put_errors:
                    coro.close()
                    return
        except Exception as exc:
            coro = queue.put(exc)
            try:
                asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=_put_timeout)
            except _put_errors:
                coro.close()
        finally:
            coro = queue.put(None)
            try:
                asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=_put_timeout)
            except _put_errors:
                coro.close()

    started = time.monotonic()
    semaphore = await _wait_for_capacity(RAG_REQUEST_TIMEOUT_SECONDS)
    remaining = RAG_REQUEST_TIMEOUT_SECONDS - (time.monotonic() - started)
    if remaining <= 0:
        semaphore.release()
        raise TimeoutError
    future = _submit_rag_job(semaphore, _run)
    return queue, cancel_event, future


async def _watch_disconnect(request: Request, cancel_event: threading.Event) -> None:
    while not cancel_event.is_set():
        await asyncio.sleep(_DISCONNECT_POLL_SECONDS)
        if await request.is_disconnected():
            cancel_event.set()
            logger.info("Client disconnected — cancelling stream")
            return


async def _stream_queue_events(
    queue: asyncio.Queue[str | Exception | None],
    cancel_event: threading.Event,
    request_id: str,
    created: int,
    model: str,
) -> AsyncIterator[str]:
    try:
        while True:
            item = await asyncio.wait_for(queue.get(), timeout=STREAM_TIMEOUT_SECONDS)
            if item is None:
                break
            if isinstance(item, Exception):
                logger.error("RAG stream error: %s", item)
                yield make_stream_chunk(
                    request_id, created, model, content="\n\n[Generation error — please retry]"
                )
                break
            yield make_stream_chunk(request_id, created, model, content=item)
    except TimeoutError:
        cancel_event.set()
        logger.warning("RAG stream timed out waiting for next chunk")
        yield make_stream_chunk(
            request_id, created, model, content="\n\n[Error: generation timed out]"
        )


async def rag_stream_response(
    question: str,
    model: str,
    graph_mode: str = "auto",
    http_request: Request | None = None,
) -> AsyncIterator[str]:
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    try:
        queue, cancel_event, _ = await _start_stream_worker(question, model, graph_mode)
    except TimeoutError:
        logger.warning("RAG stream timed out waiting for capacity")
        yield make_stream_chunk(
            request_id, created, model, content="\n\n[Error: server at capacity, please retry]"
        )
        yield make_stream_chunk(request_id, created, model, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    disconnect_task = (
        asyncio.create_task(_watch_disconnect(http_request, cancel_event))
        if http_request is not None
        else None
    )

    try:
        async for chunk in _stream_queue_events(queue, cancel_event, request_id, created, model):
            yield chunk
    finally:
        cancel_event.set()
        if disconnect_task is not None:
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task

    yield make_stream_chunk(request_id, created, model, finish_reason="stop")
    yield "data: [DONE]\n\n"
