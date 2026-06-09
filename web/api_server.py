"""
OpenAI-compatible chat completions endpoint backed by the local Graph RAG pipeline.

Implements POST /v1/chat/completions so OpenAI-compatible clients
(Chatbox, Open WebUI, LangChain, etc.) can query the local knowledge base.

Run with:

    uvicorn web.api_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import concurrent.futures
import ipaddress
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import bcrypt as _bcrypt
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient

import api.ollama_client as ollama_client  # noqa: E402 (side-effectful import)
from api.query_graph_rag import ask, ask_stream_sync
from common.qdrant import get_qdrant_client
from graph.store import GraphStore
from settings import (
    ALLOW_INSECURE_LOCALONLY,
    CORS_ORIGINS,
    GEN_MODEL,
    GENERATION_CONCURRENCY_LIMIT,
    OLLAMA_MODEL_LIST_TIMEOUT_SECONDS,
    RAG_EXECUTOR_WORKERS,
    RAG_REQUEST_TIMEOUT_SECONDS,
    SESSION_EXPIRY_HOURS,
    STREAM_TIMEOUT_SECONDS,
    TRUSTED_PROXY_IPS,
)
from web import user_store
from web.auth import create_session, is_valid_token, revoke_session
from web.openai_compat import build_chat_response, make_stream_chunk, model_entry
from web.rate_limit import check_login_rate_limit, check_rate_limit, start_sweep_tasks
from web.schemas import (
    ChatRequest,
    LoginRequest,
    extract_question_from_messages,
    validate_chat_request,
)

logger = logging.getLogger(__name__)

_SERVER_START = int(time.time())
_WEB_DIR = Path(__file__).parent
_STATIC_DIR = _WEB_DIR / "static"
_AUTH_COOKIE = "rag_token"
_DISCONNECT_POLL_SECONDS = 2.0
_RAG_CAPACITY_TIMEOUT_DETAIL = "RAG pipeline timed out waiting for capacity."
# Precomputed sentinel so login always runs bcrypt regardless of whether the username exists,
# preventing timing-based username enumeration.
_DUMMY_HASH: bytes = _bcrypt.hashpw(b"__sentinel__", _bcrypt.gensalt())

# Initialized in lifespan after the event loop is running.
_RAG_EXECUTOR: ThreadPoolExecutor | None = None
_RAG_CONCURRENCY: asyncio.Semaphore | None = None
_store: GraphStore | None = None
_client: QdrantClient | None = None


def _get_rag_executor() -> ThreadPoolExecutor:
    if _RAG_EXECUTOR is None:
        raise RuntimeError("RAG executor has not been initialized — lifespan not started")
    return _RAG_EXECUTOR


def _get_rag_concurrency() -> asyncio.Semaphore:
    if _RAG_CONCURRENCY is None:
        raise RuntimeError(
            "RAG concurrency limiter has not been initialized — lifespan not started"
        )
    return _RAG_CONCURRENCY


def _get_store() -> GraphStore:
    if _store is None:
        raise RuntimeError("GraphStore has not been initialized — lifespan not started")
    return _store


def _get_client() -> QdrantClient:
    if _client is None:
        raise RuntimeError("QdrantClient has not been initialized — lifespan not started")
    return _client


def resolve_client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    if peer in TRUSTED_PROXY_IPS:
        forwarded = request.headers.get("X-Forwarded-For", "")
        first = forwarded.split(",", 1)[0].strip()
        if first:
            try:
                ipaddress.ip_address(first)
                return first
            except ValueError:
                pass
    return peer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _RAG_EXECUTOR, _RAG_CONCURRENCY, _store, _client
    _RAG_EXECUTOR = ThreadPoolExecutor(max_workers=RAG_EXECUTOR_WORKERS)
    _RAG_CONCURRENCY = asyncio.Semaphore(GENERATION_CONCURRENCY_LIMIT)
    _store = GraphStore()
    _client = get_qdrant_client()

    user_store.init_db()
    try:
        user_store.purge_expired_sessions()
    except Exception as exc:
        logger.warning("Failed to purge expired sessions on startup: %s", exc)

    if ALLOW_INSECURE_LOCALONLY:
        logger.warning(
            "Authentication is DISABLED for local-only mode because ALLOW_INSECURE_LOCALONLY=true"
        )
    if GENERATION_CONCURRENCY_LIMIT > 1:
        logger.warning(
            "GENERATION_CONCURRENCY_LIMIT=%d: GraphStore uses a shared SQLite connection "
            "not safe for concurrent access. Keep at 1 until GraphStore is thread-hardened.",
            GENERATION_CONCURRENCY_LIMIT,
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
        if _store is not None:
            _store.close()
        futs = [
            _get_rag_executor().submit(ollama_client.close_session)
            for _ in range(RAG_EXECUTOR_WORKERS)
        ]
        for f in futs:
            with suppress(Exception):
                f.result(timeout=2)
        _get_rag_executor().shutdown(wait=True)


app = FastAPI(title="Graph RAG API", lifespan=lifespan)
app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")


def _extract_bearer_token(request: Request) -> str:
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    return token or request.cookies.get(_AUTH_COOKIE, "")


def _is_secure_request(request: Request) -> bool:
    """Return True if HTTPS; trusts X-Forwarded-Proto only from TRUSTED_PROXY_IPS."""
    if request.url.scheme == "https":
        return True
    peer = request.client.host if request.client else ""
    if peer in TRUSTED_PROXY_IPS:
        proto = request.headers.get("X-Forwarded-Proto", "").split(",", 1)[0].strip()
        return proto == "https"
    return False


async def _purge_sessions_periodically() -> None:
    while True:
        await asyncio.sleep(3600)
        try:
            user_store.purge_expired_sessions()
        except Exception as exc:
            logger.warning("Periodic session purge failed: %s", exc)


@app.middleware("http")
async def security_middleware(request: Request, call_next: Callable[..., Any]) -> Response:
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    logger.info("[%s] %s %s", request_id, request.method, request.url.path)

    if (
        request.url.path == "/favicon.ico"
        or request.url.path == "/ui"
        or request.url.path.startswith("/ui/")
    ):
        return await call_next(request)
    if request.url.path == "/healthz":
        return await call_next(request)

    client_ip = resolve_client_ip(request)
    if request.url.path == "/auth/login":
        if not await check_login_rate_limit(client_ip):
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        return await call_next(request)

    if not await check_rate_limit(client_ip):
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    if request.url.path == "/auth/logout":
        return await call_next(request)

    if request.url.path == "/auth/status":
        return await call_next(request)

    if not ALLOW_INSECURE_LOCALONLY and not is_valid_token(_extract_bearer_token(request)):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "User-Agent"],
)


@app.middleware("http")
async def _security_headers_middleware(request: Request, call_next: Callable[..., Any]) -> Response:
    try:
        response = await call_next(request)
    except Exception:
        response = Response(status_code=500)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'"
    )
    return response


async def _wait_for_capacity(timeout: float) -> asyncio.Semaphore:
    """Acquire the RAG semaphore, raising TimeoutError if capacity is not available in time."""
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
        future = loop.run_in_executor(_get_rag_executor(), fn, *args)
    except BaseException:
        semaphore.release()
        raise
    future.add_done_callback(lambda _f: semaphore.release())
    return future


async def _acquire_and_submit(
    fn: Callable[[], Any],
    timeout: float = RAG_REQUEST_TIMEOUT_SECONDS,
) -> tuple[asyncio.Future[Any], float]:
    """Acquire the RAG semaphore, compute remaining time, and submit fn to the executor."""
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


async def _run_rag_with_timeout(
    question: str,
    model: str,
    graph_mode: str = "auto",
    timeout: float = RAG_REQUEST_TIMEOUT_SECONDS,
) -> str:
    """Execute the RAG pipeline with a timeout and bounded in-flight work."""
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
    """Acquire the semaphore and return (queue, cancel_event, future)."""
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
    """Poll for client disconnect and set cancel_event when detected."""
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
    """Drain the worker queue, mapping exceptions and timeouts to SSE error chunks."""
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


async def _rag_stream_response(
    question: str,
    model: str,
    graph_mode: str = "auto",
    http_request: Request | None = None,
) -> AsyncIterator[str]:
    """Bridge ask_stream_sync (sync generator) to an async SSE generator."""
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


@app.get("/v1/models")
@app.get("/models")
async def models() -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    try:
        resp = await loop.run_in_executor(
            _get_rag_executor(),
            lambda: ollama_client.get("/api/tags", timeout=OLLAMA_MODEL_LIST_TIMEOUT_SECONDS),
        )
        resp.raise_for_status()
        data = [model_entry(m["name"], _SERVER_START) for m in resp.json().get("models", [])]
    except Exception:
        logger.warning("Failed to list Ollama models, returning default")
        data = [model_entry(GEN_MODEL, _SERVER_START)]
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat(request: Request, req: ChatRequest) -> Response:
    validate_chat_request(req)
    question = extract_question_from_messages(req.messages)

    if req.stream:
        return StreamingResponse(
            _rag_stream_response(question, req.model, req.graph_mode, request),
            media_type="text/event-stream",
        )

    answer = await _run_rag_with_timeout(question, req.model, req.graph_mode)
    return build_chat_response(answer, req.model)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.post("/auth/login")
async def login(request: Request, response: Response, credentials: LoginRequest) -> dict[str, bool]:
    stored = user_store.get_hash(credentials.username)
    hash_to_check = stored.encode() if stored else _DUMMY_HASH
    password_matches = await asyncio.to_thread(
        _bcrypt.checkpw, credentials.password.encode(), hash_to_check
    )
    if not stored or not password_matches:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_session(credentials.username)
    response.set_cookie(
        _AUTH_COOKIE,
        token,
        httponly=True,
        secure=_is_secure_request(request),
        samesite="lax",
        max_age=SESSION_EXPIRY_HOURS * 3600,
        path="/",
    )
    return {"ok": True}


@app.post("/auth/logout")
async def logout(request: Request, response: Response) -> dict[str, bool]:
    token = _extract_bearer_token(request)
    if token:
        revoke_session(token)
    response.delete_cookie(_AUTH_COOKIE, path="/")
    return {"ok": True}


@app.get("/auth/status")
def auth_status(request: Request) -> Response:
    authenticated = ALLOW_INSECURE_LOCALONLY or is_valid_token(_extract_bearer_token(request))
    return JSONResponse(content={"authenticated": authenticated})
