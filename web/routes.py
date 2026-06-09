"""FastAPI route handlers for the Graph RAG API."""

import asyncio
import logging
import time
from typing import Any

import bcrypt as _bcrypt
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

import api.ollama_client as ollama_client
from settings import (
    ALLOW_INSECURE_LOCALONLY,
    GEN_MODEL,
    OLLAMA_MODEL_LIST_TIMEOUT_SECONDS,
    SESSION_EXPIRY_HOURS,
)
from web import user_store
from web.auth import create_session, is_valid_token, revoke_session
from web.middleware import _AUTH_COOKIE, _extract_bearer_token, _is_secure_request
from web.openai_compat import build_chat_response, model_entry
from web.rag_executor import get_rag_executor, rag_stream_response, run_rag_with_timeout
from web.schemas import (
    ChatRequest,
    LoginRequest,
    extract_question_from_messages,
    validate_chat_request,
)

logger = logging.getLogger(__name__)

_SERVER_START = int(time.time())
# Precomputed sentinel: login always runs bcrypt regardless of whether the username exists,
# preventing timing-based username enumeration.
_DUMMY_HASH: bytes = _bcrypt.hashpw(b"__sentinel__", _bcrypt.gensalt())

router = APIRouter()


@router.get("/v1/models")
@router.get("/models")
async def models() -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    try:
        resp = await loop.run_in_executor(
            get_rag_executor(),
            lambda: ollama_client.get("/api/tags", timeout=OLLAMA_MODEL_LIST_TIMEOUT_SECONDS),
        )
        resp.raise_for_status()
        data = [model_entry(m["name"], _SERVER_START) for m in resp.json().get("models", [])]
    except Exception:
        logger.warning("Failed to list Ollama models, returning default")
        data = [model_entry(GEN_MODEL, _SERVER_START)]
    return {"object": "list", "data": data}


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat(request: Request, req: ChatRequest) -> Response:
    validate_chat_request(req)
    question = extract_question_from_messages(req.messages)

    if req.stream:
        return StreamingResponse(
            rag_stream_response(question, req.model, req.graph_mode, request),
            media_type="text/event-stream",
        )

    answer = await run_rag_with_timeout(question, req.model, req.graph_mode)
    return build_chat_response(answer, req.model)


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@router.post("/auth/login")
async def login(
    request: Request, response: Response, credentials: LoginRequest
) -> dict[str, bool]:
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


@router.post("/auth/logout")
async def logout(request: Request, response: Response) -> dict[str, bool]:
    token = _extract_bearer_token(request)
    if token:
        revoke_session(token)
    response.delete_cookie(_AUTH_COOKIE, path="/")
    return {"ok": True}


@router.get("/auth/status")
def auth_status(request: Request) -> Response:
    authenticated = ALLOW_INSECURE_LOCALONLY or is_valid_token(_extract_bearer_token(request))
    return JSONResponse(content={"authenticated": authenticated})
