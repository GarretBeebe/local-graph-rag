"""FastAPI route handlers for the Graph RAG API."""

import asyncio
import time
from typing import Any

import bcrypt as _bcrypt
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

from local_graph_rag.settings import (
    ALLOW_INSECURE_LOCALONLY,
    CHAT_MODELS,
    SESSION_EXPIRY_SECONDS,
)
from local_graph_rag.web import user_store
from local_graph_rag.web.auth import create_session, is_valid_token, revoke_session
from local_graph_rag.web.openai_compat import build_chat_response, model_entry
from local_graph_rag.web.rag_executor import rag_stream_response, run_rag_with_timeout
from local_graph_rag.web.schemas import (
    ChatRequest,
    LoginRequest,
    extract_question_from_messages,
    validate_chat_request,
)
from local_graph_rag.web.security import (
    AUTH_COOKIE,
    extract_bearer_token,
    is_secure_request,
    password_fits_bcrypt,
)

_SERVER_START = int(time.time())
# Precomputed sentinel: login always runs bcrypt regardless of whether the username exists,
# preventing timing-based username enumeration.
_DUMMY_HASH: bytes = _bcrypt.hashpw(b"__sentinel__", _bcrypt.gensalt())

router = APIRouter()


@router.get("/v1/models")
@router.get("/models")
def models() -> dict[str, Any]:
    return {"object": "list", "data": [model_entry(m, _SERVER_START) for m in CHAT_MODELS]}


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
    if not password_fits_bcrypt(credentials.password):
        raise HTTPException(status_code=400, detail="Password exceeds bcrypt size limit")
    stored = user_store.get_hash(credentials.username)
    hash_to_check = stored.encode() if stored else _DUMMY_HASH
    password_matches = await asyncio.to_thread(
        _bcrypt.checkpw, credentials.password.encode(), hash_to_check
    )
    if not stored or not password_matches:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_session(credentials.username)
    response.set_cookie(
        AUTH_COOKIE,
        token,
        httponly=True,
        secure=is_secure_request(request),
        samesite="lax",
        max_age=SESSION_EXPIRY_SECONDS,
        path="/",
    )
    return {"ok": True}


@router.post("/auth/logout")
async def logout(request: Request, response: Response) -> dict[str, bool]:
    token = extract_bearer_token(request)
    if token:
        revoke_session(token)
    response.delete_cookie(AUTH_COOKIE, path="/")
    return {"ok": True}


@router.get("/auth/status")
def auth_status(request: Request) -> Response:
    authenticated = ALLOW_INSECURE_LOCALONLY or is_valid_token(extract_bearer_token(request))
    return JSONResponse(content={"authenticated": authenticated})
