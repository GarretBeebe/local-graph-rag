"""Auth and security middleware for the Graph RAG API."""

import logging
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from local_graph_rag.settings import ALLOW_INSECURE_LOCALONLY
from local_graph_rag.web.auth import is_valid_token
from local_graph_rag.web.rate_limit import check_login_rate_limit, check_rate_limit
from local_graph_rag.web.security import extract_bearer_token, resolve_client_ip

logger = logging.getLogger(__name__)


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

    if not ALLOW_INSECURE_LOCALONLY and not is_valid_token(extract_bearer_token(request)):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)


async def security_headers_middleware(
    request: Request, call_next: Callable[..., Any]
) -> Response:
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled exception in request handler")
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
