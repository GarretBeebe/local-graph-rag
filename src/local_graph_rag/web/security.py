"""Shared web security helpers."""

import ipaddress

from fastapi import Request

from local_graph_rag.settings import TRUSTED_PROXY_IPS

AUTH_COOKIE = "rag_token"
BCRYPT_MAX_PASSWORD_BYTES = 72


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


def extract_bearer_token(request: Request) -> str:
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    return token or request.cookies.get(AUTH_COOKIE, "")


def is_secure_request(request: Request) -> bool:
    """Return True if HTTPS; trusts X-Forwarded-Proto only from TRUSTED_PROXY_IPS."""
    if request.url.scheme == "https":
        return True
    peer = request.client.host if request.client else ""
    if peer in TRUSTED_PROXY_IPS:
        proto = request.headers.get("X-Forwarded-Proto", "").split(",", 1)[0].strip()
        return proto == "https"
    return False


def password_fits_bcrypt(password: str) -> bool:
    return len(password.encode()) <= BCRYPT_MAX_PASSWORD_BYTES

