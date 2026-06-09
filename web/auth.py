"""Session creation and validation for API key and opaque session bearer auth."""

import hmac

from settings import API_KEY, SESSION_EXPIRY_HOURS
from web import user_store


def create_session(username: str) -> str:
    return user_store.create_session(username, SESSION_EXPIRY_HOURS)


def revoke_session(token: str) -> None:
    user_store.delete_session(token)


def is_valid_token(token: str) -> bool:
    if not token:
        return False
    if API_KEY and hmac.compare_digest(token, API_KEY):
        return True
    return user_store.validate_session(token) is not None
