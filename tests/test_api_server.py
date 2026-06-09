"""Unit tests for web/api_server.py — auth, endpoints, streaming gate."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from local_graph_rag.common.sqlite_store import SqliteStore

_TEST_API_KEY = "test-bearer-key-abc"


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    import local_graph_rag.web.rate_limit as rl

    rl._rate_buckets.clear()
    rl._login_rate_buckets.clear()
    yield
    rl._rate_buckets.clear()
    rl._login_rate_buckets.clear()


@contextmanager
def _client_ctx(
    tmp_dir: Path,
    *,
    api_key: str = "",
    insecure: bool = False,
) -> Generator[TestClient, None, None]:
    """Context manager that yields a TestClient with mocked store/qdrant and a temp user DB."""
    temp_user_store = SqliteStore(tmp_dir / "users.sqlite3")
    mock_store = MagicMock()
    mock_qdrant = MagicMock()
    with (
        patch("local_graph_rag.web.api_server.GraphStore", return_value=mock_store),
        patch("local_graph_rag.web.api_server.get_qdrant_client", return_value=mock_qdrant),
        patch("local_graph_rag.web.middleware.ALLOW_INSECURE_LOCALONLY", insecure),
        patch("local_graph_rag.web.routes.ALLOW_INSECURE_LOCALONLY", insecure),
        patch("local_graph_rag.web.auth.API_KEY", api_key),
        patch("local_graph_rag.web.user_store._store", temp_user_store),
    ):
        from local_graph_rag.web.api_server import app
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


@pytest.fixture()
def authed_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    with _client_ctx(tmp_path, api_key=_TEST_API_KEY) as client:
        yield client


@pytest.fixture()
def insecure_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    with _client_ctx(tmp_path, insecure=True) as client:
        yield client


# ---------------------------------------------------------------------------
# Public endpoints (no auth required)
# ---------------------------------------------------------------------------


def test_healthz_is_public(authed_client):
    res = authed_client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_root_redirects_to_ui(authed_client):
    res = authed_client.get(
        "/",
        headers={"Authorization": f"Bearer {_TEST_API_KEY}"},
        follow_redirects=False,
    )
    assert res.status_code in (301, 302, 307, 308)
    assert res.headers["location"].startswith("/ui")


def test_auth_status_valid_bearer_returns_true(authed_client):
    res = authed_client.get("/auth/status", headers={"Authorization": f"Bearer {_TEST_API_KEY}"})
    assert res.status_code == 200
    assert res.json()["authenticated"] is True


def test_auth_status_no_token_returns_false(authed_client):
    res = authed_client.get("/auth/status")
    assert res.status_code == 200
    assert res.json()["authenticated"] is False


def test_auth_status_insecure_local_bypasses_check(insecure_client):
    res = insecure_client.get("/auth/status")
    assert res.status_code == 200
    assert res.json()["authenticated"] is True


# ---------------------------------------------------------------------------
# Auth enforcement on protected endpoints
# ---------------------------------------------------------------------------


def test_chat_no_token_returns_401(authed_client):
    res = authed_client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert res.status_code == 401


def test_chat_invalid_bearer_returns_401(authed_client):
    res = authed_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer wrong-key"},
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert res.status_code == 401


def test_models_no_token_returns_401(authed_client):
    res = authed_client.get("/v1/models")
    assert res.status_code == 401


def test_models_valid_bearer_returns_list(authed_client):
    with patch("local_graph_rag.web.routes.ollama_client.get") as mock_get:
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.return_value = {"models": [{"name": "test-model"}]}
        res = authed_client.get(
            "/v1/models", headers={"Authorization": f"Bearer {_TEST_API_KEY}"}
        )
    assert res.status_code == 200
    assert "data" in res.json()


# ---------------------------------------------------------------------------
# Login / logout
# ---------------------------------------------------------------------------


def test_login_invalid_credentials_returns_401(authed_client):
    res = authed_client.post(
        "/auth/login",
        json={"username": "nobody", "password": "wrong"},
    )
    assert res.status_code == 401


def test_logout_always_succeeds(authed_client):
    res = authed_client.post("/auth/logout")
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# Chat request validation (insecure mode — no auth needed)
# ---------------------------------------------------------------------------


def test_chat_missing_user_message_returns_400(insecure_client):
    res = insecure_client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "system", "content": "sys"}]},
    )
    assert res.status_code == 400


def test_chat_graph_mode_defaults_to_auto(insecure_client):
    """graph_mode defaults to 'auto' when not specified."""
    with patch("local_graph_rag.web.rag_executor.ask") as mock_ask:
        mock_ask.return_value = "answer"
        res = insecure_client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert res.status_code == 200
    assert mock_ask.call_args[0][2] == "auto"


def test_chat_explicit_graph_mode_is_forwarded(insecure_client):
    """graph_mode value is passed through to ask()."""
    with patch("local_graph_rag.web.rag_executor.ask") as mock_ask:
        mock_ask.return_value = "answer"
        insecure_client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "graph_mode": "local",
            },
        )
    assert mock_ask.call_args[0][2] == "local"
