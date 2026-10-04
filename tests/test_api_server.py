"""Unit tests for web/api_server.py — auth, endpoints, streaming gate."""

import asyncio
import hashlib
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import bcrypt
import pytest
from fastapi.testclient import TestClient

import local_graph_rag.web.rag_executor as rag_executor
from local_graph_rag.common.sqlite_store import SqliteStore
from local_graph_rag.web import user_store
from tests.helpers import CHAT_COMPLETIONS_PATH, bearer_headers, chat_payload

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
        headers=bearer_headers(_TEST_API_KEY),
        follow_redirects=False,
    )
    assert res.status_code in (301, 302, 307, 308)
    assert res.headers["location"].startswith("/ui")


def test_auth_status_valid_bearer_returns_true(authed_client):
    res = authed_client.get("/auth/status", headers=bearer_headers(_TEST_API_KEY))
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
        CHAT_COMPLETIONS_PATH,
        json=chat_payload(),
    )
    assert res.status_code == 401


def test_chat_invalid_bearer_returns_401(authed_client):
    res = authed_client.post(
        CHAT_COMPLETIONS_PATH,
        headers=bearer_headers("wrong-key"),
        json=chat_payload(),
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
            "/v1/models", headers=bearer_headers(_TEST_API_KEY)
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


def test_login_rejects_password_over_bcrypt_byte_limit(authed_client):
    res = authed_client.post(
        "/auth/login",
        json={"username": "nobody", "password": "a" * 73},
    )
    assert res.status_code == 400
    assert "bcrypt" in res.json()["detail"]


def test_logout_always_succeeds(authed_client):
    res = authed_client.post("/auth/logout")
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# Chat request validation (insecure mode — no auth needed)
# ---------------------------------------------------------------------------


def test_chat_missing_user_message_returns_400(insecure_client):
    res = insecure_client.post(
        CHAT_COMPLETIONS_PATH,
        json=chat_payload("sys", role="system"),
    )
    assert res.status_code == 400


def test_chat_graph_mode_defaults_to_auto(insecure_client):
    """graph_mode defaults to 'auto' when not specified."""
    with patch("local_graph_rag.web.rag_executor.ask") as mock_ask:
        mock_ask.return_value = "answer"
        res = insecure_client.post(
            CHAT_COMPLETIONS_PATH,
            json=chat_payload(),
        )
    assert res.status_code == 200
    assert mock_ask.call_args[0][2] == "auto"


def test_chat_explicit_graph_mode_is_forwarded(insecure_client):
    """graph_mode value is passed through to ask()."""
    with patch("local_graph_rag.web.rag_executor.ask") as mock_ask:
        mock_ask.return_value = "answer"
        insecure_client.post(
            CHAT_COMPLETIONS_PATH,
            json=chat_payload(graph_mode="local"),
        )
    assert mock_ask.call_args[0][2] == "local"



# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_chat_stream_relays_generated_text(insecure_client):
    with patch.object(rag_executor, "ask_stream_sync", return_value=iter(["Hello", " world"])):
        resp = insecure_client.post(CHAT_COMPLETIONS_PATH, json=chat_payload(stream=True))

    assert resp.status_code == 200
    assert '"content": "Hello"' in resp.text
    assert '"content": " world"' in resp.text
    assert resp.text.rstrip().endswith("data: [DONE]")


def test_stream_worker_finishes_when_the_consumer_never_reads(monkeypatch: pytest.MonkeyPatch):
    """A consumer that stops reading must not strand the worker on a full queue."""
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(rag_executor, "_RAG_EXECUTOR", executor)
    monkeypatch.setattr(rag_executor, "_RAG_CONCURRENCY", asyncio.Semaphore(1))
    monkeypatch.setattr(rag_executor, "_store", MagicMock())
    monkeypatch.setattr(rag_executor, "_client", MagicMock())
    monkeypatch.setattr(rag_executor, "STREAM_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(rag_executor, "ask_stream_sync", lambda *a, **k: iter(["x"] * 100))

    async def _never_read() -> int:
        queue, _, future = await rag_executor._start_stream_worker("q", "model", "auto")
        await asyncio.wait_for(future, timeout=5)
        return queue.qsize()

    try:
        assert asyncio.run(_never_read()) == 32  # the queue filled, then the worker gave up
    finally:
        executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def _login(client: TestClient, username: str = "alice", password: str = "correct horse") -> str:
    user_store.upsert_user(username, bcrypt.hashpw(password.encode(), bcrypt.gensalt(4)).decode())
    resp = client.post("/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200
    return client.cookies["rag_token"]


def _stored_tokens() -> list[str]:
    return [row[0] for row in user_store._store.conn.execute("SELECT token FROM sessions")]


def test_login_session_authenticates_requests(authed_client):
    _login(authed_client)
    assert authed_client.get("/auth/status").json() == {"authenticated": True}


def test_session_tokens_are_stored_only_as_hashes(authed_client):
    raw = _login(authed_client)

    stored = _stored_tokens()

    assert stored == [hashlib.sha256(raw.encode()).hexdigest()]
    assert user_store.validate_session(raw) == "alice"
    assert user_store.validate_session(stored[0]) is None  # a leaked hash is not a credential


def test_logout_deletes_the_session(authed_client):
    _login(authed_client)

    authed_client.post("/auth/logout")

    assert _stored_tokens() == []
    assert authed_client.get("/auth/status").json() == {"authenticated": False}


def test_changing_a_password_revokes_sessions(authed_client):
    _login(authed_client)

    user_store.upsert_user("alice", bcrypt.hashpw(b"new password", bcrypt.gensalt(4)).decode())

    assert authed_client.get("/auth/status").json() == {"authenticated": False}
