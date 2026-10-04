"""Unit tests for the Ollama HTTP client and batch embedding. No Ollama required."""

import logging

import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout

import local_graph_rag.rag.embed as embed_mod
import local_graph_rag.rag.ollama_client as ollama_client


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    """Returns (or raises) one scripted outcome per POST, recording request kwargs."""

    def __init__(self, *outcomes: _FakeResponse | Exception):
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    def post(self, url: str, **kwargs) -> _FakeResponse:
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch):
    def _install(*outcomes: _FakeResponse | Exception) -> _FakeSession:
        session = _FakeSession(*outcomes)
        monkeypatch.setattr(ollama_client, "_get_session", lambda: session)
        monkeypatch.setattr(ollama_client, "OLLAMA_RETRY_DELAY_SECONDS", 0)
        return session

    return _install


def test_post_with_retry_does_not_retry_read_timeouts(fake_session):
    session = fake_session(ReadTimeout("still generating"), _FakeResponse({}))

    with pytest.raises(RuntimeError, match="timed out"):
        ollama_client.post_with_retry("/api/generate", json={}, timeout=1)
    assert len(session.calls) == 1


def test_post_with_retry_retries_connection_errors(fake_session):
    session = fake_session(RequestsConnectionError("refused"), _FakeResponse({"ok": True}))

    response = ollama_client.post_with_retry("/api/generate", json={}, timeout=1)

    assert response.json() == {"ok": True}
    assert len(session.calls) == 2


def test_generate_sends_options_and_logs_timings(fake_session, caplog):
    session = fake_session(
        _FakeResponse(
            {"response": "hi", "load_duration": 2_000_000_000, "prompt_eval_count": 12}
        )
    )

    with caplog.at_level(logging.INFO, logger=ollama_client.__name__):
        answer = ollama_client.generate("prompt", "model-x", options={"temperature": 0})

    assert answer == "hi"
    assert session.calls[0]["json"]["options"] == {
        "num_ctx": ollama_client.OLLAMA_NUM_CTX,
        "temperature": 0,
    }
    assert "load 2.0s, prompt 12 tok" in caplog.text


def test_embed_batch_splits_large_inputs_into_bounded_requests(monkeypatch: pytest.MonkeyPatch):
    request_sizes: list[int] = []

    def _fake_post(path: str, **kwargs) -> _FakeResponse:
        batch = kwargs["json"]["input"]
        request_sizes.append(len(batch))
        return _FakeResponse({"embeddings": [[0.0] * embed_mod.VECTOR_SIZE for _ in batch]})

    monkeypatch.setattr(embed_mod.ollama_client, "post_with_retry", _fake_post)

    vectors = embed_mod.embed_batch([f"text {i}" for i in range(130)])

    assert request_sizes == [64, 64, 2]
    assert len(vectors) == 130


def test_keep_alive_is_sent_when_configured_but_not_in_the_cache_keyed_payload(
    fake_session, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ollama_client, "OLLAMA_KEEP_ALIVE", "30m")
    session = fake_session(_FakeResponse({"response": "hi"}))

    ollama_client.generate("prompt", "model-x")

    assert session.calls[0]["json"]["keep_alive"] == "30m"
    # The extraction cache hashes this payload; keep-alive must not change the key.
    assert "keep_alive" not in ollama_client.build_generate_payload("model-x", "p", stream=False)


def test_keep_alive_is_omitted_by_default(fake_session, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ollama_client, "OLLAMA_KEEP_ALIVE", "")
    session = fake_session(_FakeResponse({"response": "hi"}))

    ollama_client.generate("prompt", "model-x")

    assert "keep_alive" not in session.calls[0]["json"]


def test_embed_batch_sends_keep_alive_when_configured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ollama_client, "OLLAMA_KEEP_ALIVE", "30m")
    sent: list[dict] = []

    def _fake_post(path: str, **kwargs) -> _FakeResponse:
        sent.append(kwargs["json"])
        return _FakeResponse({"embeddings": [[0.0] * embed_mod.VECTOR_SIZE]})

    monkeypatch.setattr(embed_mod.ollama_client, "post_with_retry", _fake_post)

    embed_mod.embed_batch(["text"])

    assert sent[0]["keep_alive"] == "30m"
