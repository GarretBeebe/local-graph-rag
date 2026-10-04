"""Per-thread Ollama HTTP sessions — each worker thread gets its own requests.Session."""

import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import requests
from requests import RequestException
from requests.exceptions import ReadTimeout

from local_graph_rag.settings import (
    GENERATION_CONCURRENCY_LIMIT,
    OLLAMA_BASE_URL,
    OLLAMA_GENERATE_TIMEOUT_SECONDS,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MAX_RETRIES,
    OLLAMA_NUM_CTX,
    OLLAMA_RETRY_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)

_thread_local = threading.local()
_generation_slots = threading.BoundedSemaphore(GENERATION_CONCURRENCY_LIMIT)
_GENERATION_SLOT_POLL_SECONDS = 0.1
_NS_PER_SECOND = 1e9


def _url(path: str) -> str:
    return f"{OLLAMA_BASE_URL}{path}"


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def close_session() -> None:
    """Close and discard the calling thread's Ollama session. Safe to call from any thread."""
    if hasattr(_thread_local, "session"):
        _thread_local.session.close()
        del _thread_local.session


def post(path: str, **kwargs: Any) -> requests.Response:
    return _get_session().post(_url(path), **kwargs)


def get(path: str, **kwargs: Any) -> requests.Response:
    return _get_session().get(_url(path), **kwargs)


def post_with_retry(
    path: str,
    cancel: threading.Event | None = None,
    **kwargs: Any,
) -> requests.Response:
    """POST with up to OLLAMA_MAX_RETRIES retries on 5xx responses and connection errors."""
    url = _url(path)
    last_exc: Exception | None = None
    for attempt in range(OLLAMA_MAX_RETRIES + 1):
        if cancel and cancel.is_set():
            raise RuntimeError(f"Ollama request to {path} cancelled")
        try:
            r = _get_session().post(url, **kwargs)
            if r.status_code >= 500 and attempt < OLLAMA_MAX_RETRIES:
                logger.warning(
                    "Ollama returned HTTP %d for %s (attempt %d/%d), retrying",
                    r.status_code,
                    path,
                    attempt + 1,
                    OLLAMA_MAX_RETRIES + 1,
                )
                time.sleep(OLLAMA_RETRY_DELAY_SECONDS)
                continue
            if not r.ok:
                raise RuntimeError(f"Ollama request to {path} failed: HTTP {r.status_code}")
            return r
        except ReadTimeout as e:
            # The server took the request and is still working on it: a retry would queue the
            # same expensive generation again behind the abandoned one.
            raise RuntimeError(f"Ollama request to {path} timed out") from e
        except RequestException as e:
            last_exc = e
            if attempt < OLLAMA_MAX_RETRIES:
                logger.warning("Ollama request to %s failed: %s (retrying)", path, e)
                time.sleep(OLLAMA_RETRY_DELAY_SECONDS)
    raise RuntimeError(f"Ollama request to {path} failed after retries: {last_exc}")


def build_generate_payload(
    model: str,
    prompt: str,
    *,
    stream: bool,
    format: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the /api/generate request body (also used to key caches of its responses)."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {"num_ctx": OLLAMA_NUM_CTX, **(options or {})},
    }
    if format is not None:
        payload["format"] = format
    return payload


def with_keep_alive(payload: dict[str, Any]) -> dict[str, Any]:
    """Add OLLAMA_KEEP_ALIVE to a request body when configured.

    Applied at send time rather than in build_generate_payload, whose output keys the
    extraction cache: residency is not part of what a response means.
    """
    return {**payload, "keep_alive": OLLAMA_KEEP_ALIVE} if OLLAMA_KEEP_ALIVE else payload


def _log_timings(model: str, data: dict[str, Any]) -> None:
    """Log where a request's time went; a nonzero load means the model was cold-loaded."""
    logger.info(
        "ollama %s: load %.1fs, prompt %d tok in %.1fs, output %d tok in %.1fs",
        model,
        data.get("load_duration", 0) / _NS_PER_SECOND,
        data.get("prompt_eval_count", 0),
        data.get("prompt_eval_duration", 0) / _NS_PER_SECOND,
        data.get("eval_count", 0),
        data.get("eval_duration", 0) / _NS_PER_SECOND,
    )


@contextmanager
def _generation_slot(
    *,
    cancel: threading.Event | None = None,
    timeout: float = OLLAMA_GENERATE_TIMEOUT_SECONDS,
) -> Iterator[None]:
    """Bound concurrent Ollama generations so local model hardware is not oversubscribed."""
    start = time.monotonic()
    acquired = False
    while not acquired:
        if cancel and cancel.is_set():
            raise RuntimeError("Ollama generation cancelled while waiting for capacity")
        remaining = timeout - (time.monotonic() - start)
        if remaining <= 0:
            raise RuntimeError("Timed out waiting for Ollama generation capacity")
        acquired = _generation_slots.acquire(
            timeout=min(_GENERATION_SLOT_POLL_SECONDS, remaining)
        )
    try:
        yield
    finally:
        _generation_slots.release()


def generate(
    prompt: str,
    model: str,
    timeout: float = OLLAMA_GENERATE_TIMEOUT_SECONDS,
    cancel: threading.Event | None = None,
    format: str | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    """Return a complete generated response from Ollama."""
    payload = build_generate_payload(model, prompt, stream=False, format=format, options=options)
    with _generation_slot(cancel=cancel, timeout=timeout):
        r = post_with_retry(
            "/api/generate", cancel=cancel, json=with_keep_alive(payload), timeout=timeout
        )
    try:
        data = r.json()
    except ValueError as e:
        raise RuntimeError(f"Ollama generate returned invalid JSON: {e}") from e
    if "response" not in data:
        raise RuntimeError(f"Ollama generate missing 'response' field: {data.get('error', data)}")
    _log_timings(model, data)
    return data["response"]


def stream_generate(
    prompt: str,
    model: str,
    timeout: float = OLLAMA_GENERATE_TIMEOUT_SECONDS,
    cancel: threading.Event | None = None,
) -> Iterator[str]:
    """Yield text chunks from Ollama's streaming generation API."""
    with (
        _generation_slot(cancel=cancel, timeout=timeout),
        _get_session().post(
            _url("/api/generate"),
            json=with_keep_alive(build_generate_payload(model, prompt, stream=True)),
            stream=True,
            timeout=timeout,
        ) as resp,
    ):
        if not resp.ok:
            raise RuntimeError(f"Ollama stream request failed: HTTP {resp.status_code}")
        for line in resp.iter_lines(decode_unicode=True):
            if cancel and cancel.is_set():
                break
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ollama stream: skipping malformed line: %r", line[:120])
                continue
            if data.get("error"):
                raise RuntimeError(f"Ollama stream error: {data['error']}")
            if data.get("response"):
                yield data["response"]
            if data.get("done"):
                _log_timings(model, data)
                break
