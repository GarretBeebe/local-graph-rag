"""
Shared embedding helper used by both the retrieval pipeline and the ingest pipeline.

Kept separate from other API modules to avoid loading unrelated dependencies
in contexts that only need embedding (e.g. batch indexing).
"""

import logging

import local_graph_rag.rag.ollama_client as ollama_client
from local_graph_rag.settings import (
    EMBED_MODEL,
    MAX_EMBED_CHARS,
    OLLAMA_EMBED_TIMEOUT_SECONDS,
    VECTOR_SIZE,
)

logger = logging.getLogger(__name__)

# Inputs per /api/embed request: keeps one large file from exceeding the request timeout in
# a single call, and makes a retry repeat one slice instead of the whole file.
_EMBED_BATCH_SIZE = 64


def _prepare_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("Cannot embed empty text")
    if len(text) > MAX_EMBED_CHARS:
        logger.warning(
            "Truncating text from %d to %d chars for embedding", len(text), MAX_EMBED_CHARS
        )
        text = text[:MAX_EMBED_CHARS]
    return text


def _validate_vector(vector: list[float], model: str = EMBED_MODEL) -> list[float]:
    if len(vector) != VECTOR_SIZE:
        raise RuntimeError(
            f"Embedding model {model!r} returned {len(vector)} dimensions; "
            f"configured VECTOR_SIZE is {VECTOR_SIZE}"
        )
    return vector


def embed(text: str) -> list[float]:
    """Return an embedding vector for the given text via the Ollama embeddings API."""
    text = _prepare_text(text)

    response = ollama_client.post_with_retry(
        "/api/embeddings",
        json=ollama_client.with_keep_alive({"model": EMBED_MODEL, "prompt": text}),
        timeout=OLLAMA_EMBED_TIMEOUT_SECONDS,
    )

    try:
        data = response.json()
    except ValueError as e:
        raise RuntimeError(f"Embedding service returned invalid JSON: {e}") from e

    if "embedding" not in data:
        raise RuntimeError("Embedding response missing 'embedding' field")

    return _validate_vector(data["embedding"])


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for multiple texts via Ollama's batch embed API."""
    prepared = [_prepare_text(text) for text in texts]
    vectors: list[list[float]] = []
    for start in range(0, len(prepared), _EMBED_BATCH_SIZE):
        batch = prepared[start : start + _EMBED_BATCH_SIZE]
        response = ollama_client.post_with_retry(
            "/api/embed",
            json=ollama_client.with_keep_alive({"model": EMBED_MODEL, "input": batch}),
            timeout=OLLAMA_EMBED_TIMEOUT_SECONDS,
        )

        try:
            data = response.json()
        except ValueError as e:
            raise RuntimeError(f"Batch embedding service returned invalid JSON: {e}") from e

        if "embeddings" not in data:
            raise RuntimeError("Batch embedding response missing 'embeddings' field")
        if len(data["embeddings"]) != len(batch):
            raise RuntimeError(
                f"Batch embedding returned {len(data['embeddings'])} vectors "
                f"for {len(batch)} texts"
            )
        vectors.extend(_validate_vector(vector) for vector in data["embeddings"])
    return vectors
