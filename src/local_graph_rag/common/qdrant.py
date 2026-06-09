"""Shared QdrantClient singleton."""

import threading

from qdrant_client import QdrantClient

from local_graph_rag.settings import QDRANT_API_KEY, QDRANT_HOST, QDRANT_PORT

_qdrant_client: QdrantClient | None = None
_qdrant_lock = threading.Lock()


def get_qdrant_client() -> QdrantClient:
    """Return the shared QdrantClient, creating it on first call."""
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_lock:
            if _qdrant_client is None:
                _qdrant_client = QdrantClient(
                    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
                    api_key=QDRANT_API_KEY or None,
                )
    return _qdrant_client
