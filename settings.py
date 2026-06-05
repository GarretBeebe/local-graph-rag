"""
Project-level settings for local-graph-rag.

All values overridable via environment variables.
Defaults assume a local install with Ollama and Qdrant on localhost.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_PATH = Path(os.environ.get("DOCS_PATH", str(PROJECT_ROOT / "documents")))
SQLITE_PATH = Path(os.environ.get("SQLITE_PATH", str(DATA_DIR / "graph.db")))

# Qdrant
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "graph_documents")
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "768"))

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", "qwen2.5:3b")
SUMMARIZE_MODEL = os.environ.get("SUMMARIZE_MODEL", "qwen2.5:7b")
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen2.5:14b")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
OLLAMA_GENERATE_TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_GENERATE_TIMEOUT_SECONDS", "120.0"))
OLLAMA_EMBED_TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_EMBED_TIMEOUT_SECONDS", "60.0"))
OLLAMA_MAX_RETRIES = int(os.environ.get("OLLAMA_MAX_RETRIES", "2"))
OLLAMA_RETRY_DELAY_SECONDS = float(os.environ.get("OLLAMA_RETRY_DELAY_SECONDS", "1.0"))
GENERATION_CONCURRENCY_LIMIT = int(os.environ.get("GENERATION_CONCURRENCY_LIMIT", "1"))

# Chunking
MAX_EMBED_CHARS = 6000
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", "2000"))
MAX_MD_CHUNK = MAX_CHUNK_CHARS

# File filtering
ALLOWED_EXTENSIONS = {
    ".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml",
    ".ts", ".tsx", ".js", ".jsx",
}

# Graph retrieval
EXTRACT_BATCH_TOKENS = int(os.environ.get("EXTRACT_BATCH_TOKENS", "4000"))
ENTITY_RETRIEVAL_K = int(os.environ.get("ENTITY_RETRIEVAL_K", "5"))
ENTITY_NEIGHBORHOOD_HOPS = int(os.environ.get("ENTITY_NEIGHBORHOOD_HOPS", "2"))
COMMUNITY_RETRIEVAL_N = int(os.environ.get("COMMUNITY_RETRIEVAL_N", "5"))


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"settings: {name} must be > 0, got {value}")


def _validate_settings() -> None:
    for name, value in {
        "QDRANT_PORT": QDRANT_PORT,
        "VECTOR_SIZE": VECTOR_SIZE,
        "OLLAMA_NUM_CTX": OLLAMA_NUM_CTX,
        "GENERATION_CONCURRENCY_LIMIT": GENERATION_CONCURRENCY_LIMIT,
        "MAX_CHUNK_CHARS": MAX_CHUNK_CHARS,
        "CHUNK_SIZE": CHUNK_SIZE,
        "EXTRACT_BATCH_TOKENS": EXTRACT_BATCH_TOKENS,
        "ENTITY_RETRIEVAL_K": ENTITY_RETRIEVAL_K,
        "ENTITY_NEIGHBORHOOD_HOPS": ENTITY_NEIGHBORHOOD_HOPS,
        "COMMUNITY_RETRIEVAL_N": COMMUNITY_RETRIEVAL_N,
    }.items():
        _require_positive(name, value)
    for name, value in {
        "OLLAMA_GENERATE_TIMEOUT_SECONDS": OLLAMA_GENERATE_TIMEOUT_SECONDS,
        "OLLAMA_EMBED_TIMEOUT_SECONDS": OLLAMA_EMBED_TIMEOUT_SECONDS,
        "OLLAMA_RETRY_DELAY_SECONDS": OLLAMA_RETRY_DELAY_SECONDS,
    }.items():
        _require_positive(name, value)
    if OLLAMA_MAX_RETRIES < 0:
        raise ValueError(f"settings: OLLAMA_MAX_RETRIES must be >= 0, got {OLLAMA_MAX_RETRIES}")
    if CHUNK_OVERLAP < 0:
        raise ValueError(f"settings: CHUNK_OVERLAP must be >= 0, got {CHUNK_OVERLAP}")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            f"settings: CHUNK_OVERLAP must be smaller than CHUNK_SIZE, "
            f"got {CHUNK_OVERLAP} >= {CHUNK_SIZE}"
        )
    if MAX_CHUNK_CHARS < CHUNK_SIZE:
        raise ValueError(
            f"settings: MAX_CHUNK_CHARS must be >= CHUNK_SIZE, "
            f"got {MAX_CHUNK_CHARS} < {CHUNK_SIZE}"
        )


_validate_settings()
