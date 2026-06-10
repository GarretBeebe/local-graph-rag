# local-graph-rag

A local Graph RAG system that builds a knowledge graph over your documents and uses it alongside
vector search to answer both specific and thematic questions.

Runs entirely on local hardware via [Ollama](https://ollama.ai) and
[Qdrant](https://qdrant.tech). No external API calls required.

> **Status:** Phases 1–6 complete. Ingestion pipeline, community detection, local/global
> retrieval, and the OpenAI-compatible web server with full auth are all operational.
> See [`notes/planning/graph-rag-plan.md`](notes/planning/graph-rag-plan.md) for the full implementation plan.

---

## Why Graph RAG?

Standard RAG retrieves isolated text chunks. It answers "find me the passage about X" well, but
struggles with questions that require connecting information across documents:

- *"How does the fingerprint store relate to the watcher?"*
- *"What are the main architectural themes in this codebase?"*
- *"What depends on the embedding module?"*

Graph RAG builds a knowledge graph during ingestion — extracting entities and relationships from
each document — then uses that graph structure at query time to expand context beyond what any
single chunk contains.

---

## Architecture

Two indexes are built during ingestion:

```
Documents
    │
    ▼
[Chunker]
    │
    ├──► [Embedder] ──► Qdrant (dense vectors for semantic search)
    │
    └──► [Entity Extractor] ──► Graph Store (NetworkX + SQLite)
                                      │
                                      ▼
                              [Community Detection]
                                      │
                                      ▼
                              [Community Summaries] ──► SQLite cache
```

At query time, a router classifies the question and picks a retrieval strategy:

| Query type | Example | Retrieval strategy |
|---|---|---|
| **Local** | "What does the fingerprint store do?" | Entity lookup → graph traversal → supporting chunks |
| **Global** | "What are the main themes?" | Community summary search → map-reduce |

Local queries combine vector search (to find seed entities) with graph traversal (to expand
context). Global queries use pre-built community summaries retrieved by embedding similarity.

### Resumable ingestion

Ingestion is safe to interrupt and guarantees consistent state across restarts:

- **Fingerprint written last.** Each file's SHA-256 fingerprint is written to SQLite only after
  all Qdrant and graph writes complete. A crashed run leaves no fingerprint, so the file is
  retried on the next run. Files that completed are skipped via hash comparison.

- **Cleanup ordering.** When re-indexing a changed file, vectors are deleted from Qdrant
  *before* removing chunk records from SQLite. If Qdrant is unreachable, the SQLite chunk IDs
  survive and the next run can retry the Qdrant delete. When writing new vectors, chunk IDs are
  registered in SQLite *before* upserting to Qdrant, so any Qdrant failure leaves behind
  deletable IDs for the next run.

- **Stale file cleanup.** At startup, the pipeline compares tracked paths against files on disk
  and removes data for any files that were deleted — vectors from Qdrant, entities and
  relationships from SQLite, and the fingerprint record.

---

## Stack

| Concern | Choice |
|---|---|
| Vector store | Qdrant |
| Graph store | NetworkX (in-memory) + SQLite (persistence) |
| Community detection | Louvain (`python-louvain`) |
| LLM / embeddings | Ollama |
| Web framework | FastAPI |
| Package manager | uv |

### Tiered model strategy

Three different Ollama models are used for different stages, tuned for the CPU-only K16 hardware:

| Stage | Default model | Why |
|---|---|---|
| Entity extraction (`EXTRACT_MODEL`) | `qwen2.5:3b` | Structured JSON only; ~4× faster than 14B |
| Community summarization (`SUMMARIZE_MODEL`) | `qwen2.5:7b` | Needs more synthesis; still 2× faster than 14B |
| Answer generation (`GEN_MODEL`) | `qwen2.5:14b` | Quality matters for the final response |

All three are overridable via environment variables.

---

## Quick start

```bash
cp .env.example .env          # set QDRANT_API_KEY, NEXTCLOUD_PATH, CODE_PATH, and optional API_KEY

docker compose up -d qdrant

# Index documents
docker compose --profile indexer run --rm indexer

# Detect communities and build LLM summaries
docker compose --profile summarizer run --rm summarizer

# Re-run with --force to regenerate all summaries even if membership is unchanged
docker compose --profile summarizer run --rm summarizer graph-rag-summarize --force
```

The summarizer is idempotent: it skips communities whose membership hasn't changed since the
last run (tracked by a SHA-256 member hash). Run it after any indexer run that adds new
documents.

### Running the web server

```bash
# Start the API server (serves the chat UI at http://localhost:8000/ui)
docker compose up -d api

# Or run locally during development
uv run uvicorn local_graph_rag.web.api_server:app --host 127.0.0.1 --port 8000
```

Docker binds the API to `127.0.0.1` by default through `API_BIND_HOST`. Only bind to a LAN
interface or `0.0.0.0` when the service is behind a trusted TLS-terminating reverse proxy.

### User management

Before connecting a browser-based client, create at least one user:

```bash
uv run graph-rag-users add alice              # prompts for password
uv run graph-rag-users add alice --password-stdin  # read password from stdin (scripting)
uv run graph-rag-users list
uv run graph-rag-users remove alice
```

Users are stored in `data/users.sqlite3`. All sessions for a user are revoked when their
password is updated or they are removed.

### Authentication

Three authentication paths are supported:

| Path | Use case |
|---|---|
| Browser login (username + password) | Chat UI at `/ui` |
| `Authorization: Bearer <API_KEY>` | OpenAI-compatible clients (Chatbox, Open WebUI) |
| `ALLOW_INSECURE_LOCALONLY=true` | Bypasses all auth checks for local-only installs |

`API_KEY` is set in `.env`, must be at least 32 characters when set, and is intended for
OpenAI-compatible clients. When using Chatbox, point it at `http://localhost:8000` and set the
API key to match.

Authorization is intentionally coarse-grained for this local implementation: any authenticated
user can query the full indexed corpus. Do not use this as a multi-tenant RBAC system.

### Querying from the CLI

With Qdrant and Ollama running and at least one document indexed:

```bash
# Auto-routes between local (entity-specific) and global (thematic) retrieval
uv run graph-rag-query "What does GraphStore do?"

# Force a specific retrieval mode: auto (default), local, or global
uv run graph-rag-query "What are the main themes in this codebase?" --mode global
```

`auto` mode classifies the question with `EXTRACT_MODEL` (falling back to a keyword heuristic
if the LLM call fails or returns something unexpected) and picks local or global retrieval
accordingly. Global mode falls back to local retrieval if no community summaries exist yet.

---

## Project Layout

```
local-graph-rag/
├── src/local_graph_rag/
│   ├── cli/
│   │   └── manage_users.py       # User management command
│   ├── rag/
│   │   ├── embed.py              # Ollama embedding
│   │   ├── ollama_client.py      # Ollama HTTP client
│   │   ├── query_graph_rag.py    # Query entry point — local + global, CLI
│   │   ├── query_router.py       # Local vs. global classifier
│   │   ├── local_retrieval.py    # Entity lookup + graph traversal
│   │   └── global_retrieval.py   # Community summary retrieval
│   ├── graph/
│   │   ├── store.py              # NetworkX + SQLite graph store
│   │   ├── extractor.py          # LLM entity/relationship extraction
│   │   └── summarizer.py         # Louvain community detection + summarization
│   ├── ingest/
│   │   ├── chunkers.py           # Document chunking
│   │   └── index_documents.py    # Full ingestion pipeline
│   ├── common/
│   │   ├── config.py             # YAML config loader
│   │   ├── paths.py              # Path normalization
│   │   ├── qdrant.py             # Qdrant client singleton
│   │   └── sqlite_store.py       # Thread-local SQLite connection wrapper
│   ├── web/
│   │   ├── api_server.py         # FastAPI server
│   │   ├── routes.py             # HTTP routes
│   │   ├── auth.py               # Session token + API key validation
│   │   ├── user_store.py         # bcrypt user store
│   │   ├── rate_limit.py         # Token-bucket rate limiter
│   │   ├── schemas.py            # Request/response models
│   │   ├── openai_compat.py      # OpenAI-compatible response builders
│   │   └── static/               # Chat UI, served at /ui
│   └── settings.py               # Env-var driven config
├── tests/
│   ├── test_graph.py             # GraphStore + extractor unit tests
│   ├── test_ingestion.py         # Fingerprint store + hash utility tests
│   ├── test_summarizer.py        # Community summarizer unit tests
│   ├── test_retrieval.py         # Local + global retrieval unit tests
│   ├── test_query_router.py      # Local/global routing unit tests
│   ├── test_query_graph_rag.py   # End-to-end query module tests
│   ├── test_api_server.py        # API server auth + endpoint tests
│   └── test_index_security.py    # Ingestion security (path traversal, size limits)
├── notes/
│   ├── planning/                 # Architecture plans and punch lists
│   └── security/                 # Security review notes
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Implementation Phases

- [x] **Phase 1** — Project skeleton, copied infrastructure, `settings.py`, `pyproject.toml`
- [x] **Phase 2** — Graph store (`graph/store.py`) + entity extractor (`graph/extractor.py`)
- [x] **Phase 3** — Full ingestion pipeline: fingerprint-based change detection, chunks registry, crash-safe Qdrant ↔ SQLite ordering, stale file cleanup
- [x] **Phase 4** — Louvain community detection, LLM-based community summarization with member-hash skip logic, community embedding store (`graph/summarizer.py`)
- [x] **Phase 5** — Local retrieval (vector search → entity neighborhood expansion), global retrieval (cosine similarity over community summaries), LLM query router with heuristic fallback, and a CLI query interface (`src/local_graph_rag/rag/query_graph_rag.py`)
- [x] **Phase 6** — FastAPI web server with OpenAI-compatible streaming endpoint, chat UI, bcrypt/session/API-key auth stack, rate limiting, security headers, and ingestion security hardening (symlink guard, file-size limit, path traversal prevention)

---

## Relation to `rag-system`

This is a **standalone project**. It does not import from `rag-system` as a package dependency.
Five small utility modules are copied and adapted; everything else is built from scratch.

Both systems can run concurrently — they use separate Qdrant collections and separate SQLite
databases.
