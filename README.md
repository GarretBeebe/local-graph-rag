# local-graph-rag

A local Graph RAG system that builds a knowledge graph over your documents and uses it alongside
vector search to answer both specific and thematic questions.

Runs entirely on local hardware via [Ollama](https://ollama.ai) and
[Qdrant](https://qdrant.tech). No external API calls required.

> **Status:** Planning phase. See [`context/GRAPH-RAG-PLAN.md`](context/GRAPH-RAG-PLAN.md) for the
> full implementation plan.

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

Ingestion is safe to interrupt. A SQLite `chunks` registry maps every Qdrant point ID back to
its source file. Before re-processing any file, the pipeline deletes its prior vectors from
Qdrant and its prior graph data from SQLite, then re-ingests cleanly. The file fingerprint is
written last, so any interrupted file has no fingerprint and is automatically retried — without
re-processing files that already completed successfully.

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

## Project Layout (planned)

```
local-graph-rag/
├── api/
│   ├── embed.py              # Ollama embedding (copied from rag-system)
│   ├── ollama_client.py      # Ollama HTTP client (copied from rag-system)
│   ├── query_graph_rag.py    # Query entry point (local + global)
│   ├── query_router.py       # Local vs. global classifier
│   ├── local_retrieval.py    # Entity lookup + graph traversal
│   └── global_retrieval.py   # Community summary retrieval + map-reduce
├── graph/
│   ├── store.py              # NetworkX + SQLite graph manager
│   ├── extractor.py          # LLM entity/relationship extraction
│   └── summarizer.py         # Community summarization
├── ingest/
│   ├── chunkers.py           # Document chunking (copied from rag-system)
│   └── index_documents.py    # Full ingestion pipeline
├── common/
│   ├── config.py             # YAML config loader
│   ├── paths.py              # Path normalization
│   └── qdrant.py             # Qdrant client singleton
├── web/
│   └── api_server.py         # FastAPI server (OpenAI-compat chat endpoint)
├── context/
│   └── GRAPH-RAG-PLAN.md     # Full architecture and implementation plan
└── settings.py               # Env-var driven config
```

---

## Implementation Phases

- [x] **Phase 1** — Project skeleton, copied infrastructure, `settings.py`, `pyproject.toml`
- [x] **Phase 2** — Graph store (`graph/store.py`) + entity extractor (`graph/extractor.py`)
- [ ] **Phase 3** — Full ingestion pipeline: fingerprint-based change detection, chunks registry, Qdrant + graph cleanup on re-ingestion, crash recovery
- [ ] **Phase 4** — Community detection + summarization
- [ ] **Phase 5** — Local and global retrieval paths + CLI query interface
- [ ] **Phase 6** — FastAPI web server with streaming

---

## Relation to `rag-system`

This is a **standalone project**. It does not import from `rag-system` as a package dependency.
Five small utility modules are copied and adapted; everything else is built from scratch.

Both systems can run concurrently — they use separate Qdrant collections and separate SQLite
databases.
