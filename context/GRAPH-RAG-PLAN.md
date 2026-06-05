# Graph RAG — Implementation Plan

## Executive Summary

Build `local-graph-rag` as a standalone project. Do **not** import from `rag-system` as a package
dependency. A handful of modules are worth copying and adapting; the retrieval layer, storage layer,
and ingestion pipeline are architecturally different enough that coupling them would cause more
confusion than they save. The Ollama infrastructure and chunking logic are the two genuine wins.

---

## What Exists in `rag-system`

The existing system is a well-built flat-vector RAG with:

- **Chunking** (`ingest/chunkers.py`): AST-based for Python, header-based for Markdown, recursive
  character splitting for everything else. Solid, reusable as-is.
- **Embedding** (`api/embed.py` + `api/ollama_client.py`): Calls Ollama's `/api/embeddings` and
  `/api/embed` (batch). Fully decoupled from retrieval logic.
- **Vector store**: Qdrant for dense vectors; BM25 in-memory index for keyword recall.
- **Retrieval pipeline** (`api/retrieval.py`): Hybrid recall (Qdrant + BM25) → MMR →
  cross-encoder reranking. Three-stage, tuned for flat vector recall.
- **Generation** (`api/query_rag.py`): Prompt builder + streaming Ollama output.
- **Ingestion** (`ingest/index_documents.py`): Read → chunk → embed_batch → upsert to Qdrant.
  Fingerprint-based change detection via SQLite.
- **Web layer** (`web/`): FastAPI, auth, rate limiting, streaming endpoint.
- **Settings** (`settings.py`): Flat env-var driven config with validation.

---

## Reuse Decision

| Component | Decision | Rationale |
|---|---|---|
| `api/ollama_client.py` | **Copy** | Pure HTTP infrastructure; no RAG coupling |
| `api/embed.py` | **Copy** | Calls ollama_client; pure embedding logic |
| `ingest/chunkers.py` | **Copy** | Chunking is still needed; works unchanged |
| `common/config.py` | **Copy** | 20-line YAML loader utility |
| `common/paths.py` | **Copy** | Path normalization utility |
| `settings.py` pattern | **Replicate** | Env-var config is the right pattern here too |
| `web/api_server.py` | **Do not reuse** | Graph RAG needs different endpoints |
| `api/retrieval.py` | **Do not reuse** | Hybrid recall + MMR is vector-specific |
| `api/keyword_index.py` | **Do not reuse** | Graph traversal replaces BM25 recall |
| `common/qdrant.py` | **Do not reuse** | Graph structure needs a different store |
| `ingest/index_documents.py` | **Do not reuse** | Must add entity extraction step |
| `web/auth.py` / `web/rate_limit.py` | **Optional later** | Out of scope for initial build |

---

## Architecture Overview

Graph RAG adds a knowledge graph on top of the vector store. The two key new capabilities are:

1. **Local queries**: retrieve specific entity neighborhoods from the graph, then answer with
   entity summaries + supporting chunks.
2. **Global queries**: retrieve pre-built community summaries, then answer with a map-reduce over
   them.

```
Documents
    │
    ▼
[Chunker]  ─────────── (same as rag-system) ───────────────────────
    │
    ├──► [Embedder] ──► Qdrant (vector store — entity + chunk nodes)
    │
    └──► [Entity Extractor (LLM)] ──► Graph Store (NetworkX + SQLite)
              │
              ▼
         [Community Detection] ──► Community Summaries (SQLite)

Query
    │
    ▼
[Query Router] ─┬─ local  ──► Entity lookup ──► Graph traversal ──► LLM
                └─ global ──► Community summaries ──► Map-reduce ──► LLM
```

---

## Hardware Context

Target hardware: GMKtec K16 (Ryzen 7 7735HS 8C/16T, 32 GB LPDDR5, no discrete GPU, OCuLink port).
CPU-only inference via Ollama/llama.cpp. Rough throughput at Q4 quantization:

| Model size | Approx tok/sec |
|---|---|
| 3B | 40–60 |
| 7B | 20–35 |
| 14B | 8–15 |

The 14B model at 10 tok/sec costs ~20s per 200-token extraction response. Across 500 files that is
~2.5 hours — too slow to be the extraction model. The solution is a tiered model strategy: use the
smallest model that produces reliable structured JSON for extraction, reserve the large model for
final answer generation. Microsoft's own GraphRAG does the same thing (different models per stage).

OCuLink provides an eGPU upgrade path if throughput becomes a blocker.

---

## Technology Choices

| Concern | Choice | Rationale |
|---|---|---|
| Graph storage | **NetworkX + SQLite** | No new service; NetworkX handles traversal; SQLite persists nodes/edges and community summaries. Neo4j is overkill for local use. |
| Vector store | **Qdrant** | Already in the stack; entity embeddings live here for semantic entity lookup |
| Entity extraction model (`EXTRACT_MODEL`) | **`qwen2.5:3b`** (default) | Structured JSON output; 3B is sufficient for entity/relationship extraction; ~4× faster than 14B on this hardware |
| Community summarization model (`SUMMARIZE_MODEL`) | **`qwen2.5:7b`** (default) | Needs more synthesis capability than extraction; still 2× faster than 14B |
| Answer generation model (`GEN_MODEL`) | **`qwen2.5:14b`** (default) | Quality matters for the final response; same model as rag-system |
| Community detection | **NetworkX Louvain** (`community` package) | Standard algorithm, built into Python ecosystem |
| Web framework | **FastAPI** | Same as rag-system |
| Package manager | **uv** | Same as rag-system |

---

## Graph Store Schema (SQLite)

```sql
-- Canonical entity nodes
CREATE TABLE entities (
    id          TEXT PRIMARY KEY,   -- slugified name
    name        TEXT NOT NULL,
    type        TEXT,               -- PERSON, ORG, CONCEPT, etc.
    description TEXT,
    community   INTEGER,            -- assigned after community detection
    embedding   BLOB                -- serialized float list for semantic lookup
);

-- Directed relationships between entities
CREATE TABLE relationships (
    id          TEXT PRIMARY KEY,
    source_id   TEXT REFERENCES entities(id),
    target_id   TEXT REFERENCES entities(id),
    label       TEXT NOT NULL,      -- e.g. "uses", "depends_on", "authored_by"
    weight      REAL DEFAULT 1.0,
    source_doc  TEXT                -- filepath the relationship was extracted from
);

-- Chunk-to-entity membership
CREATE TABLE chunk_entities (
    chunk_id    TEXT,               -- Qdrant point UUID
    entity_id   TEXT REFERENCES entities(id),
    PRIMARY KEY (chunk_id, entity_id)
);

-- Community summary cache
CREATE TABLE communities (
    id          INTEGER PRIMARY KEY,
    summary     TEXT NOT NULL,
    entity_ids  TEXT,               -- JSON array of entity IDs
    embedding   BLOB                -- embedded summary for semantic lookup
);
```

---

## New Components

### 1. Entity Extractor (`graph/extractor.py`)

Runs an LLM prompt over each chunk. Returns a structured list of entities and relationships.
Use structured output (JSON) if the model supports it; fall back to regex extraction.

**Prompt template (simplified):**
```
Extract all named entities and their relationships from the text below.

Return JSON with:
{
  "entities": [{"name": str, "type": str, "description": str}],
  "relationships": [{"source": str, "target": str, "label": str}]
}

Text:
{chunk_text}
```

**Key considerations:**
- Uses `EXTRACT_MODEL` (default: `qwen2.5:3b`), not `GEN_MODEL`. Never use the 14B model here —
  on the K16, that turns a 30-minute index into a 3-hour index for no quality gain on structured
  JSON output.
- Entity names must be normalized (lowercase, stripped) before merge.
- Budget ~1 LLM call per file. Not per chunk — batch all chunks for a file into one extraction
  call. Keep the combined input under ~4000 tokens to avoid quality degradation on smaller models.
- If a file's chunks exceed the batch token budget, split into 2-3 calls and merge results.

### 2. Graph Manager (`graph/store.py`)

Wraps NetworkX + SQLite. Responsibilities:
- `upsert_entity(name, type, description)` — merge by name slug, update description
- `upsert_relationship(source, target, label, source_doc)` — add edges
- `link_chunk(chunk_id, entity_ids)` — populate `chunk_entities`
- `build_networkx_graph()` — load edges from SQLite into an in-memory NetworkX DiGraph
- `detect_communities()` — run Louvain on the graph, write `community` back to `entities`
- `get_entity_neighborhood(entity_id, hops=2)` — return subgraph for local queries

### 3. Community Summarizer (`graph/summarizer.py`)

After community detection, iterate over each community, gather all entity descriptions and
relationship labels, prompt the LLM for a summary, write to `communities` table.

Uses `SUMMARIZE_MODEL` (default: `qwen2.5:7b`). Community summarization requires more synthesis
than entity extraction — the 3B model produces thin summaries — but does not need the full 14B.
The 7B hits a good quality/speed balance on this hardware (~20-35 tok/sec).

This runs once after initial ingestion, and again any time entities are added. Summaries are
cached in SQLite; only communities whose entity membership changed need re-summarization.

### 4. Ingestion Pipeline (`ingest/index_documents.py`)

New pipeline, replacing (not extending) the rag-system version:

```
read_file
    → chunk_document          (copied from rag-system)
    → embed_batch             (copied from rag-system)
    → upsert to Qdrant        (same as rag-system)
    → extract_entities_batch  (new: one LLM call per file over concatenated chunks)
    → graph_store.upsert_*    (new)
    → fingerprint (SQLite)    (same pattern as rag-system)
```

### 5. Query Router (`api/query_router.py`)

Classify the query as local or global:
- **Local**: mentions specific entity names, asks "what does X do", "how does A relate to B"
- **Global**: asks "what are the main themes", "summarize the codebase", "what topics exist"

Simple heuristic first (keyword detection); upgrade to LLM classification if needed.

### 6. Local Retrieval (`api/local_retrieval.py`)

1. Embed the question.
2. Search Qdrant for top-K entities by embedding similarity.
3. For each matched entity, retrieve its 1-2 hop neighborhood from the graph.
4. Collect all unique entity descriptions + relationship labels.
5. Also pull supporting chunks from Qdrant (via `chunk_entities` join).
6. Build context: entity summaries + relationship list + supporting chunk text.

### 7. Global Retrieval (`api/global_retrieval.py`)

1. Embed the question.
2. Find top-N community summaries by embedding similarity.
3. If N > 1, run map-reduce: prompt LLM once per community summary, then reduce to final answer.
4. Return the final reduced answer.

### 8. Query Interface (`api/query_graph_rag.py`)

Same pattern as `rag-system/api/query_rag.py`:

```python
def ask(question, model, cancel=None):
    mode = route_query(question)        # "local" or "global"
    if mode == "local":
        context = local_retrieve(question)
    else:
        context = global_retrieve(question)
    prompt = build_prompt(question, context, mode)
    return ollama_client.generate(prompt, model, cancel=cancel)
```

---

## Implementation Steps

### Phase 1 — Skeleton + copied infrastructure

1. Create project layout: `api/`, `graph/`, `ingest/`, `common/`, `web/`
2. Copy `ollama_client.py`, `embed.py`, `chunkers.py`, `config.py`, `paths.py` from rag-system
3. Write `settings.py` (env-var config; reuse same env var names where overlap exists)
   Key new settings:
   - `EXTRACT_MODEL` (default: `qwen2.5:3b`) — entity/relationship extraction
   - `SUMMARIZE_MODEL` (default: `qwen2.5:7b`) — community summarization
   - `GEN_MODEL` (default: `qwen2.5:14b`) — final answer generation (same as rag-system)
4. Write `pyproject.toml` with new dependencies

**New dependencies vs. rag-system:**
```toml
# Keep
qdrant-client
sentence-transformers
requests
fastapi
uvicorn
pyyaml
tqdm

# Add
networkx
python-louvain    # community detection (networkx-community)

# Remove
rank-bm25
watchdog          # defer file watching to later
torch             # defer — only needed for reranker
bcrypt            # defer auth to later
```

### Phase 2 — Graph storage + entity extraction

1. Implement `graph/store.py` (SQLite schema + NetworkX wrapper)
2. Implement `graph/extractor.py` (LLM entity/relationship extraction)
3. Write unit tests for entity normalization and merging
4. Test entity extraction against a small corpus

### Phase 3 — Ingestion pipeline

1. Implement `ingest/index_documents.py` (full pipeline including extraction step)
2. Implement fingerprint store (copy from rag-system or reimplement simply)
3. Run against a test document set; verify Qdrant and SQLite both populated correctly

### Phase 4 — Community detection + summarization

1. Implement `graph/summarizer.py`
2. Wire community detection to run after batch ingestion completes
3. Verify community summaries are sensible on a real corpus

### Phase 5 — Retrieval + query interface

1. Implement `api/local_retrieval.py`
2. Implement `api/global_retrieval.py`
3. Implement `api/query_router.py`
4. Implement `api/query_graph_rag.py`
5. Test both local and global query paths end-to-end via CLI script

### Phase 6 — Web API

1. Implement `web/api_server.py` (FastAPI, OpenAI-compat chat endpoint)
2. Wire local vs. global mode as a request parameter (user can override the router)
3. Add streaming support (same pattern as rag-system)

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Entity extraction is slow on CPU hardware | Use `EXTRACT_MODEL=qwen2.5:3b` (~4× faster than 14B); batch all chunks per file into one call; accept that initial index of large corpora takes time (one-time cost) |
| 3B extraction model produces malformed JSON or misses entities | Validate JSON output; fall back to regex parsing; test extraction quality early (Phase 2) before committing to 3B — if unacceptable, try `qwen2.5:7b` as `EXTRACT_MODEL` |
| Entity merging is lossy (same entity, different names) | Normalize aggressively; optionally embed entity names and merge by cosine similarity |
| Community quality depends on graph density | Enforce minimum edge weight threshold before including in community detection |
| Global query map-reduce costs many LLM calls | Cap community count fed to global retrieval; cache community embeddings |
| Qdrant already running for rag-system on same ports | Use a separate Qdrant collection name (`graph_documents`) or a separate port in docker-compose |

---

## What This Is NOT

- A drop-in replacement for `rag-system`. Both can run concurrently.
- A port of Microsoft's GraphRAG. That system uses Azure OpenAI and is designed for large
  corpora. This is a local, smaller-scale version using the same conceptual approach.
- An agentic system. No tool-calling loops, no dynamic re-retrieval. Plain retrieval → generation.
