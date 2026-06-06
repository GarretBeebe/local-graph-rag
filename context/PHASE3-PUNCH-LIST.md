# Phase 3 Issues Punch List — Findings 1–5

From the `/code-review` pass on Phase 3 (2026-06-06). Each item needs a design decision before implementation. Finding 5 is also fixed in the same session.

---

## 1. Qdrant orphans on re-index: SQLite deletes before Qdrant

**File:** `ingest/index_documents.py:77-82`
**Severity:** High — data integrity

### Problem

`store.delete_file_data()` commits first (chunk rows deleted from SQLite, prior IDs returned). `client.delete()` runs second. If Qdrant is unavailable or times out, the chunk IDs are permanently gone from SQLite — there is no way to find those vectors on the next run. They accumulate as orphans on every re-index attempt.

### Recommended fix

Reverse the order: call `client.delete()` first, then `store.delete_file_data()`. If Qdrant fails, the next run still has the chunk IDs in SQLite and can retry. If SQLite fails after a successful Qdrant delete, the chunk rows remain and the next run will issue a Qdrant delete for IDs that no longer exist — Qdrant handles idempotent deletes gracefully.

### Tradeoff

The window between Qdrant delete and SQLite commit leaves the system in a state where search results no longer include the file but graph queries still do. This window is milliseconds in practice.

---

## 2. Qdrant orphans on first index: Qdrant upsert before SQLite register

**File:** `ingest/index_documents.py:108-110`
**Severity:** High — data integrity

### Problem

`client.upsert()` writes N vectors to Qdrant. `store.register_chunks()` commits chunk IDs to SQLite. If `register_chunks` fails (disk full, locked DB), the vectors exist in Qdrant with no SQLite record. The next run's `delete_file_data` returns `[]` and cannot clean them. Each retry adds a fresh set of duplicates to Qdrant.

### Recommended fix

Register chunks in SQLite first (using pre-generated UUIDs), then upsert to Qdrant. If Qdrant fails, the next run finds the chunk rows in SQLite, returns them as `prior_ids`, deletes them via `delete_file_data`, and retries cleanly. Chunk rows briefly exist in SQLite without corresponding Qdrant vectors — acceptable because search uses Qdrant for retrieval, not SQLite.

### Tradeoff

The brief window where the chunk registry is ahead of the vector store means any query during that window won't find these chunks — same as not having indexed them yet, which is correct.

---

## 3. No cleanup for files deleted from disk

**File:** `ingest/index_documents.py:132-154`
**Severity:** Medium — data staleness accumulates over time

### Problem

`main()` only calls `_index_file` for files that exist on disk. Files deleted from disk leave behind: fingerprint rows, extracted entities, relationships, chunk metadata, and Qdrant vectors. These accumulate indefinitely, polluting search results and graph traversal.

`store.list_all_paths()` already returns all tracked filepaths — the machinery is there.

### Recommended fix

After scanning `DOCS_PATH`, compute `set(store.list_all_paths()) - {normalize_path(f) for f in files}`. For each orphaned path, call `store.delete_file_data(filepath)`, `client.delete(prior_ids)`, and `store.delete_hash(filepath)`. Add this as a cleanup step before the indexing loop.

### Consideration

On very large document sets, `list_all_paths()` pulls all paths into memory. Acceptable for typical RAG corpus sizes (thousands, not millions of files).

---

## 4. _prepare_file reads the entire file before the hash-skip check

**File:** `ingest/index_documents.py:35-74`
**Severity:** Medium — wasted I/O on every incremental re-run for unchanged files

### Problem

`_prepare_file` reads and buffers the entire file into `raw: list[bytes]`, joins and decodes it, then returns `(text, sha256)`. Only after returning does `_index_file` compare hashes and possibly return `"skipped"`. For unchanged files (the common case on incremental re-runs), the full allocation and I/O were wasted.

### Recommended fix

Split into two functions:

```python
def _hash_file(path: Path) -> str | None:
    """Streaming SHA-256, no text accumulation. Returns hex digest or None on error."""

def _read_file(path: Path) -> str | None:
    """Read and decode file text. Returns decoded text or None on error."""
```

In `_index_file`: call `_hash_file` first, compare against `stored_hash`, return `"skipped"` on match, then call `_read_file` only on cache miss. `_compute_hash` (already used by tests) is effectively `_hash_file` — consolidate them.

### Tradeoff

Changed files now open twice (hash pass + text pass). Both opens hit the OS page cache for small files; for large files, the two-pass cost is marginal compared to the embedding + LLM extraction that follows.

---

## 5. Uncaught exception in main() loop skips store.close()

**File:** `ingest/index_documents.py:146-149`
**Severity:** Medium — WAL not checkpointed on crash

### Problem

If `_index_file` raises (Ollama unreachable with no extraction cache, SQLite disk full during entity write), the exception propagates through line 147, crashing `main()` before `store.close()` at line 149. The SQLite WAL is left open. GC closes the connection on Linux, but the WAL checkpoint is skipped.

### Fix

Wrap the indexing loop:

```python
try:
    for f in tqdm(files, desc="Indexing"):
        counts[_index_file(f, store, client)] += 1
finally:
    store.close()
```

**Status: fixed in this session.**
