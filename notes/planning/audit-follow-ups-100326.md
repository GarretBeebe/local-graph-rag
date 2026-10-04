# Audit Follow-ups — 2026-10-03

Source: the principal-engineer audit of `7b4900b` (2026-10-03). All 4 **P1** and 13 **P2** items
were fixed in one change set; this document tracks what remains open (**P3**) and how to roll out
the P1/P2 changes safely.

---

## Deploy checklist for the P1/P2 changes

The changes include one-way SQLite schema migrations, a docker-compose environment change, and a
dependency removal. In order:

1. **Back up `graph.db`** before the new image first starts. Migrations v1–v3 run automatically on
   the first `GraphStore()` open (api, indexer, or summarizer) and cannot be undone. The SQLite
   backup API copies a consistent snapshot even while the api is running:

   ```bash
   docker compose run --rm --no-deps api python -c "import sqlite3; \
     sqlite3.connect('/app/data/graph.db').backup(sqlite3.connect('/app/data/graph.db.bak-100326'))"
   ```

2. **Review `.env`.** `api`, `indexer` and `summarizer` now load **every** variable in `.env`
   (`env_file`). Only container wiring is pinned in `docker-compose.yml`: `QDRANT_HOST/PORT/URL`,
   `OLLAMA_BASE_URL`, `SQLITE_PATH`, `INDEX_CONFIG_PATH`.
   - `TRUSTED_PROXY_IPS`: **checked 2026-10-04, then removed from `.env`.**
     - Requests through Caddy reach the API from `172.22.0.1`, Docker Desktop's gateway, so Caddy's
       address (`192.168.68.69`) never matched and the setting never took effect.
     - Decision: rely on Caddy's per-IP rate limits. The app's limiter stays one global bucket,
       and the login cookie has no Secure flag behind Caddy.
     - Trusting the gateway instead would let any LAN device that reaches port 8003 directly
       spoof its client IP.
     - If you ever do trust a proxy, `web/security.py` uses the **leftmost** `X-Forwarded-For`
       entry. That is safe only if the proxy overwrites the header.
   - Variables exported in the shell (`GEN_MODEL=x docker compose up`) no longer reach containers;
     put them in `.env`.
   - An uncommented but empty numeric value (e.g. `CHUNK_SIZE=`) now fails at startup instead of
     silently falling back.

3. **Rebuild and recreate:** `docker compose build api && docker compose up -d api`. This
   rebuilds the image shared with `indexer` and `summarizer`; `python-louvain` is gone from
   `uv.lock`. Recreate the api **before** the next indexer run: an old-image indexer writing after
   the migration would create entities without provenance rows.

4. **Check the migration** (expect `3`, `ok`, `[]`):

   ```bash
   docker compose exec api python -c "import sqlite3; c = sqlite3.connect('/app/data/graph.db'); \
     print(c.execute('PRAGMA user_version').fetchone()[0], \
           c.execute('PRAGMA integrity_check').fetchone()[0], \
           c.execute('PRAGMA foreign_key_check').fetchall())"
   ```

5. **Run the summarizer once.** Only communities whose content changed get a new summary. Any
   community whose members, descriptions and relationships match a stored summary reuses it, even
   if NetworkX's Louvain gave it a new id, so no LLM call is made. Later runs on an unchanged graph
   make no LLM calls.
   - Actual result on 2026-10-04: 3 of 105 communities summarized and 102 reused, in about
     2 minutes; a second run summarized none.
   - Changes can raise the count: a different partition on a denser graph, or relationship lines
     that were duplicated across documents (now de-duplicated). Each re-summarized community
     costs about 20 s on CPU.

6. **Expect re-extraction only for files that get reprocessed.** Migration v3 dropped the old
   extraction cache because its rows had no request hash and could not be replayed safely. Files
   whose fingerprint is unchanged are still skipped.

7. **Optional: re-index markdown.** Two fixes only reach files that are re-indexed:
   - the fence-aware markdown chunking;
   - exact entity provenance. The v2 backfill attributes each entity's merged description to every
     referencing file, so retraction is exact only once those files have been re-indexed.

   To apply them everywhere, drop fingerprints for the affected files. That costs re-chunking,
   re-embedding and re-extraction time on CPU.

---

## Deliberate deviations from the audit wording

- **Pipeline concurrency still equals `GENERATION_CONCURRENCY_LIMIT`.** Only the forced clamp to 1
  was removed, now that `GraphStore` uses thread-local connections.
  - Decoupling the two would move queueing from the 240 s capacity wait into Ollama's 120 s
    generation-slot wait. On CPU hardware that turns ordinary queueing into stream timeouts.
  - Settings now require `RAG_EXECUTOR_WORKERS > GENERATION_CONCURRENCY_LIMIT`, because `/v1/models`
    shares the executor.
- **The generation slot is not released during retry sleeps.** Read timeouts are no longer retried,
  so the remaining hold is at most `OLLAMA_MAX_RETRIES × 1 s` on fast-fail paths. That is not worth
  a new parameter.
- **`expand_neighborhood` returns boundary edges.** The plan said relationships would be limited to
  pairs of selected entities. Verification showed that when seeds alone fill the 20-entity budget,
  that leaves 0–1 relationship lines. Relationships among the selected entities still come first;
  remaining slots go to edges linking them to unselected entities, heaviest first.
- **P3 #6 is partly done.** The test-only `upsert_entity` / `upsert_relationship` were removed: the
  provenance model needs a `source_doc` on every entity write. Tests use
  `tests/helpers.add_entity` / `add_relationship` instead.

---

## P3 — open (audit punch list)

```
PRIORITY 3 — NICE TO FIX (readability, style, minor cleanup)
  [ ] Replace the pre-route get_communities() with an EXISTS check; load communities only on the global path — rag/query_graph_rag.py (_build_prompt)
  [ ] Make embed() a wrapper over embed_batch(); drop the superseded /api/embeddings endpoint — rag/embed.py (embed)
  [ ] A/B nomic-embed-text task prefixes (search_query: / search_document:) on the retrieval-ranking queries before any full re-embed — rag/embed.py (_prepare_text)
  [ ] Summarizer prompt: entity names instead of slugs, cap entities per prompt; remove log-and-reraise try blocks — graph/summarizer.py (_build_summary_prompt, summarize_community)
  [ ] Single GraphMode definition; derive _VALID_MODES with typing.get_args — web/schemas.py:17, rag/query_graph_rag.py:21,125
  [ ] Delete the unused ollama_client.post (the test-only upsert_entity/upsert_relationship are already gone) — rag/ollama_client.py (post)
  [ ] Extract _put_with_timeout in the stream worker; use a seconds-per-hour constant for the cookie max_age — web/rag_executor.py (_start_stream_worker), web/routes.py (login)
  [ ] ENV UV_COMPILE_BYTECODE=1; drop UV_SYSTEM_PYTHON (no-op for uv sync); try removing gcc/g++ — Dockerfile
  [ ] Throttle markdown rendering to one per requestAnimationFrame — web/static/app.js (_renderStream)
  [ ] Store sha256(token) instead of raw session tokens — web/user_store.py (create/validate/delete_session)
```

## Follow-up candidates found while fixing P1/P2

- **Non-streaming `generate()` can't be cancelled mid-request.** After a 504 the worker keeps the
  generation slot until Ollama finishes. Consider streaming internally and stopping on `cancel`.
- **Relationship drop rate.** Each "Indexed …" log line now reports
  `N of M extracted relationships kept`. If the drop rate is high, auto-create the missing endpoint
  entities (type `OTHER`) instead of discarding the edge.
- **Seed by identifier.** When `slugify(<identifier in the question>)` is itself an entity id, seed
  `expand_neighborhood` with it directly.
- **Empty `.env` values.** Treat `""` as unset in `settings.py` so an empty line falls back to the
  default instead of failing (numbers) or passing an empty model name (strings).
- **`X-Forwarded-For` parsing.** If a proxy that appends to the header is ever used, switch to
  rightmost-untrusted parsing.
- **Unguarded fingerprint read.** The `get_hash` call at the top of `_index_file` isn't guarded,
  so a SQLite "database is locked" after the 30 s busy timeout aborts the whole run instead of
  counting one failure. The matching write for empty files is guarded as of 2026-10-04.
- **`SqliteStore.close()` racing a thread's first connection** (read from the code, not
  reproduced). A thread opening its connection while another thread runs `close()` can cache a
  closed connection. API shutdown stops the executor before `close()`, which limits the exposure.

## Product decisions (out of scope for code review)

- Chat history is discarded: only the latest user message drives retrieval and generation, so
  follow-up questions lose context.
- `/v1/models` lists every Ollama model, including embedding models that cannot generate.
