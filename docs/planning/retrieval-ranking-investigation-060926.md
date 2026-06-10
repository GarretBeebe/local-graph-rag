# Retrieval quality investigation — findings (2026-06-09)

Findings/handoff doc from a debugging session.

**Status (2026-06-10):** Both issues resolved at the code level. Issue 2's
combined Option 1 + revised Option 2 design (below) is implemented and
tested in worktree `retrieval-ranking` (branch `worktree-retrieval-ranking`).
The live `.py` re-index (see "Re-index mechanics" below) has NOT yet been
run against the shared `data/graph.db`/Qdrant collection — until it runs, no
chunk in the live index has a `def_name` payload, so the new exact-match
fallback is inert in production despite passing tests.

## Issue 1: Query misrouting — FIXED & DEPLOYED

**Symptom:** "tell me what `_parse_extraction_response` within the
local-graph-rag project does" returned a vague, inference-only answer
("it is imported from local_graph_rag.graph.extractor, which suggests...").

**Root cause:** `_GLOBAL_KEYWORDS` in `query_router.py:13-16` included
`"project"`. Any question phrased "...within the X project" got routed to
global mode, which only sees community summaries (`global_retrieval.py`) —
never the actual source chunk.

**Fix applied:** removed `"project"` from `_GLOBAL_KEYWORDS`
(`query_router.py:15`). All 12 router tests still pass, including the
`"give me an overview of the project"` → global case (covered by
`"overview"` instead).

**Deployed:** `docker compose up -d --build api` — `graph-rag-api` rebuilt
and healthy.

## Issue 2: Embedding ranks import statements above definitions — RESOLVED (code+tests; re-index pending)

**Resolution:** Implemented as "Option 1 + Option 2, revised" below —
`chunk_python` drops bare `import`/`from...import` chunks entirely and tags
every emitted chunk with its enclosing top-level `def`/`class` name
(`def_name`); this is stored as a Qdrant `KEYWORD`-indexed payload field;
`local_retrieval.py` extracts identifier-shaped tokens from the question and
exact-matches them against `def_name`, injecting hits into `chunk_texts`. 13
new tests in `tests/test_chunkers.py` + 3 new tests in `tests/test_retrieval.py`;
ruff clean, 136/136 tests pass, build-validator GO. Remaining: live `.py`
re-index (see "Re-index mechanics" below) to populate `def_name` in the
production Qdrant collection.

**Symptom:** after the routing fix, the same question now correctly uses
*local* retrieval (response showed the `--[USES]-->` relationship format from
`_format_local`), but the "supporting text" was still just an import
statement, not the function body.

**Empirical evidence** (live queries against the `graph_documents` Qdrant
collection, `qdrant-client==1.18.0`):

```
1. score=0.8479  tests/test_graph.py#3
   'from local_graph_rag.graph.extractor import (ExtractionResult, _parse_extraction_response, ...'
...
(top 10 are ALL import statements / module docstrings)
```

The actual def-chunk (`extractor.py#13`, 1599 chars, confirmed current/correct
— matches source read directly) scores **0.7109** and doesn't make top-10.
`ENTITY_RETRIEVAL_K` defaults to 5.

Even non-import chunks rank above it: `README.md#1` (0.7645), `README.md#0`
(0.7541), `__init__.py` docstring (0.7403) — all higher than 0.7109.

**Root cause:** nomic-embed-text embeds short, identifier-dense chunks (a
12-token import line that's ~25% identifier names) closer to "what does
`<symbol>` do" queries than a 1599-char chunk where the same identifier is
~1% of the tokens, diluted among docstring + JSON-parsing/fallback logic.

## Options considered for Issue 2

**Option 1 — stop indexing bare import statements as standalone chunks.**
In `chunk_python` (`chunkers.py:108-118`), every top-level AST node —
including each `Import`/`ImportFrom` — becomes its own chunk via `_emit`.
Skip emitting import nodes. Cheap, broadly useful (these chunks carry ~no
"what does X do" signal for *any* query), but **not sufficient alone**: even
with all imports removed, README/`__init__.py` chunks (0.74–0.76) would still
outrank the def-chunk (0.711) for this query.

**Option 2, graph-based (chunk_entities lookup) — RULED OUT.**
Originally proposed: look up the entity named `_parse_extraction_response` in
`entities`, find its linked chunk via `chunk_entities`, fetch that chunk
directly. **Empirically dead**: queried the live `graph.db` —

```sql
SELECT id, name, type FROM entities WHERE id LIKE '%parse_extraction%'
  OR name LIKE '%parse_extraction%'   -- zero rows

SELECT DISTINCT e.id, e.name, e.type FROM chunk_entities ce
  JOIN chunks c ON ce.chunk_id = c.chunk_id
  JOIN entities e ON ce.entity_id = e.id
  WHERE c.filepath LIKE '%extractor.py'   -- zero rows
```

`_parse_extraction_response` was never extracted as an entity under any name,
and **no entity is linked to any chunk from `extractor.py` at all**. LLM
entity extraction doesn't reliably cover private helper functions — exactly
the symbols most likely to need this fallback. A `chunk_entities`-based
lookup would return nothing for this case.

**Option 2, revised (recommended direction) — AST-driven exact lookup.**
`chunk_python` already knows, from the AST, the `name` of every top-level
`def`/`class` it emits as a chunk. Carry that through:

- In `chunk_python`, tag each emitted chunk with its definition name (e.g.
  `node.name` for top-level `FunctionDef`/`AsyncFunctionDef`/`ClassDef`,
  `None` otherwise).
- In `index_documents.py`, store this as Qdrant payload (`def_name`,
  alongside existing `text`/`filepath`/`chunk_index`).
- Add a Qdrant `keyword` payload index on `def_name` (exact match — avoids
  full-text tokenizer ambiguity around leading underscores entirely).
- New retrieval-side lookup in `local_retrieval.py`: extract
  identifier-shaped tokens from the question (snake_case/leading-underscore/
  CamelCase), exact-match against `def_name` via `MatchValue`, inject any hit
  into `LocalContext.chunk_texts` alongside the vector-search results.

This is deterministic and independent of LLM entity extraction — the gap that
killed the graph-based approach. Naturally pairs with Option 1: both require
re-chunking `.py` files with a new payload shape, so they share one re-index
pass.

## Re-index mechanics (applies to either/both options)

`_index_file` (`index_documents.py:202`) skips files whose content hash is
unchanged — a chunker-only change won't trigger re-processing automatically.

Targeted approach: for every `.py` path in `store.list_all_paths()`, call
`store.delete_hash(path)`, then re-run
`docker compose --profile indexer run --rm indexer`. Non-`.py` files skip
(hash unchanged); `.py` files get fully re-chunked/re-embedded/re-extracted.

**Cost/risk to flag before running:**
- Re-runs LLM entity extraction (`EXTRACT_MODEL`) for every chunk of every
  `.py` file — could be slow depending on corpus size.
- `_delete_file` removes old entities/relationships sourced from `.py` files
  before re-extraction rebuilds them — `communities` table is NOT
  auto-rebuilt by the indexer (`detect_communities`/summarizer is a separate
  profile). May need `docker compose --profile summarizer run --rm
  summarizer` afterward so community summaries stay consistent with the
  rebuilt entity graph.

## Next steps (pick up next session)

1. ~~Finalize file-level plan~~ — done. See
   `/home/garret/.claude/plans/reflective-roaming-parasol.md`.
2. ~~Get user approval on the combined Option 1 + revised Option 2 design~~ — done.
3. ~~Implement~~ — done in worktree `retrieval-ranking`
   (branch `worktree-retrieval-ranking`). Remaining: merge to main, then run
   the targeted `.py` re-index and verify with the same live-Qdrant query
   technique used above (def-chunk should now be retrievable via exact
   `def_name` match regardless of its cosine rank).
4. Decide whether to re-run `summarizer` profile afterward.
