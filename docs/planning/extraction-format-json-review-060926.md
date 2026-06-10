# Extraction format="json" Fix - Code Review - 2026-06-09

Reviewed the uncommitted diff implementing the `@@@@@`-collapse fix: `format="json"`
on extraction-model `generate()` calls, plus two new `_parse_extraction_response`
fallback parsers and 3 new `ignore_patterns`.

Files: `config/index_config.yaml`, `src/local_graph_rag/graph/extractor.py`,
`src/local_graph_rag/rag/ollama_client.py`, `tests/test_graph.py`.

Verification at time of review:

- `pytest` passes: `116 passed`
- `ruff check .` passes
- End-to-end: 70/70 reprocessed files clean, 0 `@@@@@` DB-wide

This review is of the fix itself, not the data it already produced — the points
below are about robustness for *future* runs and stale-cache replays.

## 1. Parser Robustness - `_dict_to_result`

### P1 - Non-dict Top-Level JSON Raises Uncaught AttributeError

Location:

- `src/local_graph_rag/graph/extractor.py:88-91`

Problem:

`_dict_to_result(data)` calls `data.get(...)` assuming `data` is always a dict.
Any valid top-level JSON value that isn't a dict (`null`, a list, a string, a
number, a bool) raises `AttributeError: '...' object has no attribute 'get'`,
which is **not** caught by the `except (json.JSONDecodeError, ValueError)`
clauses around any of the 4 parse attempts.

Concretely reachable via:

- `_parse_extraction_response('null')` / `'[1,2,3]'`
- New Attempt 1b: `'{}\nnull'`, `'{}\n[1,2,3]'`, `'{}\n42'`
- New Attempt 3: bare response `'None'` -> substituted to `'null'` -> `json.loads` -> `None`

Fix:

Add an `isinstance(data, dict)` check in `_dict_to_result` (or at each call site)
before calling `.get()`.

### P1 - Dict Present but `entities`/`relationships` Value Not a List Raises Uncaught TypeError

Location:

- `src/local_graph_rag/graph/extractor.py:89-90`

Problem:

`data.get("entities", [])` only returns the `[]` default if the key is
**absent**. If the key is present with value `null`, an int, or a bool, `.get()`
returns that value, and the list comprehension raises
`TypeError: '...' object is not iterable`. This is **not** fixed by the
`isinstance(data, dict)` check above — `data` is a dict here, just with a
wrong-shaped value.

Concrete triggers:

- `_parse_extraction_response('{"entities": null, "relationships": []}')` -> `TypeError: 'NoneType' object is not iterable`
- `_parse_extraction_response('{"entities": 5, "relationships": []}')` -> `TypeError: 'int' object is not iterable`

`"entities": null` is a plausible model output for "found nothing".

Fix:

In `_dict_to_result`, coerce non-list values to `[]` before iterating, e.g.
`entities_raw = data.get("entities") or []` then guard with
`isinstance(entities_raw, list)` (an `or []` alone doesn't handle a truthy
non-list like `5`).

### P1 - Either Crash Cascades Through `_write_index_data` Into a Retry-Waste Loop

Location:

- `src/local_graph_rag/graph/extractor.py:172-191` (`extract_entities_for_file`, no try/except around the parse call)
- `src/local_graph_rag/ingest/index_documents.py` (`_write_index_data`, `_index_file`)

Problem:

If either exception above escapes `extract_entities_for_file`'s per-batch loop,
it propagates into `_write_index_data`, which has **already** run
`store.register_chunks` and `client.upsert` but **not yet** `store.upsert_hash`
(the last line). `_index_file`'s broad `except Exception` catches it and returns
`"failed"`.

Consequence on the next run:

- `current_hash != stored_hash` still holds (hash was never updated).
- `_delete_file` removes the chunks/Qdrant points just written this run.
- `clear_extraction_cache` wipes even the batches that successfully cached
  *before* the crash.
- The file re-embeds and re-issues LLM calls for batches that already
  succeeded — rolling waste on every failed attempt, and a file can get stuck
  indefinitely if the triggering batch produces a similarly-shaped response
  on retry.

Fix:

Wrap the per-batch parse/extract step in `extract_entities_for_file` (or the
call to `_parse_extraction_response`) so a single bad batch degrades to an
empty result for that batch instead of aborting the whole file.

## 2. Attempt 3 (`None` -> `null` substitution)

### P2 - Regex Can Silently Corrupt Legitimate String Content

Location:

- `src/local_graph_rag/graph/extractor.py:74`

Problem:

`re.sub(r"\bNone\b", "null", text)` operates on the whole response text,
including inside JSON string values. If a string value legitimately contains
the word "None" (e.g. a Python-docstring-derived description like "Returns
None if not found") **and** the response is *also* invalid elsewhere due to an
unrelated bare Python `None`, the substitution rewrites both occurrences. The
result parses successfully, so the corrupted description ("Returns null if not
found") is silently persisted to `extraction_cache` and the entity DB — no
warning.

Given this tool indexes Python source, "Returns None"/"or None"-style
descriptions in docstrings are common.

Fix:

Scope the substitution to bareword `None` in non-string positions (e.g. only
where it follows `:` or `,` and is followed by `,`/`}`/`]`, not preceded by a
quote), or accept the risk and document it explicitly.

### P2 - Regex Operates on Full Text, Missing Cases Attempt 2 Already Isolated

Location:

- `src/local_graph_rag/graph/extractor.py:74`

Problem:

Attempt 3 runs `re.sub` on the original `text`, not on Attempt 2's
already-isolated `match.group()`. A response with leading prose plus an
embedded bare `None` — e.g.
`'Here is the result: {"entities": [], "relationships": [{"source":"A","target":"B","label":"uses","extra": None}]}'`
— has Attempt 2 successfully isolate the `{...}` block (still invalid due to
`None`), but Attempt 3's substitution on the full text (with leading prose)
still fails `json.loads`, falling through to the empty-result warning. The
recoverable case is missed.

Fix:

Also try the substitution on Attempt 2's `match.group()` if Attempt 2's
`json.loads` failed, before falling through to the empty-result warning.

## 3. Configuration - `ignore_patterns`

### P3 - `.claude` Added to Global `ignore_patterns`, Not Scoped

Location:

- `config/index_config.yaml` (`.claude` under the top-level `ignore_patterns:`, alongside `.codex`/`.idea`/`.vscode` which are scoped only to the Code path's `exclude_dirs`)

Problem:

The global `ignore_patterns` list applies to all 7 Nextcloud index paths plus
Code. `matches_ignore_pattern` does an exact path-component match
(`pattern in parts`). Any directory literally named `.claude` anywhere under
the indexed Nextcloud trees — e.g. a user-curated notes/reference folder — is
now silently excluded from the RAG index, with no log distinguishing
"intentional tool-config exclusion" from "user content gone missing".

Fix:

Confirm no Nextcloud path actually contains a directory named `.claude` that
should be indexed. If there's any doubt, scope `.claude` to the Code path's
`exclude_dirs` instead of the global list.

### P3 - `.claude` Not Mirrored Into Code Path's `exclude_dirs`

Location:

- `config/index_config.yaml`

Problem:

The new `.claude` entry lives only in the global `ignore_patterns`, while
`.codex`/`.idea`/`.vscode` live only in the Code path's `exclude_dirs` — two
partially-overlapping exclusion lists with different scopes. Functionally
harmless today, but a future edit to one list without the other could
silently diverge.

Fix:

Either consolidate into one list/scope, or leave a short comment noting the
two lists are intentionally separate and why.

## 4. Code Cleanliness & Simplification

### P3 - Four Parse Attempts Share Near-Identical Structure

Location:

- `src/local_graph_rag/graph/extractor.py:43-85`

Problem:

Attempts 1/1b/2/3 each follow the same
try/`json.loads`/`_dict_to_result`/`return` shape (~36 lines total). This also
means the `_dict_to_result` validation fixes from Section 1 would need to be
correct in one place but are currently *called* from 4 sites.

Fix:

Collapse into a candidate-list + shared loop (~18 lines), trying each
candidate string in order and returning on the first that produces a valid
`ExtractionResult`. Natural to do alongside the Section 1 fixes.

### P3 - Attempts 1b/3 Are Likely Cache-Replay-Only, Not Documented as Such

Location:

- `src/local_graph_rag/graph/extractor.py:54-65, 71-81`

Problem:

Attempts 1b and 3 are likely only reachable via stale pre-`format="json"`
cached responses being replayed (`extract_entities_for_file`'s
`if i in cached:` branch) — grammar-constrained fresh output shouldn't produce
`'{}\n...'` or bare Python `None`. This isn't documented in code. A future
reader might assume `format="json"` makes these branches fully dead and remove
them, when they're still live for the cache-replay path.

Fix:

Add a one-line comment noting these attempts exist for stale cached responses
predating `format="json"`, and can be removed once those caches are no longer
relevant.

## 5. Actionable Punch List

### PRIORITY 1 - MUST FIX

- [ ] Add `isinstance(data, dict)` guard in `_dict_to_result` - `extractor.py:88-91`
- [ ] Guard against `entities`/`relationships` values that are present but not lists (`null`, int, bool) - `extractor.py:89-90`
- [ ] Prevent a single bad batch from aborting the whole file's extraction (and thus skipping `upsert_hash`) - `extractor.py:172-191`, `index_documents.py` (`_write_index_data`)

### PRIORITY 2 - SHOULD FIX

- [ ] Scope the `\bNone\b` -> `null` substitution away from JSON string values, or document the corruption risk - `extractor.py:74`
- [ ] Also try the `None`->`null` substitution on Attempt 2's isolated match before giving up - `extractor.py:74`

### PRIORITY 3 - NICE TO FIX

- [ ] Verify no Nextcloud path has a `.claude` directory that should be indexed; otherwise scope the pattern - `config/index_config.yaml`
- [ ] Reconcile/document the two IDE-tooling exclusion lists (`ignore_patterns` vs Code path `exclude_dirs`) - `config/index_config.yaml`
- [ ] Collapse the 4 parse attempts into a candidate-list + shared loop - `extractor.py:43-85`
- [ ] Comment that Attempts 1b/3 are for stale-cache replay only - `extractor.py:54-65, 71-81`
