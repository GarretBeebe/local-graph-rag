# Post-Phase-6 Grade-It Review - 2026-06-09

Reviewed commit `fff0b9e` (`Post-Phase-6 review fixes: module split, correctness, simplify pass`).

Stack: Python 3.11, FastAPI, SQLite, Qdrant, Ollama, pytest, ruff.

Verification:

- `uv run pytest -q` passes: `96 passed`
- `uv run ruff check .` passes

## 1. Architecture & Design Patterns

### P2 - Routes Import Private Middleware Internals

`web.routes` imports private middleware internals from `web.middleware`.

Locations:

- `web/routes.py:21`
- `web/middleware.py:18`
- `web/middleware.py:35`
- `web/middleware.py:40`

Problem:

`web/routes.py` pulls `_AUTH_COOKIE`, `_extract_bearer_token`, and `_is_secure_request` out of the middleware module. That makes route handlers depend on middleware implementation details.

Fix:

Move shared auth/request helpers into a public module such as `web.auth_request` or `web.security_context`, then import them from there.

### P2 - RAG Executor Module Is Accumulating Responsibilities

Locations:

- `web/rag_executor.py:34`
- `web/rag_executor.py:156`
- `web/rag_executor.py:238`

Problem:

`web.rag_executor` now mixes executor lifecycle, global store/client state, timeout/capacity policy, sync-to-async queue bridging, disconnect watching, and OpenAI SSE chunk formatting.

Fix:

Split lifecycle/state from streaming response production if this module grows again.

## 2. Code Cleanliness & Readability

### P2 - Stream Worker Still Has Dense Queue/Error/Cancellation Logic

Location:

- `web/rag_executor.py:156`

Problem:

`_start_stream_worker` is 44 lines and nests thread worker queue publishing, exception delivery, sentinel delivery, capacity timing, and cancellation setup in one function.

Fix:

Extract only the queue-put-with-timeout operation into a named helper. Leave the surrounding control flow intact.

### P3 - Remaining Magic Hour Conversion

Location:

- `web/routes.py:102`

Problem:

`SESSION_EXPIRY_HOURS * 3600` still uses an inline conversion constant. `web/user_store.py` now has `_SECONDS_PER_HOUR`; this route should not keep a raw `3600`.

Fix:

Reuse a shared constant or define the same named constant in `web.routes`.

## 3. Best Practices & Anti-Patterns

### P2 - Highest-Value Fixes Lack Regression Tests

Locations:

- `web/middleware.py:86`
- `web/rag_executor.py:34`

Problem:

The two highest-value correctness fixes in the commit are untested:

- Security-header middleware should log handler exceptions and still return security headers.
- `init_rag_executor(..., concurrency_limit=2)` should clamp effective RAG capacity to one.

Fix:

Add focused unit tests around `security_headers_middleware` and `init_rag_executor`.

### P3 - Shutdown Cleanup Suppresses Failures Silently

Location:

- `web/rag_executor.py:59`

Problem:

`shutdown_rag_executor` suppresses `ollama_client.close_session` failures without logging. If cleanup fails, shutdown continues silently and the failure is not diagnosable.

Fix:

Log at debug or warning level before suppressing cleanup failures.

## 4. Actionable Punch List

### PRIORITY 1 - MUST FIX

None found.

### PRIORITY 2 - SHOULD FIX

- [ ] Move shared request/auth helpers out of `web.middleware` private symbols - `web/routes.py:21`
- [ ] Add regression tests for security-header exception handling and RAG concurrency clamping - `web/middleware.py:86`, `web/rag_executor.py:34`
- [ ] Extract queue-put-with-timeout helper from stream worker - `web/rag_executor.py:156`
- [ ] Keep `web.rag_executor` from accumulating route/SSE formatting responsibility - `web/rag_executor.py:238`

### PRIORITY 3 - NICE TO FIX

- [ ] Replace route-level `3600` with a named seconds-per-hour constant - `web/routes.py:102`
- [ ] Log suppressed shutdown cleanup failures - `web/rag_executor.py:59`
