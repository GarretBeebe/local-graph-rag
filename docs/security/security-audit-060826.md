# Security Audit - 2026-06-08

## Security Architecture Map

Entry points reviewed: CLI ingestion (`python -m ingest.index_documents`), CLI query (`python -m api.query_graph_rag`), Docker services, Qdrant/Ollama clients. The planned FastAPI server is referenced in Compose, but `web/api_server.py` is not present.

Authentication and session model: no implemented application authentication or session layer exists in the current codebase. Qdrant uses an API key via environment configuration. Ollama is trusted as a local service.

Authorization model: no implemented user, role, or permission model exists in the current codebase.

Major trust boundaries:

- `DOCS_PATH` files are treated as ingestible content.
- Qdrant and SQLite persist document chunks, embeddings, entities, relationships, and summaries.
- Ollama receives document text and user questions as prompts.
- Docker services communicate with Qdrant and host Ollama.

Critical data flows:

- Document file -> chunking -> embeddings -> Qdrant payloads.
- Document file -> LLM entity extraction -> SQLite graph store.
- SQLite community graph -> LLM summarization -> summary embeddings.
- User question -> embedding/router/retrieval -> prompt construction -> Ollama generation.

Sensitive assets handled by the system:

- Indexed document contents.
- Vector payloads containing raw chunks.
- Graph database contents.
- Qdrant API key.
- Local filesystem files readable by the indexer process.

High-risk sinks and integration points reviewed:

- Filesystem reads.
- Qdrant writes/searches.
- SQLite queries.
- HTTP calls to Ollama.
- LLM prompt construction.

## Findings

### 1. Symlinked Files Under `DOCS_PATH` Can Be Indexed Outside the Intended Root

Vulnerability Type: Arbitrary local file indexing / sensitive file disclosure

Risk Level: Medium

Confidence: High, if untrusted users or processes can write into `DOCS_PATH`

Location:

- `ingest/index_documents.py:191`
- `ingest/index_documents.py:117`
- `common/paths.py:8`

Relevant OWASP ASVS Category: V8 Data Protection, V12 File and Resources

Affected Data Flow: filesystem entry under `DOCS_PATH` -> `Path.is_file()` follows symlink -> `normalize_path(...resolve())` resolves outside root -> `_read_file()` reads contents -> raw chunks stored in Qdrant payload and SQLite metadata.

Description: the indexer enumerates `DOCS_PATH.rglob("*")` and accepts any path where `p.is_file()` and the extension matches. `Path.is_file()` follows symlinks. The path is then normalized with `.resolve()`, but the resolved target is not checked to remain under `DOCS_PATH`. A symlink such as `documents/secret.txt -> /some/readable/secret.txt` can cause the indexer to read and persist a file outside the intended document root.

Exploit Scenario: an attacker with write access to the document folder adds a symlink named `notes.txt` pointing to a readable sensitive file. The next ingestion run hashes, reads, embeds, and stores that file's text. A later query can retrieve or summarize the sensitive contents.

Impact: unintended disclosure of local files into Qdrant/SQLite and model context.

Remediation: reject symlinks or enforce that the resolved file remains under the resolved document root before hashing/reading.

Secure Code Example:

```python
# ingest/index_documents.py

DOCS_ROOT = DOCS_PATH.resolve()

def _is_safe_indexable_file(path: Path) -> bool:
    if path.is_symlink():
        return False

    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False

    return (
        resolved.is_file()
        and resolved.is_relative_to(DOCS_ROOT)
        and has_allowed_extension(resolved, _ALLOWED)
    )

files = [p.resolve(strict=True) for p in DOCS_PATH.rglob("*") if _is_safe_indexable_file(p)]
```

Tests to Add: create a temp `DOCS_PATH`, add an allowed `.txt` symlink pointing outside the root, run file discovery, and assert it is not indexed. Add a positive test for a normal file inside the root.

### 2. Unbounded File Reads Allow Document-Driven Denial of Service

Vulnerability Type: Resource exhaustion / ingestion DoS

Risk Level: Medium

Confidence: High, if `DOCS_PATH` can contain untrusted or very large files

Location:

- `ingest/index_documents.py:27`
- `ingest/index_documents.py:45`
- `ingest/index_documents.py:191`

Relevant OWASP ASVS Category: V12 File and Resources

Affected Data Flow: document file -> `_compute_hash()` reads full stream -> `_read_file()` loads full file into memory -> chunking/embedding/extraction.

Description: the indexer has no maximum file size, file count, or aggregate ingestion limit. `_read_file()` uses `path.read_bytes().decode(...)`, which loads the full file into memory. A very large allowed-extension file, or many large files, can exhaust memory, disk, Qdrant storage, Ollama time, or CPU.

Exploit Scenario: an attacker places a multi-GB `.txt`, `.json`, or `.py` file in the indexed directory. The indexer loads it into memory and then attempts chunking/embedding/extraction, degrading or killing the ingestion process.

Impact: local denial of service and uncontrolled resource consumption.

Remediation: enforce per-file size limits before hashing/reading, and optionally enforce max file count / aggregate bytes per ingestion run.

Secure Code Example:

```python
# settings.py
MAX_INDEX_FILE_BYTES = int(os.environ.get("MAX_INDEX_FILE_BYTES", str(5 * 1024 * 1024)))

# ingest/index_documents.py
from settings import MAX_INDEX_FILE_BYTES

def _is_within_size_limit(path: Path) -> bool:
    try:
        size = path.stat().st_size
    except OSError as e:
        logger.warning("Skipping unreadable file %s: %s", path, e)
        return False

    if size > MAX_INDEX_FILE_BYTES:
        logger.warning(
            "Skipping oversized file %s: %d bytes > %d",
            path,
            size,
            MAX_INDEX_FILE_BYTES,
        )
        return False

    return True
```

Tests to Add: create a file one byte over the configured limit and assert `_index_file` or discovery skips it without reading or embedding it.

## Exploitability Notes

The strongest confirmed issue is local file disclosure through symlink indexing, but it requires an attacker or untrusted process to write into `DOCS_PATH`. If `DOCS_PATH` is strictly controlled by the operator, risk drops.

No high-confidence SQL injection, command injection, unsafe deserialization, hard-coded secrets, or authentication bypass was identified in the implemented Python code. Most SQLite access is parameterized. YAML parsing uses `safe_load`. There is no subprocess/eval sink in the active code paths reviewed.

The Docker `api` service currently points to `web.api_server:app`, but `web/api_server.py` is absent. That is a deployment correctness issue, not a confirmed security vulnerability in the present code.

## Recommended Fix Priority

1. Block symlink/out-of-root indexing before any production or shared-folder use.
2. Add file size and ingestion budget limits.
3. When the FastAPI phase is implemented, add authentication/authorization before exposing port `8000` beyond localhost.

## Suggested Security Tests

- Symlink rejection for allowed-extension files under `DOCS_PATH`.
- Resolved path must remain under `DOCS_PATH`.
- Oversized file rejection before read, chunk, embed, or extract.
- Regression tests proving document text cannot trigger filesystem, shell, SQL, or HTTP sinks.

## Areas Reviewed With No Confirmed Findings

Reviewed filesystem ingestion, SQLite graph storage, Qdrant client usage, Ollama HTTP calls, prompt construction, YAML config loading, Dockerfile, and Compose.

No confirmed SQL injection, command injection, unsafe pickle/deserialization, hard-coded application secrets, or cryptographic misuse was found in the implemented code.
