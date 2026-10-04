"""Document ingestion pipeline: chunk, embed, extract entities, upsert into Qdrant + graph."""

import hashlib
import logging
import os
import re
import sys
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from local_graph_rag.common.logging import configure_cli_logging
from local_graph_rag.common.paths import (
    has_allowed_extension,
    is_under_any_root,
    matches_ignore_pattern,
    normalize_extensions,
    normalize_path,
)
from local_graph_rag.common.qdrant import get_qdrant_client
from local_graph_rag.graph.extractor import ExtractionResult, extract_entities_for_file
from local_graph_rag.graph.store import GraphStore, slugify
from local_graph_rag.ingest.chunkers import chunk_document
from local_graph_rag.ingest.doc_config import IndexConfig, IndexPath, load_index_config
from local_graph_rag.rag.embed import embed_batch
from local_graph_rag.settings import (
    ALLOWED_EXTENSIONS,
    COLLECTION,
    DOCS_PATH,
    MAX_INDEX_FILE_BYTES,
    VECTOR_SIZE,
)

logger = logging.getLogger(__name__)

try:
    _INDEX_CONFIG: IndexConfig | None = load_index_config()
    _CONFIG_LOAD_ERROR: BaseException | None = None
except Exception as _e:
    _INDEX_CONFIG = None
    _CONFIG_LOAD_ERROR = _e

_ALLOWED: frozenset[str] = (
    _INDEX_CONFIG.allowed_extensions if _INDEX_CONFIG
    else normalize_extensions(ALLOWED_EXTENSIONS)
)
_DOCS_ROOTS: list[Path] = (
    _INDEX_CONFIG.roots if _INDEX_CONFIG else [DOCS_PATH.resolve()]
)
_collection_ensured = False
_HASH_READ_BLOCK_BYTES = 65536


def _is_safe_indexable_file(path: Path) -> bool:
    """Return True only if path is a real file inside a known root with an allowed extension."""
    if path.is_symlink():
        return False
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False
    return (
        resolved.is_file()
        and is_under_any_root(resolved, _DOCS_ROOTS)
        and has_allowed_extension(resolved, _ALLOWED)
        and _is_within_size_limit(resolved)
    )


def _is_within_size_limit(path: Path) -> bool:
    """Return True if the file is within MAX_INDEX_FILE_BYTES."""
    try:
        size = path.stat().st_size
    except OSError as e:
        _warn_unreadable_file(path, e)
        return False
    if size > MAX_INDEX_FILE_BYTES:
        logger.warning(
            "Skipping oversized file %s: %d bytes > %d", path, size, MAX_INDEX_FILE_BYTES
        )
        return False
    return True


def _compute_hash(path: Path) -> str:
    """Return SHA-256 hex digest of a file, reading in 64 KB blocks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_HASH_READ_BLOCK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


def _hash_file(path: Path) -> str | None:
    """Return SHA-256 hex digest without buffering file contents. Returns None on error."""
    try:
        return _compute_hash(path)
    except Exception as e:
        _warn_unreadable_file(path, e)
        return None


def _read_file(path: Path) -> str | None:
    """Read and decode file text. Only called after a hash-miss confirms the file changed."""
    try:
        return path.read_bytes().decode(errors="ignore")
    except Exception as e:
        _warn_unreadable_file(path, e)
        return None


def _warn_unreadable_file(path: Path, error: BaseException) -> None:
    logger.warning("Skipping unreadable file %s: %s", path, error)


def _delete_vectors(store: GraphStore, client: QdrantClient, filepath: str) -> None:
    prior_ids = store.get_chunks_for_file(filepath)
    if prior_ids:
        client.delete(collection_name=COLLECTION, points_selector=PointIdsList(points=prior_ids))


def _vector_config_value(config: object, name: str) -> object:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _normalize_distance(value: object) -> str:
    text = str(value or "")
    return text.rsplit(".", 1)[-1].lower()


def _validate_collection_config(client: QdrantClient) -> None:
    """Fail fast if an existing collection cannot accept this app's embeddings."""
    info = client.get_collection(collection_name=COLLECTION)
    params = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    if isinstance(vectors, dict) and "" in vectors:
        vectors = vectors[""]

    actual_size = _vector_config_value(vectors, "size")
    actual_distance = _normalize_distance(_vector_config_value(vectors, "distance"))
    expected_distance = _normalize_distance(Distance.COSINE)
    if actual_size != VECTOR_SIZE or actual_distance != expected_distance:
        raise RuntimeError(
            f"Qdrant collection {COLLECTION!r} has vector config "
            f"size={actual_size}, distance={actual_distance!r}; expected "
            f"size={VECTOR_SIZE}, distance={expected_distance!r}. Recreate the collection "
            "or set VECTOR_SIZE/embedding model to match existing data."
        )


def ensure_collection(client: QdrantClient) -> None:
    """Ensure the Qdrant collection exists and has the def_name payload index.

    Two responsibilities gated by one _collection_ensured flag: create the
    collection if missing, and ensure the def_name keyword payload index exists
    for exact-match lookups in local_retrieval. create_payload_index is
    idempotent, so calling it on an already-indexed field is a cheap no-op.
    """
    global _collection_ensured
    if _collection_ensured:
        return
    if not client.collection_exists(COLLECTION):
        logger.info("Collection %r not found — creating", COLLECTION)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        _validate_collection_config(client)
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="def_name",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    _collection_ensured = True


# Below this length, word-boundary matches on common short tokens are mostly noise.
_MIN_ENTITY_NAME_CHARS = 3


def _match_entity_chunks(
    chunks: list[str],
    chunk_ids: list[str],
    entities: list[dict],
    entity_ids: list[str],
) -> list[tuple[str, str]]:
    """Pair each chunk with entities whose name appears in that chunk's text.

    Word-boundary, case-insensitive matching: avoids "Go" matching inside
    "Going"/"Embargo", and "GraphStore" matching inside "GraphStoreImpl" (no
    `\\w`-boundary at that join point) the way plain substring search would.

    Still a heuristic, not a resolver: it cannot unify spelling/spacing variants
    ("GraphStore" vs "Graph Store"), aliases, pluralization, or entities the LLM
    only referenced by pronoun/paraphrase. It is nonetheless strictly more precise
    than linking every entity in the file to every chunk. Names shorter than
    _MIN_ENTITY_NAME_CHARS are skipped outright — short tokens ("Go", "It", "C")
    match constantly via word boundaries too and would poison links broadly.
    """
    pairs: list[tuple[str, str]] = []
    for entity, entity_id in zip(entities, entity_ids, strict=True):
        name = (entity.get("name") or "").strip()
        if len(name) < _MIN_ENTITY_NAME_CHARS:
            continue
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        for chunk_id, chunk_text in zip(chunk_ids, chunks, strict=True):
            if pattern.search(chunk_text):
                pairs.append((chunk_id, entity_id))
    return pairs


def _write_index_data(
    filepath: str,
    chunks: list[str],
    points: list[PointStruct],
    store: GraphStore,
    client: QdrantClient,
    current_hash: str,
) -> ExtractionResult:
    """Register chunks, upsert to Qdrant, write graph data. Returns the extraction result.

    SQLite chunk IDs registered BEFORE Qdrant upsert: if Qdrant fails, IDs survive
    in SQLite for retry. Fingerprint written last — a crash before that line leaves no
    fingerprint so the file is retried on the next run.
    """
    point_ids = [p.id for p in points]
    store.register_chunks([(pid, filepath, i) for i, pid in enumerate(point_ids)])
    client.upsert(collection_name=COLLECTION, points=points)
    result = extract_entities_for_file(chunks, filepath, store)
    entity_ids = store.upsert_entities(result.entities, filepath)
    store.upsert_relationships([
        (slugify(rel["source"]), slugify(rel["target"]), rel["label"], filepath)
        for rel in result.relationships
    ])
    store.link_chunks(_match_entity_chunks(chunks, point_ids, result.entities, entity_ids))
    if not result.had_failure:
        store.upsert_hash(filepath, current_hash)
    return result


def _cleanup_changed_file(filepath: str, store: GraphStore, client: QdrantClient) -> None:
    """Delete a changed file's Qdrant vectors, then its SQLite data and fingerprint.

    Qdrant first: if Qdrant fails, chunk IDs remain in SQLite so the next run
    can retry. If SQLite fails after Qdrant, stale SQLite rows are cleaned on
    next run's delete_file_data and the Qdrant delete is idempotent.
    """
    _delete_vectors(store, client, filepath)
    store.delete_file_data(filepath)


def _chunk_file(path: Path, text: str) -> list[tuple[str, str | None]]:
    return [(c.strip(), name) for c, name in chunk_document(path, text) if c.strip()]


def _build_points(
    filepath: str,
    chunked: list[tuple[str, str | None]],
    vectors: list[list[float]],
) -> list[PointStruct]:
    return [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"text": chunk, "filepath": filepath, "chunk_index": i, "def_name": def_name},
        )
        for i, ((chunk, def_name), vec) in enumerate(zip(chunked, vectors, strict=True))
    ]


def _index_file(path: Path, store: GraphStore, client: QdrantClient) -> str:
    """Process one file through the full pipeline. Returns 'indexed' | 'skipped' | 'failed'.

    'failed' covers an unreadable file; any other error propagates to main(). The new
    version is chunked and embedded before the old one is deleted, so a failure there
    (e.g. Ollama unreachable) leaves the previous version searchable.
    """
    filepath = normalize_path(path)

    current_hash = _hash_file(path)
    if current_hash is None:
        return "failed"
    if current_hash == store.get_hash(filepath):
        return "skipped"

    text = _read_file(path)
    if text is None:
        return "failed"

    chunked = _chunk_file(path, text)
    if not chunked:
        logger.info("No chunks produced for %s — removing any previous index data", path)
        _cleanup_changed_file(filepath, store, client)
        # Mark the empty content as processed so it isn't re-read every run; restoring the
        # old content changes the hash again, so it gets re-indexed.
        store.upsert_hash(filepath, current_hash)
        return "skipped"
    chunks = [c for c, _ in chunked]

    vectors = embed_batch(chunks)
    _cleanup_changed_file(filepath, store, client)
    points = _build_points(filepath, chunked, vectors)
    result = _write_index_data(filepath, chunks, points, store, client, current_hash)

    logger.info(
        "Indexed %s: %d chunks, %d entities, %d of %d extracted relationships kept",
        path.name,
        len(points),
        len(result.entities),
        len(result.relationships),
        result.relationships_extracted,
    )
    return "indexed"


def _accept(fpath: Path, ignore: list[str]) -> bool:
    return not matches_ignore_pattern(fpath.name, ignore) and _is_safe_indexable_file(fpath)


def _walk_index_path(ip: IndexPath, ignore: list[str], unscanned: list[Path]) -> list[Path]:
    """Return accepted files under ip, appending any directory that fails to list."""
    if not ip.path.is_dir():
        return []

    def _mark_unscanned(path: str, error: OSError) -> None:
        logger.warning("Cannot scan %s: %s", path, error)
        unscanned.append(Path(normalize_path(path)))

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(
        ip.path, topdown=True, onerror=lambda error: _mark_unscanned(error.filename, error)
    ):
        # Emptying dirnames stops a non-recursive index path at its own directory.
        dirnames[:] = [
            d for d in dirnames
            if ip.recursive and d not in ip.exclude_dirs and not matches_ignore_pattern(d, ignore)
        ]
        try:
            files.extend([p for p in (Path(dirpath) / f for f in filenames) if _accept(p, ignore)])
        except OSError as error:
            # Readable but not enterable: names are listed, but the files can't be inspected.
            _mark_unscanned(dirpath, error)
    return files


def _collect_files() -> tuple[list[Path], list[Path]]:
    """Return (indexable files, unscanned paths) from index_paths or the DOCS_PATH fallback.

    Unscanned paths are roots that are missing, empty, or unreadable, plus subdirectories
    that failed to list. A file's absence there proves nothing — a bind-mounted share that
    is offline looks empty — so main() must not treat their indexed files as deleted.
    """
    if _CONFIG_LOAD_ERROR is not None:
        raise RuntimeError(
            f"Failed to load index config: {_CONFIG_LOAD_ERROR}"
        ) from _CONFIG_LOAD_ERROR

    index_paths = _INDEX_CONFIG.index_paths if _INDEX_CONFIG else [IndexPath(DOCS_PATH)]
    ignore = _INDEX_CONFIG.ignore_patterns if _INDEX_CONFIG else []
    files: list[Path] = []
    unscanned: list[Path] = []
    for ip in index_paths:
        found = _walk_index_path(ip, ignore, unscanned)
        if not found:
            logger.warning(
                "No indexable files under %s (missing, unreadable, or empty) — "
                "keeping its existing index entries",
                ip.path,
            )
            unscanned.append(ip.path)
        files.extend(found)

    logger.info("Found %d indexable files across %d index path(s)", len(files), len(index_paths))
    return files, unscanned


def main() -> None:
    configure_cli_logging()

    store = GraphStore()
    client = get_qdrant_client()
    ensure_collection(client)

    files, unscanned = _collect_files()

    on_disk = {normalize_path(p) for p in files}
    missing = set(store.list_all_paths()) - on_disk
    stale = {p for p in missing if not is_under_any_root(Path(p), unscanned)}
    if len(stale) < len(missing):
        logger.warning(
            "Keeping %d indexed file(s) under paths that could not be scanned",
            len(missing) - len(stale),
        )
    for stale_path in sorted(stale):
        _delete_vectors(store, client, stale_path)
        store.purge_file(stale_path)
        logger.info("Removed stale: %s", stale_path)
    if stale:
        logger.info("Cleaned up %d stale file(s)", len(stale))

    counts: dict[str, int] = {"indexed": 0, "skipped": 0, "failed": 0}
    try:
        for f in tqdm(files, desc="Indexing"):
            try:
                outcome = _index_file(f, store, client)
            except Exception:
                # e.g. Ollama or Qdrant unreachable, "database is locked", or a parser
                # RecursionError: count one failed file instead of aborting the run.
                logger.exception("Indexing failed for %s", f)
                outcome = "failed"
            counts[outcome] += 1
    finally:
        store.close()
    print(
        f"Done — indexed: {counts['indexed']}, "
        f"skipped: {counts['skipped']}, "
        f"failed: {counts['failed']}"
    )
    if counts["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
