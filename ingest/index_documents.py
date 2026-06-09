"""Document ingestion pipeline: chunk, embed, extract entities, upsert into Qdrant + graph."""

import hashlib
import logging
import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams
from tqdm import tqdm

from api.embed import embed_batch
from common.paths import has_allowed_extension, normalize_extensions, normalize_path
from common.qdrant import get_qdrant_client
from graph.extractor import extract_entities_for_file
from graph.store import GraphStore, slugify
from ingest.chunkers import chunk_document
from settings import ALLOWED_EXTENSIONS, COLLECTION, DOCS_PATH, MAX_INDEX_FILE_BYTES, VECTOR_SIZE

logger = logging.getLogger(__name__)

_ALLOWED = normalize_extensions(ALLOWED_EXTENSIONS)
_collection_ensured = False
_DOCS_ROOT = DOCS_PATH.resolve()


def _is_safe_indexable_file(path: Path) -> bool:
    """Return True only if path is a real file inside DOCS_PATH with an allowed extension."""
    if path.is_symlink():
        return False
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False
    return (
        resolved.is_file()
        and resolved.is_relative_to(_DOCS_ROOT)
        and has_allowed_extension(resolved, _ALLOWED)
        and _is_within_size_limit(resolved)
    )


def _is_within_size_limit(path: Path) -> bool:
    """Return True if the file is within MAX_INDEX_FILE_BYTES."""
    try:
        size = path.stat().st_size
    except OSError as e:
        logger.warning("Skipping unreadable file %s: %s", path, e)
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
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _hash_file(path: Path) -> str | None:
    """Return SHA-256 hex digest without buffering file contents. Returns None on error."""
    try:
        return _compute_hash(path)
    except Exception as e:
        logger.warning("Skipping unreadable file %s: %s", path, e)
        return None


def _read_file(path: Path) -> str | None:
    """Read and decode file text. Only called after a hash-miss confirms the file changed."""
    try:
        return path.read_bytes().decode(errors="ignore")
    except Exception as e:
        logger.warning("Skipping unreadable file %s: %s", path, e)
        return None


def _delete_file(store: GraphStore, client: QdrantClient, filepath: str) -> None:
    """Delete Qdrant vectors then SQLite data for a file.

    Qdrant first: if Qdrant fails, chunk IDs remain in SQLite so the next run
    can retry. If SQLite fails after Qdrant, stale SQLite rows are cleaned on
    next run's delete_file_data and the Qdrant delete is idempotent.
    """
    prior_ids = store.get_chunks_for_file(filepath)
    if prior_ids:
        client.delete(collection_name=COLLECTION, points_selector=PointIdsList(points=prior_ids))
    store.delete_file_data(filepath)


def ensure_collection(client: QdrantClient) -> None:
    global _collection_ensured
    if _collection_ensured:
        return
    if not client.collection_exists(COLLECTION):
        logger.info("Collection %r not found — creating", COLLECTION)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
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


def _index_file(path: Path, store: GraphStore, client: QdrantClient) -> str:
    """Process one file through the full pipeline. Returns 'indexed' | 'skipped' | 'failed'."""
    filepath = normalize_path(path)

    # Hash-only pass: no text buffered, so unchanged files skip immediately
    current_hash = _hash_file(path)
    if current_hash is None:
        return "failed"
    stored_hash = store.get_hash(filepath)
    if current_hash == stored_hash:
        return "skipped"

    try:
        _delete_file(store, client, filepath)
        if stored_hash is not None:
            store.clear_extraction_cache(filepath)
    except Exception as e:
        logger.error("Cleanup before indexing failed for %s: %s", path, e)
        return "failed"

    text = _read_file(path)
    if text is None:
        return "failed"

    chunks = [c.strip() for c in chunk_document(path, text) if c.strip()]
    if not chunks:
        logger.info("No chunks produced for %s — skipping", path)
        return "skipped"

    try:
        vectors = embed_batch(chunks)
    except Exception as e:
        logger.error("Embedding failed for %s: %s", path, e)
        return "failed"

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"text": chunk, "filepath": filepath, "chunk_index": i},
        )
        for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
    ]

    try:
        # SQLite register BEFORE Qdrant upsert: if Qdrant fails, IDs in SQLite for next retry
        point_ids = [p.id for p in points]
        store.register_chunks([(pid, filepath, i) for i, pid in enumerate(point_ids)])
        client.upsert(collection_name=COLLECTION, points=points)

        result = extract_entities_for_file(chunks, filepath, store)
        entity_ids = store.upsert_entities(result.entities)
        store.upsert_relationships([
            (slugify(rel["source"]), slugify(rel["target"]), rel["label"], filepath)
            for rel in result.relationships
        ])

        store.link_chunks(_match_entity_chunks(chunks, point_ids, result.entities, entity_ids))
        # Write fingerprint last — crash before this line means the file has no fingerprint
        # and will be retried on next run.
        store.upsert_hash(filepath, current_hash)
    except Exception as e:
        logger.error("Indexing failed for %s: %s", path, e)
        return "failed"

    logger.info("Indexed %s: %d chunks, %d entities", path.name, len(points), len(entity_ids))
    return "indexed"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    store = GraphStore()
    client = get_qdrant_client()
    ensure_collection(client)

    files = [p for p in DOCS_PATH.rglob("*") if _is_safe_indexable_file(p)]
    logger.info("Found %d indexable files in %s", len(files), DOCS_PATH)

    on_disk = {normalize_path(p) for p in files}
    stale = set(store.list_all_paths()) - on_disk
    for stale_path in stale:
        _delete_file(store, client, stale_path)
        store.delete_hash(stale_path)
        logger.info("Removed stale: %s", stale_path)
    if stale:
        logger.info("Cleaned up %d stale file(s)", len(stale))

    counts: dict[str, int] = {"indexed": 0, "skipped": 0, "failed": 0}
    try:
        for f in tqdm(files, desc="Indexing"):
            counts[_index_file(f, store, client)] += 1
    finally:
        store.close()
    print(
        f"Done — indexed: {counts['indexed']}, "
        f"skipped: {counts['skipped']}, "
        f"failed: {counts['failed']}"
    )


if __name__ == "__main__":
    main()
