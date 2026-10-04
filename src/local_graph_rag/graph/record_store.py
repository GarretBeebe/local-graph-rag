"""Entity, relationship, chunk, cache, and fingerprint store methods."""

import json
import sqlite3

from local_graph_rag.graph.store_utils import slugify

_UPSERT_RELATIONSHIP_SQL = """
INSERT INTO relationships (source_id, target_id, label, weight, source_doc)
VALUES (?, ?, ?, 1.0, ?)
ON CONFLICT(source_id, target_id, label, source_doc)
DO UPDATE SET weight = weight + 1.0
"""

_UPSERT_MENTION_SQL = """
INSERT INTO entity_mentions (entity_id, source_doc, type, description) VALUES (?, ?, ?, ?)
ON CONFLICT(entity_id, source_doc)
DO UPDATE SET type = excluded.type, description = excluded.description
"""

# An entity lives while a current document mentions it or a relationship or chunk link still
# references it (the last three also cover rows written before entity_mentions existed).
UNREFERENCED_ENTITY_SQL = """
NOT EXISTS (SELECT 1 FROM entity_mentions m WHERE m.entity_id = entities.id)
AND NOT EXISTS (SELECT 1 FROM relationships r WHERE r.source_id = entities.id)
AND NOT EXISTS (SELECT 1 FROM relationships r WHERE r.target_id = entities.id)
AND NOT EXISTS (SELECT 1 FROM chunk_entities ce WHERE ce.entity_id = entities.id)
"""

# Derive type/description from the documents that currently mention the entity: the most
# common type and the longest description, with order-independent tie-breaks so re-indexing
# an unrelated document never flips them (that would invalidate community summaries).
_REFRESH_ENTITIES_SQL = """
UPDATE entities SET
    type = (
        SELECT m.type FROM entity_mentions m
        WHERE m.entity_id = entities.id AND m.type IS NOT NULL
        GROUP BY m.type ORDER BY COUNT(*) DESC, m.type LIMIT 1
    ),
    description = (
        SELECT m.description FROM entity_mentions m
        WHERE m.entity_id = entities.id AND COALESCE(m.description, '') != ''
        ORDER BY length(m.description) DESC, m.source_doc LIMIT 1
    )
WHERE id IN (SELECT value FROM json_each(?))
  AND EXISTS (SELECT 1 FROM entity_mentions m WHERE m.entity_id = entities.id)
"""

_FILE_ENTITY_IDS_SQL = """
SELECT entity_id FROM entity_mentions WHERE source_doc = :filepath
UNION SELECT ce.entity_id FROM chunk_entities ce
      JOIN chunks c ON c.chunk_id = ce.chunk_id WHERE c.filepath = :filepath
UNION SELECT source_id FROM relationships WHERE source_doc = :filepath
UNION SELECT target_id FROM relationships WHERE source_doc = :filepath
"""


class RecordStoreMixin:
    def upsert_entities(self, entities: list[dict], source_doc: str) -> list[str]:
        """Record entities as mentioned by source_doc. Returns slug IDs in input order.

        Stored type/description are derived from every document currently mentioning the
        entity, so a document's contribution disappears once that document is deleted.
        """
        rows: list[tuple[str, str, str | None, str | None]] = []
        for entity in entities:
            name = str(entity.get("name", "")).strip()
            slug = slugify(name)
            if not slug:
                raise ValueError(f"Entity name {name!r} produces an empty slug")
            rows.append((slug, name, entity.get("type"), entity.get("description")))
        slugs = [slug for slug, _, _, _ in rows]
        with self._write() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)",
                [(slug, name) for slug, name, _, _ in rows],
            )
            conn.executemany(
                _UPSERT_MENTION_SQL,
                [(slug, source_doc, type_, desc) for slug, _, type_, desc in rows],
            )
            conn.execute(_REFRESH_ENTITIES_SQL, (json.dumps(slugs),))
        return slugs

    def upsert_relationships(self, relationships: list[tuple[str, str, str, str]]) -> None:
        """Batch upsert (source_id, target_id, label, source_doc) tuples in one transaction."""
        if not relationships:
            return
        with self._write():
            self.conn.executemany(
                _UPSERT_RELATIONSHIP_SQL,
                relationships,
            )

    def register_chunks(self, chunks: list[tuple[str, str, int]]) -> None:
        """Batch-insert (chunk_id, filepath, chunk_index) tuples in one transaction."""
        with self._write():
            self.conn.executemany(
                "INSERT OR REPLACE INTO chunks (chunk_id, filepath, chunk_index) VALUES (?, ?, ?)",
                chunks,
            )

    def get_chunks_for_file(self, filepath: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT chunk_id FROM chunks WHERE filepath = ? ORDER BY chunk_index", (filepath,)
        ).fetchall()
        return [row["chunk_id"] for row in rows]

    def link_chunks(self, pairs: list[tuple[str, str]]) -> None:
        """Insert (chunk_id, entity_id) link rows, ignoring duplicates."""
        if not pairs:
            return
        with self._write():
            self.conn.executemany(
                "INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id) VALUES (?, ?)",
                pairs,
            )

    def get_entities_by_chunk_ids(self, chunk_ids: list[str]) -> list[str]:
        """Return distinct entity_ids linked to the given chunks, in chunk order (then by id)."""
        rows = self.conn.execute(
            "SELECT ce.entity_id FROM json_each(?) AS j "
            "JOIN chunk_entities ce ON ce.chunk_id = j.value ORDER BY j.key, ce.entity_id",
            (json.dumps(chunk_ids),),
        ).fetchall()
        return list(dict.fromkeys(row["entity_id"] for row in rows))

    def delete_file_data(self, filepath: str) -> None:
        """Remove a file's chunks, graph data, and fingerprint (delete its vectors first).

        Dropping the fingerprint with the data means the file is re-indexed by the next run
        unless the caller records a new one — even if its content reverts to the old hash.
        """
        with self._write() as conn:
            self._delete_file_rows(conn, filepath)

    def _delete_file_rows(self, conn: sqlite3.Connection, filepath: str) -> None:
        touched = json.dumps(
            [row[0] for row in conn.execute(_FILE_ENTITY_IDS_SQL, {"filepath": filepath})]
        )
        conn.execute(
            "DELETE FROM chunk_entities WHERE chunk_id IN "
            "(SELECT chunk_id FROM chunks WHERE filepath = ?)",
            (filepath,),
        )
        conn.execute("DELETE FROM chunks WHERE filepath = ?", (filepath,))
        conn.execute("DELETE FROM relationships WHERE source_doc = ?", (filepath,))
        conn.execute("DELETE FROM entity_mentions WHERE source_doc = ?", (filepath,))
        # Only entities this file touched can have become unreferenced.
        conn.execute(
            "DELETE FROM entities WHERE id IN (SELECT value FROM json_each(?)) "
            f"AND {UNREFERENCED_ENTITY_SQL}",
            (touched,),
        )
        conn.execute(_REFRESH_ENTITIES_SQL, (touched,))
        conn.execute("DELETE FROM fingerprints WHERE filepath = ?", (filepath,))

    def purge_file(self, filepath: str) -> None:
        """Remove every trace of a file that no longer exists, including its cached extractions.

        Call after deleting its vectors (get_chunks_for_file) from Qdrant.
        """
        with self._write() as conn:
            self._delete_file_rows(conn, filepath)
            conn.execute("DELETE FROM extraction_cache WHERE filepath = ?", (filepath,))

    def cache_extraction(
        self, filepath: str, batch_index: int, prompt_sha256: str, result_json: str
    ) -> None:
        with self._write():
            self.conn.execute(
                "INSERT OR REPLACE INTO extraction_cache "
                "(filepath, batch_index, prompt_sha256, result) VALUES (?, ?, ?, ?)",
                (filepath, batch_index, prompt_sha256, result_json),
            )

    def get_cached_extraction(
        self, filepath: str, batch_index: int, prompt_sha256: str
    ) -> str | None:
        """Return the cached response only if it was produced for this exact request."""
        row = self.conn.execute(
            "SELECT result FROM extraction_cache "
            "WHERE filepath = ? AND batch_index = ? AND prompt_sha256 = ?",
            (filepath, batch_index, prompt_sha256),
        ).fetchone()
        return row["result"] if row else None

    def get_hash(self, filepath: str) -> str | None:
        row = self.conn.execute(
            "SELECT sha256 FROM fingerprints WHERE filepath = ?", (filepath,)
        ).fetchone()
        return row["sha256"] if row else None

    def upsert_hash(self, filepath: str, sha256: str) -> None:
        with self._write():
            self.conn.execute(
                """
                INSERT INTO fingerprints (filepath, sha256, updated_at)
                VALUES (?, ?, strftime('%s', 'now'))
                ON CONFLICT(filepath)
                DO UPDATE SET sha256 = excluded.sha256, updated_at = strftime('%s', 'now')
                """,
                (filepath, sha256),
            )

    def list_all_paths(self) -> list[str]:
        """Return every file with stored data — including never-fingerprinted ones whose
        indexing failed partway, so their leftovers are still cleaned up once deleted."""
        rows = self.conn.execute(
            "SELECT filepath FROM fingerprints UNION SELECT filepath FROM chunks "
            "UNION SELECT filepath FROM extraction_cache"
        ).fetchall()
        return [row["filepath"] for row in rows]
