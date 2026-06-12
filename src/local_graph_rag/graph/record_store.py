"""Entity, relationship, chunk, cache, and fingerprint store methods."""

from local_graph_rag.graph.store_utils import slugify

_UPSERT_RELATIONSHIP_SQL = """
INSERT INTO relationships (source_id, target_id, label, weight, source_doc)
VALUES (?, ?, ?, 1.0, ?)
ON CONFLICT(source_id, target_id, label, source_doc)
DO UPDATE SET weight = weight + 1.0
"""


class RecordStoreMixin:
    def _upsert_entity_unlocked(
        self,
        name: str,
        type: str | None = None,
        description: str | None = None,
    ) -> str:
        """Insert or merge an entity. Caller must already hold the write lock."""
        slug = slugify(name)
        if not slug:
            raise ValueError(f"Entity name {name!r} produces an empty slug")
        existing = self._conn.execute(
            "SELECT type, description FROM entities WHERE id = ?", (slug,)
        ).fetchone()
        if existing is None:
            self._conn.execute(
                "INSERT INTO entities (id, name, type, description) VALUES (?, ?, ?, ?)",
                (slug, name.strip(), type, description),
            )
        else:
            new_type = type if (existing["type"] is None and type is not None) else existing["type"]
            merged_desc = max(str(existing["description"] or ""), str(description or ""), key=len)
            self._conn.execute(
                "UPDATE entities SET type = ?, description = ? WHERE id = ?",
                (new_type, merged_desc or None, slug),
            )
        return slug

    def upsert_entity(
        self,
        name: str,
        type: str | None = None,
        description: str | None = None,
    ) -> str:
        """Insert or merge an entity. Returns the entity's slug id."""
        with self._write():
            return self._upsert_entity_unlocked(name, type, description)

    def upsert_entities(self, entities: list[dict]) -> list[str]:
        """Batch upsert entities in one transaction. Returns slug IDs in input order."""
        slugs: list[str] = []
        with self._write():
            for e in entities:
                slugs.append(
                    self._upsert_entity_unlocked(
                        str(e.get("name", "")),
                        e.get("type"),
                        e.get("description"),
                    )
                )
        return slugs

    def upsert_relationship(
        self, source_id: str, target_id: str, label: str, source_doc: str
    ) -> None:
        """Insert a relationship or increment its weight if it already exists."""
        with self._write():
            self._conn.execute(
                _UPSERT_RELATIONSHIP_SQL,
                (source_id, target_id, label, source_doc),
            )

    def upsert_relationships(self, relationships: list[tuple[str, str, str, str]]) -> None:
        """Batch upsert (source_id, target_id, label, source_doc) tuples in one transaction."""
        if not relationships:
            return
        with self._write():
            self._conn.executemany(
                _UPSERT_RELATIONSHIP_SQL,
                relationships,
            )

    def register_chunks(self, chunks: list[tuple[str, str, int]]) -> None:
        """Batch-insert (chunk_id, filepath, chunk_index) tuples in one transaction."""
        with self._write():
            self._conn.executemany(
                "INSERT OR REPLACE INTO chunks (chunk_id, filepath, chunk_index) VALUES (?, ?, ?)",
                chunks,
            )

    def get_chunks_for_file(self, filepath: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE filepath = ? ORDER BY chunk_index", (filepath,)
        ).fetchall()
        return [row["chunk_id"] for row in rows]

    def link_chunks(self, pairs: list[tuple[str, str]]) -> None:
        """Insert (chunk_id, entity_id) link rows, ignoring duplicates."""
        if not pairs:
            return
        with self._write():
            self._conn.executemany(
                "INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id) VALUES (?, ?)",
                pairs,
            )

    def get_entities_by_chunk_ids(self, chunk_ids: list[str]) -> list[str]:
        """Return distinct entity_ids linked to any of the given chunk_ids."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT DISTINCT entity_id FROM chunk_entities WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return [row["entity_id"] for row in rows]

    def delete_file_data(self, filepath: str) -> list[str]:
        """Remove all data for a file. Returns prior chunk IDs for Qdrant deletion."""
        with self._write():
            prior_ids = [
                row["chunk_id"]
                for row in self._conn.execute(
                    "SELECT chunk_id FROM chunks WHERE filepath = ?", (filepath,)
                ).fetchall()
            ]
            self._conn.execute(
                "DELETE FROM chunk_entities WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE filepath = ?)",
                (filepath,),
            )
            self._conn.execute("DELETE FROM chunks WHERE filepath = ?", (filepath,))
            self._conn.execute("DELETE FROM relationships WHERE source_doc = ?", (filepath,))
            self._conn.execute(
                "DELETE FROM entities WHERE id NOT IN ("
                "  SELECT source_id FROM relationships"
                "  UNION"
                "  SELECT target_id FROM relationships"
                "  UNION"
                "  SELECT entity_id FROM chunk_entities"
                ")"
            )
        return prior_ids

    def cache_extraction(self, filepath: str, batch_index: int, result_json: str) -> None:
        with self._write():
            self._conn.execute(
                "INSERT OR REPLACE INTO extraction_cache (filepath, batch_index, result) "
                "VALUES (?, ?, ?)",
                (filepath, batch_index, result_json),
            )

    def get_cached_extractions(self, filepath: str) -> dict[int, str]:
        rows = self._conn.execute(
            "SELECT batch_index, result FROM extraction_cache WHERE filepath = ?", (filepath,)
        ).fetchall()
        return {row["batch_index"]: row["result"] for row in rows}

    def clear_extraction_cache(self, filepath: str) -> None:
        with self._write():
            self._conn.execute("DELETE FROM extraction_cache WHERE filepath = ?", (filepath,))

    def get_hash(self, filepath: str) -> str | None:
        row = self._conn.execute(
            "SELECT sha256 FROM fingerprints WHERE filepath = ?", (filepath,)
        ).fetchone()
        return row["sha256"] if row else None

    def upsert_hash(self, filepath: str, sha256: str) -> None:
        with self._write():
            self._conn.execute(
                """
                INSERT INTO fingerprints (filepath, sha256, updated_at)
                VALUES (?, ?, strftime('%s', 'now'))
                ON CONFLICT(filepath)
                DO UPDATE SET sha256 = excluded.sha256, updated_at = strftime('%s', 'now')
                """,
                (filepath, sha256),
            )

    def delete_hash(self, filepath: str) -> None:
        with self._write():
            self._conn.execute("DELETE FROM fingerprints WHERE filepath = ?", (filepath,))

    def list_all_paths(self) -> list[str]:
        rows = self._conn.execute("SELECT filepath FROM fingerprints").fetchall()
        return [row["filepath"] for row in rows]
