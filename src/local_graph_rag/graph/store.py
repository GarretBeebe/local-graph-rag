"""SQLite-backed graph store facade."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from local_graph_rag.common.sqlite_store import SqliteStore
from local_graph_rag.graph.community_store import CommunityStoreMixin
from local_graph_rag.graph.record_store import UNREFERENCED_ENTITY_SQL, RecordStoreMixin
from local_graph_rag.graph.store_utils import slugify
from local_graph_rag.settings import SQLITE_PATH

__all__ = ["GraphStore", "slugify"]

# Version-0 baseline. Never edit it: existing databases were created from it, so every
# schema change belongs in _MIGRATIONS, which old and new databases both run.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT,
    description TEXT,
    community   INTEGER,
    embedding   BLOB
);
CREATE TABLE IF NOT EXISTS relationships (
    source_id   TEXT REFERENCES entities(id),
    target_id   TEXT REFERENCES entities(id),
    label       TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    source_doc  TEXT,
    PRIMARY KEY (source_id, target_id, label, source_doc)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    filepath    TEXT NOT NULL,
    chunk_index INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_id    TEXT REFERENCES chunks(chunk_id),
    entity_id   TEXT REFERENCES entities(id),
    PRIMARY KEY (chunk_id, entity_id)
);
CREATE TABLE IF NOT EXISTS communities (
    id          INTEGER PRIMARY KEY,
    summary     TEXT NOT NULL,
    entity_ids  TEXT,
    member_hash TEXT,
    embedding   BLOB
);
CREATE TABLE IF NOT EXISTS extraction_cache (
    filepath    TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    result      TEXT NOT NULL,
    PRIMARY KEY (filepath, batch_index)
);
CREATE TABLE IF NOT EXISTS fingerprints (
    filepath   TEXT PRIMARY KEY,
    sha256     TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""

# Entry N upgrades a database from user_version N to N + 1.
_MIGRATIONS: list[list[str]] = [
    # v1: indexes for per-file deletes, reverse-edge traversal, entity reference checks, and
    # per-community reads — each of these lookups was a full table scan.
    [
        "CREATE INDEX IF NOT EXISTS idx_chunks_filepath ON chunks(filepath, chunk_index)",
        "CREATE INDEX IF NOT EXISTS idx_relationships_source_doc ON relationships(source_doc)",
        "CREATE INDEX IF NOT EXISTS idx_relationships_target_id ON relationships(target_id)",
        "CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity_id ON chunk_entities(entity_id)",
        "CREATE INDEX IF NOT EXISTS idx_entities_community ON entities(community)",
    ],
    # v2: per-document entity mentions, so editing or deleting a document retracts what it
    # contributed. Backfill one approximate mention (the entity's current merged type and
    # description) per referencing document, then drop entities nothing references.
    [
        """
        CREATE TABLE IF NOT EXISTS entity_mentions (
            entity_id   TEXT NOT NULL REFERENCES entities(id),
            source_doc  TEXT NOT NULL,
            type        TEXT,
            description TEXT,
            PRIMARY KEY (entity_id, source_doc)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_entity_mentions_source_doc ON entity_mentions(source_doc)",
        """
        INSERT OR IGNORE INTO entity_mentions (entity_id, source_doc, type, description)
        SELECT e.id, refs.source_doc, e.type, e.description
        FROM entities e JOIN (
            SELECT source_id AS entity_id, source_doc FROM relationships
            WHERE source_doc IS NOT NULL
            UNION SELECT target_id, source_doc FROM relationships WHERE source_doc IS NOT NULL
            UNION SELECT ce.entity_id, c.filepath FROM chunk_entities ce
                  JOIN chunks c ON c.chunk_id = ce.chunk_id
        ) refs ON refs.entity_id = e.id
        """,
        f"DELETE FROM entities WHERE {UNREFERENCED_ENTITY_SQL}",
    ],
    # v3: key cached extraction responses by a hash of their exact request. Old rows have no
    # hash and can never be safely replayed, so they are dropped rather than kept as misses.
    [
        "DROP TABLE extraction_cache",
        """
        CREATE TABLE extraction_cache (
            filepath      TEXT NOT NULL,
            batch_index   INTEGER NOT NULL,
            prompt_sha256 TEXT NOT NULL,
            result        TEXT NOT NULL,
            PRIMARY KEY (filepath, batch_index)
        )
        """,
    ],
]


class GraphStore(SqliteStore, RecordStoreMixin, CommunityStoreMixin):
    """Thread-local SQLite connections plus a stable facade over focused store mixins."""

    def __init__(self, db_path: Path = SQLITE_PATH) -> None:
        super().__init__(db_path)
        self.conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        conn = self.conn
        if conn.execute("PRAGMA user_version").fetchone()[0] >= len(_MIGRATIONS):
            return
        with self._write():
            # Re-read under the write lock: another process may have migrated meanwhile.
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            for target, statements in enumerate(_MIGRATIONS[version:], start=version + 1):
                for statement in statements:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {target}")

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """Run one write transaction on this thread's connection, rolling back on error."""
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            # Inside the try: a COMMIT that fails leaves the transaction open, and every later
            # write on this thread would then fail to BEGIN.
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
