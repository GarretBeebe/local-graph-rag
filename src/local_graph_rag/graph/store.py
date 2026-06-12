"""SQLite-backed graph store facade."""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from local_graph_rag.graph.community_store import CommunityStoreMixin
from local_graph_rag.graph.record_store import RecordStoreMixin
from local_graph_rag.graph.store_utils import slugify
from local_graph_rag.settings import SQLITE_PATH

__all__ = ["GraphStore", "slugify"]

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


class GraphStore(RecordStoreMixin, CommunityStoreMixin):
    """SQLite connection owner plus stable facade over focused store mixins."""

    def __init__(self, db_path: Path = SQLITE_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.Lock()
        with self._write():
            self._conn.executescript(_SCHEMA)

    @contextmanager
    def _write(self):
        with self._lock:
            try:
                yield
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        self._conn.close()
