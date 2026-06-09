"""Shared SQLite connection helper for thread-local stores."""

import sqlite3
import threading
from contextlib import suppress
from pathlib import Path


class SqliteStore:
    """Owns a thread-local SQLite connection for a single database file."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._local = threading.local()

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            with suppress(sqlite3.OperationalError):
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn
