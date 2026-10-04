"""Shared SQLite connection helper for thread-local stores."""

import sqlite3
import threading
from contextlib import suppress
from pathlib import Path

# The api, indexer, and summarizer are separate processes sharing one database file: wait
# out each other's short write transactions instead of failing with "database is locked".
_BUSY_TIMEOUT_SECONDS = 30.0


class SqliteStore:
    """Owns one SQLite connection per thread for a single database file."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._conns: list[tuple[threading.Thread, sqlite3.Connection]] = []
        self._conns_lock = threading.Lock()

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False only so close() can run on another thread; each
            # connection is otherwise used solely by the thread that opened it.
            conn = sqlite3.connect(
                str(self._db_path), timeout=_BUSY_TIMEOUT_SECONDS, check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            with suppress(sqlite3.OperationalError):
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            with self._conns_lock:
                # Tracking connections for close() would otherwise keep those of exited
                # threads (e.g. pruned threadpool workers) open forever.
                live: list[tuple[threading.Thread, sqlite3.Connection]] = []
                for thread, existing in self._conns:
                    if thread.is_alive():
                        live.append((thread, existing))
                    else:
                        existing.close()
                live.append((threading.current_thread(), conn))
                self._conns = live
            self._local.conn = conn
        return self._local.conn

    def close(self) -> None:
        """Close every thread's connection; later access opens fresh ones."""
        with self._conns_lock:
            for _, conn in self._conns:
                conn.close()
            self._conns.clear()
            self._local = threading.local()
