"""
SQLite-backed user and session store for web UI authentication.

Stores bcrypt password hashes and opaque session tokens in data/users.sqlite3,
which is persisted via the graph-data Docker volume.
"""

import hashlib
import secrets
import time

from local_graph_rag.common.sqlite_store import SqliteStore
from local_graph_rag.settings import DATA_DIR

DB_PATH = DATA_DIR / "users.sqlite3"
_store = SqliteStore(DB_PATH)


def _token_hash(token: str) -> str:
    """Sessions are stored by hash, so a leaked database can't be replayed as live tokens."""
    return hashlib.sha256(token.encode()).hexdigest()


def init_db() -> None:
    conn = _store.conn
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,  -- sha256 of the session token; raw tokens never stored
                username TEXT NOT NULL,
                expires_at REAL NOT NULL
            )
        """)


def get_hash(username: str) -> str | None:
    conn = _store.conn
    with conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username=?", (username,)
        ).fetchone()
    return row[0] if row else None


def upsert_user(username: str, password_hash: str) -> None:
    conn = _store.conn
    with conn:
        conn.execute(
            """
            INSERT INTO users(username, password_hash, created_at)
            VALUES(?, ?, strftime('%s','now'))
            ON CONFLICT(username) DO UPDATE SET
                password_hash = excluded.password_hash
            """,
            (username, password_hash),
        )
        conn.execute("DELETE FROM sessions WHERE username=?", (username,))


def delete_user(username: str) -> None:
    conn = _store.conn
    with conn:
        conn.execute("DELETE FROM users WHERE username=?", (username,))
        conn.execute("DELETE FROM sessions WHERE username=?", (username,))


def list_users() -> list[str]:
    conn = _store.conn
    with conn:
        rows = conn.execute("SELECT username FROM users ORDER BY username").fetchall()
    return [row[0] for row in rows]


def create_session(username: str, expiry_seconds: int) -> str:
    """Create a session and return its token."""
    token = secrets.token_hex(32)
    expires_at = time.time() + expiry_seconds
    conn = _store.conn
    with conn:
        conn.execute(
            "INSERT INTO sessions(token, username, expires_at) VALUES(?, ?, ?)",
            (_token_hash(token), username, expires_at),
        )
    return token


def validate_session(token: str) -> str | None:
    conn = _store.conn
    with conn:
        row = conn.execute(
            """
            SELECT sessions.username
            FROM sessions
            INNER JOIN users ON users.username = sessions.username
            WHERE sessions.token=? AND sessions.expires_at > ?
            """,
            (_token_hash(token), time.time()),
        ).fetchone()
    return row[0] if row else None


def delete_session(token: str) -> None:
    conn = _store.conn
    with conn:
        conn.execute("DELETE FROM sessions WHERE token=?", (_token_hash(token),))


def purge_expired_sessions() -> None:
    conn = _store.conn
    with conn:
        conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (time.time(),))
