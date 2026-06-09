"""SQLite + NetworkX graph store for entities, relationships, and chunk metadata."""

import json
import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import networkx as nx

try:
    import community as community_louvain
except ImportError:
    community_louvain = None  # type: ignore[assignment]

from local_graph_rag.settings import ENTITY_NEIGHBORHOOD_HOPS, SQLITE_PATH

logger = logging.getLogger(__name__)

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


def slugify(name: str) -> str:
    """Return a normalised ASCII slug for an entity name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower().strip()).strip("_")


class GraphStore:
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

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def upsert_entity(
        self,
        name: str,
        type: str | None = None,
        description: str | None = None,
    ) -> str:
        """Insert or merge an entity. Returns the entity's slug id."""
        slug = slugify(name)
        if not slug:
            raise ValueError(f"Entity name {name!r} produces an empty slug")
        with self._write():
            existing = self._conn.execute(
                "SELECT type, description FROM entities WHERE id = ?", (slug,)
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO entities (id, name, type, description) VALUES (?, ?, ?, ?)",
                    (slug, name.strip(), type, description),
                )
            else:
                new_type = (
                    type if (existing["type"] is None and type is not None) else existing["type"]
                )
                merged_desc = max(
                    str(existing["description"] or ""), str(description or ""), key=len
                )
                self._conn.execute(
                    "UPDATE entities SET type = ?, description = ? WHERE id = ?",
                    (new_type, merged_desc or None, slug),
                )
        return slug

    def upsert_entities(self, entities: list[dict]) -> list[str]:
        """Batch upsert entities in one transaction. Returns slug IDs in input order.

        Duplicates upsert_entity's INSERT/UPDATE body intentionally: threading.Lock
        is non-reentrant, so calling upsert_entity() inside _write() would deadlock.
        """
        slugs: list[str] = []
        with self._write():
            for e in entities:
                slug = slugify(e.get("name", ""))
                if not slug:
                    raise ValueError(f"Entity name {e.get('name')!r} produces an empty slug")
                existing = self._conn.execute(
                    "SELECT type, description FROM entities WHERE id = ?", (slug,)
                ).fetchone()
                if existing is None:
                    self._conn.execute(
                        "INSERT INTO entities (id, name, type, description) VALUES (?, ?, ?, ?)",
                        (slug, str(e.get("name", "")).strip(), e.get("type"), e.get("description")),
                    )
                else:
                    new_type = (
                        e.get("type")
                        if (existing["type"] is None and e.get("type") is not None)
                        else existing["type"]
                    )
                    merged_desc = max(
                        str(existing["description"] or ""),
                        str(e.get("description") or ""),
                        key=len,
                    )
                    self._conn.execute(
                        "UPDATE entities SET type = ?, description = ? WHERE id = ?",
                        (new_type, merged_desc or None, slug),
                    )
                slugs.append(slug)
        return slugs

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def upsert_relationship(
        self, source_id: str, target_id: str, label: str, source_doc: str
    ) -> None:
        """Insert a relationship or increment its weight if it already exists."""
        with self._write():
            self._conn.execute(
                """
                INSERT INTO relationships (source_id, target_id, label, weight, source_doc)
                VALUES (?, ?, ?, 1.0, ?)
                ON CONFLICT(source_id, target_id, label, source_doc)
                DO UPDATE SET weight = weight + 1.0
                """,
                (source_id, target_id, label, source_doc),
            )

    def upsert_relationships(self, relationships: list[tuple[str, str, str, str]]) -> None:
        """Batch upsert (source_id, target_id, label, source_doc) tuples in one transaction."""
        if not relationships:
            return
        with self._write():
            self._conn.executemany(
                """
                INSERT INTO relationships (source_id, target_id, label, weight, source_doc)
                VALUES (?, ?, ?, 1.0, ?)
                ON CONFLICT(source_id, target_id, label, source_doc)
                DO UPDATE SET weight = weight + 1.0
                """,
                relationships,
            )

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Cleanup / re-ingestion
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Extraction cache
    # ------------------------------------------------------------------

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
            self._conn.execute(
                "DELETE FROM extraction_cache WHERE filepath = ?", (filepath,)
            )

    # ------------------------------------------------------------------
    # Fingerprints
    # ------------------------------------------------------------------

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
            self._conn.execute(
                "DELETE FROM fingerprints WHERE filepath = ?", (filepath,)
            )

    def list_all_paths(self) -> list[str]:
        rows = self._conn.execute("SELECT filepath FROM fingerprints").fetchall()
        return [row["filepath"] for row in rows]

    # ------------------------------------------------------------------
    # Graph / community detection
    # ------------------------------------------------------------------

    def build_networkx_graph(self) -> nx.DiGraph:
        """Load all relationships into an in-memory DiGraph."""
        graph: nx.DiGraph = nx.DiGraph()
        rows = self._conn.execute(
            "SELECT source_id, target_id, label, weight FROM relationships"
        ).fetchall()
        for row in rows:
            graph.add_edge(
                row["source_id"],
                row["target_id"],
                label=row["label"],
                weight=row["weight"],
            )
        return graph

    def detect_communities(self) -> None:
        """Run Louvain community detection and write community IDs back to entities.

        Always clears existing assignments first, in the same transaction as any
        new ones, so entities that fall out of the graph — including the
        empty-graph case where no partition is computed at all — don't retain a
        stale community ID from a previous run.
        """
        if community_louvain is None:
            raise RuntimeError(
                "python-louvain is not installed; run `uv add python-louvain`"
            )
        graph = self.build_networkx_graph()
        partition: dict[str, int] = {}
        if len(graph.nodes) == 0:
            logger.warning(
                "detect_communities: graph is empty — clearing all community assignments"
            )
        else:
            partition = community_louvain.best_partition(graph.to_undirected())

        with self._write():
            self._conn.execute("UPDATE entities SET community = NULL")
            if partition:
                self._conn.executemany(
                    "UPDATE entities SET community = ? WHERE id = ?",
                    [(comm_id, entity_id) for entity_id, comm_id in partition.items()],
                )

        if partition:
            logger.info(
                "detect_communities: assigned %d entities to %d communities",
                len(partition),
                len(set(partition.values())),
            )

    def get_entity_neighborhood(
        self,
        entity_id: str,
        hops: int = ENTITY_NEIGHBORHOOD_HOPS,
        *,
        graph: nx.DiGraph | None = None,
    ) -> dict[str, Any]:
        """Return entity rows and relationship rows within `hops` of entity_id.

        Pass a pre-built graph (from `build_networkx_graph`) to avoid rebuilding
        it on every call when looking up neighborhoods for multiple entities.
        """
        if graph is None:
            graph = self.build_networkx_graph()
        if entity_id not in graph:
            # Entity exists but has no relationships — look it up directly.
            row = self._conn.execute(
                "SELECT id, name, type, description, community FROM entities WHERE id = ?",
                (entity_id,),
            ).fetchone()
            if row is None:
                return {"entities": [], "relationships": []}
            return {"entities": [dict(row)], "relationships": []}
        subgraph = nx.ego_graph(graph, entity_id, radius=hops, undirected=True)
        node_ids = list(subgraph.nodes)
        placeholders = ",".join("?" * len(node_ids))
        entity_rows = self._conn.execute(
            f"SELECT id, name, type, description, community FROM entities "
            f"WHERE id IN ({placeholders})",
            node_ids,
        ).fetchall()
        rel_rows = self._conn.execute(
            f"SELECT source_id, target_id, label, weight FROM relationships "
            f"WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})",
            node_ids + node_ids,
        ).fetchall()
        return {
            "entities": [dict(row) for row in entity_rows],
            "relationships": [dict(row) for row in rel_rows],
        }

    def get_entity_neighborhoods(
        self, entity_ids: list[str], hops: int = ENTITY_NEIGHBORHOOD_HOPS
    ) -> dict[str, dict[str, Any]]:
        """Return {entity_id: neighborhood} for each id, building the graph once.

        Prefer this over calling get_entity_neighborhood in a loop — it avoids
        rebuilding the relationships graph from scratch for every entity.
        """
        if not entity_ids:
            return {}
        graph = self.build_networkx_graph()
        return {
            entity_id: self.get_entity_neighborhood(entity_id, hops, graph=graph)
            for entity_id in entity_ids
        }

    # ------------------------------------------------------------------
    # Communities
    # ------------------------------------------------------------------

    def get_entities_for_community(self, community_id: int) -> list[dict]:
        """Return entity rows assigned to a given Louvain community."""
        rows = self._conn.execute(
            "SELECT id, name, type, description FROM entities WHERE community = ?",
            (community_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_relationships_for_community(self, community_id: int) -> list[dict]:
        """Return relationships where both endpoints belong to the given community."""
        rows = self._conn.execute(
            """
            SELECT r.source_id, r.target_id, r.label, r.weight
            FROM relationships r
            JOIN entities s ON r.source_id = s.id
            JOIN entities t ON r.target_id = t.id
            WHERE s.community = ? AND t.community = ?
            """,
            (community_id, community_id),
        ).fetchall()
        return [dict(row) for row in rows]

    def upsert_community(
        self,
        community_id: int,
        summary: str,
        entity_ids: list[str],
        member_hash: str,
        embedding: bytes,
    ) -> None:
        """Insert or replace a community summary row.

        embedding must be np.array(vec, dtype=np.float32).tobytes().
        Deserialize with np.frombuffer(embedding, dtype=np.float32).
        """
        with self._write():
            self._conn.execute(
                "INSERT OR REPLACE INTO communities "
                "(id, summary, entity_ids, member_hash, embedding) VALUES (?, ?, ?, ?, ?)",
                (community_id, summary, json.dumps(entity_ids), member_hash, embedding),
            )

    def _deserialize_community_row(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        d["entity_ids"] = json.loads(d["entity_ids"]) if d["entity_ids"] else []
        return d

    def get_community(self, community_id: int) -> dict | None:
        """Return a single community row, or None if not found."""
        row = self._conn.execute(
            "SELECT id, summary, entity_ids, member_hash, embedding FROM communities WHERE id = ?",
            (community_id,),
        ).fetchone()
        return self._deserialize_community_row(row) if row else None

    def get_communities(self) -> list[dict]:
        """Return all community rows (id, summary, entity_ids, member_hash, embedding)."""
        rows = self._conn.execute(
            "SELECT id, summary, entity_ids, member_hash, embedding FROM communities"
        ).fetchall()
        return [self._deserialize_community_row(row) for row in rows]

    def get_active_community_ids(self) -> set[int]:
        """Return distinct community IDs currently assigned to entities."""
        rows = self._conn.execute(
            "SELECT DISTINCT community FROM entities WHERE community IS NOT NULL"
        ).fetchall()
        return {row["community"] for row in rows}

    def delete_stale_communities(self) -> None:
        """Remove community rows whose id is no longer assigned to any entity."""
        with self._write():
            self._conn.execute(
                "DELETE FROM communities WHERE id NOT IN "
                "(SELECT DISTINCT community FROM entities WHERE community IS NOT NULL)"
            )

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
