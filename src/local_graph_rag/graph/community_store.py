"""NetworkX graph and community-summary store methods."""

import json
import logging
import sqlite3
from typing import Any

import networkx as nx

try:
    import community as community_louvain
except ImportError:
    community_louvain = None  # type: ignore[assignment]

from local_graph_rag.settings import ENTITY_NEIGHBORHOOD_HOPS

logger = logging.getLogger(__name__)

_COMMUNITY_FIELDS = "id, summary, entity_ids, member_hash, embedding"
_ACTIVE_COMMUNITY_IDS_SQL = "SELECT DISTINCT community FROM entities WHERE community IS NOT NULL"


class CommunityStoreMixin:
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
        """Run Louvain community detection and write community IDs back to entities."""
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
        """Return entity rows and relationship rows within `hops` of entity_id."""
        if graph is None:
            graph = self.build_networkx_graph()
        if entity_id not in graph:
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
        """Return {entity_id: neighborhood} for each id, building the graph once."""
        if not entity_ids:
            return {}
        graph = self.build_networkx_graph()
        return {
            entity_id: self.get_entity_neighborhood(entity_id, hops, graph=graph)
            for entity_id in entity_ids
        }

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
        """Insert or replace a community summary row."""
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
            f"SELECT {_COMMUNITY_FIELDS} FROM communities WHERE id = ?",
            (community_id,),
        ).fetchone()
        return self._deserialize_community_row(row) if row else None

    def get_communities(self) -> list[dict]:
        """Return all community rows (id, summary, entity_ids, member_hash, embedding)."""
        rows = self._conn.execute(f"SELECT {_COMMUNITY_FIELDS} FROM communities").fetchall()
        return [self._deserialize_community_row(row) for row in rows]

    def get_active_community_ids(self) -> set[int]:
        """Return distinct community IDs currently assigned to entities."""
        rows = self._conn.execute(_ACTIVE_COMMUNITY_IDS_SQL).fetchall()
        return {row["community"] for row in rows}

    def delete_stale_communities(self) -> None:
        """Remove community rows whose id is no longer assigned to any entity."""
        with self._write():
            self._conn.execute(
                f"DELETE FROM communities WHERE id NOT IN ({_ACTIVE_COMMUNITY_IDS_SQL})"
            )
