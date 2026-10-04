"""Graph traversal, community detection, and community-summary store methods."""

import json
import logging
from collections import defaultdict
from itertools import zip_longest

import networkx as nx

logger = logging.getLogger(__name__)

_ACTIVE_COMMUNITY_IDS_SQL = "SELECT DISTINCT community FROM entities WHERE community IS NOT NULL"
# Fixed so an unchanged graph always yields the same partition and community ids.
_LOUVAIN_SEED = 42
# Weight a chunk adds between each pair of entities it mentions, split across them
# (0.5 / (n - 1)) so chunks naming many entities don't dominate.
_COOCCURRENCE_WEIGHT = 0.5


def _louvain(graph: nx.Graph) -> list[set[str]]:
    """Seeded Louvain; communities ordered by their smallest member so ids are stable."""
    if graph.number_of_nodes() == 0:
        return []
    communities = nx.community.louvain_communities(graph, weight="weight", seed=_LOUVAIN_SEED)
    return sorted(communities, key=min)


def _cluster(llm_graph: nx.Graph, cooccurrence: dict[tuple[str, str], float]) -> list[set[str]]:
    """Partition entities: Louvain over LLM relationships, then place the rest by co-occurrence.

    An entity with no LLM relationship joins the community it shares the most chunk
    co-occurrence with (ties: lowest community). Attaching repeats until nothing new joins, so
    an entity that only co-occurs with attached entities is placed too; entities with no
    co-occurrence path to any LLM community get Louvain communities of their own. Running
    Louvain over the far denser co-occurrence graph instead reshuffles many communities on
    every edit, and each reshuffled community costs a fresh LLM summary.
    """
    communities = _louvain(llm_graph)
    community_of = {entity: i for i, members in enumerate(communities) for entity in members}
    while True:
        affinity: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for (a, b), weight in cooccurrence.items():
            if a in community_of and b not in community_of:
                affinity[b][community_of[a]] += weight
            elif b in community_of and a not in community_of:
                affinity[a][community_of[b]] += weight
        if not affinity:
            break
        for entity, scores in affinity.items():
            best = min(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            communities[best].add(entity)
            community_of[entity] = best

    residual = nx.Graph()
    residual.add_weighted_edges_from(
        (a, b, w)
        for (a, b), w in cooccurrence.items()
        if a not in community_of and b not in community_of
    )
    return communities + _louvain(residual)


class CommunityStoreMixin:
    def build_networkx_graph(self) -> nx.Graph:
        """Load relationships as an undirected graph with one edge per entity pair.

        Edge weight is the sum across labels, directions, and source documents. Rows are
        added in a fixed order because the seeded Louvain partition depends on it.
        """
        rows = self.conn.execute(
            "SELECT MIN(source_id, target_id) AS a, MAX(source_id, target_id) AS b, "
            "SUM(weight) AS weight FROM relationships GROUP BY a, b ORDER BY a, b"
        ).fetchall()
        graph = nx.Graph()
        graph.add_weighted_edges_from((row["a"], row["b"], row["weight"]) for row in rows)
        return graph

    def _cooccurrence_weights(self) -> dict[tuple[str, str], float]:
        """Return {(a, b): weight} (a < b) for entity pairs linked to the same chunks."""
        rows = self.conn.execute(
            """
            SELECT a.entity_id AS a, b.entity_id AS b, SUM(:weight / (n.n - 1)) AS weight
            FROM chunk_entities a
            JOIN chunk_entities b ON b.chunk_id = a.chunk_id AND a.entity_id < b.entity_id
            JOIN (SELECT chunk_id, COUNT(*) AS n FROM chunk_entities GROUP BY chunk_id) n
              ON n.chunk_id = a.chunk_id
            GROUP BY a.entity_id, b.entity_id
            ORDER BY a.entity_id, b.entity_id
            """,
            {"weight": float(_COOCCURRENCE_WEIGHT)},  # an int here would divide as integers
        )
        return {(row["a"], row["b"]): row["weight"] for row in rows}

    def detect_communities(self) -> None:
        """Assign community ids (see _cluster) and write them back to entities."""
        communities = _cluster(self.build_networkx_graph(), self._cooccurrence_weights())
        assignments = [
            (community_id, entity_id)
            for community_id, members in enumerate(communities)
            for entity_id in members
        ]
        with self._write():
            self.conn.execute("UPDATE entities SET community = NULL")
            self.conn.executemany("UPDATE entities SET community = ? WHERE id = ?", assignments)

        if assignments:
            logger.info(
                "detect_communities: assigned %d entities to %d communities",
                len(assignments),
                len(communities),
            )
        else:
            logger.warning(
                "detect_communities: graph is empty — clearing all community assignments"
            )

    def expand_neighborhood(
        self,
        seed_ids: list[str],
        hops: int,
        *,
        max_entities: int,
        max_relationships: int,
    ) -> tuple[list[dict], list[dict]]:
        """Return (entities, relationships) around seed_ids, most relevant first.

        Breadth-first with one indexed query per hop, so cost tracks the result size rather
        than the graph size. Seeds keep their given (relevance) order, and each hop takes
        neighbors round-robin across their discovering parents — heaviest edges first — so a
        single hub seed cannot spend the whole budget. Relationships among the selected
        entities come first, then edges linking them to other entities — the latter still
        carry facts when the seeds alone fill max_entities — heaviest first within each group.
        """
        selected = list(dict.fromkeys(seed_ids))[:max_entities]
        chosen = set(selected)
        frontier = selected
        for _ in range(hops):
            if not frontier or len(selected) >= max_entities:
                break
            frontier = self._next_hop(frontier, chosen, max_entities - len(selected))
            selected.extend(frontier)

        ids = json.dumps(selected)
        rows_by_id = {
            row["id"]: dict(row)
            for row in self.conn.execute(
                "SELECT id, name, type, description, community FROM entities "
                "WHERE id IN (SELECT value FROM json_each(?))",
                (ids,),
            )
        }
        relationships = self.conn.execute(
            "SELECT source_id, target_id, label, SUM(weight) AS weight FROM relationships "
            "WHERE source_id IN (SELECT value FROM json_each(:ids)) "
            "OR target_id IN (SELECT value FROM json_each(:ids)) "
            "GROUP BY source_id, target_id, label "
            "ORDER BY (source_id IN (SELECT value FROM json_each(:ids)) "
            "AND target_id IN (SELECT value FROM json_each(:ids))) DESC, "
            "weight DESC, source_id, target_id, label LIMIT :limit",
            {"ids": ids, "limit": max_relationships},
        ).fetchall()
        entities = [rows_by_id[entity_id] for entity_id in selected if entity_id in rows_by_id]
        return entities, [dict(row) for row in relationships]

    def _next_hop(self, frontier: list[str], chosen: set[str], budget: int) -> list[str]:
        """Pick up to `budget` unchosen neighbors of frontier and add them to `chosen`."""
        # Each edge is seen from both ends; per parent, heaviest neighbors first, ties by id.
        rows = self.conn.execute(
            "SELECT parent, neighbor FROM ("
            " SELECT source_id AS parent, target_id AS neighbor, weight FROM relationships"
            " WHERE source_id IN (SELECT value FROM json_each(:ids))"
            " UNION ALL"
            " SELECT target_id, source_id, weight FROM relationships"
            " WHERE target_id IN (SELECT value FROM json_each(:ids))"
            ") GROUP BY parent, neighbor ORDER BY SUM(weight) DESC, neighbor",
            {"ids": json.dumps(frontier)},
        ).fetchall()
        ranked: dict[str, list[str]] = {parent: [] for parent in frontier}
        for row in rows:
            if row["neighbor"] not in chosen:
                ranked[row["parent"]].append(row["neighbor"])
        picked: list[str] = []
        for tier in zip_longest(*ranked.values()):
            for neighbor in tier:
                if neighbor is None or neighbor in chosen:
                    continue
                chosen.add(neighbor)
                picked.append(neighbor)
                if len(picked) == budget:
                    return picked
        return picked

    def get_entities_for_community(self, community_id: int) -> list[dict]:
        """Return entity rows assigned to a given Louvain community."""
        rows = self.conn.execute(
            "SELECT id, name, type, description FROM entities WHERE community = ?",
            (community_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_relationships_for_community(self, community_id: int) -> list[dict]:
        """Return relationships within the community, one row per (source, target, label)."""
        rows = self.conn.execute(
            """
            SELECT r.source_id, r.target_id, r.label, SUM(r.weight) AS weight
            FROM relationships r
            JOIN entities s ON r.source_id = s.id
            JOIN entities t ON r.target_id = t.id
            WHERE s.community = ? AND t.community = ?
            GROUP BY r.source_id, r.target_id, r.label
            ORDER BY r.source_id, r.target_id, r.label
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
            self.conn.execute(
                "INSERT OR REPLACE INTO communities "
                "(id, summary, entity_ids, member_hash, embedding) VALUES (?, ?, ?, ?, ?)",
                (community_id, summary, json.dumps(entity_ids), member_hash, embedding),
            )

    def get_communities(self) -> list[dict]:
        """Return all community rows (id, summary, entity_ids, member_hash, embedding)."""
        rows = self.conn.execute(
            "SELECT id, summary, entity_ids, member_hash, embedding FROM communities"
        ).fetchall()
        return [
            dict(row, entity_ids=json.loads(row["entity_ids"]) if row["entity_ids"] else [])
            for row in rows
        ]

    def has_community_summaries(self) -> bool:
        """Return True if any community has both a summary and an embedding."""
        row = self.conn.execute(
            "SELECT EXISTS (SELECT 1 FROM communities "
            "WHERE embedding IS NOT NULL AND summary != '')"
        ).fetchone()
        return bool(row[0])

    def get_active_community_ids(self) -> set[int]:
        """Return distinct community IDs currently assigned to entities."""
        rows = self.conn.execute(_ACTIVE_COMMUNITY_IDS_SQL).fetchall()
        return {row["community"] for row in rows}

    def delete_stale_communities(self) -> None:
        """Remove community rows whose id is no longer assigned to any entity."""
        with self._write():
            self.conn.execute(
                f"DELETE FROM communities WHERE id NOT IN ({_ACTIVE_COMMUNITY_IDS_SQL})"
            )
