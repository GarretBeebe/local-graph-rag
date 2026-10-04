"""Community detection and LLM summarization for Graph RAG global retrieval."""

import hashlib
import logging
import sys
from collections import Counter

import numpy as np

import local_graph_rag.rag.ollama_client as ollama_client
from local_graph_rag.common.logging import configure_cli_logging
from local_graph_rag.graph.store import GraphStore
from local_graph_rag.rag.embed import embed
from local_graph_rag.settings import SUMMARIZE_MODEL

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = """\
You are summarizing a knowledge graph community for retrieval-augmented generation.

{entity_block}

{relationship_block}
Write a concise summary (3-5 sentences) of what this community is about, \
the key entities and their roles, and how they relate to each other.

Summary:"""

# Large communities would otherwise outgrow the 120 s generation budget on CPU: show the
# best-connected entities, the heaviest relationships, and clipped descriptions.
_MAX_SUMMARY_ENTITIES = 40
_MAX_SUMMARY_RELATIONSHIPS = 60
_MAX_DESCRIPTION_CHARS = 300


def _compute_member_hash(entities: list[dict], relationships: list[dict]) -> str:
    """Return SHA-256 over entity membership/metadata and relationship topology.

    Order-independent: each side is sorted before joining, so the same community
    state always yields the same hash regardless of query result ordering.
    Covers more than membership — entity type/description and relationship
    endpoints/labels are included so a content-only change (e.g. a richer
    description from re-extraction) invalidates the cached summary too.
    """
    entity_lines = sorted(
        f"{e['id']}\x1f{e.get('type') or ''}\x1f{e.get('description') or ''}"
        for e in entities
    )
    relationship_lines = sorted(
        f"{r['source_id']}\x1f{r['target_id']}\x1f{r['label']}"
        for r in relationships
    )
    joined = "\n".join(entity_lines + ["---"] + relationship_lines)
    return hashlib.sha256(joined.encode()).hexdigest()


def _build_summary_prompt(entities: list[dict], relationships: list[dict]) -> str:
    names = {e["id"]: e["name"] for e in entities}
    degree = Counter(r["source_id"] for r in relationships)
    degree.update(r["target_id"] for r in relationships)
    shown = sorted(entities, key=lambda e: (-degree[e["id"]], e["id"]))[:_MAX_SUMMARY_ENTITIES]
    entity_lines = [
        f"- {e['name']} ({e.get('type') or 'unknown'}): "
        f"{(e.get('description') or '')[:_MAX_DESCRIPTION_CHARS]}"
        for e in shown
    ]
    if len(entities) > len(shown):
        entity_lines.append(f"- ... and {len(entities) - len(shown)} more")
    entity_block = "Entities:\n" + "\n".join(entity_lines)

    heaviest = sorted(
        relationships,
        key=lambda r: (-r["weight"], r["source_id"], r["target_id"], r["label"]),
    )[:_MAX_SUMMARY_RELATIONSHIPS]
    if heaviest:
        rel_lines = "\n".join(
            f"- {names.get(r['source_id'], r['source_id'])} --[{r['label']}]--> "
            f"{names.get(r['target_id'], r['target_id'])}"
            for r in heaviest
        )
        relationship_block = f"Relationships:\n{rel_lines}\n"
    else:
        relationship_block = ""

    return _SUMMARIZE_PROMPT.format(
        entity_block=entity_block,
        relationship_block=relationship_block,
    )


def _summaries_by_hash(store: GraphStore) -> dict[str, dict]:
    return {c["member_hash"]: c for c in store.get_communities() if c["member_hash"]}


def summarize_community(
    community_id: int,
    store: GraphStore,
    *,
    previous: dict[str, dict] | None = None,
    force: bool = False,
) -> bool:
    """Summarize one community. Returns True if generated, False if skipped or reused.

    Skips the LLM when a stored community has the same member_hash under any id —
    `previous` maps member_hash to community row (read from the store when omitted) — and
    copies that summary to this id if Louvain renumbered the community. force=True always
    regenerates.
    """
    entities = store.get_entities_for_community(community_id)
    if not entities:
        logger.debug("Community %d has no entities — skipping", community_id)
        return False

    entity_ids = [e["id"] for e in entities]
    relationships = store.get_relationships_for_community(community_id)
    new_hash = _compute_member_hash(entities, relationships)

    if not force:
        if previous is None:
            previous = _summaries_by_hash(store)
        prior = previous.get(new_hash)
        if prior is not None:
            if prior["id"] != community_id:
                store.upsert_community(
                    community_id, prior["summary"], entity_ids, new_hash, prior["embedding"]
                )
            logger.debug("Community %d unchanged — reusing summary", community_id)
            return False

    prompt = _build_summary_prompt(entities, relationships)
    summary = ollama_client.generate(prompt, SUMMARIZE_MODEL).strip()
    embedding_blob = np.array(embed(summary), dtype=np.float32).tobytes()
    store.upsert_community(community_id, summary, entity_ids, new_hash, embedding_blob)
    logger.info(
        "Community %d summarized: %d entities, %d relationships",
        community_id,
        len(entities),
        len(relationships),
    )
    return True


def summarize_all_communities(
    store: GraphStore, *, force: bool = False
) -> dict[str, int]:
    """Run Louvain detection, summarize every community, then drop stale summary rows.

    Existing summaries are matched by member_hash across community ids, so a renumbered but
    unchanged community is reused instead of re-summarized. Stale rows are deleted last so
    current summaries keep serving global retrieval during a long run.

    Returns counts: {summarized, skipped, failed}.
    """
    previous = None if force else _summaries_by_hash(store)
    store.detect_communities()

    active_ids = store.get_active_community_ids()
    logger.info("Detected %d active communities", len(active_ids))

    counts: dict[str, int] = {"summarized": 0, "skipped": 0, "failed": 0}
    for community_id in sorted(active_ids):
        try:
            generated = summarize_community(
                community_id, store, previous=previous, force=force
            )
            counts["summarized" if generated else "skipped"] += 1
        except Exception:
            logger.exception("Failed to summarize community %d", community_id)
            counts["failed"] += 1

    store.delete_stale_communities()
    return counts


def main() -> None:
    configure_cli_logging()
    force = "--force" in sys.argv
    store = GraphStore()
    try:
        counts = summarize_all_communities(store, force=force)
        print(
            f"Done — summarized: {counts['summarized']}, "
            f"skipped: {counts['skipped']}, "
            f"failed: {counts['failed']}"
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
