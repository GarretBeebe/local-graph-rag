"""Community detection and LLM summarization for Graph RAG global retrieval."""

import hashlib
import logging
import sys

import numpy as np

import api.ollama_client as ollama_client
from api.embed import embed
from graph.store import GraphStore
from settings import SUMMARIZE_MODEL

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = """\
You are summarizing a knowledge graph community for retrieval-augmented generation.

{entity_block}

{relationship_block}
Write a concise summary (3-5 sentences) of what this community is about, \
the key entities and their roles, and how they relate to each other.

Summary:"""


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
    entity_lines = "\n".join(
        f"- {e['name']} ({e.get('type') or 'unknown'}): {e.get('description') or ''}"
        for e in entities
    )
    entity_block = f"Entities:\n{entity_lines}"

    if relationships:
        rel_lines = "\n".join(
            f"- {r['source_id']} --[{r['label']}]--> {r['target_id']}"
            for r in relationships
        )
        relationship_block = f"Relationships:\n{rel_lines}\n"
    else:
        relationship_block = ""

    return _SUMMARIZE_PROMPT.format(
        entity_block=entity_block,
        relationship_block=relationship_block,
    )


def summarize_community(community_id: int, store: GraphStore, *, force: bool = False) -> bool:
    """Summarize one community. Returns True if generated, False if skipped.

    Skips when the stored member_hash matches current membership, unless force=True.
    """
    entities = store.get_entities_for_community(community_id)
    if not entities:
        logger.debug("Community %d has no entities — skipping", community_id)
        return False

    entity_ids = [e["id"] for e in entities]
    relationships = store.get_relationships_for_community(community_id)
    new_hash = _compute_member_hash(entities, relationships)

    if not force:
        existing_row = store.get_community(community_id)
        if existing_row and existing_row["member_hash"] == new_hash:
            logger.debug("Community %d unchanged — skipping", community_id)
            return False

    prompt = _build_summary_prompt(entities, relationships)

    try:
        summary = ollama_client.generate(prompt, SUMMARIZE_MODEL).strip()
    except Exception as e:
        logger.error("LLM summarization failed for community %d: %s", community_id, e)
        raise

    try:
        embedding_vec = embed(summary)
    except Exception as e:
        logger.error("Embedding failed for community %d: %s", community_id, e)
        raise

    embedding_blob = np.array(embedding_vec, dtype=np.float32).tobytes()
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
    """Run Louvain detection, clean stale communities, then summarize all.

    Returns counts: {summarized, skipped, failed}.
    """
    store.detect_communities()

    active_ids = store.get_active_community_ids()
    logger.info("Detected %d active communities", len(active_ids))

    store.delete_stale_communities()

    counts: dict[str, int] = {"summarized": 0, "skipped": 0, "failed": 0}
    for community_id in sorted(active_ids):
        try:
            generated = summarize_community(community_id, store, force=force)
            counts["summarized" if generated else "skipped"] += 1
        except Exception:
            logger.exception("Failed to summarize community %d", community_id)
            counts["failed"] += 1

    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
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
