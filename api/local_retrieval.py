"""Local retrieval: Qdrant vector search → entity neighborhood → context."""

import logging
from dataclasses import dataclass, field

from qdrant_client import QdrantClient

from api.embed import embed
from graph.store import GraphStore
from settings import COLLECTION, ENTITY_NEIGHBORHOOD_HOPS, ENTITY_RETRIEVAL_K

logger = logging.getLogger(__name__)

_MAX_ENTITIES = 20
_MAX_RELATIONSHIPS = 40


@dataclass
class LocalContext:
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    chunk_texts: list[str] = field(default_factory=list)


def local_retrieve(
    question: str,
    store: GraphStore,
    client: QdrantClient,
    *,
    k: int = ENTITY_RETRIEVAL_K,
    hops: int = ENTITY_NEIGHBORHOOD_HOPS,
) -> LocalContext:
    """Embed question → Qdrant search → entity neighborhood expansion."""
    vector = embed(question)
    results = client.query_points(COLLECTION, query=vector, limit=k, with_payload=True)

    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
    for point in results.points:
        chunk_ids.append(str(point.id))
        if point.payload and "text" in point.payload:
            chunk_texts.append(point.payload["text"])

    entity_ids = store.get_entities_by_chunk_ids(chunk_ids)
    neighborhoods = store.get_entity_neighborhoods(entity_ids, hops)

    seen_entities: dict[str, dict] = {}
    seen_rels: dict[tuple[str, str, str], dict] = {}
    for neighborhood in neighborhoods.values():
        for e in neighborhood["entities"]:
            seen_entities.setdefault(e["id"], e)
        for r in neighborhood["relationships"]:
            key = (r["source_id"], r["target_id"], r["label"])
            seen_rels.setdefault(key, r)

    entities = list(seen_entities.values())[:_MAX_ENTITIES]
    relationships = list(seen_rels.values())[:_MAX_RELATIONSHIPS]

    logger.debug(
        "local_retrieve: %d chunks → %d entities → %d neighborhood entities, %d rels",
        len(chunk_ids),
        len(entity_ids),
        len(entities),
        len(relationships),
    )
    return LocalContext(entities=entities, relationships=relationships, chunk_texts=chunk_texts)
