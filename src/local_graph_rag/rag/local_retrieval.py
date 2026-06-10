"""Local retrieval: Qdrant vector search → entity neighborhood → context."""

import logging
import re
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from local_graph_rag.graph.store import GraphStore
from local_graph_rag.rag.embed import embed
from local_graph_rag.settings import COLLECTION, ENTITY_NEIGHBORHOOD_HOPS, ENTITY_RETRIEVAL_K

logger = logging.getLogger(__name__)

_MAX_ENTITIES = 20
_MAX_RELATIONSHIPS = 40
_MAX_IDENTIFIER_LOOKUPS = 5
_MAX_PIECES_PER_DEF = 10  # generous cap on chunks-per-def for an oversized def

# Tokens that "look like code": contain an underscore (snake_case / leading underscore)
# or have a lower→upper case transition (CamelCase). This filters out ordinary English
# words while catching def_name shapes chunk_python emits (_parse_extraction_response,
# GraphStore).
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_CAMEL_RE = re.compile(r"[a-z][A-Z]")


@dataclass
class LocalContext:
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    chunk_texts: list[str] = field(default_factory=list)


def _extract_identifiers(question: str) -> list[str]:
    """Return code-symbol-shaped tokens from the question (snake_case, _leading, CamelCase)."""
    return [t for t in _IDENTIFIER_RE.findall(question) if "_" in t or _CAMEL_RE.search(t)]


def _lookup_by_def_names(client: QdrantClient, def_names: list[str]) -> dict[str, list[str]]:
    """Batch exact-match chunks whose def_name payload is in def_names.

    Returns {def_name: [chunk_text, ...]}, each list ordered by chunk_index —
    recovers original order for an oversized def split into multiple chunks.
    """
    if not def_names:
        return {}

    points, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="def_name", match=MatchAny(any=def_names))]
        ),
        limit=_MAX_IDENTIFIER_LOOKUPS * _MAX_PIECES_PER_DEF,
        with_payload=True,
    )

    by_def: dict[str, list[tuple[int, str]]] = {}
    for p in points:
        if not p.payload or "text" not in p.payload or "def_name" not in p.payload:
            continue
        by_def.setdefault(p.payload["def_name"], []).append(
            (p.payload.get("chunk_index", 0), p.payload["text"])
        )

    return {
        name: [text for _, text in sorted(pieces, key=lambda t: t[0])]
        for name, pieces in by_def.items()
    }


def local_retrieve(
    question: str,
    store: GraphStore,
    client: QdrantClient,
    *,
    k: int = ENTITY_RETRIEVAL_K,
    hops: int = ENTITY_NEIGHBORHOOD_HOPS,
) -> LocalContext:
    """Embed question → Qdrant search (+ exact def_name lookup) → entity neighborhood expansion."""
    vector = embed(question)
    results = client.query_points(COLLECTION, query=vector, limit=k, with_payload=True)

    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
    for point in results.points:
        chunk_ids.append(str(point.id))
        if point.payload and "text" in point.payload:
            chunk_texts.append(point.payload["text"])

    seen_texts = set(chunk_texts)
    deduped_identifiers = list(dict.fromkeys(_extract_identifiers(question)))[
        :_MAX_IDENTIFIER_LOOKUPS
    ]
    matches_by_def = _lookup_by_def_names(client, deduped_identifiers)
    for ident in deduped_identifiers:
        for text in matches_by_def.get(ident, []):
            if text not in seen_texts:
                chunk_texts.append(text)
                seen_texts.add(text)

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
