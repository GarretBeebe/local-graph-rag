"""LLM-based entity and relationship extraction from document chunks."""

import json
import logging
import re
from dataclasses import dataclass, field

import local_graph_rag.rag.ollama_client as ollama_client
from local_graph_rag.graph.store import GraphStore, slugify
from local_graph_rag.settings import EXTRACT_BATCH_TOKENS, EXTRACT_MODEL

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are an information extraction assistant. Extract named entities and relationships \
from the text below. Return ONLY valid JSON, no other text:

{{
  "entities": [{{"name": "string", "type": "string", "description": "string"}}],
  "relationships": [{{"source": "string", "target": "string", "label": "string"}}]
}}

Entity types: FUNCTION, CLASS, MODULE, CONCEPT, PERSON, ORG, FILE, OTHER
Relationship labels: snake_case verbs (uses, depends_on, calls, returns, extends, etc.)
Only extract entities clearly present in the text. Return empty arrays if none found.

Text:
{text}"""


@dataclass
class ExtractionResult:
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _parse_extraction_response(response: str) -> ExtractionResult:
    """Parse LLM output into an ExtractionResult. Falls back gracefully on bad JSON."""
    text = response.strip()

    # Attempt 1: direct parse
    try:
        data = json.loads(text)
        return _dict_to_result(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 1b: model emitted '{}' followed by the real JSON on the next line
    if text.startswith("{}"):
        remainder = text[2:].lstrip()
        if remainder:
            try:
                data = json.loads(remainder)
                return _dict_to_result(data)
            except (json.JSONDecodeError, ValueError):
                pass

    # Attempt 2: extract first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return _dict_to_result(data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Attempt 3: last resort - Python `None` where JSON needs `null`. Only fires
    # after every well-formed-JSON attempt has failed, so this cannot corrupt
    # otherwise-valid JSON. \b prevents matching inside words like "Noneable".
    fixed = re.sub(r"\bNone\b", "null", text)
    if fixed != text:
        try:
            data = json.loads(fixed)
            return _dict_to_result(data)
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning(
        "Failed to parse extraction response; returning empty result. Response: %r", text[:200]
    )
    return ExtractionResult()


def _dict_to_result(data: dict) -> ExtractionResult:
    entities = [e for e in data.get("entities", []) if isinstance(e, dict)]
    relationships = [r for r in data.get("relationships", []) if isinstance(r, dict)]
    return ExtractionResult(entities=entities, relationships=relationships)


def _normalize_entities(entities: list[dict]) -> tuple[list[dict], set[str]]:
    """Strip whitespace, filter empty names, deduplicate by slug (keep longest description).

    Returns (entity_list, slug_set) so callers don't need to recompute slugs.
    """
    by_slug: dict[str, dict] = {}
    for entity in entities:
        name = (entity.get("name") or "").strip()
        if not name:
            continue
        slug = slugify(name)
        if not slug:
            continue
        existing = by_slug.get(slug)
        if existing is None:
            by_slug[slug] = {
                "name": name,
                "type": entity.get("type"),
                "description": entity.get("description", ""),
            }
        else:
            by_slug[slug]["description"] = max(
                str(existing.get("description") or ""),
                str(entity.get("description") or ""),
                key=len,
            )
            if existing.get("type") is None and entity.get("type"):
                by_slug[slug]["type"] = entity["type"]
    return list(by_slug.values()), set(by_slug.keys())


def _normalize_relationships(relationships: list[dict], valid_slugs: set[str]) -> list[dict]:
    """Filter relationships to those whose endpoints exist in valid_slugs."""
    result = []
    for rel in relationships:
        source = (rel.get("source") or "").strip()
        target = (rel.get("target") or "").strip()
        label = (rel.get("label") or "").strip()
        if not (source and target and label):
            continue
        if slugify(source) not in valid_slugs or slugify(target) not in valid_slugs:
            continue
        result.append({"source": source, "target": target, "label": label})
    return result


def extract_entities_for_file(
    chunks: list[str],
    filepath: str,
    store: GraphStore,
) -> ExtractionResult:
    """Extract entities and relationships from all chunks of a file.

    Batches chunks to stay within EXTRACT_BATCH_TOKENS. Caches each batch result in
    the store so interrupted runs resume without redundant LLM calls.
    """
    if not chunks:
        return ExtractionResult()

    # Build batches
    batches: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0
    for chunk in chunks:
        chunk_tokens = _estimate_tokens(chunk)
        if current_parts and current_tokens + chunk_tokens > EXTRACT_BATCH_TOKENS:
            batches.append("\n\n---\n\n".join(current_parts))
            current_parts = []
            current_tokens = 0
        current_parts.append(chunk)
        current_tokens += chunk_tokens
    if current_parts:
        batches.append("\n\n---\n\n".join(current_parts))

    cached = store.get_cached_extractions(filepath)
    all_entities: list[dict] = []
    all_relationships: list[dict] = []

    for i, batch_text in enumerate(batches):
        if i in cached:
            logger.debug("extraction cache hit for %s batch %d", filepath, i)
            result = _parse_extraction_response(cached[i])
        else:
            prompt = _PROMPT_TEMPLATE.format(text=batch_text)
            response = ollama_client.generate(prompt, EXTRACT_MODEL, format="json")
            store.cache_extraction(filepath, i, response)
            result = _parse_extraction_response(response)

        all_entities.extend(result.entities)
        all_relationships.extend(result.relationships)

    normalized_entities, valid_slugs = _normalize_entities(all_entities)
    normalized_relationships = _normalize_relationships(all_relationships, valid_slugs)

    return ExtractionResult(entities=normalized_entities, relationships=normalized_relationships)
