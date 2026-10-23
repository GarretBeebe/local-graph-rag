"""LLM-based entity and relationship extraction from document chunks."""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

import local_graph_rag.rag.ollama_client as ollama_client
from local_graph_rag.graph.store import GraphStore, slugify
from local_graph_rag.settings import EXTRACT_BATCH_TOKENS, EXTRACT_MODEL

logger = logging.getLogger(__name__)

_ENTITY_TYPES = ("FUNCTION", "CLASS", "MODULE", "CONCEPT", "PERSON", "ORG", "FILE", "OTHER")

# The text is fenced, and the instruction repeated after it, because documents that hold
# prompts of their own (this file, for one) otherwise win: the model answered them with `{}`
# or enumerated module-level names until the timeout. Also telling it not to follow the
# document's instructions fixed nothing more and cost a sixth of the entities.
_PROMPT_TEMPLATE = """\
You are an information extraction assistant. Extract named entities and relationships \
from the text below. Return ONLY valid JSON, no other text:

{{
  "entities": [{{"name": "string", "type": "string", "description": "string"}}],
  "relationships": [{{"source": "string", "target": "string", "label": "string"}}]
}}

Entity types: {types}
Relationship labels: snake_case verbs (uses, depends_on, calls, returns, extends, etc.)
Only extract entities clearly present in the text. Return empty arrays if none found.

Text:
<document>
{text}
</document>

Now extract the entities and relationships from the document above."""

# Ollama compiles this into a decoding grammar: the reply always has both arrays and a known
# entity type. The caps sit just above the largest legitimate batch seen (59 entities, 27
# relationships, p99 description 236 chars) and only break loops; num_predict is the hard
# bound. A required description cost recall, so it stays optional.
_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "maxItems": 60,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 80},
                    "type": {"type": "string", "enum": list(_ENTITY_TYPES)},
                    "description": {"type": "string", "maxLength": 240},
                },
                "required": ["name", "type"],
                "additionalProperties": False,
            },
        },
        "relationships": {
            "type": "array",
            "maxItems": 30,
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "maxLength": 80},
                    "target": {"type": "string", "maxLength": 80},
                    "label": {"type": "string", "maxLength": 40},
                },
                "required": ["source", "target", "label"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "relationships"],
    "additionalProperties": False,
}


# Deterministic decoding: repeatable extractions, and cached results mean what they say.
# repeat_penalty: Ollama's llama-server runner defaults it to 1.0, which let one dense batch
# repeat description boilerplate for 2,333 tokens (830 with 1.1).
# num_predict: ~1.5x the longest legitimate reply (2,680 tokens). A reply cut off there is
# invalid JSON, so it parses as empty and is cached like any reply: at temperature 0 a retry
# would only reproduce it.
_EXTRACT_OPTIONS = {"temperature": 0, "repeat_penalty": 1.1, "num_predict": 4096}
# Covers a capped reply (~4 min at ~18 tok/s) plus time queued behind other generations, so
# it only fires on a hang.
_EXTRACT_TIMEOUT_SECONDS = 600.0


@dataclass
class ExtractionResult:
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    had_failure: bool = False
    # Relationships the LLM returned, before dropping ones whose source or target was not
    # also extracted as an entity — the gap is a measure of graph sparsity.
    relationships_extracted: int = 0


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _parse_extraction_response(response: str) -> ExtractionResult:
    """Parse LLM output into an ExtractionResult. Falls back gracefully on bad JSON.

    Tries the whole text, then its outermost {...} block (leading/trailing prose).
    """
    text = response.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    for candidate in (text, match.group() if match else ""):
        try:
            data = json.loads(candidate)
        # JSONDecodeError subclasses ValueError; absurdly deep nesting raises RecursionError.
        # Both must degrade to "no result": the raw reply is cached and replayed every run.
        except (ValueError, RecursionError):
            continue
        return _dict_to_result(data)

    logger.warning(
        "Failed to parse extraction response; returning empty result. Response: %r", text[:200]
    )
    return ExtractionResult()


def _as_dict_list(value: object) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _dict_to_result(data: object) -> ExtractionResult:
    if not isinstance(data, dict):
        return ExtractionResult()
    return ExtractionResult(
        entities=_as_dict_list(data.get("entities")),
        relationships=_as_dict_list(data.get("relationships")),
    )


def _text(value: object) -> str:
    """Return a stripped string field, or "" when the LLM returned another type.

    Raw replies are cached and replayed, so a malformed field must degrade to "missing"
    rather than raise on every run.
    """
    return value.strip() if isinstance(value, str) else ""


def _normalize_entities(entities: list[dict]) -> tuple[list[dict], set[str]]:
    """Strip whitespace, filter empty names, deduplicate by slug (keep longest description).

    Returns (entity_list, slug_set) so callers don't need to recompute slugs.
    """
    by_slug: dict[str, dict] = {}
    for entity in entities:
        name = _text(entity.get("name"))
        slug = slugify(name)
        if not slug:
            continue
        entity_type = _text(entity.get("type")) or None
        description = _text(entity.get("description"))
        existing = by_slug.get(slug)
        if existing is None:
            by_slug[slug] = {"name": name, "type": entity_type, "description": description}
        else:
            existing["description"] = max(existing["description"], description, key=len)
            if existing["type"] is None:
                existing["type"] = entity_type
    return list(by_slug.values()), set(by_slug.keys())


def _normalize_relationships(relationships: list[dict], valid_slugs: set[str]) -> list[dict]:
    """Filter relationships to those whose endpoints exist in valid_slugs."""
    result = []
    for rel in relationships:
        source, target, label = (_text(rel.get(key)) for key in ("source", "target", "label"))
        if not (source and target and label):
            continue
        if slugify(source) not in valid_slugs or slugify(target) not in valid_slugs:
            continue
        result.append({"source": source, "target": target, "label": label})
    return result


def _build_batches(chunks: list[str]) -> list[str]:
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
    return batches


def _load_or_extract_batch(
    batch_text: str,
    batch_index: int,
    filepath: str,
    store: GraphStore,
) -> ExtractionResult:
    prompt = _PROMPT_TEMPLATE.format(
        types=", ".join(_ENTITY_TYPES),
        text=batch_text.replace("</document>", "<\\/document>"),
    )
    # Key the cache on the exact request body, so edited text, a new model, prompt, or
    # option all miss instead of replaying a result computed for something else.
    payload = ollama_client.build_generate_payload(
        EXTRACT_MODEL, prompt, stream=False, format=_EXTRACT_SCHEMA, options=_EXTRACT_OPTIONS
    )
    prompt_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    cached = store.get_cached_extraction(filepath, batch_index, prompt_sha256)
    if cached is not None:
        logger.debug("extraction cache hit for %s batch %d", filepath, batch_index)
        return _parse_extraction_response(cached)

    response = ollama_client.generate(
        prompt,
        EXTRACT_MODEL,
        timeout=_EXTRACT_TIMEOUT_SECONDS,
        format=_EXTRACT_SCHEMA,
        options=_EXTRACT_OPTIONS,
    )
    store.cache_extraction(filepath, batch_index, prompt_sha256, response)
    return _parse_extraction_response(response)


def _normalize_extraction_result(
    entities: list[dict],
    relationships: list[dict],
    *,
    had_failure: bool,
) -> ExtractionResult:
    normalized_entities, valid_slugs = _normalize_entities(entities)
    return ExtractionResult(
        entities=normalized_entities,
        relationships=_normalize_relationships(relationships, valid_slugs),
        had_failure=had_failure,
        relationships_extracted=len(relationships),
    )


def extract_entities_for_file(
    chunks: list[str],
    filepath: str,
    store: GraphStore,
) -> ExtractionResult:
    """Extract entities and relationships from all chunks of a file.

    Batches chunks to stay within EXTRACT_BATCH_TOKENS. Caches each batch result keyed by
    its exact request, so interrupted runs and unchanged batches skip redundant LLM calls.
    """
    if not chunks:
        return ExtractionResult()

    batches = _build_batches(chunks)
    all_entities: list[dict] = []
    all_relationships: list[dict] = []
    any_failure = False

    for i, batch_text in enumerate(batches):
        try:
            result = _load_or_extract_batch(batch_text, i, filepath, store)
        except Exception:
            logger.exception(
                "Extraction failed for %s batch %d; treating as empty", filepath, i
            )
            result = ExtractionResult()
            any_failure = True

        all_entities.extend(result.entities)
        all_relationships.extend(result.relationships)

    return _normalize_extraction_result(
        all_entities,
        all_relationships,
        had_failure=any_failure,
    )
