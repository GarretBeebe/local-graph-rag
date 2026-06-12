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
    had_failure: bool = False


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _none_to_null(text: str) -> str:
    """Replace bareword Python `None` with JSON `null` in clear value positions only.

    Only matches `None` immediately after `:`, `,`, or `[` and immediately before
    `,`, `}`, or `]` — i.e. where JSON requires a value, not inside string content.
    Residual risk: a string containing literal ", None," at a value-boundary could
    still be rewritten. Accepted because this only fires on already-invalid JSON as
    last-resort recovery (stale-cache-replay only — see _candidates).
    """
    return re.sub(r"([:,\[]\s*)None(?=\s*[,}\]])", r"\1null", text)


def _candidates(text: str) -> list[str]:
    """Generate candidate JSON strings to try, in order of preference.

    Candidates 2 and 4/5 below exist only to recover stale cached responses that
    predate `format="json"` — fresh grammar-constrained output shouldn't need them
    and they can be removed once such caches are no longer relevant.
    """
    out = [text]

    # Candidate 2: model emitted '{}' followed by the real JSON on the next line.
    if text.startswith("{}"):
        remainder = text[2:].lstrip()
        if remainder:
            out.append(remainder)
            # Candidate 2b: same, with None -> null substitution on the remainder.
            fixed_remainder = _none_to_null(remainder)
            if fixed_remainder != remainder:
                out.append(fixed_remainder)

    # Candidate 3: extract first {...} block (handles leading/trailing prose).
    match = re.search(r"\{.*\}", text, re.DOTALL)
    block = match.group() if match else None
    if block is not None and block != text:
        out.append(block)

    # Candidate 4: substitute None -> null on the full text.
    fixed_text = _none_to_null(text)
    if fixed_text != text:
        out.append(fixed_text)

    # Candidate 5: substitute None -> null on the isolated {...} block (if any).
    if block is not None and block != text:
        fixed_block = _none_to_null(block)
        if fixed_block != block:
            out.append(fixed_block)

    return out


def _parse_extraction_response(response: str) -> ExtractionResult:
    """Parse LLM output into an ExtractionResult. Falls back gracefully on bad JSON."""
    text = response.strip()

    for candidate in _candidates(text):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
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
    cached: dict[int, str],
    store: GraphStore,
) -> ExtractionResult:
    if batch_index in cached:
        logger.debug("extraction cache hit for %s batch %d", filepath, batch_index)
        return _parse_extraction_response(cached[batch_index])

    prompt = _PROMPT_TEMPLATE.format(text=batch_text)
    response = ollama_client.generate(prompt, EXTRACT_MODEL, format="json")
    store.cache_extraction(filepath, batch_index, response)
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
    )


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

    batches = _build_batches(chunks)
    cached = store.get_cached_extractions(filepath)
    all_entities: list[dict] = []
    all_relationships: list[dict] = []
    any_failure = False

    for i, batch_text in enumerate(batches):
        try:
            result = _load_or_extract_batch(batch_text, i, filepath, cached, store)
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
