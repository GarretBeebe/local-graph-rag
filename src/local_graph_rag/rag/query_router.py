"""Route a question to 'local' or 'global' retrieval."""

import logging
import re
import threading
from typing import Literal

import local_graph_rag.rag.ollama_client as ollama_client
from local_graph_rag.settings import EXTRACT_MODEL

logger = logging.getLogger(__name__)

_GLOBAL_KEYWORDS = frozenset({
    "theme", "themes", "overview", "summarize", "summary",
    "main", "overall", "across", "everything", "codebase",
})

_ROUTER_PROMPT = """\
Classify this question as either "local" or "global".

- "local": asks about a specific entity, function, class, file, or precise relationship
- "global": asks for themes, summaries, overviews, or patterns across the whole codebase

Reply with exactly one word: local or global.

Question: {question}
Answer:"""


_WORD_RE = re.compile(r"[a-z]+")


def _heuristic(question: str) -> Literal["local", "global"]:
    words = set(_WORD_RE.findall(question.lower()))
    return "global" if words & _GLOBAL_KEYWORDS else "local"


def route_query(
    question: str,
    communities_available: bool,
    cancel: threading.Event | None = None,
) -> Literal["local", "global"]:
    """Return 'local' or 'global' retrieval mode for the given question."""
    if not communities_available:
        return "local"

    try:
        response = ollama_client.generate(
            _ROUTER_PROMPT.format(question=question), EXTRACT_MODEL, cancel=cancel
        ).strip().lower()
        if response in ("local", "global"):
            logger.debug("route_query: LLM classified %r → %r", question[:60], response)
            return response  # type: ignore[return-value]
        logger.warning("route_query: unexpected LLM response %r — using heuristic", response)
    except Exception as e:
        logger.warning("route_query: LLM failed (%s) — using heuristic", e)

    return _heuristic(question)
