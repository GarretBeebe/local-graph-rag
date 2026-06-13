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

- "local": asks about a specific entity, function, class, file, or project by name —
  including "summarize <a named file or project>"
- "global": asks for themes, summaries, or patterns across the WHOLE codebase, not
  about one named file or project

Examples:
Question: Summarize vectorless-rag.md
Answer: local

Question: What does GraphStore do?
Answer: local

Question: Summarize the codebase
Answer: global

Question: What are the main themes across the codebase?
Answer: global

Reply with exactly one word: local or global.

Question: {question}
Answer:"""


_WORD_RE = re.compile(r"[a-z]+")

# A "name.ext" for one of the indexed extensions (config/index_config.yaml) names a
# specific file — "summarize foo.md" is local even though "summarize" is a global
# keyword on its own.
_FILENAME_RE = re.compile(r"\b[\w-]+\.(?:md|py|tsx?|jsx?|json|ya?ml|toml|go|rs|txt)\b")


def _heuristic(question: str) -> Literal["local", "global"]:
    lowered = question.lower()
    if _FILENAME_RE.search(lowered):
        return "local"
    words = set(_WORD_RE.findall(lowered))
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
