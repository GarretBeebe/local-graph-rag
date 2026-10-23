"""Narrow retrieval to the projects a question names.

A project is a direct subdirectory of an index path marked `projects: true` in the index
config (each repo under the Code mount), together with its git history document. Unscoped,
a question about one repo also pulled in a sibling repo's code for a setting of the same name.
"""

import logging
import re
from collections.abc import Iterable
from pathlib import Path

from local_graph_rag.graph.store import GraphStore
from local_graph_rag.ingest.doc_config import load_index_config
from local_graph_rag.settings import GIT_HISTORY_DIR

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")
# A name in plain words, a one-word name, or a part of a name ("graph rag") scopes a question
# only next to one of these: otherwise phrases that merely occur in repo names ("rag system",
# "code assistant") would narrow general questions.
_CUE_WORDS = frozenset({"project", "projects", "repo", "repos", "repository", "codebase"})


def _project_roots() -> tuple[Path, ...]:
    """Return the index paths whose subdirectories are projects; none if the config fails.

    Read per question (a small YAML file), so config edits apply without an api restart.
    """
    try:
        config = load_index_config()
    except Exception as e:
        logger.warning("Project scoping is off: cannot load the index config (%s)", e)
        return ()
    return tuple(ip.path for ip in config.index_paths if ip.projects) if config else ()


def _group_by_project(paths: Iterable[str], roots: tuple[Path, ...]) -> dict[str, list[str]]:
    """Map each project name to its indexed paths, including its git history document."""
    projects: dict[str, list[str]] = {}
    histories: dict[str, str] = {}
    history_dir = GIT_HISTORY_DIR.resolve()
    for p in paths:
        path = Path(p)
        if path.parent == history_dir:
            histories[path.stem] = p
            continue
        for root in roots:
            if path.is_relative_to(root) and len(path.relative_to(root).parts) > 1:
                projects.setdefault(path.relative_to(root).parts[0], []).append(p)
                break
    for name, p in histories.items():
        if name in projects:
            projects[name].append(p)
    return projects


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def match_projects(question: str, names: Iterable[str]) -> list[str]:
    """Return the projects a question names, sorted.

    A name of two or more words matches when written as an identifier, ignoring case and
    whether its words are joined by -, _ or nothing ("local_graph_rag", "LocalGraphRag").
    Vaguer mentions match only when the question also says project, repo, repository or
    codebase: the name as plain words ("local graph rag"), a one-word name, or a run of two
    or more words that occurs in only one name ("graph rag").
    """
    lowered = question.lower()
    words = _words(question)
    text = f" {' '.join(words)} "
    cued = not _CUE_WORDS.isdisjoint(words)

    def spelled(phrase: tuple[str, ...]) -> bool:
        # Words are [a-z0-9]+, so they need no regex escaping.
        pattern = rf"(?<![a-z0-9]){'[-_]?'.join(phrase)}(?![a-z0-9])"
        return re.search(pattern, lowered) is not None

    def mentions(phrase: tuple[str, ...]) -> bool:
        return spelled(phrase) or f" {' '.join(phrase)} " in text

    name_words = {name: tuple(_words(name)) for name in names}
    owners: dict[tuple[str, ...], set[str]] = {}
    for name, w in name_words.items():
        for start in range(len(w)):
            for end in range(start + 2, len(w) + 1):
                owners.setdefault(w[start:end], set()).add(name)

    def named(name: str) -> bool:
        w = name_words[name]
        if len(w) > 1 and spelled(w):
            return True
        unique_parts = [p for p, o in owners.items() if o == {name}]
        return cued and bool(w) and (mentions(w) or any(map(mentions, unique_parts)))

    return sorted(filter(named, name_words))


def project_paths(question: str, store: GraphStore) -> list[str] | None:
    """Return the indexed paths of the projects the question names, or None if it names none."""
    roots = _project_roots()
    if not roots:
        return None
    projects = _group_by_project(store.list_all_paths(), roots)
    names = match_projects(question, projects)
    if names:
        logger.info("Retrieval scoped to project(s): %s", ", ".join(names))
    return [p for name in names for p in projects[name]] or None
