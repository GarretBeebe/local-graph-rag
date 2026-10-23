"""Render each project repo's git history as a Markdown document for the index.

Indexed files say what the code is, not when it changed, so "what changed recently" had
nothing to retrieve. For each git repo directly under an index path marked `projects: true`,
the indexer writes <GIT_HISTORY_DIR>/<repo>.md before collecting files: a "Recent changes"
section, then one section per commit, newest first. The documents are embedded but not
entity-extracted (see index_documents._write_index_data).
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from local_graph_rag.common.paths import matches_ignore_pattern
from local_graph_rag.ingest.doc_config import IndexPath

logger = logging.getLogger(__name__)

_RECENT_COMMITS = 15
# Some commits add whole trees (one repo's 4 commits list 1.2 MB of file names).
_MAX_FILES_PER_COMMIT = 15
_GIT_TIMEOUT_SECONDS = 60
# Record and field separators: control characters that never occur in commit text.
_RECORD_SEP, _FIELD_SEP = "\x1e", "\x1f"
_LOG_FORMAT = _RECORD_SEP + _FIELD_SEP.join(["%h", "%ad", "%s", "%b"]) + _FIELD_SEP


@dataclass
class Commit:
    short_hash: str
    date: str
    subject: str
    body: str
    files: list[str]


def parse_log(output: str) -> list[Commit]:
    """Parse `git log --name-only --format=_LOG_FORMAT` output, newest commit first."""
    commits = []
    for record in output.split(_RECORD_SEP)[1:]:
        short_hash, date, subject, body, files = record.split(_FIELD_SEP)
        commits.append(Commit(
            short_hash=short_hash,
            date=date,
            subject=subject.strip(),
            body=body.strip(),
            files=[line for line in files.splitlines() if line.strip()],
        ))
    return commits


def render_history(repo: str, commits: list[Commit]) -> str:
    """Return the Markdown history of repo. Same commits, same text, so the hash skips it."""
    lines = [f"# Git history of {repo}", "", f"## Recent changes to {repo} (newest first)", ""]
    lines += [f"- {c.date} {c.short_hash} {c.subject}" for c in commits[:_RECENT_COMMITS]]
    for c in commits:
        lines += ["", f"## {repo} commit {c.short_hash} ({c.date}): {c.subject}"]
        if c.body:
            # Indented, so a body line can't open a Markdown heading or code fence and split
            # the chunker's sections.
            lines += [""] + [f"    {line}" if line.strip() else "" for line in c.body.splitlines()]
        if c.files:
            shown = c.files[:_MAX_FILES_PER_COMMIT]
            more = len(c.files) - len(shown)
            lines += ["", "Files: " + ", ".join(shown) + (f" (+{more} more)" if more else "")]
    return "\n".join(lines) + "\n"


def _git_log(repo: Path) -> str:
    # safe.directory: the mounted repos belong to the host user, and git refuses a repo owned
    # by another user unless told to trust it. Passed per command, never written to a config.
    return subprocess.run(
        [
            "git", "-c", "safe.directory=*", "-C", str(repo),
            "log", "--no-merges", "--date=short", "--name-only", f"--format={_LOG_FORMAT}",
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=_GIT_TIMEOUT_SECONDS,
    ).stdout


def _list_dir(path: Path) -> list[Path]:
    try:
        return sorted(path.iterdir())
    except OSError:
        return []


def sync_git_histories(index_paths: list[IndexPath], ignore: list[str], out_dir: Path) -> None:
    """Write out_dir/<repo>.md for every git repo directly under a `projects: true` path.

    Never raises: the history only supplements the documents, so a failure is logged and
    indexing goes on with whatever history documents are already there.
    """
    try:
        _sync(index_paths, ignore, out_dir)
    except Exception:
        logger.exception("Git history sync failed; indexing continues with the existing files")


def _sync(index_paths: list[IndexPath], ignore: list[str], out_dir: Path) -> None:
    project_roots = [ip for ip in index_paths if ip.projects]
    if not project_roots:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    present: set[str] = set()
    all_roots_listed = True
    for ip in project_roots:
        children = _list_dir(ip.path)
        if not children:
            # Missing, unreadable, or empty (an offline mount looks empty): which repos are
            # gone can't be told, so no history document is deleted this run.
            all_roots_listed = False
            continue
        for repo in children:
            if repo.name in ip.exclude_dirs or matches_ignore_pattern(repo.name, ignore):
                continue
            # A .git file (worktree or submodule) whose target the container can't see makes
            # git fail below, which skips the repo like any other git error.
            if not (repo / ".git").exists():
                continue
            # Before running git, so a git error keeps the existing document rather than
            # deleting it.
            present.add(repo.name)
            try:
                text = render_history(repo.name, parse_log(_git_log(repo)))
            except (OSError, subprocess.SubprocessError, ValueError) as e:
                # ValueError: a commit message holding the log's separator characters.
                logger.warning("Skipping git history of %s: %s", repo, e)
                continue
            (out_dir / f"{repo.name}.md").write_text(text, encoding="utf-8")
    if all_roots_listed:
        for doc in out_dir.glob("*.md"):
            if doc.stem not in present:
                doc.unlink()
                logger.info("Removed git history of %s: the repo is gone", doc.stem)
