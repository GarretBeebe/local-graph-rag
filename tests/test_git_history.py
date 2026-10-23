"""Unit tests for ingest/git_history.py. The last test needs git; the rest fake it."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

import local_graph_rag.ingest.git_history as git_history
from local_graph_rag.ingest.chunkers import _split_markdown_sections
from local_graph_rag.ingest.doc_config import IndexPath
from local_graph_rag.ingest.git_history import (
    Commit,
    _git_log,
    parse_log,
    render_history,
    sync_git_histories,
)

_RS, _US = "\x1e", "\x1f"


def _log_record(short_hash: str, subject: str, body: str = "", files: tuple = ()) -> str:
    """One commit as `git log --name-only --format=_LOG_FORMAT` prints it."""
    return f"{_RS}{short_hash}{_US}2026-10-22{_US}{subject}{_US}{body}{_US}\n\n" + "".join(
        f"{f}\n" for f in files
    )


def _commit(short_hash: str, subject: str, body: str = "", files: list | None = None) -> Commit:
    return Commit(short_hash, "2026-10-22", subject, body, files or [])


# ---------------------------------------------------------------------------
# parse_log / render_history
# ---------------------------------------------------------------------------


def test_parse_log_reads_each_commit():
    output = _log_record("bbb2222", "Second", "Why.\n", ("a.py", "b.py")) + _log_record(
        "aaa1111", "First"
    )

    assert parse_log(output) == [
        _commit("bbb2222", "Second", "Why.", ["a.py", "b.py"]),
        _commit("aaa1111", "First"),
    ]


def test_render_history_lists_recent_changes_then_each_commit():
    commits = [_commit(f"h{i:06d}", f"Change {i}") for i in range(20, 0, -1)]

    text = render_history("repo", commits)

    recent = text.split("## Recent changes to repo (newest first)\n\n", 1)[1].split("\n\n")[0]
    assert recent.splitlines() == [f"- 2026-10-22 h{i:06d} Change {i}" for i in range(20, 5, -1)]
    assert text.count("\n## repo commit ") == 20
    assert "## repo commit h000020 (2026-10-22): Change 20" in text


def test_render_history_caps_long_file_lists():
    files = [f"f{i}.txt" for i in range(40)]

    text = render_history("repo", [_commit("abc1234", "Add data", files=files)])

    assert "Files: f0.txt, f1.txt" in text
    assert "f14.txt (+25 more)" in text
    assert "f15.txt" not in text


def test_commit_bodies_cannot_split_markdown_sections():
    body = "# not a heading\n```\nfenced\n```\nend"

    sections = _split_markdown_sections(render_history("repo", [_commit("abc1234", "S", body)]))

    commit_section = [s for s in sections if s.startswith("## repo commit abc1234")]
    assert len(commit_section) == 1
    assert "    # not a heading" in commit_section[0]
    assert commit_section[0].rstrip().endswith("    end")


# ---------------------------------------------------------------------------
# sync_git_histories
# ---------------------------------------------------------------------------


def _make_repo(root: Path, name: str, *, git: bool = True) -> Path:
    repo = root / name
    repo.mkdir(parents=True)
    if git:
        (repo / ".git").mkdir()
    return repo


@pytest.fixture
def fake_git(monkeypatch: pytest.MonkeyPatch):
    """Replace git with a one-commit log per repo. Map a repo name to an exception to raise,
    or to the raw log text to return, to script a failure."""
    scripted: dict[str, Exception | str] = {}

    def _fake_git_log(repo: Path) -> str:
        outcome = scripted.get(repo.name, _log_record("abc1234", f"Work on {repo.name}"))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(git_history, "_git_log", _fake_git_log)
    return scripted


def test_sync_writes_one_document_per_git_repo(tmp_path: Path, fake_git):
    code, out = tmp_path / "code", tmp_path / "out"
    _make_repo(code, "alpha")
    _make_repo(code, "notes", git=False)
    _make_repo(code, "skipped")
    _make_repo(code, "ignored")

    sync_git_histories(
        [IndexPath(code, projects=True, exclude_dirs=frozenset({"skipped"}))], ["ignored"], out
    )

    assert sorted(p.name for p in out.iterdir()) == ["alpha.md"]
    assert "Work on alpha" in (out / "alpha.md").read_text()


def test_sync_removes_the_document_of_a_deleted_repo(tmp_path: Path, fake_git):
    code, out = tmp_path / "code", tmp_path / "out"
    _make_repo(code, "alpha")
    out.mkdir()
    (out / "gone.md").write_text("old history")

    sync_git_histories([IndexPath(code, projects=True)], [], out)

    assert sorted(p.name for p in out.iterdir()) == ["alpha.md"]


def test_sync_keeps_documents_when_a_projects_root_is_empty_or_missing(tmp_path: Path, fake_git):
    empty, out = tmp_path / "empty", tmp_path / "out"
    empty.mkdir()
    out.mkdir()
    (out / "alpha.md").write_text("old history")

    sync_git_histories(
        [IndexPath(empty, projects=True), IndexPath(tmp_path / "missing", projects=True)], [], out
    )

    assert (out / "alpha.md").read_text() == "old history"


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(128, ["git", "log"]),
        # A commit message holding the log's field separator.
        _log_record("abc1234", f"odd{_US}subject"),
    ],
)
def test_sync_keeps_the_document_of_a_repo_git_fails_on(tmp_path: Path, fake_git, failure, caplog):
    code, out = tmp_path / "code", tmp_path / "out"
    _make_repo(code, "alpha")
    _make_repo(code, "beta")
    out.mkdir()
    (out / "alpha.md").write_text("old history")
    fake_git["alpha"] = failure

    sync_git_histories([IndexPath(code, projects=True)], [], out)

    assert (out / "alpha.md").read_text() == "old history"
    assert "Work on beta" in (out / "beta.md").read_text()
    assert "Skipping git history of" in caplog.text


def test_sync_does_nothing_without_projects(tmp_path: Path, fake_git):
    out = tmp_path / "out"

    sync_git_histories([IndexPath(tmp_path)], [], out)

    assert not out.exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_log_reads_a_real_repository(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.invalid",
        "PATH": os.environ["PATH"], "HOME": str(tmp_path),
    }

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, env=env, capture_output=True)

    git("init", "-q")
    (repo / "a.py").write_text("print(1)\n")
    git("add", "a.py")
    git("commit", "-q", "-m", "Add a", "-m", "Because.")
    (repo / "b.py").write_text("print(2)\n")
    git("add", "b.py")
    git("commit", "-q", "-m", "Add b")

    commits = parse_log(_git_log(repo))

    assert [(c.subject, c.body, c.files) for c in commits] == [
        ("Add b", "", ["b.py"]),
        ("Add a", "Because.", ["a.py"]),
    ]
