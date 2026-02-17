"""
Tests for lumen.incremental — file change detection and reporting.

Database-dependent functions (detect_file_changes, save_file_states)
are tested with mocked sessions. Pure functions are tested directly.
"""

from __future__ import annotations

import hashlib
import subprocess
import uuid
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from lumen.incremental import (
    FileChange,
    _get_head_commit,
    _git_diff_files,
    _git_untracked_files,
    _is_commit_reachable,
    _is_git_repo,
    changed_file_paths,
    compute_file_hash,
    detect_file_changes_git,
    get_change_summary,
    stale_file_paths,
)


# ── compute_file_hash ────────────────────────────────────────────────


class TestComputeFileHash:
    def test_deterministic(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2

    def test_correct_sha256(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        content = b"hello world"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert compute_file_hash(f) == expected

    def test_different_content_different_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert compute_file_hash(f) == expected

    def test_binary_file(self, tmp_path: Path):
        f = tmp_path / "binary.bin"
        data = bytes(range(256))
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert compute_file_hash(f) == expected


# ── FileChange dataclass ─────────────────────────────────────────────


class TestFileChange:
    def test_new_file(self):
        fc = FileChange(path="src/new.py", status="new", content_hash="abc123")
        assert fc.status == "new"

    def test_modified_file(self):
        fc = FileChange(path="src/mod.py", status="modified", content_hash="def456")
        assert fc.status == "modified"

    def test_deleted_file(self):
        fc = FileChange(path="src/old.py", status="deleted", content_hash="")
        assert fc.status == "deleted"
        assert fc.content_hash == ""

    def test_unchanged_file(self):
        fc = FileChange(path="src/ok.py", status="unchanged", content_hash="ghi789")
        assert fc.status == "unchanged"


# ── get_change_summary ───────────────────────────────────────────────


class TestGetChangeSummary:
    def test_mixed_changes(self):
        changes = [
            FileChange(path="a.py", status="new", content_hash="a"),
            FileChange(path="b.py", status="new", content_hash="b"),
            FileChange(path="c.py", status="modified", content_hash="c"),
            FileChange(path="d.py", status="deleted", content_hash=""),
            FileChange(path="e.py", status="unchanged", content_hash="e"),
            FileChange(path="f.py", status="unchanged", content_hash="f"),
        ]
        summary = get_change_summary(changes)
        assert "2 new" in summary
        assert "1 modified" in summary
        assert "1 deleted" in summary
        assert "2 unchanged" in summary

    def test_all_new(self):
        changes = [
            FileChange(path="a.py", status="new", content_hash="a"),
        ]
        summary = get_change_summary(changes)
        assert "1 new" in summary
        assert "modified" not in summary

    def test_empty_changes(self):
        summary = get_change_summary([])
        assert summary == "no files found"

    def test_only_unchanged(self):
        changes = [
            FileChange(path="a.py", status="unchanged", content_hash="a"),
        ]
        summary = get_change_summary(changes)
        assert "1 unchanged" in summary


# ── changed_file_paths ───────────────────────────────────────────────


class TestChangedFilePaths:
    def test_returns_new_and_modified(self):
        changes = [
            FileChange(path="new.py", status="new", content_hash="a"),
            FileChange(path="mod.py", status="modified", content_hash="b"),
            FileChange(path="del.py", status="deleted", content_hash=""),
            FileChange(path="ok.py", status="unchanged", content_hash="c"),
        ]
        result = changed_file_paths(changes)
        assert result == {"new.py", "mod.py"}

    def test_empty_changes(self):
        assert changed_file_paths([]) == set()

    def test_no_new_or_modified(self):
        changes = [
            FileChange(path="a.py", status="unchanged", content_hash="a"),
            FileChange(path="b.py", status="deleted", content_hash=""),
        ]
        assert changed_file_paths(changes) == set()


# ── stale_file_paths ─────────────────────────────────────────────────


class TestStaleFilePaths:
    def test_returns_modified_and_deleted(self):
        changes = [
            FileChange(path="new.py", status="new", content_hash="a"),
            FileChange(path="mod.py", status="modified", content_hash="b"),
            FileChange(path="del.py", status="deleted", content_hash=""),
            FileChange(path="ok.py", status="unchanged", content_hash="c"),
        ]
        result = stale_file_paths(changes)
        assert result == {"mod.py", "del.py"}

    def test_empty_changes(self):
        assert stale_file_paths([]) == set()

    def test_no_stale(self):
        changes = [
            FileChange(path="a.py", status="new", content_hash="a"),
            FileChange(path="b.py", status="unchanged", content_hash="b"),
        ]
        assert stale_file_paths(changes) == set()


# ── _is_git_repo ─────────────────────────────────────────────────────


class TestIsGitRepo:
    def test_true_for_git_repo(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        assert _is_git_repo(tmp_path) is True

    def test_false_for_non_git_dir(self, tmp_path: Path):
        assert _is_git_repo(tmp_path) is False

    def test_false_when_git_is_file(self, tmp_path: Path):
        (tmp_path / ".git").write_text("not a directory")
        assert _is_git_repo(tmp_path) is False


# ── _get_head_commit ─────────────────────────────────────────────────


class TestGetHeadCommit:
    def test_returns_sha_for_real_git_repo(self, tmp_path: Path):
        """Create a real git repo with a commit and verify we get a SHA."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(tmp_path), capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )

        sha = _get_head_commit(tmp_path)
        assert sha is not None
        assert len(sha) == 40  # Full SHA-1 hex

    def test_returns_none_for_non_git_dir(self, tmp_path: Path):
        assert _get_head_commit(tmp_path) is None


# ── _is_commit_reachable ─────────────────────────────────────────────


class TestIsCommitReachable:
    def test_reachable_commit(self, tmp_path: Path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(tmp_path), capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )

        sha = _get_head_commit(tmp_path)
        assert _is_commit_reachable(tmp_path, sha) is True

    def test_unreachable_commit(self, tmp_path: Path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        assert _is_commit_reachable(tmp_path, "deadbeef" * 5) is False

    def test_non_git_dir(self, tmp_path: Path):
        assert _is_commit_reachable(tmp_path, "abc123") is False


# ── _git_diff_files ──────────────────────────────────────────────────


def _init_git_repo(path: Path) -> str:
    """Helper: create a git repo with one commit, return the SHA."""
    subprocess.run(["git", "init"], cwd=str(path), capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(path), capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(path), capture_output=True,
    )
    (path / "main.py").write_text("print('hello')\n")
    (path / "utils.py").write_text("def add(a, b): return a + b\n")
    subprocess.run(["git", "add", "."], cwd=str(path), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=str(path), capture_output=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(path), capture_output=True, text=True,
    )
    return result.stdout.strip()


class TestGitDiffFiles:
    def test_detects_modified_file(self, tmp_path: Path):
        base_sha = _init_git_repo(tmp_path)
        (tmp_path / "main.py").write_text("print('modified')\n")
        diff = _git_diff_files(tmp_path, base_sha)
        assert diff is not None
        assert "main.py" in diff
        assert diff["main.py"] == "M"

    def test_detects_deleted_file(self, tmp_path: Path):
        base_sha = _init_git_repo(tmp_path)
        (tmp_path / "utils.py").unlink()
        diff = _git_diff_files(tmp_path, base_sha)
        assert diff is not None
        assert "utils.py" in diff
        assert diff["utils.py"] == "D"

    def test_detects_added_file(self, tmp_path: Path):
        base_sha = _init_git_repo(tmp_path)
        (tmp_path / "new_file.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "new_file.py"], cwd=str(tmp_path), capture_output=True)
        diff = _git_diff_files(tmp_path, base_sha)
        assert diff is not None
        assert "new_file.py" in diff
        assert diff["new_file.py"] == "A"

    def test_no_changes_returns_empty(self, tmp_path: Path):
        base_sha = _init_git_repo(tmp_path)
        diff = _git_diff_files(tmp_path, base_sha)
        assert diff is not None
        assert diff == {}

    def test_returns_none_for_non_git_dir(self, tmp_path: Path):
        assert _git_diff_files(tmp_path, "abc123") is None


# ── _git_untracked_files ─────────────────────────────────────────────


class TestGitUntrackedFiles:
    def test_finds_untracked_files(self, tmp_path: Path):
        _init_git_repo(tmp_path)
        (tmp_path / "untracked.py").write_text("# new\n")
        untracked = _git_untracked_files(tmp_path)
        assert untracked is not None
        assert "untracked.py" in untracked

    def test_no_untracked_returns_empty(self, tmp_path: Path):
        _init_git_repo(tmp_path)
        untracked = _git_untracked_files(tmp_path)
        assert untracked is not None
        assert untracked == []

    def test_returns_none_for_non_git_dir(self, tmp_path: Path):
        assert _git_untracked_files(tmp_path) is None


# ── detect_file_changes_git ──────────────────────────────────────────


class TestDetectFileChangesGit:
    """End-to-end tests using real temporary git repos."""

    EXTENSIONS = frozenset({".py", ".ts"})

    def test_detects_modified_and_new(self, tmp_path: Path):
        base_sha = _init_git_repo(tmp_path)
        # Modify tracked file
        (tmp_path / "main.py").write_text("print('changed')\n")
        # Add untracked file
        (tmp_path / "new_module.py").write_text("x = 42\n")

        changes = detect_file_changes_git(
            tmp_path, base_sha, self.EXTENSIONS,
        )
        assert changes is not None
        paths = {c.path: c.status for c in changes}
        assert paths["main.py"] == "modified"
        assert paths["new_module.py"] == "new"
        # utils.py is unchanged — should not appear
        assert "utils.py" not in paths

    def test_detects_deletion(self, tmp_path: Path):
        base_sha = _init_git_repo(tmp_path)
        (tmp_path / "utils.py").unlink()

        changes = detect_file_changes_git(
            tmp_path, base_sha, self.EXTENSIONS,
        )
        assert changes is not None
        paths = {c.path: c.status for c in changes}
        assert paths["utils.py"] == "deleted"

    def test_filters_by_extension(self, tmp_path: Path):
        base_sha = _init_git_repo(tmp_path)
        (tmp_path / "readme.md").write_text("# Hello\n")  # Not in extensions

        changes = detect_file_changes_git(
            tmp_path, base_sha, self.EXTENSIONS,
        )
        assert changes is not None
        assert all(c.path != "readme.md" for c in changes)

    def test_filters_by_ignore_patterns(self, tmp_path: Path):
        base_sha = _init_git_repo(tmp_path)
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "dep.py").write_text("x = 1\n")

        changes = detect_file_changes_git(
            tmp_path, base_sha, self.EXTENSIONS,
            ignore_patterns=["node_modules"],
        )
        assert changes is not None
        assert all("node_modules" not in c.path for c in changes)

    def test_returns_none_for_unreachable_commit(self, tmp_path: Path):
        _init_git_repo(tmp_path)
        result = detect_file_changes_git(
            tmp_path, "deadbeef" * 5, self.EXTENSIONS,
        )
        assert result is None

    def test_returns_none_for_non_git_dir(self, tmp_path: Path):
        result = detect_file_changes_git(
            tmp_path, "abc123", self.EXTENSIONS,
        )
        assert result is None

    def test_content_hash_computed_for_changed_files(self, tmp_path: Path):
        base_sha = _init_git_repo(tmp_path)
        (tmp_path / "main.py").write_text("print('new')\n")

        changes = detect_file_changes_git(
            tmp_path, base_sha, self.EXTENSIONS,
        )
        assert changes is not None
        modified = [c for c in changes if c.path == "main.py"][0]
        assert modified.content_hash != ""
        expected = hashlib.sha256(b"print('new')\n").hexdigest()
        assert modified.content_hash == expected

    def test_deleted_file_has_empty_hash(self, tmp_path: Path):
        base_sha = _init_git_repo(tmp_path)
        (tmp_path / "utils.py").unlink()

        changes = detect_file_changes_git(
            tmp_path, base_sha, self.EXTENSIONS,
        )
        deleted = [c for c in changes if c.path == "utils.py"][0]
        assert deleted.content_hash == ""


# ── Fallback behavior ────────────────────────────────────────────────


class TestGitFallback:
    """Verify that detect_file_changes_git returns None when it should."""

    def test_non_git_repo_returns_none(self, tmp_path: Path):
        assert detect_file_changes_git(tmp_path, "abc", frozenset({".py"})) is None

    def test_missing_commit_returns_none(self, tmp_path: Path):
        _init_git_repo(tmp_path)
        assert detect_file_changes_git(
            tmp_path, "0" * 40, frozenset({".py"}),
        ) is None
