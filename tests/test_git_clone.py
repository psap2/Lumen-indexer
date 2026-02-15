"""
Tests for lumen.git_clone — URL utilities, clone, and cleanup.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lumen.git_clone import (
    cleanup_clone,
    clone_repo,
    is_git_url,
    normalize_git_url,
    repo_name_from_url,
)


# ── is_git_url ───────────────────────────────────────────────────────


class TestIsGitUrl:
    @pytest.mark.parametrize("url", [
        "https://github.com/org/repo.git",
        "http://github.com/org/repo.git",
        "git@github.com:org/repo.git",
        "ssh://git@github.com/org/repo.git",
        "git://github.com/public/repo.git",
        "  https://github.com/org/repo.git  ",  # leading/trailing spaces
    ])
    def test_valid_urls(self, url: str):
        assert is_git_url(url) is True

    @pytest.mark.parametrize("url", [
        "/path/to/local/repo",
        "./relative/repo",
        "just-a-name",
        "",
        "ftp://files.example.com/repo.git",
    ])
    def test_invalid_urls(self, url: str):
        assert is_git_url(url) is False


# ── normalize_git_url ────────────────────────────────────────────────


class TestNormalizeGitUrl:
    def test_removes_token_credentials(self):
        url = "https://x-access-token:ghp_xxxxx@github.com/org/repo.git"
        normalized = normalize_git_url(url)
        assert "ghp_xxxxx" not in normalized
        assert "x-access-token" not in normalized
        assert "github.com/org/repo.git" in normalized

    def test_removes_user_pass(self):
        url = "https://user:password@github.com/org/repo.git"
        normalized = normalize_git_url(url)
        assert "password" not in normalized
        assert "user" not in normalized

    def test_preserves_clean_url(self):
        url = "https://github.com/org/repo.git"
        assert normalize_git_url(url) == url

    def test_ssh_url_unchanged(self):
        url = "git@github.com:org/repo.git"
        assert normalize_git_url(url) == url

    def test_strips_whitespace(self):
        url = "  git@github.com:org/repo.git  "
        assert normalize_git_url(url) == "git@github.com:org/repo.git"


# ── repo_name_from_url ──────────────────────────────────────────────


class TestRepoNameFromUrl:
    @pytest.mark.parametrize("url,expected", [
        ("https://github.com/org/my-project.git", "my-project"),
        ("https://github.com/org/my-project", "my-project"),
        ("git@github.com:org/my-project.git", "my-project"),
        ("https://github.com/org/repo/", "repo"),
        ("https://github.com/org/repo.git/", "repo"),
    ])
    def test_extracts_name(self, url: str, expected: str):
        assert repo_name_from_url(url) == expected

    def test_fallback_for_empty(self):
        assert repo_name_from_url("") == "repo"


# ── clone_repo ───────────────────────────────────────────────────────


class TestCloneRepo:
    @patch("lumen.git_clone.subprocess.run")
    @patch("lumen.git_clone.CLONE_DIR", "")
    def test_successful_clone(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        with patch("lumen.git_clone.tempfile.mkdtemp", return_value=str(tmp_path / "clone")):
            (tmp_path / "clone").mkdir()
            result = clone_repo("https://github.com/org/repo.git")
            assert result == tmp_path / "clone"

        # Verify git clone was called with correct args
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "git" in cmd
        assert "clone" in cmd
        assert "--depth" in cmd
        assert "1" in cmd

    @patch("lumen.git_clone.subprocess.run")
    @patch("lumen.git_clone.CLONE_DIR", "")
    def test_clone_with_branch(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        with patch("lumen.git_clone.tempfile.mkdtemp", return_value=str(tmp_path / "clone")):
            (tmp_path / "clone").mkdir()
            clone_repo("https://github.com/org/repo.git", branch="develop")

        cmd = mock_run.call_args[0][0]
        assert "--branch" in cmd
        assert "develop" in cmd

    @patch("lumen.git_clone.subprocess.run")
    @patch("lumen.git_clone.CLONE_DIR", "")
    def test_clone_failure_raises(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = MagicMock(
            returncode=128, stderr="fatal: repo not found", stdout=""
        )
        with patch("lumen.git_clone.tempfile.mkdtemp", return_value=str(tmp_path / "clone")):
            (tmp_path / "clone").mkdir()
            with pytest.raises(RuntimeError, match="Git clone failed"):
                clone_repo("https://github.com/org/nonexistent.git")

    @patch("lumen.git_clone.subprocess.run")
    @patch("lumen.git_clone.CLONE_DIR", "")
    def test_clone_timeout_raises(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=300)
        with patch("lumen.git_clone.tempfile.mkdtemp", return_value=str(tmp_path / "clone")):
            (tmp_path / "clone").mkdir()
            with pytest.raises(RuntimeError, match="timed out"):
                clone_repo("https://github.com/org/huge-repo.git")


# ── cleanup_clone ────────────────────────────────────────────────────


class TestCleanupClone:
    def test_removes_directory(self, tmp_path: Path):
        clone_dir = tmp_path / "clone"
        clone_dir.mkdir()
        (clone_dir / "file.txt").write_text("test")
        cleanup_clone(clone_dir)
        assert not clone_dir.exists()

    def test_nonexistent_path(self, tmp_path: Path):
        """cleanup_clone should not raise for a nonexistent path."""
        nonexistent = tmp_path / "does_not_exist"
        cleanup_clone(nonexistent)  # Should not raise
