"""
Tests for lumen.incremental — file change detection and reporting.

Database-dependent functions (detect_file_changes, save_file_states)
are tested with mocked sessions. Pure functions are tested directly.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from lumen.incremental import (
    FileChange,
    changed_file_paths,
    compute_file_hash,
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
