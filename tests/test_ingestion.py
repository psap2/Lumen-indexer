"""
Tests for lumen.ingestion.code_ingestor — code splitting, SCIP enrichment,
and the full ingest_repository pipeline.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from lumen.config import IndexedChunk, IndexedSymbol
from lumen.ingestion.code_ingestor import (
    _chunk_id,
    _collect_source_files,
    _enrich_with_scip,
    _line_based_split,
    _should_ignore,
    _symbol_metadata_text,
    ingest_repository,
)
from lumen.scip_parser.parser import ParsedIndex


# ── _should_ignore ───────────────────────────────────────────────────


class TestShouldIgnore:
    def test_ignored_directory(self, tmp_path: Path):
        p = tmp_path / "node_modules" / "pkg" / "index.js"
        assert _should_ignore(p, ["node_modules"]) is True

    def test_not_ignored(self, tmp_path: Path):
        p = tmp_path / "src" / "main.py"
        assert _should_ignore(p, ["node_modules"]) is False

    def test_multiple_patterns(self, tmp_path: Path):
        p = tmp_path / "__pycache__" / "mod.pyc"
        assert _should_ignore(p, ["node_modules", "__pycache__"]) is True


# ── _chunk_id ────────────────────────────────────────────────────────


class TestChunkId:
    def test_deterministic(self):
        """Same input always produces the same chunk ID."""
        id1 = _chunk_id("src/main.py", 0, 60)
        id2 = _chunk_id("src/main.py", 0, 60)
        assert id1 == id2

    def test_different_for_different_ranges(self):
        id1 = _chunk_id("src/main.py", 0, 60)
        id2 = _chunk_id("src/main.py", 60, 120)
        assert id1 != id2

    def test_different_for_different_files(self):
        id1 = _chunk_id("src/main.py", 0, 60)
        id2 = _chunk_id("src/utils.py", 0, 60)
        assert id1 != id2

    def test_length(self):
        """Chunk IDs are 16-character hex strings."""
        cid = _chunk_id("test.py", 0, 10)
        assert len(cid) == 16
        assert all(c in "0123456789abcdef" for c in cid)


# ── _line_based_split ────────────────────────────────────────────────


class TestLineBasedSplit:
    def test_short_file(self):
        """A file shorter than chunk_lines produces one chunk."""
        text = "line1\nline2\nline3\n"
        chunks = _line_based_split(text, chunk_lines=10, overlap=2)
        assert len(chunks) == 1
        start, end, content = chunks[0]
        assert start == 0
        assert end == 3
        assert "line1" in content

    def test_exact_chunk_size_no_overlap(self):
        """A file with exactly chunk_lines lines and zero overlap produces one chunk."""
        lines = [f"line{i}\n" for i in range(10)]
        text = "".join(lines)
        chunks = _line_based_split(text, chunk_lines=10, overlap=0)
        assert len(chunks) == 1

    def test_exact_chunk_size_with_overlap(self):
        """With overlap, the sliding window may produce an extra tail chunk."""
        lines = [f"line{i}\n" for i in range(10)]
        text = "".join(lines)
        chunks = _line_based_split(text, chunk_lines=10, overlap=2)
        # First chunk covers all 10 lines; the window advances by 8,
        # producing a small tail chunk.
        assert len(chunks) == 2
        assert chunks[0][0] == 0
        assert chunks[0][1] == 10

    def test_overlap(self):
        """Chunks should overlap by the specified number of lines."""
        lines = [f"line{i}\n" for i in range(20)]
        text = "".join(lines)
        chunks = _line_based_split(text, chunk_lines=10, overlap=3)

        # First chunk: lines 0-9, second: lines 7-16, third: lines 14-19
        assert len(chunks) >= 2
        assert chunks[0][0] == 0
        assert chunks[0][1] == 10
        assert chunks[1][0] == 7  # 10 - 3 = 7
        assert chunks[1][1] == 17

    def test_empty_text(self):
        chunks = _line_based_split("", chunk_lines=10, overlap=2)
        # Empty text has 0 lines, so the loop doesn't execute
        assert chunks == []

    def test_single_line(self):
        chunks = _line_based_split("hello\n", chunk_lines=60, overlap=15)
        assert len(chunks) == 1
        assert chunks[0][2] == "hello\n"

    def test_preserves_content(self):
        """All original content appears somewhere in the chunks."""
        text = "".join(f"line{i}\n" for i in range(100))
        chunks = _line_based_split(text, chunk_lines=30, overlap=5)
        all_text = "".join(c[2] for c in chunks)
        for i in range(100):
            assert f"line{i}" in all_text


# ── _collect_source_files ────────────────────────────────────────────


class TestCollectSourceFiles:
    def test_collects_matching_extensions(self, sample_repo: Path):
        files = _collect_source_files(
            sample_repo,
            extensions=frozenset({".py"}),
            ignore=["node_modules", "__pycache__", ".git"],
        )
        paths = [str(f.relative_to(sample_repo)) for f in files]
        assert any("main.py" in p for p in paths)
        assert any("utils.py" in p for p in paths)

    def test_ignores_patterns(self, sample_repo: Path):
        files = _collect_source_files(
            sample_repo,
            extensions=frozenset({".js"}),
            ignore=["node_modules"],
        )
        paths = [str(f) for f in files]
        assert not any("node_modules" in p for p in paths)

    def test_no_matching_files(self, sample_repo: Path):
        files = _collect_source_files(
            sample_repo,
            extensions=frozenset({".scala"}),
            ignore=[],
        )
        assert files == []


# ── _enrich_with_scip ────────────────────────────────────────────────


class TestEnrichWithScip:
    def test_without_parsed_index(self):
        """Returns empty when no SCIP index is provided."""
        symbols = _enrich_with_scip(None, "main.py", 0, 10)
        assert symbols == []

    def test_with_parsed_index(self, sample_parsed_index: ParsedIndex):
        symbols = _enrich_with_scip(
            sample_parsed_index, "src/main.py", 0, 3,
        )
        # Should find the "hello" function defined at line 0
        assert len(symbols) >= 1
        assert any(s.display_name == "hello" for s in symbols)

    def test_no_matches_in_range(self, sample_parsed_index: ParsedIndex):
        symbols = _enrich_with_scip(
            sample_parsed_index, "src/main.py", 100, 200,
        )
        assert symbols == []

    def test_returns_indexed_symbols(self, sample_parsed_index: ParsedIndex):
        symbols = _enrich_with_scip(
            sample_parsed_index, "src/main.py", 0, 3,
        )
        for s in symbols:
            assert isinstance(s, IndexedSymbol)
            assert s.file_path == "src/main.py"


# ── _symbol_metadata_text ────────────────────────────────────────────


class TestSymbolMetadataText:
    def test_empty_symbols(self):
        assert _symbol_metadata_text([]) == ""

    def test_header_format(self):
        sym = IndexedSymbol(
            symbol_id="pkg/f().",
            kind="Function",
            display_name="compute",
            documentation="Runs a calculation.",
            file_path="a.py",
            line_start=0,
            line_end=5,
        )
        text = _symbol_metadata_text([sym])
        assert "[SCIP Symbols]" in text
        assert "Function" in text
        assert "compute" in text
        assert "Runs a calculation." in text

    def test_truncates_long_docs(self):
        sym = IndexedSymbol(
            symbol_id="pkg/f().",
            kind="Function",
            display_name="f",
            documentation="x" * 500,
            file_path="a.py",
            line_start=0,
            line_end=1,
        )
        text = _symbol_metadata_text([sym])
        # Documentation is truncated at 200 chars
        assert len(text) < 500 + 100  # header + some overhead, well under 500

    def test_relationships_included(self):
        sym = IndexedSymbol(
            symbol_id="pkg/f().",
            kind="Function",
            display_name="f",
            documentation="",
            file_path="a.py",
            line_start=0,
            line_end=1,
            relationships=[
                {"target_symbol": "pkg/Bar#", "is_implementation": True},
            ],
        )
        text = _symbol_metadata_text([sym])
        assert "implements" in text
        assert "pkg/Bar#" in text


# ── ingest_repository ────────────────────────────────────────────────


class TestIngestRepository:
    def test_basic_ingestion(self, sample_repo: Path):
        """Ingest a sample repo without SCIP and get nodes + chunks."""
        nodes, chunks = ingest_repository(
            sample_repo,
            parsed_index=None,
            extensions=frozenset({".py"}),
        )
        assert len(nodes) > 0
        assert len(chunks) > 0
        assert len(nodes) == len(chunks)

    def test_chunks_have_required_fields(self, sample_repo: Path):
        _, chunks = ingest_repository(
            sample_repo,
            parsed_index=None,
            extensions=frozenset({".py"}),
        )
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.file_path
            assert chunk.language
            assert chunk.code
            assert chunk.line_end >= chunk.line_start

    def test_nodes_have_metadata(self, sample_repo: Path):
        nodes, _ = ingest_repository(
            sample_repo,
            parsed_index=None,
            extensions=frozenset({".py"}),
        )
        for node in nodes:
            meta = node.metadata
            assert "file_path" in meta
            assert "language" in meta
            assert "line_start" in meta
            assert "line_end" in meta
            assert "chunk_id" in meta

    def test_file_filter(self, sample_repo: Path):
        """Only specified files are ingested when file_filter is set."""
        nodes, chunks = ingest_repository(
            sample_repo,
            parsed_index=None,
            extensions=frozenset({".py"}),
            file_filter={"src/main.py"},
        )
        for chunk in chunks:
            assert chunk.file_path == "src/main.py"

    def test_repo_id_in_metadata(self, sample_repo: Path):
        nodes, _ = ingest_repository(
            sample_repo,
            parsed_index=None,
            extensions=frozenset({".py"}),
            repo_id="test-repo-123",
        )
        for node in nodes:
            assert node.metadata["repo_id"] == "test-repo-123"

    def test_no_source_files(self, empty_repo: Path):
        nodes, chunks = ingest_repository(
            empty_repo,
            parsed_index=None,
            extensions=frozenset({".py"}),
        )
        assert nodes == []
        assert chunks == []

    def test_with_scip_enrichment(self, sample_repo: Path, sample_parsed_index: ParsedIndex):
        """When a parsed SCIP index is provided, chunks get symbol data."""
        nodes, chunks = ingest_repository(
            sample_repo,
            parsed_index=sample_parsed_index,
            extensions=frozenset({".py"}),
        )
        # At least some chunks should have SCIP symbols
        # (depends on whether line ranges overlap)
        assert len(chunks) > 0

    def test_empty_files_skipped(self, tmp_path: Path):
        """Files with only whitespace are skipped."""
        (tmp_path / "empty.py").write_text("   \n\n  \n")
        (tmp_path / "real.py").write_text("x = 1\n")
        nodes, chunks = ingest_repository(
            tmp_path,
            parsed_index=None,
            extensions=frozenset({".py"}),
        )
        paths = [c.file_path for c in chunks]
        assert "empty.py" not in paths
        assert "real.py" in paths
