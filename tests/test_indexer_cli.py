"""
Tests for lumen.indexer — CLI argument parsing and orchestration logic.

Tests the argument parser and main entry-point routing, mocking out
heavy operations (SCIP, embedding, database).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lumen.indexer import (
    _build_parser,
    _ensure_proto_compiled,
    _merge_parsed_indexes,
    _should_ignore,
    detect_languages,
    main,
)
from lumen.scip_parser.parser import ParsedDocument, ParsedIndex, ParsedSymbol


# ── _build_parser ────────────────────────────────────────────────────


class TestBuildParser:
    def test_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["/path/to/repo"])
        assert args.repo_path == "/path/to/repo"
        assert args.language == "auto"
        assert args.skip_scip is False
        assert args.query_only is False
        assert args.question is None
        assert args.export_chunks is None
        assert args.incremental is False
        assert args.verbose is False
        assert args.git_url is None
        assert args.branch is None

    def test_language_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["/repo", "--language", "python"])
        assert args.language == "python"

    def test_language_short_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["/repo", "-l", "typescript"])
        assert args.language == "typescript"

    def test_skip_scip(self):
        parser = _build_parser()
        args = parser.parse_args(["/repo", "--skip-scip"])
        assert args.skip_scip is True

    def test_query_only(self):
        parser = _build_parser()
        args = parser.parse_args(["/repo", "--query-only"])
        assert args.query_only is True

    def test_question(self):
        parser = _build_parser()
        args = parser.parse_args(["/repo", "-q", "What does foo do?"])
        assert args.question == "What does foo do?"

    def test_export_chunks(self):
        parser = _build_parser()
        args = parser.parse_args(["/repo", "--export-chunks", "out.json"])
        assert args.export_chunks == "out.json"

    def test_incremental(self):
        parser = _build_parser()
        args = parser.parse_args(["/repo", "--incremental"])
        assert args.incremental is True

    def test_incremental_short(self):
        parser = _build_parser()
        args = parser.parse_args(["/repo", "-i"])
        assert args.incremental is True

    def test_verbose(self):
        parser = _build_parser()
        args = parser.parse_args(["/repo", "--verbose"])
        assert args.verbose is True

    def test_git_url(self):
        parser = _build_parser()
        args = parser.parse_args(["--git-url", "https://github.com/org/repo.git"])
        assert args.git_url == "https://github.com/org/repo.git"
        assert args.repo_path is None

    def test_git_url_with_branch(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--git-url", "https://github.com/org/repo.git",
            "--branch", "develop",
        ])
        assert args.branch == "develop"

    def test_no_args_allowed(self):
        """Parser allows no positional arg when --git-url is used."""
        parser = _build_parser()
        args = parser.parse_args(["--git-url", "https://github.com/org/repo.git"])
        assert args.repo_path is None


# ── _should_ignore ───────────────────────────────────────────────────


class TestShouldIgnore:
    def test_node_modules(self, tmp_path: Path):
        p = tmp_path / "node_modules" / "pkg"
        assert _should_ignore(p) is True

    def test_normal_path(self, tmp_path: Path):
        p = tmp_path / "src" / "main.py"
        assert _should_ignore(p) is False


# ── _merge_parsed_indexes ────────────────────────────────────────────


class TestMergeParsedIndexes:
    def test_single_index_returns_same(self):
        idx = ParsedIndex(documents=[
            ParsedDocument(relative_path="a.py", language="python"),
        ])
        result = _merge_parsed_indexes([idx])
        assert result is idx

    def test_merge_two_indexes(self):
        idx1 = ParsedIndex(documents=[
            ParsedDocument(
                relative_path="a.py",
                language="python",
                symbols=[ParsedSymbol(symbol_id="a/f().")],
            ),
        ])
        idx2 = ParsedIndex(documents=[
            ParsedDocument(
                relative_path="b.ts",
                language="typescript",
                symbols=[ParsedSymbol(symbol_id="b/g().")],
            ),
        ])
        merged = _merge_parsed_indexes([idx1, idx2])
        assert len(merged.documents) == 2
        assert "a/f()." in merged.symbol_table
        assert "b/g()." in merged.symbol_table

    def test_merge_preserves_external_symbols(self):
        ext = ParsedSymbol(symbol_id="ext/Foo#", kind="Class")
        idx1 = ParsedIndex(external_symbols=[ext])
        idx2 = ParsedIndex()
        merged = _merge_parsed_indexes([idx1, idx2])
        assert "ext/Foo#" in merged.symbol_table


# ── _ensure_proto_compiled ───────────────────────────────────────────


class TestEnsureProtoCompiled:
    @patch("lumen.indexer.Path.exists", return_value=True)
    def test_skips_when_pb2_exists(self, mock_exists):
        """Should not call compile_proto when scip_pb2.py exists."""
        with patch("lumen.indexer.Path.resolve", return_value=Path("/fake")):
            # This should not raise or try to compile
            _ensure_proto_compiled()

    @patch("lumen.indexer.Path.exists", return_value=False)
    @patch("lumen.proto.compile.compile_proto")
    def test_compiles_when_missing(self, mock_compile, mock_exists):
        _ensure_proto_compiled()
        mock_compile.assert_called_once()


# ── main() entry point ──────────────────────────────────────────────


class TestMain:
    def test_no_args_exits(self):
        """main() with no args should exit with error."""
        with pytest.raises(SystemExit):
            main([])

    def test_both_sources_exits(self, tmp_path: Path):
        """Providing both repo_path and --git-url should exit."""
        with pytest.raises(SystemExit):
            main([str(tmp_path), "--git-url", "https://github.com/org/repo.git"])

    def test_nonexistent_path_exits(self):
        """A non-existent local path should exit."""
        with pytest.raises(SystemExit):
            main(["/nonexistent/path/to/repo"])

    @patch("lumen.indexer.run_repl")
    @patch("lumen.indexer.run_query")
    def test_query_only_with_question(self, mock_query, mock_repl, tmp_path: Path):
        """--query-only with --question should call run_query."""
        main([str(tmp_path), "--query-only", "-q", "What is foo?"])
        mock_query.assert_called_once_with("What is foo?")
        mock_repl.assert_not_called()

    @patch("lumen.indexer.run_repl")
    def test_query_only_without_question(self, mock_repl, tmp_path: Path):
        """--query-only without --question should start REPL."""
        main([str(tmp_path), "--query-only"])
        mock_repl.assert_called_once()
