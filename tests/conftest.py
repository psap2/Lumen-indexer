"""
Shared fixtures for the Lumen test suite.

Provides reusable fixtures for temporary repos, parsed indexes,
sample chunks, and mocked dependencies so individual tests stay focused.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List

import pytest

from lumen.config import IndexedChunk, IndexedSymbol, LanguageProfile
from lumen.scip_parser.parser import (
    ParsedDocument,
    ParsedIndex,
    ParsedSymbol,
    SymbolOccurrence,
    SymbolRelationship,
)


# ── Temporary repo fixtures ──────────────────────────────────────────


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a minimal multi-language repository on disk."""
    # Python files
    py_dir = tmp_path / "src"
    py_dir.mkdir()
    (py_dir / "main.py").write_text(
        textwrap.dedent("""\
        def hello(name: str) -> str:
            \"\"\"Greet someone.\"\"\"
            return f"Hello, {name}!"

        class Config:
            debug = True
            port = 8080
        """),
        encoding="utf-8",
    )
    (py_dir / "utils.py").write_text(
        textwrap.dedent("""\
        import hashlib

        def compute_hash(data: bytes) -> str:
            return hashlib.sha256(data).hexdigest()
        """),
        encoding="utf-8",
    )

    # TypeScript file
    ts_dir = tmp_path / "frontend"
    ts_dir.mkdir()
    (ts_dir / "app.ts").write_text(
        textwrap.dedent("""\
        export function greet(name: string): string {
            return `Hello, ${name}!`;
        }

        export interface User {
            id: number;
            name: string;
        }
        """),
        encoding="utf-8",
    )

    # Go file
    (tmp_path / "main.go").write_text(
        textwrap.dedent("""\
        package main

        import "fmt"

        func main() {
            fmt.Println("Hello, world!")
        }
        """),
        encoding="utf-8",
    )

    # Files that should be ignored
    node_modules = tmp_path / "node_modules" / "dep"
    node_modules.mkdir(parents=True)
    (node_modules / "index.js").write_text("module.exports = {};")

    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "main.cpython-310.pyc").write_bytes(b"\x00" * 10)

    # Non-code file
    (tmp_path / "README.md").write_text("# My Project")

    return tmp_path


@pytest.fixture
def python_only_repo(tmp_path: Path) -> Path:
    """Create a repo with only Python files."""
    (tmp_path / "app.py").write_text("def run(): pass\n")
    (tmp_path / "tests.py").write_text("def test_run(): pass\n")
    return tmp_path


@pytest.fixture
def empty_repo(tmp_path: Path) -> Path:
    """Create an empty repository (no source files)."""
    (tmp_path / "README.md").write_text("# Empty")
    return tmp_path


# ── SCIP parser fixtures ─────────────────────────────────────────────


@pytest.fixture
def sample_parsed_symbol() -> ParsedSymbol:
    """A sample parsed SCIP symbol."""
    return ParsedSymbol(
        symbol_id="scip-python python pkg main.py/hello().",
        kind="Function",
        display_name="hello",
        documentation="Greet someone.",
        enclosing_symbol="",
        relationships=[
            SymbolRelationship(
                target_symbol="scip-python python pkg main.py/Config#",
                is_reference=True,
            )
        ],
    )


@pytest.fixture
def sample_parsed_index() -> ParsedIndex:
    """Build a small in-memory ParsedIndex with lookup tables."""
    sym_hello = ParsedSymbol(
        symbol_id="pkg/main.py/hello().",
        kind="Function",
        display_name="hello",
        documentation="Greet someone.",
    )
    sym_config = ParsedSymbol(
        symbol_id="pkg/main.py/Config#",
        kind="Class",
        display_name="Config",
        documentation="App config.",
    )
    sym_hash = ParsedSymbol(
        symbol_id="pkg/utils.py/compute_hash().",
        kind="Function",
        display_name="compute_hash",
        documentation="Compute SHA-256.",
    )

    doc_main = ParsedDocument(
        relative_path="src/main.py",
        language="python",
        symbols=[sym_hello, sym_config],
        occurrences=[
            SymbolOccurrence(
                symbol="pkg/main.py/hello().",
                start_line=0,
                start_char=4,
                end_line=0,
                end_char=9,
                is_definition=True,
            ),
            SymbolOccurrence(
                symbol="pkg/main.py/Config#",
                start_line=4,
                start_char=6,
                end_line=4,
                end_char=12,
                is_definition=True,
            ),
        ],
    )
    doc_utils = ParsedDocument(
        relative_path="src/utils.py",
        language="python",
        symbols=[sym_hash],
        occurrences=[
            SymbolOccurrence(
                symbol="pkg/utils.py/compute_hash().",
                start_line=2,
                start_char=4,
                end_line=2,
                end_char=16,
                is_definition=True,
            ),
        ],
    )

    idx = ParsedIndex(
        project_root="/tmp/test",
        tool_name="scip-python",
        tool_version="1.0",
        documents=[doc_main, doc_utils],
    )
    idx.build_lookup_tables()
    return idx


# ── IndexedChunk / IndexedSymbol fixtures ────────────────────────────


@pytest.fixture
def sample_indexed_symbol() -> IndexedSymbol:
    return IndexedSymbol(
        symbol_id="pkg/main.py/hello().",
        kind="Function",
        display_name="hello",
        documentation="Greet someone.",
        file_path="src/main.py",
        line_start=0,
        line_end=3,
        relationships=[
            {"target_symbol": "pkg/main.py/Config#", "is_reference": True}
        ],
    )


@pytest.fixture
def sample_indexed_chunk(sample_indexed_symbol: IndexedSymbol) -> IndexedChunk:
    return IndexedChunk(
        chunk_id="abc123",
        file_path="src/main.py",
        language="python",
        code='def hello(name: str) -> str:\n    return f"Hello, {name}!"\n',
        line_start=0,
        line_end=3,
        symbols=[sample_indexed_symbol],
    )


# ── Language profile fixture ─────────────────────────────────────────


@pytest.fixture
def python_profile() -> LanguageProfile:
    return LanguageProfile(
        name="python",
        extensions=frozenset({".py", ".pyi"}),
        scip_command=["npx", "--yes", "@sourcegraph/scip-python", "index", "."],
        tree_sitter_lang="python",
        install_hint="npm install -g @sourcegraph/scip-python",
    )


@pytest.fixture
def typescript_profile() -> LanguageProfile:
    return LanguageProfile(
        name="typescript",
        extensions=frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}),
        scip_command=[
            "npx", "--yes", "@sourcegraph/scip-typescript",
            "index", "--infer-tsconfig",
        ],
        tree_sitter_lang="typescript",
    )


@pytest.fixture
def cpp_profile() -> LanguageProfile:
    """C/C++ profile — no SCIP command available."""
    return LanguageProfile(
        name="cpp",
        extensions=frozenset({".cpp", ".cc", ".h", ".c"}),
        scip_command=None,
        tree_sitter_lang="cpp",
        install_hint="No turnkey SCIP indexer for C/C++ yet.",
    )
