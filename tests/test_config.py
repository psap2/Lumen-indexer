"""
Tests for lumen.config — data structures, language registry, and constants.
"""

from __future__ import annotations

import pytest

from lumen.config import (
    CHUNK_LINES,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_LINES,
    DEFAULT_IGNORE_PATTERNS,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LANGUAGE_REGISTRY,
    SCIP_INDEX_FILENAME,
    IndexedChunk,
    IndexedSymbol,
    LanguageProfile,
    resolve_language,
)


# ── LanguageProfile ──────────────────────────────────────────────────


class TestLanguageProfile:
    def test_frozen(self, python_profile: LanguageProfile):
        """LanguageProfile is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            python_profile.name = "java"  # type: ignore[misc]

    def test_fields(self, python_profile: LanguageProfile):
        assert python_profile.name == "python"
        assert ".py" in python_profile.extensions
        assert ".pyi" in python_profile.extensions
        assert python_profile.tree_sitter_lang == "python"
        assert python_profile.scip_command is not None

    def test_no_scip_command(self, cpp_profile: LanguageProfile):
        """C/C++ has no SCIP indexer."""
        assert cpp_profile.scip_command is None
        assert cpp_profile.install_hint  # should have a hint


# ── LANGUAGE_REGISTRY ────────────────────────────────────────────────


class TestLanguageRegistry:
    def test_known_languages(self):
        expected = {"typescript", "python", "go", "rust", "java", "ruby", "cpp"}
        assert set(LANGUAGE_REGISTRY.keys()) == expected

    def test_all_have_extensions(self):
        for key, profile in LANGUAGE_REGISTRY.items():
            assert len(profile.extensions) > 0, f"{key} has no extensions"

    def test_all_have_tree_sitter_lang(self):
        for key, profile in LANGUAGE_REGISTRY.items():
            assert profile.tree_sitter_lang, f"{key} missing tree_sitter_lang"

    def test_extensions_are_frozensets(self):
        for key, profile in LANGUAGE_REGISTRY.items():
            assert isinstance(profile.extensions, frozenset), (
                f"{key} extensions should be frozenset"
            )

    def test_extensions_start_with_dot(self):
        for key, profile in LANGUAGE_REGISTRY.items():
            for ext in profile.extensions:
                assert ext.startswith("."), (
                    f"{key} extension '{ext}' should start with '.'"
                )


# ── resolve_language ─────────────────────────────────────────────────


class TestResolveLanguage:
    def test_exact_key(self):
        profile = resolve_language("python")
        assert profile.name == "python"

    def test_alias_py(self):
        profile = resolve_language("py")
        assert profile.name == "python"

    def test_alias_ts(self):
        profile = resolve_language("ts")
        assert profile.name == "typescript"

    def test_alias_js(self):
        profile = resolve_language("js")
        assert profile.name == "typescript"

    def test_alias_javascript(self):
        profile = resolve_language("javascript")
        assert profile.name == "typescript"

    def test_alias_golang(self):
        profile = resolve_language("golang")
        assert profile.name == "go"

    def test_alias_rs(self):
        profile = resolve_language("rs")
        assert profile.name == "rust"

    def test_alias_c(self):
        profile = resolve_language("c")
        assert profile.name == "cpp"

    def test_alias_cpp(self):
        profile = resolve_language("c++")
        assert profile.name == "cpp"

    def test_case_insensitive(self):
        profile = resolve_language("Python")
        assert profile.name == "python"

    def test_unknown_language_raises(self):
        with pytest.raises(ValueError, match="Unknown language"):
            resolve_language("fortran")

    def test_error_message_lists_supported(self):
        with pytest.raises(ValueError, match="Supported:"):
            resolve_language("brainfuck")


# ── IndexedSymbol ────────────────────────────────────────────────────


class TestIndexedSymbol:
    def test_fields(self, sample_indexed_symbol: IndexedSymbol):
        sym = sample_indexed_symbol
        assert sym.symbol_id == "pkg/main.py/hello()."
        assert sym.kind == "Function"
        assert sym.display_name == "hello"
        assert sym.file_path == "src/main.py"
        assert sym.line_start == 0
        assert sym.line_end == 3

    def test_to_dict(self, sample_indexed_symbol: IndexedSymbol):
        d = sample_indexed_symbol.to_dict()
        assert d["symbol_id"] == "pkg/main.py/hello()."
        assert d["kind"] == "Function"
        assert d["display_name"] == "hello"
        assert isinstance(d["relationships"], list)
        assert len(d["relationships"]) == 1

    def test_default_relationships(self):
        sym = IndexedSymbol(
            symbol_id="x",
            kind="Variable",
            display_name="x",
            documentation="",
            file_path="a.py",
            line_start=0,
            line_end=1,
        )
        assert sym.relationships == []


# ── IndexedChunk ─────────────────────────────────────────────────────


class TestIndexedChunk:
    def test_symbol_count(self, sample_indexed_chunk: IndexedChunk):
        assert sample_indexed_chunk.symbol_count == 1

    def test_definition_count(self, sample_indexed_chunk: IndexedChunk):
        # "Function" is a definition kind
        assert sample_indexed_chunk.definition_count == 1

    def test_relationship_count(self, sample_indexed_chunk: IndexedChunk):
        assert sample_indexed_chunk.relationship_count == 1

    def test_complexity_hint(self, sample_indexed_chunk: IndexedChunk):
        chunk = sample_indexed_chunk
        lines = max(chunk.line_end - chunk.line_start, 1)
        expected = round((chunk.symbol_count + chunk.relationship_count) / lines, 4)
        assert chunk.complexity_hint == expected

    def test_complexity_hint_zero_lines(self):
        """When line_start == line_end, complexity_hint should not divide by zero."""
        chunk = IndexedChunk(
            chunk_id="z",
            file_path="a.py",
            language="python",
            code="x = 1",
            line_start=5,
            line_end=5,
        )
        # lines = max(5 - 5, 1) = 1, so no ZeroDivisionError
        assert chunk.complexity_hint == 0.0

    def test_to_dict(self, sample_indexed_chunk: IndexedChunk):
        d = sample_indexed_chunk.to_dict()
        assert d["chunk_id"] == "abc123"
        assert d["file_path"] == "src/main.py"
        assert d["language"] == "python"
        assert d["symbol_count"] == 1
        assert d["definition_count"] == 1
        assert d["relationship_count"] == 1
        assert "complexity_hint" in d
        assert isinstance(d["symbols"], list)
        assert len(d["symbols"]) == 1

    def test_to_dict_empty_symbols(self):
        chunk = IndexedChunk(
            chunk_id="empty",
            file_path="test.py",
            language="python",
            code="pass",
            line_start=0,
            line_end=1,
        )
        d = chunk.to_dict()
        assert d["symbols"] == []
        assert d["symbol_count"] == 0
        assert d["definition_count"] == 0

    def test_definition_count_multiple_kinds(self):
        """Only specific kinds count as definitions."""
        symbols = [
            IndexedSymbol(
                symbol_id="a", kind="Function", display_name="f",
                documentation="", file_path="x.py", line_start=0, line_end=1,
            ),
            IndexedSymbol(
                symbol_id="b", kind="Class", display_name="C",
                documentation="", file_path="x.py", line_start=0, line_end=1,
            ),
            IndexedSymbol(
                symbol_id="c", kind="Variable", display_name="v",
                documentation="", file_path="x.py", line_start=0, line_end=1,
            ),
            IndexedSymbol(
                symbol_id="d", kind="Method", display_name="m",
                documentation="", file_path="x.py", line_start=0, line_end=1,
            ),
            IndexedSymbol(
                symbol_id="e", kind="Property", display_name="p",
                documentation="", file_path="x.py", line_start=0, line_end=1,
            ),
        ]
        chunk = IndexedChunk(
            chunk_id="multi",
            file_path="x.py",
            language="python",
            code="...",
            line_start=0,
            line_end=10,
            symbols=symbols,
        )
        # Function, Class, Method are definitions; Variable, Property are not
        assert chunk.definition_count == 3


# ── Constants ────────────────────────────────────────────────────────


class TestConstants:
    def test_scip_index_filename(self):
        assert SCIP_INDEX_FILENAME == "index.scip"

    def test_chunk_lines_positive(self):
        assert CHUNK_LINES > 0

    def test_chunk_overlap_less_than_chunk(self):
        assert CHUNK_OVERLAP_LINES < CHUNK_LINES

    def test_chunk_max_chars_positive(self):
        assert CHUNK_MAX_CHARS > 0

    def test_embedding_model_set(self):
        assert EMBEDDING_MODEL  # non-empty

    def test_embedding_dim_positive(self):
        assert EMBEDDING_DIM > 0

    def test_default_ignore_patterns(self):
        assert "node_modules" in DEFAULT_IGNORE_PATTERNS
        assert ".git" in DEFAULT_IGNORE_PATTERNS
        assert "__pycache__" in DEFAULT_IGNORE_PATTERNS
