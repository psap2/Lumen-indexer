"""
Tests for lumen.scip_parser.parser — dataclasses, kind inference, and lookup tables.

These tests exercise the pure-Python logic without needing a real SCIP protobuf
file or compiled scip_pb2 module.
"""

from __future__ import annotations

import pytest

from lumen.scip_parser.parser import (
    ParsedDocument,
    ParsedIndex,
    ParsedSymbol,
    SymbolOccurrence,
    SymbolRelationship,
    _infer_kind_from_docs,
    _infer_kind_from_symbol_id,
)


# ── SymbolRelationship ───────────────────────────────────────────────


class TestSymbolRelationship:
    def test_defaults(self):
        rel = SymbolRelationship(target_symbol="pkg/Foo#")
        assert rel.is_reference is False
        assert rel.is_implementation is False
        assert rel.is_type_definition is False
        assert rel.is_definition is False

    def test_to_dict(self):
        rel = SymbolRelationship(
            target_symbol="pkg/Foo#",
            is_reference=True,
            is_implementation=True,
        )
        d = rel.to_dict()
        assert d["target_symbol"] == "pkg/Foo#"
        assert d["is_reference"] is True
        assert d["is_implementation"] is True
        assert d["is_type_definition"] is False
        assert d["is_definition"] is False


# ── SymbolOccurrence ─────────────────────────────────────────────────


class TestSymbolOccurrence:
    def test_defaults(self):
        occ = SymbolOccurrence(
            symbol="sym",
            start_line=5,
            start_char=0,
            end_line=5,
            end_char=10,
        )
        assert occ.is_definition is False
        assert occ.is_import is False
        assert occ.syntax_kind == ""
        assert occ.enclosing_start_line is None
        assert occ.enclosing_end_line is None


# ── ParsedSymbol ─────────────────────────────────────────────────────


class TestParsedSymbol:
    def test_defaults(self):
        sym = ParsedSymbol(symbol_id="test/sym")
        assert sym.kind == "UnspecifiedKind"
        assert sym.display_name == ""
        assert sym.documentation == ""
        assert sym.relationships == []


# ── Kind inference from documentation ────────────────────────────────


class TestInferKindFromDocs:
    @pytest.mark.parametrize("doc,expected", [
        # TypeScript patterns in fenced code blocks
        ("```typescript\nfunction greet()\n```", "Function"),
        ("```ts\nclass UserService\n```", "Class"),
        ("```typescript\ninterface Config\n```", "Interface"),
        ("```ts\nenum Color\n```", "Enum"),
        ("```typescript\ntype UserId = string\n```", "TypeAlias"),
        ("```ts\n(method) getName()\n```", "Method"),
        ("```ts\n(property) name: string\n```", "Property"),
        ("```ts\n(parameter) id: number\n```", "Parameter"),
        ("```ts\n(enum member) RED\n```", "EnumMember"),
        ("```ts\nconstructor()\n```", "Constructor"),
        ("```ts\nnamespace Utils\n```", "Namespace"),
        ("```ts\nmodule MyMod\n```", "Module"),
        ("```ts\nconst MAX = 100\n```", "Variable"),
        ("```ts\nvar x = 1\n```", "Variable"),
        ("```ts\nlet y = 2\n```", "Variable"),
        # Python patterns
        ("```python\ndef compute()\n```", "Function"),
        ("```python\nclass Engine\n```", "Class"),
        ("```python\nasync def fetch()\n```", "Function"),
        # Python module style (unfenced)
        ("(module) apps.server.main", "Module"),
        # No match
        ("Just some text", ""),
        ("", ""),
    ])
    def test_patterns(self, doc: str, expected: str):
        assert _infer_kind_from_docs(doc) == expected


class TestInferKindFromSymbolId:
    @pytest.mark.parametrize("sym_id,expected", [
        # Function / method
        ("scip-python python pkg main.py/hello().", "Function"),
        # Class / type
        ("scip-python python pkg main.py/Config#", "Type"),
        # Property / term
        ("scip-python python pkg main.py/Config#debug.", "Property"),
        # Module
        ("scip-python python pkg main.py/", "Module"),
        # Local variable
        ("local 5", "Variable"),
        # Constructor
        ("scip-python python pkg main.py/Config#`<constructor>`().", "Constructor"),
        # Parameter
        ("scip-python python pkg main.py/hello().(name)", "Parameter"),
        # Empty
        ("", ""),
        # No recognisable suffix
        ("unknown", ""),
    ])
    def test_patterns(self, sym_id: str, expected: str):
        assert _infer_kind_from_symbol_id(sym_id) == expected


# ── ParsedIndex ──────────────────────────────────────────────────────


class TestParsedIndex:
    def test_build_lookup_tables(self, sample_parsed_index: ParsedIndex):
        idx = sample_parsed_index
        # Symbol table should contain all symbols from documents
        assert "pkg/main.py/hello()." in idx.symbol_table
        assert "pkg/main.py/Config#" in idx.symbol_table
        assert "pkg/utils.py/compute_hash()." in idx.symbol_table

    def test_file_symbols(self, sample_parsed_index: ParsedIndex):
        idx = sample_parsed_index
        assert "src/main.py" in idx.file_symbols
        assert len(idx.file_symbols["src/main.py"]) == 2  # hello + Config

    def test_file_occurrences(self, sample_parsed_index: ParsedIndex):
        idx = sample_parsed_index
        assert "src/main.py" in idx.file_occurrences
        assert len(idx.file_occurrences["src/main.py"]) == 2

    def test_symbols_in_range_hit(self, sample_parsed_index: ParsedIndex):
        idx = sample_parsed_index
        # hello is defined at line 0
        syms = idx.symbols_in_range("src/main.py", 0, 3)
        names = [s.display_name for s in syms]
        assert "hello" in names

    def test_symbols_in_range_miss(self, sample_parsed_index: ParsedIndex):
        idx = sample_parsed_index
        # Lines 100-200 should have nothing
        syms = idx.symbols_in_range("src/main.py", 100, 200)
        assert syms == []

    def test_symbols_in_range_nonexistent_file(self, sample_parsed_index: ParsedIndex):
        idx = sample_parsed_index
        syms = idx.symbols_in_range("nonexistent.py", 0, 100)
        assert syms == []

    def test_occurrences_in_range_hit(self, sample_parsed_index: ParsedIndex):
        idx = sample_parsed_index
        occs = idx.occurrences_in_range("src/main.py", 0, 1)
        assert len(occs) >= 1
        assert occs[0].symbol == "pkg/main.py/hello()."

    def test_occurrences_in_range_miss(self, sample_parsed_index: ParsedIndex):
        idx = sample_parsed_index
        occs = idx.occurrences_in_range("src/main.py", 50, 60)
        assert occs == []

    def test_occurrences_in_range_nonexistent_file(self, sample_parsed_index: ParsedIndex):
        idx = sample_parsed_index
        occs = idx.occurrences_in_range("nope.py", 0, 100)
        assert occs == []

    def test_empty_index(self):
        idx = ParsedIndex()
        idx.build_lookup_tables()
        assert idx.symbol_table == {}
        assert idx.file_symbols == {}
        assert idx.file_occurrences == {}

    def test_external_symbols_in_table(self):
        ext_sym = ParsedSymbol(
            symbol_id="external/pkg/Foo#",
            kind="Class",
            display_name="Foo",
        )
        idx = ParsedIndex(external_symbols=[ext_sym])
        idx.build_lookup_tables()
        assert "external/pkg/Foo#" in idx.symbol_table

    def test_symbols_in_range_only_definitions(self, sample_parsed_index: ParsedIndex):
        """symbols_in_range only returns symbols for definition occurrences."""
        idx = sample_parsed_index
        # Add a non-definition occurrence
        idx.file_occurrences["src/main.py"].append(
            SymbolOccurrence(
                symbol="pkg/main.py/hello().",
                start_line=10,
                start_char=0,
                end_line=10,
                end_char=5,
                is_definition=False,
            )
        )
        syms = idx.symbols_in_range("src/main.py", 10, 11)
        # Should NOT find anything because the occurrence is not a definition
        assert syms == []


# ── Merge indexes ────────────────────────────────────────────────────


class TestMergeIndexes:
    def test_merge_two_indexes(self):
        doc1 = ParsedDocument(
            relative_path="a.py", language="python",
            symbols=[ParsedSymbol(symbol_id="a/f().", kind="Function")],
        )
        doc2 = ParsedDocument(
            relative_path="b.ts", language="typescript",
            symbols=[ParsedSymbol(symbol_id="b/g().", kind="Function")],
        )

        idx1 = ParsedIndex(documents=[doc1])
        idx2 = ParsedIndex(documents=[doc2])

        merged = ParsedIndex()
        merged.documents.extend(idx1.documents)
        merged.documents.extend(idx2.documents)
        merged.build_lookup_tables()

        assert len(merged.documents) == 2
        assert "a/f()." in merged.symbol_table
        assert "b/g()." in merged.symbol_table
