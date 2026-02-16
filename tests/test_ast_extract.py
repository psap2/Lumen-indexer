"""
Tests for lumen.ingestion.ast_extract — tree-sitter AST metadata extraction.

Verifies that signatures, imports, complexity, and node types are
extracted correctly for Python, TypeScript, and Go.
"""

from __future__ import annotations

import textwrap

import pytest

from lumen.config import ASTMetadata
from lumen.ingestion.ast_extract import (
    ast_metadata_text,
    extract_ast_metadata,
)


# ── Python ───────────────────────────────────────────────────────────


class TestPythonExtraction:
    """AST metadata extraction for Python code."""

    SAMPLE = textwrap.dedent("""\
        import os
        from pathlib import Path

        class UserService:
            \"\"\"Manages user lifecycle.\"\"\"

            def get_user(self, user_id: int) -> dict:
                if user_id <= 0:
                    raise ValueError("bad id")
                return {"id": user_id}

            def delete_user(self, user_id: int) -> bool:
                for hook in self._hooks:
                    hook(user_id)
                return True

        def standalone_function(x: int, y: int) -> int:
            while x > 0:
                x -= 1
            return x + y
    """)

    def test_signatures_extracted(self):
        meta = extract_ast_metadata(self.SAMPLE, "python")
        assert meta is not None
        sigs = meta.signatures
        # Should find the class, its two methods, and the standalone function
        assert any("UserService" in s for s in sigs), f"Missing class sig in {sigs}"
        assert any("get_user" in s for s in sigs), f"Missing get_user in {sigs}"
        assert any("delete_user" in s for s in sigs), f"Missing delete_user in {sigs}"
        assert any("standalone_function" in s for s in sigs), f"Missing standalone_function in {sigs}"

    def test_imports_extracted(self):
        meta = extract_ast_metadata(self.SAMPLE, "python")
        assert meta is not None
        assert len(meta.imports) == 2
        assert any("import os" in i for i in meta.imports)
        assert any("from pathlib import Path" in i for i in meta.imports)

    def test_complexity_counts_branches(self):
        meta = extract_ast_metadata(self.SAMPLE, "python")
        assert meta is not None
        # 1 if + 1 for + 1 while = 3
        assert meta.complexity >= 3, f"Expected >= 3 branches, got {meta.complexity}"

    def test_node_types(self):
        meta = extract_ast_metadata(self.SAMPLE, "python")
        assert meta is not None
        assert "import_statement" in meta.node_types
        assert "import_from_statement" in meta.node_types
        assert "class_definition" in meta.node_types
        assert "function_definition" in meta.node_types

    def test_nesting_depth(self):
        meta = extract_ast_metadata(self.SAMPLE, "python")
        assert meta is not None
        assert meta.nesting_depth >= 2

    def test_to_dict_roundtrip(self):
        meta = extract_ast_metadata(self.SAMPLE, "python")
        assert meta is not None
        d = meta.to_dict()
        assert isinstance(d["signatures"], list)
        assert isinstance(d["imports"], list)
        assert isinstance(d["complexity"], int)
        assert isinstance(d["nesting_depth"], int)
        assert isinstance(d["node_types"], list)


class TestPythonDecorated:
    """Ensure decorated functions/classes are captured."""

    SAMPLE = textwrap.dedent("""\
        from functools import lru_cache

        @lru_cache(maxsize=128)
        def expensive_compute(n: int) -> int:
            return n * n
    """)

    def test_decorated_function_found(self):
        meta = extract_ast_metadata(self.SAMPLE, "python")
        assert meta is not None
        assert any("expensive_compute" in s for s in meta.signatures)


# ── TypeScript ───────────────────────────────────────────────────────


class TestTypeScriptExtraction:
    """AST metadata extraction for TypeScript code."""

    SAMPLE = textwrap.dedent("""\
        import { Request, Response } from 'express';

        interface UserPayload {
            id: number;
            name: string;
        }

        export class UserController {
            getUser(req: Request, res: Response): void {
                if (!req.params.id) {
                    res.status(400).send('Missing id');
                    return;
                }
                try {
                    const user = this.findUser(req.params.id);
                    res.json(user);
                } catch (e) {
                    res.status(500).send('Error');
                }
            }
        }

        export function healthCheck(): string {
            return 'ok';
        }
    """)

    def test_signatures_extracted(self):
        meta = extract_ast_metadata(self.SAMPLE, "typescript")
        assert meta is not None
        sigs = meta.signatures
        assert any("UserController" in s for s in sigs), f"Missing class in {sigs}"
        assert any("healthCheck" in s for s in sigs), f"Missing function in {sigs}"

    def test_imports_extracted(self):
        meta = extract_ast_metadata(self.SAMPLE, "typescript")
        assert meta is not None
        assert len(meta.imports) >= 1
        assert any("express" in i for i in meta.imports)

    def test_complexity(self):
        meta = extract_ast_metadata(self.SAMPLE, "typescript")
        assert meta is not None
        # if + try = at least 2 branch constructs
        assert meta.complexity >= 2

    def test_interface_detected(self):
        meta = extract_ast_metadata(self.SAMPLE, "typescript")
        assert meta is not None
        # interface should appear as a type signature or node type
        sigs = meta.signatures
        assert any("UserPayload" in s for s in sigs), f"Missing interface in {sigs}"


# ── Go ───────────────────────────────────────────────────────────────


class TestGoExtraction:
    """AST metadata extraction for Go code."""

    SAMPLE = textwrap.dedent("""\
        package main

        import (
            "fmt"
            "net/http"
        )

        type UserService struct {
            db *Database
        }

        func (s *UserService) GetUser(id int) (*User, error) {
            if id <= 0 {
                return nil, fmt.Errorf("invalid id: %d", id)
            }
            for _, u := range s.cache {
                if u.ID == id {
                    return u, nil
                }
            }
            return s.db.FindByID(id)
        }

        func NewUserService(db *Database) *UserService {
            return &UserService{db: db}
        }
    """)

    def test_signatures_extracted(self):
        meta = extract_ast_metadata(self.SAMPLE, "go")
        assert meta is not None
        sigs = meta.signatures
        # Method: GetUser, Function: NewUserService, Type: UserService
        assert any("GetUser" in s for s in sigs), f"Missing method in {sigs}"
        assert any("NewUserService" in s for s in sigs), f"Missing function in {sigs}"

    def test_imports_extracted(self):
        meta = extract_ast_metadata(self.SAMPLE, "go")
        assert meta is not None
        assert len(meta.imports) >= 1
        assert any("fmt" in i for i in meta.imports)

    def test_complexity(self):
        meta = extract_ast_metadata(self.SAMPLE, "go")
        assert meta is not None
        # 2 if + 1 for = at least 3
        assert meta.complexity >= 3

    def test_type_declaration(self):
        meta = extract_ast_metadata(self.SAMPLE, "go")
        assert meta is not None
        sigs = meta.signatures
        assert any("UserService" in s for s in sigs), f"Missing struct in {sigs}"


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_code(self):
        meta = extract_ast_metadata("", "python")
        assert meta is not None
        assert meta.signatures == []
        assert meta.imports == []
        assert meta.complexity == 0

    def test_unsupported_language(self):
        meta = extract_ast_metadata("SELECT 1;", "sql")
        # sql is not in _LANG_CONFIG → should return None
        assert meta is None

    def test_unknown_language(self):
        meta = extract_ast_metadata("hello", "klingon")
        assert meta is None

    def test_plain_text_language(self):
        meta = extract_ast_metadata("just some text", "text")
        assert meta is None

    def test_single_function(self):
        code = "def hello():\n    print('hi')\n"
        meta = extract_ast_metadata(code, "python")
        assert meta is not None
        assert len(meta.signatures) == 1
        assert "hello" in meta.signatures[0]

    def test_comments_only(self):
        code = "# Just a comment\n# Another comment\n"
        meta = extract_ast_metadata(code, "python")
        assert meta is not None
        assert meta.signatures == []
        assert meta.complexity == 0


# ── ast_metadata_text rendering ──────────────────────────────────────


class TestASTMetadataText:
    """Test the text renderer used for embedding enrichment."""

    def test_none_returns_empty(self):
        assert ast_metadata_text(None) == ""

    def test_empty_metadata_returns_empty(self):
        meta = ASTMetadata()
        assert ast_metadata_text(meta) == ""

    def test_signatures_rendered(self):
        meta = ASTMetadata(
            signatures=["def get_user(id: int) -> User"],
            complexity=2,
        )
        text = ast_metadata_text(meta)
        assert "[AST Context]" in text
        assert "get_user" in text
        assert "complexity: 2" in text

    def test_imports_rendered(self):
        meta = ASTMetadata(
            imports=["import os", "from pathlib import Path"],
            signatures=["def main()"],
        )
        text = ast_metadata_text(meta)
        assert "import: import os" in text
        assert "import: from pathlib import Path" in text

    def test_header_ends_with_double_newline(self):
        meta = ASTMetadata(signatures=["def foo()"])
        text = ast_metadata_text(meta)
        assert text.endswith("\n\n")
