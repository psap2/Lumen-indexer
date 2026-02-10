"""
Lumen — centralised configuration.

All tunables live here so the rest of the codebase stays free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ── File-type defaults ───────────────────────────────────────────────

#: Extensions we index per language.  Extend as new SCIP indexers land.
TYPESCRIPT_EXTENSIONS: frozenset[str] = frozenset(
    {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}
)

#: Files / directories to always skip during repo traversal.
DEFAULT_IGNORE_PATTERNS: list[str] = [
    "node_modules",
    ".git",
    "dist",
    "build",
    ".next",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    "coverage",
]

# ── SCIP ─────────────────────────────────────────────────────────────

#: Name of the SCIP index file produced by scip-typescript.
SCIP_INDEX_FILENAME: str = "index.scip"

#: npx command to invoke scip-typescript.
SCIP_TS_CMD: list[str] = [
    "npx",
    "--yes",
    "@sourcegraph/scip-typescript",
    "index",
    "--infer-tsconfig",
]

# ── Code splitting ───────────────────────────────────────────────────

#: Target chunk size (in lines) for the code splitter.
CHUNK_LINES: int = 60

#: Overlap between consecutive chunks (in lines).
CHUNK_OVERLAP_LINES: int = 15

#: Maximum characters per chunk — acts as a hard ceiling.
CHUNK_MAX_CHARS: int = 3000

# ── Embeddings ───────────────────────────────────────────────────────

#: HuggingFace model for local embedding generation (no API key needed).
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

#: Dimensionality of the embedding model above.
EMBEDDING_DIM: int = 384

# ── Vector store ─────────────────────────────────────────────────────

#: Default directory for ChromaDB persistent storage.
CHROMA_PERSIST_DIR: str = ".lumen/chroma_db"

#: ChromaDB collection name used by the indexer.
CHROMA_COLLECTION: str = "lumen_code_index"


# ── Standardised output schema for the Friction Scoring Engine ───────

@dataclass
class IndexedSymbol:
    """A single SCIP symbol attached to a code chunk."""

    symbol_id: str
    kind: str
    display_name: str
    documentation: str
    file_path: str
    line_start: int
    line_end: int
    relationships: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol_id": self.symbol_id,
            "kind": self.kind,
            "display_name": self.display_name,
            "documentation": self.documentation,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "relationships": self.relationships,
        }


@dataclass
class IndexedChunk:
    """
    A code chunk enriched with SCIP intelligence.

    This is the canonical interchange format consumed by the
    Friction Scoring Engine downstream.
    """

    chunk_id: str
    file_path: str
    language: str
    code: str
    line_start: int
    line_end: int
    symbols: List[IndexedSymbol] = field(default_factory=list)

    # ── Friction-scoring helpers ─────────────────────────────────
    @property
    def symbol_count(self) -> int:
        return len(self.symbols)

    @property
    def definition_count(self) -> int:
        return sum(
            1 for s in self.symbols if s.kind.lower() in {
                "function", "method", "class", "interface", "type",
                "constructor", "enum",
            }
        )

    @property
    def relationship_count(self) -> int:
        return sum(len(s.relationships) for s in self.symbols)

    @property
    def complexity_hint(self) -> float:
        """Naive complexity proxy: symbols × relationships / lines."""
        lines = max(self.line_end - self.line_start, 1)
        return round(
            (self.symbol_count + self.relationship_count) / lines, 4
        )

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "language": self.language,
            "code": self.code,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "symbols": [s.to_dict() for s in self.symbols],
            "symbol_count": self.symbol_count,
            "definition_count": self.definition_count,
            "relationship_count": self.relationship_count,
            "complexity_hint": self.complexity_hint,
        }
