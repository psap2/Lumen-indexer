"""
Lumen — centralised configuration.

All tunables live here so the rest of the codebase stays free of magic numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  

# ── Language registry ────────────────────────────────────────────────


@dataclass(frozen=True)
class LanguageProfile:
    """Everything Lumen needs to know about a supported language."""

    #: Human-readable name (used in --language flag).
    name: str
    #: File extensions to scan for.
    extensions: frozenset[str]
    #: Shell command that produces index.scip when run inside the repo.
    #: ``None`` means "no SCIP indexer available — skip SCIP".
    scip_command: Optional[list[str]]
    #: The tree-sitter grammar identifier used by CodeSplitter.
    tree_sitter_lang: str
    #: Extra install instructions shown if the SCIP command is missing.
    install_hint: str = ""


#: Master registry — add new languages here.
LANGUAGE_REGISTRY: Dict[str, LanguageProfile] = {
    "typescript": LanguageProfile(
        name="typescript",
        extensions=frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}),
        scip_command=[
            "npx", "--yes", "@sourcegraph/scip-typescript",
            "index", "--infer-tsconfig",
        ],
        tree_sitter_lang="typescript",
        install_hint="npm install -g @sourcegraph/scip-typescript  (or npx handles it)",
    ),
    "python": LanguageProfile(
        name="python",
        extensions=frozenset({".py", ".pyi"}),
        scip_command=[
            "npx", "--yes", "@sourcegraph/scip-python",
            "index", ".",
        ],
        tree_sitter_lang="python",
        install_hint="npm install -g @sourcegraph/scip-python  (requires Node >= 16)",
    ),
    "go": LanguageProfile(
        name="go",
        extensions=frozenset({".go"}),
        scip_command=["scip-go"],
        tree_sitter_lang="go",
        install_hint="go install github.com/sourcegraph/scip-go/cmd/scip-go@latest",
    ),
    "rust": LanguageProfile(
        name="rust",
        extensions=frozenset({".rs"}),
        scip_command=["rust-analyzer", "scip", "."],
        tree_sitter_lang="rust",
        install_hint="Install rust-analyzer: https://rust-analyzer.github.io/manual.html",
    ),
    "java": LanguageProfile(
        name="java",
        extensions=frozenset({".java"}),
        scip_command=["scip-java", "index"],
        tree_sitter_lang="java",
        install_hint="cs install scip-java  (requires Coursier: https://get-coursier.io)",
    ),
    "ruby": LanguageProfile(
        name="ruby",
        extensions=frozenset({".rb", ".rake", ".gemspec"}),
        scip_command=["scip-ruby"],
        tree_sitter_lang="ruby",
        install_hint="gem install scip-ruby  (requires Ruby >= 3.0)",
    ),
    "cpp": LanguageProfile(
        name="cpp",
        extensions=frozenset({".cpp", ".cc", ".cxx", ".hpp", ".h", ".c"}),
        scip_command=None,  # No turnkey SCIP indexer yet
        tree_sitter_lang="cpp",
        install_hint="No turnkey SCIP indexer for C/C++ yet — indexing proceeds without SCIP.",
    ),
}

#: Convenience aliases (so users can type --language js or --language ts)
_LANGUAGE_ALIASES: Dict[str, str] = {
    "ts": "typescript",
    "js": "typescript",
    "javascript": "typescript",
    "py": "python",
    "golang": "go",
    "rs": "rust",
    "rb": "ruby",
    "c": "cpp",
    "c++": "cpp",
}


def resolve_language(name: str) -> LanguageProfile:
    """Resolve a user-provided language string to a ``LanguageProfile``."""
    key = _LANGUAGE_ALIASES.get(name.lower(), name.lower())
    if key not in LANGUAGE_REGISTRY:
        supported = ", ".join(sorted(LANGUAGE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown language '{name}'. Supported: {supported}"
        )
    return LANGUAGE_REGISTRY[key]



# ── General SCIP settings ────────────────────────────────────────────

SCIP_INDEX_FILENAME: str = "index.scip"

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
    "target",       # Rust/Java
    "vendor",       # Go
    "Pods",         # iOS
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

# ── Supabase / Postgres ──────────────────────────────────────────────

#: Supabase project URL (used by the REST client, not DB directly).
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")

#: Supabase anonymous/service-role key.
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")

#: Full Postgres connection string.
#: When using Supabase this looks like:
#:   ``postgresql://postgres:<pw>@db.<project>.supabase.co:5432/postgres``
DATABASE_URL: str = os.environ.get("DATABASE_URL", "")

#: The Postgres table LlamaIndex will use for pgvector embeddings.
PG_EMBED_TABLE: str = "code_embeddings"


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
                "typealias", "constructor", "enum", "module", "namespace",
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
