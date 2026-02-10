"""
SCIP index parser.

Reads a binary ``index.scip`` file (protobuf-encoded) and materialises
it into lightweight Python dataclasses that the rest of the pipeline
consumes.

Design notes
────────────
* The protobuf bindings (``scip_pb2``) are generated once by
  ``python -m lumen.proto.compile``.  This module lazy-imports them and
  gives a clear error if they are missing.
* All public helpers return plain dataclasses — no protobuf types leak
  beyond this module boundary.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Lightweight domain objects ───────────────────────────────────────


@dataclass
class SymbolRelationship:
    """A directed edge from one symbol to another."""

    target_symbol: str
    is_reference: bool = False
    is_implementation: bool = False
    is_type_definition: bool = False
    is_definition: bool = False

    def to_dict(self) -> dict:
        return {
            "target_symbol": self.target_symbol,
            "is_reference": self.is_reference,
            "is_implementation": self.is_implementation,
            "is_type_definition": self.is_type_definition,
            "is_definition": self.is_definition,
        }


@dataclass
class SymbolOccurrence:
    """Where a symbol appears in a file (line-based, 0-indexed)."""

    symbol: str
    start_line: int
    start_char: int
    end_line: int
    end_char: int
    is_definition: bool = False
    is_import: bool = False
    syntax_kind: str = ""
    enclosing_start_line: Optional[int] = None
    enclosing_end_line: Optional[int] = None


@dataclass
class ParsedSymbol:
    """Rich metadata for a single symbol extracted from SCIP."""

    symbol_id: str
    kind: str = "UnspecifiedKind"
    display_name: str = ""
    documentation: str = ""
    enclosing_symbol: str = ""
    relationships: List[SymbolRelationship] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """One source file as seen by SCIP."""

    relative_path: str
    language: str
    occurrences: List[SymbolOccurrence] = field(default_factory=list)
    symbols: List[ParsedSymbol] = field(default_factory=list)
    text: str = ""


@dataclass
class ParsedIndex:
    """The entire SCIP index in memory."""

    project_root: str = ""
    tool_name: str = ""
    tool_version: str = ""
    documents: List[ParsedDocument] = field(default_factory=list)
    external_symbols: List[ParsedSymbol] = field(default_factory=list)

    # ── Derived look-ups (built by build_lookup_tables) ──────────
    symbol_table: Dict[str, ParsedSymbol] = field(default_factory=dict)
    file_symbols: Dict[str, List[ParsedSymbol]] = field(default_factory=dict)
    file_occurrences: Dict[str, List[SymbolOccurrence]] = field(default_factory=dict)

    def build_lookup_tables(self) -> None:
        """Populate convenience indexes after parsing."""
        for doc in self.documents:
            path = doc.relative_path

            # File → symbols
            self.file_symbols.setdefault(path, []).extend(doc.symbols)

            # File → occurrences
            self.file_occurrences.setdefault(path, []).extend(doc.occurrences)

            # Global symbol table
            for sym in doc.symbols:
                self.symbol_table[sym.symbol_id] = sym

        for sym in self.external_symbols:
            self.symbol_table[sym.symbol_id] = sym

    def symbols_in_range(
        self, file_path: str, start_line: int, end_line: int
    ) -> List[ParsedSymbol]:
        """Return symbols whose *definition occurrence* overlaps [start, end)."""
        hits: List[ParsedSymbol] = []
        for occ in self.file_occurrences.get(file_path, []):
            if not occ.is_definition:
                continue
            if occ.end_line < start_line or occ.start_line >= end_line:
                continue
            sym = self.symbol_table.get(occ.symbol)
            if sym:
                hits.append(sym)
        return hits

    def occurrences_in_range(
        self, file_path: str, start_line: int, end_line: int
    ) -> List[SymbolOccurrence]:
        """Return *all* occurrences that overlap [start, end)."""
        return [
            occ
            for occ in self.file_occurrences.get(file_path, [])
            if not (occ.end_line < start_line or occ.start_line >= end_line)
        ]


# ── Protobuf → dataclass conversion ─────────────────────────────────

# Symbol role bitmask constants (from scip.proto SymbolRole enum).
_ROLE_DEFINITION = 0x1
_ROLE_IMPORT = 0x2


def _load_pb2():
    """Lazy-import the compiled protobuf module."""
    try:
        from lumen.proto import scip_pb2  # type: ignore[import-untyped]

        return scip_pb2
    except ImportError:
        logger.error(
            "scip_pb2 module not found.  Run:  python -m lumen.proto.compile"
        )
        sys.exit(1)


def _kind_name(pb2, kind_value: int) -> str:
    """Resolve a SymbolInformation.Kind enum int to its name."""
    try:
        return pb2.SymbolInformation.Kind.Name(kind_value)
    except ValueError:
        return f"Unknown({kind_value})"


def _syntax_kind_name(pb2, value: int) -> str:
    try:
        return pb2.SyntaxKind.Name(value)
    except ValueError:
        return ""


def _parse_occurrence(pb2, occ) -> SymbolOccurrence:
    r = list(occ.range)
    if len(r) == 3:
        start_line, start_char, end_char = r
        end_line = start_line
    elif len(r) >= 4:
        start_line, start_char, end_line, end_char = r[:4]
    else:
        start_line = start_char = end_line = end_char = 0

    enc = list(occ.enclosing_range)
    enc_start = enc[0] if len(enc) >= 1 else None
    enc_end = enc[2] if len(enc) >= 4 else (enc[0] if len(enc) >= 1 else None)

    return SymbolOccurrence(
        symbol=occ.symbol,
        start_line=start_line,
        start_char=start_char,
        end_line=end_line,
        end_char=end_char,
        is_definition=bool(occ.symbol_roles & _ROLE_DEFINITION),
        is_import=bool(occ.symbol_roles & _ROLE_IMPORT),
        syntax_kind=_syntax_kind_name(pb2, occ.syntax_kind),
        enclosing_start_line=enc_start,
        enclosing_end_line=enc_end,
    )


def _parse_symbol(pb2, sym) -> ParsedSymbol:
    return ParsedSymbol(
        symbol_id=sym.symbol,
        kind=_kind_name(pb2, sym.kind),
        display_name=sym.display_name,
        documentation="\n".join(sym.documentation) if sym.documentation else "",
        enclosing_symbol=sym.enclosing_symbol,
        relationships=[
            SymbolRelationship(
                target_symbol=rel.symbol,
                is_reference=rel.is_reference,
                is_implementation=rel.is_implementation,
                is_type_definition=rel.is_type_definition,
                is_definition=rel.is_definition,
            )
            for rel in sym.relationships
        ],
    )


def _parse_document(pb2, doc) -> ParsedDocument:
    return ParsedDocument(
        relative_path=doc.relative_path,
        language=doc.language if doc.language else "unknown",
        occurrences=[_parse_occurrence(pb2, o) for o in doc.occurrences],
        symbols=[_parse_symbol(pb2, s) for s in doc.symbols],
        text=doc.text,
    )


# ── Public API ───────────────────────────────────────────────────────


def parse_scip_index(path: Path) -> ParsedIndex:
    """
    Read a binary SCIP index and return a fully-resolved ``ParsedIndex``.

    Parameters
    ----------
    path:
        Filesystem path to an ``index.scip`` file.

    Returns
    -------
    ParsedIndex
        Structured representation with pre-built lookup tables.
    """
    pb2 = _load_pb2()

    raw = path.read_bytes()
    index = pb2.Index()
    index.ParseFromString(raw)

    parsed = ParsedIndex(
        project_root=index.metadata.project_root if index.metadata else "",
        tool_name=(
            index.metadata.tool_info.name
            if index.metadata and index.metadata.tool_info
            else ""
        ),
        tool_version=(
            index.metadata.tool_info.version
            if index.metadata and index.metadata.tool_info
            else ""
        ),
        documents=[_parse_document(pb2, d) for d in index.documents],
        external_symbols=[_parse_symbol(pb2, s) for s in index.external_symbols],
    )

    parsed.build_lookup_tables()

    logger.info(
        "Parsed SCIP index: %d documents, %d symbols in table",
        len(parsed.documents),
        len(parsed.symbol_table),
    )
    return parsed
