"""
Code ingestion pipeline — reads source files, splits them, and enriches
each chunk with SCIP symbol intelligence.

This module intentionally keeps tree-sitter optional: when the
``llama-index-core`` ``CodeSplitter`` is available **and** the tree-sitter
grammar for the target language is installed, we use language-aware
splitting.  Otherwise we fall back to a simple line-based splitter so
the pipeline never hard-fails on a missing native dependency.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from llama_index.core.schema import Document, TextNode

from lumen.config import (
    CHUNK_LINES,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_LINES,
    DEFAULT_IGNORE_PATTERNS,
    IndexedChunk,
    IndexedSymbol,
    LANGUAGE_REGISTRY,
)
from lumen.scip_parser.parser import ParsedIndex, ParsedSymbol

logger = logging.getLogger(__name__)


# ── Language mapping ─────────────────────────────────────────────────

#: Map file extension → tree-sitter language identifier.
#: Built dynamically from the language registry so new languages only
#: need to be added in config.py.
_EXT_TO_LANGUAGE: Dict[str, str] = {}
for _profile in LANGUAGE_REGISTRY.values():
    for _ext in _profile.extensions:
        _EXT_TO_LANGUAGE[_ext] = _profile.tree_sitter_lang

# Extras for extensions that map to a *different* tree-sitter grammar
# than their language registry entry (e.g. .js → "javascript" not "typescript")
_EXT_TO_LANGUAGE.update({
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".c": "c",
    ".h": "c",
    ".pyi": "python",
    ".cs": "c_sharp",
})


# ── Helpers ──────────────────────────────────────────────────────────


def _should_ignore(path: Path, ignore: List[str]) -> bool:
    parts = path.parts
    return any(ig in parts for ig in ignore)


def _chunk_id(file_path: str, start: int, end: int) -> str:
    raw = f"{file_path}:{start}-{end}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _collect_source_files(
    repo_root: Path,
    extensions: frozenset[str],
    ignore: List[str],
) -> List[Path]:
    """Recursively collect source files matching *extensions*."""
    files: List[Path] = []
    for p in sorted(repo_root.rglob("*")):
        if not p.is_file():
            continue
        if _should_ignore(p, ignore):
            continue
        if p.suffix.lower() in extensions:
            files.append(p)
    return files


# ── Splitters ────────────────────────────────────────────────────────


def _try_code_splitter(language: str):
    """
    Attempt to create a tree-sitter-backed ``CodeSplitter``.

    Returns ``None`` if tree-sitter or the grammar is unavailable.
    """
    try:
        from llama_index.core.node_parser import CodeSplitter  # noqa: WPS433

        splitter = CodeSplitter(
            language=language,
            chunk_lines=CHUNK_LINES,
            chunk_lines_overlap=CHUNK_OVERLAP_LINES,
            max_chars=CHUNK_MAX_CHARS,
        )
        return splitter
    except Exception as exc:  # noqa: BLE001
        logger.debug("CodeSplitter unavailable for %s: %s", language, exc)
        return None


def _line_based_split(
    text: str,
    chunk_lines: int = CHUNK_LINES,
    overlap: int = CHUNK_OVERLAP_LINES,
) -> List[tuple[int, int, str]]:
    """
    Fallback splitter: fixed-window with overlap.

    Returns a list of ``(start_line_0idx, end_line_0idx, chunk_text)``.
    """
    lines = text.splitlines(keepends=True)
    chunks: List[tuple[int, int, str]] = []
    start = 0
    while start < len(lines):
        end = min(start + chunk_lines, len(lines))
        chunk_text = "".join(lines[start:end])
        chunks.append((start, end, chunk_text))
        start += chunk_lines - overlap
    return chunks


# ── SCIP enrichment ─────────────────────────────────────────────────


def _enrich_with_scip(
    parsed_index: Optional[ParsedIndex],
    rel_path: str,
    start_line: int,
    end_line: int,
) -> List[IndexedSymbol]:
    """Attach SCIP symbol metadata to a code chunk."""
    if parsed_index is None:
        return []

    symbols: List[IndexedSymbol] = []
    for sym in parsed_index.symbols_in_range(rel_path, start_line, end_line):
        symbols.append(
            IndexedSymbol(
                symbol_id=sym.symbol_id,
                kind=sym.kind,
                display_name=sym.display_name,
                documentation=sym.documentation,
                file_path=rel_path,
                line_start=start_line,
                line_end=end_line,
                relationships=[r.to_dict() for r in sym.relationships],
            )
        )
    return symbols


def _symbol_metadata_text(symbols: List[IndexedSymbol]) -> str:
    """
    Render SCIP symbols into a human-readable header that is prepended
    to the chunk text before embedding.  This gives the embedding model
    semantic context it would otherwise lack.
    """
    if not symbols:
        return ""

    parts = ["[SCIP Symbols]"]
    for s in symbols:
        line = f"  • {s.kind}: {s.display_name or s.symbol_id}"
        if s.documentation:
            # Take first 200 chars of docs to stay concise.
            doc_preview = s.documentation[:200].replace("\n", " ")
            line += f"  — {doc_preview}"
        if s.relationships:
            rel_strs = []
            for r in s.relationships:
                edges = []
                if r.get("is_implementation"):
                    edges.append("implements")
                if r.get("is_reference"):
                    edges.append("refs")
                if r.get("is_type_definition"):
                    edges.append("type-def")
                if edges:
                    rel_strs.append(
                        f"{','.join(edges)} → {r['target_symbol'][:80]}"
                    )
            if rel_strs:
                line += f"  [{'; '.join(rel_strs)}]"
        parts.append(line)
    return "\n".join(parts) + "\n\n"


# ── Public API ───────────────────────────────────────────────────────


def ingest_repository(
    repo_root: Path,
    parsed_index: Optional[ParsedIndex] = None,
    extensions: Optional[frozenset[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    file_filter: Optional[Set[str]] = None,
    repo_id: Optional[str] = None,
) -> tuple[List[TextNode], List[IndexedChunk]]:
    """
    Walk a repository, split source files, and produce LlamaIndex
    ``TextNode`` objects enriched with SCIP symbol data.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository root.
    parsed_index:
        Pre-parsed SCIP index (``None`` to skip SCIP enrichment).
    extensions:
        File extensions to include.
    ignore_patterns:
        Directory / file names to skip.
    file_filter:
        When provided, only files whose *relative path* is in this set
        will be ingested.  Used by the incremental indexing pipeline to
        restrict processing to new and modified files.
    repo_id:
        String UUID of the repository.  Stored in each ``TextNode``'s
        metadata so downstream queries can filter by repository.

    Returns
    -------
    (nodes, chunks)
        *nodes* are ready for vector-store ingestion; *chunks* are the
        standardised ``IndexedChunk`` objects for the Friction Scoring
        Engine.
    """
    if extensions is None:
        # Default: combine all known extensions from every registered language.
        extensions = frozenset().union(
            *(p.extensions for p in LANGUAGE_REGISTRY.values())
        )

    if ignore_patterns is None:
        ignore_patterns = DEFAULT_IGNORE_PATTERNS

    source_files = _collect_source_files(repo_root, extensions, ignore_patterns)

    # Apply incremental file filter if provided
    if file_filter is not None:
        source_files = [
            f for f in source_files
            if str(f.relative_to(repo_root)) in file_filter
        ]

    logger.info("Found %d source files in %s", len(source_files), repo_root)

    nodes: List[TextNode] = []
    chunks: List[IndexedChunk] = []

    for fpath in source_files:
        rel_path = str(fpath.relative_to(repo_root))
        lang = _EXT_TO_LANGUAGE.get(fpath.suffix.lower(), "text")

        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Skipping %s: %s", rel_path, exc)
            continue

        if not text.strip():
            continue

        # ── Split ────────────────────────────────────────────────
        raw_chunks: List[tuple[int, int, str]] = []
        splitter = _try_code_splitter(lang)

        if splitter is not None:
            try:
                doc = Document(text=text, metadata={"file_path": rel_path})
                split_nodes = splitter.get_nodes_from_documents([doc])
                # CodeSplitter nodes carry the text; we need to recover
                # line offsets for SCIP correlation.
                offset = 0
                for sn in split_nodes:
                    chunk_text = sn.get_content()
                    start_line = text.count("\n", 0, text.find(chunk_text, offset))
                    end_line = start_line + chunk_text.count("\n") + 1
                    offset = text.find(chunk_text, offset) + len(chunk_text)
                    raw_chunks.append((start_line, end_line, chunk_text))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CodeSplitter failed for %s, falling back: %s",
                    rel_path,
                    exc,
                )
                raw_chunks = _line_based_split(text)
        else:
            raw_chunks = _line_based_split(text)

        # ── Enrich & emit ────────────────────────────────────────
        for start_line, end_line, chunk_text in raw_chunks:
            cid = _chunk_id(rel_path, start_line, end_line)

            scip_symbols = _enrich_with_scip(
                parsed_index, rel_path, start_line, end_line
            )
            symbol_header = _symbol_metadata_text(scip_symbols)

            # The embedded text includes the symbol header so the
            # vector representation captures semantic relationships.
            enriched_text = f"{symbol_header}{chunk_text}"

            node = TextNode(
                text=enriched_text,
                id_=cid,
                metadata={
                    "repo_id": repo_id or "",
                    "file_path": rel_path,
                    "language": lang,
                    "line_start": start_line,
                    "line_end": end_line,
                    "symbol_count": len(scip_symbols),
                    "definition_count": sum(
                        1
                        for s in scip_symbols
                        if s.kind.lower()
                        in {
                            "function",
                            "method",
                            "class",
                            "interface",
                            "type",
                            "constructor",
                            "enum",
                        }
                    ),
                    "chunk_id": cid,
                },
                excluded_embed_metadata_keys=["chunk_id", "repo_id"],
                excluded_llm_metadata_keys=["chunk_id", "repo_id"],
            )
            nodes.append(node)

            indexed_chunk = IndexedChunk(
                chunk_id=cid,
                file_path=rel_path,
                language=lang,
                code=chunk_text,
                line_start=start_line,
                line_end=end_line,
                symbols=scip_symbols,
            )
            chunks.append(indexed_chunk)

    logger.info(
        "Ingestion complete: %d nodes from %d files", len(nodes), len(source_files)
    )
    return nodes, chunks
