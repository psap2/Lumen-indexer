#!/usr/bin/env python3
"""
Lumen — Repository Indexing Service.

CLI entry-point that orchestrates the full pipeline:

    1. Run the SCIP indexer (scip-typescript) against a local repo.
    2. Parse the resulting ``index.scip`` protobuf.
    3. Walk the repo, split code, and enrich chunks with SCIP symbols.
    4. Embed and persist to ChromaDB.
    5. (Optional) Run an interactive query REPL.

Usage
─────
    # Full index + query
    python -m lumen.indexer /path/to/repo

    # Skip SCIP generation (if index.scip already exists)
    python -m lumen.indexer /path/to/repo --skip-scip

    # Query-only mode (index must already exist)
    python -m lumen.indexer /path/to/repo --query-only

    # Ask a single question
    python -m lumen.indexer /path/to/repo --question "What does parseConfig do?"
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from lumen.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    SCIP_INDEX_FILENAME,
    SCIP_TS_CMD,
    TYPESCRIPT_EXTENSIONS,
)

logger = logging.getLogger("lumen")


# ── Step 0: Proto compilation guard ─────────────────────────────────


def _ensure_proto_compiled() -> None:
    """Compile scip.proto → scip_pb2.py if the latter is missing."""
    proto_dir = Path(__file__).resolve().parent / "proto"
    pb2 = proto_dir / "scip_pb2.py"
    if pb2.exists():
        return

    logger.info("scip_pb2.py not found — compiling protobuf …")
    from lumen.proto.compile import compile_proto

    compile_proto()


# ── Step 1: SCIP indexer execution ───────────────────────────────────


def run_scip_indexer(repo_root: Path) -> Path:
    """
    Invoke ``scip-typescript`` and return the path to the generated
    ``index.scip`` file.

    Raises
    ------
    RuntimeError
        If the indexer process exits with a non-zero code.
    FileNotFoundError
        If the output file was not created.
    """
    output_path = repo_root / SCIP_INDEX_FILENAME
    logger.info("Running SCIP indexer in %s …", repo_root)
    logger.info("  Command: %s", " ".join(SCIP_TS_CMD))

    try:
        result = subprocess.run(
            SCIP_TS_CMD,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "npx not found.  Make sure Node.js (>= 16) is on your PATH.\n"
            "  brew install node   # macOS\n"
            "  sudo apt install nodejs npm   # Debian/Ubuntu"
        ) from None

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"scip-typescript exited with code {result.returncode}.\n"
            f"stderr:\n{stderr}"
        )

    if result.stdout.strip():
        logger.debug("scip-typescript stdout:\n%s", result.stdout.strip())

    if not output_path.exists():
        raise FileNotFoundError(
            f"SCIP indexer completed but {output_path} was not created."
        )

    size_kb = output_path.stat().st_size / 1024
    logger.info("Generated %s (%.1f KB)", output_path, size_kb)
    return output_path


# ── Step 2: Parse SCIP ───────────────────────────────────────────────


def parse_scip(scip_path: Path):
    from lumen.scip_parser.parser import parse_scip_index

    return parse_scip_index(scip_path)


# ── Step 3: Ingest ───────────────────────────────────────────────────


def ingest(repo_root: Path, parsed_index):
    from lumen.ingestion.code_ingestor import ingest_repository

    nodes, chunks = ingest_repository(
        repo_root,
        parsed_index=parsed_index,
        extensions=TYPESCRIPT_EXTENSIONS,
    )
    return nodes, chunks


# ── Step 4: Embed + persist ──────────────────────────────────────────


def embed_and_persist(nodes, persist_dir: str, collection_name: str):
    from lumen.storage.vector_store import build_index

    return build_index(
        nodes,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )


# ── Step 5: Query ────────────────────────────────────────────────────


def run_query(question: str, persist_dir: str, collection_name: str) -> None:
    from lumen.query.engine import query_index

    results = query_index(
        question,
        persist_dir=persist_dir,
        collection_name=collection_name,
        top_k=5,
    )

    if not results:
        print("No results found.")
        return

    print(f"\n{'─' * 60}")
    print(f"Query: {question}")
    print(f"{'─' * 60}\n")

    for i, r in enumerate(results, 1):
        print(f"  Result {i}  {r.summary()}")
        preview_lines = r.text.splitlines()[:15]
        for line in preview_lines:
            print(f"    {line}")
        if len(r.text.splitlines()) > 15:
            print("    …")
        print()


def run_repl(persist_dir: str, collection_name: str) -> None:
    from lumen.query.engine import interactive_repl

    interactive_repl(persist_dir=persist_dir, collection_name=collection_name)


# ── CLI ──────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lumen.indexer",
        description="Lumen Repository Indexing Service — SCIP + LlamaIndex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m lumen.indexer ./my-ts-project\n"
            "  python -m lumen.indexer ./my-ts-project --skip-scip\n"
            "  python -m lumen.indexer ./my-ts-project --query-only\n"
            '  python -m lumen.indexer ./my-ts-project --question "What does parseConfig do?"'
        ),
    )
    p.add_argument(
        "repo_path",
        type=str,
        help="Path to the local repository to index.",
    )
    p.add_argument(
        "--skip-scip",
        action="store_true",
        default=False,
        help="Skip running the SCIP indexer (expects index.scip to exist).",
    )
    p.add_argument(
        "--query-only",
        action="store_true",
        default=False,
        help="Skip indexing; go straight to the query REPL.",
    )
    p.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Ask a single question and exit (non-interactive).",
    )
    p.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help=f"ChromaDB storage directory (default: <repo>/{CHROMA_PERSIST_DIR}).",
    )
    p.add_argument(
        "--collection",
        type=str,
        default=CHROMA_COLLECTION,
        help=f"ChromaDB collection name (default: {CHROMA_COLLECTION}).",
    )
    p.add_argument(
        "--export-chunks",
        type=str,
        default=None,
        metavar="PATH",
        help="Export enriched chunks as JSON (for the Friction Scoring Engine).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    # ── Logging ──────────────────────────────────────────────────
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    repo_root = Path(args.repo_path).resolve()
    if not repo_root.is_dir():
        logger.error("Repository path does not exist: %s", repo_root)
        sys.exit(1)

    persist_dir = args.persist_dir or str(repo_root / CHROMA_PERSIST_DIR)
    collection = args.collection

    # ── Query-only mode ──────────────────────────────────────────
    if args.query_only:
        if args.question:
            run_query(args.question, persist_dir, collection)
        else:
            run_repl(persist_dir, collection)
        return

    # ── Ensure proto bindings exist ──────────────────────────────
    _ensure_proto_compiled()

    # ── Step 1: SCIP indexer ─────────────────────────────────────
    scip_path = repo_root / SCIP_INDEX_FILENAME
    if not args.skip_scip:
        try:
            scip_path = run_scip_indexer(repo_root)
        except (RuntimeError, FileNotFoundError) as exc:
            logger.error("SCIP indexer failed: %s", exc)
            logger.info(
                "Continuing without SCIP data. "
                "Chunks will still be indexed but without symbol enrichment."
            )
            scip_path = None  # type: ignore[assignment]

    # ── Step 2: Parse SCIP ───────────────────────────────────────
    parsed_index = None
    if scip_path and scip_path.exists():
        logger.info("Parsing SCIP index: %s", scip_path)
        parsed_index = parse_scip(scip_path)
        logger.info(
            "  → %d documents, %d symbols",
            len(parsed_index.documents),
            len(parsed_index.symbol_table),
        )
    else:
        logger.warning("No SCIP index found — proceeding without symbol data.")

    # ── Step 3: Ingest ───────────────────────────────────────────
    logger.info("Ingesting repository: %s", repo_root)
    nodes, chunks = ingest(repo_root, parsed_index)
    logger.info("  → %d enriched code chunks", len(chunks))

    # ── Step 3b (optional): Export chunks ────────────────────────
    if args.export_chunks:
        export_path = Path(args.export_chunks)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2)
        logger.info("Exported %d chunks to %s", len(chunks), export_path)

    # ── Step 4: Embed + persist ──────────────────────────────────
    if not nodes:
        logger.warning("No code chunks to index.  Exiting.")
        sys.exit(0)

    logger.info("Building vector index (%d nodes) …", len(nodes))
    index = embed_and_persist(nodes, persist_dir, collection)

    # ── Step 5: Query ────────────────────────────────────────────
    if args.question:
        run_query(args.question, persist_dir, collection)
    else:
        print(f"\n{'═' * 60}")
        print("  Lumen indexing complete!")
        print(f"  Chunks:  {len(chunks)}")
        print(f"  Storage: {persist_dir}")
        print(f"{'═' * 60}\n")
        run_repl(persist_dir, collection)


if __name__ == "__main__":
    main()
