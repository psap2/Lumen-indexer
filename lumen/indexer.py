#!/usr/bin/env python3
"""
Lumen — Repository Indexing Service.

CLI entry-point that orchestrates the full pipeline:

    1. Detect (or accept) the target language(s).
    2. Run the appropriate SCIP indexer to generate ``index.scip``.
    3. Parse the resulting protobuf.
    4. Walk the repo, split code, and enrich chunks with SCIP symbols.
    5. Embed and persist to Supabase (Postgres + pgvector).
    6. (Optional) Run an interactive query REPL.

Usage
─────
    # Auto-detect language and index
    python -m lumen.indexer /path/to/repo

    # Explicit language
    python -m lumen.indexer /path/to/repo --language python
    python -m lumen.indexer /path/to/repo --language typescript

    # Multi-language repo (comma-separated)
    python -m lumen.indexer /path/to/repo --language typescript,python

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
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from lumen.config import (
    DEFAULT_IGNORE_PATTERNS,
    LANGUAGE_REGISTRY,
    SCIP_INDEX_FILENAME,
    LanguageProfile,
    resolve_language,
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


# ── Language auto-detection ──────────────────────────────────────────


def _should_ignore(path: Path) -> bool:
    return any(ig in path.parts for ig in DEFAULT_IGNORE_PATTERNS)


def detect_languages(repo_root: Path) -> List[LanguageProfile]:
    """
    Scan the repo's file extensions and return the matching
    ``LanguageProfile``(s), ordered by file count (most files first).

    Only returns languages that are in our ``LANGUAGE_REGISTRY``.
    """
    # Build a reverse map:  extension → language key
    ext_to_lang: Dict[str, str] = {}
    for lang_key, profile in LANGUAGE_REGISTRY.items():
        for ext in profile.extensions:
            ext_to_lang[ext] = lang_key

    # Count extensions
    lang_counts: Counter[str] = Counter()
    for p in repo_root.rglob("*"):
        if not p.is_file() or _should_ignore(p):
            continue
        ext = p.suffix.lower()
        lang_key = ext_to_lang.get(ext)
        if lang_key:
            lang_counts[lang_key] += 1

    if not lang_counts:
        logger.warning("No recognised source files found in %s", repo_root)
        return []

    # Return profiles ordered by file count
    profiles = []
    for lang_key, count in lang_counts.most_common():
        profile = LANGUAGE_REGISTRY[lang_key]
        logger.info("  Detected: %-12s (%d files)", profile.name, count)
        profiles.append(profile)

    return profiles


# ── Step 1: SCIP indexer execution ───────────────────────────────────


def run_scip_indexer(
    repo_root: Path,
    profile: LanguageProfile,
    project_name: Optional[str] = None,
) -> Optional[Path]:
    """
    Invoke the SCIP indexer for *profile* and return the path to the
    generated ``index.scip``, or ``None`` if no SCIP indexer is available.
    """
    if profile.scip_command is None:
        logger.info(
            "No SCIP indexer for %s. %s",
            profile.name,
            profile.install_hint,
        )
        return None

    output_path = repo_root / SCIP_INDEX_FILENAME

    # Build the command — some indexers need extra flags.
    cmd = list(profile.scip_command)

    # scip-python needs --project-name
    if profile.name == "python":
        name = project_name or repo_root.name
        cmd.extend(["--project-name", name])

    logger.info("Running SCIP indexer (%s) in %s …", profile.name, repo_root)
    logger.info("  Command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=300,
            shell=True,  # Required on Windows to find .cmd/.bat files like npx
        )
    except FileNotFoundError:
        tool = cmd[0]
        raise RuntimeError(
            f"'{tool}' not found on PATH.\n"
            f"  Install hint: {profile.install_hint}"
        ) from None

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"SCIP indexer ({profile.name}) exited with code {result.returncode}.\n"
            f"stderr:\n{stderr}"
        )

    if result.stdout.strip():
        logger.debug("SCIP stdout:\n%s", result.stdout.strip())

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


def _merge_parsed_indexes(indexes):
    """Merge multiple ParsedIndex objects into one."""
    from lumen.scip_parser.parser import ParsedIndex

    if len(indexes) == 1:
        return indexes[0]

    merged = ParsedIndex()
    for idx in indexes:
        merged.documents.extend(idx.documents)
        merged.external_symbols.extend(idx.external_symbols)

    merged.build_lookup_tables()
    logger.info(
        "Merged %d SCIP indexes → %d documents, %d symbols",
        len(indexes),
        len(merged.documents),
        len(merged.symbol_table),
    )
    return merged


# ── Step 3: Ingest ───────────────────────────────────────────────────


def ingest(
    repo_root: Path,
    parsed_index,
    extensions: frozenset[str],
    file_filter: Optional[set[str]] = None,
    repo_id: Optional[str] = None,
):
    from lumen.ingestion.code_ingestor import ingest_repository

    nodes, chunks = ingest_repository(
        repo_root,
        parsed_index=parsed_index,
        extensions=extensions,
        file_filter=file_filter,
        repo_id=repo_id,
    )
    return nodes, chunks


# ── Step 4: Embed + persist ──────────────────────────────────────────


def embed_and_persist(nodes, repo_id=None):
    """Embed and persist to Supabase pgvector."""
    from lumen.storage.supabase_store import build_index

    return build_index(nodes, repo_id=repo_id)


# ── Step 5: Query ────────────────────────────────────────────────────


def run_query(question: str) -> None:
    """Query the Supabase-backed index."""
    from lumen.storage.supabase_store import load_index
    from lumen.query.engine import query_index

    index = load_index()
    if index is None:
        print("ERROR: No index available. Run indexing first.")
        return
    results = query_index(question, index=index, top_k=5)
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


def run_repl(repo_id: Optional[str] = None) -> None:
    """Interactive REPL backed by Supabase, optionally scoped to a repo."""
    from lumen.query.engine import interactive_repl

    interactive_repl(repo_id=repo_id)


# ── Incremental indexing ─────────────────────────────────────────────


def incremental_index(
    repo_root: Path,
    repo_id,
    profiles: List[LanguageProfile],
    skip_scip: bool = False,
) -> tuple[int, int]:
    """
    Re-index a previously indexed repository, processing only files
    that have been added, modified, or deleted since the last run.

    Returns ``(chunk_count, symbol_count)`` totals after the update.
    """
    import uuid as _uuid

    from lumen.incremental import (
        changed_file_paths,
        detect_file_changes,
        get_change_summary,
        save_file_states,
        stale_file_paths,
    )
    from lumen.storage.supabase_store import (
        delete_file_data,
        get_repo_totals,
        persist_chunks_incremental,
        update_repository_status,
    )

    if isinstance(repo_id, str):
        repo_id = _uuid.UUID(repo_id)

    lang_names = [p.name for p in profiles]
    all_extensions: frozenset[str] = frozenset().union(
        *(p.extensions for p in profiles)
    )

    # ── 1. Detect changes ─────────────────────────────────────────
    logger.info("Detecting file changes for incremental re-index …")
    changes = detect_file_changes(repo_id, repo_root, all_extensions)
    summary = get_change_summary(changes)
    logger.info("Change summary: %s", summary)

    new_or_modified = changed_file_paths(changes)
    stale = stale_file_paths(changes)

    if not new_or_modified and not stale:
        logger.info("Nothing changed — skipping re-index.")
        return get_repo_totals(repo_id)

    update_repository_status(repo_id, "indexing")

    # ── 2. Run SCIP on full repo (needs full context) ─────────────
    _ensure_proto_compiled()

    parsed_indexes = []
    scip_path = repo_root / SCIP_INDEX_FILENAME

    if not skip_scip:
        for profile in profiles:
            try:
                result_path = run_scip_indexer(
                    repo_root, profile, project_name=repo_root.name,
                )
                if result_path and result_path.exists():
                    parsed = parse_scip(result_path)
                    parsed_indexes.append(parsed)
            except (RuntimeError, FileNotFoundError) as exc:
                logger.warning("SCIP failed for %s: %s", profile.name, exc)
    else:
        if scip_path.exists():
            parsed = parse_scip(scip_path)
            parsed_indexes.append(parsed)

    parsed_index = None
    if parsed_indexes:
        parsed_index = _merge_parsed_indexes(parsed_indexes)

    # ── 3. Delete stale data ────────────────────────────────────────
    # Clean up ALL non-unchanged files.  "New" files may still have
    # leftover chunks from a full index that ran before file-state
    # tracking was introduced, so we delete for new+modified+deleted.
    all_affected = {c.path for c in changes if c.status != "unchanged"}
    if all_affected:
        logger.info("Cleaning up data for %d affected files …", len(all_affected))
        delete_file_data(repo_id, list(all_affected))

    # ── 4. Ingest only new + modified files ───────────────────────
    if new_or_modified:
        logger.info(
            "Ingesting %d new/modified files …", len(new_or_modified),
        )
        nodes, chunks = ingest(
            repo_root, parsed_index, all_extensions,
            file_filter=new_or_modified,
            repo_id=str(repo_id),
        )

        # ── 5. Persist new structured data + embeddings ───────────
        if nodes:
            persist_chunks_incremental(repo_id, chunks)
            embed_and_persist(nodes, repo_id=repo_id)
    else:
        chunks = []

    # ── 6. Update file state tracking ─────────────────────────────
    chunk_counts: Dict[str, int] = {}
    for c in chunks:
        chunk_counts[c.file_path] = chunk_counts.get(c.file_path, 0) + 1

    save_file_states(repo_id, changes, chunk_counts)

    # ── 7. Recount totals and update repo status ──────────────────
    total_chunks, total_symbols = get_repo_totals(repo_id)
    update_repository_status(
        repo_id, "ready",
        chunk_count=total_chunks,
        symbol_count=total_symbols,
    )

    logger.info(
        "Incremental re-index complete: %d total chunks, %d total symbols",
        total_chunks,
        total_symbols,
    )
    return total_chunks, total_symbols


# ── CLI ──────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    supported = ", ".join(sorted(LANGUAGE_REGISTRY.keys()))

    p = argparse.ArgumentParser(
        prog="lumen.indexer",
        description="Lumen Repository Indexing Service — SCIP + LlamaIndex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m lumen.indexer ./my-project                        # local repo, auto-detect\n"
            "  python -m lumen.indexer ./my-project --language python       # explicit language\n"
            "  python -m lumen.indexer --git-url https://github.com/org/repo.git  # remote repo\n"
            "  python -m lumen.indexer --git-url https://github.com/org/repo.git --branch dev\n"
            "  python -m lumen.indexer ./my-project --skip-scip\n"
            "  python -m lumen.indexer ./my-project --query-only\n"
            '  python -m lumen.indexer ./my-project -q "What does parseConfig do?"'
        ),
    )
    p.add_argument(
        "repo_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the local repository to index (or use --git-url).",
    )
    p.add_argument(
        "--git-url",
        type=str,
        default=None,
        help="Git clone URL (HTTPS or SSH).  Clones to a temp dir, indexes, then cleans up.",
    )
    p.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Branch to clone (only used with --git-url).  Defaults to repo default branch.",
    )
    p.add_argument(
        "--language", "-l",
        type=str,
        default="auto",
        help=(
            f"Language(s) to index. Use 'auto' to detect from file extensions, "
            f"or specify one or more (comma-separated). Supported: {supported}"
        ),
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
        "--export-chunks",
        type=str,
        default=None,
        metavar="PATH",
        help="Export enriched chunks as JSON (for the Friction Scoring Engine).",
    )
    p.add_argument(
        "--incremental", "-i",
        action="store_true",
        default=False,
        help=(
            "Incremental re-index: only process files that have changed "
            "since the last run.  Requires a prior full index for this repo."
        ),
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

    # ── Resolve source: local path or Git URL ────────────────────
    clone_path: Optional[Path] = None

    if args.git_url:
        if args.repo_path:
            logger.error("Provide either repo_path or --git-url, not both.")
            sys.exit(1)
        from lumen.git_clone import clone_repo, repo_name_from_url
        try:
            clone_path = clone_repo(args.git_url, branch=args.branch)
        except RuntimeError as exc:
            logger.error("Clone failed: %s", exc)
            sys.exit(1)
        repo_root = clone_path
    elif args.repo_path:
        repo_root = Path(args.repo_path).resolve()
        if not repo_root.is_dir():
            logger.error("Repository path does not exist: %s", repo_root)
            sys.exit(1)
    else:
        logger.error("Provide either a repo_path argument or --git-url.")
        sys.exit(1)

    try:
        _main_inner(args, repo_root, clone_path)
    finally:
        if clone_path:
            from lumen.git_clone import cleanup_clone
            cleanup_clone(clone_path)


def _main_inner(args, repo_root: Path, clone_path: Optional[Path]) -> None:
    """Inner main logic, separated so clone cleanup runs in finally."""

    # ── Query-only mode ──────────────────────────────────────────
    if args.query_only:
        if args.question:
            run_query(args.question)
        else:
            run_repl()
        return

    # ── Resolve language(s) ──────────────────────────────────────
    if args.language.lower() == "auto":
        logger.info("Auto-detecting language(s) in %s …", repo_root)
        profiles = detect_languages(repo_root)
        if not profiles:
            logger.error(
                "Could not detect any supported language. "
                "Use --language to specify one explicitly."
            )
            sys.exit(1)
    else:
        # Parse comma-separated list: --language typescript,python
        raw_langs = [s.strip() for s in args.language.split(",") if s.strip()]
        try:
            profiles = [resolve_language(lang) for lang in raw_langs]
        except ValueError as exc:
            logger.error("%s", exc)
            sys.exit(1)

    lang_names = [p.name for p in profiles]
    logger.info("Languages to index: %s", ", ".join(lang_names))

    # ── Ensure proto bindings exist ──────────────────────────────
    _ensure_proto_compiled()

    # ── Incremental mode ─────────────────────────────────────────
    if args.incremental:
        from lumen.storage.supabase_store import find_repository_by_path

        existing = find_repository_by_path(str(repo_root))
        if existing is None:
            logger.warning(
                "No previous index found for %s — falling back to full index.",
                repo_root,
            )
        else:
            import uuid as _uuid

            repo_id = _uuid.UUID(existing["id"])
            logger.info(
                "Incremental re-index of %s (repo_id=%s)", repo_root, repo_id,
            )

            total_chunks, total_symbols = incremental_index(
                repo_root,
                repo_id=repo_id,
                profiles=profiles,
                skip_scip=args.skip_scip,
            )

            if args.question:
                run_query(args.question)
            else:
                print(f"\n{'═' * 60}")
                print("  Lumen incremental re-index complete!")
                print(f"  Languages: {', '.join(lang_names)}")
                print(f"  Chunks:    {total_chunks}")
                print(f"  Symbols:   {total_symbols}")
                print(f"  Repo ID:   {repo_id}")
                print(f"{'═' * 60}\n")

                run_repl(repo_id=str(repo_id))
            return

    # ── Step 1: SCIP indexer(s) ──────────────────────────────────
    parsed_indexes = []
    scip_path = repo_root / SCIP_INDEX_FILENAME

    if not args.skip_scip:
        for profile in profiles:
            try:
                result_path = run_scip_indexer(
                    repo_root, profile, project_name=repo_root.name,
                )
                if result_path and result_path.exists():
                    logger.info("Parsing SCIP index for %s …", profile.name)
                    parsed = parse_scip(result_path)
                    logger.info(
                        "  → %d documents, %d symbols",
                        len(parsed.documents),
                        len(parsed.symbol_table),
                    )
                    parsed_indexes.append(parsed)
            except (RuntimeError, FileNotFoundError) as exc:
                logger.warning("SCIP indexer failed for %s: %s", profile.name, exc)
                logger.info("  Continuing without SCIP data for %s.", profile.name)
    else:
        if scip_path.exists():
            logger.info("Parsing existing SCIP index: %s", scip_path)
            parsed = parse_scip(scip_path)
            logger.info(
                "  → %d documents, %d symbols",
                len(parsed.documents),
                len(parsed.symbol_table),
            )
            parsed_indexes.append(parsed)
        else:
            logger.warning("--skip-scip but no index.scip found. Proceeding without SCIP.")

    # Merge SCIP indexes
    parsed_index = None
    if parsed_indexes:
        parsed_index = _merge_parsed_indexes(parsed_indexes)
    else:
        logger.warning("No SCIP data available — chunks will lack symbol enrichment.")

    # ── Create repository record early so repo_id is available ────
    from lumen.storage.supabase_store import (
        create_repository,
        persist_chunks,
        update_repository_status,
    )

    repo_id = create_repository(
        name=repo_root.name,
        path_or_url=str(repo_root),
        languages=lang_names,
    )
    update_repository_status(repo_id, "indexing")

    # ── Step 3: Ingest ───────────────────────────────────────────
    all_extensions: frozenset[str] = frozenset().union(
        *(p.extensions for p in profiles)
    )
    logger.info(
        "Ingesting repository: %s  (extensions: %s)",
        repo_root,
        ", ".join(sorted(all_extensions)),
    )
    nodes, chunks = ingest(repo_root, parsed_index, all_extensions, repo_id=str(repo_id))
    logger.info("  → %d enriched code chunks", len(chunks))

    # ── Step 3b (optional): Export chunks ────────────────────────
    if args.export_chunks:
        export_path = Path(args.export_chunks)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2)
        logger.info("Exported %d chunks to %s", len(chunks), export_path)

    # ── Step 4: Embed + persist to Supabase ──────────────────────
    if not nodes:
        logger.warning("No code chunks to index.  Exiting.")
        sys.exit(0)

    logger.info("Building vector index (%d nodes) …", len(nodes))

    try:
        persist_chunks(repo_id, chunks)
        index = embed_and_persist(nodes, repo_id=repo_id)
        total_symbols = sum(c.symbol_count for c in chunks)
        update_repository_status(
            repo_id, "ready",
            chunk_count=len(chunks),
            symbol_count=total_symbols,
        )

        # Save file states so incremental re-index has a baseline
        # (skip for cloned repos — they don't persist on disk)
        if not clone_path:
            from lumen.incremental import (
                detect_file_changes,
                save_file_states,
            )

            initial_changes = detect_file_changes(
                repo_id, repo_root, all_extensions,
            )
            chunk_counts: Dict[str, int] = {}
            for c in chunks:
                chunk_counts[c.file_path] = chunk_counts.get(c.file_path, 0) + 1
            save_file_states(repo_id, initial_changes, chunk_counts)

    except Exception:
        update_repository_status(repo_id, "failed", error_message="Embedding failed")
        raise

    # ── Step 5: Query ────────────────────────────────────────────
    source_label = args.git_url or str(repo_root)
    if args.question:
        run_query(args.question)
    else:
        print(f"\n{'═' * 60}")
        print("  Lumen indexing complete!")
        print(f"  Source:    {source_label}")
        print(f"  Languages: {', '.join(lang_names)}")
        print(f"  Chunks:    {len(chunks)}")
        print(f"  Repo ID:   {repo_id}")
        print(f"  Storage:   Supabase (pgvector)")
        print(f"{'═' * 60}\n")

        # Skip REPL for cloned repos (clone is about to be deleted)
        if not clone_path:
            run_repl(repo_id=str(repo_id))


if __name__ == "__main__":
    main()
