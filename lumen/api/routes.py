"""
REST endpoint definitions for the Lumen API.

All endpoints live under ``/api/v1/``.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from lumen import __version__
from lumen.api.deps import get_query_index, invalidate_index_cache
from lumen.api.schemas import (
    ChunkListResponse,
    ChunkResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    QueryResultItem,
    RepoResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


# ── Health ───────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Quick liveness / readiness probe."""
    from lumen.db.session import check_connection

    db_connected = check_connection()

    return HealthResponse(
        status="ok",
        storage_backend="supabase",
        db_connected=db_connected,
        version=__version__,
    )


# ── Index a repository ──────────────────────────────────────────────


@router.post("/repos/index", response_model=IndexResponse)
async def index_repo(req: IndexRequest, background_tasks: BackgroundTasks):
    """
    Kick off background indexing of a repository.

    Accepts either a local ``repo_path`` or a remote ``git_url``.
    When a ``git_url`` is provided the repo is shallow-cloned to a temp
    directory, indexed, and then cleaned up automatically.

    Returns immediately with a ``repo_id`` and ``status: indexing``.
    The caller can poll ``GET /repos/{repo_id}`` to check progress.

    When ``incremental=True``, the service looks up the most recent
    index for this repo path and only re-processes changed files.
    Falls back to a full index if no previous index exists.
    """
    print("POST /repos/index — request body: %s", req.model_dump_json())

    from lumen.storage.supabase_store import (
        create_repository,
        find_repository_by_path,
        update_repository_status,
    )

    # ── Resolve source: local path or Git URL ────────────────────
    clone_path = None  # set when we clone a remote repo

    if req.git_url:
        from lumen.git_clone import clone_repo, normalize_git_url, repo_name_from_url

        try:
            clone_path = clone_repo(req.git_url, branch=req.branch)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        repo_path = clone_path
        repo_display_name = repo_name_from_url(req.git_url)
        # Keep original URL for lookup, normalize for storage
        original_url = req.git_url
        normalized_url = normalize_git_url(req.git_url)
        path_or_url = original_url  # Use original for initial lookup
    else:
        repo_path = Path(req.repo_path).resolve()
        if not repo_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Path does not exist: {repo_path}",
            )
        repo_display_name = repo_path.name
        path_or_url = str(repo_path)

    language = req.language or "auto"

    # ── Incremental path (local repos only) ───────────────────────
    if req.incremental and not clone_path:
        existing = find_repository_by_path(path_or_url)
        if existing is not None:
            rid = uuid.UUID(existing["id"])
            # Normalize the stored URL if it's a git URL
            if clone_path:
                _normalize_stored_url(rid, normalized_url)
            update_repository_status(rid, "indexing")
            background_tasks.add_task(
                _run_incremental_pipeline,
                repo_id=rid,
                repo_path=repo_path,
                language=language,
            )
            return IndexResponse(
                repo_id=str(rid),
                status="indexing",
                message=(
                    f"Incremental re-index started for {repo_display_name}. "
                    f"Poll GET /repos/{rid} for status."
                ),
            )
        # No prior index — fall through to full index
        logger.info("No prior index for %s — performing full index.", repo_path)

    # ── Full index path ───────────────────────────────────────────
    # Try to find existing repo by original URL first (for backward compatibility)
    existing = find_repository_by_path(path_or_url)

    # If not found and this is a git URL, try normalized URL
    if not existing and clone_path:
        existing = find_repository_by_path(normalized_url)

    if existing is not None:
        repo_id = uuid.UUID(existing["id"])
        logger.info("Found existing repository %s for %s — re-indexing.", repo_id, path_or_url)
        # Normalize the stored URL if it's a git URL with credentials
        if clone_path and existing.get("path_or_url") != normalized_url:
            _normalize_stored_url(repo_id, normalized_url)
            logger.info("Normalized stored URL for repo %s", repo_id)
    else:
        # Create new repo with normalized URL for git URLs
        repo_id = create_repository(
            name=repo_display_name,
            path_or_url=normalized_url if clone_path else path_or_url,
            languages=[language],
        )
    update_repository_status(repo_id, "indexing")

    background_tasks.add_task(
        _run_indexing_pipeline,
        repo_id=repo_id,
        repo_path=repo_path,
        language=language,
        clone_path=clone_path,
    )

    return IndexResponse(
        repo_id=str(repo_id),
        status="indexing",
        message=f"Indexing started for {repo_display_name}. Poll GET /repos/{repo_id} for status.",
    )


# ── List repositories ───────────────────────────────────────────────


@router.get("/repos", response_model=List[RepoResponse])
async def list_repos():
    """List all indexed repositories."""
    from lumen.storage.supabase_store import list_repositories

    rows = list_repositories()
    return [RepoResponse(**r) for r in rows]


# ── Get single repository ───────────────────────────────────────────


@router.get("/repos/{repo_id}", response_model=RepoResponse)
async def get_repo(repo_id: str):
    """Get details for a single repository."""
    from lumen.storage.supabase_store import get_repository

    try:
        uid = uuid.UUID(repo_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid repo_id (must be UUID).")

    repo = get_repository(uid)
    if repo is None:
        raise HTTPException(status_code=404, detail="Repository not found.")
    return RepoResponse(**repo)


# ── List chunks for a repo ──────────────────────────────────────────


@router.get("/repos/{repo_id}/chunks", response_model=ChunkListResponse)
async def list_chunks(
    repo_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    """
    Paginated list of code chunks for a repository.

    This is the endpoint the Friction Scoring Engine calls.
    """
    from lumen.storage.supabase_store import get_chunks_for_repo, get_repository

    try:
        uid = uuid.UUID(repo_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid repo_id (must be UUID).")

    repo = get_repository(uid)
    if repo is None:
        raise HTTPException(status_code=404, detail="Repository not found.")

    chunks = get_chunks_for_repo(uid, offset=offset, limit=limit)
    return ChunkListResponse(
        repo_id=repo_id,
        chunks=[ChunkResponse(**c) for c in chunks],
        offset=offset,
        limit=limit,
        total=repo.get("chunk_count", 0),
    )


# ── Incremental re-index ─────────────────────────────────────────────


@router.put("/repos/{repo_id}/reindex", response_model=IndexResponse)
async def reindex_repo(repo_id: str, background_tasks: BackgroundTasks):
    """
    Trigger an incremental re-index of an existing repository.

    Only files that have changed since the last index run will be
    re-processed.  The repo must already have been indexed at least once.
    """
    from lumen.storage.supabase_store import get_repository, update_repository_status

    try:
        uid = uuid.UUID(repo_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid repo_id (must be UUID).")

    repo = get_repository(uid)
    if repo is None:
        raise HTTPException(status_code=404, detail="Repository not found.")

    repo_path = Path(repo["path_or_url"])
    if not repo_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Repository path no longer exists on disk: {repo_path}",
        )

    update_repository_status(uid, "indexing")

    # Determine language from the stored repo record
    language = ",".join(repo.get("languages", [])) or "auto"

    background_tasks.add_task(
        _run_incremental_pipeline,
        repo_id=uid,
        repo_path=repo_path,
        language=language,
    )

    return IndexResponse(
        repo_id=repo_id,
        status="indexing",
        message=(
            f"Incremental re-index started for {repo_path.name}. "
            f"Poll GET /repos/{repo_id} for status."
        ),
    )


# ── Semantic query ──────────────────────────────────────────────────


@router.post("/query", response_model=QueryResponse)
async def query_code(req: QueryRequest):
    """
    Run a semantic search against the indexed codebase.

    When ``repo_id`` or ``repo_ids`` is provided, results are scoped
    to the specified repository/repositories.  Otherwise, all indexed
    repos are searched.

    Returns ranked code chunks with similarity scores.
    """
    from lumen.query.engine import query_index

    # Validate mutual exclusivity of repo_id / repo_ids
    if req.repo_id and req.repo_ids:
        raise HTTPException(
            status_code=400,
            detail="Provide either repo_id or repo_ids, not both.",
        )

    index = get_query_index()
    if index is None:
        raise HTTPException(
            status_code=503,
            detail="No index available. Index a repository first.",
        )

    results = query_index(
        req.question,
        index=index,
        top_k=req.top_k,
        repo_id=req.repo_id,
        repo_ids=req.repo_ids,
    )

    items = [
        QueryResultItem(
            score=r.score,
            file_path=r.file_path,
            line_start=r.metadata.get("line_start"),
            line_end=r.metadata.get("line_end"),
            language=r.metadata.get("language"),
            symbol_count=r.symbol_count,
            text_preview=r.text[:1000],
        )
        for r in results
    ]

    return QueryResponse(
        question=req.question,
        results=items,
        total=len(items),
    )


# ── Background indexing pipeline ─────────────────────────────────────


def _normalize_stored_url(repo_id: uuid.UUID, normalized_url: str) -> None:
    """Update a repository's path_or_url to the normalized version."""
    from lumen.db.session import get_session
    from lumen.db.models import Repository

    with get_session() as session:
        repo = session.get(Repository, repo_id)
        if repo:
            repo.path_or_url = normalized_url


def _run_indexing_pipeline(
    repo_id: uuid.UUID,
    repo_path: Path,
    language: str,
    clone_path: Path | None = None,
) -> None:
    """
    Execute the full Lumen indexing pipeline in a background thread.

    Wires results to the Supabase tables via ``repo_id``.
    When *clone_path* is set, the cloned directory is deleted in the
    ``finally`` block regardless of success or failure.
    """
    import traceback

    from lumen.indexer import (
        _ensure_proto_compiled,
        _merge_parsed_indexes,
        detect_languages,
        embed_and_persist,
        ingest,
        parse_scip,
        run_scip_indexer,
    )
    from lumen.config import (
        SCIP_INDEX_FILENAME,
        resolve_language,
    )

    try:
        logger.info("Background indexing started for %s", repo_path)

        # ── Resolve languages ────────────────────────────────────
        if language.lower() == "auto":
            profiles = detect_languages(repo_path)
            if not profiles:
                raise RuntimeError(f"No supported languages detected in {repo_path}")
        else:
            raw = [s.strip() for s in language.split(",") if s.strip()]
            profiles = [resolve_language(lang) for lang in raw]

        lang_names = [p.name for p in profiles]

        # Update languages on the repo record
        from lumen.db.session import get_session
        from lumen.db.models import Repository

        with get_session() as session:
            repo = session.get(Repository, repo_id)
            if repo:
                repo.languages = lang_names

        # ── Proto compilation ────────────────────────────────────
        _ensure_proto_compiled()

        # ── SCIP indexers ────────────────────────────────────────
        parsed_indexes = []
        for profile in profiles:
            try:
                result_path = run_scip_indexer(
                    repo_path, profile, project_name=repo_path.name,
                )
                if result_path and result_path.exists():
                    parsed = parse_scip(result_path)
                    parsed_indexes.append(parsed)
            except (RuntimeError, FileNotFoundError) as exc:
                logger.warning("SCIP failed for %s: %s", profile.name, exc)

        parsed_index = None
        if parsed_indexes:
            parsed_index = _merge_parsed_indexes(parsed_indexes)

        # ── Ingest ───────────────────────────────────────────────
        all_extensions = frozenset().union(*(p.extensions for p in profiles))
        nodes, chunks = ingest(repo_path, parsed_index, all_extensions, repo_id=str(repo_id))
        logger.info("Ingested %d chunks for %s", len(chunks), repo_path.name)

        # ── Persist structured data ──────────────────────────────
        from lumen.storage.supabase_store import persist_chunks
        persist_chunks(repo_id, chunks)

        # ── Embed + persist vectors ──────────────────────────────
        from lumen.storage.supabase_store import build_index as sb_build
        sb_build(nodes, repo_id=repo_id)

        # ── Update status → ready ────────────────────────────────
        total_symbols = sum(c.symbol_count for c in chunks)
        from lumen.storage.supabase_store import update_repository_status
        update_repository_status(
            repo_id,
            "ready",
            chunk_count=len(chunks),
            symbol_count=total_symbols,
        )

        # Save file states so incremental re-index has a baseline
        # (skip for cloned repos — they don't persist on disk)
        if not clone_path:
            from lumen.incremental import (
                detect_file_changes,
                save_file_states,
                save_last_indexed_commit,
            )

            initial_changes = detect_file_changes(repo_id, repo_path, all_extensions)
            chunk_counts: dict[str, int] = {}
            for c in chunks:
                chunk_counts[c.file_path] = chunk_counts.get(c.file_path, 0) + 1
            save_file_states(repo_id, initial_changes, chunk_counts)
            save_last_indexed_commit(repo_id, repo_path)

        # Invalidate query cache so next query picks up new data
        invalidate_index_cache()

        logger.info(
            "Background indexing complete: %s (%d chunks, %d symbols)",
            repo_path.name,
            len(chunks),
            total_symbols,
        )

    except Exception as exc:
        logger.error("Background indexing failed: %s\n%s", exc, traceback.format_exc())
        try:
            from lumen.storage.supabase_store import update_repository_status
            update_repository_status(
                repo_id, "failed", error_message=str(exc)
            )
        except Exception:
            logger.error("Could not update repo status to 'failed'.")

    finally:
        # Always clean up cloned repos
        if clone_path:
            from lumen.git_clone import cleanup_clone
            cleanup_clone(clone_path)


def _run_incremental_pipeline(
    repo_id: uuid.UUID,
    repo_path: Path,
    language: str,
) -> None:
    """
    Execute incremental re-indexing in a background thread.

    Only files that have changed since the last run are re-processed.
    """
    import traceback

    from lumen.config import resolve_language
    from lumen.indexer import detect_languages, incremental_index

    try:
        logger.info("Background incremental re-index started for %s", repo_path)

        # ── Resolve languages ─────────────────────────────────────
        if language.lower() == "auto":
            profiles = detect_languages(repo_path)
            if not profiles:
                raise RuntimeError(f"No supported languages detected in {repo_path}")
        else:
            raw = [s.strip() for s in language.split(",") if s.strip()]
            profiles = [resolve_language(lang) for lang in raw]

        # ── Run incremental index ─────────────────────────────────
        total_chunks, total_symbols = incremental_index(
            repo_root=repo_path,
            repo_id=repo_id,
            profiles=profiles,
        )

        # Invalidate query cache so next query picks up new data
        invalidate_index_cache()

        logger.info(
            "Background incremental re-index complete: %s (%d chunks, %d symbols)",
            repo_path.name,
            total_chunks,
            total_symbols,
        )

    except Exception as exc:
        logger.error(
            "Background incremental re-index failed: %s\n%s",
            exc,
            traceback.format_exc(),
        )
        try:
            from lumen.storage.supabase_store import update_repository_status

            update_repository_status(
                repo_id, "failed", error_message=str(exc)
            )
        except Exception:
            logger.error("Could not update repo status to 'failed'.")
