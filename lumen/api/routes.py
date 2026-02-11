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
from lumen.config import STORAGE_BACKEND

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


# ── Health ───────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Quick liveness / readiness probe."""
    db_connected = None
    if STORAGE_BACKEND == "supabase":
        from lumen.db.session import check_connection
        db_connected = check_connection()

    return HealthResponse(
        status="ok",
        storage_backend=STORAGE_BACKEND,
        db_connected=db_connected,
        version=__version__,
    )


# ── Index a repository ──────────────────────────────────────────────


@router.post("/repos/index", response_model=IndexResponse)
async def index_repo(req: IndexRequest, background_tasks: BackgroundTasks):
    """
    Kick off background indexing of a local repository.

    Returns immediately with a ``repo_id`` and ``status: indexing``.
    The caller can poll ``GET /repos/{repo_id}`` to check progress.
    """
    repo_path = Path(req.repo_path).resolve()
    if not repo_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {repo_path}")

    language = req.language or "auto"

    if STORAGE_BACKEND == "supabase":
        from lumen.storage.supabase_store import create_repository, update_repository_status

        repo_id = create_repository(
            name=repo_path.name,
            path_or_url=str(repo_path),
            languages=[language],
        )
        update_repository_status(repo_id, "indexing")

        background_tasks.add_task(
            _run_indexing_pipeline,
            repo_id=repo_id,
            repo_path=repo_path,
            language=language,
        )

        return IndexResponse(
            repo_id=str(repo_id),
            status="indexing",
            message=f"Indexing started for {repo_path.name}. Poll GET /repos/{repo_id} for status.",
        )
    else:
        # ChromaDB mode — run synchronously (no repo_id tracking).
        # We still background it so the HTTP response returns quickly.
        fake_id = str(uuid.uuid4())
        background_tasks.add_task(
            _run_indexing_pipeline,
            repo_id=None,
            repo_path=repo_path,
            language=language,
        )
        return IndexResponse(
            repo_id=fake_id,
            status="indexing",
            message=f"Indexing started for {repo_path.name} (ChromaDB mode).",
        )


# ── List repositories ───────────────────────────────────────────────


@router.get("/repos", response_model=List[RepoResponse])
async def list_repos():
    """List all indexed repositories (Supabase only)."""
    if STORAGE_BACKEND != "supabase":
        raise HTTPException(
            status_code=501,
            detail="Repository listing is only available with STORAGE_BACKEND=supabase.",
        )
    from lumen.storage.supabase_store import list_repositories
    rows = list_repositories()
    return [RepoResponse(**r) for r in rows]


# ── Get single repository ───────────────────────────────────────────


@router.get("/repos/{repo_id}", response_model=RepoResponse)
async def get_repo(repo_id: str):
    """Get details for a single repository."""
    if STORAGE_BACKEND != "supabase":
        raise HTTPException(
            status_code=501,
            detail="Repository details are only available with STORAGE_BACKEND=supabase.",
        )
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
    if STORAGE_BACKEND != "supabase":
        raise HTTPException(
            status_code=501,
            detail="Chunk listing is only available with STORAGE_BACKEND=supabase.",
        )
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


# ── Semantic query ──────────────────────────────────────────────────


@router.post("/query", response_model=QueryResponse)
async def query_code(req: QueryRequest):
    """
    Run a semantic search against the indexed codebase.

    Returns ranked code chunks with similarity scores.
    """
    from lumen.query.engine import query_index

    # Determine which index to load
    persist_dir = None
    if STORAGE_BACKEND != "supabase" and req.repo_path:
        from pathlib import Path
        from lumen.config import CHROMA_PERSIST_DIR
        persist_dir = str(Path(req.repo_path).resolve() / CHROMA_PERSIST_DIR)

    index = get_query_index(persist_dir=persist_dir)
    if index is None:
        raise HTTPException(
            status_code=503,
            detail="No index available. Index a repository first.",
        )

    results = query_index(req.question, index=index, top_k=req.top_k)

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


def _run_indexing_pipeline(
    repo_id: uuid.UUID | None,
    repo_path: Path,
    language: str,
) -> None:
    """
    Execute the full Lumen indexing pipeline in a background thread.

    This reuses the same functions from ``lumen.indexer`` but wires
    results to the Supabase tables when a ``repo_id`` is provided.
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
        CHROMA_COLLECTION,
        CHROMA_PERSIST_DIR,
        SCIP_INDEX_FILENAME,
        STORAGE_BACKEND,
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
        if repo_id and STORAGE_BACKEND == "supabase":
            from lumen.storage.supabase_store import update_repository_status
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
        nodes, chunks = ingest(repo_path, parsed_index, all_extensions)
        logger.info("Ingested %d chunks for %s", len(chunks), repo_path.name)

        # ── Persist structured data (Supabase only) ──────────────
        if repo_id and STORAGE_BACKEND == "supabase":
            from lumen.storage.supabase_store import persist_chunks
            persist_chunks(repo_id, chunks)

        # ── Embed + persist vectors ──────────────────────────────
        if STORAGE_BACKEND == "supabase":
            from lumen.storage.supabase_store import build_index as sb_build
            sb_build(nodes, repo_id=repo_id)
        else:
            chroma_dir = str(repo_path / CHROMA_PERSIST_DIR)
            embed_and_persist(nodes, chroma_dir, CHROMA_COLLECTION)

        # ── Update status → ready ────────────────────────────────
        total_symbols = sum(c.symbol_count for c in chunks)
        if repo_id and STORAGE_BACKEND == "supabase":
            from lumen.storage.supabase_store import update_repository_status
            update_repository_status(
                repo_id,
                "ready",
                chunk_count=len(chunks),
                symbol_count=total_symbols,
            )

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
        if repo_id and STORAGE_BACKEND == "supabase":
            try:
                from lumen.storage.supabase_store import update_repository_status
                update_repository_status(
                    repo_id, "failed", error_message=str(exc)
                )
            except Exception:
                logger.error("Could not update repo status to 'failed'.")
