"""
REST endpoint definitions for the Lumen API.

All endpoints live under ``/api/v1/``.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List
from urllib.parse import urlparse

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


# ── Git cloning helpers ──────────────────────────────────────────────


def _is_git_url(path_or_url: str) -> bool:
    """Check if the input is a git URL."""
    # Check for common git URL patterns
    git_patterns = [
        r'^https?://.*\.git$',
        r'^https?://.*\.git/',
        r'^git@.*:.*\.git$',
        r'^git://.*\.git$',
    ]
    return any(re.match(pattern, path_or_url) for pattern in git_patterns)


def _clone_repository(git_url: str) -> Path:
    """
    Clone a git repository to a temporary directory.
    
    Supports access tokens in the URL format:
    - https://token@github.com/user/repo.git
    - https://username:token@github.com/user/repo.git
    
    Returns the path to the cloned repository.
    """
    # Create a temporary directory for the clone
    temp_dir = tempfile.mkdtemp(prefix="lumen-clone-")
    
    try:
        logger.info("Cloning repository: %s", git_url)
        
        # Run git clone
        result = subprocess.run(
            ["git", "clone", "--depth", "1", git_url, temp_dir],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"Git clone failed: {error_msg}\n"
                f"URL: {git_url}"
            )
        
        cloned_path = Path(temp_dir)
        if not cloned_path.is_dir():
            raise RuntimeError(f"Cloned repository not found at {cloned_path}")
        
        logger.info("Successfully cloned to %s", cloned_path)
        return cloned_path
        
    except subprocess.TimeoutExpired:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Git clone timed out after 5 minutes: {git_url}")
    except FileNotFoundError:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(
            "Git not found on PATH. Install git to clone repositories."
        )
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _extract_repo_name_from_url(git_url: str) -> str:
    """Extract repository name from a git URL."""
    # Remove .git suffix if present
    url = git_url.rstrip('/').rstrip('.git')
    # Extract the last part of the path
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split('/') if p]
    if path_parts:
        return path_parts[-1]
    # Fallback: use a sanitized version of the URL
    return re.sub(r'[^\w-]', '_', url)[:50]


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
    Kick off background indexing of a local repository or clone and index from a git URL.

    Returns immediately with a ``repo_id`` and ``status: indexing``.
    The caller can poll ``GET /repos/{repo_id}`` to check progress.

    When ``incremental=True``, the service looks up the most recent
    index for this repo path and only re-processes changed files.
    Falls back to a full index if no previous index exists.
    
    Supports git URLs with access tokens:
    - https://token@github.com/user/repo.git
    - https://username:token@github.com/user/repo.git
    """
    # Determine if input is a git URL or local path
    is_git_url = _is_git_url(req.repo_path)
    
    if is_git_url:
        # Clone the repository
        try:
            repo_path = _clone_repository(req.repo_path)
            repo_name = _extract_repo_name_from_url(req.repo_path)
            is_cloned = True
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to clone repository: {str(e)}"
            )
    else:
        # Use local path
        repo_path = Path(req.repo_path).resolve()
        if not repo_path.is_dir():
            raise HTTPException(
                status_code=400, 
                detail=f"Path does not exist: {repo_path}"
            )
        repo_name = repo_path.name
        is_cloned = False

    language = req.language or "auto"

    from lumen.storage.supabase_store import (
        create_repository,
        find_repository_by_path,
        update_repository_status,
    )

    # ── Incremental path ──────────────────────────────────────────
    if req.incremental:
        # For git URLs, use the URL as the lookup key
        lookup_key = req.repo_path if is_git_url else str(repo_path)
        existing = find_repository_by_path(lookup_key)
        if existing is not None:
            rid = uuid.UUID(existing["id"])
            update_repository_status(rid, "indexing")
            background_tasks.add_task(
                _run_incremental_pipeline,
                repo_id=rid,
                repo_path=repo_path,
                language=language,
                cleanup_clone=is_cloned,  # Clean up cloned repos after indexing
            )
            return IndexResponse(
                repo_id=str(rid),
                status="indexing",
                message=(
                    f"Incremental re-index started for {repo_name}. "
                    f"Poll GET /repos/{rid} for status."
                ),
            )
        # No prior index — fall through to full index
        logger.info("No prior index for %s — performing full index.", repo_path)

    # ── Full index path ───────────────────────────────────────────
    # Store the original URL for git repos, path for local repos
    path_or_url = req.repo_path if is_git_url else str(repo_path)
    
    repo_id = create_repository(
        name=repo_name,
        path_or_url=path_or_url,
        languages=[language],
    )
    update_repository_status(repo_id, "indexing")

    background_tasks.add_task(
        _run_indexing_pipeline,
        repo_id=repo_id,
        repo_path=repo_path,
        language=language,
        cleanup_clone=is_cloned,  # Clean up cloned repos after indexing
    )

    return IndexResponse(
        repo_id=str(repo_id),
        status="indexing",
        message=f"Indexing started for {repo_name}. Poll GET /repos/{repo_id} for status.",
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

    Returns ranked code chunks with similarity scores.
    """
    from lumen.query.engine import query_index

    index = get_query_index()
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
    repo_id: uuid.UUID,
    repo_path: Path,
    language: str,
    cleanup_clone: bool = False,
) -> None:
    """
    Execute the full Lumen indexing pipeline in a background thread.

    Wires results to the Supabase tables via ``repo_id``.
    
    Args:
        cleanup_clone: If True, remove the cloned repository after indexing.
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
        nodes, chunks = ingest(repo_path, parsed_index, all_extensions)
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
        from lumen.incremental import detect_file_changes, save_file_states

        initial_changes = detect_file_changes(repo_id, repo_path, all_extensions)
        chunk_counts: dict[str, int] = {}
        for c in chunks:
            chunk_counts[c.file_path] = chunk_counts.get(c.file_path, 0) + 1
        save_file_states(repo_id, initial_changes, chunk_counts)

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
        # Clean up cloned repository if needed
        if cleanup_clone and repo_path.exists():
            try:
                logger.info("Cleaning up cloned repository: %s", repo_path)
                shutil.rmtree(repo_path, ignore_errors=True)
            except Exception as e:
                logger.warning("Failed to clean up cloned repository: %s", e)


def _run_incremental_pipeline(
    repo_id: uuid.UUID,
    repo_path: Path,
    language: str,
    cleanup_clone: bool = False,
) -> None:
    """
    Execute incremental re-indexing in a background thread.

    Only files that have changed since the last run are re-processed.
    
    Args:
        cleanup_clone: If True, remove the cloned repository after indexing.
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
    finally:
        # Clean up cloned repository if needed
        if cleanup_clone and repo_path.exists():
            try:
                logger.info("Cleaning up cloned repository: %s", repo_path)
                shutil.rmtree(repo_path, ignore_errors=True)
            except Exception as e:
                logger.warning("Failed to clean up cloned repository: %s", e)
