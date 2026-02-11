"""
Supabase Postgres + pgvector storage backend for Lumen.

Supabase Postgres + pgvector storage backend for Lumen.

Responsibilities:
    1. Write structured data (repos, chunks, symbols) to relational tables.
    2. Write + query vector embeddings via LlamaIndex's ``PGVectorStore``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from lumen.config import (
    DATABASE_URL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    IndexedChunk,
    PG_EMBED_TABLE,
)
from lumen.db.models import CodeChunk, CodeEmbedding, Repository, Symbol
from lumen.db.session import get_session

logger = logging.getLogger(__name__)


# ── Embedding singleton ───────────────────────────────────────────────

_embedding_initialised = False


def _ensure_embedding_model() -> None:
    """Set the global LlamaIndex embedding model (local HuggingFace)."""
    global _embedding_initialised
    if _embedding_initialised:
        return
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    _embedding_initialised = True


# ── PGVectorStore helpers ────────────────────────────────────────────


def _get_pg_vector_store():
    """Lazy-import and instantiate the PGVectorStore."""
    from llama_index.vector_stores.postgres import PGVectorStore

    return PGVectorStore.from_params(
        database=_extract_dbname(DATABASE_URL),
        host=_extract_host(DATABASE_URL),
        port=str(_extract_port(DATABASE_URL)),
        user=_extract_user(DATABASE_URL),
        password=_extract_password(DATABASE_URL),
        table_name=PG_EMBED_TABLE,
        embed_dim=EMBEDDING_DIM,
    )


# ── Public API ────────────────────────────────────────────────────────


def build_index(
    nodes: List[TextNode],
    repo_id: Optional[uuid.UUID] = None,
) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex backed by Supabase pgvector.

    Also writes structured relational data (chunks + symbols) to
    Postgres if *repo_id* is provided.
    """
    _ensure_embedding_model()

    vector_store = _get_pg_vector_store()
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    logger.info("Building pgvector index with %d nodes …", len(nodes))
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_ctx,
        show_progress=True,
    )
    logger.info("pgvector index built (%d nodes).", len(nodes))
    return index


def load_index() -> Optional[VectorStoreIndex]:
    """Load an existing index from the Supabase pgvector table."""
    _ensure_embedding_model()

    vector_store = _get_pg_vector_store()
    try:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        logger.info("Loaded pgvector index from Supabase.")
        return index
    except Exception as exc:
        logger.warning("Could not load pgvector index: %s", exc)
        return None


# ── Relational writes ────────────────────────────────────────────────


def create_repository(
    name: str,
    path_or_url: str,
    languages: List[str],
) -> uuid.UUID:
    """Insert a new ``repositories`` row and return its UUID."""
    repo_id = uuid.uuid4()
    with get_session() as session:
        repo = Repository(
            id=repo_id,
            name=name,
            path_or_url=path_or_url,
            languages=languages,
            status="pending",
        )
        session.add(repo)
    logger.info("Created repository %s (id=%s)", name, repo_id)
    return repo_id


def update_repository_status(
    repo_id: uuid.UUID,
    status: str,
    *,
    chunk_count: Optional[int] = None,
    symbol_count: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """Update a repository's status (and optional counters)."""
    with get_session() as session:
        repo = session.get(Repository, repo_id)
        if repo is None:
            logger.warning("Repository %s not found — skipping status update.", repo_id)
            return
        repo.status = status
        if chunk_count is not None:
            repo.chunk_count = chunk_count
        if symbol_count is not None:
            repo.symbol_count = symbol_count
        if error_message is not None:
            repo.error_message = error_message
        repo.updated_at = datetime.now(timezone.utc)
    logger.info("Repository %s → status=%s", repo_id, status)


def get_repository(repo_id: uuid.UUID) -> Optional[Dict[str, Any]]:
    """Fetch a single repository as a dict."""
    with get_session() as session:
        repo = session.get(Repository, repo_id)
        return repo.to_dict() if repo else None


def list_repositories() -> List[Dict[str, Any]]:
    """Return all repositories (newest first)."""
    with get_session() as session:
        repos = (
            session.query(Repository)
            .order_by(Repository.created_at.desc())
            .all()
        )
        return [r.to_dict() for r in repos]


def persist_chunks(
    repo_id: uuid.UUID,
    chunks: List[IndexedChunk],
) -> int:
    """
    Bulk-insert ``IndexedChunk`` data into the ``code_chunks`` and
    ``symbols`` tables.  Returns the number of chunks written.
    """
    with get_session() as session:
        # Delete old data for this repo (allows re-index)
        session.query(Symbol).filter(Symbol.repo_id == repo_id).delete()
        session.query(CodeChunk).filter(CodeChunk.repo_id == repo_id).delete()

        chunk_rows = []
        symbol_rows = []

        for chunk in chunks:
            chunk_rows.append(
                CodeChunk(
                    repo_id=repo_id,
                    chunk_id=chunk.chunk_id,
                    file_path=chunk.file_path,
                    language=chunk.language,
                    code=chunk.code,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    symbol_count=chunk.symbol_count,
                    definition_count=chunk.definition_count,
                    relationship_count=chunk.relationship_count,
                    complexity_hint=chunk.complexity_hint,
                )
            )
            for sym in chunk.symbols:
                symbol_rows.append(
                    Symbol(
                        repo_id=repo_id,
                        chunk_id=chunk.chunk_id,
                        symbol_id=sym.symbol_id,
                        kind=sym.kind,
                        display_name=sym.display_name,
                        documentation=sym.documentation,
                        file_path=sym.file_path,
                        line_start=sym.line_start,
                        line_end=sym.line_end,
                        relationships=sym.relationships,
                    )
                )

        session.bulk_save_objects(chunk_rows)
        session.bulk_save_objects(symbol_rows)

    logger.info(
        "Persisted %d chunks and %d symbols for repo %s",
        len(chunk_rows),
        len(symbol_rows),
        repo_id,
    )
    return len(chunk_rows)


def get_chunks_for_repo(
    repo_id: uuid.UUID,
    *,
    offset: int = 0,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Paginated list of code chunks for a repository."""
    with get_session() as session:
        rows = (
            session.query(CodeChunk)
            .filter(CodeChunk.repo_id == repo_id)
            .order_by(CodeChunk.file_path, CodeChunk.line_start)
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [
            {
                "id": str(r.id),
                "chunk_id": r.chunk_id,
                "file_path": r.file_path,
                "language": r.language,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "symbol_count": r.symbol_count,
                "definition_count": r.definition_count,
                "relationship_count": r.relationship_count,
                "complexity_hint": r.complexity_hint,
                "code": r.code,
            }
            for r in rows
        ]


def delete_repo_data(repo_id: uuid.UUID) -> None:
    """Delete all data for a repository (cascades via FK)."""
    with get_session() as session:
        repo = session.get(Repository, repo_id)
        if repo:
            session.delete(repo)
            logger.info("Deleted repository %s and all related data.", repo_id)


# ── URL parsing helpers ──────────────────────────────────────────────
# We parse the DATABASE_URL ourselves so we can pass individual params
# to PGVectorStore.from_params().


def _extract_host(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).hostname or "localhost"


def _extract_port(url: str) -> int:
    from urllib.parse import urlparse
    return urlparse(url).port or 5432


def _extract_user(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).username or "postgres"


def _extract_password(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).password or ""


def _extract_dbname(url: str) -> str:
    from urllib.parse import urlparse
    path = urlparse(url).path
    return path.lstrip("/") if path else "postgres"
