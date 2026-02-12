"""
Pydantic request / response models for the Lumen REST API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────────────


class IndexRequest(BaseModel):
    """POST /api/v1/repos/index"""

    repo_path: str = Field(
        ..., description="Absolute path to the local repository to index."
    )
    language: Optional[str] = Field(
        None,
        description=(
            "Language(s) to index. 'auto' (default) to detect, "
            "or a comma-separated list like 'typescript,python'."
        ),
    )
    incremental: bool = Field(
        False,
        description=(
            "When True, only re-index files that changed since the last "
            "run.  Falls back to a full index if no previous index exists."
        ),
    )


class QueryRequest(BaseModel):
    """POST /api/v1/query"""

    repo_id: Optional[str] = Field(
        None,
        description=(
            "UUID of a single repository to scope the query to. "
            "Mutually exclusive with repo_ids."
        ),
    )
    repo_ids: Optional[List[str]] = Field(
        None,
        description=(
            "List of repository UUIDs to scope the query to (OR logic). "
            "Use this when querying across a known set of repos, e.g. "
            "a frontend + backend pair.  Mutually exclusive with repo_id."
        ),
    )
    question: str = Field(
        ..., description="Natural-language question about the codebase."
    )
    top_k: int = Field(
        5, ge=1, le=50, description="Number of results to return."
    )


# ── Responses ────────────────────────────────────────────────────────


class IndexResponse(BaseModel):
    """Response for POST /api/v1/repos/index"""

    repo_id: str
    status: str
    message: str


class RepoResponse(BaseModel):
    """A repository record."""

    id: str
    name: str
    path_or_url: str
    languages: List[str]
    status: str
    chunk_count: int
    symbol_count: int
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class QueryResultItem(BaseModel):
    """A single search hit."""

    score: float
    file_path: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    language: Optional[str] = None
    symbol_count: int = 0
    text_preview: str


class QueryResponse(BaseModel):
    """Response for POST /api/v1/query"""

    question: str
    results: List[QueryResultItem]
    total: int


class ChunkResponse(BaseModel):
    """A code chunk record (paginated list item)."""

    id: str
    chunk_id: str
    file_path: str
    language: str
    line_start: int
    line_end: int
    symbol_count: int
    definition_count: int
    relationship_count: int
    complexity_hint: float
    code: str


class ChunkListResponse(BaseModel):
    """Paginated list of chunks."""

    repo_id: str
    chunks: List[ChunkResponse]
    offset: int
    limit: int
    total: int


class HealthResponse(BaseModel):
    """GET /api/v1/health"""

    status: str
    storage_backend: str
    db_connected: Optional[bool] = None
    version: str
