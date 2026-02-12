"""
Pydantic request / response models for the Lumen REST API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ── Requests ─────────────────────────────────────────────────────────


class IndexRequest(BaseModel):
    """POST /api/v1/repos/index

    Provide **either** ``repo_path`` (local directory) **or**
    ``git_url`` (remote clone URL), not both.
    """

    repo_path: Optional[str] = Field(
        None, description="Absolute path to a local repository to index."
    )
    git_url: Optional[str] = Field(
        None,
        description=(
            "Git clone URL (HTTPS or SSH).  The repo will be shallow-cloned "
            "to a temp directory, indexed, then cleaned up."
        ),
    )
    branch: Optional[str] = Field(
        None,
        description=(
            "Branch to clone when using git_url.  "
            "Defaults to the repository's default branch."
        ),
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

    @model_validator(mode="after")
    def _check_source(self) -> "IndexRequest":
        if not self.repo_path and not self.git_url:
            raise ValueError("Provide either repo_path or git_url.")
        if self.repo_path and self.git_url:
            raise ValueError("Provide either repo_path or git_url, not both.")
        if self.branch and not self.git_url:
            raise ValueError("branch is only valid when using git_url.")
        return self


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
