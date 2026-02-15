"""
Tests for lumen.api.schemas — Pydantic request/response model validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lumen.api.schemas import (
    ChunkResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    QueryResultItem,
    RepoResponse,
)


# ── IndexRequest ─────────────────────────────────────────────────────


class TestIndexRequest:
    def test_valid_repo_path(self):
        req = IndexRequest(repo_path="/path/to/repo")
        assert req.repo_path == "/path/to/repo"
        assert req.git_url is None

    def test_valid_git_url(self):
        req = IndexRequest(git_url="https://github.com/org/repo.git")
        assert req.git_url == "https://github.com/org/repo.git"
        assert req.repo_path is None

    def test_git_url_with_branch(self):
        req = IndexRequest(
            git_url="https://github.com/org/repo.git",
            branch="develop",
        )
        assert req.branch == "develop"

    def test_neither_source_raises(self):
        with pytest.raises(ValidationError, match="repo_path or git_url"):
            IndexRequest()

    def test_both_sources_raises(self):
        with pytest.raises(ValidationError, match="repo_path or git_url"):
            IndexRequest(
                repo_path="/path/to/repo",
                git_url="https://github.com/org/repo.git",
            )

    def test_branch_without_git_url_raises(self):
        with pytest.raises(ValidationError, match="branch is only valid"):
            IndexRequest(repo_path="/path/to/repo", branch="main")

    def test_incremental_default_false(self):
        req = IndexRequest(repo_path="/path")
        assert req.incremental is False

    def test_incremental_true(self):
        req = IndexRequest(repo_path="/path", incremental=True)
        assert req.incremental is True

    def test_language_optional(self):
        req = IndexRequest(repo_path="/path")
        assert req.language is None

    def test_language_provided(self):
        req = IndexRequest(repo_path="/path", language="python")
        assert req.language == "python"


# ── QueryRequest ─────────────────────────────────────────────────────


class TestQueryRequest:
    def test_basic_query(self):
        req = QueryRequest(question="What does foo do?")
        assert req.question == "What does foo do?"
        assert req.top_k == 5

    def test_custom_top_k(self):
        req = QueryRequest(question="test", top_k=10)
        assert req.top_k == 10

    def test_top_k_min(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="test", top_k=0)

    def test_top_k_max(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="test", top_k=51)

    def test_repo_id_filter(self):
        req = QueryRequest(
            question="test",
            repo_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert req.repo_id is not None

    def test_repo_ids_filter(self):
        req = QueryRequest(
            question="test",
            repo_ids=["id-1", "id-2"],
        )
        assert req.repo_ids is not None
        assert len(req.repo_ids) == 2

    def test_question_required(self):
        with pytest.raises(ValidationError):
            QueryRequest()  # type: ignore[call-arg]


# ── Response models ──────────────────────────────────────────────────


class TestIndexResponse:
    def test_valid(self):
        resp = IndexResponse(
            repo_id="123", status="indexing", message="Started",
        )
        assert resp.repo_id == "123"
        assert resp.status == "indexing"


class TestRepoResponse:
    def test_valid(self):
        resp = RepoResponse(
            id="123",
            name="my-repo",
            path_or_url="/path",
            languages=["python"],
            status="ready",
            chunk_count=100,
            symbol_count=50,
        )
        assert resp.name == "my-repo"
        assert resp.languages == ["python"]

    def test_optional_fields(self):
        resp = RepoResponse(
            id="123",
            name="my-repo",
            path_or_url="/path",
            languages=[],
            status="ready",
            chunk_count=0,
            symbol_count=0,
        )
        assert resp.error_message is None
        assert resp.created_at is None


class TestQueryResultItem:
    def test_valid(self):
        item = QueryResultItem(
            score=0.95,
            file_path="src/main.py",
            text_preview="def hello():",
        )
        assert item.score == 0.95
        assert item.symbol_count == 0  # default

    def test_all_fields(self):
        item = QueryResultItem(
            score=0.9,
            file_path="src/main.py",
            line_start=10,
            line_end=25,
            language="python",
            symbol_count=3,
            text_preview="def foo():",
        )
        assert item.line_start == 10
        assert item.language == "python"


class TestQueryResponse:
    def test_valid(self):
        resp = QueryResponse(
            question="What does foo do?",
            results=[],
            total=0,
        )
        assert resp.total == 0
        assert resp.results == []


class TestChunkResponse:
    def test_valid(self):
        chunk = ChunkResponse(
            id="uuid-1",
            chunk_id="abc123",
            file_path="src/main.py",
            language="python",
            line_start=0,
            line_end=60,
            symbol_count=5,
            definition_count=2,
            relationship_count=3,
            complexity_hint=0.1234,
            code="def hello(): pass",
        )
        assert chunk.chunk_id == "abc123"
        assert chunk.complexity_hint == 0.1234


class TestHealthResponse:
    def test_valid(self):
        resp = HealthResponse(
            status="ok",
            storage_backend="supabase",
            version="0.1.0",
        )
        assert resp.status == "ok"
        assert resp.db_connected is None

    def test_with_db_connected(self):
        resp = HealthResponse(
            status="ok",
            storage_backend="supabase",
            db_connected=True,
            version="0.1.0",
        )
        assert resp.db_connected is True
