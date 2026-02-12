"""
Query engine — semantic search over the SCIP-enriched code index.

Provides both a programmatic API and a simple REPL for interactive
exploration.  Supports optional metadata filtering so callers can
scope queries to specific repositories.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

logger = logging.getLogger(__name__)


# ── Data transfer object ─────────────────────────────────────────────


class QueryResult:
    """Lightweight wrapper around a ranked retrieval hit."""

    def __init__(self, node: NodeWithScore) -> None:
        self._node = node

    @property
    def score(self) -> float:
        return self._node.score or 0.0

    @property
    def text(self) -> str:
        return self._node.node.get_content()

    @property
    def metadata(self) -> dict:
        return self._node.node.metadata  # type: ignore[union-attr]

    @property
    def file_path(self) -> str:
        return self.metadata.get("file_path", "")

    @property
    def line_range(self) -> str:
        return f"L{self.metadata.get('line_start', '?')}–L{self.metadata.get('line_end', '?')}"

    @property
    def symbol_count(self) -> int:
        return self.metadata.get("symbol_count", 0)

    def summary(self) -> str:
        return (
            f"[{self.score:.4f}]  {self.file_path}  {self.line_range}  "
            f"({self.symbol_count} SCIP symbols)"
        )

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "file_path": self.file_path,
            "line_start": self.metadata.get("line_start"),
            "line_end": self.metadata.get("line_end"),
            "symbol_count": self.symbol_count,
            "language": self.metadata.get("language"),
            "text_preview": self.text[:500],
        }


# ── Query helpers ────────────────────────────────────────────────────


def _build_repo_filters(
    repo_id: Optional[str] = None,
    repo_ids: Optional[List[str]] = None,
) -> Optional[MetadataFilters]:
    """
    Build a ``MetadataFilters`` object for repository-scoped queries.

    * ``repo_id``  — filter to a single repository (EQ).
    * ``repo_ids`` — filter to one of several repositories (OR of EQs).
    * Both ``None`` — no filter (search across all repos).
    """
    if repo_id:
        return MetadataFilters(
            filters=[
                MetadataFilter(
                    key="repo_id",
                    value=repo_id,
                    operator=FilterOperator.EQ,
                ),
            ],
        )

    if repo_ids:
        return MetadataFilters(
            filters=[
                MetadataFilter(
                    key="repo_id",
                    value=rid,
                    operator=FilterOperator.EQ,
                )
                for rid in repo_ids
            ],
            condition=FilterCondition.OR,
        )

    return None


def query_index(
    question: str,
    index: Optional[VectorStoreIndex] = None,
    top_k: int = 5,
    repo_id: Optional[str] = None,
    repo_ids: Optional[List[str]] = None,
) -> List[QueryResult]:
    """
    Run a semantic search against the Lumen code index.

    Parameters
    ----------
    question:
        Natural-language question, e.g.
        *"What does the authenticateUser function do?"*
    index:
        An already-loaded ``VectorStoreIndex``.  If ``None``, the index
        is loaded from Supabase.
    top_k:
        Number of results to return.
    repo_id:
        When provided, restrict results to chunks from this single
        repository (UUID string).
    repo_ids:
        When provided, restrict results to chunks from any of these
        repositories (list of UUID strings, OR logic).

    Returns
    -------
    list[QueryResult]
        Ranked list of code chunks relevant to the question.
    """
    if index is None:
        from lumen.storage.supabase_store import load_index

        index = load_index()
        if index is None:
            logger.error("No index available. Run the indexer first.")
            return []

    filters = _build_repo_filters(repo_id=repo_id, repo_ids=repo_ids)

    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=filters,
    )
    raw_results: List[NodeWithScore] = retriever.retrieve(question)
    return [QueryResult(r) for r in raw_results]


def query_function(
    function_name: str,
    index: Optional[VectorStoreIndex] = None,
    top_k: int = 3,
    repo_id: Optional[str] = None,
    repo_ids: Optional[List[str]] = None,
) -> List[QueryResult]:
    """
    Convenience wrapper: ask the index about a specific function.

    Constructs a targeted question and returns the top matches.
    """
    question = (
        f"What is the purpose of the function or method named '{function_name}'? "
        f"Explain its role, parameters, return value, and how it relates to "
        f"other parts of the codebase."
    )
    return query_index(
        question,
        index=index,
        top_k=top_k,
        repo_id=repo_id,
        repo_ids=repo_ids,
    )


# ── Interactive REPL ─────────────────────────────────────────────────


def interactive_repl(
    repo_id: Optional[str] = None,
    repo_ids: Optional[List[str]] = None,
) -> None:
    """
    Launch a minimal interactive query loop.

    Parameters
    ----------
    repo_id:
        When provided, every query is scoped to this single repo.
    repo_ids:
        When provided, every query is scoped to these repos (OR).
    """
    from lumen.storage.supabase_store import load_index

    index = load_index()
    if index is None:
        print("ERROR: No index available. Run the indexer first.")
        return

    print("─" * 60)
    print("Lumen Query REPL  (type 'exit' or Ctrl-C to quit)")
    if repo_id:
        print(f"  Scoped to repo: {repo_id}")
    elif repo_ids:
        print(f"  Scoped to repos: {', '.join(repo_ids)}")
    print("─" * 60)

    while True:
        try:
            question = input("\n❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question or question.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        results = query_index(
            question,
            index=index,
            top_k=5,
            repo_id=repo_id,
            repo_ids=repo_ids,
        )
        if not results:
            print("  (no results)")
            continue

        for i, r in enumerate(results, 1):
            print(f"\n  ── Result {i} {r.summary()}")
            # Show first 20 lines of the chunk text.
            preview = "\n".join(r.text.splitlines()[:20])
            print(f"{preview}")
            if len(r.text.splitlines()) > 20:
                print("    …")
