"""
Shared dependencies for the FastAPI application.

Provides lazy-loaded singletons and dependency injection helpers
for use with ``Depends()``.
"""

from __future__ import annotations

import logging
from typing import Optional

from llama_index.core import VectorStoreIndex

logger = logging.getLogger(__name__)

# ── Cached index ─────────────────────────────────────────────────────

_cached_index: Optional[VectorStoreIndex] = None


def get_query_index() -> Optional[VectorStoreIndex]:
    """
    Return the Supabase-backed ``VectorStoreIndex`` for querying.

    Caches the result so subsequent requests reuse it.
    """
    global _cached_index

    if _cached_index is not None:
        return _cached_index

    from lumen.storage.supabase_store import load_index

    idx = load_index()
    if idx is not None:
        _cached_index = idx
    return idx


def invalidate_index_cache() -> None:
    """Clear the cached index so the next query reloads from storage."""
    global _cached_index
    _cached_index = None
    logger.info("Query index cache invalidated.")
