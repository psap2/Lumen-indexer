"""
Shared dependencies for the FastAPI application.

Provides lazy-loaded singletons and dependency injection helpers
for use with ``Depends()``.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from llama_index.core import VectorStoreIndex

from lumen.config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, STORAGE_BACKEND

logger = logging.getLogger(__name__)

# ── Cached index map ─────────────────────────────────────────────────
# In Supabase mode there's one global index.
# In ChromaDB mode each repo has its own index keyed by persist_dir.

_index_cache: Dict[str, VectorStoreIndex] = {}


def get_query_index(
    persist_dir: Optional[str] = None,
    collection_name: str = CHROMA_COLLECTION,
) -> Optional[VectorStoreIndex]:
    """
    Return a ``VectorStoreIndex`` for querying.

    Loads from the configured backend (ChromaDB or Supabase) and caches
    the result so subsequent requests reuse it.

    For ChromaDB mode, *persist_dir* identifies which repo's index to load.
    """
    cache_key = "supabase" if STORAGE_BACKEND == "supabase" else (persist_dir or "default")

    if cache_key in _index_cache:
        return _index_cache[cache_key]

    if STORAGE_BACKEND == "supabase":
        from lumen.storage.supabase_store import load_index
        idx = load_index()
    else:
        from lumen.storage.vector_store import load_index
        idx = load_index(
            persist_dir=persist_dir or CHROMA_PERSIST_DIR,
            collection_name=collection_name,
        )

    if idx is not None:
        _index_cache[cache_key] = idx
    return idx


def invalidate_index_cache() -> None:
    """Clear all cached indexes so the next query reloads from storage."""
    _index_cache.clear()
    logger.info("Query index cache invalidated.")
