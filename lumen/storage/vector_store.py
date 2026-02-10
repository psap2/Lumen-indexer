"""
ChromaDB-backed vector store for Lumen.

Wraps LlamaIndex's ``ChromaVectorStore`` with Lumen-specific defaults
and provides helpers for index lifecycle management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from lumen.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


_embedding_initialised = False


def _ensure_embedding_model() -> None:
    """
    Configure the global LlamaIndex embedding model.

    We use a local HuggingFace model so that indexing works entirely
    offline without an API key.

    NOTE: We must NOT read ``Settings.embed_model`` before setting it,
    because LlamaIndex lazily resolves the default (OpenAI) on first
    access, which throws if ``llama-index-embeddings-openai`` is not
    installed.  Instead we use a module-level flag.
    """
    global _embedding_initialised
    if _embedding_initialised:
        return

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
    )
    _embedding_initialised = True


def get_chroma_client(persist_dir: str = CHROMA_PERSIST_DIR) -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client rooted at *persist_dir*."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def build_index(
    nodes: List[TextNode],
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION,
) -> VectorStoreIndex:
    """
    Build (or overwrite) a VectorStoreIndex backed by ChromaDB.

    Parameters
    ----------
    nodes:
        Enriched ``TextNode`` objects from the ingestion pipeline.
    persist_dir:
        Filesystem path for ChromaDB's persistent storage.
    collection_name:
        Name of the ChromaDB collection.

    Returns
    -------
    VectorStoreIndex
        Ready for querying.
    """
    _ensure_embedding_model()

    client = get_chroma_client(persist_dir)

    # Delete existing collection to allow full re-index.
    try:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection '%s'", collection_name)
    except Exception:  # noqa: BLE001
        pass

    chroma_collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    logger.info("Building vector index with %d nodes …", len(nodes))
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_ctx,
        show_progress=True,
    )
    logger.info("Vector index built and persisted to %s", persist_dir)
    return index


def load_index(
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION,
) -> Optional[VectorStoreIndex]:
    """
    Load an existing VectorStoreIndex from a persisted ChromaDB collection.

    Returns ``None`` if the collection does not exist or is empty.
    """
    _ensure_embedding_model()

    client = get_chroma_client(persist_dir)

    try:
        chroma_collection = client.get_collection(collection_name)
    except Exception:  # noqa: BLE001
        logger.warning("Collection '%s' not found in %s", collection_name, persist_dir)
        return None

    if chroma_collection.count() == 0:
        logger.warning("Collection '%s' is empty", collection_name)
        return None

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    logger.info(
        "Loaded index from %s (%d vectors)",
        persist_dir,
        chroma_collection.count(),
    )
    return index
