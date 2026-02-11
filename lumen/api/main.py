"""
Lumen API — FastAPI entry point.

Start with:
    python -m lumen.api.main

Or:
    uvicorn lumen.api.main:app --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lumen import __version__
from lumen.api.routes import router

logger = logging.getLogger("lumen.api")


# ── Lifespan ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    # Try loading .env if python-dotenv is installed
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Loaded .env file.")
    except ImportError:
        pass

    logger.info("Lumen API v%s starting  (storage=supabase/pgvector)", __version__)
    yield
    logger.info("Lumen API shutting down.")


# ── App factory ──────────────────────────────────────────────────────


app = FastAPI(
    title="Lumen — Repository Indexing Service",
    description=(
        "SCIP-enriched code intelligence API.\n\n"
        "Index local repositories and run semantic queries against them."
    ),
    version=__version__,
    lifespan=lifespan,
)

# CORS — permissive for now; tighten when there's a frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ── Convenience: ``python -m lumen.api.main`` ───────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))

    uvicorn.run(
        "lumen.api.main:app",
        host=host,
        port=port,
        reload=True,
    )
