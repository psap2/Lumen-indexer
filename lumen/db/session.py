"""
Database session management for the Supabase / Postgres backend.

Provides a ``get_engine()`` singleton and a ``get_session()`` context manager
for transactional work.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def get_database_url() -> str:
    """
    Read the Postgres connection string from the environment.

    Looks for ``DATABASE_URL`` first, then falls back to constructing one
    from individual ``PGHOST``, ``PGUSER``, etc. vars (common in Supabase).
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return url

    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    user = os.environ.get("PGUSER", "postgres")
    password = os.environ.get("PGPASSWORD", "")
    dbname = os.environ.get("PGDATABASE", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def get_engine() -> Engine:
    """Return (or create) the singleton SQLAlchemy ``Engine``."""
    global _engine
    if _engine is None:
        url = get_database_url()
        logger.info("Creating DB engine → %s", _redact(url))
        _engine = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> sessionmaker:
    """Return (or create) the singleton ``sessionmaker``."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager that yields a SQLAlchemy ``Session``.

    Commits on clean exit, rolls back on exception.
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """Quick connectivity check — returns ``True`` if the DB is reachable."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database connection check failed: %s", exc)
        return False


def _redact(url: str) -> str:
    """Redact password from a connection URL for logging."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.password:
            return url.replace(f":{parsed.password}@", ":***@")
    except Exception:
        pass
    return url
