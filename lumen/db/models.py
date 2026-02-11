"""
SQLAlchemy ORM models mirroring the Supabase schema.

These models are used by the storage backend to read/write
structured data to Postgres.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ── Base ─────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """Shared declarative base for all Lumen ORM models."""

    pass


# ── Repositories ─────────────────────────────────────────────────────


class Repository(Base):
    __tablename__ = "repositories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    path_or_url: Mapped[str] = mapped_column(String, nullable=False)
    languages: Mapped[List[str]] = mapped_column(
        ARRAY(String), nullable=False, default=list
    )
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="pending"
    )
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    symbol_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    chunks: Mapped[List["CodeChunk"]] = relationship(
        back_populates="repository", cascade="all, delete-orphan"
    )
    symbols: Mapped[List["Symbol"]] = relationship(
        back_populates="repository", cascade="all, delete-orphan"
    )
    embeddings: Mapped[List["CodeEmbedding"]] = relationship(
        back_populates="repository", cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "path_or_url": self.path_or_url,
            "languages": self.languages,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "symbol_count": self.symbol_count,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ── Code Chunks ──────────────────────────────────────────────────────


class CodeChunk(Base):
    __tablename__ = "code_chunks"
    __table_args__ = (
        UniqueConstraint("repo_id", "chunk_id", name="uq_code_chunks_repo_chunk"),
        Index("idx_code_chunks_repo", "repo_id"),
        Index("idx_code_chunks_file", "repo_id", "file_path"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    repo_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    language: Mapped[str] = mapped_column(String, nullable=False)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    line_start: Mapped[int] = mapped_column(Integer, nullable=False)
    line_end: Mapped[int] = mapped_column(Integer, nullable=False)
    symbol_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    definition_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    relationship_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    complexity_hint: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    repository: Mapped["Repository"] = relationship(back_populates="chunks")


# ── Symbols ──────────────────────────────────────────────────────────


class Symbol(Base):
    __tablename__ = "symbols"
    __table_args__ = (
        Index("idx_symbols_repo", "repo_id"),
        Index("idx_symbols_kind", "repo_id", "kind"),
        Index("idx_symbols_chunk", "chunk_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    repo_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id: Mapped[str] = mapped_column(String, nullable=False)
    symbol_id: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False, default="UnspecifiedKind")
    display_name: Mapped[str] = mapped_column(String, nullable=False, default="")
    documentation: Mapped[str] = mapped_column(Text, nullable=False, default="")
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    line_start: Mapped[int] = mapped_column(Integer, nullable=False)
    line_end: Mapped[int] = mapped_column(Integer, nullable=False)
    relationships: Mapped[Any] = mapped_column(
        JSONB, nullable=False, default=list
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    repository: Mapped["Repository"] = relationship(back_populates="symbols")


# ── Code Embeddings ──────────────────────────────────────────────────


class CodeEmbedding(Base):
    """
    Stores vector embeddings for code chunks.

    NOTE: The ``embedding`` column is handled by LlamaIndex's
    ``PGVectorStore`` adapter and is defined as a pgvector ``vector(384)``
    in the migration SQL.  In the ORM we keep it as a generic Column
    because SQLAlchemy does not natively understand the ``vector`` type
    without the pgvector SQLAlchemy extension.  For direct ORM writes
    we bypass this column and let LlamaIndex manage it.
    """

    __tablename__ = "code_embeddings"
    __table_args__ = (
        Index("idx_code_embeddings_repo", "repo_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    repo_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[Any] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    repository: Mapped["Repository"] = relationship(back_populates="embeddings")
