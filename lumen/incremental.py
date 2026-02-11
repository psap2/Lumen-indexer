"""
Incremental indexing — file-level change detection.

Computes SHA-256 content hashes for every source file and compares
them against previously stored state so the indexing pipeline can
skip files that have not changed since the last run.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

from lumen.config import DEFAULT_IGNORE_PATTERNS
from lumen.db.models import FileIndexState
from lumen.db.session import get_session

logger = logging.getLogger(__name__)


# ── Data types ───────────────────────────────────────────────────────


@dataclass
class FileChange:
    """Describes the change status of a single source file."""

    path: str  # Relative to repo root
    status: Literal["new", "modified", "deleted", "unchanged"]
    content_hash: str  # SHA-256 hex; empty string for deleted files


# ── Hashing ──────────────────────────────────────────────────────────


def compute_file_hash(filepath: Path) -> str:
    """Return the SHA-256 hex digest of *filepath*'s contents."""
    h = hashlib.sha256()
    with open(filepath, "rb") as fh:
        while True:
            block = fh.read(65_536)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ── Change detection ─────────────────────────────────────────────────


def _should_ignore(path: Path, ignore: List[str]) -> bool:
    return any(ig in path.parts for ig in ignore)


def _collect_files_with_hashes(
    repo_root: Path,
    extensions: frozenset[str],
    ignore_patterns: List[str],
) -> Dict[str, str]:
    """
    Walk the repo and return ``{relative_path: sha256_hex}`` for every
    source file matching *extensions*.
    """
    result: Dict[str, str] = {}
    for p in sorted(repo_root.rglob("*")):
        if not p.is_file():
            continue
        if _should_ignore(p, ignore_patterns):
            continue
        if p.suffix.lower() in extensions:
            rel = str(p.relative_to(repo_root))
            result[rel] = compute_file_hash(p)
    return result


def detect_file_changes(
    repo_id: uuid.UUID,
    repo_root: Path,
    extensions: frozenset[str],
    ignore_patterns: Optional[List[str]] = None,
) -> List[FileChange]:
    """
    Compare current on-disk files against stored ``FileIndexState`` rows
    to produce a categorised list of changes.

    Returns a list of :class:`FileChange` — one per file that is new,
    modified, deleted, or unchanged.
    """
    if ignore_patterns is None:
        ignore_patterns = list(DEFAULT_IGNORE_PATTERNS)

    # 1. Compute current file hashes
    current_files = _collect_files_with_hashes(repo_root, extensions, ignore_patterns)

    # 2. Load stored file states
    stored: Dict[str, str] = {}  # file_path → content_hash
    with get_session() as session:
        rows = (
            session.query(FileIndexState)
            .filter(FileIndexState.repo_id == repo_id)
            .all()
        )
        for row in rows:
            stored[row.file_path] = row.content_hash

    # 3. Categorise
    changes: List[FileChange] = []

    for rel_path, current_hash in current_files.items():
        if rel_path not in stored:
            changes.append(FileChange(path=rel_path, status="new", content_hash=current_hash))
        elif stored[rel_path] != current_hash:
            changes.append(FileChange(path=rel_path, status="modified", content_hash=current_hash))
        else:
            changes.append(FileChange(path=rel_path, status="unchanged", content_hash=current_hash))

    for rel_path in stored:
        if rel_path not in current_files:
            changes.append(FileChange(path=rel_path, status="deleted", content_hash=""))

    return changes


# ── Persistence ──────────────────────────────────────────────────────


def save_file_states(
    repo_id: uuid.UUID,
    changes: List[FileChange],
    chunk_counts: Optional[Dict[str, int]] = None,
) -> None:
    """
    Upsert ``FileIndexState`` rows after a successful indexing run.

    - *new* and *modified* files get their hash updated.
    - *deleted* files get their row removed.
    - *unchanged* files are left as-is.

    *chunk_counts* maps ``relative_path → number_of_chunks`` for files
    that were actually ingested.
    """
    if chunk_counts is None:
        chunk_counts = {}

    with get_session() as session:
        for change in changes:
            if change.status == "deleted":
                session.query(FileIndexState).filter(
                    FileIndexState.repo_id == repo_id,
                    FileIndexState.file_path == change.path,
                ).delete()
                continue

            if change.status in ("new", "modified"):
                existing = (
                    session.query(FileIndexState)
                    .filter(
                        FileIndexState.repo_id == repo_id,
                        FileIndexState.file_path == change.path,
                    )
                    .first()
                )
                if existing:
                    existing.content_hash = change.content_hash
                    existing.chunk_count = chunk_counts.get(change.path, 0)
                    existing.indexed_at = datetime.now(timezone.utc)
                else:
                    session.add(
                        FileIndexState(
                            repo_id=repo_id,
                            file_path=change.path,
                            content_hash=change.content_hash,
                            chunk_count=chunk_counts.get(change.path, 0),
                        )
                    )

    logger.info("Saved file index state for repo %s", repo_id)


# ── Reporting ────────────────────────────────────────────────────────


def get_change_summary(changes: List[FileChange]) -> str:
    """Return a human-readable one-line summary of changes."""
    counts: Counter[str] = Counter(c.status for c in changes)
    parts = []
    for status in ("new", "modified", "deleted", "unchanged"):
        n = counts.get(status, 0)
        if n:
            parts.append(f"{n} {status}")
    return ", ".join(parts) if parts else "no files found"


def changed_file_paths(changes: List[FileChange]) -> set[str]:
    """Return the set of relative paths for new + modified files."""
    return {c.path for c in changes if c.status in ("new", "modified")}


def stale_file_paths(changes: List[FileChange]) -> set[str]:
    """Return the set of relative paths for modified + deleted files."""
    return {c.path for c in changes if c.status in ("modified", "deleted")}
