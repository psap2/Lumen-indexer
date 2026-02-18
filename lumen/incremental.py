"""
Incremental indexing — file-level change detection.

Two strategies for detecting changed files:

1. **Git-diff (fast path):** When the repo is a git repo and we have a
   stored commit SHA from the last index, we run ``git diff`` to find
   changed files instantly.  This is O(1) regardless of repo size.

2. **Hash-based (fallback):** Walk every source file and compute a
   SHA-256 content hash, comparing against stored ``FileIndexState``
   rows.  This is O(N) in the number of files but works for non-git
   repos and as a safety net.

The public ``detect_file_changes()`` function tries git-diff first and
falls back to hash-based automatically.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

from lumen.config import DEFAULT_IGNORE_PATTERNS
from lumen.db.models import FileIndexState, Repository
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


# ── Git helpers ──────────────────────────────────────────────────────


def _is_git_repo(repo_root: Path) -> bool:
    """Return ``True`` if *repo_root* contains a ``.git`` directory."""
    return (repo_root / ".git").is_dir()


def _get_head_commit(repo_root: Path) -> Optional[str]:
    """Return the current HEAD commit SHA, or ``None`` on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _is_commit_reachable(repo_root: Path, commit_sha: str) -> bool:
    """Return ``True`` if *commit_sha* exists in the repo's object store."""
    try:
        result = subprocess.run(
            ["git", "cat-file", "-t", commit_sha],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() == "commit"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _git_diff_files(
    repo_root: Path,
    base_commit: str,
) -> Optional[Dict[str, str]]:
    """
    Run ``git diff --name-status <base_commit>`` to find tracked files
    that differ between *base_commit* and the current working tree.

    Returns ``{relative_path: status}`` where status is one of
    ``A`` (added), ``M`` (modified), ``D`` (deleted), ``R`` (renamed).
    Returns ``None`` on failure.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", base_commit],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    changes: Dict[str, str] = {}
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        # Renames: R100\told_path\tnew_path
        if status.startswith("R"):
            old_path = parts[1]
            new_path = parts[2] if len(parts) > 2 else parts[1]
            changes[old_path] = "D"
            changes[new_path] = "A"
        else:
            changes[parts[1]] = status[0]  # Take first char (e.g. "M", "A", "D")

    return changes


def _git_untracked_files(repo_root: Path) -> Optional[List[str]]:
    """
    Run ``git ls-files --others --exclude-standard`` to find untracked
    files (new files not yet added to git).

    Returns a list of relative paths, or ``None`` on failure.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    return [
        line for line in result.stdout.strip().splitlines() if line
    ]


# ── Git-diff change detection (fast path) ────────────────────────────


def detect_file_changes_git(
    repo_root: Path,
    last_commit: str,
    extensions: frozenset[str],
    ignore_patterns: Optional[List[str]] = None,
) -> Optional[List[FileChange]]:
    """
    Detect changed files using ``git diff`` against a stored commit.

    Returns a list of :class:`FileChange` objects for files that are
    new, modified, or deleted — but does **not** enumerate unchanged
    files (unlike the hash-based approach).  This is fine because
    downstream consumers only care about non-unchanged entries.

    Returns ``None`` if git operations fail (caller should fall back
    to hash-based detection).
    """
    if ignore_patterns is None:
        ignore_patterns = list(DEFAULT_IGNORE_PATTERNS)

    if not _is_git_repo(repo_root):
        return None

    if not _is_commit_reachable(repo_root, last_commit):
        logger.warning(
            "Stored commit %s not reachable — falling back to hash-based detection.",
            last_commit[:12],
        )
        return None

    # Get tracked file changes
    diff_files = _git_diff_files(repo_root, last_commit)
    if diff_files is None:
        return None

    # Get untracked (new) files
    untracked = _git_untracked_files(repo_root)
    if untracked is None:
        return None

    changes: List[FileChange] = []

    # Map git statuses → FileChange statuses
    _STATUS_MAP = {"A": "new", "M": "modified", "D": "deleted"}

    for rel_path, git_status in diff_files.items():
        # Filter by extension
        suffix = Path(rel_path).suffix.lower()
        if suffix not in extensions:
            continue
        # Filter by ignore patterns
        if _should_ignore(Path(rel_path), ignore_patterns):
            continue

        fc_status = _STATUS_MAP.get(git_status, "modified")
        content_hash = ""
        if fc_status != "deleted":
            full_path = repo_root / rel_path
            if full_path.is_file():
                content_hash = compute_file_hash(full_path)
        changes.append(FileChange(path=rel_path, status=fc_status, content_hash=content_hash))

    # Add untracked files as "new"
    for rel_path in untracked:
        if rel_path in diff_files:
            continue  # Already handled above
        suffix = Path(rel_path).suffix.lower()
        if suffix not in extensions:
            continue
        if _should_ignore(Path(rel_path), ignore_patterns):
            continue
        full_path = repo_root / rel_path
        if full_path.is_file():
            content_hash = compute_file_hash(full_path)
            changes.append(FileChange(path=rel_path, status="new", content_hash=content_hash))

    return changes


# ── Hash-based change detection (fallback) ───────────────────────────


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
    Compare current on-disk files against the last indexed state to
    produce a categorised list of changes.

    Tries the **git-diff fast path** first (O(1) via ``git diff``).
    Falls back to **hash-based detection** (O(N) full scan) when git
    is unavailable, the stored commit is missing, or git commands fail.

    Returns a list of :class:`FileChange` — one per file that is new,
    modified, deleted, or unchanged.
    """
    if ignore_patterns is None:
        ignore_patterns = list(DEFAULT_IGNORE_PATTERNS)

    # ── Try git-diff fast path ────────────────────────────────────
    last_commit = _load_last_indexed_commit(repo_id)
    if last_commit and _is_git_repo(repo_root):
        result = detect_file_changes_git(
            repo_root, last_commit, extensions, ignore_patterns,
        )
        if result is not None:
            logger.info(
                "Used git-diff for change detection (fast path, base=%s)",
                last_commit[:12],
            )
            return result

    # ── Fallback: hash-based detection ────────────────────────────
    logger.info("Using hash-based change detection (full scan)")

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


# ── Commit SHA persistence ───────────────────────────────────────────


def _load_last_indexed_commit(repo_id: uuid.UUID) -> Optional[str]:
    """Read the stored ``last_indexed_commit`` for a repository."""
    with get_session() as session:
        repo = session.get(Repository, repo_id)
        if repo is None:
            return None
        return repo.last_indexed_commit


def save_last_indexed_commit(
    repo_id: uuid.UUID,
    repo_root: Path,
) -> None:
    """
    Resolve the current HEAD commit and persist it to the repository row.

    Called after a successful index (full or incremental) so the next
    incremental run can use git-diff against this commit.  Silently
    skips if the repo is not a git repo.
    """
    if not _is_git_repo(repo_root):
        return

    sha = _get_head_commit(repo_root)
    if sha is None:
        return

    with get_session() as session:
        repo = session.get(Repository, repo_id)
        if repo is not None:
            repo.last_indexed_commit = sha

    logger.info("Saved last_indexed_commit=%s for repo %s", sha[:12], repo_id)
