"""
Git clone layer — fetches remote repositories for indexing.

Provides a thin wrapper around ``git clone`` so the rest of the
indexing pipeline can work with a local directory regardless of
whether the source is a local path or a remote Git URL.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from lumen.config import CLONE_DIR

logger = logging.getLogger(__name__)

#: Pattern that loosely matches common Git URL formats.
_GIT_URL_PATTERN = re.compile(
    r"^(https?://|git@|ssh://|git://)",
    re.IGNORECASE,
)


def is_git_url(value: str) -> bool:
    """Return ``True`` if *value* looks like a Git clone URL."""
    return bool(_GIT_URL_PATTERN.match(value.strip()))


def normalize_git_url(url: str) -> str:
    """
    Normalize a Git URL by removing embedded credentials (tokens, passwords).

    This ensures that URLs with different credentials are treated as the same
    repository, preventing duplicate entries and lookup issues.

    Examples::

        https://x-access-token:ghp_xxx@github.com/org/repo.git  →  https://github.com/org/repo.git
        https://user:pass@github.com/org/repo.git               →  https://github.com/org/repo.git
        https://github.com/org/repo.git                         →  https://github.com/org/repo.git
        git@github.com:org/repo.git                             →  git@github.com:org/repo.git
    """
    from urllib.parse import urlparse, urlunparse

    # Only normalize HTTP(S) URLs; SSH URLs (git@...) don't have embedded creds
    if not url.startswith(("http://", "https://")):
        return url.strip()

    parsed = urlparse(url.strip())

    # Reconstruct URL without username/password
    normalized = urlunparse((
        parsed.scheme,
        parsed.hostname + (f":{parsed.port}" if parsed.port else ""),
        parsed.path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))

    return normalized


def repo_name_from_url(url: str) -> str:
    """
    Extract a human-readable repo name from a Git URL.

    Examples::

        https://github.com/psap2/Lumen-indexer.git  →  Lumen-indexer
        git@github.com:psap2/Lumen-indexer.git       →  Lumen-indexer
    """
    # Strip trailing slashes and .git suffix
    name = url.rstrip("/")
    if name.endswith(".git"):
        name = name[:-4]
    # Take the last path segment
    name = name.split("/")[-1].split(":")[-1]
    return name or "repo"


def clone_repo(
    git_url: str,
    branch: Optional[str] = None,
    depth: int = 1,
) -> Path:
    """
    Shallow-clone a remote repository into a temporary directory.

    Parameters
    ----------
    git_url:
        The HTTPS or SSH clone URL.
    branch:
        Branch to clone.  ``None`` clones the default branch.
    depth:
        Git clone depth.  Defaults to ``1`` (latest commit only)
        for speed — we only need the current source, not history.

    Returns
    -------
    Path
        Absolute path to the cloned repository on disk.

    Raises
    ------
    RuntimeError
        If ``git clone`` fails.
    """
    # Determine base directory for clones
    base = Path(CLONE_DIR) if CLONE_DIR else Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)

    clone_dir = Path(tempfile.mkdtemp(
        prefix="lumen-clone-",
        dir=str(base),
    ))

    cmd = ["git", "clone", "--depth", str(depth)]
    if branch:
        cmd += ["--branch", branch]
    cmd += [git_url, str(clone_dir)]

    logger.info("Cloning %s → %s", git_url, clone_dir)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout for large repos
        )
    except subprocess.TimeoutExpired:
        cleanup_clone(clone_dir)
        raise RuntimeError(
            f"Git clone timed out after 300s for {git_url}"
        )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        cleanup_clone(clone_dir)
        raise RuntimeError(
            f"Git clone failed (exit {result.returncode}): {stderr}"
        )

    logger.info("Clone complete: %s (%s)", repo_name_from_url(git_url), clone_dir)
    return clone_dir


def cleanup_clone(clone_path: Path) -> None:
    """
    Delete a previously cloned repository directory.

    Safe to call even if the path doesn't exist.
    """
    if clone_path.exists():
        logger.info("Cleaning up clone: %s", clone_path)
        shutil.rmtree(clone_path, ignore_errors=True)
