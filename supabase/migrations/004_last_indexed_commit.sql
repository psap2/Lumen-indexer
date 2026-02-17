-- Migration 004: Track the git commit SHA at last successful index.
--
-- Used by the git-diff fast path for incremental change detection.
-- When present, the indexer runs `git diff <last_indexed_commit>` instead
-- of hashing every file, making change detection O(1) instead of O(N).

ALTER TABLE repositories
    ADD COLUMN IF NOT EXISTS last_indexed_commit TEXT DEFAULT NULL;

COMMENT ON COLUMN repositories.last_indexed_commit IS
    'Git commit SHA at last successful index — used for git-diff change detection';
