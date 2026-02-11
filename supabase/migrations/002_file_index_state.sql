-- Lumen — Incremental Indexing Support
-- Tracks per-file content hashes so re-indexing only processes changed files.

CREATE TABLE IF NOT EXISTS file_index_state (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id      UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    file_path    TEXT NOT NULL,
    content_hash TEXT NOT NULL,       -- SHA256 hex digest of the file contents
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    indexed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (repo_id, file_path)
);

CREATE INDEX idx_file_state_repo ON file_index_state (repo_id);
