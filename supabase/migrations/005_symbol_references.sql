-- Migration 005: Add symbol-level references table.
--
-- Stores precise source-code reference edges (from_symbol -> to_symbol)
-- extracted from SCIP occurrences, with exact line/column locations.

CREATE TABLE IF NOT EXISTS symbol_references (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id         UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    from_symbol_id  TEXT NOT NULL DEFAULT '',
    to_symbol_id    TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    line            INTEGER NOT NULL,
    col             INTEGER NOT NULL,
    reference_kind  TEXT NOT NULL DEFAULT 'reference',
    snippet         TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_symbol_refs_repo
    ON symbol_references (repo_id);

CREATE INDEX IF NOT EXISTS idx_symbol_refs_from
    ON symbol_references (repo_id, from_symbol_id);

CREATE INDEX IF NOT EXISTS idx_symbol_refs_to
    ON symbol_references (repo_id, to_symbol_id);

CREATE INDEX IF NOT EXISTS idx_symbol_refs_file_loc
    ON symbol_references (repo_id, file_path, line);
