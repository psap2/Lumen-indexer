-- Lumen Repository Indexing Service — Initial Schema
-- Run this in your Supabase SQL Editor (or via supabase db push).
-- Requires the pgvector extension to be enabled first.

-- ── Enable pgvector ─────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS vector;

-- ── Repositories ────────────────────────────────────────────────────

CREATE TYPE repo_status AS ENUM ('pending', 'indexing', 'ready', 'failed');

CREATE TABLE IF NOT EXISTS repositories (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    path_or_url TEXT NOT NULL,
    languages   TEXT[] NOT NULL DEFAULT '{}',
    status      repo_status NOT NULL DEFAULT 'pending',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    symbol_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_repositories_status ON repositories (status);

-- ── Code chunks ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS code_chunks (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id             UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    chunk_id            TEXT NOT NULL,
    file_path           TEXT NOT NULL,
    language            TEXT NOT NULL,
    code                TEXT NOT NULL,
    line_start          INTEGER NOT NULL,
    line_end            INTEGER NOT NULL,
    symbol_count        INTEGER NOT NULL DEFAULT 0,
    definition_count    INTEGER NOT NULL DEFAULT 0,
    relationship_count  INTEGER NOT NULL DEFAULT 0,
    complexity_hint     REAL NOT NULL DEFAULT 0.0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (repo_id, chunk_id)
);

CREATE INDEX idx_code_chunks_repo ON code_chunks (repo_id);
CREATE INDEX idx_code_chunks_file ON code_chunks (repo_id, file_path);

-- ── Symbols ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS symbols (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id         UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    chunk_id        TEXT NOT NULL,
    symbol_id       TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'UnspecifiedKind',
    display_name    TEXT NOT NULL DEFAULT '',
    documentation   TEXT NOT NULL DEFAULT '',
    file_path       TEXT NOT NULL,
    line_start      INTEGER NOT NULL,
    line_end        INTEGER NOT NULL,
    relationships   JSONB NOT NULL DEFAULT '[]',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_symbols_repo ON symbols (repo_id);
CREATE INDEX idx_symbols_kind ON symbols (repo_id, kind);
CREATE INDEX idx_symbols_chunk ON symbols (chunk_id);

-- ── Code embeddings (pgvector) ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS code_embeddings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id     UUID NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    chunk_id    TEXT NOT NULL,
    embedding   vector(384) NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_code_embeddings_repo ON code_embeddings (repo_id);

-- HNSW index for fast cosine similarity search
CREATE INDEX idx_code_embeddings_vector ON code_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ── Auto-update updated_at ──────────────────────────────────────────

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER repositories_updated_at
    BEFORE UPDATE ON repositories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
