-- Migration 003: Add AST metadata column to code_chunks
--
-- Stores tree-sitter AST structural metadata (signatures, imports,
-- complexity, node types) as JSONB.  This data is always available
-- regardless of whether SCIP ran successfully.

ALTER TABLE code_chunks
    ADD COLUMN IF NOT EXISTS ast_metadata jsonb DEFAULT NULL;

COMMENT ON COLUMN code_chunks.ast_metadata IS
    'Tree-sitter AST structural metadata: node_types, signatures, imports, complexity, nesting_depth';
