# Lumen — Repository Indexing Service

A component of the **Lumen Product Lifecycle Engine** that indexes codebases
using [Sourcegraph SCIP](https://github.com/sourcegraph/scip) and
[LlamaIndex](https://docs.llamaindex.ai/) to provide semantic code
understanding.

---

## What it does

1. **Runs the SCIP indexer** (`scip-typescript`) against a local TypeScript
   repository to produce an `index.scip` file.
2. **Parses SCIP** — extracts every symbol definition, reference, and
   relationship from the protobuf index.
3. **Splits & enriches** — walks the repo, splits source files into chunks,
   and attaches SCIP symbol metadata (function names, kinds, docs,
   dependency edges) to each chunk.
4. **Embeds & stores** — generates embeddings with a local HuggingFace model
   (`BAAI/bge-small-en-v1.5`) and persists them in ChromaDB.
5. **Queries** — exposes a retrieval API and interactive REPL to verify that
   the indexer "understands" a function's purpose.

All output is structured as `IndexedChunk` objects ready for the downstream
**Friction Scoring Engine**.

---

## Quick start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | >= 3.10 | Indexing script |
| Node.js | >= 16 | `scip-typescript` |
| npm / npx | (bundled) | Running the SCIP indexer |

### Setup

```bash
cd Lumen-indexer

# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Compile the SCIP protobuf schema (one-time)
python -m lumen.proto.compile
```

### Index a repository

```bash
# Full pipeline: SCIP → parse → ingest → embed → REPL
python -m lumen.indexer /path/to/your-ts-project

# Skip SCIP generation (if index.scip already exists)
python -m lumen.indexer /path/to/your-ts-project --skip-scip

# Ask a single question
python -m lumen.indexer /path/to/your-ts-project \
  --question "What does the authenticateUser function do?"

# Export chunks as JSON for the Friction Scoring Engine
python -m lumen.indexer /path/to/your-ts-project \
  --export-chunks ./output/chunks.json
```

### Query an existing index

```bash
python -m lumen.indexer /path/to/your-ts-project --query-only
```

---

## Project structure

```
Lumen-indexer/
├── README.md
├── requirements.txt
└── lumen/
    ├── __init__.py
    ├── config.py                # Centralised tunables & output schema
    ├── indexer.py               # CLI entry-point & pipeline orchestrator
    ├── proto/
    │   ├── scip.proto           # Vendored SCIP protobuf definition
    │   └── compile.py           # Proto → Python compilation helper
    ├── scip_parser/
    │   └── parser.py            # Reads index.scip → ParsedIndex
    ├── ingestion/
    │   └── code_ingestor.py     # Code splitting + SCIP enrichment
    ├── storage/
    │   └── vector_store.py      # ChromaDB persistence layer
    └── query/
        └── engine.py            # Semantic retrieval + REPL
```

---

## Architecture

```
                    ┌──────────────┐
   Local repo ────►│ scip-typescript│────► index.scip
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ SCIP Parser  │  (protobuf → ParsedIndex)
                    └──────┬───────┘
                           │
         ┌─────────────────┼──────────────────┐
         │                 │                  │
         ▼                 ▼                  ▼
   Source files      Symbol table      Relationships
         │                 │                  │
         └─────────┬───────┘──────────────────┘
                   │
                   ▼
            ┌─────────────┐
            │ Code Ingest  │  (split + enrich with SCIP)
            └──────┬──────┘
                   │
                   ▼
            ┌─────────────┐
            │  Embeddings  │  (HuggingFace bge-small)
            └──────┬──────┘
                   │
                   ▼
            ┌─────────────┐       ┌────────────────────────┐
            │  ChromaDB    │◄─────│  Friction Scoring Engine │
            └──────┬──────┘       └────────────────────────┘
                   │
                   ▼
            ┌─────────────┐
            │  Query REPL  │
            └─────────────┘
```

---

## Standardised output format

Each `IndexedChunk` (exported via `--export-chunks`) contains:

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | `str` | Deterministic hash of file + line range |
| `file_path` | `str` | Path relative to repo root |
| `language` | `str` | Detected language |
| `code` | `str` | Raw source code for the chunk |
| `line_start` | `int` | 0-based start line |
| `line_end` | `int` | 0-based end line (exclusive) |
| `symbols` | `list` | SCIP symbols with kind, docs, relationships |
| `symbol_count` | `int` | Total symbols in chunk |
| `definition_count` | `int` | Definitions (functions, classes, etc.) |
| `relationship_count` | `int` | Cross-symbol edges |
| `complexity_hint` | `float` | `(symbols + relationships) / lines` |

---

## Configuration

All tunables are in `lumen/config.py`:

- **Chunk size / overlap** — `CHUNK_LINES`, `CHUNK_OVERLAP_LINES`, `CHUNK_MAX_CHARS`
- **Embedding model** — `EMBEDDING_MODEL` (default: `BAAI/bge-small-en-v1.5`)
- **ChromaDB** — `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION`
- **Ignore patterns** — `DEFAULT_IGNORE_PATTERNS`

---

## License

Internal — Lumen Product Lifecycle Engine.
# Lumen-indexer
