# Lumen — Repository Indexing Service

A component of the **Lumen Product Lifecycle Engine** that indexes codebases
using [Sourcegraph SCIP](https://github.com/sourcegraph/scip) and
[LlamaIndex](https://docs.llamaindex.ai/) to provide semantic code
understanding.

---

## Quick start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | >= 3.10 | Indexing script |
| Node.js | >= 16 | `scip-typescript` and `scip-python` (via npx) |
| npm / npx | (bundled) | Running SCIP indexers |

> **Other languages** (Go, Rust, Java, Ruby) require their own SCIP indexer
> binaries — see the [Supported languages](#supported-languages) table below.

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
# Auto-detect language and index (works for any supported language)
python -m lumen.indexer /path/to/your-project

# Explicit language
python -m lumen.indexer /path/to/your-project --language python
python -m lumen.indexer /path/to/your-project --language typescript

# Multi-language repo (comma-separated)
python -m lumen.indexer /path/to/your-project --language typescript,python

# Skip SCIP generation (if index.scip already exists)
python -m lumen.indexer /path/to/your-project --skip-scip

# Ask a single question
python -m lumen.indexer /path/to/your-project \
  --question "What does the authenticateUser function do?"

# Export chunks as JSON for the Friction Scoring Engine
python -m lumen.indexer /path/to/your-project \
  --export-chunks ./output/chunks.json
```

### Query an existing index

```bash
python -m lumen.indexer /path/to/your-project --query-only
```

---

## How the whole thing works, step by step

When you run:

```bash
python -m lumen.indexer ~/Desktop/typescript-clean-architecture
```

Here is exactly what happens, in order:

---

### Step 0 — Proto compilation guard (`indexer.py` → `proto/compile.py`)

Before anything else, the script checks if the file `lumen/proto/scip_pb2.py`
exists. This file is auto-generated Python code that knows how to read SCIP's
binary protobuf format. If it's missing (first run), the script calls
`lumen/proto/compile.py`, which uses `grpcio-tools` to compile `scip.proto`
into `scip_pb2.py`. You only need this once — after that the file sticks
around and this step is skipped.

**Think of it like this:** SCIP stores its data in a binary format called
protobuf. Python can't read that natively. `scip_pb2.py` is the "translator"
that teaches Python how to decode that binary data. We generate this translator
from the `scip.proto` blueprint.

---

### Step 1 — Run the SCIP indexer (`indexer.py` → `run_scip_indexer()`)

**File:** `lumen/indexer.py`, line 68

The script shells out (runs a terminal command) inside your target repo:

```
npx --yes @sourcegraph/scip-typescript index --infer-tsconfig
```

This is a Sourcegraph tool that **statically analyses** your TypeScript code —
similar to how a compiler reads your code, but instead of producing a running
program, it produces a **map of every symbol** in the codebase. This map is
saved as `index.scip` inside the target repo.

What's in that map? For every file in your project, it records:
- Every **symbol** — functions, classes, interfaces, variables, parameters, enums
- Where each symbol is **defined** (file + line number)
- Where each symbol is **referenced** (used) from other files
- **Relationships** between symbols — e.g. "class Dog *implements* interface Animal"
- **Documentation** — JSDoc comments, type signatures like `function add(a: number, b: number): number`

**If this step fails** (say Node.js isn't installed, or it's not a TypeScript
project), the pipeline doesn't crash. It logs a warning and continues to
Step 3 without SCIP data — you'll still get code chunks, just without the
symbol intelligence.

---

### Step 2 — Parse the SCIP index (`indexer.py` → `scip_parser/parser.py`)

**File:** `lumen/scip_parser/parser.py`, line 245

Now we have `index.scip` — a binary protobuf file. This step reads that file
and converts it into Python dataclasses that the rest of our code can work
with.

Here's what happens inside `parse_scip_index()`:

1. It reads the raw bytes from `index.scip`
2. Uses the generated `scip_pb2.py` to decode the protobuf into structured data
3. Loops through every **document** (source file) in the index
4. For each document, it extracts:
   - **Occurrences** — each place a symbol appears, with its exact line/column
     position and whether it's a definition or just a reference
   - **Symbols** — rich metadata: the symbol's kind, display name, docs, and
     relationships to other symbols
5. Builds **lookup tables** so later steps can quickly ask:
   - "Give me all symbols defined between lines 10–30 of `User.ts`"
   - "What are the relationships for symbol X?"

The output is a `ParsedIndex` object that looks roughly like:

```
ParsedIndex:
  project_root: "/Users/you/Desktop/your-project"
  documents: [
    ParsedDocument:
      relative_path: "src/app/domain/User.ts"
      symbols: [
        ParsedSymbol(id="...User#", kind="Class", display_name="User", ...)
        ParsedSymbol(id="...User#name.", kind="Property", ...)
        ...
      ]
      occurrences: [
        SymbolOccurrence(symbol="...User#", line=15, is_definition=True)
        SymbolOccurrence(symbol="...User#", line=42, is_definition=False)  ← reference
        ...
      ]
  ]
  symbol_table: { "...User#": ParsedSymbol(...), ... }   ← fast lookup by ID
```

---

### Step 3 — Ingest: split code + enrich with SCIP (`ingestion/code_ingestor.py`)

**File:** `lumen/ingestion/code_ingestor.py`, line 200

This is where the magic happens — where SCIP intelligence meets LlamaIndex.

**3a. Collect files**

First, it walks the target repo and collects every file matching the target
extensions (`.ts`, `.tsx`, `.js`, `.jsx`, `.mjs`, `.cjs`). It skips
`node_modules/`, `.git/`, `dist/`, `build/`, and other junk directories
(configured in `config.py`).

**3b. Split each file into chunks**

You can't embed an entire file into a vector database — it would be too big and
the meaning would get diluted. So we split each file into smaller **chunks** of
roughly 60 lines, with 15 lines of overlap between consecutive chunks (so
context isn't lost at chunk boundaries).

The splitter tries to use **tree-sitter** (via LlamaIndex's `CodeSplitter`),
which understands the language grammar and tries to split at natural boundaries
(between functions, between classes, etc.) rather than cutting mid-statement.
If tree-sitter isn't available, it falls back to a simple line-based split.

**3c. Enrich each chunk with SCIP symbols**

This is the key step that makes Lumen different from just embedding raw code.

For each chunk, we know its file path and line range (e.g. `User.ts` lines
0–60). We ask the `ParsedIndex`: "What symbols are **defined** between lines
0 and 60 of `User.ts`?" It returns things like:

- `Class: User`
- `Interface: ICreateUserRequestDTO`
- `Property: User#email`
- `Constructor: User(id, name, email, password, createdAt, updatedAt)`
- `Relationship: MongooseUserRepository implements UserRepository`

We render all of this into a **text header** that gets **prepended to the code
chunk** before embedding:

```
[SCIP Symbols]
  • Class: User  — ```ts class User ```
  • Interface: ICreateUserRequestDTO  — ```ts interface ICreateUserRequestDTO ```
  • Property: User#email  — ```ts (property) email: string ```
  • Method: MongooseUserRepository#add()  [implements,refs → UserRepository]

// ... actual code follows ...
export class User {
  public id: string;
  ...
```

**Why do this?** Because the embedding model will now encode not just the raw
code, but also the fact that this chunk contains a `Class` called `User` with
specific properties, and that something else *implements* a `UserRepository`.
When you later search for "How does the User entity interact with the
Infrastructure layer?", the embedding captures those relationships and returns
this chunk as a match — something that wouldn't happen with raw code alone.

**3d. Produce two outputs**

For each chunk, we create:
1. A **TextNode** (LlamaIndex object) — the enriched text + metadata, ready for
   embedding
2. An **IndexedChunk** (our own object) — the standardised output format for the
   Friction Scoring Engine, with fields like `symbol_count`,
   `definition_count`, `relationship_count`, and `complexity_hint`

---

### Step 4 — Embed and persist to ChromaDB (`storage/vector_store.py`)

**File:** `lumen/storage/vector_store.py`, line 62

Now we have a list of TextNode objects (e.g. 20 enriched code chunks). This
step turns them into **vectors** (arrays of 384 numbers) and stores them.

Here's what happens:

1. **Load the embedding model.** We use `BAAI/bge-small-en-v1.5`, a HuggingFace
   model that runs **locally on your machine** — no API key, no internet
   needed. It converts text into a 384-dimensional vector that captures the
   *meaning* of that text.

2. **Create a ChromaDB collection.** ChromaDB is a local vector database that
   stores embeddings on disk (inside `<your-repo>/.lumen/chroma_db/`). If a
   previous collection exists, we delete it and rebuild from scratch.

3. **Build the VectorStoreIndex.** LlamaIndex takes each TextNode, passes its
   text through the embedding model, gets back a 384-number vector, and stores
   that vector + the original text + metadata in ChromaDB.

After this step, you have a searchable database where you can find code by
*meaning*, not just by keyword matching.

**Where does the data live?** Inside the target repo at
`.lumen/chroma_db/`. So if you indexed
`~/Desktop/typescript-clean-architecture`, the database is at
`~/Desktop/typescript-clean-architecture/.lumen/chroma_db/`.

---

### Step 5 — Query (`query/engine.py`)

**File:** `lumen/query/engine.py`, line 77

Now the index is built. The script either:
- Runs a single question (if you used `--question "..."`)
- Drops you into the **interactive REPL** (the `❯` prompt)

When you type a question like:

```
❯ What does the registerUser function do?
```

Here's what happens behind the scenes:

1. **Embed your question.** The same `bge-small-en-v1.5` model converts your
   question into a 384-dimensional vector.

2. **Cosine similarity search.** ChromaDB compares your question vector against
   every stored code chunk vector. It measures how "close" each chunk's meaning
   is to your question using cosine similarity (1.0 = identical meaning,
   0.0 = completely unrelated).

3. **Return top-K results.** The 5 most similar chunks are returned, ranked by
   score.

4. **Display.** Each result shows:
   - The **score** (e.g. `[0.7262]`)
   - The **file path** and **line range** (e.g. `src/app/domain/User.ts L0–L60`)
   - The **number of SCIP symbols** in that chunk
   - The **chunk text** — including the `[SCIP Symbols]` header and the code

**Why does this work?** Because in Step 3, we prepended symbol metadata to
each chunk. The embedding model encoded meanings like "this class implements
that interface" and "this method takes a UserDTO parameter". So when you
ask about function purposes or architectural relationships, the vector
search finds chunks whose *meaning* matches your question — not just chunks
that happen to contain the same keywords.

---

## Project structure

```
Lumen-indexer/                        ← You run commands from here
├── README.md
├── requirements.txt
├── .gitignore
└── lumen/                            ← Python package (don't cd into here)
    ├── __init__.py                   # Package root (v0.1.0)
    ├── __main__.py                   # Allows `python -m lumen` shortcut
    ├── config.py                     # All tunables + IndexedChunk/IndexedSymbol schemas
    ├── indexer.py                    # CLI entry-point — orchestrates steps 0–5
    ├── proto/
    │   ├── scip.proto                # SCIP protobuf definition (vendored from Sourcegraph)
    │   ├── compile.py                # Compiles scip.proto → scip_pb2.py
    │   └── scip_pb2.py               # (generated, git-ignored) Python protobuf bindings
    ├── scip_parser/
    │   └── parser.py                 # Reads index.scip binary → ParsedIndex dataclasses
    ├── ingestion/
    │   └── code_ingestor.py          # Walks repo, splits code, enriches with SCIP symbols
    ├── storage/
    │   └── vector_store.py           # ChromaDB: build and load vector indexes
    └── query/
        └── engine.py                 # Semantic search + interactive REPL
```

### What each file does

| File | One-line purpose |
|------|-----------------|
| `config.py` | Every tunable number and the `IndexedChunk`/`IndexedSymbol` data schemas live here |
| `indexer.py` | The conductor — calls each step in order and handles CLI flags |
| `proto/scip.proto` | The Sourcegraph blueprint that defines what an `index.scip` file contains |
| `proto/compile.py` | One-time script that generates `scip_pb2.py` from `scip.proto` |
| `scip_parser/parser.py` | Reads the binary `index.scip`, converts everything into Python dataclasses, builds fast lookup tables |
| `ingestion/code_ingestor.py` | Walks the repo, splits files into chunks, asks the parser "what symbols are in lines X–Y?", prepends that info to each chunk |
| `storage/vector_store.py` | Takes the enriched chunks, runs them through the embedding model, stores the vectors in ChromaDB on disk |
| `query/engine.py` | Takes your question, embeds it with the same model, does a cosine similarity search against ChromaDB, returns the top matches |

---

## The data flow at a glance

```
 You run:  python -m lumen.indexer ~/Desktop/my-ts-project

 Step 1    npx scip-typescript runs inside your repo
           └──► produces index.scip (binary protobuf)

 Step 2    parser.py reads index.scip
           └──► produces ParsedIndex (Python object with symbol tables)

 Step 3    code_ingestor.py walks your repo
           ├── splits each file into ~60-line chunks
           ├── asks ParsedIndex: "what symbols are in this chunk?"
           ├── prepends [SCIP Symbols] header to each chunk
           └──► produces TextNodes (for embedding) + IndexedChunks (for export)

 Step 4    vector_store.py embeds the TextNodes
           ├── runs each chunk through bge-small-en-v1.5 (local model)
           ├── gets a 384-number vector per chunk
           └──► stores vectors + text in ChromaDB (on disk at .lumen/chroma_db/)

 Step 5    engine.py handles your questions
           ├── embeds your question with the same model
           ├── cosine similarity search against all stored vectors
           └──► returns the 5 closest code chunks, ranked by score
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

- **Chunk size / overlap** — `CHUNK_LINES` (60), `CHUNK_OVERLAP_LINES` (15), `CHUNK_MAX_CHARS` (3000)
- **Embedding model** — `EMBEDDING_MODEL` (default: `BAAI/bge-small-en-v1.5`, 384 dimensions, runs locally)
- **ChromaDB** — `CHROMA_PERSIST_DIR` (`.lumen/chroma_db`), `CHROMA_COLLECTION` (`lumen_code_index`)
- **Ignore patterns** — `DEFAULT_IGNORE_PATTERNS` (skips `node_modules`, `.git`, `dist`, `build`, etc.)

---

## Supported languages

| Language | `--language` flag | SCIP indexer | Install |
|----------|------------------|-------------|---------|
| TypeScript / JavaScript | `typescript` (or `ts`, `js`) | `@sourcegraph/scip-typescript` | Automatic via npx |
| Python | `python` (or `py`) | `@sourcegraph/scip-python` | Automatic via npx |
| Go | `go` (or `golang`) | `scip-go` | `go install github.com/sourcegraph/scip-go/cmd/scip-go@latest` |
| Rust | `rust` (or `rs`) | `rust-analyzer scip` | [rust-analyzer install](https://rust-analyzer.github.io/manual.html) |
| Java | `java` | `scip-java` | `cs install scip-java` ([Coursier](https://get-coursier.io)) |
| Ruby | `ruby` (or `rb`) | `scip-ruby` | `gem install scip-ruby` |
| C / C++ | `cpp` (or `c`, `c++`) | *(none yet)* | Code is still chunked and embedded, just without SCIP symbol data |

**Auto-detection:** If you don't pass `--language`, Lumen scans the repo's file
extensions and picks the language(s) with the most files. For a repo with both
`.ts` and `.py` files, it will run both SCIP indexers and merge the results.

**Multi-language:** Pass comma-separated values:
`--language typescript,python`

---

## Current limitations

- **Symbol kinds show as "UnspecifiedKind"** — `scip-typescript` stores kind info in the signature docs string rather than the protobuf `kind` field. The data is there, just in a different place.
- **No incremental indexing** — every run rebuilds the full index. For large repos, this can be slow.
- **SCIP indexer availability** — Go, Rust, Java, and Ruby SCIP indexers must be installed separately (see table above). If they're not on PATH, Lumen logs a warning and continues without SCIP enrichment for that language.

---

## License

Internal — Lumen Product Lifecycle Engine.
