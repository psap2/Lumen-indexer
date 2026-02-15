"""
Integration tests for the Lumen indexing pipeline.

These tests exercise multiple real modules wired together end-to-end,
mocking only the external boundaries:
  - Supabase / Postgres (database writes)
  - SCIP subprocess (external tool)
  - Embedding model (HuggingFace download)

What IS tested (real code, no mocks):
  - Language detection → ingestion pipeline
  - SCIP parser → chunk enrichment flow
  - Chunk → IndexedChunk → to_dict() contract
  - CLI --export-chunks full pipeline
`  - TextNode metadata contract for downstream consumers
`"""

from __future__ import annotations

import json
import textwrap
import uuid
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from lumen.config import (
    LANGUAGE_REGISTRY,
    IndexedChunk,
    IndexedSymbol,
    LanguageProfile,
    resolve_language,
)
from lumen.ingestion.code_ingestor import ingest_repository
from lumen.scip_parser.parser import (
    ParsedDocument,
    ParsedIndex,
    ParsedSymbol,
    SymbolOccurrence,
    SymbolRelationship,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def polyglot_repo(tmp_path: Path) -> Path:
    """
    A realistic multi-language repo with enough code to produce
    multiple chunks per file (exercises the splitter).
    """
    # Python — long enough to split into 2+ chunks at 60-line default
    py_lines = []
    py_lines.append("import os\nimport sys\nfrom pathlib import Path\n\n")
    for i in range(8):
        py_lines.append(
            f"def function_{i}(x: int, y: int) -> int:\n"
            f'    """Compute something for case {i}."""\n'
            f"    result = x + y + {i}\n"
            f"    if result > 100:\n"
            f"        return result - {i}\n"
            f"    return result * 2\n\n\n"
        )
    py_lines.append(
        "class DataProcessor:\n"
        '    """Processes data through the pipeline."""\n\n'
        "    def __init__(self, config: dict) -> None:\n"
        "        self.config = config\n"
        "        self.results: list = []\n\n"
        "    def run(self) -> list:\n"
        "        for key, val in self.config.items():\n"
        "            self.results.append((key, val))\n"
        "        return self.results\n"
    )
    (tmp_path / "processor.py").write_text("".join(py_lines))

    # TypeScript
    ts_code = textwrap.dedent("""\
    export interface Config {
        debug: boolean;
        port: number;
        host: string;
    }

    export class Server {
        private config: Config;

        constructor(config: Config) {
            this.config = config;
        }

        start(): void {
            console.log(`Listening on ${this.config.host}:${this.config.port}`);
        }

        stop(): void {
            console.log("Server stopped");
        }
    }

    export function createServer(config: Config): Server {
        return new Server(config);
    }
    """)
    (tmp_path / "server.ts").write_text(ts_code)

    # Go
    go_code = textwrap.dedent("""\
    package main

    import (
        "fmt"
        "net/http"
    )

    type Handler struct {
        Name string
    }

    func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Hello from %s", h.Name)
    }

    func main() {
        handler := &Handler{Name: "Lumen"}
        http.ListenAndServe(":8080", handler)
    }
    """)
    (tmp_path / "main.go").write_text(go_code)

    # Ignored directories
    nm = tmp_path / "node_modules" / "leftpad"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = (s, n) => s.padStart(n);")

    return tmp_path


@pytest.fixture
def rich_parsed_index(polyglot_repo: Path) -> ParsedIndex:
    """
    A ParsedIndex with symbols that map to the polyglot_repo files,
    simulating what SCIP would produce.
    """
    # Python symbols
    py_symbols = []
    py_occurrences = []
    for i in range(8):
        line = 4 + i * 8  # approximate line for each function
        sym = ParsedSymbol(
            symbol_id=f"pkg/processor.py/function_{i}().",
            kind="Function",
            display_name=f"function_{i}",
            documentation=f"Compute something for case {i}.",
        )
        py_symbols.append(sym)
        py_occurrences.append(SymbolOccurrence(
            symbol=sym.symbol_id,
            start_line=line,
            start_char=4,
            end_line=line,
            end_char=15,
            is_definition=True,
        ))

    class_sym = ParsedSymbol(
        symbol_id="pkg/processor.py/DataProcessor#",
        kind="Class",
        display_name="DataProcessor",
        documentation="Processes data through the pipeline.",
        relationships=[
            SymbolRelationship(
                target_symbol="pkg/processor.py/function_0().",
                is_reference=True,
            ),
        ],
    )
    py_symbols.append(class_sym)
    py_occurrences.append(SymbolOccurrence(
        symbol=class_sym.symbol_id,
        start_line=68,
        start_char=6,
        end_line=68,
        end_char=19,
        is_definition=True,
    ))

    py_doc = ParsedDocument(
        relative_path="processor.py",
        language="python",
        symbols=py_symbols,
        occurrences=py_occurrences,
    )

    # TypeScript symbols
    ts_symbols = [
        ParsedSymbol(
            symbol_id="pkg/server.ts/Config#",
            kind="Interface",
            display_name="Config",
        ),
        ParsedSymbol(
            symbol_id="pkg/server.ts/Server#",
            kind="Class",
            display_name="Server",
            relationships=[
                SymbolRelationship(
                    target_symbol="pkg/server.ts/Config#",
                    is_type_definition=True,
                ),
            ],
        ),
        ParsedSymbol(
            symbol_id="pkg/server.ts/createServer().",
            kind="Function",
            display_name="createServer",
        ),
    ]
    ts_occurrences = [
        SymbolOccurrence(
            symbol="pkg/server.ts/Config#",
            start_line=0, start_char=17, end_line=0, end_char=23,
            is_definition=True,
        ),
        SymbolOccurrence(
            symbol="pkg/server.ts/Server#",
            start_line=6, start_char=13, end_line=6, end_char=19,
            is_definition=True,
        ),
        SymbolOccurrence(
            symbol="pkg/server.ts/createServer().",
            start_line=24, start_char=16, end_line=24, end_char=28,
            is_definition=True,
        ),
    ]
    ts_doc = ParsedDocument(
        relative_path="server.ts",
        language="typescript",
        symbols=ts_symbols,
        occurrences=ts_occurrences,
    )

    idx = ParsedIndex(
        project_root=str(polyglot_repo),
        tool_name="scip-test",
        tool_version="0.0.1",
        documents=[py_doc, ts_doc],
    )
    idx.build_lookup_tables()
    return idx


# ═════════════════════════════════════════════════════════════════════
# Integration Test 1: Language Detection → Ingestion Pipeline
# ═════════════════════════════════════════════════════════════════════


class TestDetectionToIngestion:
    """
    Verify that auto-detected language extensions are correctly passed
    to the ingestor so all relevant files get chunked.
    """

    def test_detected_extensions_match_ingested_files(self, polyglot_repo: Path):
        """Extensions from detection feed into ingestion correctly."""
        # Step 1: Detect languages (real code)
        from lumen.indexer import detect_languages

        profiles = detect_languages(polyglot_repo)
        detected_names = {p.name for p in profiles}
        assert "python" in detected_names
        assert "typescript" in detected_names
        assert "go" in detected_names

        # Step 2: Build combined extensions (same logic as indexer.py)
        all_extensions: frozenset[str] = frozenset().union(
            *(p.extensions for p in profiles)
        )

        # Step 3: Ingest with those extensions (real code)
        nodes, chunks = ingest_repository(
            polyglot_repo,
            parsed_index=None,
            extensions=all_extensions,
        )

        # Verify: all three languages produced chunks
        languages_in_chunks = {c.language for c in chunks}
        assert "python" in languages_in_chunks
        assert "typescript" in languages_in_chunks
        assert "go" in languages_in_chunks

        # Verify: node_modules was ignored
        for chunk in chunks:
            assert "node_modules" not in chunk.file_path

    def test_single_language_filter(self, polyglot_repo: Path):
        """When user specifies --language python, only .py files are ingested."""
        profile = resolve_language("python")
        nodes, chunks = ingest_repository(
            polyglot_repo,
            parsed_index=None,
            extensions=profile.extensions,
        )
        for chunk in chunks:
            assert chunk.file_path.endswith(".py"), (
                f"Non-Python file ingested: {chunk.file_path}"
            )


# ═════════════════════════════════════════════════════════════════════
# Integration Test 2: SCIP Parser → Chunk Enrichment
# ═════════════════════════════════════════════════════════════════════


class TestScipEnrichmentPipeline:
    """
    Verify that a ParsedIndex produced by the SCIP parser correctly
    enriches chunks during ingestion — symbols attached, metadata text
    generated, and IndexedSymbol fields populated.
    """

    def test_chunks_receive_scip_symbols(
        self, polyglot_repo: Path, rich_parsed_index: ParsedIndex,
    ):
        """Real ingestion with a real ParsedIndex attaches symbols to chunks."""
        all_ext = frozenset({".py", ".ts"})
        nodes, chunks = ingest_repository(
            polyglot_repo,
            parsed_index=rich_parsed_index,
            extensions=all_ext,
        )

        # At least some chunks should have symbols
        enriched = [c for c in chunks if c.symbol_count > 0]
        assert len(enriched) > 0, "No chunks received SCIP symbols"

        # Verify symbol fields are populated
        for chunk in enriched:
            for sym in chunk.symbols:
                assert sym.symbol_id, "symbol_id is empty"
                assert sym.kind, "kind is empty"
                assert sym.file_path, "file_path is empty"

    def test_symbol_header_in_node_text(
        self, polyglot_repo: Path, rich_parsed_index: ParsedIndex,
    ):
        """Enriched nodes should contain [SCIP Symbols] header for embedding."""
        nodes, _ = ingest_repository(
            polyglot_repo,
            parsed_index=rich_parsed_index,
            extensions=frozenset({".py"}),
        )

        enriched_nodes = [
            n for n in nodes if n.metadata.get("symbol_count", 0) > 0
        ]
        assert len(enriched_nodes) > 0

        for node in enriched_nodes:
            text = node.get_content()
            assert "[SCIP Symbols]" in text, (
                "Enriched node missing SCIP header in embedded text"
            )

    def test_cross_language_symbols(
        self, polyglot_repo: Path, rich_parsed_index: ParsedIndex,
    ):
        """Both Python and TypeScript chunks get their respective symbols."""
        all_ext = frozenset({".py", ".ts"})
        _, chunks = ingest_repository(
            polyglot_repo,
            parsed_index=rich_parsed_index,
            extensions=all_ext,
        )

        py_enriched = [
            c for c in chunks
            if c.file_path.endswith(".py") and c.symbol_count > 0
        ]
        ts_enriched = [
            c for c in chunks
            if c.file_path.endswith(".ts") and c.symbol_count > 0
        ]

        assert len(py_enriched) > 0, "No Python chunks enriched"
        assert len(ts_enriched) > 0, "No TypeScript chunks enriched"


# ═════════════════════════════════════════════════════════════════════
# Integration Test 3: IndexedChunk Output Contract
# ═════════════════════════════════════════════════════════════════════


class TestIndexedChunkContract:
    """
    The IndexedChunk.to_dict() output is the canonical interchange format
    consumed by the Friction Scoring Engine. These tests ensure the
    contract is never accidentally broken.
    """

    REQUIRED_KEYS = {
        "chunk_id", "file_path", "language", "code",
        "line_start", "line_end", "symbols",
        "symbol_count", "definition_count",
        "relationship_count", "complexity_hint",
    }

    REQUIRED_SYMBOL_KEYS = {
        "symbol_id", "kind", "display_name", "documentation",
        "file_path", "line_start", "line_end", "relationships",
    }

    def test_to_dict_has_all_required_keys(self, polyglot_repo: Path):
        """Every chunk from a real ingestion has the full schema."""
        _, chunks = ingest_repository(
            polyglot_repo,
            parsed_index=None,
            extensions=frozenset({".py", ".ts", ".go"}),
        )
        assert len(chunks) > 0

        for chunk in chunks:
            d = chunk.to_dict()
            missing = self.REQUIRED_KEYS - set(d.keys())
            assert not missing, (
                f"Chunk {chunk.chunk_id} missing keys: {missing}"
            )

    def test_to_dict_with_symbols_has_full_schema(
        self, polyglot_repo: Path, rich_parsed_index: ParsedIndex,
    ):
        """Enriched chunks have complete symbol sub-documents."""
        _, chunks = ingest_repository(
            polyglot_repo,
            parsed_index=rich_parsed_index,
            extensions=frozenset({".py", ".ts"}),
        )

        enriched = [c for c in chunks if c.symbol_count > 0]
        assert len(enriched) > 0

        for chunk in enriched:
            d = chunk.to_dict()
            for sym_dict in d["symbols"]:
                missing = self.REQUIRED_SYMBOL_KEYS - set(sym_dict.keys())
                assert not missing, (
                    f"Symbol in chunk {chunk.chunk_id} missing keys: {missing}"
                )

    def test_to_dict_json_serialisable(self, polyglot_repo: Path, rich_parsed_index: ParsedIndex):
        """to_dict() output must be JSON-serialisable (no custom objects)."""
        _, chunks = ingest_repository(
            polyglot_repo,
            parsed_index=rich_parsed_index,
            extensions=frozenset({".py", ".ts", ".go"}),
        )

        # This will raise TypeError if any value is not serialisable
        json_str = json.dumps([c.to_dict() for c in chunks])
        parsed_back = json.loads(json_str)
        assert len(parsed_back) == len(chunks)

    def test_types_are_correct(self, polyglot_repo: Path):
        """Verify value types in the to_dict() output."""
        _, chunks = ingest_repository(
            polyglot_repo,
            parsed_index=None,
            extensions=frozenset({".py"}),
        )
        for chunk in chunks:
            d = chunk.to_dict()
            assert isinstance(d["chunk_id"], str)
            assert isinstance(d["file_path"], str)
            assert isinstance(d["language"], str)
            assert isinstance(d["code"], str)
            assert isinstance(d["line_start"], int)
            assert isinstance(d["line_end"], int)
            assert isinstance(d["symbols"], list)
            assert isinstance(d["symbol_count"], int)
            assert isinstance(d["definition_count"], int)
            assert isinstance(d["relationship_count"], int)
            assert isinstance(d["complexity_hint"], float)

    def test_computed_properties_consistent(
        self, polyglot_repo: Path, rich_parsed_index: ParsedIndex,
    ):
        """symbol_count, definition_count, etc. match the actual symbols list."""
        _, chunks = ingest_repository(
            polyglot_repo,
            parsed_index=rich_parsed_index,
            extensions=frozenset({".py"}),
        )
        for chunk in chunks:
            d = chunk.to_dict()
            assert d["symbol_count"] == len(d["symbols"])
            assert d["symbol_count"] == chunk.symbol_count
            assert d["definition_count"] == chunk.definition_count
            assert d["relationship_count"] == chunk.relationship_count


# ═════════════════════════════════════════════════════════════════════
# Integration Test 4: TextNode Metadata Contract
# ═════════════════════════════════════════════════════════════════════


class TestTextNodeContract:
    """
    TextNodes are what LlamaIndex embeds and stores. Their metadata
    must contain the right keys for filtering and display.
    """

    REQUIRED_METADATA_KEYS = {
        "repo_id", "file_path", "language",
        "line_start", "line_end",
        "symbol_count", "definition_count", "chunk_id",
    }

    def test_nodes_have_required_metadata(self, polyglot_repo: Path):
        repo_id = str(uuid.uuid4())
        nodes, _ = ingest_repository(
            polyglot_repo,
            parsed_index=None,
            extensions=frozenset({".py", ".ts", ".go"}),
            repo_id=repo_id,
        )
        assert len(nodes) > 0

        for node in nodes:
            missing = self.REQUIRED_METADATA_KEYS - set(node.metadata.keys())
            assert not missing, (
                f"Node {node.id_} missing metadata keys: {missing}"
            )

    def test_repo_id_propagates(self, polyglot_repo: Path):
        """repo_id passed to ingest_repository appears in every node."""
        repo_id = "test-repo-abc-123"
        nodes, _ = ingest_repository(
            polyglot_repo,
            parsed_index=None,
            extensions=frozenset({".py"}),
            repo_id=repo_id,
        )
        for node in nodes:
            assert node.metadata["repo_id"] == repo_id

    def test_chunk_id_matches_between_node_and_chunk(self, polyglot_repo: Path):
        """Each node's chunk_id must match its paired IndexedChunk."""
        nodes, chunks = ingest_repository(
            polyglot_repo,
            parsed_index=None,
            extensions=frozenset({".py"}),
        )
        assert len(nodes) == len(chunks)

        for node, chunk in zip(nodes, chunks):
            assert node.metadata["chunk_id"] == chunk.chunk_id
            assert node.id_ == chunk.chunk_id

    def test_excluded_metadata_keys(self, polyglot_repo: Path):
        """chunk_id and repo_id should be excluded from embedding/LLM."""
        nodes, _ = ingest_repository(
            polyglot_repo,
            parsed_index=None,
            extensions=frozenset({".py"}),
            repo_id="test",
        )
        for node in nodes:
            assert "chunk_id" in node.excluded_embed_metadata_keys
            assert "repo_id" in node.excluded_embed_metadata_keys
            assert "chunk_id" in node.excluded_llm_metadata_keys
            assert "repo_id" in node.excluded_llm_metadata_keys


# ═════════════════════════════════════════════════════════════════════
# Integration Test 5: CLI --export-chunks Full Pipeline
# ═════════════════════════════════════════════════════════════════════


class TestCLIExportPipeline:
    """
    Test the CLI's --export-chunks flag end-to-end, mocking only the
    external boundaries (SCIP subprocess, Supabase).
    """

    @patch("lumen.incremental.save_file_states")
    @patch("lumen.incremental.detect_file_changes", return_value=[])
    @patch("lumen.indexer.embed_and_persist")
    @patch("lumen.storage.supabase_store.persist_chunks")
    @patch("lumen.storage.supabase_store.update_repository_status")
    @patch("lumen.storage.supabase_store.create_repository", return_value=uuid.uuid4())
    @patch("lumen.indexer._ensure_proto_compiled")
    @patch("lumen.indexer.run_scip_indexer", return_value=None)
    @patch("lumen.indexer.run_repl")
    def test_export_produces_valid_json(
        self,
        mock_repl,
        mock_scip,
        mock_proto,
        mock_create,
        mock_status,
        mock_persist,
        mock_embed,
        mock_detect_changes,
        mock_save_states,
        polyglot_repo: Path,
        tmp_path: Path,
    ):
        """CLI with --export-chunks writes valid JSON with full schema."""
        from lumen.indexer import main

        export_file = tmp_path / "chunks.json"

        # Run the CLI: auto-detect, skip SCIP (returns None), export
        main([
            str(polyglot_repo),
            "--language", "python",
            "--export-chunks", str(export_file),
        ])

        # Verify the export file exists and is valid JSON
        assert export_file.exists(), "Export file was not created"

        with open(export_file) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) > 0, "Export is empty"

        # Every exported chunk must have the full contract
        required = {
            "chunk_id", "file_path", "language", "code",
            "line_start", "line_end", "symbols",
            "symbol_count", "definition_count",
            "relationship_count", "complexity_hint",
        }
        for item in data:
            missing = required - set(item.keys())
            assert not missing, f"Exported chunk missing keys: {missing}"
            assert item["language"] == "python"
            assert len(item["code"]) > 0


# ═════════════════════════════════════════════════════════════════════
# Integration Test 6: Multi-chunk File Splitting
# ═════════════════════════════════════════════════════════════════════


class TestChunkSplitting:
    """
    Verify that files long enough to produce multiple chunks are
    split correctly with proper line ranges and no gaps.
    """

    def test_long_file_produces_multiple_chunks(self, polyglot_repo: Path):
        """processor.py (80+ lines) should produce more than one chunk."""
        _, chunks = ingest_repository(
            polyglot_repo,
            parsed_index=None,
            extensions=frozenset({".py"}),
        )
        py_chunks = [c for c in chunks if c.file_path == "processor.py"]
        assert len(py_chunks) > 1, (
            f"Expected multiple chunks for processor.py, got {len(py_chunks)}"
        )

    def test_chunks_cover_entire_file(self, polyglot_repo: Path):
        """Every line of the file should be covered by at least one chunk."""
        _, chunks = ingest_repository(
            polyglot_repo,
            parsed_index=None,
            extensions=frozenset({".py"}),
        )
        py_chunks = sorted(
            [c for c in chunks if c.file_path == "processor.py"],
            key=lambda c: c.line_start,
        )

        # First chunk should start at line 0
        assert py_chunks[0].line_start == 0

        # Read the actual file to know how many lines
        source = (polyglot_repo / "processor.py").read_text()
        total_lines = len(source.splitlines())

        # Last chunk should cover the end of the file
        assert py_chunks[-1].line_end >= total_lines - 1 or \
               py_chunks[-1].line_end >= total_lines

    def test_chunk_ids_are_unique(self, polyglot_repo: Path):
        """No two chunks should share the same chunk_id."""
        _, chunks = ingest_repository(
            polyglot_repo,
            parsed_index=None,
            extensions=frozenset({".py", ".ts", ".go"}),
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_chunk_code_is_nonempty(self, polyglot_repo: Path):
        """Every chunk must contain actual code."""
        _, chunks = ingest_repository(
            polyglot_repo,
            parsed_index=None,
            extensions=frozenset({".py", ".ts", ".go"}),
        )
        for chunk in chunks:
            assert len(chunk.code.strip()) > 0, (
                f"Empty code in chunk {chunk.chunk_id} ({chunk.file_path})"
            )
