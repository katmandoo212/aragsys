# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project State

This repository contains the **Scientific Agentic RAG Framework** - a pluggable, extensible Retrieval-Augmented Generation framework for multi-hop reasoning over academic and clinical literature.

### Current Status

**Phase:** Phase 7 Complete - Generation (2026-02-07)

**Phase 7 adds:**
- Answer dataclass for generation responses (content, metadata, citations)
- OllamaClient generation model configuration
- SimpleGenerationTechnique: Basic LLM generation with document context
- ContextGenerationTechnique: Citation-aware generation with source markers
- ChainOfThoughtGenerationTechnique: Step-by-step reasoning with answer extraction
- **120 tests passing** (99 Phase 1-6 + 21 Phase 7)

**Supported formats:** .txt, .pdf, .md, .markdown with graceful fallback for unknown types

**Generation models:** glm-4.7:cloud, llama3:8b, llama3:70b (via Ollama)

## Planned Architecture

The project implements an iterative RAG system using:
- **Python 3.13+** with uv for package management
- **PocketFlow** for node-based workflow orchestration
- **Ollama** for local LLM inference
- **pdfplumber** for PDF structure extraction
- **YAML** for all configuration

### Core Design Patterns
- **Registry Pattern:** Dynamic technique loading from YAML configuration
- **Strategy Pattern:** Swappable retrieval strategies
- **Builder Pattern:** Pipeline composition
- **Duck typing interface** for flexibility

### Implementation Phases

1. **Phase 1 (Complete):** Architecture foundation with mocked techniques
2. **Phase 2 (Complete):** Naive RAG with dense retrieval (NaiveRAGTechnique, Document dataclass)
3. **Phase 3 (Complete):** Document Formats - PDF/Markdown chunking with structure preservation
4. **Phase 4 (Complete):** Advanced Retrieval (HyDE, Multi-Query, Hybrid search)
5. **Phase 5 (Complete):** Precision (Reranking, Contextual Compression)
6. **Phase 6 (Complete):** GraphRAG with multi-hop reasoning
7. **Phase 7 (Complete):** Generation (Simple, Context, Chain-of-Thought)

## Key Files

### Configuration
- `config/techniques.yaml` - Technique definitions and their configs
- `config/pipelines.yaml` - Pipeline compositions
- `config/models.yaml` - Ollama and model settings

### Core Components
- `registry/technique_registry.py` - TechniqueRegistry, TechniqueMetadata
- `nodes/technique_node.py` - Generic TechniqueNode for PocketFlow
- `pipeline/builder.py` - PipelineBuilder
- `ollama/client.py` - OllamaClient
- `utils/pdf_chunker.py` - PDFChunker with structure preservation (tables, figures, page numbers)
- `utils/markdown_chunker.py` - MarkdownChunker with section-aware H1-H3 heading paths
- `utils/text_chunker.py` - Format-aware dispatcher (.txt, .pdf, .md) with fallback
- `techniques/naive_rag.py` - NaiveRAGTechnique implementation
- `stores/vector_store.py` - VectorStore with ChromaDB integration

### Tests
- `tests/test_registry.py` - Registry tests (7 tests)
- `tests/test_nodes.py` - Node tests (2 tests)
- `tests/test_builder.py` - Builder tests (2 tests)
- `tests/test_ollama.py` - Ollama client tests (4 tests)
- `tests/test_integration.py` - Round-trip integration test (1 test)
- `tests/test_naive_rag_technique.py` - NaiveRAGTechnique tests (3 tests)
- `tests/test_vector_store.py` - VectorStore tests (4 tests)
- `tests/test_pdf_chunker.py` - PDFChunker tests (11 tests)
- `tests/test_markdown_chunker.py` - MarkdownChunker tests (15 tests)
- `tests/test_text_chunker.py` - TextChunker format dispatch tests (5 tests)
- `tests/test_neo4j_store.py` - Neo4jStore tests (15 tests)
- `tests/test_entity_extractor.py` - EntityExtractor tests (4 tests)
- `tests/test_graph_entity_technique.py` - GraphEntityTechnique tests (3 tests)
- `tests/test_graph_multi_hop_technique.py` - GraphMultiHopTechnique tests (3 tests)
- `tests/test_graph_expand_technique.py` - GraphExpandTechnique tests (3 tests)
- `tests/test_answer.py` - Answer dataclass tests (3 tests)
- `tests/test_generate_simple_technique.py` - SimpleGenerationTechnique tests (4 tests)
- `tests/test_generate_context_technique.py` - ContextGenerationTechnique tests (3 tests)
- `tests/test_generate_cot_technique.py` - ChainOfThoughtGenerationTechnique tests (7 tests)
- `tests/test_generation_config.py` - Generation config tests (2 tests)

**Total: 120 tests, all passing**

## Commands

```bash
# Run tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_registry.py -v

# Install dependencies
uv sync
```

## Session Management

**IMPORTANT:** Follow session-management.md skill. Update session state at natural breakpoints.

### After Every Task Completion
Ask yourself:
1. Was a decision made? → Log to `_project_specs/session/decisions.md`
2. Did this take >10 tool calls? → Full checkpoint to `_project_specs/session/current-state.md`
3. Is a major feature complete? → Create archive entry in `_project_specs/session/archive/`
4. Otherwise → Quick update to `_project_specs/session/current-state.md`

### Checkpoint Triggers
**Quick Update** (current-state.md):
- After any todo completion
- After small changes

**Full Checkpoint** (current-state.md + decisions.md):
- After significant changes
- After ~20 tool calls
- After any decision
- When switching focus areas

**Archive** (archive/ + full checkpoint):
- End of session
- Major feature complete
- Context feels heavy (~50+ tool calls)

### Session Start Protocol
When beginning work:
1. Read `_project_specs/session/current-state.md`
2. Review recent `decisions.md` entries if needed
3. Continue from "Next Steps"

### Session End Protocol
Before ending or when context limit approaches:
1. Create archive: `_project_specs/session/archive/YYYY-MM-DD.md`
2. Update current-state.md with handoff format
3. Ensure next steps are specific and actionable

## Coding Standards

- **Type hints** on all function signatures
- **Duck typing interface** - all techniques follow protocol
- **Dependency injection** - all dependencies injected via YAML config
- **YAGNI** - only build what's needed for current phase
- **TDD** - write tests before implementation

## Phase 3 Learnings (Document Formats)

### Design Decisions
- **pdfplumber over pypdf:** Chose pdfplumber for better table extraction and layout analysis
- **Heading path metadata:** Tracks full H1-H3 hierarchy (e.g., ["Introduction", "Background", "Methods"])
- **Figure caption patterns:** Multiple regex variants for "Figure X:", "Fig. X", "FIGURE X" (case-insensitive)
- **Graceful fallback:** Unknown formats attempt plain text extraction, return empty list on failure

### Architecture Patterns
- **Dispatch pattern:** TextChunker maps extensions to handler functions via `_handlers` dict
- **Unified metadata schema:** All chunkers return `list[tuple[str, dict]]` with consistent metadata fields
- **Rich PDF metadata:** page_number, section_title, figure_captions, table_data

### Testing Insights
- PDFChunker: 11 tests cover basic extraction, table handling, figure detection, edge cases
- MarkdownChunker: 15 tests cover heading paths, section splitting, nested headings, empty sections
- Format dispatch: 5 tests for extension detection, fallback, TXT path

### Future Considerations
- PDF section detection uses basic heuristics (font size/weight not implemented per YAGNI)
- Table extraction preserves structure but no semantic understanding yet
- Heading paths are simple strings - could be enriched with IDs or anchors

## Learnings from Phase 3 (Document Formats)

**PDFChunker with pdfplumber:**
- Use `pdfplumber.open(path)` to parse PDFs page by page
- Extract text via `page.extract_text()` for each page
- Tables: `page.extract_tables()` returns structured data, preserve as metadata
- Figure captions: Regex patterns like `r"Figure\s+\d+:"` and `r"Fig\.\s*\d+"` work well
- Split oversized content by paragraphs (double newlines `\\n\\n`)
- Metadata: `source`, `chunk_index`, `page_number`, `section_title`, `figure_captions`, `table_data`

**MarkdownChunker section-aware parsing:**
- Parse line by line to detect H1-H3 headings (`# `, `## `, `### `)
- Maintain heading stack for path tracking (push/pop as nesting changes)
- Heading path format: `["H1", "H2", "H3"]` - provides context for retrieval
- Split oversized sections by same paragraph logic as PDF
- Metadata: `source`, `chunk_index`, `heading_path`, `heading_level`

**TextChunker dispatcher pattern:**
- Handler dict maps extensions to methods: `{'.pdf': self._chunk_pdf, ...}`
- Fallback handler attempts plain text for unknown formats
- Consistent return type: `list[tuple[str, dict]]` for all formats

**Testing strategies:**
- Use temporary files with `tempfile.NamedTemporaryFile()` for test fixtures
- Mock PDF/Markdown content with realistic structure (headings, tables, figures)
- Test both happy paths and error cases (file not found, invalid PDF)
- Verify metadata completeness per chunk