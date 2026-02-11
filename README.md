# Scientific Agentic RAG Framework

A pluggable, extensible Retrieval-Augmented Generation framework for multi-hop reasoning over academic and clinical literature.

[![Tests](https://img.shields.io/badge/tests-144%20passing-brightgreen)](tests/)

## Overview

aragsys is a complete RAG framework designed for scientific literature analysis. It provides:

- **Web Interface** - Interactive query workspace with real-time progress
- **Document Ingestion** - Process PDF, Markdown, and TXT files with structure preservation
- **Advanced Retrieval** - Multiple strategies (Naive, HyDE, Multi-Query, Hybrid search)
- **Precision Enhancement** - Reranking and contextual compression
- **GraphRAG** - Multi-hop entity reasoning using Neo4j
- **LLM Generation** - Multiple strategies (Simple, Context-aware, Chain-of-Thought)

The framework uses YAML-driven configuration for flexible composition of RAG techniques.

## Quick Start

**Get started in 5 minutes:**

```bash
# Clone the repository
git clone https://github.com/katmandoo212/aragsys.git
cd aragsys

# Install dependencies
uv sync

# Install Ollama (from ollama.com) and pull models
ollama pull glm-4.7:cloud
ollama pull bge-m3:latest

# Start the web server
uv run uvicorn backend.main:app --reload --port 8000

# Open http://localhost:8000 in your browser
```

That's it! The web interface provides everything you need.

## Documentation

- **[User Guide](docs/USER_GUIDE.md)** - Complete guide for using the web interface and API
- **[CLAUDE.md](CLAUDE.md)** - Development guidance for contributors
- **[Design Documents](docs/plans/)** - Architecture and implementation plans

## Current Status

**Phase:** Phase 8 Complete - Web Frontend (2026-02-10)

Phase 8 adds:
- FastAPI-based web frontend with query workspace
- Real-time progress streaming via SSE
- Document management (web fetch + file upload)
- Monitoring dashboard with analytics
- Bootstrap 5 + HTMX for responsive UI

**Total: 144 tests passing** (120 Phase 1-7 + 24 Phase 8)

## Requirements

- Python 3.13+
- uv (for package management)
- Ollama (for local LLM inference)
- Neo4j (optional, for GraphRAG features)

## Installation

### Step 1: Clone and Install

```bash
git clone https://github.com/katmandoo212/aragsys.git
cd aragsys
uv venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
uv sync
```

### Step 2: Install Ollama

Download from [ollama.com](https://ollama.com) and pull required models:

```bash
ollama pull glm-4.7:cloud       # Generation model
ollama pull bge-m3:latest        # Embedding model
```

### Step 3: (Optional) Neo4j for GraphRAG

Install from [neo4j.com](https://neo4j.com) and configure in `config/neo4j.yaml`.

## Running the Web Frontend

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

**Web Frontend Features:**
- Query workspace with real-time progress (SSE)
- Document management (web fetch + file upload)
- Monitoring dashboard with analytics
- Responsive UI built with Bootstrap 5 + HTMX

## Configuration

The framework uses YAML files for all configuration.

### models.yaml - Ollama connection and models

```yaml
ollama:
  base_url: "http://localhost:11434"

query_models:
  default: "glm-4.7:cloud"

embedding_models:
  default: "bge-m3:latest"
```

### techniques.yaml - RAG techniques

```yaml
techniques:
  naive_rag:
    class: techniques.naive_rag.NaiveRAGTechnique
    enabled: true
    config:
      chunk_size: 500
      top_k: 5
      embedding_model: "bge-m3:latest"
```

### pipelines.yaml - Pipeline compositions

```yaml
pipelines:
  naive_flow:
    query_model: "glm-4.7:cloud"
    techniques: [naive_rag, simple_generation]

  advanced_flow:
    query_model: "glm-4.7:cloud"
    techniques: [hyde, naive_rag, rerank, context_generation]
```

## Query Pipelines

| Pipeline | Description | Best For |
|----------|-------------|---------|
| `naive_flow` | Simple dense retrieval + generation | Quick answers, small collections |
| `advanced_flow` | HyDE + Reranking + Citations | Complex questions, medium collections |
| `graph_flow` | Entity-based multi-hop reasoning | Relationship queries |
| `full_flow` | All techniques combined | Maximum quality, large collections |

## Supported Document Formats

| Format | Extension | Features |
|--------|-----------|----------|
| PDF | .pdf | Structure preservation, tables, figures, page numbers |
| Markdown | .md, .markdown | Heading hierarchy (H1-H3), section paths |
| Text | .txt | Paragraph-based chunking |

## API Reference

### Endpoints

- `GET /api/health` - Health check
- `GET /api/pipelines` - Available pipelines
- `GET /api/query/history` - Query history
- `GET /api/query/metrics` - Query metrics
- `POST /api/query` - Submit a query
- `GET /api/query/stream/{task_id}` - Stream progress (SSE)
- `GET /api/documents` - List documents
- `POST /api/documents/fetch` - Fetch from URL
- `POST /api/documents/upload` - Upload file

### Example: Submit Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "pipeline": "naive_flow"}'
```

Response:
```json
{
  "task_id": "8f3a42a5-4e87-47d5-b2ff-2da5a1f59942",
  "status": "pending"
}
```

### Example: Stream Progress

```bash
curl -N http://localhost:8000/api/query/stream/{task_id}
```

Events:
```
data: {'status': 'embedding_query', 'progress': 10, ...}
data: {'status': 'retrieving', 'progress': 30, ...}
data: {'status': 'reranking', 'progress': 50, ...}
data: {'status': 'generating', 'progress': 70, ...}
data: {'status': 'complete', 'progress': 100, ...}
data: [DONE]
```

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_registry.py -v

# Run with coverage
uv run pytest tests/ --cov=. --cov-report=html
```

## Key Design Patterns

1. **Registry Pattern** - Dynamic technique loading from YAML configuration
2. **Strategy Pattern** - Swappable retrieval/generation techniques
3. **Builder Pattern** - Pipeline composition
4. **Duck Typing** - Flexible technique interfaces

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Configuration Layer (YAML)                │
│  - models.yaml      - Ollama and model settings             │
│  - techniques.yaml  - Technique definitions                 │
│  - pipelines.yaml   - Pipeline compositions                 │
│  - generation.yaml  - Generation settings                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Registry Layer                            │
│  - TechniqueRegistry loads and caches metadata              │
│  - Provides technique discovery and lookup                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Execution Layer (PocketFlow)              │
│  - NaiveRAGTechnique - Dense vector retrieval               │
│  - HyDETechnique - Hypothetical document embeddings         │
│  - GraphMultiHopTechnique - Multi-hop reasoning             │
│  - SimpleGenerationTechnique - Basic LLM generation         │
│  - ContextGenerationTechnique - Citation-aware generation   │
│  - ChainOfThoughtGenerationTechnique - Reasoning generation │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | Complete | Architecture foundation with mocked techniques |
| 2 | Complete | Naive RAG with dense retrieval (NaiveRAGTechnique) |
| 3 | Complete | Document Formats - PDF/Markdown chunking |
| 4 | Complete | Advanced Retrieval (HyDE, Multi-Query, Hybrid) |
| 5 | Complete | Precision (Reranking, Contextual Compression) |
| 6 | Complete | GraphRAG with multi-hop reasoning |
| 7 | Complete | Generation (Simple, Context, Chain-of-Thought) |
| 8 | Complete | Web Frontend (FastAPI + Bootstrap + HTMX) |

## Project Structure

```
aragsys/
├── backend/                          # FastAPI web application
│   ├── main.py                      # FastAPI app entry point
│   ├── db.py                        # SQLite database layer
│   ├── models/                      # Pydantic models
│   │   ├── query.py
│   │   ├── document.py
│   │   └── pipeline.py
│   ├── routers/                     # API routers
│   │   ├── query.py
│   │   ├── documents.py
│   │   ├── health.py
│   │   └── pipelines.py
│   ├── services/                    # Business logic
│   │   ├── query_engine.py
│   │   └── document_service.py
│   └── utils/                       # Backend utilities
│       └── web_fetcher.py
├── frontend/                         # Web UI
│   ├── templates/                   # Jinja2 templates
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── workspace.html
│   │   ├── documents.html
│   │   └── monitoring.html
│   └── static/                      # Static assets
│       ├── css/
│       └── js/
├── config/                           # YAML configuration
│   ├── models.yaml
│   ├── techniques.yaml
│   ├── pipelines.yaml
│   ├── generation.yaml
│   ├── graphrag.yaml
│   └── neo4j.yaml
├── techniques/                       # RAG technique implementations
│   ├── naive_rag.py
│   ├── hyde.py
│   ├── multi_query.py
│   ├── hybrid.py
│   ├── rerank.py
│   ├── compress.py
│   ├── graph_entity.py
│   ├── graph_multihop.py
│   ├── graph_expand.py
│   ├── generate_simple.py
│   ├── generate_context.py
│   └── generate_cot.py
├── stores/                           # Storage backends
│   ├── vector_store.py
│   └── neo4j_store.py
├── registry/                         # Technique metadata registry
│   └── technique_registry.py
├── nodes/                            # PocketFlow workflow nodes
│   └── technique_node.py
├── pipeline/                         # Pipeline composition
│   └── builder.py
├── ollama/                           # Ollama API client
│   └── client.py
├── utils/                            # Utilities
│   ├── document.py
│   ├── answer.py
│   ├── vector_store.py
│   ├── entity_extractor.py
│   ├── pdf_chunker.py
│   ├── markdown_chunker.py
│   └── text_chunker.py
├── tests/                            # Test suite (144 tests)
│   ├── test_*.py
│   └── backend/
│       └── test_*.py
├── docs/                             # Documentation
│   └── USER_GUIDE.md
└── pyproject.toml                    # Dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`uv run pytest tests/ -v`)
5. Update documentation as needed
6. Commit your changes
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [PocketFlow](https://the-pocket.github.io/PocketFlow/) for workflow orchestration
- Uses [Ollama](https://ollama.com/) for local LLM inference
- Inspired by modern RAG architecture patterns
- Uses [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF structure extraction
- Uses [ChromaDB](https://www.chromadb.com/) for vector storage
- Uses [FastAPI](https://fastapi.tiangolo.com/) for the web API
- Uses [Bootstrap](https://getbootstrap.com/) and [HTMX](https://htmx.org/) for the UI