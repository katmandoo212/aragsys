# User Guide

Welcome to the Scientific Agentic RAG Framework (aragsys). This guide will help you get started with using the framework for interactive RAG queries over academic and clinical literature.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Setting Up Ollama](#setting-up-ollama)
4. [Running the Web Frontend](#running-the-web-frontend)
5. [Using the Web Interface](#using-the-web-interface)
6. [Working with Documents](#working-with-documents)
7. [Query Pipelines](#query-pipelines)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

## Quick Start

The fastest way to get started is with the web interface:

```bash
# Start the web server
uv run uvicorn backend.main:app --reload --port 8000

# Open http://localhost:8000 in your browser
```

That's it! The web interface provides everything you need to query documents and see results.

## Installation

### Prerequisites

- **Python 3.13+** - The framework requires Python 3.13 or later
- **uv** - Fast Python package manager (recommended)
- **Ollama** - Local LLM inference engine
- **Neo4j** (Optional) - For GraphRAG features

### Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/katmandoo212/aragsys.git
cd aragsys

# Create virtual environment
uv venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv sync
```

## Setting Up Ollama

### Install Ollama

Download and install Ollama from [ollama.com](https://ollama.com).

### Pull Required Models

The framework requires at least one generation model and one embedding model:

```bash
# Generation models (required)
ollama pull glm-4.7:cloud

# Embedding models (required)
ollama pull bge-m3:latest
```

### Recommended Additional Models

```bash
# Additional generation models
ollama pull llama3:8b
ollama pull llama3:70b

# Additional embedding models
ollama pull nomic-embed-text-v2-moe:latest
ollama pull qwen3-embedding:latest
ollama pull granite-embedding:278m
ollama pull mxbai-embed-large:latest
```

### Verify Ollama is Running

```bash
# Check Ollama is running
ollama list

# Test a model
ollama run glm-4.7:cloud "Hello, world!"
```

### Optional: Neo4j for GraphRAG

If you want to use GraphRAG features (multi-hop entity reasoning), install Neo4j:

1. Download from [neo4j.com](https://neo4j.com)
2. Start Neo4j and create a database
3. Update `config/neo4j.yaml` with your credentials:

```yaml
neo4j:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "your_password"
  database: "rag"
```

## Running the Web Frontend

### Start the Server

```bash
# From the project root
uv run uvicorn backend.main:app --reload --port 8000
```

### Access the Application

Open your browser and navigate to: **http://localhost:8000**

You'll see the landing page with:
- Feature overview
- Quick links to key sections
- Getting started tips

## Using the Web Interface

### Query Workspace

The query workspace is your primary interface for asking questions:

1. **Enter your query** in the text field
2. **Select a pipeline** (naive_flow, advanced_flow, graph_flow, full_flow)
3. **Click "Submit Query"**
4. **Watch real-time progress** as the query is processed
5. **View results** with citations and metadata

**Progress Stages:**
- **Embedding Query** (10%) - Converting your query to vector space
- **Retrieving Documents** (30%) - Finding relevant documents
- **Reranking** (50%) - Reordering by relevance
- **Generating Answer** (70%) - Creating the response
- **Complete** (100%) - Results ready

### Document Management

Add documents to your knowledge base from two sources:

#### Web Fetch
1. Navigate to the **Documents** page
2. Enter a URL in the "Fetch from URL" field
3. Click "Fetch Document"
4. The document will be processed and added to your collection

#### File Upload
1. Navigate to the **Documents** page
2. Click "Choose File" or drag and drop
3. Select a PDF, Markdown, or text file
4. Click "Upload Document"

**Supported Formats:**
- PDF (.pdf) - Preserves tables, figures, page numbers
- Markdown (.md, .markdown) - Preserves heading hierarchy
- Text (.txt) - Simple text files

### Monitoring Dashboard

View analytics about your queries:
- **Total Queries** - All-time query count
- **Success Rate** - Percentage of successful queries
- **Average Response Time** - Mean time in milliseconds
- **Recent 24h** - Queries in the last day

View charts showing:
- Query volume over time
- Response time trends
- Pipeline usage distribution

## Working with Documents

### Adding Sample Documents

Place sample documents in the `data/samples/` directory:

```bash
data/samples/
├── sample1.pdf
├── sample2.md
└── sample3.txt
```

### Document Chunking

Documents are automatically chunked based on format:

**PDF:**
- Split by paragraphs (double newlines)
- Preserves page numbers, tables, figure captions
- Maximum chunk size: 500 tokens (configurable)

**Markdown:**
- Split by section (H1-H3 headings)
- Preserves heading paths for context
- Maximum chunk size: 500 tokens (configurable)

**Text:**
- Split by paragraphs
- Simple metadata (source, chunk_index)

### Understanding Citations

Answers include citations in the format `[1]`, `[2]`, etc.:

```
Paris is the capital of France [1]. The city is known for...
```

Each citation number corresponds to a document in the retrieved results list.

## Query Pipelines

The framework provides pre-configured pipelines:

### naive_flow

Simple and fast. Good for straightforward questions.

**Techniques:** `naive_rag` → `simple_generation`

**Use when:**
- Questions have clear, direct answers
- You need quick responses
- Document collection is small

### advanced_flow

Uses multiple retrieval strategies and reranking.

**Techniques:** `hyde` → `naive_rag` → `rerank` → `context_generation`

**Use when:**
- Questions are complex or ambiguous
- You need higher quality results
- Document collection is medium-sized

### graph_flow

Entity-based multi-hop reasoning.

**Techniques:** `graph_entity` → `graph_multihop` → `cot_generation`

**Use when:**
- Questions involve relationships between entities
- You need step-by-step reasoning
- Neo4j is configured

### full_flow

All techniques combined for maximum quality.

**Techniques:** `hyde` → `naive_rag` → `multi_query` → `hybrid` → `rerank` → `compress` → `graph_entity` → `graph_multihop` → `context_generation`

**Use when:**
- Questions are complex and multi-faceted
- Quality is more important than speed
- Document collection is large

## API Reference

### Health Check

Check if the server is running:

```bash
curl http://localhost:8000/api/health
```

Response:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "models": {
    "generation": "glm-4.7:cloud",
    "embedding": "bge-m3:latest"
  }
}
```

### List Pipelines

Get available query pipelines:

```bash
curl http://localhost:8000/api/pipelines
```

Response:
```json
{
  "pipelines": {
    "naive_flow": {
      "name": "naive_flow",
      "query_model": "glm-4.7:cloud",
      "techniques": ["naive_rag", "generation"],
      "description": "naive_flow pipeline"
    }
  },
  "default_pipeline": "glm-4.7:cloud"
}
```

### Submit Query

Submit a new query:

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

### Stream Query Progress

Stream real-time progress using SSE:

```bash
curl -N http://localhost:8000/api/query/stream/8f3a42a5-4e87-47d5-b2ff-2da5a1f59942
```

Events:
```
data: {'status': 'embedding_query', 'progress': 10, 'message': 'Embedding query...', 'data': None}

data: {'status': 'retrieving', 'progress': 30, 'message': 'Retrieving documents...', 'data': None}

data: {'status': 'reranking', 'progress': 50, 'message': 'Reranking documents...', 'data': None}

data: {'status': 'generating', 'progress': 70, 'message': 'Generating answer...', 'data': None}

data: {'status': 'complete', 'progress': 100, 'message': 'Complete', 'data': {...}}

data: [DONE]
```

### Query History

Get recent queries:

```bash
curl http://localhost:8000/api/query/history?limit=10
```

Response:
```json
{
  "queries": [
    {
      "query_id": 1,
      "query": "What is the capital of France?",
      "pipeline": "naive_flow",
      "timestamp": "2026-02-10T22:59:08.308061+00:00",
      "success": true
    }
  ]
}
```

### Query Metrics

Get aggregate metrics:

```bash
curl http://localhost:8000/api/query/metrics
```

Response:
```json
{
  "total_queries": 100,
  "success_rate": 95.5,
  "avg_response_time_ms": 1250,
  "recent_24h": 25
}
```

### List Documents

Get all documents:

```bash
curl http://localhost:8000/api/documents
```

Response:
```json
{
  "documents": [
    {
      "doc_id": 1,
      "title": "Sample Document",
      "source": "sample.pdf",
      "chunk_count": 15,
      "added_at": "2026-02-10T22:59:08.308061+00:00"
    }
  ]
}
```

### Fetch Document from URL

```bash
curl -X POST http://localhost:8000/api/documents/fetch \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/document.pdf"}'
```

### Upload Document

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@/path/to/document.pdf"
```

## Troubleshooting

### Ollama Not Responding

**Problem:** Queries fail with connection errors.

**Solution:**
1. Check Ollama is running: `ollama list`
2. Verify Ollama is listening: `curl http://localhost:11434/api/tags`
3. Check `config/models.yaml` for correct base_url

### Documents Not Being Found

**Problem:** Queries return no results.

**Solution:**
1. Check documents have been added: `curl http://localhost:8000/api/documents`
2. Verify documents are in supported formats (PDF, MD, TXT)
3. Try a simpler pipeline (naive_flow)
4. Check embedding model is available: `ollama list`

### Server Won't Start

**Problem:** `uvicorn` fails to start.

**Solution:**
1. Check port 8000 is available: `netstat -ano | findstr :8000` (Windows) or `lsof -i :8000` (macOS/Linux)
2. Use a different port: `uvicorn backend.main:app --port 8001`
3. Check all dependencies are installed: `uv sync`

### Slow Query Performance

**Problem:** Queries take too long.

**Solution:**
1. Use a faster pipeline (naive_flow instead of full_flow)
2. Use a smaller embedding model (granite-embedding:278m)
3. Reduce chunk size in `config/techniques.yaml`
4. Reduce `top_k` in technique configurations

### Database Errors

**Problem:** Query history or metrics fail to load.

**Solution:**
1. Delete `backend/data/aragsys.db` (it will be recreated)
2. Check directory permissions for `backend/data/`
3. Verify SQLite is available

## Next Steps

- **Configuration:** Learn about customizing pipelines and techniques in `config/`
- **API Development:** Use the REST API to build custom integrations
- **Python API:** Use the Python library directly for programmatic access
- **Testing:** Run the test suite to verify your setup: `uv run pytest tests/ -v`

## Support

- **GitHub Issues:** Report bugs and feature requests
- **Documentation:** See `docs/plans/` for design documents
- **Tests:** Check `tests/` for usage examples