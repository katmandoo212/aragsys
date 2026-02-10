---
title: Web Frontend Design - Scientific Agentic RAG Framework
type: feat
date: 2026-02-09
status: Approved
---

# Web Frontend Design - Scientific Agentic RAG Framework

## Overview

Build a FastAPI-based web application frontend for the Scientific Agentic RAG Framework. This is a **demo/prototype** to showcase the framework's capabilities with a polished, interactive UI.

### Purpose
- Demonstrate RAG pipeline features (retrieval, generation, citations)
- Visualize multi-hop reasoning and graph-based queries
- Showcase document ingestion from multiple sources
- Provide monitoring and analytics dashboard

### Key Requirements
- **Pre-loaded sample data** for immediate demo capability
- **Web document fetching** (HTML, PDF, MD from URLs)
- **File upload** as secondary option
- **Local only** (localhost, no authentication)
- **Complete UI** (query, documents, monitoring)
- **HTMX** for server-driven interactivity (minimal JS)

---

## Architecture

### Backend Layer (FastAPI)

```
backend/
├── main.py                    # FastAPI app entry point
├── config.py                  # Configuration loading
├── db.py                      # SQLite session management
├── models/
│   ├── query.py               # QueryRequest, QueryResponse
│   ├── document.py            # DocumentCreate, DocumentResponse
│   └── pipeline.py            # PipelineConfig
├── routers/
│   ├── query.py               # POST /api/query, GET /api/query/history
│   ├── documents.py           # GET/POST/DELETE /api/documents, POST /fetch
│   ├── pipelines.py           # GET /api/pipelines
│   └── health.py              # GET /api/health
├── services/
│   ├── query_engine.py        # RAG pipeline orchestration
│   ├── document_service.py    # Document CRUD + web fetching
│   └── metrics_service.py     # Analytics from SQLite
└── utils/
    ├── web_fetcher.py         # NEW: URL fetching and extraction
    └── streaming.py           # SSE response helpers
```

### Frontend Layer (Bootstrap 5 + HTMX)

```
frontend/
├── index.html                 # Landing page with quick search
├── workspace.html             # Query workspace (split-pane)
├── documents.html             # Document management
├── monitoring.html            # Analytics dashboard
├── css/
│   └── styles.css             # Custom styles
└── js/
    ├── app.js                 # App initialization
    ├── query.js               # SSE handling
    └── documents.js           # Upload/fetch logic
```

---

## Component Specifications

### Document Service (with Web Fetching)

**Endpoints:**
- `GET /api/documents` - List all indexed documents
- `POST /api/documents/upload` - Upload file (multipart form)
- `POST /api/documents/fetch` - **NEW:** Fetch from URL
- `DELETE /api/documents/{id}` - Delete document

**Web Fetching Flow:**
```python
async def fetch_document(url: str) -> Document:
    # 1. Validate URL format
    # 2. HTTP GET with timeout (30s)
    # 3. Extract content based on content-type:
    #    - text/html → BeautifulSoup → body text
    #    - application/pdf → PDFChunker
    #    - text/markdown → MarkdownChunker
    # 4. Chunk → Embed → Store in ChromaDB
    # 5. Return metadata (title, url, chunk_count)
```

**Web Fetching Dependencies:**
```toml
beautifulsoup4 = "^4.12.0"
lxml = "^5.0.0"
```

### Query Engine (with SSE Progress)

**Endpoint:** `POST /api/query`

**Async Flow:**
1. Start async task, return `task_id`
2. Client subscribes to SSE: `GET /api/query/stream/{task_id}`
3. Server sends progress events:
   - `{"status": "embedding_query"}`
   - `{"status": "retrieving", "count": 5}`
   - `{"status": "reranking", "count": 5}`
   - `{"status": "generating"}`
   - `{"status": "complete", "answer": {...}}`

### Frontend Pages

#### Query Workspace (`/workspace`)
- Split-pane layout: query input (left) + answer display (right)
- Pipeline selector (naive_flow, advanced_flow, graph_flow)
- Real-time progress indicator (SSE)
- Answer with citations as superscript `[1]`
- Retrieved documents sidebar with scores
- Source document modal on citation click

#### Document Management (`/documents`)
- Document list with metadata (source, chunk count, date)
- **URL fetch form** - input URL, click fetch
- File upload drag-and-drop zone
- Delete with confirmation
- Document preview modal

#### Monitoring Dashboard (`/monitoring`)
- Daily query count chart (Chart.js)
- Response time percentiles (P50, P95, P99)
- Success rate gauge
- Recent error log with filter
- Token usage tracking

---

## Data Flow

### Query Flow with SSE

```
User Input → POST /api/query → task_id
    ↓
Client SSE: GET /api/query/stream/{task_id}
    ↓
Server Progress Events:
    embedding_query → retrieving → reranking → generating → complete
    ↓
Final Answer rendered with citations [1][2][3]
```

### Document Fetching Flow

```
User enters URL → POST /api/documents/fetch
    ↓
Validate URL → HTTP GET (30s timeout)
    ↓
Extract content by type:
    HTML → BeautifulSoup body
    PDF → PDFChunker
    MD → MarkdownChunker
    ↓
Chunk → Embed (OllamaClient) → Store (ChromaDB)
    ↓
Return document metadata
```

---

## Error Handling

| Scenario | Status Code | User Message |
|----------|-------------|--------------|
| Invalid URL | 400 | "Invalid URL format" |
| URL fetch timeout | 504 | "URL fetch timed out (30s). Retry?" |
| Content too large | 413 | "Content exceeds 5MB limit" |
| Unsupported content type | 415 | "Unsupported content type" |
| Ollama unavailable | 503 | "LLM service unavailable - check Ollama" |
| Empty search results | 200 | "No documents found for query" |
| Malformed content | 200 | "Partial content extracted" |

---

## Sample Data

Pre-loaded scientific documents in `data/samples/`:
- `clinical_trial.pdf` - Clinical research paper
- `ai_medical_review.md` - AI in medicine review
- `pharmacology_study.pdf` - Drug interaction study
- `medical_guidelines.txt` - Clinical guidelines

---

## Testing Strategy

| Test Type | Coverage |
|-----------|----------|
| Unit Tests | FastAPI routes, web fetching, query engine |
| Integration Tests | Full query flow, fetch → embed → store |
| E2E Tests | Playwright for key user journeys |
| Mocking | OllamaClient, HTTP requests |

**Test Files:**
- `tests/backend/test_api_routes.py`
- `tests/backend/test_web_fetcher.py`
- `tests/backend/test_query_engine.py`
- `tests/e2e/test_query_workspace.py`

---

## Deployment (Local Demo)

```bash
# Start Ollama
ollama serve

# Start Neo4j (optional, for GraphRAG)
neo4j start

# Run the web app
uv run uvicorn backend.main:app --reload --port 8000

# Open browser to http://localhost:8000
```

---

## Success Metrics

- Queries complete in < 5 seconds for simple requests
- Web document fetch completes in < 30 seconds
- UI is responsive and polished (Bootstrap 5)
- All 3 generation strategies demonstrate clearly
- Citations render correctly with source links
- 95% of queries complete without errors

---

## References

- Existing plan: `docs/plans/2026-02-08-feat-web-frontend-plan.md`
- FastAPI Documentation: https://fastapi.tiangolo.com/
- HTMX Documentation: https://htmx.org/
- Bootstrap 5 Documentation: https://getbootstrap.com/docs/5.3/