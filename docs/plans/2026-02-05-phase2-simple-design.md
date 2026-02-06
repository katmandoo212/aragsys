# Phase 2: Simple Naive RAG - Design

**Status:** Approved for Implementation
**Date:** 2026-02-05

---

## Overview

Phase 2 implements naive RAG retrieval for TXT documents only. Minimal, straightforward design integrating with Phase 1's architecture.

**Scope:**
- TXT document ingestion with paragraph-based chunking
- Dense embeddings via Ollama
- ChromaDB vector storage (minimal wrapper)
- Similarity search returning top-k results
- Retrieval only (generation in later phase)

**Simplifications:**
- TXT format only (no PDF/Markdown parsing)
- Minimal ChromaDB wrapper (basic add/search)
- ~10 tests vs previous 28-task plan
- Extend existing YAML configs (no new config files)

---

## Architecture

### Core Components

| Component | Purpose |
|-----------|---------|
| **TextChunker** | Paragraph-based TXT chunking |
| **VectorStore** | Minimal ChromaDB wrapper |
| **NaiveRAGTechnique** | Technique protocol implementation |
| **Enhanced OllamaClient** | Add embed() method |

### Data Flow

**Ingestion:** TXT → TextChunker → OllamaClient.embed() → VectorStore → ChromaDB

**Retrieval:** Query → TechniqueNode → NaiveRAGTechnique.retrieve() → OllamaClient.embed() → VectorStore.search() → Results

---

## Component Specifications

### TextChunker

**Location:** `utils/text_chunker.py`

```python
class TextChunker:
    def chunk_file(self, path: str, max_chunk_size: int) -> list[tuple[str, dict]]:
        """Chunk TXT file by paragraphs.

        Returns list of (content, metadata) tuples.
        Metadata includes: source, chunk_index.
        """
```

- Splits on `\n\n` (double newline) for paragraphs
- Returns empty list for empty files
- Raises FileNotFoundError if file doesn't exist

### VectorStore

**Location:** `utils/vector_store.py`

```python
class VectorStore:
    def add_documents(self, chunks: list[tuple[str, dict, list[float]]]) -> None:
        """Add documents with embeddings to ChromaDB."""

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        """Search for similar documents.

        Returns list of dicts with: content, metadata, distance.
        """
```

- Wraps ChromaDB PersistentClient
- Auto-creates collection if needed
- Returns formatted results

### NaiveRAGTechnique

**Location:** `techniques/naive_rag.py`

```python
from utils.document import Document

class NaiveRAGTechnique:
    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = config.get("embedding_model", "bge-m3:latest")
        self.top_k = config.get("top_k", 5)

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve relevant documents for query."""
```

- Uses OllamaClient.embed() for queries
- Calls VectorStore.search()
- Converts results to Document objects

### OllamaClient Enhancement

**Location:** `ollama/client.py`

```python
def embed(self, text: str, model: str) -> list[float]:
    """Generate embeddings via Ollama."""
    url = f"{self.base_url}/api/embed"
    # Call API with retry logic
```

---

## Configuration

### Extend config/techniques.yaml

```yaml
techniques:
  naive_rag:
    enabled: true
    config:
      chunk_size: 500
      top_k: 5
      embedding_model: "bge-m3:latest"
      collection_name: "documents"
```

### Add to config/models.yaml

```yaml
vector_store:
  persist_directory: "./data/chroma"
```

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| File not found | FileNotFoundError with path |
| Empty file | Return empty chunks (no error) |
| Ollama not running | Clear error with startup suggestion |
| ChromaDB error | Propagate ChromaDB exception |
| Empty search | Return empty list (valid) |

---

## Testing

### Test Files

| File | Tests |
|------|-------|
| `tests/test_text_chunker.py` | 3 tests |
| `tests/test_vector_store.py` | 3 tests |
| `tests/test_naive_rag_technique.py` | 2 tests |
| `tests/test_ollama_client.py` (extend) | 2 tests |

**Total:** ~10 tests

### Test Approach

- Use tmp_path for temp files and ChromaDB dirs
- Mock OllamaClient for most tests
- Simple pytest fixtures

---

## Files to Create

```
utils/
  ├── __init__.py
  ├── text_chunker.py
  └── vector_store.py

techniques/
  ├── __init__.py
  └── naive_rag.py

tests/
  ├── test_text_chunker.py
  ├── test_vector_store.py
  └── test_naive_rag_technique.py
```

## Files to Modify

- `ollama/client.py` - Add embed() method
- `tests/test_ollama.py` - Add embed tests
- `config/techniques.yaml` - Extend naive_rag config
- `config/models.yaml` - Add vector_store section
- `pyproject.toml` - Add chromadb dependency

---

## Success Criteria

Phase 2 is complete when:
1. All ~10 tests pass
2. TXT files can be chunked and embedded
3. Similarity search returns top-k results
4. NaiveRAGTechnique integrates with existing technique system
5. ChromaDB persists across sessions

---

## Next Phases

This minimal foundation enables:
- Phase 3: Add PDF/Markdown chunking
- Phase 4: Add advanced retrieval (HyDE, Multi-Query)
- Phase 5: Add reranking and compression
- Phase 6: Add generation