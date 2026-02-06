# Phase 2 Design: Naive RAG with Dense Retrieval

**Status:** Approved for Implementation
**Date:** 2026-02-05

---

## Overview

Phase 2 implements the retrieval layer of the RAG system, focusing on document ingestion, embedding, and similarity search. The architecture integrates with Phase 1's pluggable technique system while introducing new components for vector storage and document processing.

**Scope:** Retrieval only. Generation will be implemented in a later phase.

---

## Architecture

### Core Components

| Component | Purpose |
|-----------|---------|
| **DocumentStore** | Wraps ChromaDB for persistent vector storage |
| **Chunker** | Semantic/paragraph-based chunking of documents |
| **NaiveRAGTechnique** | Protocol-based technique implementing retrieve() |
| **Enhanced OllamaClient** | Extended with embed() method |
| **Document** | Dataclass for retrieval results |

### Data Flow

**Ingestion:** Documents → Chunker → OllamaClient.embed() → DocumentStore → Persistence

**Retrieval:** Query → TechniqueNode → NaiveRAGTechnique.retrieve() → OllamaClient.embed() → DocumentStore.search() → Results

### Three-Layer Architecture (Phase 1 + 2)

1. **Configuration Layer (YAML):**
   - `config/techniques.yaml` - technique configs (chunk_size, top_k, embedding_model)
   - `config/vector_stores.yaml` - ChromaDB settings (collection name, persistence path)

2. **Registry Layer:**
   - `TechniqueRegistry` - loads technique metadata
   - NaiveRAGTechnique registered dynamically

3. **Execution Layer:**
   - `TechniqueNode` - wraps NaiveRAGTechnique via PocketFlow
   - PipelineBuilder composes workflows

---

## Component Specifications

### DocumentStore (new)

Wraps ChromaDB's embedded client.

**Methods:**
- `add_documents(chunks: list[tuple[str, dict, list[float]]])` - Insert documents
- `search(query_vector: list[float], top_k: int)` - Return top-k similar docs
- `clear_collection()` - Reset collection

**Config (vector_stores.yaml):**
```yaml
chromadb:
  persist_directory: "./data/vectors"
  collection_name: "scientific_docs"
```

### Chunker (new)

Semantic/paragraph-based chunking for TXT, PDF, and Markdown.

**Methods:**
- `chunk_document(file_path: str, max_chunk_size: int) -> list[tuple[str, dict]]`

**Metadata per chunk:**
- `source` - file path
- `chunk_index` - position in document
- `position` - character offsets

Uses regex for sentence/paragraph boundary detection. Configurable max chunk size.

### NaiveRAGTechnique (technique implementation)

Implements technique protocol via duck typing.

**Protocol:**
- `__init__(config: dict)` - constructor
- `retrieve(query: str) -> list[Document]` - retrieval method

**Behavior:**
1. Embeds query via OllamaClient
2. Calls DocumentStore.search() with embedded query
3. Returns list of Document objects

Config loaded from techniques.yaml (embedding_model, top_k).

### Document (dataclass)

```python
@dataclass
class Document:
    content: str
    metadata: dict
    score: float
```

Standard return type for all retrieval operations.

### Enhanced OllamaClient

**New method:**
```python
def embed(self, text: str, model: str) -> list[float]:
```

Calls Ollama's `/api/embed` endpoint. Includes retry logic for failures.

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| Unsupported file format | Clear error with supported formats list |
| File not found | FileNotFoundError with path |
| Empty document | Returns empty chunks, logs warning |
| Ollama not running | ConnectionError with startup suggestion |
| Model not found | Error with available models list |
| Rate limiting | Retry with exponential backoff (max 3) |
| Empty vector store | Returns empty results, logs warning |
| No search results | Returns empty list (valid outcome) |
| Invalid config | ValueError with descriptive message |

No silent failures. All errors guide user toward resolution.

---

## Dependencies

```toml
[project.dependencies]
# Existing from Phase 1
httpx = ">=0.28.1"
pocketflow = ">=0.0.3"
pydantic = ">=2.12.5"
pytest = ">=9.0.2"
pyyaml = ">=6.0.3"

# New for Phase 2
chromadb = ">=0.6.0"
pypdf = ">=5.0.0"  # PDF parsing
```

---

## Testing Strategy

### Unit Tests

`tests/test_chunker.py` (5 tests):
- TXT chunking with paragraphs
- PDF extraction and chunking
- Markdown structure preservation
- Empty file handling
- Large file with chunk_size limit

`tests/test_document_store.py` (5 tests):
- Add and retrieve round-trip
- Search empty store
- Search with top_k
- Clear collection
- Data persistence

`tests/test_naive_rag_technique.py` (4 tests):
- Retrieve returns correct Document type
- Retrieve with no results
- Retrieve respects top_k config
- Config values applied correctly

`tests/test_ollama_client.py` (extend existing):
- Embed returns vector
- Embed uses correct model
- Embed handles connection error

### Integration Tests

`tests/test_retrieval_integration.py` (3 tests):
- Full ingestion flow (file → chunk → embed → store)
- Full retrieval flow (query → embed → search → results)
- TechniqueNode integration with NaiveRAGTechnique

**Total: ~18 new tests**

### Test Fixtures

Sample documents in `tests/fixtures/`:
- `sample.txt`
- `sample.pdf`
- `sample.md`

---

## Files to Create

```
stores/
  ├── __init__.py
  ├── document_store.py
  └── __init__.py

utils/
  ├── __init__.py
  ├── chunker.py
  └── __init__.py

techniques/
  ├── __init__.py
  ├── naive_rag.py
  └── base.py (protocol)

config/
  └── vector_stores.yaml

tests/
  ├── fixtures/
  │   ├── sample.txt
  │   ├── sample.pdf
  │   └── sample.md
  ├── test_chunker.py
  ├── test_document_store.py
  ├── test_naive_rag_technique.py
  └── test_retrieval_integration.py
```

## Files to Modify

- `ollama/client.py` - Add `embed()` method
- `tests/test_ollama.py` - Add embedding tests
- `pyproject.toml` - Add chromadb and pypdf dependencies

---

## Success Criteria

Phase 2 is complete when:
1. All new unit tests pass
2. Integration test passes full ingestion flow
3. Integration test passes full retrieval flow
4. OllamaClient.embed() generates valid vectors
5. DocumentStore persists and retrieves documents correctly
6. Chunker handles TXT, PDF, and Markdown formats
7. NaiveRAGTechnique integrates with existing TechniqueNode pattern
8. Error handling covers all documented scenarios

---

## Future Work (Phases 3-6)

Phase 2 provides the foundation for:
- Phase 3: Advanced retrieval (HyDE, Multi-Query, Hybrid)
- Phase 4: Precision (Reranking, Contextual Compression)
- Phase 5: GraphRAG with multi-hop reasoning
- Phase 6: Agentic self-correction

The retrieval architecture is pluggable, enabling easy addition of new techniques in later phases.