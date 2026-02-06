# Phase 2 Simple Naive RAG - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a minimal retrieval layer for the RAG system with TXT document ingestion, Ollama embeddings, ChromaDB vector storage, and similarity search.

**Architecture:** Three components integrate with Phase 1's technique system - TextChunker splits TXT by paragraphs, VectorStore wraps ChromaDB for persistence, NaiveRAGTechnique implements the retrieve() protocol using OllamaClient for embeddings.

**Tech Stack:** Python 3.13+, ChromaDB (vector store), Ollama (embeddings), pytest (testing), TDD workflow.

---

## Prerequisites

**Before starting:**

```bash
# Ensure we're in the worktree
cd .worktrees/phase2-simple

# Install dependencies
uv add chromadb

# Create new directories
mkdir -p utils techniques
touch utils/__init__.py techniques/__init__.py
```

---

## Task 1: Add ChromaDB Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add chromadb to dependencies**

Edit `pyproject.toml` and add `chromadb>=0.6.0` to the dependencies list.

**Step 2: Install dependencies**

Run: `uv sync`
Expected: Dependencies installed successfully

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add ChromaDB for vector storage"
```

---

## Task 2: TextChunker - Basic Structure

**Files:**
- Create: `utils/text_chunker.py`
- Create: `tests/test_text_chunker.py`

**Step 1: Write failing test**

Create `tests/test_text_chunker.py`:

```python
import pytest
from utils.text_chunker import TextChunker

def test_chunk_file_splits_by_paragraphs(tmp_path):
    # Create test file with paragraphs
    test_file = tmp_path / "test.txt"
    test_file.write_text("Paragraph one.\n\nParagraph two.\n\nParagraph three.")

    chunker = TextChunker()
    chunks = chunker.chunk_file(str(test_file), max_chunk_size=100)

    assert len(chunks) == 3
    assert all(isinstance(content, str) for content, meta in chunks)
    assert all("source" in meta for _, meta in chunks)
    assert all("chunk_index" in meta for _, meta in chunks)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_text_chunker.py::test_chunk_file_splits_by_paragraphs -v`
Expected: FAIL with "TextChunker not defined"

**Step 3: Write minimal implementation**

Create `utils/text_chunker.py`:

```python
from pathlib import Path


class TextChunker:
    def chunk_file(self, path: str, max_chunk_size: int) -> list[tuple[str, dict]]:
        """Chunk TXT file by paragraphs.

        Returns list of (content, metadata) tuples.
        Metadata includes: source, chunk_index.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            return []

        # Split by double newline (paragraphs)
        paragraphs = content.split('\n\n')
        chunks = []

        for idx, para in enumerate(paragraphs):
            if not para.strip():
                continue

            chunks.append((
                para.strip(),
                {"source": str(file_path), "chunk_index": idx}
            ))

        return chunks
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_text_chunker.py::test_chunk_file_splits_by_paragraphs -v`
Expected: PASS

**Step 5: Commit**

```bash
git add utils/text_chunker.py tests/test_text_chunker.py
git commit -m "feat: add TextChunker for TXT paragraph chunking

- Split TXT files by double newline
- Include source and chunk_index in metadata
- Handle empty files gracefully"
```

---

## Task 3: TextChunker - Empty File

**Files:**
- Modify: `tests/test_text_chunker.py`

**Step 1: Write test**

Add to `tests/test_text_chunker.py`:

```python
def test_chunk_empty_file_returns_empty_list(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")

    chunker = TextChunker()
    chunks = chunker.chunk_file(str(empty_file), max_chunk_size=100)

    assert chunks == []
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_text_chunker.py::test_chunk_empty_file_returns_empty_list -v`
Expected: PASS (already handled)

**Step 3: Commit**

```bash
git add tests/test_text_chunker.py
git commit -m "test: add empty file test for TextChunker"
```

---

## Task 4: TextChunker - File Not Found

**Files:**
- Modify: `tests/test_text_chunker.py`

**Step 1: Write test**

Add to `tests/test_text_chunker.py`:

```python
def test_chunk_file_not_found_raises_error():
    chunker = TextChunker()

    with pytest.raises(FileNotFoundError, match="File not found"):
        chunker.chunk_file("nonexistent.txt", max_chunk_size=100)
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_text_chunker.py::test_chunk_file_not_found_raises_error -v`
Expected: PASS (already handled)

**Step 3: Commit**

```bash
git add tests/test_text_chunker.py
git commit -m "test: add file not found test for TextChunker"
```

---

## Task 5: OllamaClient - Add embed() Method

**Files:**
- Modify: `ollama/client.py`
- Modify: `tests/test_ollama.py`

**Step 1: Write failing test**

Add to `tests/test_ollama.py`:

```python
def test_ollama_embed_returns_vector(tmp_path):
    # Create mock config
    config_file = tmp_path / "models.yaml"
    config_file.write_text('ollama:\n  base_url: "http://localhost:11434"')

    client = OllamaClient.from_config(str(config_file))

    with patch('httpx.Client.post') as mock_post:
        # Mock the embedding response
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response

        vector = client.embed("test query", "bge-m3:latest")

        assert vector == [0.1, 0.2, 0.3]
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ollama.py::test_ollama_embed_returns_vector -v`
Expected: FAIL with "embed method not found"

**Step 3: Add embed method**

Modify `ollama/client.py`:

```python
from dataclasses import dataclass
import yaml
import httpx

@dataclass
class OllamaClient:
    base_url: str

    @classmethod
    def from_config(cls, config_path: str) -> "OllamaClient":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(base_url=config["ollama"]["base_url"])

    def embed(self, text: str, model: str) -> list[float]:
        """Generate embeddings for the given text using the specified model."""
        url = f"{self.base_url}/api/embed"
        payload = {
            "model": model,
            "input": text
        }

        response = httpx.Client().post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        return data.get("embedding", [])
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ollama.py::test_ollama_embed_returns_vector -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ollama/client.py tests/test_ollama.py
git commit -m "feat: add embed() method to OllamaClient

- Call Ollama /api/embed endpoint
- Return embedding vector as list of floats"
```

---

## Task 6: OllamaClient - Connection Error

**Files:**
- Modify: `tests/test_ollama.py`

**Step 1: Write test**

Add to `tests/test_ollama.py`:

```python
def test_ollama_embed_connection_error(tmp_path):
    config_file = tmp_path / "models.yaml"
    config_file.write_text('ollama:\n  base_url: "http://localhost:11434"')

    client = OllamaClient.from_config(str(config_file))

    with patch('httpx.Client.post') as mock_post:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(httpx.ConnectError):
            client.embed("test", "bge-m3:latest")
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_ollama.py::test_ollama_embed_connection_error -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_ollama.py
git commit -m "test: add connection error test for OllamaClient.embed()"
```

---

## Task 7: VectorStore - Basic Structure

**Files:**
- Create: `utils/vector_store.py`
- Create: `tests/test_vector_store.py`

**Step 1: Write failing test**

Create `tests/test_vector_store.py`:

```python
import pytest
from utils.vector_store import VectorStore

def test_vector_store_creation(tmp_path):
    config = {
        "persist_directory": str(tmp_path / "vectors"),
        "collection_name": "test_collection"
    }

    store = VectorStore(config)

    assert store.config == config
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vector_store.py::test_vector_store_creation -v`
Expected: FAIL with "VectorStore not defined"

**Step 3: Write minimal implementation**

Create `utils/vector_store.py`:

```python
from dataclasses import dataclass
import chromadb

@dataclass
class VectorStore:
    config: dict

    def __post_init__(self):
        self.client = chromadb.PersistentClient(
            path=self.config["persist_directory"]
        )
        self.collection = self.client.get_or_create_collection(
            name=self.config["collection_name"]
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_vector_store.py::test_vector_store_creation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add utils/vector_store.py tests/test_vector_store.py
git commit -m "feat: add VectorStore with ChromaDB

- Wrap ChromaDB PersistentClient
- Auto-create collection from config"
```

---

## Task 8: VectorStore - Add Documents

**Files:**
- Modify: `utils/vector_store.py`
- Modify: `tests/test_vector_store.py`

**Step 1: Write failing test**

Add to `tests/test_vector_store.py`:

```python
def test_vector_store_add_documents(tmp_path):
    config = {
        "persist_directory": str(tmp_path / "vectors"),
        "collection_name": "test_collection"
    }

    store = VectorStore(config)

    chunks = [
        ("First chunk", {"source": "test.txt", "index": 0}, [0.1, 0.2, 0.3]),
        ("Second chunk", {"source": "test.txt", "index": 1}, [0.4, 0.5, 0.6]),
    ]

    store.add_documents(chunks)

    count = store.collection.count()
    assert count == 2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vector_store.py::test_vector_store_add_documents -v`
Expected: FAIL with "add_documents method not found"

**Step 3: Add add_documents method**

Modify `utils/vector_store.py`:

```python
    def add_documents(self, chunks: list[tuple[str, dict, list[float]]]) -> None:
        """Add documents with embeddings to ChromaDB."""
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for idx, (content, metadata, embedding) in enumerate(chunks):
            ids.append(f"doc_{idx}")
            documents.append(content)
            metadatas.append(metadata)
            embeddings.append(embedding)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_vector_store.py::test_vector_store_add_documents -v`
Expected: PASS

**Step 5: Commit**

```bash
git add utils/vector_store.py tests/test_vector_store.py
git commit -m "feat: add add_documents() to VectorStore

- Add documents with embeddings to ChromaDB
- Handle empty chunk lists gracefully"
```

---

## Task 9: VectorStore - Search

**Files:**
- Modify: `utils/vector_store.py`
- Modify: `tests/test_vector_store.py`

**Step 1: Write failing test**

Add to `tests/test_vector_store.py`:

```python
def test_vector_store_search(tmp_path):
    config = {
        "persist_directory": str(tmp_path / "vectors"),
        "collection_name": "test_collection"
    }

    store = VectorStore(config)

    # Add test documents
    chunks = [
        ("First chunk", {"source": "test.txt"}, [0.1, 0.2, 0.3]),
        ("Second chunk", {"source": "test.txt"}, [0.4, 0.5, 0.6]),
    ]
    store.add_documents(chunks)

    # Search with query vector
    results = store.search([0.1, 0.2, 0.3], top_k=2)

    assert len(results) == 2
    assert all("content" in r for r in results)
    assert all("metadata" in r for r in results)
    assert all("distance" in r for r in results)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vector_store.py::test_vector_store_search -v`
Expected: FAIL with "search method not found"

**Step 3: Add search method**

Modify `utils/vector_store.py`:

```python
    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        """Search for similar documents using the query vector."""
        if not query_vector:
            return []

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        # Format results
        formatted_results = []
        for idx in range(len(results["ids"][0])):
            formatted_results.append({
                "content": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "distance": results.get("distances", [[0]])[0][idx] if "distances" in results else 0.0
            })

        return formatted_results
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_vector_store.py::test_vector_store_search -v`
Expected: PASS

**Step 5: Commit**

```bash
git add utils/vector_store.py tests/test_vector_store.py
git commit -m "feat: add search() to VectorStore

- Query ChromaDB with embedding vector
- Return formatted results with content, metadata, distance"
```

---

## Task 10: VectorStore - Empty Search

**Files:**
- Modify: `tests/test_vector_store.py`

**Step 1: Write test**

Add to `tests/test_vector_store.py`:

```python
def test_vector_store_search_empty_store(tmp_path):
    config = {
        "persist_directory": str(tmp_path / "vectors"),
        "collection_name": "test_collection"
    }

    store = VectorStore(config)

    results = store.search([0.1, 0.2, 0.3], top_k=5)

    assert results == []
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_vector_store.py::test_vector_store_search_empty_store -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_vector_store.py
git commit -m "test: add empty store search test"
```

---

## Task 11: NaiveRAGTechnique - Basic Structure

**Files:**
- Create: `techniques/naive_rag.py`
- Create: `tests/test_naive_rag_technique.py`

**Step 1: Write failing test**

Create `tests/test_naive_rag_technique.py`:

```python
import pytest
from techniques.naive_rag import NaiveRAGTechnique

def test_naive_rag_technique_creation():
    config = {
        "embedding_model": "bge-m3:latest",
        "top_k": 5,
        "collection_name": "documents"
    }

    technique = NaiveRAGTechnique(config)

    assert technique.config == config
    assert technique.embedding_model == "bge-m3:latest"
    assert technique.top_k == 5
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_naive_rag_technique.py::test_naive_rag_technique_creation -v`
Expected: FAIL with "NaiveRAGTechnique not defined"

**Step 3: Write minimal implementation**

Create `techniques/naive_rag.py`:

```python
from dataclasses import dataclass

from utils.document import Document


@dataclass
class NaiveRAGTechnique:
    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = config.get("embedding_model", "bge-m3:latest")
        self.top_k = config.get("top_k", 5)
        self.collection_name = config.get("collection_name", "documents")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_naive_rag_technique.py::test_naive_rag_technique_creation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add techniques/naive_rag.py tests/test_naive_rag_technique.py
git commit -m "feat: add NaiveRAGTechnique basic structure

- Initialize with config dict
- Store embedding_model, top_k, collection_name"
```

---

## Task 12: NaiveRAGTechnique - Retrieve Method

**Files:**
- Modify: `techniques/naive_rag.py`
- Modify: `tests/test_naive_rag_technique.py`

**Step 1: Write failing test**

Add to `tests/test_naive_rag_technique.py`:

```python
def test_naive_rag_technique_retrieve_returns_documents():
    config = {
        "embedding_model": "bge-m3:latest",
        "top_k": 2,
        "collection_name": "test_collection"
    }

    technique = NaiveRAGTechnique(config)

    with patch.object(technique, '_embed_query') as mock_embed, \
         patch('utils.vector_store.VectorStore') as mock_store_class:

        mock_store = Mock()
        mock_store.search.return_value = [
            {"content": "Doc 1", "metadata": {"source": "test"}, "distance": 0.1},
            {"content": "Doc 2", "metadata": {"source": "test"}, "distance": 0.2}
        ]
        mock_store_class.return_value = mock_store

        mock_embed.return_value = [0.1, 0.2, 0.3]

        results = technique.retrieve("test query")

        assert len(results) == 2
        assert all(isinstance(r, Document) for r in results)
        assert results[0].content == "Doc 1"
        assert results[0].score == 0.1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_naive_rag_technique.py::test_naive_rag_technique_retrieve_returns_documents -v`
Expected: FAIL with "retrieve method not found"

**Step 3: Add retrieve method**

Modify `techniques/naive_rag.py`:

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ollama.client import OllamaClient
from utils.vector_store import VectorStore

if TYPE_CHECKING:
    from utils.document import Document


@dataclass
class NaiveRAGTechnique:
    def __init__(self, config: dict, ollama_client: OllamaClient = None, vector_store: VectorStore = None):
        self.config = config
        self.embedding_model = config.get("embedding_model", "bge-m3:latest")
        self.top_k = config.get("top_k", 5)
        self.collection_name = config.get("collection_name", "documents")

        self.ollama_client = ollama_client
        self.vector_store = vector_store

    def retrieve(self, query: str) -> list["Document"]:
        """Retrieve relevant documents for the given query."""
        # Embed the query
        query_vector = self._embed_query(query)

        # Search the store
        results = self.vector_store.search(query_vector, self.top_k)

        # Convert to Document objects
        documents = []
        for result in results:
            # Use distance as inverse score
            score = 1.0 - min(result["distance"], 1.0)

            from utils.document import Document
            documents.append(Document(
                content=result["content"],
                metadata=result["metadata"],
                score=score
            ))

        return documents

    def _embed_query(self, query: str) -> list[float]:
        """Embed the query using Ollama."""
        return self.ollama_client.embed(query, self.embedding_model)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_naive_rag_technique.py::test_naive_rag_technique_retrieve_returns_documents -v`
Expected: PASS

**Step 5: Commit**

```bash
git add techniques/naive_rag.py tests/test_naive_rag_technique.py
git commit -m "feat: add retrieve() method to NaiveRAGTechnique

- Embed query using OllamaClient
- Search VectorStore with embedded query
- Convert results to Document objects"
```

---

## Task 13: NaiveRAGTechnique - No Results

**Files:**
- Modify: `tests/test_naive_rag_technique.py`

**Step 1: Write test**

Add to `tests/test_naive_rag_technique.py`:

```python
def test_naive_rag_technique_retrieve_no_results():
    config = {"embedding_model": "bge-m3:latest", "top_k": 5}

    technique = NaiveRAGTechnique(config)

    with patch('utils.vector_store.VectorStore') as mock_store_class:
        mock_store = Mock()
        mock_store.search.return_value = []
        mock_store_class.return_value = mock_store

        results = technique.retrieve("test query")

        assert results == []
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_naive_rag_technique.py::test_naive_rag_technique_retrieve_no_results -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_naive_rag_technique.py
git commit -m "test: add no results test for NaiveRAGTechnique.retrieve()"
```

---

## Task 14: Extend Configuration

**Files:**
- Modify: `config/techniques.yaml`
- Modify: `config/models.yaml`

**Step 1: Add collection_name to techniques.yaml**

Modify `config/techniques.yaml`:

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

**Step 2: Add vector_store section to models.yaml**

Modify `config/models.yaml`:

```yaml
ollama:
  base_url: "http://localhost:11434"

query_models:
  default: "glm-4.7:cloud"

embedding_models:
  default: "bge-m3:latest"
  available:
    - "nomic-embed-text-v2-moe:latest"
    - "qwen3-embedding:latest"
    - "granite-embedding:278m"
    - "bge-m3:latest"
    - "mxbai-embed-large:latest"

vector_store:
  persist_directory: "./data/chroma"
```

**Step 3: Commit**

```bash
git add config/techniques.yaml config/models.yaml
git commit -m "config: extend config for Phase 2

- Add collection_name to naive_rag technique
- Add vector_store persist_directory to models"
```

---

## Task 15: All Tests Pass

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests pass (Phase 1 + Phase 2 tests)

**Step 2: Count tests**

Expected: ~20 tests total (14 from Phase 1 + ~6 from Phase 2)

**Step 3: Commit final state**

```bash
git add .
git commit -m "chore: Phase 2 simple implementation complete

All tests passing:
- TextChunker for TXT paragraph chunking (3 tests)
- VectorStore with ChromaDB (3 tests)
- NaiveRAGTechnique with retrieve() (2 tests)
- OllamaClient.embed() tests (2 tests)

Total: ~20 tests passing"
```

---

## Task 16: Update Design Document Status

**Files:**
- Modify: `docs/plans/2026-02-05-phase2-simple-design.md`

**Step 1: Update status line**

Change:
```
**Status:** Approved for Implementation
```

To:
```
**Status:** Implemented
```

**Step 2: Add implementation notes**

Add at end of document:

```markdown
## Implementation Notes

**Completed:** 2026-02-05
**Branch:** feature/phase2-simple
**Tests:** All passing (~20 total)

Components implemented:
- TextChunker for TXT paragraph chunking
- VectorStore wrapping ChromaDB
- NaiveRAGTechnique implementing retrieve protocol
- OllamaClient.embed() for query embeddings
```

**Step 3: Commit**

```bash
git add docs/plans/2026-02-05-phase2-simple-design.md
git commit -m "docs: mark Phase 2 simple as implemented"
```

---

## Summary

This plan implements the retrieval layer of the RAG system with 16 tasks:

**New Components:**
- `utils/text_chunker.py` - TXT paragraph chunking
- `utils/vector_store.py` - ChromaDB wrapper
- `techniques/naive_rag.py` - Retrieval technique

**Modified Components:**
- `ollama/client.py` - Added embed() method
- `config/techniques.yaml` - Extended naive_rag config
- `config/models.yaml` - Added vector_store section
- `pyproject.toml` - Added chromadb dependency

**Test Coverage:**
- `tests/test_text_chunker.py` - 3 tests
- `tests/test_vector_store.py` - 3 tests
- `tests/test_naive_rag_technique.py` - 2 tests
- `tests/test_ollama.py` - 2 tests (extended)

**Total:** 10 new tests + 14 from Phase 1 = 24 tests

---

**For Execution:** Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.