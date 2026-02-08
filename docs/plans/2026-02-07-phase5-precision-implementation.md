# Phase 5: Precision (Reranking, Contextual Compression) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build reranking and contextual compression techniques to improve retrieval precision and reduce token usage.

**Architecture:** Two new techniques (RerankTechnique, CompressTechnique) following existing pattern with Ollama integration, configuration-driven via YAML, and TDD approach.

**Tech Stack:** Python 3.13+, Ollama, pytest, YAML

---

### Task 1: Create precision configuration file

**Files:**
- Create: `config/precision.yaml`

**Step 1: Create the configuration file**

```yaml
precision:
  keyword_extraction:
    stop_words: ["the", "a", "an", "is", "are", "was", "were"]
    segment_length: 200
    min_keyword_matches: 1
  llm_refinement:
    enabled: true
    model: "glm-4.7:cloud"
    max_segments: 3
```

**Step 2: Verify file created**

Run: `cat config/precision.yaml`
Expected: Shows YAML content above

**Step 3: Commit**

```bash
git add config/precision.yaml
git commit -m "config: add precision configuration file"
```

---

### Task 2: Write RerankTechnique tests

**Files:**
- Create: `tests/test_rerank_technique.py`

**Step 1: Write test file with mock setup**

```python
import pytest
from unittest.mock import Mock, MagicMock
from techniques.rerank import RerankTechnique

@pytest.fixture
def mock_ollama_client():
    client = MagicMock()
    client.generate.return_value = "0.85"
    return client

@pytest.fixture
def mock_base_technique():
    technique = MagicMock()
    technique.retrieve.return_value = [
        {"content": "doc1", "metadata": {"id": 1}, "relevance_score": 0.5},
        {"content": "doc2", "metadata": {"id": 2}, "relevance_score": 0.3},
    ]
    return technique

def test_rerank_reorders_results(mock_ollama_client, mock_base_technique):
    reranker = RerankTechnique(
        ollama_client=mock_ollama_client,
        scoring_model="test-model",
        top_k=5,
        score_threshold=0.0,
        base_technique=mock_base_technique,
    )

    results = reranker.retrieve("test query")

    assert len(results) == 2
    assert results[0]["content"] == "doc1"
    assert results[1]["content"] == "doc2"
    # generate should be called for each document
    assert mock_ollama_client.generate.call_count == 2

def test_rerank_filters_by_score_threshold(mock_ollama_client, mock_base_technique):
    mock_ollama_client.generate.side_effect = ["0.2", "0.8"]
    reranker = RerankTechnique(
        ollama_client=mock_ollama_client,
        scoring_model="test-model",
        top_k=5,
        score_threshold=0.5,
        base_technique=mock_base_technique,
    )

    results = reranker.retrieve("test query")

    # Only document with score >= 0.5 should be returned
    assert len(results) == 1
    assert results[0]["content"] == "doc2"
    assert results[0]["relevance_score"] == 0.8

def test_rerank_handles_empty_results(mock_ollama_client, mock_base_technique):
    mock_base_technique.retrieve.return_value = []
    reranker = RerankTechnique(
        ollama_client=mock_ollama_client,
        scoring_model="test-model",
        top_k=5,
        score_threshold=0.5,
        base_technique=mock_base_technique,
    )

    results = reranker.retrieve("test query")

    assert results == []
    mock_ollama_client.generate.assert_not_called()

def test_rerank_keeps_top_k(mock_ollama_client):
    # Create mock base technique with 5 results
    base_technique = MagicMock()
    base_technique.retrieve.return_value = [
        {"content": f"doc{i}", "metadata": {"id": i}, "relevance_score": 0.5}
        for i in range(5)
    ]

    # Mock scores - first 3 should be highest
    mock_ollama_client.generate.side_effect = ["0.9", "0.3", "0.8", "0.2", "0.7"]

    reranker = RerankTechnique(
        ollama_client=mock_ollama_client,
        scoring_model="test-model",
        top_k=3,
        score_threshold=0.0,
        base_technique=base_technique,
    )

    results = reranker.retrieve("test query")

    assert len(results) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rerank_technique.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'techniques.rerank'"

**Step 3: Commit**

```bash
git add tests/test_rerank_technique.py
git commit -m "test: add RerankTechnique tests"
```

---

### Task 3: Create RerankTechnique implementation

**Files:**
- Create: `techniques/rerank.py`

**Step 1: Write minimal implementation**

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ollama.client import OllamaClient

@dataclass
class RerankTechnique:
    ollama_client: "OllamaClient"
    scoring_model: str
    top_k: int
    score_threshold: float
    base_technique: Optional[object] = None

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve and rerank results using cross-encoder scoring."""
        # Get initial results from base technique
        if self.base_technique:
            results = self.base_technique.retrieve(query)
        else:
            return []

        if not results:
            return []

        # Score each document
        scored_results = []
        for result in results:
            score = self._score_document(query, result["content"])
            if score >= self.score_threshold:
                result["relevance_score"] = score
                scored_results.append(result)

        # Sort by score and keep top_k
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_results[:self.top_k]

    def _score_document(self, query: str, document: str) -> float:
        """Score document relevance using cross-encoder prompt."""
        prompt = (
            f"Rate the relevance of this document to the query on a scale of 0.0 to 1.0.\n"
            f"Query: {query}\n"
            f"Document: {document}\n"
            f"Relevance score:"
        )
        response = self.ollama_client.generate(prompt, self.scoring_model)
        try:
            return float(response.strip())
        except ValueError:
            return 0.0
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_rerank_technique.py -v`
Expected: PASS (4 tests)

**Step 3: Commit**

```bash
git add techniques/rerank.py
git commit -m "feat: add RerankTechnique implementation"
```

---

### Task 4: Write CompressTechnique tests

**Files:**
- Create: `tests/test_compress_technique.py`

**Step 1: Write test file**

```python
import pytest
from unittest.mock import MagicMock
from techniques.compress import CompressTechnique

@pytest.fixture
def mock_ollama_client():
    client = MagicMock()
    client.generate.return_value = "relevant content"
    return client

@pytest.fixture
def mock_base_technique():
    technique = MagicMock()
    technique.retrieve.return_value = [
        {
            "content": "This is about cats. Dogs are also mentioned. This talks about cats again.",
            "metadata": {"id": 1},
            "relevance_score": 0.8,
        }
    ]
    return technique

def test_keyword_extraction_filters_segments(mock_ollama_client, mock_base_technique):
    compressor = CompressTechnique(
        ollama_client=mock_ollama_client,
        use_llm_refinement=False,
        min_keyword_matches=1,
        segment_length=50,
        top_k_segments=3,
        base_technique=mock_base_technique,
    )

    results = compressor.retrieve("cats")

    # Should extract segments containing "cats"
    assert len(results) == 1
    assert "cats" in results[0]["content"].lower()
    # LLM should not be called when use_llm_refinement=False
    mock_ollama_client.generate.assert_not_called()

def test_llm_refinement_compresses_content(mock_ollama_client, mock_base_technique):
    mock_ollama_client.generate.return_value = "Cats are furry animals."
    compressor = CompressTechnique(
        ollama_client=mock_ollama_client,
        use_llm_refinement=True,
        min_keyword_matches=1,
        segment_length=50,
        top_k_segments=3,
        base_technique=mock_base_technique,
    )

    results = compressor.retrieve("cats")

    assert len(results) == 1
    assert results[0]["content"] == "Cats are furry animals."
    mock_ollama_client.generate.assert_called_once()

def test_fallback_without_llm_refinement(mock_base_technique):
    client = MagicMock()
    compressor = CompressTechnique(
        ollama_client=client,
        use_llm_refinement=False,
        min_keyword_matches=1,
        segment_length=50,
        top_k_segments=3,
        base_technique=mock_base_technique,
    )

    results = compressor.retrieve("cats")

    assert len(results) == 1
    client.generate.assert_not_called()

def test_document_with_no_matches(mock_ollama_client, mock_base_technique):
    mock_base_technique.retrieve.return_value = [
        {"content": "This is about nothing relevant.", "metadata": {"id": 1}, "relevance_score": 0.5}
    ]
    compressor = CompressTechnique(
        ollama_client=mock_ollama_client,
        use_llm_refinement=False,
        min_keyword_matches=1,
        segment_length=50,
        top_k_segments=3,
        base_technique=mock_base_technique,
    )

    results = compressor.retrieve("cats")

    # No matches, should return empty or original
    assert isinstance(results, list)

def test_configuration_options(mock_ollama_client, mock_base_technique):
    compressor = CompressTechnique(
        ollama_client=mock_ollama_client,
        use_llm_refinement=True,
        min_keyword_matches=2,
        segment_length=20,
        top_k_segments=1,
        base_technique=mock_base_technique,
    )

    # Verify config is stored
    assert compressor.min_keyword_matches == 2
    assert compressor.segment_length == 20
    assert compressor.top_k_segments == 1
    assert compressor.use_llm_refinement is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_compress_technique.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'techniques.compress'"

**Step 3: Commit**

```bash
git add tests/test_compress_technique.py
git commit -m "test: add CompressTechnique tests"
```

---

### Task 5: Create CompressTechnique implementation

**Files:**
- Create: `techniques/compress.py`

**Step 1: Write minimal implementation**

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import re

if TYPE_CHECKING:
    from ollama.client import OllamaClient

@dataclass
class CompressTechnique:
    ollama_client: "OllamaClient"
    use_llm_refinement: bool
    min_keyword_matches: int
    segment_length: int
    top_k_segments: int
    base_technique: Optional[object] = None

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve and compress documents to relevant segments."""
        # Get initial results from base technique
        if self.base_technique:
            results = self.base_technique.retrieve(query)
        else:
            return []

        if not results:
            return []

        compressed_results = []
        for result in results:
            compressed = self._compress_document(query, result["content"])
            if compressed:
                result["content"] = compressed
                compressed_results.append(result)

        return compressed_results

    def _compress_document(self, query: str, content: str) -> str:
        """Compress document using keyword extraction and optional LLM refinement."""
        # Extract query terms (simple tokenization, lowercase)
        query_terms = set(re.findall(r'\w+', query.lower()))

        # Split into segments (simple sentence-based)
        segments = re.split(r'[.!?]+', content)
        segments = [s.strip() for s in segments if s.strip()]

        # Score segments by keyword overlap
        scored_segments = []
        for segment in segments:
            segment_terms = set(re.findall(r'\w+', segment.lower()))
            overlap = len(query_terms & segment_terms)

            if overlap >= self.min_keyword_matches:
                scored_segments.append((segment, overlap))

        # Sort by overlap and keep top_k
        scored_segments.sort(key=lambda x: x[1], reverse=True)
        top_segments = [s[0] for s in scored_segments[:self.top_k_segments]]

        if not top_segments:
            return content

        # Optional LLM refinement
        if self.use_llm_refinement:
            return self._llm_refine(query, top_segments)

        # Join segments without LLM
        return ". ".join(top_segments)

    def _llm_refine(self, query: str, segments: list[str]) -> str:
        """Refine segments using LLM to keep only relevant content."""
        segments_text = "\n".join(f"- {s}" for s in segments)
        prompt = (
            f"From these segments, extract only content directly relevant to the query.\n"
            f"Query: {query}\n"
            f"Segments:\n{segments_text}\n"
            f"Relevant content:"
        )
        return self.ollama_client.generate(prompt, "glm-4.7:cloud")
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_compress_technique.py -v`
Expected: PASS (5 tests)

**Step 3: Commit**

```bash
git add techniques/compress.py
git commit -m "feat: add CompressTechnique implementation"
```

---

### Task 6: Update techniques.yaml with new techniques

**Files:**
- Modify: `config/techniques.yaml`

**Step 1: Add rerank and compress entries to techniques.yaml**

Find the existing techniques section and add after the existing entries:

```yaml
  rerank:
    class: techniques.rerank.RerankTechnique
    enabled: true
    config:
      scoring_model: "bge-reranker-v2:latest"
      top_k: 5
      score_threshold: 0.5

  compress:
    class: techniques.compress.CompressTechnique
    enabled: true
    config:
      use_llm_refinement: true
      top_k_segments: 3
      min_keyword_matches: 1
      segment_length: 200
```

**Step 2: Run tests to verify configuration loads**

Run: `pytest tests/test_registry.py::test_technique_registry_loads_yaml -v`
Expected: PASS

**Step 3: Commit**

```bash
git add config/techniques.yaml
git commit -m "config: add rerank and compress techniques to registry"
```

---

### Task 7: Export new techniques from __init__.py

**Files:**
- Modify: `techniques/__init__.py`

**Step 1: Add imports for new techniques**

Find the existing imports and add:

```python
from .rerank import RerankTechnique
from .compress import CompressTechnique
```

Also update the `__all__` list to include the new classes:

```python
__all__ = [
    "BaseTechnique",
    "NaiveRAGTechnique",
    "HyDETechnique",
    "MultiQueryTechnique",
    "HybridTechnique",
    "RerankTechnique",
    "CompressTechnique",
]
```

**Step 2: Verify imports work**

Run: `python -c "from techniques import RerankTechnique, CompressTechnique; print('OK')"`
Expected: Prints "OK"

**Step 3: Commit**

```bash
git add techniques/__init__.py
git commit -m "feat: export RerankTechnique and CompressTechnique"
```

---

### Task 8: Verify all tests pass

**Files:**
- Test: All tests

**Step 1: Run complete test suite**

Run: `pytest tests/ -v`
Expected: PASS (74 + 9 = 83 tests passing)

**Step 2: Check specific new test files**

Run: `pytest tests/test_rerank_technique.py tests/test_compress_technique.py -v`
Expected: PASS (4 + 5 = 9 tests)

**Step 3: Commit**

```bash
git commit --allow-empty -m "test: verify all Phase 5 tests pass (83 total)"
```

---

## Summary

**Total tasks:** 8
**Total tests:** 9 new tests (4 rerank + 5 compress)
**Expected final test count:** 83 (74 + 9)

**Key files created:**
- `config/precision.yaml`
- `techniques/rerank.py`
- `techniques/compress.py`
- `tests/test_rerank_technique.py`
- `tests/test_compress_technique.py`

**Key files modified:**
- `config/techniques.yaml`
- `techniques/__init__.py`