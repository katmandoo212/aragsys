# Phase 7 - Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build LLM response generation capabilities with context-aware prompting, answer synthesis, and citation generation.

**Architecture:** Use existing OllamaClient.generate() to build generation techniques that take retrieved documents as context and produce answers. Create Answer dataclass to store responses with metadata. Follow the same pattern as retrieval techniques (dataclass, config injection, duck typing).

**Tech Stack:** Python 3.13+, Ollama (already installed), pytest (already installed), existing OllamaClient, existing technique patterns.

---

## Prerequisites

**Before starting:**

```bash
# Ensure we're on master branch
git checkout master
git pull origin master

# Create feature branch
git checkout -b feature/phase7-generation

# Install dependencies (should already be installed)
uv sync
```

---

## Task 1: Answer dataclass

**Files:**
- Create: `utils/answer.py`
- Test: `tests/test_answer.py`

**Step 1: Write the failing test**

```python
# tests/test_answer.py (new file)

"""Tests for Answer dataclass."""

from utils.answer import Answer
from dataclasses import asdict


class TestAnswer:
    """Test Answer dataclass for generation responses."""

    def test_answer_creation(self):
        """Answer can be created with content and metadata."""
        answer = Answer(
            content="Paris is the capital of France.",
            metadata={"source": "test", "model": "llama3"},
            citations=["doc1", "doc2"]
        )
        assert answer.content == "Paris is the capital of France."
        assert answer.metadata["source"] == "test"
        assert answer.citations == ["doc1", "doc2"]

    def test_answer_with_empty_metadata(self):
        """Answer can be created with minimal metadata."""
        answer = Answer(content="Test answer")
        assert answer.content == "Test answer"
        assert answer.metadata == {}
        assert answer.citations == []

    def test_answer_to_dict(self):
        """Answer can be converted to dictionary."""
        answer = Answer(
            content="Test",
            metadata={"key": "value"},
            citations=["doc1"]
        )
        result = asdict(answer)
        assert result["content"] == "Test"
        assert result["metadata"] == {"key": "value"}
        assert result["citations"] == ["doc1"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_answer.py -v`
Expected: FAIL with "No module named 'utils.answer'"

**Step 3: Write minimal implementation**

```python
# utils/answer.py (new file)

"""Answer dataclass for generation responses."""

from dataclasses import dataclass, field


@dataclass
class Answer:
    """Response from LLM generation."""

    content: str
    metadata: dict = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_answer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add utils/answer.py tests/test_answer.py
git commit -m "feat: add Answer dataclass

- Add content, metadata, citations fields
- Support default empty metadata and citations
- Add tests for Answer creation and conversion"
```

---

## Task 2: OllamaClient - Add generation model config

**Files:**
- Modify: `ollama/client.py`
- Modify: `config/models.yaml`
- Test: `tests/test_ollama.py`

**Step 1: Write the failing test**

Add this to tests/test_ollama.py:

```python
def test_ollama_client_has_generation_models():
    """OllamaClient config includes generation models."""
    from ollama.client import OllamaClient

    client = OllamaClient.from_config("config/models.yaml")
    assert hasattr(client, 'generation_model')
    assert client.generation_model is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ollama.py::test_ollama_client_has_generation_models -v`
Expected: FAIL with "AttributeError: 'OllamaClient' object has no attribute 'generation_model'"

**Step 3: Write minimal implementation**

First, update `config/models.yaml` to add generation model config:

```yaml
# Add after embedding_models section
generation_models:
  default: "glm-4.7:cloud"
  available:
    - "glm-4.7:cloud"
    - "llama3:8b"
    - "llama3:70b"
```

Then modify `ollama/client.py`:

```python
# ollama/client.py (modify existing file)

@dataclass
class OllamaClient:
    base_url: str
    generation_model: str = None

    @classmethod
    def from_config(cls, config_path: str) -> "OllamaClient":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(
            base_url=config["ollama"]["base_url"],
            generation_model=config["generation_models"]["default"]
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ollama.py::test_ollama_client_has_generation_models -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ollama/client.py config/models.yaml tests/test_ollama.py
git commit -m "feat: add generation model config to OllamaClient

- Add generation_models section to models.yaml
- Add generation_model attribute to OllamaClient
- Set default to glm-4.7:cloud"
```

---

## Task 3: SimpleGenerationTechnique

**Files:**
- Create: `techniques/generate_simple.py`
- Test: `tests/test_generate_simple_technique.py`

**Step 1: Write the failing test**

```python
# tests/test_generate_simple_technique.py (new file)

"""Tests for SimpleGenerationTechnique."""

from unittest.mock import MagicMock, patch
from techniques.generate_simple import SimpleGenerationTechnique
from utils.answer import Answer


class TestSimpleGenerationTechnique:
    """Test SimpleGenerationTechnique for basic LLM generation."""

    def test_initialization(self):
        """SimpleGenerationTechnique initializes with config."""
        config = {"model": "llama3:8b", "max_context_docs": 5}
        technique = SimpleGenerationTechnique(config, ollama_client=MagicMock())
        assert technique.model == "llama3:8b"
        assert technique.max_context_docs == 5

    def test_generate_empty_query(self):
        """Empty query returns empty answer."""
        config = {"model": "llama3:8b"}
        technique = SimpleGenerationTechnique(config, ollama_client=MagicMock())
        result = technique.generate("", [])
        assert isinstance(result, Answer)
        assert result.content == ""

    def test_generate_calls_ollama(self):
        """Generate calls OllamaClient.generate with prompt."""
        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = "Paris is France's capital."

        technique = SimpleGenerationTechnique(config, ollama_client=mock_client)
        result = technique.generate("What is France's capital?", [])

        assert isinstance(result, Answer)
        assert result.content == "Paris is France's capital."
        mock_client.generate.assert_called_once()

    def test_generate_includes_context(self):
        """Generate includes document context in prompt."""
        from utils.document import Document

        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = "Answer"

        docs = [
            Document(content="Paris is a city.", metadata={"source": "doc1"}, score=0.9),
            Document(content="France is a country.", metadata={"source": "doc2"}, score=0.8)
        ]

        technique = SimpleGenerationTechnique(config, ollama_client=mock_client)
        technique.generate("Query", docs)

        # Verify generate was called
        assert mock_client.generate.called
        call_args = mock_client.generate.call_args
        prompt = call_args[0][0]
        assert "Paris is a city" in prompt
        assert "France is a country" in prompt
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate_simple_technique.py -v`
Expected: FAIL with "No module named 'techniques.generate_simple'"

**Step 3: Write minimal implementation**

```python
# techniques/generate_simple.py (new file)

"""Simple LLM generation technique."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ollama.client import OllamaClient
    from utils.answer import Answer
    from utils.document import Document


@dataclass
class SimpleGenerationTechnique:
    """Generate answers using basic LLM prompting."""

    def __init__(self, config: dict, ollama_client=None):
        self.config = config
        self.model = config.get("model", "glm-4.7:cloud")
        self.max_context_docs = config.get("max_context_docs", 5)
        self.ollama_client = ollama_client

    def generate(self, query: str, documents: list["Document"]) -> "Answer":
        """Generate an answer based on query and retrieved documents."""
        if not query or not self.ollama_client:
            from utils.answer import Answer
            return Answer(content="")

        # Build prompt with context
        context = self._build_context(documents[:self.max_context_docs])
        prompt = f"""Query: {query}

Context:
{context}

Answer:"""

        # Generate response
        response = self.ollama_client.generate(prompt, self.model)

        from utils.answer import Answer
        return Answer(content=response.strip())

    def _build_context(self, documents: list["Document"]) -> str:
        """Build context string from documents."""
        if not documents:
            return "No context available."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", f"doc_{i}")
            context_parts.append(f"[{i}] {source}: {doc.content}")

        return "\n\n".join(context_parts)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate_simple_technique.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add techniques/generate_simple.py tests/test_generate_simple_technique.py
git commit -m "feat: add SimpleGenerationTechnique

- Basic LLM generation with document context
- Support configurable model and max_context_docs
- Return Answer object with content"
```

---

## Task 4: ContextGenerationTechnique with citations

**Files:**
- Create: `techniques/generate_context.py`
- Test: `tests/test_generate_context_technique.py`

**Step 1: Write the failing test**

```python
# tests/test_generate_context_technique.py (new file)

"""Tests for ContextGenerationTechnique."""

from unittest.mock import MagicMock
from techniques.generate_context import ContextGenerationTechnique
from utils.answer import Answer
from utils.document import Document


class TestContextGenerationTechnique:
    """Test ContextGenerationTechnique for citation-aware generation."""

    def test_initialization(self):
        """ContextGenerationTechnique initializes with config."""
        config = {"model": "llama3:8b", "max_context_docs": 3}
        technique = ContextGenerationTechnique(config, ollama_client=MagicMock())
        assert technique.model == "llama3:8b"
        assert technique.max_context_docs == 3

    def test_generate_returns_citations(self):
        """Generate includes citations from used documents."""
        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = "Paris is the capital [1]."

        docs = [
            Document(content="Paris is France's capital.", metadata={"source": "doc1"}, score=0.9)
        ]

        technique = ContextGenerationTechnique(config, ollama_client=mock_client)
        result = technique.generate("Query", docs)

        assert isinstance(result, Answer)
        assert "doc1" in result.citations

    def test_generate_formats_context_with_sources(self):
        """Context includes source labels for citation."""
        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = "Answer"

        docs = [
            Document(content="Fact A", metadata={"source": "source1.txt"}, score=0.9),
            Document(content="Fact B", metadata={"source": "source2.pdf"}, score=0.8)
        ]

        technique = ContextGenerationTechnique(config, ollama_client=mock_client)
        technique.generate("Query", docs)

        call_args = mock_client.generate.call_args
        prompt = call_args[0][0]
        assert "[source1.txt]" in prompt
        assert "[source2.pdf]" in prompt
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate_context_technique.py -v`
Expected: FAIL with "No module named 'techniques.generate_context'"

**Step 3: Write minimal implementation**

```python
# techniques/generate_context.py (new file)

"""Context-aware LLM generation with citations."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ollama.client import OllamaClient
    from utils.answer import Answer
    from utils.document import Document


@dataclass
class ContextGenerationTechnique:
    """Generate answers with citation formatting."""

    def __init__(self, config: dict, ollama_client=None):
        self.config = config
        self.model = config.get("model", "glm-4.7:cloud")
        self.max_context_docs = config.get("max_context_docs", 5)
        self.ollama_client = ollama_client

    def generate(self, query: str, documents: list["Document"]) -> "Answer":
        """Generate an answer with citations from context."""
        if not query or not self.ollama_client:
            from utils.answer import Answer
            return Answer(content="")

        # Build context with citation markers
        context, citations = self._build_context_with_citations(documents[:self.max_context_docs])

        # Build prompt
        prompt = f"""Query: {query}

Use the following context to answer. Cite your sources using [n] notation where n is the reference number.

Context:
{context}

Answer:"""

        # Generate response
        response = self.ollama_client.generate(prompt, self.model)

        from utils.answer import Answer
        return Answer(content=response.strip(), citations=citations)

    def _build_context_with_citations(self, documents: list["Document"]) -> tuple[str, list[str]]:
        """Build context with citation markers and return citations list."""
        if not documents:
            return "No context available.", []

        context_parts = []
        citations = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", f"doc_{i}")
            citations.append(source)
            context_parts.append(f"[{i}] {source}: {doc.content}")

        return "\n\n".join(context_parts), citations
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate_context_technique.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add techniques/generate_context.py tests/test_generate_context_technique.py
git commit -m "feat: add ContextGenerationTechnique

- Context-aware generation with citation markers
- Return Answer object with citations list
- Format context with [n] citation notation"
```

---

## Task 5: ChainOfThoughtGenerationTechnique

**Files:**
- Create: `techniques/generate_cot.py`
- Test: `tests/test_generate_cot_technique.py`

**Step 1: Write the failing test**

```python
# tests/test_generate_cot_technique.py (new file)

"""Tests for ChainOfThoughtGenerationTechnique."""

from unittest.mock import MagicMock
from techniques.generate_cot import ChainOfThoughtGenerationTechnique
from utils.answer import Answer


class TestChainOfThoughtGenerationTechnique:
    """Test ChainOfThoughtGenerationTechnique for reasoning-based generation."""

    def test_initialization(self):
        """ChainOfThoughtGenerationTechnique initializes with config."""
        config = {"model": "llama3:8b", "max_context_docs": 3}
        technique = ChainOfThoughtGenerationTechnique(config, ollama_client=MagicMock())
        assert technique.model == "llama3:8b"
        assert technique.max_context_docs == 3

    def test_generate_includes_reasoning_instructions(self):
        """Generate includes step-by-step reasoning instructions."""
        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = "Reasoning:\n1. First step.\n2. Second step.\n\nAnswer: Result."

        technique = ChainOfThoughtGenerationTechnique(config, ollama_client=mock_client)
        technique.generate("Query", [])

        call_args = mock_client.generate.call_args
        prompt = call_args[0][0]
        assert "step by step" in prompt.lower()
        assert "reasoning" in prompt.lower()

    def test_generate_returns_answer_without_reasoning(self):
        """Generate strips reasoning from final answer."""
        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = """Reasoning:
Let me think about this.
The answer is Paris.

Answer: Paris"""

        technique = ChainOfThoughtGenerationTechnique(config, ollama_client=mock_client)
        result = technique.generate("Query", [])

        # Answer should be extracted from "Answer:" section
        assert "Paris" in result.content
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate_cot_technique.py -v`
Expected: FAIL with "No module named 'techniques.generate_cot'"

**Step 3: Write minimal implementation**

```python
# techniques/generate_cot.py (new file)

"""Chain-of-thought LLM generation technique."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ollama.client import OllamaClient
    from utils.answer import Answer
    from utils.document import Document


@dataclass
class ChainOfThoughtGenerationTechnique:
    """Generate answers using chain-of-thought reasoning."""

    def __init__(self, config: dict, ollama_client=None):
        self.config = config
        self.model = config.get("model", "glm-4.7:cloud")
        self.max_context_docs = config.get("max_context_docs", 5)
        self.ollama_client = ollama_client

    def generate(self, query: str, documents: list["Document"]) -> "Answer":
        """Generate an answer using step-by-step reasoning."""
        if not query or not self.ollama_client:
            from utils.answer import Answer
            return Answer(content="")

        # Build context
        context = self._build_context(documents[:self.max_context_docs])

        # Build CoT prompt
        prompt = f"""Query: {query}

Context:
{context}

Think step by step to answer the query. Show your reasoning process, then provide the final answer.

Reasoning:"""

        # Generate response
        response = self.ollama_client.generate(prompt, self.model)

        # Extract answer from CoT format
        answer = self._extract_answer(response)

        from utils.answer import Answer
        return Answer(content=answer.strip())

    def _build_context(self, documents: list["Document"]) -> str:
        """Build context string from documents."""
        if not documents:
            return "No context available."

        return "\n\n".join([f"- {doc.content}" for doc in documents])

    def _extract_answer(self, response: str) -> str:
        """Extract final answer from CoT response."""
        # Look for "Answer:" or "Final Answer:" section
        if "Answer:" in response:
            return response.split("Answer:")[-1]
        elif "Final Answer:" in response:
            return response.split("Final Answer:")[-1]
        return response
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate_cot_technique.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add techniques/generate_cot.py tests/test_generate_cot_technique.py
git commit -m "feat: add ChainOfThoughtGenerationTechnique

- Chain-of-thought reasoning with step-by-step instructions
- Extract final answer from reasoning process
- Support configurable model and max_context_docs"
```

---

## Task 6: Update techniques __init__.py exports

**Files:**
- Modify: `techniques/__init__.py`
- Test: `tests/test_registry.py`

**Step 1: Write the failing test**

Add this to tests/test_registry.py:

```python
def test_registry_imports_generation_techniques():
    """Verify generation techniques are importable."""
    from techniques import SimpleGenerationTechnique, ContextGenerationTechnique, ChainOfThoughtGenerationTechnique
    assert SimpleGenerationTechnique is not None
    assert ContextGenerationTechnique is not None
    assert ChainOfThoughtGenerationTechnique is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py::test_registry_imports_generation_techniques -v`
Expected: FAIL with "cannot import name 'SimpleGenerationTechnique'"

**Step 3: Write minimal implementation**

Modify `techniques/__init__.py`:

```python
# techniques/__init__.py (modify existing file)

"""RAG retrieval techniques."""

from techniques.naive_rag import NaiveRAGTechnique
from techniques.hyde import HyDETechnique
from techniques.multi_query import MultiQueryTechnique
from techniques.hybrid import HybridTechnique
from techniques.rerank import RerankTechnique
from techniques.compress import CompressTechnique
from techniques.graph_entity import GraphEntityTechnique
from techniques.graph_multihop import GraphMultiHopTechnique
from techniques.graph_expand import GraphExpandTechnique
from techniques.generate_simple import SimpleGenerationTechnique
from techniques.generate_context import ContextGenerationTechnique
from techniques.generate_cot import ChainOfThoughtGenerationTechnique

__all__ = [
    "NaiveRAGTechnique",
    "HyDETechnique",
    "MultiQueryTechnique",
    "HybridTechnique",
    "RerankTechnique",
    "CompressTechnique",
    "GraphEntityTechnique",
    "GraphMultiHopTechnique",
    "GraphExpandTechnique",
    "SimpleGenerationTechnique",
    "ContextGenerationTechnique",
    "ChainOfThoughtGenerationTechnique",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py::test_registry_imports_generation_techniques -v`
Expected: PASS

**Step 5: Commit**

```bash
git add techniques/__init__.py tests/test_registry.py
git commit -m "feat: export Phase 7 generation techniques

- Add SimpleGenerationTechnique to exports
- Add ContextGenerationTechnique to exports
- Add ChainOfThoughtGenerationTechnique to exports
- Add test verifying imports work"
```

---

## Task 7: Update config/techniques.yaml

**Files:**
- Modify: `config/techniques.yaml`

**Step 1: Add generation techniques to config**

Add to `config/techniques.yaml`:

```yaml
  simple_generation:
    class: techniques.generate_simple.SimpleGenerationTechnique
    enabled: true
    config:
      model: "glm-4.7:cloud"
      max_context_docs: 5

  context_generation:
    class: techniques.generate_context.ContextGenerationTechnique
    enabled: true
    config:
      model: "glm-4.7:cloud"
      max_context_docs: 5

  cot_generation:
    class: techniques.generate_cot.ChainOfThoughtGenerationTechnique
    enabled: true
    config:
      model: "glm-4.7:cloud"
      max_context_docs: 3
```

**Step 2: Verify config loads**

Run: `python -c "import yaml; config = yaml.safe_load(open('config/techniques.yaml')); print(list(config['techniques'].keys()))"`
Expected: Shows simple_generation, context_generation, cot_generation keys

**Step 3: Commit**

```bash
git add config/techniques.yaml
git commit -m "feat: add Phase 7 generation techniques to config

- Add simple_generation for basic LLM generation
- Add context_generation for citation-aware generation
- Add cot_generation for chain-of-thought reasoning"
```

---

## Task 8: Create generation.yaml config

**Files:**
- Create: `config/generation.yaml`
- Test: `tests/test_generation_config.py`

**Step 1: Write the failing test**

```python
# tests/test_generation_config.py (new file)

"""Tests for generation configuration."""

import pytest
import yaml


def test_generation_config_exists():
    """Generation config file exists and is valid YAML."""
    with open("config/generation.yaml") as f:
        config = yaml.safe_load(f)
    assert "generation" in config
    assert "models" in config


def test_generation_config_has_defaults():
    """Config has sensible default values."""
    with open("config/generation.yaml") as f:
        config = yaml.safe_load(f)
    assert config["generation"]["default_model"] is not None
    assert config["generation"]["max_context_docs"] > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generation_config.py -v`
Expected: FAIL with "No such file or directory: 'config/generation.yaml'"

**Step 3: Write minimal implementation**

```yaml
# config/generation.yaml (new file)

generation:
  # Default model for generation
  default_model: "glm-4.7:cloud"

  # Maximum number of documents to include as context
  max_context_docs: 5

  # Generation settings
  temperature: 0.7
  max_tokens: 512

  # Prompt templates
  prompts:
    simple: |
      Query: {query}

      Context:
      {context}

      Answer:

    context: |
      Query: {query}

      Use the following context to answer. Cite your sources using [n] notation.

      Context:
      {context}

      Answer:

    cot: |
      Query: {query}

      Context:
      {context}

      Think step by step to answer the query. Show your reasoning process.

      Reasoning:

models:
  # Available generation models
  available:
    - name: "glm-4.7:cloud"
      max_context: 10000
      supports_citations: true
    - name: "llama3:8b"
      max_context: 8000
      supports_citations: false
    - name: "llama3:70b"
      max_context: 12000
      supports_citations: true
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generation_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add config/generation.yaml tests/test_generation_config.py
git commit -m "feat: add generation.yaml configuration

- Define default model and generation settings
- Add prompt templates for different strategies
- Add model specifications with context limits"
```

---

## Task 9: All tests pass

**Step 1: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: All tests pass (including Phase 7 tests)

**Step 2: Fix any failing tests if needed**

(If tests fail, debug and fix, then re-run)

**Step 3: Commit final state**

```bash
git add .
git commit -m "chore: Phase 7 Generation implementation complete

All tests passing:
- Answer dataclass for generation responses
- OllamaClient generation model config
- SimpleGenerationTechnique for basic LLM generation
- ContextGenerationTechnique with citations
- ChainOfThoughtGenerationTechnique for reasoning
- Configuration files updated
- Total tests: ~115 (99 Phase 1-6 + ~16 Phase 7)"
```

---

## Task 10: Update CLAUDE.md status

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update project status**

Change:
```
**Phase:** Phase 6 Complete - GraphRAG (2026-02-07)
```

To:
```
**Phase:** Phase 7 Complete - Generation (2026-02-07)
```

**Step 2: Add Phase 7 to Implementation Phases section**

Add after Phase 6 description:
```markdown
**Phase 7 adds:**
- Answer dataclass for generation responses (content, metadata, citations)
- OllamaClient generation model configuration
- SimpleGenerationTechnique: Basic LLM generation with document context
- ContextGenerationTechnique: Citation-aware generation with source markers
- ChainOfThoughtGenerationTechnique: Step-by-step reasoning with answer extraction
- **~115 tests passing** (99 Phase 1-6 + ~16 Phase 7)
```

**Step 3: Update Supported formats section**

Change:
```
**Supported formats:** .txt, .pdf, .md, .markdown with graceful fallback for unknown types
```

To:
```
**Supported formats:** .txt, .pdf, .md, .markdown with graceful fallback for unknown types

**Generation models:** glm-4.7:cloud, llama3:8b, llama3:70b (via Ollama)
```

**Step 4: Remove Next phases section**

Remove the "Next phases:" line since this is the final phase.

**Step 5: Update Tests section**

Add Phase 7 test files to the list:
```python
- `tests/test_answer.py` - Answer dataclass tests (3 tests)
- `tests/test_generate_simple_technique.py` - SimpleGenerationTechnique tests (4 tests)
- `tests/test_generate_context_technique.py` - ContextGenerationTechnique tests (3 tests)
- `tests/test_generate_cot_technique.py` - ChainOfThoughtGenerationTechnique tests (3 tests)
- `tests/test_generation_config.py` - Generation config tests (2 tests)
```

Update total test count:
```python
**Total: ~115 tests, all passing**
```

**Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update project status to Phase 7 complete

- Mark Phase 7 as complete (final phase)
- Document Phase 7 features (3 generation techniques, Answer dataclass)
- Update supported formats with generation models
- Update test count to ~115 passing"
```

---

## Summary

This plan implements Phase 7 (Generation) - the final phase of the Scientific Agentic RAG Framework:

**Core Components:**
- **Answer dataclass** - Response object with content, metadata, citations
- **OllamaClient enhancement** - Generation model configuration

**Generation Techniques (3):**
- **SimpleGenerationTechnique** - Basic LLM generation with document context
- **ContextGenerationTechnique** - Citation-aware generation with source markers
- **ChainOfThoughtGenerationTechnique** - Step-by-step reasoning with answer extraction

**Configuration:**
- `config/techniques.yaml` - Add generation technique definitions
- `config/generation.yaml` - Generation-specific configuration with prompts and models
- `config/models.yaml` - Add generation_models section

**Files created:**
- `utils/answer.py`
- `techniques/generate_simple.py`
- `techniques/generate_context.py`
- `techniques/generate_cot.py`
- `config/generation.yaml`
- Tests in 5 new test files

**After Phase 7:**
The Scientific Agentic RAG Framework is complete with full RAG pipeline:
- Document ingestion (PDF, Markdown, TXT)
- Advanced retrieval (HyDE, Multi-Query, Hybrid)
- Precision (Reranking, Compression)
- GraphRAG (Multi-hop reasoning with Neo4j)
- **Generation (LLM response with multiple strategies)**