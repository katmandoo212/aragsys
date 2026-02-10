# Scientific Agentic RAG Framework

A pluggable, extensible Retrieval-Augmented Generation framework for multi-hop reasoning over academic and clinical literature.

[![Tests](https://img.shields.io/badge/tests-144%20passing-brightgreen)](tests/)

## Overview

aragsys is a complete RAG framework designed for scientific literature analysis. It provides:

- **Document Ingestion** - Process PDF, Markdown, and TXT files with structure preservation
- **Advanced Retrieval** - Multiple strategies (Naive, HyDE, Multi-Query, Hybrid search)
- **Precision Enhancement** - Reranking and contextual compression
- **GraphRAG** - Multi-hop entity reasoning using Neo4j
- **LLM Generation** - Multiple strategies (Simple, Context-aware, Chain-of-Thought)

The framework uses YAML-driven configuration for flexible composition of RAG techniques.

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

## Ollama Setup

The framework uses Ollama for local LLM inference. Install Ollama from [ollama.com](https://ollama.com).

Pull required models:

```bash
# Generation models (default)
ollama pull glm-4.7:cloud
ollama pull llama3:8b
ollama pull llama3:70b

# Embedding models
ollama pull nomic-embed-text-v2-moe:latest
ollama pull qwen3-embedding:latest
ollama pull granite-embedding:278m
ollama pull bge-m3:latest
ollama pull mxbai-embed-large:latest
```

### Neo4j Setup (Optional)

For GraphRAG features, install Neo4j from [neo4j.com](https://neo4j.com).

Configure connection in `config/neo4j.yaml`:

```yaml
neo4j:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "your_password"
  database: "rag"
```

## Configuration

The framework uses YAML files for all configuration.

### models.yaml

Configure Ollama connection and available models:

```yaml
ollama:
  base_url: "http://localhost:11434"

# Query model for generation
query_models:
  default: "glm-4.7:cloud"

# Embedding models for vector search
embedding_models:
  default: "bge-m3:latest"
  available:
    - "nomic-embed-text-v2-moe:latest"
    - "qwen3-embedding:latest"
    - "granite-embedding:278m"
    - "bge-m3:latest"
    - "mxbai-embed-large:latest"

# Generation models for LLM responses
generation_models:
  default: "glm-4.7:cloud"
  available:
    - "glm-4.7:cloud"
    - "llama3:8b"
    - "llama3:70b"

# Vector store settings
vector_store:
  persist_directory: "./data/chroma"
  collection_name: "documents"
  dimension: 1024
```

### techniques.yaml

Define available RAG techniques and their configurations:

```yaml
techniques:
  # Basic dense retrieval
  naive_rag:
    class: techniques.naive_rag.NaiveRAGTechnique
    enabled: true
    config:
      chunk_size: 500
      top_k: 5
      embedding_model: "bge-m3:latest"
      collection_name: "documents"

  # Hypothetical Document Embeddings
  hyde:
    class: techniques.hyde.HyDETechnique
    enabled: true
    config:
      embedding_model: "bge-m3:latest"
      generation_model: "glm-4.7:cloud"
      top_k: 5

  # Multiple query variations
  multi_query:
    class: techniques.multi_query.MultiQueryTechnique
    enabled: true
    config:
      embedding_model: "bge-m3:latest"
      num_queries: 3

  # Hybrid search (vector + full-text)
  hybrid:
    class: techniques.hybrid.HybridTechnique
    enabled: true
    config:
      embedding_model: "bge-m3:latest"
      top_k: 5

  # Reranking with cross-encoder
  rerank:
    class: techniques.rerank.RerankTechnique
    enabled: true
    config:
      model: "bge-reranker-large:latest"
      score_threshold: 0.5

  # Contextual compression
  compress:
    class: techniques.compress.CompressTechnique
    enabled: true
    config:
      chunk_size: 500

  # GraphRAG: Entity-based retrieval
  graph_entity:
    class: techniques.graph_entity.GraphEntityTechnique
    enabled: true
    config:
      embedding_model: "bge-m3:latest"

  # GraphRAG: Multi-hop reasoning
  graph_multihop:
    class: techniques.graph_multihop.GraphMultiHopTechnique
    enabled: true
    config:
      embedding_model: "bge-m3:latest"
      max_depth: 2

  # GraphRAG: Relationship expansion
  graph_expand:
    class: techniques.graph_expand.GraphExpandTechnique
    enabled: true
    config:
      embedding_model: "bge-m3:latest"

  # Generation: Basic LLM generation
  simple_generation:
    class: techniques.generate_simple.SimpleGenerationTechnique
    enabled: true
    config:
      model: "glm-4.7:cloud"
      max_context_docs: 5

  # Generation: Citation-aware generation
  context_generation:
    class: techniques.generate_context.ContextGenerationTechnique
    enabled: true
    config:
      model: "glm-4.7:cloud"
      max_context_docs: 5

  # Generation: Chain-of-thought reasoning
  cot_generation:
    class: techniques.generate_cot.ChainOfThoughtGenerationTechnique
    enabled: true
    config:
      model: "glm-4.7:cloud"
      max_context_docs: 3
```

### pipelines.yaml

Define pipeline compositions:

```yaml
# Default query model
default_query_model: "glm-4.7:cloud"

pipelines:
  # Simple pipeline: naive retrieval + generation
  naive_flow:
    query_model: "glm-4.7:cloud"
    techniques: [naive_rag, simple_generation]

  # Advanced pipeline: HyDE + naive RAG + reranking + generation
  advanced_flow:
    query_model: "glm-4.7:cloud"
    techniques: [hyde, naive_rag, rerank, context_generation]

  # GraphRAG pipeline: entity extraction + multi-hop + generation
  graph_flow:
    query_model: "glm-4.7:cloud"
    techniques: [graph_entity, graph_multihop, cot_generation]

  # Full pipeline: all techniques
  full_flow:
    query_model: "glm-4.7:cloud"
    techniques: [hyde, naive_rag, multi_query, hybrid, rerank, compress, graph_entity, graph_multihop, context_generation]
```

### generation.yaml

Generation-specific settings:

```yaml
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

## Usage

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_registry.py -v

# Run with coverage
uv run pytest tests/ --cov=. --cov-report=html

# Run tests for specific technique
uv run pytest tests/test_generate_simple_technique.py -v
```

### Using the Technique Registry

The registry loads technique definitions from YAML and provides metadata:

```python
from registry.technique_registry import TechniqueRegistry

# Load technique definitions from YAML
registry = TechniqueRegistry("config/techniques.yaml")

# Get technique metadata
metadata = registry.get_technique("naive_rag")
print(metadata.name)       # "naive_rag"
print(metadata.enabled)    # True
print(metadata.get_config("chunk_size"))  # 500

# List all enabled techniques
enabled = registry.list_techniques()
for name, meta in enabled.items():
    print(f"{name}: {meta.class_path}")

# Check if technique exists
if registry.has_technique("hyde"):
    print("HyDE technique available")
```

### Building Pipelines

Use PipelineBuilder to compose pipelines from technique configurations:

```python
from registry.technique_registry import TechniqueRegistry
from pipeline.builder import PipelineBuilder
from nodes.technique_node import TechniqueNode

# Load registry
registry = TechniqueRegistry("config/techniques.yaml")
registry_dict = {name: meta for name, meta in registry._techniques.items()}

# Build pipeline nodes
builder = PipelineBuilder("config/pipelines.yaml", registry_dict)
nodes = builder.build_nodes("advanced_flow")

# Execute pipeline (requires PocketFlow)
from pocketflow import Flow

flow = Flow(name="rag_pipeline")
for node in nodes:
    flow.add_node(node)

# Run the flow
# flow.run({"query": "What is the capital of France?"})
```

### Using Ollama Client

The OllamaClient provides embedding generation and text generation:

```python
from ollama.client import OllamaClient

# Create client from config
client = OllamaClient.from_config("config/models.yaml")
print(client.base_url)  # "http://localhost:11434"

# Or create directly
client = OllamaClient("http://localhost:11434")

# Generate embeddings
vector = client.embed("Your text here", "bge-m3:latest")
print(f"Vector dimension: {len(vector)}")

# Generate text response
response = client.generate("What is the capital of France?", "glm-4.7:cloud")
print(response)  # "The capital of France is Paris."

# Batch embeddings for multiple texts
vectors = client.embed_batch(["Text 1", "Text 2"], "bge-m3:latest")
```

### Document Chunking

Process documents with format-aware chunking:

```python
from utils.text_chunker import TextChunker

chunker = TextChunker()

# Chunk any supported format (auto-detected from extension)
chunks = chunker.chunk_file("document.pdf", max_chunk_size=500)

# Each chunk is a (content, metadata) tuple
for content, metadata in chunks:
    print(f"Source: {metadata['source']}")
    print(f"Page: {metadata.get('page_number', 'N/A')}")
    print(f"Section: {metadata.get('section_title', 'N/A')}")
    print(f"Content preview: {content[:100]}...")
```

### Using NaiveRAG Technique

Basic retrieval using dense vector search:

```python
from techniques.naive_rag import NaiveRAGTechnique
from ollama.client import OllamaClient
from utils.vector_store import VectorStore

# Initialize components
client = OllamaClient.from_config("config/models.yaml")
store = VectorStore({"persist_directory": "./data/chroma", "collection_name": "documents"})

# Create technique
config = {
    "embedding_model": "bge-m3:latest",
    "top_k": 5,
    "collection_name": "documents"
}
technique = NaiveRAGTechnique(config, ollama_client=client, vector_store=store)

# Add documents to vector store
from utils.document import Document
docs = [
    Document(content="Paris is the capital of France.", metadata={"source": "doc1"}, score=0.9),
    Document(content="London is the capital of England.", metadata={"source": "doc2"}, score=0.8)
]
store.add_documents(docs)

# Retrieve documents
query = "What is the capital of France?"
documents = technique.retrieve(query)

for doc in documents:
    print(f"Score: {doc.score:.4f}, Content: {doc.content}")
```

### Using Generation Techniques

Generate answers using different strategies:

```python
from techniques.generate_simple import SimpleGenerationTechnique
from techniques.generate_context import ContextGenerationTechnique
from techniques.generate_cot import ChainOfThoughtGenerationTechnique
from ollama.client import OllamaClient
from utils.document import Document
from utils.answer import Answer

# Initialize
client = OllamaClient.from_config("config/models.yaml")

# 1. Simple Generation (basic LLM generation)
config = {"model": "glm-4.7:cloud", "max_context_docs": 5}
technique = SimpleGenerationTechnique(config, ollama_client=client)

query = "What is the capital of France?"
docs = [Document(content="Paris is France's capital city.", metadata={"source": "doc1"}, score=0.9)]

answer = technique.generate(query, docs)
print(answer.content)  # "The capital of France is Paris."
print(answer.citations)  # []

# 2. Context Generation (with citations)
config = {"model": "glm-4.7:cloud", "max_context_docs": 5}
technique = ContextGenerationTechnique(config, ollama_client=client)

answer = technique.generate(query, docs)
print(answer.content)    # "Paris is France's capital city [1]."
print(answer.citations)  # ["doc1"]

# 3. Chain-of-Thought Generation (reasoning)
config = {"model": "glm-4.7:cloud", "max_context_docs": 3}
technique = ChainOfThoughtGenerationTechnique(config, ollama_client=client)

answer = technique.generate(query, docs)
print(answer.content)    # Extracted answer without reasoning steps
```

### Using GraphRAG Techniques

Multi-hop reasoning using Neo4j graph storage:

```python
from techniques.graph_entity import GraphEntityTechnique
from techniques.graph_multihop import GraphMultiHopTechnique
from stores.neo4j_store import Neo4jStore
from ollama.client import OllamaClient

# Initialize Neo4j store
config = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "password",
    "database": "rag"
}
neo4j_store = Neo4jStore(config)

# Initialize technique
client = OllamaClient.from_config("config/models.yaml")
config = {"embedding_model": "bge-m3:latest"}
technique = GraphEntityTechnique(config, neo4j_store=neo4j_store)

# Find entities in query and retrieve related documents
query = "What is the relationship between Paris and France?"
documents = technique.retrieve(query)

for doc in documents:
    print(f"Entity: {doc.metadata.get('entity', 'N/A')}")
    print(f"Relationship: {doc.metadata.get('relationship', 'N/A')}")
    print(f"Content: {doc.content}")
```

### Full Example: Complete RAG Pipeline

```python
from registry.technique_registry import TechniqueRegistry
from pipeline.builder import PipelineBuilder
from ollama.client import OllamaClient
from utils.document import Document
from utils.answer import Answer

# 1. Load configuration
registry = TechniqueRegistry("config/techniques.yaml")
client = OllamaClient.from_config("config/models.yaml")

# 2. Build pipeline
registry_dict = {name: meta for name, meta in registry._techniques.items()}
builder = PipelineBuilder("config/pipelines.yaml", registry_dict)
nodes = builder.build_nodes("advanced_flow")

# 3. Prepare documents
docs = [
    Document(content="Paris is the capital of France.", metadata={"source": "doc1"}, score=0.9),
    Document(content="France is a country in Europe.", metadata={"source": "doc2"}, score=0.8)
]

# 4. Execute pipeline (simplified - typically via PocketFlow)
query = "What is the capital of France?"
results = []

for node in nodes:
    # Execute each technique in the pipeline
    if hasattr(node, 'technique'):
        technique = node.technique
        if hasattr(technique, 'retrieve'):
            docs = technique.retrieve(query)
        elif hasattr(technique, 'generate'):
            answer = technique.generate(query, docs)
            results.append(answer)

# 5. Output results
for i, answer in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"  Content: {answer.content}")
    print(f"  Citations: {answer.citations}")
```

## Supported Document Formats

| Format | Extension | Features |
|--------|-----------|----------|
| PDF | .pdf, .PDF | Structure preservation, tables, figures, page numbers |
| Markdown | .md, .markdown | Heading hierarchy (H1-H3), section paths |
| Text | .txt | Paragraph-based chunking |
| Unknown | Fallback | Plain text extraction (graceful degradation) |

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
│  - TechniqueNode wraps technique execution                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Execution Layer (PocketFlow)              │
│  - PipelineBuilder composes flows                           │
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

## Running the Web Frontend

The project includes a FastAPI-based web frontend for interactive RAG queries:

```bash
# Start the web server
uv run uvicorn backend.main:app --reload --port 8000

# Access the application
# Open http://localhost:8000 in your browser
```

**Web Frontend Features:**
- Query workspace with real-time progress (Server-Sent Events)
- Document management (web fetch + file upload)
- Monitoring dashboard with analytics
- Responsive UI built with Bootstrap 5 + HTMX

## Project Structure

```
aragsys/
├── backend/                          # FastAPI web application
│   ├── api/                         # API endpoints
│   │   ├── document.py              # Document management endpoints
│   │   ├── query.py                 # Query execution endpoints
│   │   └── monitoring.py            # Monitoring/analytics endpoints
│   ├── static/                      # Static assets
│   │   └── css/                     # Custom styles
│   ├── templates/                   # HTML templates
│   │   ├── index.html               # Main application page
│   │   └── components/              # Reusable HTMX components
│   ├── services/                    # Business logic services
│   │   ├── query_service.py         # Query orchestration service
│   │   ├── document_service.py      # Document processing service
│   │   └── monitoring_service.py    # Analytics service
│   ├── models/                      # Pydantic models
│   │   ├── document.py              # Document-related models
│   │   ├── query.py                 # Query-related models
│   │   └── monitoring.py            # Monitoring-related models
│   ├── main.py                      # FastAPI application entry point
│   └── app_config.py                # Application configuration
├── config/                           # Configuration files
│   ├── models.yaml                  # Ollama and model settings
│   ├── techniques.yaml              # Technique definitions
│   ├── pipelines.yaml               # Pipeline compositions
│   ├── generation.yaml              # Generation settings
│   ├── graphrag.yaml                # GraphRAG settings
│   ├── neo4j.yaml                   # Neo4j connection settings
│   └── entities.yaml                # Entity extraction settings
├── techniques/                       # RAG technique implementations
│   ├── naive_rag.py                 # Basic dense retrieval
│   ├── hyde.py                      # Hypothetical Document Embeddings
│   ├── multi_query.py               # Multiple query variations
│   ├── hybrid.py                    # Hybrid search
│   ├── rerank.py                    # Reranking with cross-encoder
│   ├── compress.py                  # Contextual compression
│   ├── graph_entity.py              # Entity-based retrieval
│   ├── graph_multihop.py            # Multi-hop reasoning
│   ├── graph_expand.py              # Relationship expansion
│   ├── generate_simple.py           # Basic LLM generation
│   ├── generate_context.py          # Citation-aware generation
│   └── generate_cot.py              # Chain-of-thought generation
├── stores/                           # Storage backends
│   └── neo4j_store.py               # Neo4j graph storage
├── registry/                         # Technique metadata registry
│   └── technique_registry.py        # TechniqueRegistry, TechniqueMetadata
├── nodes/                            # PocketFlow workflow nodes
│   └── technique_node.py            # Generic TechniqueNode
├── pipeline/                         # Pipeline composition
│   └── builder.py                   # PipelineBuilder
├── ollama/                           # Ollama API client
│   └── client.py                    # OllamaClient
├── utils/                            # Utilities
│   ├── document.py                  # Document dataclass
│   ├── answer.py                    # Answer dataclass
│   ├── vector_store.py              # ChromaDB vector store
│   ├── entity_extractor.py          # Named entity extraction
│   ├── pdf_chunker.py               # PDF processing with pdfplumber
│   ├── markdown_chunker.py          # Markdown section parsing
│   └── text_chunker.py              # Format-aware dispatcher
├── tests/                            # Test suite (144 tests)
│   ├── test_registry.py
│   ├── test_nodes.py
│   ├── test_builder.py
│   ├── test_ollama.py
│   ├── test_integration.py
│   ├── test_naive_rag_technique.py
│   ├── test_vector_store.py
│   ├── test_pdf_chunker.py
│   ├── test_markdown_chunker.py
│   ├── test_text_chunker.py
│   ├── test_neo4j_store.py
│   ├── test_entity_extractor.py
│   ├── test_graph_entity_technique.py
│   ├── test_graph_multihop_technique.py
│   ├── test_graph_expand_technique.py
│   ├── test_answer.py
│   ├── test_generate_simple_technique.py
│   ├── test_generate_context_technique.py
│   ├── test_generate_cot_technique.py
│   └── test_generation_config.py
├── docs/                             # Documentation
│   └── plans/                       # Design and implementation plans
├── main.py                           # CLI entry point
├── pyproject.toml                    # Dependencies
├── requirements.txt                  # Runtime dependencies
├── README.md                         # This file
└── CLAUDE.md                         # Development guidance
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

## Commands Reference

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_registry.py -v

# Run with coverage
uv run pytest tests/ --cov=. --cov-report=html

# Run specific test
uv run pytest tests/test_registry.py::TestTechniqueRegistry::test_technique_registry_loads -v

# Check code quality (if ruff is configured)
uv run ruff check .

# Type checking (if mypy is configured)
uv run mypy .
```
