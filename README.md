# Scientific Agentic RAG Framework

A pluggable, extensible Retrieval-Augmented Generation framework for multi-hop reasoning over academic and clinical literature.

## Overview

aragsys is designed as a three-layer architecture that enables flexible composition of RAG techniques through YAML-driven configuration. The framework separates concerns into:

1. **Configuration Layer** - YAML files define available techniques and pipeline compositions
2. **Registry Layer** - Loads YAML configs and provides technique metadata
3. **Execution Layer** - Composes pipelines and executes via PocketFlow

## Current Status

**Phase:** MVP Architecture Complete

The MVP establishes the foundational architecture with:
- YAML-driven technique and pipeline configuration
- Registry pattern for dynamic technique discovery
- Builder pattern for pipeline composition
- Ollama integration for LLM and embedding models
- Complete test coverage (14/14 tests passing)

**Next phases** will implement actual retrieval, generation, and multi-hop reasoning capabilities.

## Requirements

- Python 3.14+
- uv (for package management)
- Ollama (for local LLM inference)

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
# Query model (default)
ollama pull glm-4.7:cloud

# Embedding models
ollama pull nomic-embed-text-v2-moe:latest
ollama pull qwen3-embedding:latest
ollama pull granite-embedding:278m
ollama pull bge-m3:latest
ollama pull mxbai-embed-large:latest
```

## Configuration

### techniques.yaml

Define available RAG techniques and their configurations:

```yaml
techniques:
  naive_rag:
    enabled: true
    config:
      chunk_size: 500
      top_k: 5
      embedding_model: "bge-m3:latest"
  hyde:
    enabled: true
    config:
      query_expansions: 3
      embedding_model: "mxbai-embed-large:latest"
      query_model: "glm-4.7:cloud"
```

### pipelines.yaml

Define pipeline compositions:

```yaml
default_query_model: "glm-4.7:cloud"

pipelines:
  naive_flow:
    query_model: "glm-4.7:cloud"
    techniques: [naive_rag, generation]
  advanced_flow:
    query_model: "glm-4.7:cloud"
    techniques: [hyde, naive_rag, rerank, generation]
```

### models.yaml

Configure Ollama connection and available models:

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
```

### Using the Registry

```python
from registry.technique_registry import TechniqueRegistry

# Load technique definitions
registry = TechniqueRegistry("config/techniques.yaml")

# Get technique metadata
metadata = registry.get_technique("naive_rag")
print(metadata.name)  # "naive_rag"
print(metadata.enabled)  # True
print(metadata.get_config("chunk_size"))  # 500

# List available techniques
available = registry.list_techniques()
```

### Building Pipelines

```python
from registry.technique_registry import TechniqueRegistry
from pipeline.builder import PipelineBuilder

# Load registry and build pipeline
registry = TechniqueRegistry("config/techniques.yaml")
registry_dict = {name: meta for name, meta in registry._techniques.items()}

builder = PipelineBuilder("config/pipelines.yaml", registry_dict)
nodes = builder.build_nodes("naive_flow")

# Each node is a PocketFlow Node
for node in nodes:
    print(f"Node: {node.technique_name}, Config: {node.config}")
```

### Using Ollama Client

```python
from ollama.client import OllamaClient

# Create client from config
client = OllamaClient.from_config("config/models.yaml")
print(client.base_url)  # "http://localhost:11434"

# Or create directly
client = OllamaClient("http://localhost:11434")
```

## Project Structure

```
aragsys/
├── config/                    # Configuration files
│   ├── models.yaml           # Ollama and model settings
│   ├── techniques.yaml       # Technique definitions
│   └── pipelines.yaml        # Pipeline compositions
├── registry/                  # Technique metadata registry
│   └── technique_registry.py # TechniqueRegistry, TechniqueMetadata
├── nodes/                     # PocketFlow workflow nodes
│   └── technique_node.py     # Generic TechniqueNode
├── pipeline/                  # Pipeline composition
│   └── builder.py            # PipelineBuilder
├── ollama/                    # Ollama API client
│   └── client.py             # OllamaClient
├── tests/                     # Test suite
│   ├── test_registry.py      # Registry tests
│   ├── test_nodes.py         # Node tests
│   ├── test_builder.py       # Builder tests
│   ├── test_ollama.py        # Ollama client tests
│   └── test_integration.py   # Round-trip integration test
└── docs/                      # Documentation
    └── plans/                # Design and implementation plans
```

## Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────┐
│         Configuration Layer (YAML)              │
│  - techniques.yaml, pipelines.yaml, models.yaml │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│         Registry Layer                          │
│  - TechniqueRegistry loads and caches metadata  │
│  - Provides technique discovery and lookup      │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│         Execution Layer (PocketFlow)            │
│  - PipelineBuilder composes flows              │
│  - TechniqueNode wraps technique execution     │
└─────────────────────────────────────────────────┘
```

### Design Principles

- **YAGNI** - Only build what's needed for current phase
- **SOLID** - Single Responsibility, Dependency Injection, etc.
- **Extensibility** - Add techniques via YAML, not code changes
- **Testability** - All components tested in isolation

## Implementation Phases

1. **Phase 1 (MVP - Complete)**: Architecture foundation with mocked techniques
2. **Phase 2**: Naive RAG with dense retrieval
3. **Phase 3**: Advanced retrieval (HyDE, Multi-Query, Hybrid search)
4. **Phase 4**: Precision (Reranking, Contextual Compression)
5. **Phase 5**: GraphRAG with multi-hop reasoning
6. **Phase 6**: Agentic self-correction with iterative refinement

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`uv run pytest tests/ -v`)
5. Commit your changes
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [PocketFlow](https://the-pocket.github.io/PocketFlow/) for workflow orchestration
- Uses [Ollama](https://ollama.com/) for local LLM inference
- Inspired by modern RAG architecture patterns