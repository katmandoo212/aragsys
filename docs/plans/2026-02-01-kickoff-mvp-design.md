# Kickoff MVP Design: Architecture Foundation

**Date:** 2026-02-01
**Status:** Implemented

## Overview

The MVP establishes a three-layer architecture for pluggable RAG techniques, focusing on testable and extensible structure rather than functional retrieval capabilities.

### Architecture Layers

1. **Configuration Layer**: YAML files define available techniques and pipeline compositions
2. **Registry Layer**: Loads YAML configs and provides technique metadata
3. **Execution Layer**: Composes pipelines and executes via PocketFlow

### Success Criteria

Complete round-trip demonstration: config → registry → pipeline → execution, all verified with mocked techniques.

---

## Technique Registry

The `TechniqueRegistry` class reads technique definitions from YAML without importing actual classes.

### Configuration Loading

- Reads `config/techniques.yaml` on initialization
- YAML structure maps technique names to their metadata (name, enabled status, config values)
- Example:
  ```yaml
  naive_rag:
    enabled: true
    config:
      chunk_size: 500
      top_k: 5
  ```

### Technique Resolution

- Returns a `TechniqueMetadata` object containing technique name and config
- For MVP, this is a data object - execution is mocked in tests
- Serves as configuration lookup, not class factory

### API

```python
registry = TechniqueRegistry("config/techniques.yaml")
metadata = registry.get_technique("naive_rag")  # returns TechniqueMetadata
available = registry.list_techniques()  # returns list of enabled technique names
```

### Error Handling

- `TechniqueNotFoundError`: Requested technique doesn't exist in config
- Disabled techniques are skipped when listing

---

## Pipeline Builder

The `PipelineBuilder` composes PocketFlow flows from registered technique metadata.

### Builder Pattern

- Takes `TechniqueRegistry` and pipeline configuration as input
- `build_pipeline(name)` creates a Flow from `config/pipelines.yaml`
- Pipeline YAML defines technique sequence:
  ```yaml
  naive_flow:
    techniques: [naive_rag, generation]
  ```

### Flow Composition

- Creates PocketFlow `Flow` with nodes for each technique
- Each node is a generic `TechniqueNode` wrapping technique execution
- Nodes connected sequentially based on pipeline definition
- Shared state dictionary flows between nodes

### API

```python
builder = PipelineBuilder(registry, "config/pipelines.yaml")
flow = builder.build_pipeline("naive_flow")
result = flow.run({"query": "test query"})
```

### TechniqueNode

```python
# prep: extract inputs from shared state
def prep(self, shared):
    return shared["query"], shared.get("retrieved_docs", [])

# exec: call technique (mocked in MVP)
def exec(self, prep_result):
    query, docs = prep_result
    return technique.execute(query, docs)

# post: store output, return next node
def post(self, shared, prep_res, exec_res):
    shared["retrieved_docs"] = exec_res
    return "next_node_name"
```

---

## Shared State

The shared state dictionary is the communication channel between nodes.

### Standard Keys

- `query`: User's original question (input to first node)
- `retrieved_docs`: Documents from vector store (output of retrieval nodes)
- `context`: Compressed/processed context (output of compression nodes)
- `answer`: Final generated response (output of generation node)
- `metadata`: Ancillary data (iteration count, timing, debug info)

### Flow Pattern

1. **Retrieval nodes**: read `query`, write `retrieved_docs`
2. **Reranking nodes**: read `query` + `retrieved_docs`, write `retrieved_docs` (filtered)
3. **Compression nodes**: read `retrieved_docs`, write `context`
4. **Generation nodes**: read `query` + `context`, write `answer`

---

## Directory Structure

```
aragsys/
├── config/
│   ├── techniques.yaml      # Technique definitions
│   └── pipelines.yaml       # Pipeline compositions
├── registry/
│   ├── __init__.py
│   └── technique_registry.py   # TechniqueRegistry class
├── nodes/
│   ├── __init__.py
│   └── technique_node.py      # Generic TechniqueNode
├── pipeline/
│   ├── __init__.py
│   └── builder.py            # PipelineBuilder class
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Pytest fixtures
│   ├── test_registry.py      # Registry tests with mocks
│   ├── test_builder.py       # Builder tests with mocks
│   └── test_integration.py   # Round-trip demo
├── pyproject.toml
└── README.md
```

---

## YAML Configuration

### config/techniques.yaml

```yaml
techniques:
  naive_rag:
    enabled: true
    config:
      chunk_size: 500
      top_k: 5
  hyde:
    enabled: true
    config:
      query_expansions: 3
  disabled_technique:
    enabled: false
```

### config/pipelines.yaml

```yaml
pipelines:
  naive_flow:
    techniques: [naive_rag, generation]
  advanced_flow:
    techniques: [hyde, naive_rag, rerank, generation]
```

---

## Testing Strategy

### Philosophy

- No real technique implementations - all mocked
- Tests verify architecture works, not that techniques work
- Each component tested in isolation, then integrated

### Test Structure

1. **test_registry.py**: Verify YAML loads, metadata accessible, disabled techniques skipped
2. **test_builder.py**: Verify pipeline creates correct nodes in correct order
3. **test_integration.py**: Full round-trip demo

### Round-Trip Demo

```python
def test_complete_round_trip(tmp_path):
    # 1. Create mock config files
    # 2. Load config into Registry
    # 3. Build pipeline from Builder
    # 4. Mock technique execution
    # 5. Run flow
    # 6. Verify: config loaded → registry found → pipeline built → flow executed
```

### Mock Technique

```python
class MockTechnique:
    def __init__(self, metadata):
        self.metadata = metadata

    def execute(self, query, **kwargs):
        return f"Mock result for: {query}"
```

---

## Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| No dynamic import | Techniques mocked for MVP - just need metadata |
| YAML-driven config | Users change behavior without code changes |
| Generic TechniqueNode | Single implementation works for all techniques |
| Shared state dict | Simple, flexible communication between nodes |
| Disabled flag | Techniques can be temporarily disabled without removal |

---

## Implementation Notes

**Completed:** 2026-02-01
**Branch:** feature/kickoff-mvp
**Tests:** All passing

Components implemented:
- TechniqueRegistry with YAML loading
- TechniqueMetadata dataclass
- PipelineBuilder with node creation
- TechniqueNode with prep/exec/post
- OllamaClient with config loading
- Integration test for round-trip verification

---

## Implementation Notes

**Completed:** 2026-02-01
**Branch:** feature/kickoff-mvp
**Tests:** All passing

Components implemented:
- TechniqueRegistry with YAML loading
- TechniqueMetadata dataclass
- PipelineBuilder with node creation
- TechniqueNode with prep/exec/post
- OllamaClient with config loading
- Integration test for round-trip verification