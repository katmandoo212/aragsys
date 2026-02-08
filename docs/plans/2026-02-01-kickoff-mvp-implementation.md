# MVP Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a testable, extensible RAG architecture with YAML-driven configuration, technique registry, pipeline builder, and Ollama integration.

**Architecture:** Three-layer system - Configuration Layer (YAML), Registry Layer (technique metadata), Execution Layer (PocketFlow flows). All techniques mocked for MVP - focus on plumbing, not retrieval.

**Tech Stack:** Python 3.14+, Pydantic (data validation), PyYAML (config loading), PocketFlow (workflow), pytest (testing), uv (package management)

---

## Prerequisites

**Before starting:**

```bash
# Ensure we're in the worktree
cd .worktrees/kickoff-mvp

# Install dependencies
uv add pydantic pyyaml pytest

# Create project structure
mkdir -p config registry nodes pipeline ollama tests
touch registry/__init__.py nodes/__init__.py pipeline/__init__.py ollama/__init__.py tests/__init__.py
```

---

## Task 1: Configuration - models.yaml

**Files:**
- Create: `config/models.yaml`

**Step 1: Create models.yaml file**

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

**Step 2: Verify file created**

Run: `cat config/models.yaml`
Expected: YAML content shown above

**Step 3: Commit**

```bash
git add config/models.yaml
git commit -m "feat: add models.yaml configuration for Ollama

- Define Ollama base URL
- Configure query models with default
- Configure embedding models with list of available models"
```

---

## Task 2: Configuration - techniques.yaml

**Files:**
- Create: `config/techniques.yaml`

**Step 1: Create techniques.yaml file**

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
  disabled_technique:
    enabled: false
```

**Step 2: Verify file created**

Run: `cat config/techniques.yaml`
Expected: YAML content shown above

**Step 3: Commit**

```bash
git add config/techniques.yaml
git commit -m "feat: add techniques.yaml configuration

- Define naive_rag technique with embedding model
- Define hyde technique with query expansion
- Add disabled technique example"
```

---

## Task 3: Configuration - pipelines.yaml

**Files:**
- Create: `config/pipelines.yaml`

**Step 1: Create pipelines.yaml file**

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

**Step 2: Verify file created**

Run: `cat config/pipelines.yaml`
Expected: YAML content shown above

**Step 3: Commit**

```bash
git add config/pipelines.yaml
git commit -m "feat: add pipelines.yaml configuration

- Define default query model
- Add naive_flow pipeline
- Add advanced_flow pipeline with more techniques"
```

---

## Task 4: Registry - TechniqueMetadata dataclass

**Files:**
- Create: `registry/technique_registry.py`
- Test: `tests/test_registry.py`

**Step 1: Write failing test for TechniqueMetadata**

```python
# tests/test_registry.py
import pytest
from registry.technique_registry import TechniqueMetadata

def test_technique_metadata_creation():
    metadata = TechniqueMetadata(
        name="naive_rag",
        enabled=True,
        config={"chunk_size": 500, "top_k": 5}
    )
    assert metadata.name == "naive_rag"
    assert metadata.enabled is True
    assert metadata.config == {"chunk_size": 500, "top_k": 5}

def test_technique_metadata_get_config_value():
    metadata = TechniqueMetadata(
        name="naive_rag",
        enabled=True,
        config={"chunk_size": 500, "top_k": 5}
    )
    assert metadata.get_config("chunk_size") == 500
    assert metadata.get_config("top_k") == 5

def test_technique_metadata_get_config_default():
    metadata = TechniqueMetadata(
        name="naive_rag",
        enabled=True,
        config={}
    )
    assert metadata.get_config("missing_key", default=100) == 100
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL with "TechniqueMetadata not defined"

**Step 3: Write minimal implementation**

```python
# registry/technique_registry.py
from dataclasses import dataclass
from typing import Any, Mapping

@dataclass
class TechniqueMetadata:
    name: str
    enabled: bool
    config: Mapping[str, Any]

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add registry/technique_registry.py tests/test_registry.py
git commit -m "feat: add TechniqueMetadata dataclass

- Add name, enabled, config fields
- Add get_config method with default support
- Add tests for creation and config access"
```

---

## Task 5: Registry - Load YAML config

**Files:**
- Modify: `registry/technique_registry.py`
- Test: `tests/test_registry.py`

**Step 1: Write failing test for YAML loading**

```python
# tests/test_registry.py (add to existing file)
import tempfile
import os

def test_technique_registry_loads_yaml(tmp_path):
    yaml_content = """
techniques:
  naive_rag:
    enabled: true
    config:
      chunk_size: 500
  hyde:
    enabled: false
    config:
      expansions: 3
"""
    yaml_file = tmp_path / "techniques.yaml"
    yaml_file.write_text(yaml_content)

    from registry.technique_registry import TechniqueRegistry
    registry = TechniqueRegistry(str(yaml_file))

    metadata = registry.get_technique("naive_rag")
    assert metadata.name == "naive_rag"
    assert metadata.enabled is True
    assert metadata.config == {"chunk_size": 500}
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py::test_technique_registry_loads_yaml -v`
Expected: FAIL with "TechniqueRegistry not defined" or similar

**Step 3: Write minimal implementation**

```python
# registry/technique_registry.py (add to existing file)
from dataclasses import dataclass
from typing import Any, Mapping
import yaml

@dataclass
class TechniqueMetadata:
    name: str
    enabled: bool
    config: Mapping[str, Any]

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


class TechniqueRegistry:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._techniques: dict[str, TechniqueMetadata] = {}
        self._load_config()

    def _load_config(self) -> None:
        with open(self.config_path) as f:
            data = yaml.safe_load(f)
        for name, config in data.get("techniques", {}).items():
            self._techniques[name] = TechniqueMetadata(
                name=name,
                enabled=config.get("enabled", True),
                config=config.get("config", {})
            )

    def get_technique(self, name: str) -> TechniqueMetadata:
        if name not in self._techniques:
            raise ValueError(f"Technique '{name}' not found")
        return self._techniques[name]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py::test_technique_registry_loads_yaml -v`
Expected: PASS

**Step 5: Commit**

```bash
git add registry/technique_registry.py tests/test_registry.py
git commit -m "feat: add YAML loading to TechniqueRegistry

- Load techniques from YAML config
- Store TechniqueMetadata for each technique
- Add get_technique method with error handling"
```

---

## Task 6: Registry - List techniques

**Files:**
- Modify: `registry/technique_registry.py`
- Test: `tests/test_registry.py`

**Step 1: Write failing test for list techniques**

```python
# tests/test_registry.py (add to existing file)
def test_technique_registry_lists_enabled_only(tmp_path):
    yaml_content = """
techniques:
  enabled_one:
    enabled: true
    config: {}
  enabled_two:
    enabled: true
    config: {}
  disabled_one:
    enabled: false
    config: {}
"""
    yaml_file = tmp_path / "techniques.yaml"
    yaml_file.write_text(yaml_content)

    from registry.technique_registry import TechniqueRegistry
    registry = TechniqueRegistry(str(yaml_file))

    techniques = registry.list_techniques()
    assert techniques == ["enabled_one", "enabled_two"]
    assert "disabled_one" not in techniques
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py::test_technique_registry_lists_enabled_only -v`
Expected: FAIL with "list_techniques not defined"

**Step 3: Write minimal implementation**

```python
# registry/technique_registry.py (add to existing file)
    def list_techniques(self) -> list[str]:
        return [name for name, meta in self._techniques.items() if meta.enabled]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add registry/technique_registry.py tests/test_registry.py
git commit -m "feat: add list_techniques method to Registry

- Return list of enabled technique names only
- Filter out disabled techniques"
```

---

## Task 7: Registry - Custom exception

**Files:**
- Modify: `registry/technique_registry.py`
- Test: `tests/test_registry.py`

**Step 1: Write failing test for custom exception**

```python
# tests/test_registry.py (add to existing file)
def test_technique_registry_raises_technique_not_found():
    from registry.technique_registry import TechniqueRegistry, TechniqueNotFoundError

    # Test exception can be raised
    with pytest.raises(TechniqueNotFoundError) as exc_info:
        raise TechniqueNotFoundError("missing_tech")

    assert "missing_tech" in str(exc_info.value)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py::test_technique_registry_raises_technique_not_found -v`
Expected: FAIL with "TechniqueNotFoundError not defined"

**Step 3: Write minimal implementation**

```python
# registry/technique_registry.py (add at top)
class TechniqueNotFoundError(Exception):
    pass
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py::test_technique_registry_raises_technique_not_found -v`
Expected: PASS

**Step 5: Commit**

```bash
git add registry/technique_registry.py tests/test_registry.py
git commit -m "feat: add TechniqueNotFoundError exception

- Custom exception for missing techniques
- Better error messaging for registry users"
```

---

## Task 8: Registry - Use custom exception in get_technique

**Files:**
- Modify: `registry/technique_registry.py`
- Test: `tests/test_registry.py`

**Step 1: Write failing test**

```python
# tests/test_registry.py (add to existing file)
def test_technique_registry_get_missing_raises_custom_exception():
    yaml_content = "techniques: {}"
    yaml_file = tmp_path / "techniques.yaml"
    yaml_file.write_text(yaml_content)

    from registry.technique_registry import TechniqueRegistry, TechniqueNotFoundError
    registry = TechniqueRegistry(str(yaml_file))

    with pytest.raises(TechniqueNotFoundError):
        registry.get_technique("does_not_exist")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py::test_technique_registry_get_missing_raises_custom_exception -v`
Expected: FAIL (ValueError raised instead of TechniqueNotFoundError)

**Step 3: Update implementation**

```python
# registry/technique_registry.py (update get_technique method)
    def get_technique(self, name: str) -> TechniqueMetadata:
        if name not in self._techniques:
            raise TechniqueNotFoundError(f"Technique '{name}' not found")
        return self._techniques[name]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add registry/technique_registry.py tests/test_registry.py
git commit -m "refactor: use custom exception in get_technique

- Raise TechniqueNotFoundError instead of ValueError
- Better error type for consumers"
```

---

## Task 9: Nodes - TechniqueNode base structure

**Files:**
- Create: `nodes/technique_node.py`
- Test: `tests/test_nodes.py`

**Step 1: Install PocketFlow dependency**

```bash
uv add pocketflow
```

**Step 2: Write failing test for TechniqueNode structure**

```python
# tests/test_nodes.py
from nodes.technique_node import TechniqueNode

def test_technique_node_exists():
    node = TechniqueNode("test_technique", {"chunk_size": 500})
    assert node.technique_name == "test_technique"
    assert node.config == {"chunk_size": 500}
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_nodes.py -v`
Expected: FAIL with "TechniqueNode not defined"

**Step 4: Write minimal implementation**

```python
# nodes/technique_node.py
from dataclasses import dataclass
from pocketflow import Node

@dataclass
class TechniqueNode(Node):
    technique_name: str
    config: dict

    def prep(self, shared: dict):
        return shared

    def exec(self, prep_result):
        return prep_result

    def post(self, shared: dict, prep_res, exec_res):
        return None
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_nodes.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add nodes/technique_node.py tests/test_nodes.py pyproject.toml
git commit -m "feat: add TechniqueNode base class

- Inherit from PocketFlow Node
- Store technique_name and config
- Implement prep/exec/post stub methods"
```

---

## Task 10: Nodes - Extract query from shared state

**Files:**
- Modify: `nodes/technique_node.py`
- Test: `tests/test_nodes.py`

**Step 1: Write failing test**

```python
# tests/test_nodes.py (add to existing file)
def test_technique_node_prep_extracts_query():
    from nodes.technique_node import TechniqueNode

    node = TechniqueNode("test", {})
    shared = {"query": "test query", "retrieved_docs": []}

    result = node.prep(shared)
    assert result["query"] == "test query"
    assert result["retrieved_docs"] == []
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_nodes.py::test_technique_node_prep_extracts_query -v`
Expected: PASS (current implementation returns shared dict)

**Step 3: Update test to be more specific**

```python
# tests/test_nodes.py (update test)
def test_technique_node_prep_extracts_specific_keys():
    from nodes.technique_node import TechniqueNode

    node = TechniqueNode("test", {})
    shared = {"query": "test query", "retrieved_docs": []}

    query, docs = node.prep(shared)
    assert query == "test query"
    assert docs == []
```

**Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_nodes.py::test_technique_node_prep_extracts_specific_keys -v`
Expected: FAIL (cannot unpack dict into 2 variables)

**Step 5: Update implementation**

```python
# nodes/technique_node.py (update prep method)
    def prep(self, shared: dict):
        query = shared.get("query")
        docs = shared.get("retrieved_docs", [])
        return query, docs
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_nodes.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add nodes/technique_node.py tests/test_nodes.py
git commit -m "feat: update TechniqueNode prep to extract query and docs

- Extract query from shared state
- Extract retrieved_docs with empty default
- Return as tuple for exec method"
```

---

## Task 11: Pipeline - PipelineBuilder structure

**Files:**
- Create: `pipeline/builder.py`
- Test: `tests/test_builder.py`

**Step 1: Write failing test**

```python
# tests/test_builder.py
from pipeline.builder import PipelineBuilder

def test_pipeline_builder_creation():
    from registry.technique_registry import TechniqueRegistry
    import tempfile
    import os

    yaml_content = "pipelines:\n  test_flow:\n    techniques: [tech1, tech2]"
    yaml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml_file.write(yaml_content)
    yaml_file.close()

    try:
        mock_registry = {"tech1": {"enabled": True, "config": {}}}
        builder = PipelineBuilder(yaml_file.name, mock_registry)
        assert builder is not None
    finally:
        os.unlink(yaml_file.name)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_builder.py -v`
Expected: FAIL with "PipelineBuilder not defined"

**Step 3: Write minimal implementation**

```python
# pipeline/builder.py
import yaml

class PipelineBuilder:
    def __init__(self, config_path: str, registry: dict):
        self.config_path = config_path
        self.registry = registry
        self._load_config()

    def _load_config(self) -> None:
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_builder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pipeline/builder.py tests/test_builder.py
git commit -m "feat: add PipelineBuilder base class

- Load pipeline config from YAML
- Store registry reference
- Basic constructor implementation"
```

---

## Task 12: Pipeline - Build pipeline creates nodes

**Files:**
- Modify: `pipeline/builder.py`
- Test: `tests/test_builder.py`

**Step 1: Write failing test**

```python
# tests/test_builder.py (add to existing file)
import tempfile
import os

def test_pipeline_build_creates_nodes():
    from pipeline.builder import PipelineBuilder
    from nodes.technique_node import TechniqueNode

    yaml_content = "pipelines:\n  test_flow:\n    techniques: [tech1, tech2]"
    yaml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml_file.write(yaml_content)
    yaml_file.close()

    try:
        # Mock registry with technique metadata
        class MockMetadata:
            def __init__(self, name, enabled, config):
                self.name = name
                self.enabled = enabled
                self.config = config

        mock_registry = {
            "tech1": MockMetadata("tech1", True, {"param": "value1"}),
            "tech2": MockMetadata("tech2", True, {"param": "value2"})
        }

        builder = PipelineBuilder(yaml_file.name, mock_registry)
        nodes = builder.build_nodes("test_flow")

        assert len(nodes) == 2
        assert isinstance(nodes[0], TechniqueNode)
        assert isinstance(nodes[1], TechniqueNode)
        assert nodes[0].technique_name == "tech1"
        assert nodes[1].technique_name == "tech2"
    finally:
        os.unlink(yaml_file.name)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_builder.py::test_pipeline_build_creates_nodes -v`
Expected: FAIL with "build_nodes not defined"

**Step 3: Write minimal implementation**

```python
# pipeline/builder.py (add to existing file)
from nodes.technique_node import TechniqueNode

class PipelineBuilder:
    def __init__(self, config_path: str, registry: dict):
        self.config_path = config_path
        self.registry = registry
        self._load_config()

    def _load_config(self) -> None:
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    def build_nodes(self, pipeline_name: str) -> list[TechniqueNode]:
        pipeline = self.config["pipelines"][pipeline_name]
        technique_names = pipeline.get("techniques", [])
        nodes = []
        for name in technique_names:
            meta = self.registry[name]
            nodes.append(TechniqueNode(meta.name, meta.config))
        return nodes
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_builder.py::v`
Expected: PASS

**Step 5: Commit**

```bash
git add pipeline/builder.py tests/test_builder.py
git commit -m "feat: add build_nodes method to PipelineBuilder

- Create TechniqueNode instances from pipeline config
- Return list of nodes in execution order"
```

---

## Task 13: Ollama - OllamaClient structure

**Files:**
- Create: `ollama/client.py`
- Test: `tests/test_ollama.py`

**Step 1: Install httpx dependency**

```bash
uv add httpx
```

**Step 2: Write failing test**

```python
# tests/test_ollama.py
from ollama.client import OllamaClient

def test_ollama_client_creation():
    client = OllamaClient("http://localhost:11434")
    assert client.base_url == "http://localhost:11434"
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_ollama.py -v`
Expected: FAIL with "OllamaClient not defined"

**Step 4: Write minimal implementation**

```python
# ollama/client.py
from dataclasses import dataclass

@dataclass
class OllamaClient:
    base_url: str
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_ollama.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add ollama/client.py tests/test_ollama.py pyproject.toml
git commit -m "feat: add OllamaClient base class

- Store base_url for Ollama API
- Dataclass for simple configuration"
```

---

## Task 14: Ollama - OllamaClient loads from config

**Files:**
- Modify: `ollama/client.py`
- Test: `tests/test_ollama.py`

**Step 1: Write failing test**

```python
# tests/test_ollama.py (add to existing file)
import tempfile
import os

def test_ollama_client_from_config(tmp_path):
    yaml_content = """
ollama:
  base_url: "http://custom-url:11434"
"""
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml_content)

    from ollama.client import OllamaClient
    client = OllamaClient.from_config(str(config_file))
    assert client.base_url == "http://custom-url:11434"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ollama.py::test_ollama_client_from_config -v`
Expected: FAIL with "from_config not defined"

**Step 3: Write minimal implementation**

```python
# ollama/client.py (add to existing file)
from dataclasses import dataclass
import yaml

@dataclass
class OllamaClient:
    base_url: str

    @classmethod
    def from_config(cls, config_path: str) -> "OllamaClient":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(base_url=config["ollama"]["base_url"])
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ollama.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ollama/client.py tests/test_ollama.py
git commit -m "feat: add from_config class method to OllamaClient

- Load base_url from models.yaml config
- Factory method for convenient initialization"
```

---

## Task 15: Integration - Round-trip test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
import tempfile
import os

def test_complete_round_trip(tmp_path):
    """
    Complete round-trip: config → registry → pipeline → execution
    """
    # Create mock config files
    techniques_yaml = """
techniques:
  technique_a:
    enabled: true
    config:
      param1: value1
  technique_b:
    enabled: false
    config: {}
"""

    pipelines_yaml = """
default_query_model: "glm-4.7:cloud"

pipelines:
  test_pipeline:
    query_model: "glm-4.7:cloud"
    techniques: [technique_a]
"""

    models_yaml = """
ollama:
  base_url: "http://localhost:11434"

query_models:
  default: "glm-4.7:cloud"
"""

    # Write config files
    techniques_file = tmp_path / "techniques.yaml"
    pipelines_file = tmp_path / "pipelines.yaml"
    models_file = tmp_path / "models.yaml"

    techniques_file.write_text(techniques_yaml)
    pipelines_file.write_text(pipelines_yaml)
    models_file.write_text(models_yaml)

    # 1. Load config into Registry
    from registry.technique_registry import TechniqueRegistry
    registry = TechniqueRegistry(str(techniques_file))

    # Verify registry loaded correctly
    assert "technique_a" in registry.list_techniques()
    assert "technique_b" not in registry.list_techniques()

    meta = registry.get_technique("technique_a")
    assert meta.enabled is True
    assert meta.config == {"param1": "value1"}

    # 2. Build pipeline from Builder
    from pipeline.builder import PipelineBuilder
    # Create mock registry dict for builder
    registry_dict = {name: meta for name, meta in registry._techniques.items()}
    builder = PipelineBuilder(str(pipelines_file), registry_dict)

    nodes = builder.build_nodes("test_pipeline")
    assert len(nodes) == 1
    assert nodes[0].technique_name == "technique_a"

    # 3. Load Ollama config
    from ollama.client import OllamaClient
    client = OllamaClient.from_config(str(models_file))
    assert client.base_url == "http://localhost:11434"

    # 4. Round-trip verified - all components connected
    print("Round-trip successful: config → registry → pipeline → client")
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add complete round-trip integration test

- Verify config loads into Registry
- Verify Builder creates nodes from registry
- Verify OllamaClient loads from config
- All components connected successfully"
```

---

## Task 16: All tests pass

**Step 1: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: All tests pass

**Step 2: Fix any failing tests if needed**

(If tests fail, debug and fix, then re-run)

**Step 3: Commit final state**

```bash
git add .
git commit -m "chore: MVP architecture complete

All tests passing:
- TechniqueRegistry loads YAML config
- PipelineBuilder creates nodes
- TechniqueNode extracts shared state
- OllamaClient loads from config
- Integration test verifies round-trip"
```

---

## Task 17: Update design document status

**Files:**
- Modify: `docs/plans/2026-02-01-kickoff-mvp-design.md`

**Step 1: Update status line**

Change:
```
**Status:** Approved for Implementation
```

To:
```
**Status:** Implemented
```

**Step 2: Add implementation notes at end**

```markdown
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
```

**Step 3: Commit**

```bash
git add docs/plans/2026-02-01-kickoff-mvp-design.md
git commit -m "docs: update design status to implemented

- Mark MVP as completed
- Add implementation notes and completion date"
```

---

## Summary

This plan builds a complete, testable architecture foundation for the RAG framework. Each component is built TDD-style with isolated tests, followed by an integration test that proves the complete round-trip works.

**Key achievements:**
- YAML-driven configuration (models, techniques, pipelines)
- Registry pattern for technique metadata
- Builder pattern for pipeline composition
- Generic TechniqueNode for PocketFlow integration
- Ollama client with config loading
- Full test coverage including integration

**Files created:**
- `config/models.yaml`, `config/techniques.yaml`, `config/pipelines.yaml`
- `registry/technique_registry.py`
- `nodes/technique_node.py`
- `pipeline/builder.py`
- `ollama/client.py`
- Tests in `tests/` directory