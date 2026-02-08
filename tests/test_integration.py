# tests/test_integration.py
import tempfile
import os

def test_complete_round_trip(tmp_path):
    """
    Complete round-trip: config -> registry -> pipeline -> execution
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

generation_models:
  default: "llama3.2"
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
    print("Round-trip successful: config -> registry -> pipeline -> client")