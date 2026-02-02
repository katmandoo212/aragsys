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


def test_technique_registry_raises_technique_not_found():
    from registry.technique_registry import TechniqueRegistry, TechniqueNotFoundError

    # Test exception can be raised
    with pytest.raises(TechniqueNotFoundError) as exc_info:
        raise TechniqueNotFoundError("missing_tech")

    assert "missing_tech" in str(exc_info.value)