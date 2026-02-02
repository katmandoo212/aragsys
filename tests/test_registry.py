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