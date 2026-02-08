"""Tests for GraphRAG configuration."""

import pytest
import yaml


def test_graphrag_config_exists():
    """GraphRAG config file exists and is valid YAML."""
    with open("config/graphrag.yaml") as f:
        config = yaml.safe_load(f)
    assert "graph" in config
    assert "neo4j" in config


def test_graphrag_config_has_entity_types():
    """Config defines entity types for extraction."""
    with open("config/graphrag.yaml") as f:
        config = yaml.safe_load(f)
    assert "entity_types" in config["graph"]
    assert isinstance(config["graph"]["entity_types"], list)


def test_graphrag_config_has_defaults():
    """Config has sensible default values."""
    with open("config/graphrag.yaml") as f:
        config = yaml.safe_load(f)
    assert config["graph"]["max_hops"] > 0
    assert config["graph"]["top_k"] > 0