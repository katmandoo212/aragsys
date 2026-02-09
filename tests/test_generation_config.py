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