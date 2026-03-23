"""Tests for configuration loader with environment variable support."""

import os
import pytest
from utils.config_loader import expand_env_vars, load_config


class TestExpandEnvVars:
    """Tests for environment variable expansion."""

    def test_expand_simple_var(self, monkeypatch):
        """Expand simple environment variable."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        result = expand_env_vars("${TEST_VAR}")
        assert result == "test_value"

    def test_expand_var_with_default_not_set(self, monkeypatch):
        """Use default value when env var not set."""
        monkeypatch.delenv("UNSET_VAR", raising=False)
        result = expand_env_vars("${UNSET_VAR:-default_value}")
        assert result == "default_value"

    def test_expand_var_with_default_override(self, monkeypatch):
        """Env var overrides default value."""
        monkeypatch.setenv("TEST_VAR", "actual_value")
        result = expand_env_vars("${TEST_VAR:-default_value}")
        assert result == "actual_value"

    def test_expand_var_not_set_raises_error(self):
        """Raise error when env var not set and no default."""
        with pytest.raises(ValueError, match="Environment variable 'UNDEFINED_VAR' is not set"):
            expand_env_vars("${UNDEFINED_VAR}")

    def test_expand_dict_values(self, monkeypatch):
        """Expand env vars in dictionary values."""
        monkeypatch.setenv("DB_PASSWORD", "secret123")
        config = {"database": {"password": "${DB_PASSWORD}"}}
        result = expand_env_vars(config)
        assert result["database"]["password"] == "secret123"

    def test_expand_list_values(self, monkeypatch):
        """Expand env vars in list values."""
        monkeypatch.setenv("HOST", "localhost")
        config = {"hosts": ["${HOST}", "backup.example.com"]}
        result = expand_env_vars(config)
        assert result["hosts"][0] == "localhost"

    def test_no_expansion_in_regular_strings(self):
        """Regular strings pass through unchanged."""
        result = expand_env_vars("hello world")
        assert result == "hello world"

    def test_multiple_vars_in_string(self, monkeypatch):
        """Expand multiple env vars in single string."""
        monkeypatch.setenv("USER", "admin")
        monkeypatch.setenv("HOST", "localhost")
        result = expand_env_vars("${USER}@${HOST}")
        assert result == "admin@localhost"


class TestLoadConfig:
    """Tests for YAML config loading."""

    def test_load_config_with_env_vars(self, tmp_path, monkeypatch):
        """Load YAML and expand environment variables."""
        monkeypatch.setenv("TEST_PASSWORD", "mysecretpassword")

        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
database:
  user: admin
  password: "${TEST_PASSWORD}"
""")

        config = load_config(str(config_file))
        assert config["database"]["password"] == "mysecretpassword"

    def test_load_config_with_defaults(self, tmp_path, monkeypatch):
        """Load YAML and use default values."""
        monkeypatch.delenv("OPTIONAL_VAR", raising=False)

        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
settings:
  value: "${OPTIONAL_VAR:-default_value}"
""")

        config = load_config(str(config_file))
        assert config["settings"]["value"] == "default_value"

    def test_load_config_preserves_other_values(self, tmp_path):
        """Non-env-var values are preserved."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
name: test_project
count: 42
enabled: true
""")

        config = load_config(str(config_file))
        assert config["name"] == "test_project"
        assert config["count"] == 42
        assert config["enabled"] is True