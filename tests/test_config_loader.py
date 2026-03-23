"""Tests for configuration loader with environment variable support."""

import os
import pytest
from pathlib import Path
from utils.config_loader import expand_env_vars, load_config, load_env_files


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

    def test_empty_default(self, monkeypatch):
        """Empty default is valid."""
        monkeypatch.delenv("OPTIONAL_VAR", raising=False)
        result = expand_env_vars("${OPTIONAL_VAR:-}")
        assert result == ""

    def test_nested_expansion(self, monkeypatch):
        """Expand vars in nested structures."""
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")
        config = {
            "database": {
                "host": "${DB_HOST}",
                "port": "${DB_PORT}",
                "credentials": {
                    "user": "admin",
                    "password": "${DB_PASSWORD:-default_pass}"
                }
            }
        }
        result = expand_env_vars(config)
        assert result["database"]["host"] == "localhost"
        assert result["database"]["port"] == "5432"
        assert result["database"]["credentials"]["password"] == "default_pass"


class TestLoadEnvFiles:
    """Tests for .env file loading."""

    def test_load_env_file(self, tmp_path, monkeypatch):
        """Load variables from .env file."""
        monkeypatch.delenv("FROM_ENV_FILE", raising=False)

        env_file = tmp_path / ".env"
        env_file.write_text("FROM_ENV_FILE=loaded_value\n")

        load_env_files(env_file=env_file, project_root=tmp_path)
        assert os.environ.get("FROM_ENV_FILE") == "loaded_value"

    def test_load_env_local_overrides_env(self, tmp_path, monkeypatch):
        """ .env.local overrides .env values."""
        monkeypatch.delenv("OVERRIDE_VAR", raising=False)

        env_file = tmp_path / ".env"
        env_local_file = tmp_path / ".env.local"

        env_file.write_text("OVERRIDE_VAR=original\n")
        env_local_file.write_text("OVERRIDE_VAR=overridden\n")

        load_env_files(project_root=tmp_path)
        assert os.environ.get("OVERRIDE_VAR") == "overridden"

    def test_missing_env_file_no_error(self, tmp_path):
        """Missing .env file does not raise error."""
        # Should not raise any error
        load_env_files(project_root=tmp_path)

    def test_custom_env_file(self, tmp_path, monkeypatch):
        """Load custom env file."""
        monkeypatch.delenv("CUSTOM_VAR", raising=False)

        custom_env = tmp_path / "custom.env"
        custom_env.write_text("CUSTOM_VAR=custom_value\n")

        load_env_files(env_file=custom_env, project_root=tmp_path)
        assert os.environ.get("CUSTOM_VAR") == "custom_value"


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

    def test_load_config_missing_file_raises_error(self, tmp_path):
        """Missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_load_config_with_dotenv(self, tmp_path, monkeypatch):
        """Config loads .env file automatically."""
        monkeypatch.delenv("DOTENV_VAR", raising=False)

        # Create .env file
        env_file = tmp_path / ".env"
        env_file.write_text("DOTENV_VAR=from_dotenv\n")

        # Create config that references the var
        config_file = tmp_path / "config.yaml"
        config_file.write_text("value: ${DOTENV_VAR:-default}\n")

        config = load_config(str(config_file), project_root=tmp_path)
        assert config["value"] == "from_dotenv"

    def test_load_config_skip_env_loading(self, tmp_path, monkeypatch):
        """Can skip .env loading with load_env=False."""
        monkeypatch.delenv("SKIP_VAR", raising=False)

        # Create .env file
        env_file = tmp_path / ".env"
        env_file.write_text("SKIP_VAR=should_not_load\n")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("value: ${SKIP_VAR:-fallback}\n")

        config = load_config(str(config_file), load_env=False, project_root=tmp_path)
        assert config["value"] == "fallback"

    def test_load_config_with_path_object(self, tmp_path, monkeypatch):
        """Accepts Path objects as well as strings."""
        monkeypatch.setenv("PATH_VAR", "path_value")

        config_file = tmp_path / "test.yaml"
        config_file.write_text("value: ${PATH_VAR}\n")

        config = load_config(config_file)
        assert config["value"] == "path_value"

    def test_load_config_empty_file(self, tmp_path):
        """Empty config file returns empty dict."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        config = load_config(str(config_file))
        assert config == {}

    def test_load_config_with_list(self, tmp_path, monkeypatch):
        """Config with list values expands correctly."""
        monkeypatch.setenv("ITEM1", "first")
        monkeypatch.setenv("ITEM2", "second")

        config_file = tmp_path / "list.yaml"
        config_file.write_text("""
items:
  - "${ITEM1}"
  - "${ITEM2}"
  - static_value
""")

        config = load_config(str(config_file))
        assert config["items"] == ["first", "second", "static_value"]