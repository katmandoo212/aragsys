"""Configuration loader with environment variable support."""

import os
import re
import yaml
from typing import Any


def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in configuration values.

    Supports ${VAR_NAME} syntax with optional default values:
    - ${VAR_NAME} - raises error if not set
    - ${VAR_NAME:-default} - uses default if not set
    """
    if isinstance(value, str):
        # Match ${VAR_NAME} or ${VAR_NAME:-default}
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'

        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            raise ValueError(f"Environment variable '{var_name}' is not set and no default provided")

        return re.sub(pattern, replace_var, value)
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value


def load_config(path: str) -> dict:
    """Load YAML configuration with environment variable expansion.

    Args:
        path: Path to YAML configuration file

    Returns:
        Configuration dictionary with environment variables expanded
    """
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    return expand_env_vars(config)