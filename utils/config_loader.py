"""Configuration loader with environment variable support."""

import os
import re
import yaml
from pathlib import Path
from typing import Any

# Optional dotenv support - graceful fallback if not installed
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    def load_dotenv(*args, **kwargs):
        pass


# Compiled regex for env var pattern (module-level for performance)
_ENV_VAR_PATTERN = re.compile(r'\$\{([^}:]+)(?::-([^}]*))?\}')


def load_env_files(env_file: str | Path | None = None, project_root: str | Path | None = None) -> None:
    """Load environment variables from .env files.

    Loads files in order (later files override earlier):
    1. {project_root}/.env
    2. {project_root}/.env.local (for local overrides, git-ignored)
    3. Custom env_file if provided

    Args:
        env_file: Optional custom .env file path
        project_root: Project root directory. Defaults to current working directory.
    """
    if not DOTENV_AVAILABLE:
        return

    root = Path(project_root) if project_root else Path.cwd()

    # Load in order: .env, .env.local, then custom
    for env_path in [root / '.env', root / '.env.local']:
        if env_path.exists():
            load_dotenv(env_path, override=True)

    # Load custom env file if provided
    if env_file:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path, override=True)


def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in configuration values.

    Supports ${VAR_NAME} syntax with optional default values:
    - ${VAR_NAME} - raises error if not set
    - ${VAR_NAME:-default} - uses default if not set

    Args:
        value: Configuration value (string, dict, list, or other)

    Returns:
        Value with environment variables expanded

    Raises:
        ValueError: If required environment variable is not set and no default provided
    """
    if isinstance(value, str):
        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            raise ValueError(f"Environment variable '{var_name}' is not set and no default provided")

        return _ENV_VAR_PATTERN.sub(replace_var, value)
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value


def load_config(
    path: str | Path,
    *,
    env_file: str | Path | None = None,
    load_env: bool = True,
    project_root: str | Path | None = None
) -> dict:
    """Load YAML configuration with environment variable expansion.

    Automatically loads .env files before expanding variables (unless load_env=False).

    Args:
        path: Path to YAML configuration file
        env_file: Optional custom .env file path
        load_env: Whether to load .env files (default: True)
        project_root: Project root directory for finding .env files

    Returns:
        Configuration dictionary with environment variables expanded

    Raises:
        FileNotFoundError: If configuration file does not exist
        ValueError: If required environment variable is not set
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    # Load .env files if requested
    if load_env:
        # Use config file directory as project root if not specified
        root = project_root or config_path.parent.parent
        load_env_files(env_file=env_file, project_root=root)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    return expand_env_vars(config)