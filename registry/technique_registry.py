from dataclasses import dataclass
from typing import Any, Mapping
import yaml


class TechniqueNotFoundError(Exception):
    pass


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
            raise TechniqueNotFoundError(f"Technique '{name}' not found")
        return self._techniques[name]

    def list_techniques(self) -> list[str]:
        return [name for name, meta in self._techniques.items() if meta.enabled]