from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class TechniqueMetadata:
    name: str
    enabled: bool
    config: Mapping[str, Any]

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)