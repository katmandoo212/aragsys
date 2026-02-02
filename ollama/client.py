from dataclasses import dataclass
import yaml

@dataclass
class OllamaClient:
    base_url: str

    @classmethod
    def from_config(cls, config_path: str) -> "OllamaClient":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(base_url=config["ollama"]["base_url"])