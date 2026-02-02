# pipeline/builder.py
import yaml

class PipelineBuilder:
    def __init__(self, config_path: str, registry: dict):
        self.config_path = config_path
        self.registry = registry
        self._load_config()

    def _load_config(self) -> None:
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)