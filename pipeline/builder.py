# pipeline/builder.py
import yaml

from nodes.technique_node import TechniqueNode


class PipelineBuilder:
    def __init__(self, config_path: str, registry: dict):
        self.config_path = config_path
        self.registry = registry
        self._load_config()

    def _load_config(self) -> None:
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    def build_nodes(self, pipeline_name: str) -> list[TechniqueNode]:
        pipeline = self.config["pipelines"][pipeline_name]
        technique_names = pipeline.get("techniques", [])
        nodes = []
        for name in technique_names:
            meta = self.registry[name]
            nodes.append(TechniqueNode(meta.name, meta.config))
        return nodes