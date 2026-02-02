# tests/test_builder.py
from pipeline.builder import PipelineBuilder

def test_pipeline_builder_creation():
    from registry.technique_registry import TechniqueRegistry
    import tempfile
    import os

    yaml_content = "pipelines:\n  test_flow:\n    techniques: [tech1, tech2]"
    yaml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml_file.write(yaml_content)
    yaml_file.close()

    try:
        mock_registry = {"tech1": {"enabled": True, "config": {}}}
        builder = PipelineBuilder(yaml_file.name, mock_registry)
        assert builder is not None
    finally:
        os.unlink(yaml_file.name)


def test_pipeline_build_creates_nodes():
    import tempfile
    import os
    from nodes.technique_node import TechniqueNode

    yaml_content = "pipelines:\n  test_flow:\n    techniques: [tech1, tech2]"
    yaml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml_file.write(yaml_content)
    yaml_file.close()

    try:
        # Mock registry with technique metadata
        class MockMetadata:
            def __init__(self, name, enabled, config):
                self.name = name
                self.enabled = enabled
                self.config = config

        mock_registry = {
            "tech1": MockMetadata("tech1", True, {"param": "value1"}),
            "tech2": MockMetadata("tech2", True, {"param": "value2"})
        }

        builder = PipelineBuilder(yaml_file.name, mock_registry)
        nodes = builder.build_nodes("test_flow")

        assert len(nodes) == 2
        assert isinstance(nodes[0], TechniqueNode)
        assert isinstance(nodes[1], TechniqueNode)
        assert nodes[0].technique_name == "tech1"
        assert nodes[1].technique_name == "tech2"
    finally:
        os.unlink(yaml_file.name)