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