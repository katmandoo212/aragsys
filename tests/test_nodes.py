from nodes.technique_node import TechniqueNode


def test_technique_node_exists():
    node = TechniqueNode("test_technique", {"chunk_size": 500})
    assert node.technique_name == "test_technique"
    assert node.config == {"chunk_size": 500}