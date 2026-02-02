from nodes.technique_node import TechniqueNode


def test_technique_node_exists():
    node = TechniqueNode("test_technique", {"chunk_size": 500})
    assert node.technique_name == "test_technique"
    assert node.config == {"chunk_size": 500}


def test_technique_node_prep_extracts_specific_keys():
    node = TechniqueNode("test", {})
    shared = {"query": "test query", "retrieved_docs": []}

    query, docs = node.prep(shared)
    assert query == "test query"
    assert docs == []