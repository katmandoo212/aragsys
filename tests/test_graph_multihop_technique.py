"""Tests for GraphMultiHopTechnique."""

import pytest
from unittest.mock import MagicMock
from techniques.graph_multihop import GraphMultiHopTechnique


class TestGraphMultiHopTechnique:
    """Test GraphMultiHopTechnique for multi-hop reasoning."""

    def test_initialization(self):
        """GraphMultiHopTechnique initializes with config."""
        config = {"max_hops": 3, "top_k": 5}
        technique = GraphMultiHopTechnique(config, neo4j_store=MagicMock())
        assert technique.max_hops == 3
        assert technique.top_k == 5

    def test_retrieve_with_no_entities(self):
        """No entities found returns empty results."""
        from utils.document import Document

        config = {"max_hops": 3, "top_k": 5}

        mock_store = MagicMock()
        mock_store.find_entities_in_query.return_value = []

        technique = GraphMultiHopTechnique(config, neo4j_store=mock_store)
        results = technique.retrieve("Query with no entities")

        assert results == []

    def test_multi_hop_queries_entity_paths(self):
        """Multi-hop technique queries paths between entities."""
        from utils.document import Document

        config = {"max_hops": 3, "top_k": 5}

        mock_store = MagicMock()
        mock_store.find_entities_in_query.return_value = [
            {"id": "ent1", "name": "MIT", "type": "ORG"},
            {"id": "ent2", "name": "Stanford", "type": "ORG"}
        ]
        mock_store.multi_hop_query.return_value = [
            {"doc_id": "doc1", "content": "MIT-Stanford collaboration", "path_length": 1},
            {"doc_id": "doc2", "content": "Research partnership", "path_length": 2}
        ]

        technique = GraphMultiHopTechnique(config, neo4j_store=mock_store)
        results = technique.retrieve("How are MIT and Stanford connected?")

        assert len(results) == 2
        assert results[0].content == "MIT-Stanford collaboration"
        assert results[0].metadata["path_length"] == 1

        mock_store.multi_hop_query.assert_called_once_with(
            ["ent1", "ent2"], max_hops=3
        )