"""Tests for GraphEntityTechnique."""

import pytest
from unittest.mock import MagicMock, patch
from techniques.graph_entity import GraphEntityTechnique


class TestGraphEntityTechnique:
    """Test GraphEntityTechnique for entity-based retrieval."""

    def test_initialization(self):
        """GraphEntityTechnique initializes with config."""
        config = {"max_hops": 2, "top_k": 5}
        technique = GraphEntityTechnique(config, neo4j_store=MagicMock())
        assert technique.max_hops == 2
        assert technique.top_k == 5

    def test_retrieve_with_empty_query(self):
        """Empty query returns empty results."""
        config = {"max_hops": 2, "top_k": 5}
        technique = GraphEntityTechnique(config, neo4j_store=MagicMock())
        results = technique.retrieve("")
        assert results == []

    def test_entity_lookup_calls_neo4j(self):
        """Entity lookup queries Neo4j for matching entities."""
        from utils.document import Document

        config = {"max_hops": 2, "top_k": 5}

        mock_store = MagicMock()
        mock_store.find_entities_in_query.return_value = [
            {"id": "ent1", "name": "John Smith", "type": "PERSON"}
        ]
        mock_store.get_connected_documents.return_value = [
            {"content": "John Smith's research", "metadata": {"source": "doc1"}, "doc_id": "doc1"}
        ]

        technique = GraphEntityTechnique(config, neo4j_store=mock_store)
        results = technique.retrieve("What did John Smith research?")

        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert results[0].content == "John Smith's research"
        mock_store.find_entities_in_query.assert_called_once_with("What did John Smith research?")
        mock_store.get_connected_documents.assert_called_once_with("ent1", max_hops=2)