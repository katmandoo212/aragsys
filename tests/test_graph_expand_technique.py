"""Tests for GraphExpandTechnique."""

import pytest
from unittest.mock import MagicMock
from techniques.graph_expand import GraphExpandTechnique


class TestGraphExpandTechnique:
    """Test GraphExpandTechnique for entity relationship expansion."""

    def test_initialization(self):
        """GraphExpandTechnique initializes with config."""
        config = {"max_hops": 2, "min_doc_count": 1}
        technique = GraphExpandTechnique(config, neo4j_store=MagicMock())
        assert technique.max_hops == 2
        assert technique.min_doc_count == 1

    def test_expand_queries_entity_relationships(self):
        """Expand technique queries related entities."""
        from utils.document import Document

        config = {"max_hops": 2, "min_doc_count": 1}

        mock_store = MagicMock()
        mock_store.find_entities_in_query.return_value = [
            {"id": "ent1", "name": "John Smith", "type": "PERSON"}
        ]
        mock_store.get_entity_relationships.return_value = [
            {"id": "ent2", "name": "MIT", "type": "ORG", "doc_count": 3},
            {"id": "ent3", "name": "Stanford", "type": "ORG", "doc_count": 1}
        ]
        mock_store.get_connected_documents.return_value = [
            {"content": "MIT research paper", "metadata": {"source": "doc1"}, "doc_id": "doc1"}
        ]

        technique = GraphExpandTechnique(config, neo4j_store=mock_store)
        results = technique.retrieve("What organizations did John Smith work with?")

        assert len(results) >= 1
        mock_store.get_entity_relationships.assert_called_once_with("ent1", max_hops=2)