"""Tests for HybridTechnique."""

import pytest
from unittest.mock import Mock
from techniques.hybrid import HybridTechnique


class TestHybridTechniqueCreation:
    """Test Hybrid technique initialization."""

    def test_hybrid_technique_creates(self):
        """HybridTechnique creates with config."""
        ollama_client = Mock()
        vector_store = Mock()
        config = {
            "embedding_model": "bge-m3:latest",
            "top_k": 10,
            "rrf_k": 60
        }
        technique = HybridTechnique(config, ollama_client, vector_store)
        assert technique.rrf_k == 60
        assert technique.top_k == 10


class TestHybridRetrieval:
    """Test Hybrid retrieval flow."""

    def test_hybrid_search_merges_dense_and_sparse(self):
        """Hybrid search merges dense and sparse results."""
        ollama_client = Mock()
        ollama_client.embed.return_value = [0.1] * 1024

        vector_store = Mock()
        vector_store.vector_search.return_value = [
            {"content": "result 1", "metadata": {"id": "1"}, "distance": 0.2},
            {"content": "result 2", "metadata": {"id": "2"}, "distance": 0.3}
        ]
        vector_store.fulltext_search.return_value = [
            {"content": "result 1", "metadata": {"id": "1"}, "distance": 0.4},
            {"content": "result 3", "metadata": {"id": "3"}, "distance": 0.5}
        ]
        vector_store.hybrid_search.return_value = [
            {"content": "result 1", "metadata": {"id": "1"}, "score": 0.9},
            {"content": "result 2", "metadata": {"id": "2"}, "score": 0.8}
        ]

        config = {
            "embedding_model": "bge-m3:latest",
            "top_k": 10,
            "rrf_k": 60
        }
        technique = HybridTechnique(config, ollama_client, vector_store)

        documents = technique.retrieve("test query")

        assert len(documents) >= 1
        ollama_client.embed.assert_called_once()