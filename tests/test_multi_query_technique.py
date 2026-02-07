"""Tests for MultiQueryTechnique."""

import pytest
from unittest.mock import Mock
from techniques.multi_query import MultiQueryTechnique


class TestMultiQueryTechniqueCreation:
    """Test MultiQuery technique initialization."""

    def test_multi_query_technique_creates(self):
        """MultiQueryTechnique creates with config."""
        ollama_client = Mock()
        vector_store = Mock()
        config = {
            "embedding_model": "bge-m3:latest",
            "generation_model": "glm-4.7:cloud",
            "num_queries": 3,
            "top_k": 5
        }
        technique = MultiQueryTechnique(config, ollama_client, vector_store)
        assert technique.num_queries == 3
        assert technique.top_k == 5


class TestMultiQueryGeneration:
    """Test multiple query generation."""

    def test_generate_multiple_queries(self):
        """Generate multiple query variations."""
        ollama_client = Mock()
        ollama_client.generate.return_value = "Query 1\nQuery 2\nQuery 3"

        vector_store = Mock()
        config = {
            "embedding_model": "bge-m3:latest",
            "generation_model": "glm-4.7:cloud",
            "num_queries": 3,
            "top_k": 5
        }
        technique = MultiQueryTechnique(config, ollama_client, vector_store)

        queries = technique._generate_queries("What is RAG?")
        assert len(queries) >= 2
        assert "Query 1" in queries


class TestMultiQueryRetrieval:
    """Test MultiQuery retrieval flow."""

    def test_multi_query_retrieves_and_deduplicates(self):
        """MultiQuery retrieves for each query and deduplicates."""
        ollama_client = Mock()
        ollama_client.generate.return_value = "Query A\nQuery B"
        ollama_client.embed.return_value = [0.1] * 1024

        vector_store = Mock()
        vector_store.vector_search.return_value = [
            {"content": "result 1", "metadata": {"id": "1"}, "distance": 0.2},
            {"content": "result 2", "metadata": {"id": "2"}, "distance": 0.3}
        ]

        config = {
            "embedding_model": "bge-m3:latest",
            "generation_model": "glm-4.7:cloud",
            "num_queries": 3,
            "top_k": 5
        }
        technique = MultiQueryTechnique(config, ollama_client, vector_store)

        documents = technique.retrieve("What is RAG?")

        # Should deduplicate by document id
        assert len(documents) <= 4  # 2 results per query * 2 queries, but deduplicated
        assert ollama_client.generate.assert_called_once()