"""Tests for HyDETechnique."""

import pytest
from unittest.mock import Mock
from techniques.hyde import HyDETechnique


class TestHyDETechniqueCreation:
    """Test HyDE technique initialization."""

    def test_hyde_technique_creates(self):
        """HyDETechnique creates with config."""
        ollama_client = Mock()
        vector_store = Mock()
        config = {
            "embedding_model": "bge-m3:latest",
            "generation_model": "glm-4.7:cloud",
            "top_k": 5
        }
        technique = HyDETechnique(config, ollama_client, vector_store)
        assert technique.top_k == 5
        assert technique.embedding_model == "bge-m3:latest"
        assert technique.generation_model == "glm-4.7:cloud"


class TestHyDEHypotheticalGeneration:
    """Test hypothetical document generation."""

    def test_generate_hypothetical_answer(self):
        """Generate hypothetical answer for query."""
        ollama_client = Mock()
        ollama_client.generate.return_value = "This is a hypothetical answer."

        vector_store = Mock()
        config = {
            "embedding_model": "bge-m3:latest",
            "generation_model": "glm-4.7:cloud",
            "top_k": 5
        }
        technique = HyDETechnique(config, ollama_client, vector_store)

        hypothetical = technique._generate_hypothetical("What is RAG?")
        assert hypothetical == "This is a hypothetical answer."
        ollama_client.generate.assert_called_once()


class TestHyDERetrieval:
    """Test HyDE retrieval flow."""

    def test_hyde_retrieves_with_hypothetical_embedding(self):
        """HyDE retrieves using hypothetical answer embedding."""
        ollama_client = Mock()
        ollama_client.generate.return_value = "Hypothetical answer."
        ollama_client.embed.return_value = [0.1] * 1024

        vector_store = Mock()
        vector_store.vector_search.return_value = [
            {"content": "result 1", "metadata": {}, "distance": 0.2},
            {"content": "result 2", "metadata": {}, "distance": 0.3}
        ]

        config = {
            "embedding_model": "bge-m3:latest",
            "generation_model": "glm-4.7:cloud",
            "top_k": 5
        }
        technique = HyDETechnique(config, ollama_client, vector_store)

        documents = technique.retrieve("What is RAG?")

        assert len(documents) == 2
        assert documents[0].content == "result 1"
        assert documents[0].score > 0  # Distance converted to score
        ollama_client.generate.assert_called_once()
        ollama_client.embed.assert_called_once()
        vector_store.vector_search.assert_called_once()