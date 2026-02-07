import pytest
from unittest.mock import Mock, patch
from techniques.naive_rag import NaiveRAGTechnique
from utils.document import Document


def test_naive_rag_technique_creation():
    config = {
        "embedding_model": "bge-m3:latest",
        "top_k": 5,
        "collection_name": "documents"
    }

    technique = NaiveRAGTechnique(config)

    assert technique.config == config
    assert technique.embedding_model == "bge-m3:latest"
    assert technique.top_k == 5


def test_naive_rag_technique_retrieve_returns_documents():
    config = {
        "embedding_model": "bge-m3:latest",
        "top_k": 2,
        "collection_name": "test_collection"
    }

    mock_ollama = Mock()
    mock_store = Mock()

    technique = NaiveRAGTechnique(config, ollama_client=mock_ollama, vector_store=mock_store)

    mock_ollama.embed.return_value = [0.1, 0.2, 0.3]
    mock_store.search.return_value = [
        {"content": "Doc 1", "metadata": {"source": "test"}, "distance": 0.1},
        {"content": "Doc 2", "metadata": {"source": "test"}, "distance": 0.2}
    ]

    results = technique.retrieve("test query")

    assert len(results) == 2
    assert all(isinstance(r, Document) for r in results)
    assert results[0].content == "Doc 1"
    assert results[0].score == 0.9  # 1.0 - 0.1
    assert results[1].score == 0.8  # 1.0 - 0.2


def test_naive_rag_technique_retrieve_no_results():
    config = {"embedding_model": "bge-m3:latest", "top_k": 5}

    mock_ollama = Mock()
    mock_store = Mock()

    technique = NaiveRAGTechnique(config, ollama_client=mock_ollama, vector_store=mock_store)

    mock_ollama.embed.return_value = [0.1, 0.2, 0.3]
    mock_store.search.return_value = []

    results = technique.retrieve("test query")

    assert results == []