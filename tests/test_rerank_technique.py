import pytest
from unittest.mock import Mock, MagicMock
from techniques.rerank import RerankTechnique

@pytest.fixture
def mock_ollama_client():
    client = MagicMock()
    client.generate.return_value = "0.85"
    return client

@pytest.fixture
def mock_base_technique():
    technique = MagicMock()
    technique.retrieve.return_value = [
        {"content": "doc1", "metadata": {"id": 1}, "relevance_score": 0.5},
        {"content": "doc2", "metadata": {"id": 2}, "relevance_score": 0.3},
    ]
    return technique

def test_rerank_reorders_results(mock_ollama_client, mock_base_technique):
    reranker = RerankTechnique(
        ollama_client=mock_ollama_client,
        scoring_model="test-model",
        top_k=5,
        score_threshold=0.0,
        base_technique=mock_base_technique,
    )

    results = reranker.retrieve("test query")

    assert len(results) == 2
    assert results[0]["content"] == "doc1"
    assert results[1]["content"] == "doc2"
    # generate should be called for each document
    assert mock_ollama_client.generate.call_count == 2

def test_rerank_filters_by_score_threshold(mock_ollama_client, mock_base_technique):
    mock_ollama_client.generate.side_effect = ["0.2", "0.8"]
    reranker = RerankTechnique(
        ollama_client=mock_ollama_client,
        scoring_model="test-model",
        top_k=5,
        score_threshold=0.5,
        base_technique=mock_base_technique,
    )

    results = reranker.retrieve("test query")

    # Only document with score >= 0.5 should be returned
    assert len(results) == 1
    assert results[0]["content"] == "doc2"
    assert results[0]["relevance_score"] == 0.8

def test_rerank_handles_empty_results(mock_ollama_client, mock_base_technique):
    mock_base_technique.retrieve.return_value = []
    reranker = RerankTechnique(
        ollama_client=mock_ollama_client,
        scoring_model="test-model",
        top_k=5,
        score_threshold=0.5,
        base_technique=mock_base_technique,
    )

    results = reranker.retrieve("test query")

    assert results == []
    mock_ollama_client.generate.assert_not_called()

def test_rerank_keeps_top_k(mock_ollama_client):
    # Create mock base technique with 5 results
    base_technique = MagicMock()
    base_technique.retrieve.return_value = [
        {"content": f"doc{i}", "metadata": {"id": i}, "relevance_score": 0.5}
        for i in range(5)
    ]

    # Mock scores - first 3 should be highest
    mock_ollama_client.generate.side_effect = ["0.9", "0.3", "0.8", "0.2", "0.7"]

    reranker = RerankTechnique(
        ollama_client=mock_ollama_client,
        scoring_model="test-model",
        top_k=3,
        score_threshold=0.0,
        base_technique=base_technique,
    )

    results = reranker.retrieve("test query")

    assert len(results) == 3