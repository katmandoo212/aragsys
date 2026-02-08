import pytest
from unittest.mock import MagicMock
from techniques.compress import CompressTechnique

@pytest.fixture
def mock_ollama_client():
    client = MagicMock()
    client.generate.return_value = "relevant content"
    return client

@pytest.fixture
def mock_base_technique():
    technique = MagicMock()
    technique.retrieve.return_value = [
        {
            "content": "This is about cats. Dogs are also mentioned. This talks about cats again.",
            "metadata": {"id": 1},
            "relevance_score": 0.8,
        }
    ]
    return technique

def test_keyword_extraction_filters_segments(mock_ollama_client, mock_base_technique):
    compressor = CompressTechnique(
        ollama_client=mock_ollama_client,
        use_llm_refinement=False,
        min_keyword_matches=1,
        segment_length=50,
        top_k_segments=3,
        base_technique=mock_base_technique,
    )

    results = compressor.retrieve("cats")

    # Should extract segments containing "cats"
    assert len(results) == 1
    assert "cats" in results[0]["content"].lower()
    # LLM should not be called when use_llm_refinement=False
    mock_ollama_client.generate.assert_not_called()

def test_llm_refinement_compresses_content(mock_ollama_client, mock_base_technique):
    mock_ollama_client.generate.return_value = "Cats are furry animals."
    compressor = CompressTechnique(
        ollama_client=mock_ollama_client,
        use_llm_refinement=True,
        min_keyword_matches=1,
        segment_length=50,
        top_k_segments=3,
        base_technique=mock_base_technique,
    )

    results = compressor.retrieve("cats")

    assert len(results) == 1
    assert results[0]["content"] == "Cats are furry animals."
    mock_ollama_client.generate.assert_called_once()

def test_fallback_without_llm_refinement(mock_base_technique):
    client = MagicMock()
    compressor = CompressTechnique(
        ollama_client=client,
        use_llm_refinement=False,
        min_keyword_matches=1,
        segment_length=50,
        top_k_segments=3,
        base_technique=mock_base_technique,
    )

    results = compressor.retrieve("cats")

    assert len(results) == 1
    client.generate.assert_not_called()

def test_document_with_no_matches(mock_ollama_client, mock_base_technique):
    mock_base_technique.retrieve.return_value = [
        {"content": "This is about nothing relevant.", "metadata": {"id": 1}, "relevance_score": 0.5}
    ]
    compressor = CompressTechnique(
        ollama_client=mock_ollama_client,
        use_llm_refinement=False,
        min_keyword_matches=1,
        segment_length=50,
        top_k_segments=3,
        base_technique=mock_base_technique,
    )

    results = compressor.retrieve("cats")

    # No matches, should return empty or original
    assert isinstance(results, list)

def test_configuration_options(mock_ollama_client, mock_base_technique):
    compressor = CompressTechnique(
        ollama_client=mock_ollama_client,
        use_llm_refinement=True,
        min_keyword_matches=2,
        segment_length=20,
        top_k_segments=1,
        base_technique=mock_base_technique,
    )

    # Verify config is stored
    assert compressor.min_keyword_matches == 2
    assert compressor.segment_length == 20
    assert compressor.top_k_segments == 1
    assert compressor.use_llm_refinement is True