"""Tests for SimpleGenerationTechnique."""

from unittest.mock import MagicMock, patch
from techniques.generate_simple import SimpleGenerationTechnique
from utils.answer import Answer


class TestSimpleGenerationTechnique:
    """Test SimpleGenerationTechnique for basic LLM generation."""

    def test_initialization(self):
        """SimpleGenerationTechnique initializes with config."""
        config = {"model": "llama3:8b", "max_context_docs": 5}
        technique = SimpleGenerationTechnique(config, ollama_client=MagicMock())
        assert technique.model == "llama3:8b"
        assert technique.max_context_docs == 5

    def test_generate_empty_query(self):
        """Empty query returns empty answer."""
        config = {"model": "llama3:8b"}
        technique = SimpleGenerationTechnique(config, ollama_client=MagicMock())
        result = technique.generate("", [])
        assert isinstance(result, Answer)
        assert result.content == ""

    def test_generate_calls_ollama(self):
        """Generate calls OllamaClient.generate with prompt."""
        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = "Paris is France's capital."

        technique = SimpleGenerationTechnique(config, ollama_client=mock_client)
        result = technique.generate("What is France's capital?", [])

        assert isinstance(result, Answer)
        assert result.content == "Paris is France's capital."
        mock_client.generate.assert_called_once()

    def test_generate_includes_context(self):
        """Generate includes document context in prompt."""
        from utils.document import Document

        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = "Answer"

        docs = [
            Document(content="Paris is a city.", metadata={"source": "doc1"}, score=0.9),
            Document(content="France is a country.", metadata={"source": "doc2"}, score=0.8)
        ]

        technique = SimpleGenerationTechnique(config, ollama_client=mock_client)
        technique.generate("Query", docs)

        # Verify generate was called
        assert mock_client.generate.called
        call_args = mock_client.generate.call_args
        prompt = call_args[0][0]
        assert "Paris is a city" in prompt
        assert "France is a country" in prompt