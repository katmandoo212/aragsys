"""Tests for ChainOfThoughtGenerationTechnique."""

from unittest.mock import MagicMock
from techniques.generate_cot import ChainOfThoughtGenerationTechnique
from utils.answer import Answer


class TestChainOfThoughtGenerationTechnique:
    """Test ChainOfThoughtGenerationTechnique for reasoning-based generation."""

    def test_initialization(self):
        """ChainOfThoughtGenerationTechnique initializes with config."""
        config = {"model": "llama3:8b", "max_context_docs": 3}
        technique = ChainOfThoughtGenerationTechnique(config, ollama_client=MagicMock())
        assert technique.model == "llama3:8b"
        assert technique.max_context_docs == 3

    def test_generate_includes_reasoning_instructions(self):
        """Generate includes step-by-step reasoning instructions."""
        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = "Reasoning:\n1. First step.\n2. Second step.\n\nAnswer: Result."

        technique = ChainOfThoughtGenerationTechnique(config, ollama_client=mock_client)
        technique.generate("Query", [])

        call_args = mock_client.generate.call_args
        prompt = call_args[0][0]
        assert "step by step" in prompt.lower()
        assert "reasoning" in prompt.lower()

    def test_generate_returns_answer_without_reasoning(self):
        """Generate strips reasoning from final answer."""
        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = """Reasoning:
Let me think about this.
The answer is Paris.

Answer: Paris"""

        technique = ChainOfThoughtGenerationTechnique(config, ollama_client=mock_client)
        result = technique.generate("Query", [])

        # Answer should be extracted from "Answer:" section
        assert "Paris" in result.content

    def test_generate_empty_query(self):
        """Empty query returns empty answer."""
        config = {"model": "llama3:8b"}
        technique = ChainOfThoughtGenerationTechnique(config, ollama_client=MagicMock())
        result = technique.generate("", [])
        assert isinstance(result, Answer)
        assert result.content == ""

    def test_generate_calls_ollama(self):
        """Generate calls OllamaClient.generate with prompt."""
        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = "Reasoning steps...\n\nAnswer: Paris is France's capital."

        technique = ChainOfThoughtGenerationTechnique(config, ollama_client=mock_client)
        result = technique.generate("What is France's capital?", [])

        assert isinstance(result, Answer)
        mock_client.generate.assert_called_once()

    def test_extract_answer_handles_final_answer(self):
        """Extract answer handles 'Final Answer:' format."""
        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = "Reasoning...\n\nFinal Answer: Paris"

        technique = ChainOfThoughtGenerationTechnique(config, ollama_client=mock_client)
        result = technique.generate("Query", [])

        assert "Paris" in result.content

    def test_extract_answer_fallback_when_no_answer_marker(self):
        """Extract answer returns full response when no marker found."""
        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = "This is the direct answer."

        technique = ChainOfThoughtGenerationTechnique(config, ollama_client=mock_client)
        result = technique.generate("Query", [])

        assert "direct answer" in result.content