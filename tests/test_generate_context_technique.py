"""Tests for ContextGenerationTechnique."""

from unittest.mock import MagicMock
from techniques.generate_context import ContextGenerationTechnique
from utils.answer import Answer
from utils.document import Document


class TestContextGenerationTechnique:
    """Test ContextGenerationTechnique for citation-aware generation."""

    def test_initialization(self):
        """ContextGenerationTechnique initializes with config."""
        config = {"model": "llama3:8b", "max_context_docs": 3}
        technique = ContextGenerationTechnique(config, ollama_client=MagicMock())
        assert technique.model == "llama3:8b"
        assert technique.max_context_docs == 3

    def test_generate_returns_citations(self):
        """Generate includes citations from used documents."""
        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = "Paris is the capital [1]."

        docs = [
            Document(content="Paris is France's capital.", metadata={"source": "doc1"}, score=0.9)
        ]

        technique = ContextGenerationTechnique(config, ollama_client=mock_client)
        result = technique.generate("Query", docs)

        assert isinstance(result, Answer)
        assert "doc1" in result.citations

    def test_generate_formats_context_with_sources(self):
        """Context includes source labels for citation."""
        config = {"model": "llama3:8b"}
        mock_client = MagicMock()
        mock_client.generate.return_value = "Answer"

        docs = [
            Document(content="Fact A", metadata={"source": "source1.txt"}, score=0.9),
            Document(content="Fact B", metadata={"source": "source2.pdf"}, score=0.8)
        ]

        technique = ContextGenerationTechnique(config, ollama_client=mock_client)
        technique.generate("Query", docs)

        call_args = mock_client.generate.call_args
        prompt = call_args[0][0]
        assert "[1] source1.txt:" in prompt
        assert "[2] source2.pdf:" in prompt