"""Tests for Answer dataclass."""

from utils.answer import Answer
from dataclasses import asdict


class TestAnswer:
    """Test Answer dataclass for generation responses."""

    def test_answer_creation(self):
        """Answer can be created with content and metadata."""
        answer = Answer(
            content="Paris is the capital of France.",
            metadata={"source": "test", "model": "llama3"},
            citations=["doc1", "doc2"]
        )
        assert answer.content == "Paris is the capital of France."
        assert answer.metadata["source"] == "test"
        assert answer.citations == ["doc1", "doc2"]

    def test_answer_with_empty_metadata(self):
        """Answer can be created with minimal metadata."""
        answer = Answer(content="Test answer")
        assert answer.content == "Test answer"
        assert answer.metadata == {}
        assert answer.citations == []

    def test_answer_to_dict(self):
        """Answer can be converted to dictionary."""
        answer = Answer(
            content="Test",
            metadata={"key": "value"},
            citations=["doc1"]
        )
        result = asdict(answer)
        assert result["content"] == "Test"
        assert result["metadata"] == {"key": "value"}
        assert result["citations"] == ["doc1"]