"""Tests for EntityExtractor."""

import sys
import pytest
from unittest.mock import patch
from utils.entity_extractor import EntityExtractor


class TestEntityExtractorEdgeCases:
    """Test edge cases for entity extraction."""

    def test_empty_text(self):
        """Empty text returns empty list."""
        extractor = EntityExtractor()
        entities = extractor.extract("")
        assert entities == []

    def test_whitespace_only(self):
        """Whitespace-only text returns empty list."""
        extractor = EntityExtractor()
        entities = extractor.extract("   \n\n   ")
        assert entities == []

    @patch.dict("sys.modules", {"spacy": None})
    def test_spacy_not_available(self):
        """If spaCy is not available, return empty list."""
        extractor = EntityExtractor()
        text = "John Smith is a researcher."
        entities = extractor.extract(text)
        assert entities == []

    def test_initializes_with_defaults(self):
        """EntityExtractor initializes with default entity types."""
        extractor = EntityExtractor()
        assert extractor.entity_types == ["PERSON", "ORG", "GPE"]
        assert extractor.model_name == "en_core_web_sm"