"""Tests for EntityExtractor."""

import pytest
from utils.entity_extractor import EntityExtractor


class TestEntityExtractorBasicExtraction:
    """Test basic entity extraction."""

    def test_extract_person_entities(self):
        """Extract person names from text."""
        extractor = EntityExtractor()
        text = "John Smith and Jane Doe are researchers."
        entities = extractor.extract(text)
        persons = [e for e in entities if e["label"] == "PERSON"]
        assert len(persons) >= 2
        assert "John Smith" in [e["text"] for e in persons]

    def test_extract_organization_entities(self):
        """Extract organization names from text."""
        extractor = EntityExtractor()
        text = "Acme Corp and Tech Industries are companies."
        entities = extractor.extract(text)
        orgs = [e for e in entities if e["label"] == "ORG"]
        assert len(orgs) >= 1
        assert any("Acme" in e["text"] for e in orgs)

    def test_extract_location_entities(self):
        """Extract location entities from text."""
        extractor = EntityExtractor()
        text = "The conference was held in New York and Paris."
        entities = extractor.extract(text)
        locations = [e for e in entities if e["label"] in ["GPE", "LOC"]]
        assert len(locations) >= 1


class TestEntityExtractorEdgeCases:
    """Test edge cases for entity extraction."""

    def test_empty_text(self):
        """Empty text returns empty list."""
        extractor = EntityExtractor()
        entities = extractor.extract("")
        assert entities == []

    def test_no_entities(self):
        """Text with no entities returns empty list."""
        extractor = EntityExtractor()
        entities = extractor.extract("This is just some text without entities.")
        assert len(entities) == 0

    def test_whitespace_only(self):
        """Whitespace-only text returns empty list."""
        extractor = EntityExtractor()
        entities = extractor.extract("   \n\n   ")
        assert entities == []