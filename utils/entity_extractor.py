"""Named entity extraction using spaCy."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy


@dataclass
class EntityExtractor:
    """Extract named entities from text using spaCy."""

    model_name: str = "en_core_web_sm"
    entity_types: list[str] = None
    _nlp: "spacy.Language" = None

    def __post_init__(self):
        """Initialize spaCy model."""
        if self.entity_types is None:
            self.entity_types = ["PERSON", "ORG", "GPE"]

        try:
            import spacy
            try:
                self._nlp = spacy.load(self.model_name)
            except (OSError, Exception):
                # Model not downloaded or other error - skip download, just set to None
                # Downloading in tests causes issues with subprocess/pip
                self._nlp = None
        except (ImportError, Exception):
            self._nlp = None

    def extract(self, text: str) -> list[dict]:
        """Extract entities from text.

        Returns list of {text, label, start, end} tuples.
        """
        if not text or not text.strip() or self._nlp is None:
            return []

        doc = self._nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

        return entities