"""Answer dataclass for generation responses."""

from dataclasses import dataclass, field


@dataclass
class Answer:
    """Response from LLM generation."""

    content: str
    metadata: dict = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)