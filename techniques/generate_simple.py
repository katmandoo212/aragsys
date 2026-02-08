"""Simple LLM generation technique."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ollama.client import OllamaClient
    from utils.answer import Answer
    from utils.document import Document


@dataclass
class SimpleGenerationTechnique:
    """Generate answers using basic LLM prompting."""

    def __init__(self, config: dict, ollama_client=None):
        self.config = config
        self.model = config.get("model", "glm-4.7:cloud")
        self.max_context_docs = config.get("max_context_docs", 5)
        self.ollama_client = ollama_client

    def generate(self, query: str, documents: list["Document"]) -> "Answer":
        """Generate an answer based on query and retrieved documents."""
        if not query or not self.ollama_client:
            from utils.answer import Answer
            return Answer(content="")

        # Build prompt with context
        context = self._build_context(documents[:self.max_context_docs])
        prompt = f"""Query: {query}

Context:
{context}

Answer:"""

        # Generate response
        response = self.ollama_client.generate(prompt, self.model)

        from utils.answer import Answer
        return Answer(content=response.strip())

    def _build_context(self, documents: list["Document"]) -> str:
        """Build context string from documents."""
        if not documents:
            return "No context available."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", f"doc_{i}")
            context_parts.append(f"[{i}] {source}: {doc.content}")

        return "\n\n".join(context_parts)