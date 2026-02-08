"""Context-aware LLM generation with citations."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ollama.client import OllamaClient
    from utils.answer import Answer
    from utils.document import Document


@dataclass
class ContextGenerationTechnique:
    """Generate answers with citation formatting."""

    def __init__(self, config: dict, ollama_client=None):
        self.config = config
        self.model = config.get("model", "glm-4.7:cloud")
        self.max_context_docs = config.get("max_context_docs", 5)
        self.ollama_client = ollama_client

    def generate(self, query: str, documents: list["Document"]) -> "Answer":
        """Generate an answer with citations from context."""
        if not query or not self.ollama_client:
            from utils.answer import Answer
            return Answer(content="")

        # Build context with citation markers
        context, citations = self._build_context_with_citations(documents[:self.max_context_docs])

        # Build prompt
        prompt = f"""Query: {query}

Use the following context to answer. Cite your sources using [n] notation where n is the reference number.

Context:
{context}

Answer:"""

        # Generate response
        response = self.ollama_client.generate(prompt, self.model)

        from utils.answer import Answer
        return Answer(content=response.strip(), citations=citations)

    def _build_context_with_citations(self, documents: list["Document"]) -> tuple[str, list[str]]:
        """Build context with citation markers and return citations list."""
        if not documents:
            return "No context available.", []

        context_parts = []
        citations = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", f"doc_{i}")
            citations.append(source)
            context_parts.append(f"[{i}] {source}: {doc.content}")

        return "\n\n".join(context_parts), citations