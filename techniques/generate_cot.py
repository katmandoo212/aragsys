"""Chain-of-thought LLM generation technique."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ollama.client import OllamaClient
    from utils.answer import Answer
    from utils.document import Document


@dataclass
class ChainOfThoughtGenerationTechnique:
    """Generate answers using chain-of-thought reasoning."""

    def __init__(self, config: dict, ollama_client=None):
        self.model = config.get("model", "glm-4.7:cloud")
        self.max_context_docs = config.get("max_context_docs", 5)
        self.ollama_client = ollama_client

    def generate(self, query: str, documents: list["Document"]) -> "Answer":
        """Generate an answer using step-by-step reasoning."""
        if not query or not self.ollama_client:
            from utils.answer import Answer
            return Answer(content="")

        # Build context
        context = self._build_context(documents[:self.max_context_docs])

        # Build CoT prompt
        prompt = f"""Query: {query}

Context:
{context}

Think step by step to answer the query. Show your reasoning process, then provide the final answer.

Reasoning:"""

        # Generate response
        response = self.ollama_client.generate(prompt, self.model)

        # Extract answer from CoT format
        answer = self._extract_answer(response)

        from utils.answer import Answer
        return Answer(content=answer.strip())

    def _build_context(self, documents: list["Document"]) -> str:
        """Build context string from documents."""
        if not documents:
            return "No context available."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", f"doc_{i}")
            context_parts.append(f"[{i}] {source}: {doc.content}")

        return "\n\n".join(context_parts)

    def _extract_answer(self, response: str) -> str:
        """Extract final answer from CoT response."""
        # Look for "Answer:" or "Final Answer:" section
        if "Answer:" in response:
            return response.split("Answer:")[-1]
        elif "Final Answer:" in response:
            return response.split("Final Answer:")[-1]
        return response