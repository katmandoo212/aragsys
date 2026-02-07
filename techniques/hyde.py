"""HyDE (Hypothetical Document Embeddings) technique."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ollama.client import OllamaClient
    from stores.neo4j_store import Neo4jStore


@dataclass
class HyDETechnique:
    """Retrieval using Hypothetical Document Embeddings."""

    config: dict
    ollama_client: "OllamaClient"
    vector_store: "Neo4jStore"

    def __post_init__(self):
        """Initialize HyDE technique."""
        self.embedding_model = self.config.get("embedding_model", "bge-m3:latest")
        self.generation_model = self.config.get("generation_model", "glm-4.7:cloud")
        self.top_k = self.config.get("top_k", 5)

    def retrieve(self, query: str) -> list:
        """Retrieve relevant documents using HyDE."""
        # Step 1: Generate hypothetical answer
        hypothetical = self._generate_hypothetical(query)

        # Step 2: Embed hypothetical answer
        hypothetical_vector = self.ollama_client.embed(hypothetical, self.embedding_model)

        # Step 3: Retrieve using hypothetical embedding
        results = self.vector_store.vector_search(hypothetical_vector, self.top_k)

        # Step 4: Convert to Document objects
        documents = []
        for result in results:
            from utils.document import Document
            score = 1.0 - min(result["distance"], 1.0)
            documents.append(Document(
                content=result["content"],
                metadata=result["metadata"],
                score=score
            ))

        return documents

    def _generate_hypothetical(self, query: str) -> str:
        """Generate hypothetical answer for the query."""
        prompt = f"Generate a brief, factual answer to this question: {query}"
        return self.ollama_client.generate(prompt, self.generation_model)