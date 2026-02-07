"""Hybrid (Dense + Sparse) Retrieval technique."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ollama.client import OllamaClient
    from stores.neo4j_store import Neo4jStore


@dataclass
class HybridTechnique:
    """Retrieval using hybrid dense + sparse search."""

    config: dict
    ollama_client: "OllamaClient"
    vector_store: "Neo4jStore"

    def __post_init__(self):
        """Initialize Hybrid technique."""
        self.embedding_model = self.config.get("embedding_model", "bge-m3:latest")
        self.top_k = self.config.get("top_k", 10)
        self.rrf_k = self.config.get("rrf_k", 60)

    def retrieve(self, query: str) -> list:
        """Retrieve relevant documents using hybrid search."""
        # Step 1: Get query embedding
        query_vector = self.ollama_client.embed(query, self.embedding_model)

        # Step 2: Perform hybrid search in store
        results = self.vector_store.hybrid_search(query, query_vector, self.top_k)

        # Step 3: Convert to Document objects
        documents = []
        for result in results:
            from utils.document import Document
            documents.append(Document(
                content=result["content"],
                metadata=result["metadata"],
                score=result["score"]
            ))

        return documents