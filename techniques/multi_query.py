"""Multi-Query Retrieval technique."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ollama.client import OllamaClient
    from stores.neo4j_store import Neo4jStore


@dataclass
class MultiQueryTechnique:
    """Retrieval using multiple query variations."""

    config: dict
    ollama_client: "OllamaClient"
    vector_store: "Neo4jStore"

    def __post_init__(self):
        """Initialize MultiQuery technique."""
        self.embedding_model = self.config.get("embedding_model", "bge-m3:latest")
        self.generation_model = self.config.get("generation_model", "glm-4.7:cloud")
        self.num_queries = self.config.get("num_queries", 3)
        self.top_k = self.config.get("top_k", 5)

    def retrieve(self, query: str) -> list:
        """Retrieve relevant documents using multiple queries."""
        # Step 1: Generate multiple query variations
        queries = self._generate_queries(query)

        # Step 2: Retrieve for each query
        all_results = {}
        for q in queries:
            query_vector = self.ollama_client.embed(q, self.embedding_model)
            results = self.vector_store.vector_search(query_vector, self.top_k)

            for result in results:
                doc_id = result["metadata"].get("id", result["content"])
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        "content": result["content"],
                        "metadata": result["metadata"],
                        "scores": []
                    }
                score = 1.0 - min(result["distance"], 1.0)
                all_results[doc_id]["scores"].append(score)

        # Step 3: Aggregate and rank
        documents = []
        for doc_data in all_results.values():
            avg_score = sum(doc_data["scores"]) / len(doc_data["scores"])
            from utils.document import Document
            documents.append(Document(
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                score=avg_score
            ))

        return sorted(documents, key=lambda d: -d.score)[:self.top_k]

    def _generate_queries(self, query: str) -> list[str]:
        """Generate multiple query variations."""
        prompt = f"Generate {self.num_queries} diverse search queries for: {query}"
        response = self.ollama_client.generate(prompt, self.generation_model)
        # Split by newlines to get individual queries
        return [q.strip() for q in response.split("\n") if q.strip()]