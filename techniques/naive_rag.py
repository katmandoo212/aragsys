from dataclasses import dataclass
from typing import TYPE_CHECKING

from ollama.client import OllamaClient
from utils.vector_store import VectorStore

if TYPE_CHECKING:
    from utils.document import Document


@dataclass
class NaiveRAGTechnique:
    def __init__(self, config: dict, ollama_client: OllamaClient = None, vector_store: VectorStore = None):
        self.config = config
        self.embedding_model = config.get("embedding_model", "bge-m3:latest")
        self.top_k = config.get("top_k", 5)
        self.collection_name = config.get("collection_name", "documents")

        self.ollama_client = ollama_client
        self.vector_store = vector_store

    def retrieve(self, query: str) -> list["Document"]:
        """Retrieve relevant documents for the given query."""
        # Embed the query
        query_vector = self._embed_query(query)

        # Search the store
        results = self.vector_store.search(query_vector, self.top_k)

        # Convert to Document objects
        documents = []
        for result in results:
            # Use distance as inverse score
            score = 1.0 - min(result["distance"], 1.0)

            from utils.document import Document
            documents.append(Document(
                content=result["content"],
                metadata=result["metadata"],
                score=score
            ))

        return documents

    def _embed_query(self, query: str) -> list[float]:
        """Embed the query using Ollama."""
        return self.ollama_client.embed(query, self.embedding_model)