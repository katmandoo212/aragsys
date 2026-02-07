from dataclasses import dataclass
import chromadb

@dataclass
class VectorStore:
    config: dict

    def __post_init__(self):
        self.client = chromadb.PersistentClient(
            path=self.config["persist_directory"]
        )
        self.collection = self.client.get_or_create_collection(
            name=self.config["collection_name"]
        )

    def add_documents(self, chunks: list[tuple[str, dict, list[float]]]) -> None:
        """Add documents with embeddings to ChromaDB."""
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for idx, (content, metadata, embedding) in enumerate(chunks):
            ids.append(f"doc_{idx}")
            documents.append(content)
            metadatas.append(metadata)
            embeddings.append(embedding)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        """Search for similar documents using the query vector."""
        if not query_vector:
            return []

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        # Format results
        formatted_results = []
        for idx in range(len(results["ids"][0])):
            formatted_results.append({
                "content": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "distance": results.get("distances", [[0]])[0][idx] if "distances" in results else 0.0
            })

        return formatted_results