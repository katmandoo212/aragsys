"""Graph-based entity retrieval technique."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.document import Document


@dataclass
class GraphEntityTechnique:
    """Retrieve documents by matching entities in query."""

    def __init__(self, config: dict, neo4j_store=None):
        self.config = config
        self.max_hops = config.get("max_hops", 2)
        self.top_k = config.get("top_k", 5)
        self.neo4j_store = neo4j_store

    def retrieve(self, query: str) -> list["Document"]:
        """Retrieve documents by finding entities in query and fetching related documents."""
        if not query or not self.neo4j_store:
            return []

        # Find entities mentioned in query
        entities = self.neo4j_store.find_entities_in_query(query)
        if not entities:
            return []

        # Collect documents from all entities
        all_docs = []
        seen_doc_ids = set()

        for entity in entities[:self.top_k]:  # Limit entities to consider
            docs = self.neo4j_store.get_connected_documents(
                entity["id"], max_hops=self.max_hops
            )
            for doc in docs:
                if doc["doc_id"] not in seen_doc_ids:
                    seen_doc_ids.add(doc["doc_id"])
                    all_docs.append(doc)

        # Convert to Document objects
        from utils.document import Document
        return [
            Document(
                content=doc["content"],
                metadata=doc["metadata"],
                score=1.0  # Graph-based retrieval uses implicit relevance
            )
            for doc in all_docs[:self.top_k]
        ]