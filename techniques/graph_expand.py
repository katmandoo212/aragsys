"""Graph-based entity relationship expansion technique."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.document import Document


@dataclass
class GraphExpandTechnique:
    """Retrieve documents by expanding from entities to their relationships."""

    def __init__(self, config: dict, neo4j_store=None):
        self.config = config
        self.max_hops = config.get("max_hops", 2)
        self.min_doc_count = config.get("min_doc_count", 1)
        self.neo4j_store = neo4j_store

    def retrieve(self, query: str) -> list["Document"]:
        """Retrieve documents by expanding entity relationships."""
        if not query or not self.neo4j_store:
            return []

        # Find entities in query
        entities = self.neo4j_store.find_entities_in_query(query)
        if not entities:
            return []

        # Expand to related entities
        all_docs = []
        seen_doc_ids = set()

        for entity in entities:
            # Get related entities
            related = self.neo4j_store.get_entity_relationships(
                entity["id"], max_hops=self.max_hops
            )
            related = [r for r in related if r["doc_count"] >= self.min_doc_count]

            # Get documents from related entities
            for rel_entity in related:
                docs = self.neo4j_store.get_connected_documents(
                    rel_entity["id"], max_hops=1
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
                score=1.0
            )
            for doc in all_docs
        ]