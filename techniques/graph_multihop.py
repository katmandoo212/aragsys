"""Graph-based multi-hop reasoning technique."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.document import Document


@dataclass
class GraphMultiHopTechnique:
    """Retrieve documents by finding paths between entities in query."""

    def __init__(self, config: dict, neo4j_store=None):
        self.config = config
        self.max_hops = config.get("max_hops", 3)
        self.top_k = config.get("top_k", 5)
        self.neo4j_store = neo4j_store

    def retrieve(self, query: str) -> list["Document"]:
        """Retrieve documents by finding multi-hop paths between entities."""
        if not query or not self.neo4j_store:
            return []

        # Find entities in query
        entities = self.neo4j_store.find_entities_in_query(query)
        if len(entities) < 2:
            return []

        # Extract entity IDs
        entity_ids = [e["id"] for e in entities[:self.top_k]]

        # Find multi-hop paths
        paths = self.neo4j_store.multi_hop_query(entity_ids, max_hops=self.max_hops)
        if not paths:
            return []

        # Convert to Document objects with path metadata
        from utils.document import Document
        return [
            Document(
                content=path["content"],
                metadata={"path_length": path["path_length"], "doc_id": path["doc_id"]},
                score=1.0 / (1 + path["path_length"])  # Shorter paths get higher score
            )
            for path in paths[:self.top_k]
        ]