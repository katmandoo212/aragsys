"""Neo4j-based storage for documents, vectors, and entities."""

from dataclasses import dataclass
from typing import Any
from neo4j import GraphDatabase


@dataclass
class Neo4jStore:
    """Neo4j storage backend with vector and full-text search."""

    config: dict

    def __post_init__(self):
        """Initialize Neo4j driver."""
        neo4j_config = self.config["neo4j"]
        self.driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["user"], neo4j_config["password"])
        )
        self.database = neo4j_config.get("database", "neo4j")
        self.vector_index_name = self.config["vector_index"]["name"]
        self.vector_dimension = self.config["vector_index"]["dimension"]
        self.fulltext_index_name = self.config["fulltext_index"]["name"]

    def add_documents(self, chunks: list[tuple[str, dict, list[float]]]) -> None:
        """Add documents with embeddings to Neo4j."""
        if not chunks:
            return

        with self.driver.session(database=self.database) as session:
            for idx, (content, metadata, embedding) in enumerate(chunks):
                doc_id = f"doc_{idx}"
                query = """
                MERGE (d:Document {id: $doc_id})
                SET d.content = $content,
                    d.metadata = $metadata,
                    d.embedding = $embedding
                """
                session.run(query, doc_id=doc_id, content=content, metadata=metadata, embedding=embedding)

    def add_entities(self, doc_id: str, entities: list[dict]) -> None:
        """Add entities and link to document."""
        if not entities:
            return

        with self.driver.session(database=self.database) as session:
            for entity in entities:
                query = """
                MERGE (e:Entity {id: $entity_id})
                SET e.name = $name, e.type = $type
                MERGE (d:Document {id: $doc_id})
                MERGE (d)-[:CONTAINS]->(e)
                """
                entity_id = f"{doc_id}_entity_{entity['start']}_{entity['end']}"
                session.run(
                    query,
                    entity_id=entity_id,
                    name=entity["text"],
                    type=entity["label"],
                    doc_id=doc_id
                )

    def vector_search(self, query_vector: list[float], top_k: int) -> list[dict]:
        """Search using vector similarity."""
        if not query_vector:
            return []

        with self.driver.session(database=self.database) as session:
            query = f"""
            CALL db.index.vector.queryNodes('{self.vector_index_name}', $top_k, $query_vector)
            YIELD node, score
            RETURN node.content as content, node.metadata as metadata, score
            LIMIT $top_k
            """
            result = session.run(query, query_vector=query_vector, top_k=top_k)
            return [
                {
                    "content": record["content"],
                    "metadata": record["metadata"],
                    "distance": 1.0 - record["score"]  # Convert similarity to distance
                }
                for record in result
            ]

    def fulltext_search(self, query: str, top_k: int) -> list[dict]:
        """Search using full-text index."""
        if not query:
            return []

        with self.driver.session(database=self.database) as session:
            query_str = f"""
            CALL db.index.fulltext.queryNodes('{self.fulltext_index_name}', $query)
            YIELD node, score
            RETURN node.content as content, node.metadata as metadata, score
            LIMIT $top_k
            """
            result = session.run(query_str, query=query)
            return [
                {
                    "content": record["content"],
                    "metadata": record["metadata"],
                    "distance": 1.0 - record["score"]
                }
                for record in result
            ]

    def hybrid_search(self, query: str, query_vector: list[float], top_k: int) -> list[dict]:
        """Combine vector and full-text search with RRF."""
        rrf_k = self.config.get("rrf_k", 60)

        dense_results = self.vector_search(query_vector, top_k)
        sparse_results = self.fulltext_search(query, top_k)

        # Create rank maps
        dense_ranks = {r["content"]: i + 1 for i, r in enumerate(dense_results)}
        sparse_ranks = {r["content"]: i + 1 for i, r in enumerate(sparse_results)}

        # Combine with RRF
        merged = {}
        for content in set(list(dense_ranks.keys()) + list(sparse_ranks.keys())):
            rank_dense = dense_ranks.get(content, top_k)
            rank_sparse = sparse_ranks.get(content, top_k)
            rrf_score = 1.0 / (rrf_k + rank_dense) + 1.0 / (rrf_k + rank_sparse)
            merged[content] = (rrf_score, content)

        # Return sorted by combined score
        sorted_results = sorted(merged.items(), key=lambda x: -x[1][0])[:top_k]

        return [
            {
                "content": content,
                "metadata": {},  # Would need to fetch from actual node
                "score": score
            }
            for score, content in sorted_results
        ]

    def find_entities_in_query(self, query: str) -> list[dict]:
        """Find entities mentioned in query."""
        # For now, return empty list - will integrate EntityExtractor later
        # This allows us to test the Neo4j query structure first
        return []

    def get_connected_documents(self, entity_id: str, max_hops: int = 1) -> list[dict]:
        """Get documents connected to an entity within max_hops distance."""
        if max_hops < 1:
            return []

        with self.driver.session(database=self.database) as session:
            # Variable-length path query for documents within max_hops
            query = f"""
            MATCH (d:Document)-[:CONTAINS*1..{max_hops}]-(e:Entity {{id: $entity_id}})
            RETURN d.content as content, d.metadata as metadata, elementId(d) as doc_id
            """
            result = session.run(query, entity_id=entity_id)
            return [
                {
                    "content": record["content"],
                    "metadata": record["metadata"],
                    "doc_id": record["doc_id"]
                }
                for record in result.data()
            ]

    def get_entity_relationships(self, entity_id: str, max_hops: int = 2) -> list[dict]:
        """Get entities connected via shared documents within max_hops distance."""
        if max_hops < 1:
            return []

        with self.driver.session(database=self.database) as session:
            # Find entities connected through shared documents
            query = f"""
            MATCH path = (e1:Entity {{id: $entity_id}})-[*1..{max_hops}]-(d:Document)-[:CONTAINS]-(e2:Entity)
            WHERE e1 <> e2 AND length(path) <= {max_hops}
            RETURN e2.name as name, e2.type as type, e2.id as id, count(DISTINCT d) as doc_count
            ORDER BY doc_count DESC
            LIMIT 20
            """
            result = session.run(query, entity_id=entity_id)
            return [
                {
                    "name": record["name"],
                    "type": record["type"],
                    "id": record["id"],
                    "doc_count": record["doc_count"]
                }
                for record in result.data()
            ]