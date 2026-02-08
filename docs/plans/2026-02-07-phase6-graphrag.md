# Phase 6 - GraphRAG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build multi-hop reasoning capabilities using Neo4j graph traversal for entity relationship queries and knowledge graph path finding.

**Architecture:** Leverage existing Neo4jStore foundation with Document and Entity nodes. Add GraphRAG techniques that traverse entity relationships (PERSON, ORG, GPE) to find multi-hop connections, then merge results with vector retrieval for context-rich answers.

**Tech Stack:** Python 3.13+, Neo4j Python driver (already installed), pytest (already installed), existing Neo4jStore, existing EntityExtractor.

---

## Prerequisites

**Before starting:**

```bash
# Ensure we're in main branch
git checkout master
git pull origin master

# Create feature branch
git checkout -b feature/phase6-graphrag

# Install dependencies (should already be installed)
uv sync
```

---

## Task 1: Neo4jStore - Find entities in query

**Files:**
- Modify: `stores/neo4j_store.py`
- Test: `tests/test_neo4j_store.py`

**Step 1: Write the failing test**

```python
# tests/test_neo4j_store.py (add to existing file)

def test_find_entities_in_query():
    """Extract entity mentions from query and find matching nodes."""
    from unittest.mock import MagicMock
    from stores.neo4j_store import Neo4jStore

    config = {
        "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password", "database": "rag"},
        "vector_index": {"name": "doc_emb", "dimension": 1024},
        "fulltext_index": {"name": "doc_ft"}
    }

    store = Neo4jStore(config)
    store.driver = MagicMock()

    # Mock session and result
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.data.return_value = [
        {"name": "John Smith", "type": "PERSON", "id": "ent1"},
        {"name": "MIT", "type": "ORG", "id": "ent2"}
    ]
    mock_session.run.return_value = mock_result
    store.driver.session.return_value.__enter__.return_value = mock_session

    # Test entity finding (for now, returns empty until EntityExtractor integrated)
    entities = store.find_entities_in_query("John Smith worked at MIT")
    # Initially returns empty list - will integrate EntityExtractor later
    assert entities == []

    mock_session.run.assert_called_once()
    call_args = mock_session.run.call_args[0][0]
    assert "Entity" in call_args
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_neo4j_store.py::test_find_entities_in_query -v`
Expected: FAIL with "Neo4jStore has no attribute 'find_entities_in_query'"

**Step 3: Write minimal implementation**

```python
# stores/neo4j_store.py (add to existing file)

    def find_entities_in_query(self, query: str) -> list[dict]:
        """Find entities mentioned in query."""
        # For now, return empty list - will integrate EntityExtractor later
        # This allows us to test the Neo4j query structure first
        return []
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_neo4j_store.py::test_find_entities_in_query -v`
Expected: PASS

**Step 5: Commit**

```bash
git add stores/neo4j_store.py tests/test_neo4j_store.py
git commit -m "feat: add find_entities_in_query stub to Neo4jStore

- Add method for finding entities in query
- Returns empty list for now (EntityExtractor integration later)
- Add test for Neo4j query structure"
```

---

## Task 2: Neo4jStore - Get connected documents

**Files:**
- Modify: `stores/neo4j_store.py`
- Test: `tests/test_neo4j_store.py`

**Step 1: Write the failing test**

```python
# tests/test_neo4j_store.py (add to existing file)

def test_get_connected_documents():
    """Get documents connected to an entity."""
    from unittest.mock import MagicMock
    from stores.neo4j_store import Neo4jStore

    config = {
        "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password", "database": "rag"},
        "vector_index": {"name": "doc_emb", "dimension": 1024},
        "fulltext_index": {"name": "doc_ft"}
    }

    store = Neo4jStore(config)
    store.driver = MagicMock()

    # Mock session and result
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.data.return_value = [
        {"content": "John Smith is a researcher.", "metadata": {"source": "doc1"}, "doc_id": "doc1"},
        {"content": "Smith published on ML.", "metadata": {"source": "doc2"}, "doc_id": "doc2"}
    ]
    mock_session.run.return_value = mock_result
    store.driver.session.return_value.__enter__.return_value = mock_session

    docs = store.get_connected_documents("ent1", max_hops=1)
    assert len(docs) == 2
    assert docs[0]["content"] == "John Smith is a researcher."
    assert docs[0]["doc_id"] == "doc1"

    # Verify Cypher query
    call_args = mock_session.run.call_args[0][0]
    assert "MATCH" in call_args
    assert "Entity" in call_args
    assert "Document" in call_args
    assert "CONTAINS" in call_args
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_neo4j_store.py::test_get_connected_documents -v`
Expected: FAIL with "Neo4jStore has no attribute 'get_connected_documents'"

**Step 3: Write minimal implementation**

```python
# stores/neo4j_store.py (add to existing file)

    def get_connected_documents(self, entity_id: str, max_hops: int = 1) -> list[dict]:
        """Get documents connected to an entity within max_hops distance."""
        if max_hops < 1:
            return []

        with self.driver.session(database=self.database) as session:
            # Variable-length path query for documents within max_hops
            query = f"""
            MATCH (d:Document)-[:CONTAINS*1..{max_hops}]-(e:Entity {{id: $entity_id}})
            RETURN d.content as content, d.metadata as metadata, elementId(d) as doc_id
            LIMIT 50
            """
            result = session.run(query, entity_id=entity_id)
            return [
                {
                    "content": record["content"],
                    "metadata": record["metadata"],
                    "doc_id": record["doc_id"]
                }
                for record in result
            ]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_neo4j_store.py::test_get_connected_documents -v`
Expected: PASS

**Step 5: Commit**

```bash
git add stores/neo4j_store.py tests/test_neo4j_store.py
git commit -m "feat: add get_connected_documents to Neo4jStore

- Query documents connected to entity via CONTAINS relationship
- Support variable-length paths with max_hops parameter
- Return content, metadata, and doc_id for each document"
```

---

## Task 3: Neo4jStore - Get entity relationships

**Files:**
- Modify: `stores/neo4j_store.py`
- Test: `tests/test_neo4j_store.py`

**Step 1: Write the failing test**

```python
# tests/test_neo4j_store.py (add to existing file)

def test_get_entity_relationships():
    """Get entities related to a given entity."""
    from unittest.mock import MagicMock
    from stores.neo4j_store import Neo4jStore

    config = {
        "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password", "database": "rag"},
        "vector_index": {"name": "doc_emb", "dimension": 1024},
        "fulltext_index": {"name": "doc_ft"}
    }

    store = Neo4jStore(config)
    store.driver = MagicMock()

    # Mock session and result
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.data.return_value = [
        {"name": "MIT", "type": "ORG", "id": "ent2", "doc_count": 2},
        {"name": "Stanford", "type": "ORG", "id": "ent3", "doc_count": 1}
    ]
    mock_session.run.return_value = mock_result
    store.driver.session.return_value.__enter__.return_value = mock_session

    relations = store.get_entity_relationships("ent1", max_hops=2)
    assert len(relations) == 2
    assert relations[0]["name"] == "MIT"
    assert relations[0]["type"] == "ORG"
    assert relations[0]["doc_count"] == 2

    # Verify Cypher query
    call_args = mock_session.run.call_args[0][0]
    assert "MATCH" in call_args
    assert "Entity" in call_args
    assert "Document" in call_args
    assert "length(path)" in call_args
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_neo4j_store.py::test_get_entity_relationships -v`
Expected: FAIL with "Neo4jStore has no attribute 'get_entity_relationships'"

**Step 3: Write minimal implementation**

```python
# stores/neo4j_store.py (add to existing file)

    def get_entity_relationships(self, entity_id: str, max_hops: int = 2) -> list[dict]:
        """Get entities connected via shared documents within max_hops distance."""
        if max_hops < 1:
            return []

        with self.driver.session(database=self.database) as session:
            # Find entities connected through shared documents
            query = f"""
            MATCH (e1:Entity {{id: $entity_id}})-[*1..{max_hops}]-(d:Document)-[:CONTAINS]-(e2:Entity)
            WHERE e1 <> e2
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
                for record in result
            ]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_neo4j_store.py::test_get_entity_relationships -v`
Expected: PASS

**Step 5: Commit**

```bash
git add stores/neo4j_store.py tests/test_neo4j_store.py
git commit -m "feat: add get_entity_relationships to Neo4jStore

- Find entities connected through shared documents
- Support multi-hop connections with max_hops parameter
- Return related entities with document count for ranking"
```

---

## Task 4: Neo4jStore - Multi-hop path query

**Files:**
- Modify: `stores/neo4j_store.py`
- Test: `tests/test_neo4j_store.py`

**Step 1: Write the failing test**

```python
# tests/test_neo4j_store.py (add to existing file)

def test_multi_hop_query():
    """Find paths connecting multiple entities."""
    from unittest.mock import MagicMock
    from stores.neo4j_store import Neo4jStore

    config = {
        "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password", "database": "rag"},
        "vector_index": {"name": "doc_emb", "dimension": 1024},
        "fulltext_index": {"name": "doc_ft"}
    }

    store = Neo4jStore(config)
    store.driver = MagicMock()

    # Mock session and result
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.data.return_value = [
        {"doc_id": "doc1", "content": "John Smith at MIT", "path_length": 1},
        {"doc_id": "doc2", "content": "MIT collaboration with Stanford", "path_length": 2}
    ]
    mock_session.run.return_value = mock_result
    store.driver.session.return_value.__enter__.return_value = mock_session

    paths = store.multi_hop_query(["ent1", "ent2"], max_hops=3)
    assert len(paths) == 2
    assert paths[0]["doc_id"] == "doc1"
    assert paths[0]["path_length"] == 1

    # Verify Cypher query
    call_args = mock_session.run.call_args[0][0]
    assert "allShortestPaths" in call_args or "shortestPath" in call_args or "Entity" in call_args
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_neo4j_store.py::test_multi_hop_query -v`
Expected: FAIL with "Neo4jStore has no attribute 'multi_hop_query'"

**Step 3: Write minimal implementation**

```python
# stores/neo4j_store.py (add to existing file)

    def multi_hop_query(self, entity_ids: list[str], max_hops: int = 3) -> list[dict]:
        """Find documents and paths connecting multiple entities."""
        if len(entity_ids) < 2 or max_hops < 1:
            return []

        with self.driver.session(database=self.database) as session:
            # Find documents on paths connecting any pair of entities
            entity_ids_str = ", ".join([f"'{eid}'" for eid in entity_ids])
            query = f"""
            MATCH (e1:Entity)-[:CONTAINS*1..{max_hops}]-(d:Document)-[:CONTAINS*1..{max_hops}]-(e2:Entity)
            WHERE e1.id IN [{entity_ids_str}] AND e2.id IN [{entity_ids_str}] AND e1.id < e2.id
            RETURN elementId(d) as doc_id, d.content as content,
                   length(shortestPath((e1)-[*]-(e2))) as path_length
            ORDER BY path_length ASC
            LIMIT 30
            """
            result = session.run(query)
            return [
                {
                    "doc_id": record["doc_id"],
                    "content": record["content"],
                    "path_length": record["path_length"]
                }
                for record in result
            ]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_neo4j_store.py::test_multi_hop_query -v`
Expected: PASS

**Step 5: Commit**

```bash
git add stores/neo4j_store.py tests/test_neo4j_store.py
git commit -m "feat: add multi_hop_query to Neo4jStore

- Find documents on paths connecting multiple entities
- Support configurable max_hops for path depth
- Return documents sorted by shortest path length"
```

---

## Task 5: GraphRAG technique - Entity lookup

**Files:**
- Create: `techniques/graph_entity.py`
- Test: `tests/test_graph_entity_technique.py`

**Step 1: Write the failing test**

```python
# tests/test_graph_entity_technique.py (new file)

"""Tests for GraphEntityTechnique."""

import pytest
from unittest.mock import MagicMock, patch
from techniques.graph_entity import GraphEntityTechnique


class TestGraphEntityTechnique:
    """Test GraphEntityTechnique for entity-based retrieval."""

    def test_initialization(self):
        """GraphEntityTechnique initializes with config."""
        config = {"max_hops": 2, "top_k": 5}
        technique = GraphEntityTechnique(config, neo4j_store=MagicMock())
        assert technique.max_hops == 2
        assert technique.top_k == 5

    def test_retrieve_with_empty_query(self):
        """Empty query returns empty results."""
        config = {"max_hops": 2, "top_k": 5}
        technique = GraphEntityTechnique(config, neo4j_store=MagicMock())
        results = technique.retrieve("")
        assert results == []

    def test_entity_lookup_calls_neo4j(self):
        """Entity lookup queries Neo4j for matching entities."""
        from utils.document import Document

        config = {"max_hops": 2, "top_k": 5}

        mock_store = MagicMock()
        mock_store.find_entities_in_query.return_value = [
            {"id": "ent1", "name": "John Smith", "type": "PERSON"}
        ]
        mock_store.get_connected_documents.return_value = [
            {"content": "John Smith's research", "metadata": {"source": "doc1"}, "doc_id": "doc1"}
        ]

        technique = GraphEntityTechnique(config, neo4j_store=mock_store)
        results = technique.retrieve("What did John Smith research?")

        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert results[0].content == "John Smith's research"
        mock_store.find_entities_in_query.assert_called_once_with("What did John Smith research?")
        mock_store.get_connected_documents.assert_called_once_with("ent1", max_hops=2)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_entity_technique.py -v`
Expected: FAIL with "No module named 'techniques.graph_entity'"

**Step 3: Write minimal implementation**

```python
# techniques/graph_entity.py (new file)

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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_entity_technique.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add techniques/graph_entity.py tests/test_graph_entity_technique.py
git commit -m "feat: add GraphEntityTechnique for entity-based retrieval

- Find entities in query and fetch connected documents
- Support configurable max_hops and top_k
- Deduplicate documents by doc_id"
```

---

## Task 6: GraphRAG technique - Multi-hop reasoning

**Files:**
- Create: `techniques/graph_multihop.py`
- Test: `tests/test_graph_multihop_technique.py`

**Step 1: Write the failing test**

```python
# tests/test_graph_multihop_technique.py (new file)

"""Tests for GraphMultiHopTechnique."""

import pytest
from unittest.mock import MagicMock
from techniques.graph_multihop import GraphMultiHopTechnique


class TestGraphMultiHopTechnique:
    """Test GraphMultiHopTechnique for multi-hop reasoning."""

    def test_initialization(self):
        """GraphMultiHopTechnique initializes with config."""
        config = {"max_hops": 3, "top_k": 5}
        technique = GraphMultiHopTechnique(config, neo4j_store=MagicMock())
        assert technique.max_hops == 3
        assert technique.top_k == 5

    def test_retrieve_with_no_entities(self):
        """No entities found returns empty results."""
        from utils.document import Document

        config = {"max_hops": 3, "top_k": 5}

        mock_store = MagicMock()
        mock_store.find_entities_in_query.return_value = []

        technique = GraphMultiHopTechnique(config, neo4j_store=mock_store)
        results = technique.retrieve("Query with no entities")

        assert results == []

    def test_multi_hop_queries_entity_paths(self):
        """Multi-hop technique queries paths between entities."""
        from utils.document import Document

        config = {"max_hops": 3, "top_k": 5}

        mock_store = MagicMock()
        mock_store.find_entities_in_query.return_value = [
            {"id": "ent1", "name": "MIT", "type": "ORG"},
            {"id": "ent2", "name": "Stanford", "type": "ORG"}
        ]
        mock_store.multi_hop_query.return_value = [
            {"doc_id": "doc1", "content": "MIT-Stanford collaboration", "path_length": 1},
            {"doc_id": "doc2", "content": "Research partnership", "path_length": 2}
        ]

        technique = GraphMultiHopTechnique(config, neo4j_store=mock_store)
        results = technique.retrieve("How are MIT and Stanford connected?")

        assert len(results) == 2
        assert results[0].content == "MIT-Stanford collaboration"
        assert results[0].metadata["path_length"] == 1

        mock_store.multi_hop_query.assert_called_once_with(
            ["ent1", "ent2"], max_hops=3
        )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_multihop_technique.py -v`
Expected: FAIL with "No module named 'techniques.graph_multihop'"

**Step 3: Write minimal implementation**

```python
# techniques/graph_multihop.py (new file)

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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_multihop_technique.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add techniques/graph_multihop.py tests/test_graph_multihop_technique.py
git commit -m "feat: add GraphMultiHopTechnique for multi-hop reasoning

- Find paths between entities in query
- Score results by path length (shorter = better)
- Return documents on multi-hop connections"
```

---

## Task 7: GraphRAG technique - Entity relationship expansion

**Files:**
- Create: `techniques/graph_expand.py`
- Test: `tests/test_graph_expand_technique.py`

**Step 1: Write the failing test**

```python
# tests/test_graph_expand_technique.py (new file)

"""Tests for GraphExpandTechnique."""

import pytest
from unittest.mock import MagicMock
from techniques.graph_expand import GraphExpandTechnique


class TestGraphExpandTechnique:
    """Test GraphExpandTechnique for entity relationship expansion."""

    def test_initialization(self):
        """GraphExpandTechnique initializes with config."""
        config = {"max_hops": 2, "top_k": 5, "min_doc_count": 1}
        technique = GraphExpandTechnique(config, neo4j_store=MagicMock())
        assert technique.max_hops == 2
        assert technique.top_k == 5
        assert technique.min_doc_count == 1

    def test_expand_queries_entity_relationships(self):
        """Expand technique queries related entities."""
        from utils.document import Document

        config = {"max_hops": 2, "top_k": 5, "min_doc_count": 1}

        mock_store = MagicMock()
        mock_store.find_entities_in_query.return_value = [
            {"id": "ent1", "name": "John Smith", "type": "PERSON"}
        ]
        mock_store.get_entity_relationships.return_value = [
            {"id": "ent2", "name": "MIT", "type": "ORG", "doc_count": 3},
            {"id": "ent3", "name": "Stanford", "type": "ORG", "doc_count": 1}
        ]
        mock_store.get_connected_documents.return_value = [
            {"content": "MIT research paper", "metadata": {"source": "doc1"}, "doc_id": "doc1"}
        ]

        technique = GraphExpandTechnique(config, neo4j_store=mock_store)
        results = technique.retrieve("What organizations did John Smith work with?")

        assert len(results) >= 1
        mock_store.get_entity_relationships.assert_called_once_with("ent1", max_hops=2)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_expand_technique.py -v`
Expected: FAIL with "No module named 'techniques.graph_expand'"

**Step 3: Write minimal implementation**

```python
# techniques/graph_expand.py (new file)

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
        self.top_k = config.get("top_k", 5)
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
            for rel_entity in related[:self.top_k]:
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
            for doc in all_docs[:self.top_k]
        ]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_expand_technique.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add techniques/graph_expand.py tests/test_graph_expand_technique.py
git commit -m "feat: add GraphExpandTechnique for entity relationship expansion

- Find entities in query, expand to related entities
- Support min_doc_count filter for weak relationships
- Fetch documents from expanded entity network"
```

---

## Task 8: Update techniques __init__.py exports

**Files:**
- Modify: `techniques/__init__.py`
- Test: `tests/test_registry.py` (verify imports work)

**Step 1: Write the failing test**

```python
# tests/test_registry.py (add to existing file)

def test_registry_imports_graph_techniques():
    """Verify graph techniques are importable."""
    from techniques import GraphEntityTechnique, GraphMultiHopTechnique, GraphExpandTechnique
    assert GraphEntityTechnique is not None
    assert GraphMultiHopTechnique is not None
    assert GraphExpandTechnique is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py::test_registry_imports_graph_techniques -v`
Expected: FAIL with "cannot import name 'GraphEntityTechnique'"

**Step 3: Write minimal implementation**

```python
# techniques/__init__.py (modify existing file)

"""RAG retrieval techniques."""

from techniques.naive_rag import NaiveRAGTechnique
from techniques.hyde import HyDETechnique
from techniques.multi_query import MultiQueryTechnique
from techniques.hybrid import HybridTechnique
from techniques.rerank import RerankTechnique
from techniques.compress import CompressTechnique
from techniques.graph_entity import GraphEntityTechnique
from techniques.graph_multihop import GraphMultiHopTechnique
from techniques.graph_expand import GraphExpandTechnique

__all__ = [
    "NaiveRAGTechnique",
    "HyDETechnique",
    "MultiQueryTechnique",
    "HybridTechnique",
    "RerankTechnique",
    "CompressTechnique",
    "GraphEntityTechnique",
    "GraphMultiHopTechnique",
    "GraphExpandTechnique",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py::test_registry_imports_graph_techniques -v`
Expected: PASS

**Step 5: Commit**

```bash
git add techniques/__init__.py tests/test_registry.py
git commit -m "feat: export Phase 6 graph techniques

- Add GraphEntityTechnique to exports
- Add GraphMultiHopTechnique to exports
- Add GraphExpandTechnique to exports
- Add test verifying imports work"
```

---

## Task 9: Update config/techniques.yaml

**Files:**
- Modify: `config/techniques.yaml`
- Test: `tests/test_neo4j_store.py` (verify config loads)

**Step 1: Add graph techniques to config**

```yaml
# config/techniques.yaml (add to existing file)

  graph_entity:
    class: techniques.graph_entity.GraphEntityTechnique
    enabled: true
    config:
      max_hops: 2
      top_k: 5

  graph_multihop:
    class: techniques.graph_multihop.GraphMultiHopTechnique
    enabled: true
    config:
      max_hops: 3
      top_k: 5

  graph_expand:
    class: techniques.graph_expand.GraphExpandTechnique
    enabled: true
    config:
      max_hops: 2
      top_k: 5
      min_doc_count: 2
```

**Step 2: Verify config loads**

Run: `python -c "import yaml; print(yaml.safe_load(open('config/techniques.yaml'))['techniques'].keys())"`
Expected: Shows graph_entity, graph_multihop, graph_expand keys

**Step 3: Commit**

```bash
git add config/techniques.yaml
git commit -m "feat: add Phase 6 graph techniques to config

- Add graph_entity technique for entity-based retrieval
- Add graph_multihop technique for multi-hop reasoning
- Add graph_expand technique for relationship expansion"
```

---

## Task 10: Create graphrag.yaml config

**Files:**
- Create: `config/graphrag.yaml`
- Test: `tests/test_graphrag_config.py` (new test file)

**Step 1: Write the failing test**

```python
# tests/test_graphrag_config.py (new file)

"""Tests for GraphRAG configuration."""

import pytest
import yaml


def test_graphrag_config_exists():
    """GraphRAG config file exists and is valid YAML."""
    with open("config/graphrag.yaml") as f:
        config = yaml.safe_load(f)
    assert "graph" in config
    assert "neo4j" in config


def test_graphrag_config_has_entity_types():
    """Config defines entity types for extraction."""
    with open("config/graphrag.yaml") as f:
        config = yaml.safe_load(f)
    assert "entity_types" in config["graph"]
    assert isinstance(config["graph"]["entity_types"], list)


def test_graphrag_config_has_defaults():
    """Config has sensible default values."""
    with open("config/graphrag.yaml") as f:
        config = yaml.safe_load(f)
    assert config["graph"]["max_hops"] > 0
    assert config["graph"]["top_k"] > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graphrag_config.py -v`
Expected: FAIL with "No such file or directory: 'config/graphrag.yaml'"

**Step 3: Write minimal implementation**

```yaml
# config/graphrag.yaml (new file)

graph:
  # Entity types to extract and index
  entity_types:
    - PERSON
    - ORG
    - GPE
    - EVENT
    - WORK_OF_ART
    - LAW

  # Default maximum hops for graph traversal
  max_hops: 3

  # Default number of results to return
  top_k: 5

  # Minimum document count for entity relationships
  min_doc_count: 2

  # Relationship expansion settings
  expand:
    max_related_entities: 10
    min_shared_documents: 1

neo4j:
  # Reference to Neo4j config (actual values in neo4j.yaml)
  config_ref: "neo4j.yaml"

  # Index names for graph queries
  entity_index: "entity_names"
  relationship_index: "entity_relationships"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graphrag_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add config/graphrag.yaml tests/test_graphrag_config.py
git commit -m "feat: add graphrag.yaml configuration

- Define entity types for extraction
- Set default max_hops and top_k values
- Configure expansion parameters
- Add config validation tests"
```

---

## Task 11: All tests pass

**Step 1: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: All tests pass (including Phase 6 tests)

**Step 2: Fix any failing tests if needed**

(If tests fail, debug and fix, then re-run)

**Step 3: Commit final state**

```bash
git add .
git commit -m "chore: Phase 6 GraphRAG implementation complete

All tests passing:
- Neo4jStore graph query methods (4 new methods)
- GraphEntityTechnique for entity-based retrieval
- GraphMultiHopTechnique for multi-hop reasoning
- GraphExpandTechnique for relationship expansion
- Configuration files updated
- Total tests: ~100 (83 Phase 1-5 + 17 Phase 6)"
```

---

## Task 12: Update CLAUDE.md status

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update project status**

Change:
```
**Phase:** Phase 5 Complete - Precision (2026-02-07)
```

To:
```
**Phase:** Phase 6 Complete - GraphRAG (2026-02-07)
```

**Step 2: Add Phase 6 to Implementation Phases section**

Add:
```markdown
**Phase 6 adds:**
- GraphEntityTechnique: Entity-based document retrieval
- GraphMultiHopTechnique: Multi-hop path reasoning
- GraphExpandTechnique: Entity relationship expansion
- Neo4jStore graph query methods: find_entities_in_query, get_connected_documents, get_entity_relationships, multi_hop_query
- **~100 tests passing** (83 Phase 1-5 + ~17 Phase 6)
```

**Step 3: Update Next phases**

Change:
```markdown
**Next phases:** Precision (Reranking), GraphRAG with multi-hop reasoning
```

To:
```markdown
**Next phases:** Generation (LLM response generation)
```

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update project status to Phase 6 complete

- Mark Phase 6 as complete
- Document Phase 6 features (3 new techniques, 4 Neo4jStore methods)
- Update test count to ~100 passing"
```

---

## Summary

This plan implements Phase 6 (GraphRAG) with multi-hop reasoning capabilities using Neo4j:

**Neo4jStore Enhancements (4 methods):**
- `find_entities_in_query()` - Extract entities from query text
- `get_connected_documents()` - Find documents connected to entities
- `get_entity_relationships()` - Find entities related via shared documents
- `multi_hop_query()` - Find paths connecting multiple entities

**New Graph Techniques (3):**
- `GraphEntityTechnique` - Entity-based document retrieval
- `GraphMultiHopTechnique` - Multi-hop path reasoning
- `GraphExpandTechnique` - Entity relationship expansion

**Configuration:**
- `config/techniques.yaml` - Add graph technique definitions
- `config/graphrag.yaml` - GraphRAG-specific configuration

**Files created:**
- `techniques/graph_entity.py`
- `techniques/graph_multihop.py`
- `techniques/graph_expand.py`
- `config/graphrag.yaml`
- Tests in `tests/test_graph_entity_technique.py`, `tests/test_graph_multihop_technique.py`, `tests/test_graph_expand_technique.py`, `tests/test_graphrag_config.py`