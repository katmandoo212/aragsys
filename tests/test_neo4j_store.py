"""Tests for Neo4jStore."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from stores.neo4j_store import Neo4jStore


class TestNeo4jStoreCreation:
    """Test Neo4jStore initialization."""

    def test_neo4j_store_creates_client(self):
        """Neo4jStore creates a Neo4j driver client."""
        with patch("stores.neo4j_store.GraphDatabase") as mock_graph_db:
            mock_driver = Mock()
            mock_graph_db.driver.return_value = mock_driver
            config = {
                "neo4j": {
                    "uri": "bolt://localhost:7687",
                    "user": "neo4j",
                    "password": "password",
                    "database": "rag"
                },
                "vector_index": {"name": "document_embeddings", "dimension": 1024},
                "fulltext_index": {"name": "document_fulltext"}
            }
            store = Neo4jStore(config)
            assert store.config == config

    def test_neo4j_store_requires_uri(self):
        """Neo4jStore requires URI in config."""
        with pytest.raises(KeyError):
            config = {"neo4j": {"user": "neo4j", "password": "password"}}
            Neo4jStore(config)


class TestNeo4jStoreAddDocuments:
    """Test document addition to Neo4jStore."""

    def test_add_documents_with_embeddings(self):
        """Add documents with embeddings to Neo4j."""
        with patch("stores.neo4j_store.GraphDatabase") as mock_graph_db:
            mock_driver = Mock()
            mock_session = MagicMock()
            mock_session_context = MagicMock()
            mock_session_context.__enter__.return_value = mock_session
            mock_driver.session.return_value = mock_session_context
            mock_graph_db.driver.return_value = mock_driver

            config = {
                "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password", "database": "rag"},
                "vector_index": {"name": "document_embeddings", "dimension": 1024},
                "fulltext_index": {"name": "document_fulltext"}
            }
            store = Neo4jStore(config)
            chunks = [
                ("content 1", {"source": "file1.txt"}, [0.1] * 1024),
                ("content 2", {"source": "file1.txt"}, [0.2] * 1024)
            ]
            store.add_documents(chunks)
            mock_session.run.assert_called()

    def test_add_empty_documents(self):
        """Empty documents list is handled gracefully."""
        with patch("stores.neo4j_store.GraphDatabase") as mock_graph_db:
            mock_driver = Mock()
            mock_graph_db.driver.return_value = mock_driver

            config = {
                "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password", "database": "rag"},
                "vector_index": {"name": "document_embeddings", "dimension": 1024},
                "fulltext_index": {"name": "document_fulltext"}
            }
            store = Neo4jStore(config)
            store.add_documents([])  # Should not error


class TestNeo4jStoreVectorSearch:
    """Test vector similarity search."""

    def test_vector_search_returns_results(self):
        """Vector search returns document results."""
        with patch("stores.neo4j_store.GraphDatabase") as mock_graph_db:
            mock_driver = Mock()
            mock_session = MagicMock()
            mock_session_context = MagicMock()
            mock_session_context.__enter__.return_value = mock_session
            mock_driver.session.return_value = mock_session_context

            # Mock result dictionaries that are subscriptable
            result_dict1 = {"content": "result 1", "metadata": {"source": "file1.txt"}, "score": 0.9}
            result_dict2 = {"content": "result 2", "metadata": {"source": "file1.txt"}, "score": 0.8}

            # Make the session.run() return an iterable of dictionaries
            mock_session.run.return_value = [result_dict1, result_dict2]
            mock_graph_db.driver.return_value = mock_driver

            config = {
                "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password", "database": "rag"},
                "vector_index": {"name": "document_embeddings", "dimension": 1024},
                "fulltext_index": {"name": "document_fulltext"}
            }
            store = Neo4jStore(config)
            results = store.vector_search([0.1] * 1024, 5)
            assert len(results) == 2
            assert results[0]["content"] == "result 1"

    def test_vector_search_empty_query(self):
        """Empty query vector returns empty list."""
        with patch("stores.neo4j_store.GraphDatabase") as mock_graph_db:
            mock_driver = Mock()
            mock_graph_db.driver.return_value = mock_driver

            config = {
                "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password", "database": "rag"},
                "vector_index": {"name": "document_embeddings", "dimension": 1024},
                "fulltext_index": {"name": "document_fulltext"}
            }
            store = Neo4jStore(config)
            results = store.vector_search([], 5)
            assert results == []


class TestNeo4jStoreHybridSearch:
    """Test hybrid search with RRF."""

    def test_hybrid_search_merges_results(self):
        """Hybrid search merges dense and sparse results with RRF."""
        with patch("stores.neo4j_store.GraphDatabase") as mock_graph_db:
            mock_driver = Mock()
            mock_session = MagicMock()
            mock_session_context = MagicMock()
            mock_session_context.__enter__.return_value = mock_session
            mock_driver.session.return_value = mock_session_context

            # Mock results as dictionaries
            result_dict1 = {"content": "result 1", "metadata": {"source": "file1.txt"}, "score": 0.9}
            result_dict2 = {"content": "result 2", "metadata": {"source": "file1.txt"}, "score": 0.8}

            # Return these results
            mock_session.run.return_value = [result_dict1, result_dict2]
            mock_graph_db.driver.return_value = mock_driver

            config = {
                "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password", "database": "rag"},
                "vector_index": {"name": "document_embeddings", "dimension": 1024},
                "fulltext_index": {"name": "document_fulltext"}
            }
            store = Neo4jStore(config)
            results = store.hybrid_search("test query", [0.1] * 1024, 5)
            assert isinstance(results, list)


class TestNeo4jStoreEntities:
    """Test entity handling."""

    def test_add_entities_creates_relationships(self):
        """Add entities creates Document->Entity relationships."""
        with patch("stores.neo4j_store.GraphDatabase") as mock_graph_db:
            mock_driver = Mock()
            mock_session = MagicMock()
            mock_session_context = MagicMock()
            mock_session_context.__enter__.return_value = mock_session
            mock_driver.session.return_value = mock_session_context
            mock_graph_db.driver.return_value = mock_driver

            config = {
                "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password", "database": "rag"},
                "vector_index": {"name": "document_embeddings", "dimension": 1024},
                "fulltext_index": {"name": "document_fulltext"}
            }
            store = Neo4jStore(config)
            entities = [
                {"text": "John Smith", "label": "PERSON", "start": 0, "end": 10},
                {"text": "Acme Corp", "label": "ORG", "start": 20, "end": 29}
            ]
            store.add_entities("doc_1", entities)
            mock_session.run.assert_called()

    def test_find_entities_in_query(self):
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

        # Test entity finding (for now, returns empty until EntityExtractor integrated)
        entities = store.find_entities_in_query("John Smith worked at MIT")
        # Initially returns empty list - will integrate EntityExtractor later
        assert entities == []