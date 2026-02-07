import pytest
from utils.vector_store import VectorStore


def test_vector_store_creation(tmp_path):
    config = {
        "persist_directory": str(tmp_path / "vectors"),
        "collection_name": "test_collection"
    }

    store = VectorStore(config)

    assert store.config == config


def test_vector_store_add_documents(tmp_path):
    config = {
        "persist_directory": str(tmp_path / "vectors"),
        "collection_name": "test_collection"
    }

    store = VectorStore(config)

    chunks = [
        ("First chunk", {"source": "test.txt", "index": 0}, [0.1, 0.2, 0.3]),
        ("Second chunk", {"source": "test.txt", "index": 1}, [0.4, 0.5, 0.6]),
    ]

    store.add_documents(chunks)

    count = store.collection.count()
    assert count == 2


def test_vector_store_search(tmp_path):
    config = {
        "persist_directory": str(tmp_path / "vectors"),
        "collection_name": "test_collection"
    }

    store = VectorStore(config)

    # Add test documents
    chunks = [
        ("First chunk", {"source": "test.txt"}, [0.1, 0.2, 0.3]),
        ("Second chunk", {"source": "test.txt"}, [0.4, 0.5, 0.6]),
    ]
    store.add_documents(chunks)

    # Search with query vector
    results = store.search([0.1, 0.2, 0.3], top_k=2)

    assert len(results) == 2
    assert all("content" in r for r in results)
    assert all("metadata" in r for r in results)
    assert all("distance" in r for r in results)


def test_vector_store_search_empty_store(tmp_path):
    config = {
        "persist_directory": str(tmp_path / "vectors"),
        "collection_name": "test_collection"
    }

    store = VectorStore(config)

    results = store.search([0.1, 0.2, 0.3], top_k=5)

    assert results == []