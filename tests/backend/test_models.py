"""Tests for Pydantic models."""

import pytest
from backend.models.query import QueryRequest, QueryResponse, ProgressEvent
from backend.models.document import DocumentCreate, DocumentResponse, FetchRequest
from backend.models.pipeline import PipelineConfig, PipelineInfo


def test_query_request_validation():
    """QueryRequest validates query string."""
    req = QueryRequest(query="test query", pipeline="naive_flow")
    assert req.query == "test query"
    assert req.pipeline == "naive_flow"


def test_query_request_empty_query_raises_error():
    """Empty query raises validation error."""
    with pytest.raises(Exception):  # Pydantic raises ValidationError for empty query
        QueryRequest(query="", pipeline="naive_flow")


def test_document_create_validation():
    """DocumentCreate validates URL."""
    doc = DocumentCreate(url="https://example.com/doc.pdf")
    assert doc.url == "https://example.com/doc.pdf"


def test_progress_event_serialization():
    """ProgressEvent can be serialized."""
    event = ProgressEvent(status="embedding_query")
    assert event.status == "embedding_query"
    assert event.progress == 0


def test_query_response_defaults():
    """QueryResponse has sensible defaults."""
    response = QueryResponse(
        answer_id="test-id",
        query="test query",
        content="test answer",
        timestamp="2026-02-10T00:00:00Z"
    )
    assert response.citations == []
    assert response.retrieved_docs == 0
    assert response.response_time_ms == 0


def test_document_response_structure():
    """DocumentResponse has required fields."""
    response = DocumentResponse(
        document_id="doc-1",
        source="https://example.com/doc.pdf",
        title="Test Document",
        chunk_count=10,
        created_at="2026-02-10T00:00:00Z"
    )
    assert response.document_id == "doc-1"
    assert response.chunk_count == 10


def test_fetch_request_max_size():
    """FetchRequest has max_size_mb with default and constraints."""
    req = FetchRequest(url="https://example.com/doc.pdf")
    assert req.max_size_mb == 5

    req2 = FetchRequest(url="https://example.com/doc.pdf", max_size_mb=10)
    assert req2.max_size_mb == 10


def test_pipeline_config_structure():
    """PipelineConfig has required fields."""
    config = PipelineConfig(
        name="naive_flow",
        query_model="llama3:8b",
        techniques=["naive_rag"]
    )
    assert config.name == "naive_flow"
    assert config.query_model == "llama3:8b"
    assert config.techniques == ["naive_rag"]


def test_pipeline_info_structure():
    """PipelineInfo contains pipelines dict and default."""
    info = PipelineInfo(
        pipelines={
            "naive_flow": PipelineConfig(
                name="naive_flow",
                query_model="llama3:8b",
                techniques=["naive_rag"]
            )
        },
        default_pipeline="naive_flow"
    )
    assert info.default_pipeline == "naive_flow"
    assert "naive_flow" in info.pipelines


def test_query_request_max_context_validation():
    """QueryRequest validates max_context_docs constraints."""
    # Within range
    req = QueryRequest(query="test", max_context_docs=10)
    assert req.max_context_docs == 10

    # At minimum
    req2 = QueryRequest(query="test", max_context_docs=1)
    assert req2.max_context_docs == 1

    # At maximum
    req3 = QueryRequest(query="test", max_context_docs=20)
    assert req3.max_context_docs == 20


def test_progress_event_with_data():
    """ProgressEvent can include additional data."""
    event = ProgressEvent(
        status="retrieving",
        progress=50,
        message="Retrieving documents...",
        data={"retrieved": 3, "total": 5}
    )
    assert event.progress == 50
    assert event.message == "Retrieving documents..."
    assert event.data == {"retrieved": 3, "total": 5}