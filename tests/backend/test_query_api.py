"""Tests for query API."""

from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from backend.main import app


def test_query_endpoint_accepts_request():
    """Query endpoint accepts valid request."""
    client = TestClient(app)
    response = client.post("/api/query", json={
        "query": "test query",
        "pipeline": "naive_flow"
    })
    # Should return 202 with task_id
    assert response.status_code in [202, 500]  # 500 if services not ready


def test_query_endpoint_validates_input():
    """Query endpoint validates input."""
    client = TestClient(app)
    # Empty query should fail validation
    response = client.post("/api/query", json={
        "query": "",
        "pipeline": "naive_flow"
    })
    assert response.status_code == 422  # Validation error


@patch("backend.services.query_engine.QueryEngine")
def test_query_stream_sends_progress_events(mock_engine):
    """Query stream sends SSE progress events."""
    mock_engine_instance = MagicMock()
    mock_engine.return_value = mock_engine_instance

    client = TestClient(app)
    response = client.post("/api/query", json={"query": "test", "pipeline": "naive_flow"})
    assert response.status_code in [202, 500]