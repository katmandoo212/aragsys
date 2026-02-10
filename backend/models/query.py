"""Query request/response models."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class QueryRequest(BaseModel):
    """Request to run a RAG query."""

    query: str = Field(..., min_length=1, description="The query text")
    pipeline: str = Field(default="naive_flow", description="Pipeline name to use")
    max_context_docs: int = Field(default=5, ge=1, le=20, description="Max context documents")


class ProgressEvent(BaseModel):
    """Server-Sent Event for query progress."""

    status: str = Field(..., description="Current status: embedding_query, retrieving, reranking, generating, complete")
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage")
    message: str = Field(default="", description="Status message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Additional data")


class QueryResponse(BaseModel):
    """Response from a RAG query."""

    answer_id: str = Field(..., description="Unique answer identifier")
    query: str = Field(..., description="The original query")
    content: str = Field(..., description="Generated answer content")
    citations: list[str] = Field(default_factory=list, description="Citation sources")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    retrieved_docs: int = Field(default=0, description="Number of documents retrieved")
    response_time_ms: int = Field(default=0, description="Response time in milliseconds")
    timestamp: str = Field(..., description="ISO timestamp of query")