"""Document request/response models."""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any


class DocumentCreate(BaseModel):
    """Request to create a document from URL."""

    url: str = Field(..., description="Document URL")
    title: Optional[str] = Field(None, description="Document title (optional)")
    tags: list[str] = Field(default_factory=list, description="Document tags")


class FetchRequest(BaseModel):
    """Request to fetch document from web."""

    url: str = Field(..., description="URL to fetch")
    max_size_mb: int = Field(default=5, ge=1, le=50, description="Max content size in MB")


class DocumentResponse(BaseModel):
    """Response with document metadata."""

    document_id: str = Field(..., description="Unique document identifier")
    source: str = Field(..., description="Source URL or file path")
    title: str = Field(..., description="Document title")
    chunk_count: int = Field(..., description="Number of chunks")
    created_at: str = Field(..., description="ISO timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")