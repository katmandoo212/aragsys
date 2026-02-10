"""Documents API router."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.models.document import FetchRequest
from backend.services.document_service import DocumentService

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("")
async def list_documents():
    """List all indexed documents."""
    service = DocumentService()
    docs = service.list_documents()
    return {"documents": docs}


@router.post("/fetch")
async def fetch_document(request: FetchRequest):
    """Fetch document from web URL."""
    from backend.utils.web_fetcher import WebFetcher

    fetcher = WebFetcher()
    try:
        result = await fetcher.fetch(request.url, request.max_size_mb)

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)

        # In real implementation: chunk and embed content
        return {
            "success": True,
            "document": {
                "url": result.url,
                "title": result.title,
                "content_type": result.content_type,
                "chunk_count": 1,  # Placeholder
                "created_at": result.fetched_at
            }
        }
    finally:
        await fetcher.close()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload document file."""
    # Placeholder for file upload
    return {
        "success": True,
        "message": "Document uploaded (placeholder)",
        "filename": file.filename
    }


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete document by ID."""
    # Placeholder for delete
    return {"success": True, "message": "Document deleted"}