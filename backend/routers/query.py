"""Query API router."""

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from backend.models.query import QueryRequest
from backend.services.query_engine import QueryEngine

router = APIRouter(prefix="/api/query", tags=["query"])

# In-memory task storage (for demo - use Redis in production)
tasks: dict = {}


@router.post("", status_code=202)
async def submit_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Submit a query and return task ID."""
    import uuid
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "result": None}
    return {"task_id": task_id, "status": "pending"}


@router.get("/stream/{task_id}")
async def query_stream(task_id: str):
    """Stream query progress via SSE."""

    async def event_generator():
        """Generate SSE events."""
        engine = QueryEngine()

        try:
            async for event in engine.execute_query(
                query="demo query",  # In real: fetch from task storage
                pipeline="naive_flow"
            ):
                # Convert to SSE format
                yield f"data: {event}\n\n"

            # Signal end of stream
            yield "data: [DONE]\n\n"

        finally:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/history")
async def get_query_history(limit: int = 50):
    """Get recent query history."""
    from backend.db import QueryRecord
    records = QueryRecord.get_recent(limit=limit)
    return {
        "queries": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "pipeline": r.pipeline,
                "timestamp": r.timestamp,
                "success": r.success
            }
            for r in records
        ]
    }


@router.get("/metrics")
async def get_metrics():
    """Get query metrics."""
    from backend.db import QueryRecord
    return QueryRecord.get_metrics()