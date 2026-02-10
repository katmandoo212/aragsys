"""API routers."""

from backend.routers.query import router as query_router
from backend.routers.documents import router as documents_router
from backend.routers.pipelines import router as pipelines_router
from backend.routers.health import router as health_router

__all__ = ["query_router", "documents_router", "pipelines_router", "health_router"]