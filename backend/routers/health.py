"""Health API router."""

from fastapi import APIRouter
import yaml

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("")
async def health_check():
    """Health check endpoint."""
    with open("config/models.yaml") as f:
        models = yaml.safe_load(f)

    return {
        "status": "ok",
        "version": "1.0.0",
        "models": {
            "generation": models.get("generation_models", {}).get("default", ""),
            "embedding": models.get("embedding_models", {}).get("default", "")
        }
    }