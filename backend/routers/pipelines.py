"""Pipelines API router."""

from fastapi import APIRouter
from backend.models.pipeline import PipelineConfig, PipelineInfo
import yaml

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])


@router.get("", response_model=PipelineInfo)
async def list_pipelines():
    """List available pipelines."""
    with open("config/pipelines.yaml") as f:
        config = yaml.safe_load(f)

    pipelines = {}
    for name, cfg in config.get("pipelines", {}).items():
        pipelines[name] = PipelineConfig(
            name=name,
            query_model=cfg.get("query_model", ""),
            techniques=cfg.get("techniques", []),
            description=f"{name} pipeline"
        )

    return PipelineInfo(
        pipelines=pipelines,
        default_pipeline=config.get("default_query_model", "naive_flow")
    )