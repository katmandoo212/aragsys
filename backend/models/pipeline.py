"""Pipeline configuration models."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class PipelineConfig(BaseModel):
    """Pipeline configuration."""

    name: str = Field(..., description="Pipeline name")
    query_model: str = Field(..., description="Query model to use")
    techniques: list[str] = Field(..., description="List of technique names")
    description: Optional[str] = Field(None, description="Pipeline description")


class PipelineInfo(BaseModel):
    """Pipeline information."""

    pipelines: Dict[str, PipelineConfig] = Field(..., description="Available pipelines")
    default_pipeline: str = Field(..., description="Default pipeline name")