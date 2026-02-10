"""FastAPI main application."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import yaml

from backend.routers import query_router, documents_router, pipelines_router, health_router

app = FastAPI(title="Scientific Agentic RAG Framework")

# Include routers
app.include_router(health_router)
app.include_router(query_router)
app.include_router(documents_router)
app.include_router(pipelines_router)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="frontend/templates")

# Load configuration
config_path = Path("config/models.yaml")
config: dict[str, Any] = {}
try:
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
except (yaml.YAMLError, OSError) as e:
    config = {"error": f"Failed to load config: {e}"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Landing page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/workspace", response_class=HTMLResponse)
async def workspace(request: Request):
    """Query workspace page."""
    return templates.TemplateResponse("workspace.html", {"request": request})