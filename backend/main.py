"""FastAPI main application."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import yaml

app = FastAPI(title="Scientific Agentic RAG Framework")

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


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "models": config.get("generation_models", {})}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Landing page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/workspace", response_class=HTMLResponse)
async def workspace(request: Request):
    """Query workspace page."""
    return templates.TemplateResponse("workspace.html", {"request": request})