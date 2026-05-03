from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.catalog import router as catalog_router
from app.api.routes.health import router as health_router
from app.api.routes.projects import router as projects_router
from app.api.routes.training import router as training_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(catalog_router)
app.include_router(projects_router)
app.include_router(training_router)

app.mount("/assets", StaticFiles(directory=Path(__file__).resolve().parents[1] / "assets"), name="assets")
app.mount("/media", StaticFiles(directory=settings.data_dir), name="media")
