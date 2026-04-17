from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings

settings.ensure_dirs()

app = FastAPI(title=settings.app_name, version=settings.version)
app.include_router(router)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html")
