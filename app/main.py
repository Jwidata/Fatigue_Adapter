from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.services.app_context import AppContext


def create_app() -> FastAPI:
    app = FastAPI(title="Fatigue_Adapter", version="0.1.0")
    app.state.context = AppContext()
    app.include_router(api_router, prefix="/api")

    static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


app = create_app()
