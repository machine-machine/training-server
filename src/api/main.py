"""FastAPI application factory for the 2DEXY Training Server."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import data, health, models, training
from src.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    await init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="2DEXY Training Server",
        description="GPU-powered ML training and model export for HotPath inference",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(models.router, prefix="/models", tags=["models"])
    app.include_router(training.router, prefix="/jobs", tags=["training"])
    app.include_router(data.router, prefix="/data", tags=["data"])

    return app


app = create_app()
