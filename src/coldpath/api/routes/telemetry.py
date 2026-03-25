"""
Telemetry routes for model/version visibility.

Endpoints:
- GET /telemetry/models/{model_name} - List model versions
"""

import logging
import os

from fastapi import APIRouter
from pydantic import BaseModel

from coldpath.storage import ModelArtifactStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telemetry", tags=["telemetry"])


class ModelVersionInfo(BaseModel):
    version: int
    timestamp: str | None = None
    filepath: str | None = None


class ModelVersionsResponse(BaseModel):
    model_name: str
    versions: list[ModelVersionInfo]
    latest_version: int | None = None
    latest_timestamp: str | None = None


@router.get("/models/{model_name}", response_model=ModelVersionsResponse)
async def list_model_versions(model_name: str):
    """List model versions for the requested artifact."""
    model_dir = os.getenv("MODEL_DIR", "models")
    store = ModelArtifactStore(model_dir)
    versions = store.list_versions(model_name)

    latest_version = versions[-1] if versions else None
    latest_version_number = latest_version["version"] if latest_version else None
    latest_timestamp = latest_version.get("timestamp") if latest_version else None

    return ModelVersionsResponse(
        model_name=model_name,
        versions=[ModelVersionInfo(**entry) for entry in versions],
        latest_version=latest_version_number,
        latest_timestamp=latest_timestamp,
    )
