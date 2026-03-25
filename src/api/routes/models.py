"""Model registry endpoints — manifest, download, promote."""

import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import verify_api_key
from src.api.deps import get_db
from src.db.models import ModelArtifact

router = APIRouter()


@router.get(
    "/manifest",
    dependencies=[Depends(verify_api_key)],
    summary="List promoted models",
    description=(
        "Returns every model that has been promoted to 'latest' — one per ``model_type``. "
        "This is what HotPath polls to discover new model versions. "
        "``model_type`` values: ``profitability``, ``fraud``, ``slippage``, ``regime``."
    ),
    response_description="List of promoted model artifacts with metrics",
)
async def get_manifest(db: AsyncSession = Depends(get_db)):
    """Return all promoted (latest) models."""
    stmt = select(ModelArtifact).where(ModelArtifact.promoted_at.isnot(None))
    result = await db.execute(stmt)
    artifacts = result.scalars().all()

    return {
        "models": [
            {
                "id": str(a.id),
                "model_type": a.model_type,
                "version": a.version,
                "sha256": a.sha256_checksum,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "metrics": a.metrics,
            }
            for a in artifacts
        ]
    }


@router.get(
    "/latest/{model_type}",
    dependencies=[Depends(verify_api_key)],
    summary="Download the active model artifact",
    description=(
        "Returns the full artifact JSON for the currently promoted model of the given type. "
        "This is the payload the HotPath Rust engine fetches and loads. "
        "``model_type``: ``profitability`` | ``fraud`` | ``slippage`` | ``regime``."
    ),
    response_description="Full ModelArtifact JSON including weights, calibration, and checksum",
)
async def get_latest_model(model_type: str, db: AsyncSession = Depends(get_db)):
    """Download the full ModelArtifact JSON for the promoted version."""
    stmt = (
        select(ModelArtifact)
        .where(ModelArtifact.model_type == model_type)
        .where(ModelArtifact.promoted_at.isnot(None))
        .order_by(ModelArtifact.promoted_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    artifact = result.scalar_one_or_none()

    if not artifact:
        raise HTTPException(404, detail=f"No promoted model of type '{model_type}'")

    # Read the full artifact JSON from disk
    try:
        with open(artifact.file_path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(500, detail="Artifact file missing from storage")


@router.post(
    "/promote/{artifact_id}",
    dependencies=[Depends(verify_api_key)],
    summary="Promote a model artifact",
    description=(
        "Promotes the specified artifact to 'latest' for its model type, "
        "atomically demoting the previously promoted artifact of the same type. "
        "After promotion, ``GET /models/latest/{model_type}`` and ``GET /models/manifest`` "
        "will return this artifact. HotPath will pick it up on its next poll cycle."
    ),
    response_description="Confirmation with model_type and version number",
)
async def promote_model(artifact_id: UUID, db: AsyncSession = Depends(get_db)):
    """Promote a model artifact to 'latest' for its type."""
    # Fetch the artifact to promote
    stmt = select(ModelArtifact).where(ModelArtifact.id == artifact_id)
    result = await db.execute(stmt)
    artifact = result.scalar_one_or_none()

    if not artifact:
        raise HTTPException(404, detail="Artifact not found")

    # Demote current promoted model of same type
    demote_stmt = (
        update(ModelArtifact)
        .where(ModelArtifact.model_type == artifact.model_type)
        .where(ModelArtifact.promoted_at.isnot(None))
        .values(promoted_at=None)
    )
    await db.execute(demote_stmt)

    # Promote the new one
    from datetime import datetime, timezone

    artifact.promoted_at = datetime.now(timezone.utc)
    await db.commit()

    return {
        "promoted": True,
        "model_type": artifact.model_type,
        "version": artifact.version,
    }
