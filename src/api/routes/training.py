"""Training job endpoints — launch, status, cancel."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import verify_api_key
from src.api.deps import get_db
from src.db.models import ModelArtifact, TrainingJob
from src.workers.training_tasks import run_training_job

router = APIRouter()


class TrainRequest(BaseModel):
    job_type: str
    """
    Type of model to train. One of:
    - ``ensemble``  — Linear + LightGBM + XGBoost ensemble (profitability prediction).
      Dataset needs numeric feature columns plus a binary label column (default: ``profitable``).
    - ``fraud``     — RandomForest rug-detection classifier with MI feature selection.
      Dataset needs: ``liquidity_usd``, ``fdv_usd``, ``holder_count``, ``top_holder_pct``,
      ``mint_authority_enabled``, ``freeze_authority_enabled``, ``lp_burn_pct``,
      ``age_seconds``, ``volume_24h``, ``price_change_1h``, plus label column (default: ``is_rug``).
    - ``bandit``    — UCB / Thompson Sampling multi-armed bandit for slippage optimisation.
      Dataset needs: ``quoted_slippage_bps``, ``slippage_bps``, ``pnl_pct``, ``included``.
    - ``rl``        — Alias for ``bandit``.
    - ``slippage``  — L-BFGS-B linear slippage prediction model.
      Dataset needs: ``quoted_slippage_bps``, ``realized_slippage_bps``, ``liquidity_usd``,
      ``volume_usd``, ``volatility``, ``latency_ms``.
    - ``regime``    — Market regime detector (MEME_SEASON / NORMAL / BEAR_VOLATILE / HIGH_MEV / LOW_LIQUIDITY).
      Dataset needs: ``total_volume_24h_sol``, ``avg_volume_7d_sol``, ``new_tokens_24h``,
      ``sol_price_change_7d_pct``, ``volatility_24h``, ``avg_volatility_7d``,
      ``sandwich_attacks_24h``, ``jito_bundle_rate``.
    """
    dataset_id: UUID
    """UUID returned by ``POST /data/upload``."""
    hyperparams: dict = {}
    """
    Optional training hyperparameters. Common keys by job type:

    ``ensemble``: ``label_column`` (str, default ``profitable``), ``n_estimators`` (int),
    ``max_depth`` (int), ``learning_rate`` (float), ``C`` (float, logistic regression),
    ``ensemble_weights`` (dict with ``linear``, ``lgbm``, ``xgb`` floats summing to 1).

    ``fraud``: ``label_column`` (str, default ``is_rug``).

    ``bandit``: ``strategy`` (``ucb``|``thompson``|``epsilon_greedy``),
    ``exploration_param`` (float), ``reward_decay`` (float).

    ``slippage``: column name overrides — ``quoted_col``, ``realized_col``,
    ``liquidity_col``, ``volume_col``, ``volatility_col``, ``latency_col``.
    """
    gpu_id: int | None = None
    """Pin to a specific GPU index (0 or 1). Omit to let the scheduler pick."""


@router.post(
    "/train",
    dependencies=[Depends(verify_api_key)],
    status_code=202,
    summary="Launch a training job",
    description=(
        "Enqueues an async Celery training task and returns immediately with a ``job_id``. "
        "Poll ``GET /jobs/{job_id}/status`` to track progress. "
        "On completion the worker writes a model artifact to disk and registers it in the DB. "
        "Use ``POST /models/promote/{artifact_id}`` to make it the active model."
    ),
    response_description="job_id and initial pending status",
)
async def launch_training(req: TrainRequest, db: AsyncSession = Depends(get_db)):
    """Launch an async training job via Celery."""
    valid_types = {"ensemble", "fraud", "rl", "bandit", "slippage", "regime"}
    if req.job_type not in valid_types:
        raise HTTPException(400, detail=f"job_type must be one of {valid_types}")

    # Create job record
    job = TrainingJob(
        job_type=req.job_type,
        status="pending",
        hyperparams=req.hyperparams,
        gpu_id=req.gpu_id,
        dataset_id=req.dataset_id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Dispatch to Celery
    run_training_job.delay(str(job.id))

    return {
        "job_id": str(job.id),
        "status": "pending",
        "message": "Training job queued",
    }


@router.get(
    "/{job_id}/status",
    dependencies=[Depends(verify_api_key)],
    summary="Poll training job status",
    description=(
        "Returns the current status of a training job and, once complete, its metrics. "
        "``status`` progresses: ``pending`` → ``running`` → ``completed`` | ``failed``."
    ),
    response_description="Job status, metrics, and timing",
)
async def get_job_status(job_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get the current status and metrics of a training job."""
    stmt = select(TrainingJob).where(TrainingJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(404, detail="Job not found")

    elapsed = None
    if job.started_at:
        from datetime import datetime, timezone

        now = job.completed_at or datetime.now(timezone.utc)
        elapsed = int((now - job.started_at).total_seconds())

    artifact = None
    if job.status == "completed":
        artifact_stmt = (
            select(ModelArtifact)
            .where(ModelArtifact.job_id == job.id)
            .order_by(ModelArtifact.created_at.desc())
        )
        artifact_result = await db.execute(artifact_stmt)
        artifact = artifact_result.scalars().first()

    return {
        "job_id": str(job.id),
        "job_type": job.job_type,
        "status": job.status,
        "gpu_id": job.gpu_id,
        "artifact_id": str(artifact.id) if artifact else None,
        "artifact_model_type": artifact.model_type if artifact else None,
        "artifact_version": artifact.version if artifact else None,
        "metrics": job.metrics or {},
        "error_message": job.error_message,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "elapsed_seconds": elapsed,
    }
