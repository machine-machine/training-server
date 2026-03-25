"""
ML routes for proposals and feedback.

Endpoints:
- POST /ml/proposals - Get ML-ranked setting candidates
- POST /ml/feedback - Submit feedback for learning loop
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml"])


def _default_samples_dir() -> Path:
    # Persist training samples inside the repo so restarts/rebuilds keep them
    project_root = Path(__file__).resolve().parents[4]  # EngineColdPath/
    return project_root / "data" / "training"


# Request/Response Models


class ProposalConstraints(BaseModel):
    """Constraints for proposal generation."""

    max_drawdown: float | None = Field(
        None, ge=0.0, le=100.0, description="Max drawdown percentage (0-100)"
    )
    min_sharpe: float | None = Field(None, ge=-10.0, le=100.0, description="Minimum Sharpe ratio")
    max_position_sol: float | None = Field(
        None, ge=0.0001, le=10000.0, description="Max position size in SOL"
    )
    min_liquidity_usd: float | None = Field(
        None, ge=0, le=1_000_000_000, description="Min liquidity in USD"
    )


class ProposalsRequest(BaseModel):
    """Request for ML proposals."""

    current_params: dict[str, Any] = Field(..., description="Current strategy parameters")
    constraints: ProposalConstraints | None = Field(None, description="Constraints to respect")
    num_proposals: int = Field(
        default=5, ge=1, le=10, description="Number of proposals to generate"
    )
    exploration_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for exploration vs exploitation (higher = more exploration)",
    )


class ProposalCandidate(BaseModel):
    """A single proposal candidate."""

    id: str
    params: dict[str, Any]
    expected_reward: float
    confidence: float
    exploration_bonus: float
    source: str  # "bandit", "perturbation", "hybrid"


class ProposalsResponse(BaseModel):
    """Response with ML proposals."""

    proposals: list[ProposalCandidate]
    bandit_recommendation: dict[str, Any] | None = None
    total_pulls: int
    timestamp: str


class FeedbackRequest(BaseModel):
    """Request to submit trade feedback."""

    proposal_id: str | None = Field(None, description="ID of the proposal that was used")
    params_used: dict[str, Any] = Field(..., description="Parameters that were used")
    trade_result: dict[str, Any] = Field(
        ..., description="Trade outcome (pnl_pct, included, slippage_bps, etc.)"
    )
    context: dict[str, Any] | None = Field(
        None, description="Additional context (liquidity, volatility, etc.)"
    )


class FeedbackResponse(BaseModel):
    """Response after processing feedback."""

    success: bool
    reward_calculated: float
    arm_updated: str | None = None
    message: str


# Dependency injection


def get_bandit_trainer():
    """Get bandit trainer instance."""
    from coldpath.api.server import get_app_state

    return get_app_state().bandit_trainer


BanditTrainerDep = Annotated[Any, Depends(get_bandit_trainer)]


# Endpoints


@router.post("/proposals", response_model=ProposalsResponse)
async def get_proposals(
    request: ProposalsRequest,
    bandit_trainer: BanditTrainerDep,
):
    """Get ML-ranked parameter proposals.

    Uses multi-armed bandit to select candidate parameters based on
    historical performance. Balances exploration of new parameter
    combinations with exploitation of known good ones.
    """
    import uuid
    from datetime import datetime

    try:
        proposals = []

        # Get bandit recommendation
        bandit_rec = bandit_trainer.get_recommendation()
        arm_weights = bandit_trainer.get_arm_weights()

        # Generate proposals based on bandit arms
        for i in range(min(request.num_proposals, len(arm_weights))):
            # Select arm based on weights or exploration
            if i == 0:
                # First proposal: best arm (exploitation)
                arm = bandit_trainer.select_arm()
                source = "bandit"
            else:
                # Subsequent proposals: explore variations
                arm = bandit_trainer.select_arm()
                source = "perturbation" if i % 2 == 0 else "hybrid"

            # Create candidate params from current + arm's slippage
            candidate_params = request.current_params.copy()
            candidate_params["slippage_bps"] = arm.slippage_bps

            # Apply exploration perturbations for variety
            if source == "perturbation":
                candidate_params = _perturb_params(
                    candidate_params,
                    request.constraints,
                    perturbation_scale=0.1 + (i * 0.05),
                )

            # Calculate expected reward and confidence
            expected_reward = arm.mean_reward
            confidence = min(1.0, arm.count / 100) if arm.count > 0 else 0.1
            exploration_bonus = request.exploration_weight * (1.0 - confidence)

            proposals.append(
                ProposalCandidate(
                    id=str(uuid.uuid4()),
                    params=candidate_params,
                    expected_reward=expected_reward + exploration_bonus,
                    confidence=confidence,
                    exploration_bonus=exploration_bonus,
                    source=source,
                )
            )

        # Sort by expected reward (including exploration bonus)
        proposals.sort(key=lambda p: p.expected_reward, reverse=True)

        return ProposalsResponse(
            proposals=proposals[: request.num_proposals],
            bandit_recommendation=bandit_rec,
            total_pulls=bandit_trainer.total_pulls,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Proposals error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    bandit_trainer: BanditTrainerDep,
):
    """Submit trade feedback for the learning loop.

    Updates the multi-armed bandit with observed rewards from actual
    trades. This allows the system to learn which parameter combinations
    work best.
    """
    try:
        # Calculate reward from trade result
        reward = bandit_trainer.calculate_reward(request.trade_result)

        # Determine which arm was used
        slippage_used = request.params_used.get("slippage_bps", 300)
        arm_name = bandit_trainer._classify_arm(slippage_used)

        # Update the bandit
        success = reward > 0
        bandit_trainer.update(arm_name, reward, success)

        logger.info(f"Feedback processed: arm={arm_name}, reward={reward:.4f}, success={success}")

        return FeedbackResponse(
            success=True,
            reward_calculated=reward,
            arm_updated=arm_name,
            message=f"Updated {arm_name} arm with reward {reward:.4f}",
        )

    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/stats")
async def get_ml_stats(
    bandit_trainer: BanditTrainerDep,
):
    """Get ML model statistics."""
    return {
        "bandit": bandit_trainer.to_params(),
        "training_history": bandit_trainer.training_history[-10:],  # Last 10 training runs
    }


def _perturb_params(
    params: dict[str, Any],
    constraints: ProposalConstraints | None,
    perturbation_scale: float = 0.1,
) -> dict[str, Any]:
    """Apply random perturbations to parameters within constraints."""
    import random

    perturbed = params.copy()

    # Define perturbation ranges for each parameter
    perturbations = {
        "stop_loss_pct": (-2, 2),
        "take_profit_pct": (-5, 5),
        "max_hold_minutes": (-5, 5),
        "min_liquidity_usd": (-2000, 2000),
        "max_risk_score": (-0.05, 0.05),
        "max_position_sol": (-0.01, 0.01),
    }

    for key, (min_delta, max_delta) in perturbations.items():
        if key in perturbed:
            delta = random.uniform(min_delta, max_delta) * perturbation_scale
            new_value = perturbed[key] + delta

            # Apply constraints
            if constraints:
                if key == "max_position_sol" and constraints.max_position_sol:
                    new_value = min(new_value, constraints.max_position_sol)
                if key == "min_liquidity_usd" and constraints.min_liquidity_usd:
                    new_value = max(new_value, constraints.min_liquidity_usd)

            # Ensure positive values where needed
            if key in ["stop_loss_pct", "take_profit_pct", "min_liquidity_usd"]:
                new_value = max(0.1, new_value)
            elif key == "max_risk_score":
                new_value = max(0.1, min(1.0, new_value))

            perturbed[key] = new_value

    return perturbed


# ============================================================================
# V3 Model Training Endpoints
# ============================================================================


class TrainV3Request(BaseModel):
    """Request to train v3 model from collected samples."""

    samples_path: str | None = Field(
        None, description="Path to training samples JSON file (server-side)"
    )
    samples_data: dict[str, Any] | None = Field(
        None, description="Inline training samples (as exported from Swift)"
    )
    output_path: str | None = Field(None, description="Output path for trained model")
    dataset_id: str | None = Field(None, description="Dataset identifier for artifact")
    config_override: dict[str, Any] | None = Field(
        None, description="Override training config parameters"
    )


class TrainV3Response(BaseModel):
    """Response from v3 training."""

    success: bool
    message: str
    metrics: dict[str, float] | None = None
    feature_importance: dict[str, float] | None = None
    output_path: str | None = None
    checksum: str | None = None


class UploadSamplesRequest(BaseModel):
    """Request to upload training samples."""

    samples: list[dict[str, Any]] = Field(
        ..., description="List of training samples with features and outcomes"
    )
    overwrite: bool = Field(default=False, description="Whether to overwrite existing samples")


class UploadSamplesResponse(BaseModel):
    """Response from samples upload."""

    success: bool
    message: str
    samples_received: int
    samples_labeled: int
    total_samples: int


@router.post("/train/v3", response_model=TrainV3Response)
async def train_v3_model(request: TrainV3Request):
    """Train v3 profitability model on real data.

    This endpoint trains a new XGBoost model using collected samples
    from the Swift DataCollectionService. The model is trained with
    50 features and calibrated using isotonic regression.

    Target metrics:
    - Accuracy >= 65%
    - AUC-ROC >= 0.75
    - Calibration error <= 0.05

    Returns the trained model metrics and export path.
    """
    import os

    try:
        from coldpath.training.v3_profitability_trainer import (
            TrainingConfig,
            V3ProfitabilityTrainer,
        )

        # Determine samples source
        if request.samples_data:
            # Use inline data
            samples_source = request.samples_data
            logger.info("Training from inline samples data")
        elif request.samples_path:
            # Use file path
            samples_source = request.samples_path
            logger.info(f"Training from file: {request.samples_path}")
        else:
            # Fall back to samples from ALL locations (Python + Swift storage)
            all_samples = _load_all_samples()
            if not all_samples:
                raise HTTPException(
                    status_code=400,
                    detail="No samples found in any location. "
                    "Checked: Python storage (EngineColdPath/data/training/) "
                    "and Swift storage (~/Library/Application Support/2DEXY/TrainingData/)",
                )
            samples_source = {"samples": all_samples}
            logger.info(f"Training from all locations: {len(all_samples)} samples loaded")

        # Create trainer with optional config override
        config = TrainingConfig()
        if request.config_override:
            for key, value in request.config_override.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        trainer = V3ProfitabilityTrainer(config)

        # Train
        if isinstance(samples_source, str):
            result = trainer.train_from_file(samples_source)
        else:
            result = trainer.train_from_dict(samples_source)

        # Determine output path
        output_path = request.output_path
        if not output_path:
            # Default to local artifacts directory
            artifacts_dir = os.environ.get(
                "ARTIFACTS_DIR",
                str(Path(__file__).resolve().parents[4] / "artifacts" / "profitability"),
            )
            output_path = os.path.join(artifacts_dir, "v003.json")

        # Export
        dataset_id = request.dataset_id or f"real_v3_{datetime.utcnow().strftime('%Y%m%d')}"
        exported_path = trainer.export_to_hotpath(result, output_path, dataset_id)

        return TrainV3Response(
            success=True,
            message=f"Model trained successfully with {result.metrics.get('n_train', 0)} samples",
            metrics=result.metrics,
            feature_importance=result.feature_importance,
            output_path=exported_path,
            checksum=None,  # Would extract from export
        )

    except ValueError as e:
        logger.error(f"Training validation error: {e}")
        return TrainV3Response(
            success=False,
            message=str(e),
        )
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/samples/upload", response_model=UploadSamplesResponse)
async def upload_training_samples(request: UploadSamplesRequest):
    """Upload training samples from Swift DataCollectionService.

    This endpoint receives samples exported from the Swift app's
    DataCollectionService and stores them for model training.

    Samples should include:
    - features: 50-element array matching HotPath RawFeatures
    - outcome: {wasProfitable, pnlPercentage, pnlSol, exitReason}
    - metadata: {mint, symbol, timestamp, source, collectionMode}
    """
    import json

    try:
        # Get storage directory
        samples_dir = Path(os.environ.get("SAMPLES_DIR", _default_samples_dir()))
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Count labeled samples
        labeled_count = sum(1 for s in request.samples if s.get("outcome") is not None)

        # Load existing samples if not overwriting
        existing_path = samples_dir / "training_samples.jsonl"
        existing_samples = []

        if not request.overwrite and existing_path.exists():
            with open(existing_path) as f:
                for line in f:
                    if line.strip():
                        existing_samples.append(json.loads(line))

        # Merge samples (dedupe by mint + timestamp)
        existing_keys = {(s.get("mint", ""), s.get("timestamp", "")) for s in existing_samples}

        new_samples = []
        for s in request.samples:
            key = (s.get("mint", ""), s.get("timestamp", ""))
            if key not in existing_keys:
                new_samples.append(s)
                existing_keys.add(key)

        # Combine and save
        all_samples = existing_samples + new_samples

        with open(existing_path, "w") as f:
            for s in all_samples:
                f.write(json.dumps(s) + "\n")

        logger.info(f"Uploaded {len(new_samples)} new samples, total: {len(all_samples)}")

        return UploadSamplesResponse(
            success=True,
            message=f"Received {len(request.samples)} samples, {len(new_samples)} new",
            samples_received=len(request.samples),
            samples_labeled=labeled_count,
            total_samples=len(all_samples),
        )

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _swift_samples_dir() -> Path:
    """Get Swift app's training samples directory."""
    # Swift saves to ~/Library/Application Support/2DEXY/TrainingData
    home = Path.home()
    return home / "Library" / "Application Support" / "2DEXY" / "TrainingData"


def _load_all_samples() -> list:
    """Load samples from both Python and Swift storage locations."""
    import json

    samples = []
    seen_keys = set()

    # Helper to check if sample is labeled
    def is_labeled(s):
        return s.get("outcome") is not None or s.get("was_profitable") is not None

    # Locations to check
    locations = [
        Path(os.environ.get("SAMPLES_DIR", _default_samples_dir())) / "training_samples.jsonl",
        _swift_samples_dir() / "training_samples.jsonl",
    ]

    for samples_path in locations:
        if not samples_path.exists():
            continue

        try:
            with open(samples_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        sample = json.loads(line)
                        # Dedupe by mint + timestamp (handle both timestamp formats)
                        ts = sample.get("timestamp") or sample.get("timestamp_ms") or ""
                        key = (sample.get("mint", ""), str(ts))
                        if key not in seen_keys:
                            seen_keys.add(key)
                            samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to load samples from {samples_path}: {e}")

    return samples


@router.get("/samples/stats")
async def get_samples_stats():
    """Get statistics about collected training samples.

    Reads from both:
    - Python storage: EngineColdPath/data/training/
    - Swift storage: ~/Library/Application Support/2DEXY/TrainingData/
    """
    try:
        samples = _load_all_samples()

        if not samples:
            return {
                "total_samples": 0,
                "labeled_samples": 0,
                "positive_samples": 0,
                "negative_samples": 0,
                "positive_ratio": 0.0,
                "ready_for_training": False,
                "min_required": 500,
            }

        # Handle both flat schema (was_profitable) and nested schema (outcome.wasProfitable)
        def is_labeled(s):
            return s.get("outcome") is not None or s.get("was_profitable") is not None

        def is_positive(s):
            if s.get("outcome"):
                return s["outcome"].get("wasProfitable", False)
            return s.get("was_profitable", False)

        labeled = [s for s in samples if is_labeled(s)]
        positive = sum(1 for s in labeled if is_positive(s))

        return {
            "total_samples": len(samples),
            "labeled_samples": len(labeled),
            "positive_samples": positive,
            "negative_samples": len(labeled) - positive,
            "positive_ratio": positive / len(labeled) if labeled else 0.0,
            "ready_for_training": len(labeled) >= 500,
            "min_required": 500,
            "sources": list(set(s.get("source", "unknown") for s in samples)),
        }

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
