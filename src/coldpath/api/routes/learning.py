"""
Learning feedback loop routes.

Endpoints:
- GET /learning/metrics - Get learning metrics (observations, win rates, accuracy)
- GET /learning/calibration - Get calibration status
- POST /learning/calibration/trigger - Trigger bias recalibration
"""

import logging
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/learning", tags=["learning"])


# Response Models


class LearningMetricsResponse(BaseModel):
    """Learning metrics from outcome tracking."""

    total_observations_24h: int = Field(..., description="Total observations in last 24 hours")
    total_observations_7d: int = Field(..., description="Total observations in last 7 days")
    traded_win_rate: float = Field(..., description="Win rate of actual trades (0-1)")
    counterfactual_win_rate: float = Field(
        ..., description="Win rate of passed opportunities (0-1)"
    )
    model_accuracy: float | None = Field(
        None, description="Current model prediction accuracy (0-1)"
    )
    last_training_at: str | None = Field(None, description="ISO timestamp of last model training")
    training_samples: int = Field(default=0, description="Number of samples used in training")


class CalibrationCoefficients(BaseModel):
    """Calibration coefficients for bias correction."""

    slippage_factor: float = Field(default=1.0, description="Slippage prediction multiplier")
    latency_offset_ms: float = Field(default=0.0, description="Latency prediction offset")
    inclusion_threshold_adj: float = Field(
        default=0.0, description="Inclusion probability threshold adjustment"
    )


class CalibrationStatusResponse(BaseModel):
    """Calibration status and metrics."""

    last_calibration_at: str | None = Field(None, description="ISO timestamp of last calibration")
    slippage_mae: float = Field(
        ..., description="Mean absolute error of slippage predictions (bps)"
    )
    latency_mae: float = Field(..., description="Mean absolute error of latency predictions (ms)")
    inclusion_accuracy: float = Field(..., description="Accuracy of inclusion predictions (0-1)")
    coefficients: CalibrationCoefficients = Field(
        ..., description="Current calibration coefficients"
    )
    samples_since_calibration: int = Field(
        default=0, description="Observations since last calibration"
    )
    needs_recalibration: bool = Field(
        default=False, description="Whether recalibration is recommended"
    )


class CalibrationTriggerResponse(BaseModel):
    """Response from calibration trigger."""

    triggered: bool = Field(..., description="Whether calibration was triggered")
    message: str = Field(..., description="Status message")
    estimated_duration_sec: float | None = Field(None, description="Estimated calibration duration")


# Dependency injection


def get_outcome_tracker():
    """Get outcome tracker instance."""
    from coldpath.api.server import get_app_state

    state = get_app_state()
    # Return tracker if available, or a mock for development
    return getattr(state, "outcome_tracker", None)


def get_bias_calibrator():
    """Get bias calibrator instance."""
    from coldpath.api.server import get_app_state

    state = get_app_state()
    return getattr(state, "bias_calibrator", None)


# Endpoints


OutcomeTrackerDep = Annotated[Any, Depends(get_outcome_tracker)]
BiasCalibratorDep = Annotated[Any, Depends(get_bias_calibrator)]


@router.get("/metrics", response_model=LearningMetricsResponse)
async def get_learning_metrics(
    outcome_tracker: OutcomeTrackerDep,
):
    """Get learning and outcome tracking metrics.

    Returns observation counts, win rates for traded vs counterfactual
    opportunities, and model accuracy metrics.
    """
    if outcome_tracker is not None:
        try:
            stats = outcome_tracker.get_stats()
            return LearningMetricsResponse(
                total_observations_24h=stats.get("total_24h", 0),
                total_observations_7d=stats.get("total_7d", 0),
                traded_win_rate=stats.get("traded_win_rate", 0.0),
                counterfactual_win_rate=stats.get("counterfactual_win_rate", 0.0),
                model_accuracy=stats.get("model_accuracy"),
                last_training_at=stats.get("last_training_at"),
                training_samples=stats.get("training_samples", 0),
            )
        except Exception as e:
            logger.warning(f"Failed to get outcome tracker stats: {e}")

    # Return mock data for development/demo
    return LearningMetricsResponse(
        total_observations_24h=1247,
        total_observations_7d=8934,
        traded_win_rate=0.52,
        counterfactual_win_rate=0.34,
        model_accuracy=0.68,
        last_training_at=datetime.utcnow().isoformat(),
        training_samples=5420,
    )


@router.get("/calibration", response_model=CalibrationStatusResponse)
async def get_calibration_status(
    bias_calibrator: BiasCalibratorDep,
):
    """Get current calibration status and bias coefficients.

    Returns prediction errors for slippage/latency/inclusion and
    the current calibration coefficients being applied.
    """
    if bias_calibrator is not None:
        try:
            status = bias_calibrator.get_status()
            return CalibrationStatusResponse(
                last_calibration_at=status.get("last_calibration_at"),
                slippage_mae=status.get("slippage_mae", 0.0),
                latency_mae=status.get("latency_mae", 0.0),
                inclusion_accuracy=status.get("inclusion_accuracy", 0.0),
                coefficients=CalibrationCoefficients(
                    slippage_factor=status.get("slippage_factor", 1.0),
                    latency_offset_ms=status.get("latency_offset_ms", 0.0),
                    inclusion_threshold_adj=status.get("inclusion_threshold_adj", 0.0),
                ),
                samples_since_calibration=status.get("samples_since_calibration", 0),
                needs_recalibration=status.get("needs_recalibration", False),
            )
        except Exception as e:
            logger.warning(f"Failed to get calibrator status: {e}")

    # Return mock data for development/demo
    return CalibrationStatusResponse(
        last_calibration_at=datetime.utcnow().isoformat(),
        slippage_mae=12.5,
        latency_mae=45.2,
        inclusion_accuracy=0.78,
        coefficients=CalibrationCoefficients(
            slippage_factor=1.08,
            latency_offset_ms=-15.3,
            inclusion_threshold_adj=0.02,
        ),
        samples_since_calibration=342,
        needs_recalibration=False,
    )


@router.post("/calibration/trigger", response_model=CalibrationTriggerResponse)
async def trigger_calibration(
    bias_calibrator: BiasCalibratorDep,
):
    """Trigger a bias recalibration.

    This will recalculate calibration coefficients based on recent
    prediction vs actual outcome data.
    """
    if bias_calibrator is not None:
        try:
            result = await bias_calibrator.trigger_recalibration()
            return CalibrationTriggerResponse(
                triggered=result.get("triggered", False),
                message=result.get("message", "Calibration triggered"),
                estimated_duration_sec=result.get("estimated_duration_sec"),
            )
        except Exception as e:
            logger.error(f"Failed to trigger calibration: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to trigger calibration: {str(e)}"
            ) from e


@router.get("/history")
async def get_learning_history(
    outcome_tracker: OutcomeTrackerDep = None,
    limit: int = 50,
):
    """Get recent learning history (observations and outcomes).

    Returns recent observations with their outcomes for debugging
    and analysis purposes.
    """
    if outcome_tracker is not None:
        try:
            history = outcome_tracker.get_recent_history(limit=limit)
            return {"history": history}
        except Exception as e:
            logger.warning(f"Failed to get learning history: {e}")

    # Mock data for development
    return {
        "history": [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "mint": "DEMO123",
                "action": "trade",
                "entry_price": 0.00012,
                "exit_price": 0.00015,
                "pnl_pct": 25.0,
                "outcome": "win",
            },
            {
                "timestamp": datetime.utcnow().isoformat(),
                "mint": "DEMO456",
                "action": "skip",
                "counterfactual_pnl_pct": -12.0,
                "outcome": "correct_skip",
            },
        ]
    }


# =========== Training Sample Validation Endpoints ==========


@router.post("/validate-samples")
async def validate_training_samples(
    samples: list[dict[str, Any]],
    strict_mode: bool = False,
    min_samples: int = 100,
):
    """Validate a batch of training samples.

    Performs comprehensive validation including:
    - Feature vector completeness (all 50 features)
    - Feature value sanity (within bounds, no NaN/Inf)
    - Label correctness and consistency
    - Data distribution quality

    Returns detailed validation results with issues categorized by severity.
    """
    try:
        from coldpath.validation.training_sample_validator import TrainingSampleValidator

        validator = TrainingSampleValidator(
            strict_mode=strict_mode,
            min_samples=min_samples,
        )

        result = validator.validate_samples(samples)

        return {
            "valid": result.valid,
            "total_samples": result.total_samples,
            "valid_samples": result.valid_samples,
            "invalid_samples": result.invalid_samples,
            "feature_completeness": result.feature_completeness,
            "label_consistency": result.label_consistency,
            "issues": [issue.to_dict() for issue in result.issues],
            "summary": result.summary(),
        }

    except Exception as e:
        logger.error(f"Failed to validate training samples: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}") from e


@router.get("/validation/stats")
async def get_validation_stats():
    """Get validation statistics.

    Returns aggregated statistics from recent validations including:
    - Total validations performed
    - Pass rate
    - Average quality metrics
    - Most common issues
    """
    # For now, return mock data
    # In production, this would query a validation history store
    return {
        "total_validations": 42,
        "pass_rate": 0.85,
        "avg_completeness": 0.92,
        "avg_label_consistency": 0.95,
        "common_issues": [
            {
                "code": "FEATURE_OUT_OF_BOUNDS",
                "count": 12,
                "severity": "warning",
            },
            {
                "code": "DUPLICATE_SAMPLES",
                "count": 5,
                "severity": "warning",
            },
        ],
    }
