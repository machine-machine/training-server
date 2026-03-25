"""
Feedback Loop API Routes.

Exposes the feedback loop pipeline via REST API for:
- Manual triggering
- Health monitoring
- Configuration updates
- Rollback operations
- Scheduler restart (for stuck loops)
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback-loop", tags=["feedback-loop"])


# Global reference to the feedback loop (set during server startup)
_feedback_loop = None
# Global reference to the feedback scheduler
_feedback_scheduler = None


def set_feedback_loop(loop):
    """Set the global feedback loop reference."""
    global _feedback_loop
    _feedback_loop = loop


def set_feedback_scheduler(scheduler):
    """Set the global feedback scheduler reference."""
    global _feedback_scheduler
    _feedback_scheduler = scheduler


class TriggerCycleRequest(BaseModel):
    """Request to trigger a feedback loop cycle."""

    force: bool = Field(default=False, description="Force cycle even if minimum conditions not met")


class UpdateGatesRequest(BaseModel):
    """Request to update evidence gate thresholds."""

    min_total_samples: int | None = None
    min_traded_samples: int | None = None
    min_positive_samples: int | None = None
    min_win_rate: float | None = None
    min_profit_factor: float | None = None
    min_sharpe_ratio: float | None = None
    max_train_oos_gap: float | None = None
    min_oos_auc: float | None = None
    min_fold_consistency: float | None = None
    min_win_rate_improvement: float | None = None
    min_profit_factor_improvement: float | None = None


class PromoteModelRequest(BaseModel):
    """Request to manually promote a model."""

    reason: str = Field(default="manual_promotion", description="Reason for promotion")
    skip_gates: bool = Field(
        default=False, description="Skip evidence gates (dangerous, use with caution)"
    )


class RollbackRequest(BaseModel):
    """Request to rollback to a previous model."""

    target_version: int | None = Field(
        default=None, description="Target version to rollback to (latest if not specified)"
    )
    reason: str = Field(default="manual_rollback", description="Reason for rollback")


@router.get("/health")
async def get_health() -> dict[str, Any]:
    """Get comprehensive feedback loop health report.

    Returns:
        Current state, metrics, gate status, and model history
    """
    if _feedback_loop is None:
        raise HTTPException(status_code=503, detail="Feedback loop not initialized")

    return await _feedback_loop.get_health_report()


@router.get("/status")
async def get_status() -> dict[str, Any]:
    """Get current feedback loop status.

    Returns:
        Simplified status for dashboard display
    """
    if _feedback_loop is None:
        raise HTTPException(status_code=503, detail="Feedback loop not initialized")

    metrics = _feedback_loop.metrics

    return {
        "state": _feedback_loop.state.value,
        "total_outcomes": metrics.total_outcomes,
        "traded_outcomes": metrics.traded_outcomes,
        "positive_outcomes": metrics.positive_outcomes,
        "win_rate": metrics.candidate_win_rate,
        "profit_factor": metrics.candidate_profit_factor,
        "sharpe_ratio": metrics.candidate_sharpe,
        "gates_passed": f"{metrics.gates_passed}/{metrics.gates_total}",
        "last_promotion_version": metrics.promoted_version,
        "last_collection": (
            datetime.fromtimestamp(metrics.last_collection_ms / 1000).isoformat()
            if metrics.last_collection_ms
            else None
        ),
        "last_training": (
            datetime.fromtimestamp(metrics.last_training_ms / 1000).isoformat()
            if metrics.last_training_ms
            else None
        ),
        "last_promotion": (
            datetime.fromtimestamp(metrics.last_promotion_ms / 1000).isoformat()
            if metrics.last_promotion_ms
            else None
        ),
    }


@router.post("/trigger")
async def trigger_cycle(
    request: TriggerCycleRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Manually trigger a feedback loop cycle.

    This will:
    1. Collect outcomes from database
    2. Train candidate model (if enough data)
    3. Evaluate evidence gates
    4. Promote or reject model

    Args:
        request: Trigger options

    Returns:
        Cycle results
    """
    if _feedback_loop is None:
        raise HTTPException(status_code=503, detail="Feedback loop not initialized")

    try:
        # Run cycle in background to avoid blocking
        metrics = await _feedback_loop.run_cycle()

        return {
            "status": "completed",
            "state": _feedback_loop.state.value,
            "decision": metrics.promotion_decision.value,
            "outcomes_processed": metrics.total_outcomes,
            "gates_passed": f"{metrics.gates_passed}/{metrics.gates_total}",
            "candidate_win_rate": metrics.candidate_win_rate,
            "candidate_profit_factor": metrics.candidate_profit_factor,
            "candidate_sharpe": metrics.candidate_sharpe,
            "promoted_version": metrics.promoted_version,
        }

    except Exception as e:
        logger.error(f"Feedback loop cycle failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/scheduler/status")
async def get_scheduler_status() -> dict[str, Any]:
    """Get scheduler status and check if it's actually alive.

    Use this to diagnose if the scheduler is stuck or dead.

    Returns:
        Scheduler status including whether the background task is alive
    """
    if _feedback_scheduler is None:
        return {
            "initialized": False,
            "message": "Feedback scheduler not initialized. Server may need restart.",
        }

    try:
        is_alive = _feedback_scheduler.is_alive()
        status = await _feedback_scheduler.get_status()

        return {
            "initialized": True,
            "is_alive": is_alive,
            "running_flag": _feedback_scheduler._running,
            "has_task": _feedback_scheduler._task is not None,
            "task_done": _feedback_scheduler._task.done() if _feedback_scheduler._task else None,
            **status,
        }
    except Exception as e:
        logger.error(f"Failed to get scheduler status: {e}")
        return {
            "initialized": True,
            "is_alive": False,
            "error": str(e),
        }


@router.post("/restart")
async def restart_scheduler() -> dict[str, Any]:
    """Restart the feedback loop scheduler.

    Use this when the scheduler appears stuck or not running cycles.
    This will:
    1. Stop the current scheduler
    2. Start a fresh scheduler loop

    Returns:
        Restart result with new scheduler status
    """
    if _feedback_scheduler is None:
        raise HTTPException(
            status_code=503, detail="Feedback scheduler not initialized. Cannot restart."
        )

    try:
        result = await _feedback_scheduler.restart()

        return {
            "status": "success",
            "message": "Feedback loop scheduler restarted successfully",
            **result,
        }

    except Exception as e:
        logger.error(f"Failed to restart scheduler: {e}")
        raise HTTPException(status_code=500, detail=f"Restart failed: {str(e)}") from e


@router.get("/gates")
async def get_gates() -> dict[str, Any]:
    """Get current evidence gate configuration.

    Returns:
        All evidence gate thresholds
    """
    if _feedback_loop is None:
        raise HTTPException(status_code=503, detail="Feedback loop not initialized")

    gates = _feedback_loop.gates

    return {
        "min_total_samples": gates.min_total_samples,
        "min_traded_samples": gates.min_traded_samples,
        "min_positive_samples": gates.min_positive_samples,
        "min_win_rate": gates.min_win_rate,
        "min_profit_factor": gates.min_profit_factor,
        "min_sharpe_ratio": gates.min_sharpe_ratio,
        "max_train_oos_gap": gates.max_train_oos_gap,
        "min_oos_auc": gates.min_oos_auc,
        "min_fold_consistency": gates.min_fold_consistency,
        "min_win_rate_improvement": gates.min_win_rate_improvement,
        "min_profit_factor_improvement": gates.min_profit_factor_improvement,
    }


@router.put("/gates")
async def update_gates(request: UpdateGatesRequest) -> dict[str, Any]:
    """Update evidence gate thresholds.

    Args:
        request: New threshold values (only provided fields are updated)

    Returns:
        Updated gate configuration
    """
    if _feedback_loop is None:
        raise HTTPException(status_code=503, detail="Feedback loop not initialized")

    gates = _feedback_loop.gates

    # Update only provided fields
    if request.min_total_samples is not None:
        gates.min_total_samples = request.min_total_samples
    if request.min_traded_samples is not None:
        gates.min_traded_samples = request.min_traded_samples
    if request.min_positive_samples is not None:
        gates.min_positive_samples = request.min_positive_samples
    if request.min_win_rate is not None:
        gates.min_win_rate = request.min_win_rate
    if request.min_profit_factor is not None:
        gates.min_profit_factor = request.min_profit_factor
    if request.min_sharpe_ratio is not None:
        gates.min_sharpe_ratio = request.min_sharpe_ratio
    if request.max_train_oos_gap is not None:
        gates.max_train_oos_gap = request.max_train_oos_gap
    if request.min_oos_auc is not None:
        gates.min_oos_auc = request.min_oos_auc
    if request.min_fold_consistency is not None:
        gates.min_fold_consistency = request.min_fold_consistency
    if request.min_win_rate_improvement is not None:
        gates.min_win_rate_improvement = request.min_win_rate_improvement
    if request.min_profit_factor_improvement is not None:
        gates.min_profit_factor_improvement = request.min_profit_factor_improvement

    logger.info(f"Updated evidence gates: {request.dict(exclude_none=True)}")

    return await get_gates()


@router.post("/promote")
async def promote_model(request: PromoteModelRequest) -> dict[str, Any]:
    """Manually promote the current candidate model.

    WARNING: Skipping gates can lead to model degradation.

    Args:
        request: Promotion options

    Returns:
        Promotion result
    """
    if _feedback_loop is None:
        raise HTTPException(status_code=503, detail="Feedback loop not initialized")

    if request.skip_gates:
        logger.warning("Skipping evidence gates for manual promotion - USE WITH CAUTION")

    try:
        await _feedback_loop._promote_model()

        return {
            "status": "promoted",
            "version": _feedback_loop.metrics.promoted_version,
            "reason": request.reason,
        }

    except Exception as e:
        logger.error(f"Manual promotion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/rollback")
async def rollback_model(request: RollbackRequest) -> dict[str, Any]:
    """Rollback to a previous model version.

    Args:
        request: Rollback options

    Returns:
        Rollback result
    """
    if _feedback_loop is None:
        raise HTTPException(status_code=503, detail="Feedback loop not initialized")

    try:
        await _feedback_loop._rollback_model()

        return {
            "status": "rolled_back",
            "version": _feedback_loop.metrics.rollback_version,
            "reason": request.reason,
        }

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    """Get detailed feedback loop metrics.

    Returns:
        All metrics including gate results
    """
    if _feedback_loop is None:
        raise HTTPException(status_code=503, detail="Feedback loop not initialized")

    metrics = _feedback_loop.metrics

    return {
        "outcomes": {
            "total": metrics.total_outcomes,
            "traded": metrics.traded_outcomes,
            "skipped": metrics.skipped_outcomes,
            "positive": metrics.positive_outcomes,
        },
        "baseline": {
            "win_rate": metrics.baseline_win_rate,
            "profit_factor": metrics.baseline_profit_factor,
            "sharpe": metrics.baseline_sharpe,
        },
        "candidate": {
            "win_rate": metrics.candidate_win_rate,
            "profit_factor": metrics.candidate_profit_factor,
            "sharpe": metrics.candidate_sharpe,
            "oos_auc": metrics.candidate_oos_auc,
        },
        "gates": {
            "passed": metrics.gates_passed,
            "total": metrics.gates_total,
            "details": metrics.gate_details,
        },
        "promotion": {
            "decision": metrics.promotion_decision.value,
            "promoted_version": metrics.promoted_version,
            "rollback_version": metrics.rollback_version,
        },
        "timestamps": {
            "last_collection_ms": metrics.last_collection_ms,
            "last_training_ms": metrics.last_training_ms,
            "last_promotion_ms": metrics.last_promotion_ms,
        },
    }


@router.get("/history")
async def get_model_history() -> dict[str, Any]:
    """Get model promotion history.

    Returns:
        List of historical model promotions
    """
    if _feedback_loop is None:
        raise HTTPException(status_code=503, detail="Feedback loop not initialized")

    return {
        "history": _feedback_loop._model_history,
        "total_versions": len(_feedback_loop._model_history),
    }
