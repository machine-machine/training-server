"""
Optimization API routes for AI-driven strategy optimization.

Provides RESTful endpoints for:
- Triggering strategy optimization
- Querying optimization status
- Applying/rejecting optimization proposals
- Manual rollback
- Submitting trade feedback for learning
- Querying drift detection status
- Parameter effectiveness reports
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["optimization"])


# ---- Request/Response Models ----


class OptimizeStrategyRequest(BaseModel):
    """Request to trigger strategy optimization."""

    goal: str = Field(
        description="Optimization objective",
        examples=["maximize_sharpe", "minimize_drawdown"],
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Hard constraints to respect",
    )
    urgency: str = Field(
        default="normal",
        description="Priority level: low, normal, high",
    )
    reason: str = Field(
        default="manual",
        description="Why optimization was triggered",
    )


class OptimizeStrategyResponse(BaseModel):
    """Response from optimization trigger."""

    optimization_id: str
    status: str
    message: str


class ApplyOptimizationRequest(BaseModel):
    """Request to approve/reject an optimization."""

    approved: bool = Field(description="Whether to approve")
    approved_by: str = Field(default="user", description="Who approved")


class RollbackRequest(BaseModel):
    """Request to rollback an optimization."""

    reason: str = Field(description="Reason for rollback")


class TradeFeedbackRequest(BaseModel):
    """Trade outcome for learning."""

    trade_id: str
    pnl_sol: float
    pnl_pct: float
    fees_sol: float = 0.0
    slippage_bps: int = 0
    entry_price: float
    exit_price: float
    exit_reason: str = "unknown"
    hold_duration_seconds: int = 0
    regime: str = "unknown"
    volatility: float = 0.0
    liquidity_at_entry: float = 0.0
    ml_confidence: float | None = None
    source: str = "live"
    params_version: int = 0
    params_snapshot: dict[str, Any] = Field(default_factory=dict)


# ---- Helper to get components ----


def _get_orchestrator() -> Any:
    """Get the AIStrategyOrchestrator from app state."""
    try:
        from coldpath.api.server import get_app_state

        state = get_app_state()
        orchestrator = getattr(state, "strategy_orchestrator", None)
        if orchestrator is None:
            raise HTTPException(
                status_code=503,
                detail="Strategy orchestrator not initialized",
            )
        return orchestrator
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Server module not available",
        ) from None


def _get_performance_engine() -> Any:
    """Get the PerformanceFeedbackEngine from app state."""
    try:
        from coldpath.api.server import get_app_state

        state = get_app_state()
        engine = getattr(state, "performance_engine_ai", None)
        if engine is None:
            raise HTTPException(
                status_code=503,
                detail="Performance engine not initialized",
            )
        return engine
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Server module not available",
        ) from None


def _get_drift_detector() -> Any:
    """Get the DriftDetector from app state."""
    try:
        from coldpath.api.server import get_app_state

        state = get_app_state()
        detector = getattr(state, "drift_detector", None)
        if detector is None:
            raise HTTPException(
                status_code=503,
                detail="Drift detector not initialized",
            )
        return detector
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Server module not available",
        ) from None


def _get_parameter_store() -> Any:
    """Get the ParameterStore from app state."""
    try:
        from coldpath.api.server import get_app_state

        state = get_app_state()
        store = getattr(state, "parameter_store", None)
        if store is None:
            raise HTTPException(
                status_code=503,
                detail="Parameter store not initialized",
            )
        return store
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Server module not available",
        ) from None


def _get_parameter_tracker() -> Any:
    """Get the ParameterEffectivenessTracker from app state."""
    try:
        from coldpath.api.server import get_app_state

        state = get_app_state()
        tracker = getattr(state, "parameter_tracker", None)
        if tracker is None:
            raise HTTPException(
                status_code=503,
                detail="Parameter tracker not initialized",
            )
        return tracker
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Server module not available",
        ) from None


# ---- Optimization Endpoints ----


@router.post("/optimize-strategy", response_model=OptimizeStrategyResponse)
async def optimize_strategy(request: OptimizeStrategyRequest) -> dict[str, Any]:
    """
    Trigger a new strategy optimization.

    Starts the full optimization pipeline:
    analyzing -> generating -> backtesting -> validating -> deploying

    Returns immediately with an optimization_id for status polling.
    """
    orchestrator = _get_orchestrator()

    try:
        optimization_id = await orchestrator.trigger_optimization(
            reason=request.reason,
            goal=request.goal,
            constraints=request.constraints,
            urgency=request.urgency,
        )
        return {
            "optimization_id": optimization_id,
            "status": "started",
            "message": f"Optimization {optimization_id} started",
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to trigger optimization: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/optimization/{optimization_id}")
async def get_optimization_status(optimization_id: str) -> dict[str, Any]:
    """
    Get optimization status and progress.

    Returns current stage, progress, and results so far.
    """
    orchestrator = _get_orchestrator()

    status = await orchestrator.get_status(optimization_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Optimization {optimization_id} not found",
        )

    return status


@router.post("/optimization/{optimization_id}/apply")
async def apply_optimization(
    optimization_id: str,
    request: ApplyOptimizationRequest,
) -> dict[str, Any]:
    """
    Apply or reject a pending optimization.

    Only works when optimization is in AWAITING_APPROVAL stage.
    """
    orchestrator = _get_orchestrator()

    success = await orchestrator.apply_approval(
        optimization_id=optimization_id,
        approved=request.approved,
        approved_by=request.approved_by,
    )

    if not success:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot apply approval for {optimization_id}. "
                f"Optimization may not be in awaiting_approval stage."
            ),
        )

    action = "approved" if request.approved else "rejected"
    return {
        "optimization_id": optimization_id,
        "action": action,
        "approved_by": request.approved_by,
        "message": f"Optimization {optimization_id} {action}",
    }


@router.post("/optimization/{optimization_id}/rollback")
async def rollback_optimization(
    optimization_id: str,
    request: RollbackRequest,
) -> dict[str, Any]:
    """
    Manually rollback an optimization to previous parameters.
    """
    try:
        from coldpath.api.server import get_app_state

        state = get_app_state()
        rollback_mgr = getattr(state, "rollback_manager", None)

        if rollback_mgr:
            success = await rollback_mgr.rollback(
                deployment_id=optimization_id,
                reason=request.reason,
            )
        else:
            # Direct rollback via parameter store
            store = _get_parameter_store()
            await store.rollback(reason=request.reason)
            success = True

        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Rollback failed for {optimization_id}",
            )

        return {
            "optimization_id": optimization_id,
            "status": "rolled_back",
            "reason": request.reason,
            "message": f"Optimization {optimization_id} rolled back",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Rollback failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/optimization/history")
async def get_optimization_history() -> dict[str, Any]:
    """Get history of all optimization runs."""
    orchestrator = _get_orchestrator()
    history = orchestrator.get_optimization_history()
    return {
        "total": len(history),
        "optimizations": history,
    }


# ---- Trade Feedback Endpoints ----


@router.post("/trade-feedback")
async def submit_trade_feedback(
    request: TradeFeedbackRequest,
) -> dict[str, Any]:
    """
    Submit a trade outcome for learning.

    The performance engine processes the outcome to:
    - Update regime-specific performance metrics
    - Update parameter effectiveness scores
    - Check for performance drift
    - Trigger optimization if degradation detected
    """
    engine = _get_performance_engine()

    from coldpath.ai.performance_engine import TradeOutcome

    outcome = TradeOutcome(
        trade_id=request.trade_id,
        timestamp=datetime.now(),
        params_version=request.params_version,
        params_snapshot=request.params_snapshot,
        pnl_sol=request.pnl_sol,
        pnl_pct=request.pnl_pct,
        fees_sol=request.fees_sol,
        slippage_bps=request.slippage_bps,
        entry_price=request.entry_price,
        exit_price=request.exit_price,
        exit_reason=request.exit_reason,
        hold_duration_seconds=request.hold_duration_seconds,
        regime=request.regime,
        volatility=request.volatility,
        liquidity_at_entry=request.liquidity_at_entry,
        ml_confidence=request.ml_confidence,
        source=request.source,
    )

    result = await engine.process_trade_outcome(outcome)

    return {
        "trade_id": result.trade_id,
        "processed": result.processed,
        "drift_detected": result.drift_detected,
        "optimization_triggered": result.optimization_triggered,
        "details": result.details,
    }


# ---- Performance & Drift Endpoints ----


@router.get("/performance")
async def get_performance_summary() -> dict[str, Any]:
    """
    Get current performance summary including regime breakdown.
    """
    engine = _get_performance_engine()
    return await engine.analyze()


@router.get("/performance/regime/{regime}")
async def get_regime_performance(regime: str) -> dict[str, Any]:
    """Get performance metrics for a specific market regime."""
    engine = _get_performance_engine()
    return await engine.get_regime_performance(regime=regime)


@router.get("/drift")
async def get_drift_status() -> dict[str, Any]:
    """
    Get the latest drift detection report.
    """
    detector = _get_drift_detector()
    report = detector.last_report
    if report is None:
        return {
            "status": "no_report",
            "message": "No drift detection has been run yet",
        }
    return report.to_dict()


# ---- Parameter Endpoints ----


@router.get("/parameters/current")
async def get_current_parameters() -> dict[str, Any]:
    """Get currently deployed strategy parameters."""
    store = _get_parameter_store()
    version = await store.get_current_version()
    return {
        "version": version.version,
        "timestamp": version.timestamp.isoformat(),
        "deployed_by": version.deployed_by,
        "deployment_reason": version.deployment_reason,
        "params": version.params,
    }


@router.get("/parameters/history")
async def get_parameter_history() -> dict[str, Any]:
    """Get parameter version history."""
    store = _get_parameter_store()
    history = await store.get_history()
    return {
        "current_version": store.current_version_number,
        "history_count": len(history),
        "versions": [
            {
                "version": v.version,
                "timestamp": v.timestamp.isoformat(),
                "deployed_by": v.deployed_by,
                "reason": v.deployment_reason,
            }
            for v in history
        ],
    }


@router.get("/parameters/effectiveness/{parameter_name}")
async def get_parameter_effectiveness(
    parameter_name: str,
) -> dict[str, Any]:
    """Get effectiveness report for a specific parameter."""
    tracker = _get_parameter_tracker()
    return await tracker.get_parameter_report(parameter_name)


# ---- Phase 2: Advanced AI Integration Endpoints ----


def _get_cost_tracker() -> Any:
    """Get the CostTracker from app state."""
    try:
        from coldpath.api.server import get_app_state

        state = get_app_state()
        tracker = getattr(state, "cost_tracker", None)
        if tracker is None:
            raise HTTPException(
                status_code=503,
                detail="Cost tracker not initialized",
            )
        return tracker
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Server module not available",
        ) from None


def _get_task_router() -> Any:
    """Get the TaskComplexityRouter from app state."""
    try:
        from coldpath.api.server import get_app_state

        state = get_app_state()
        router = getattr(state, "task_router", None)
        if router is None:
            raise HTTPException(
                status_code=503,
                detail="Task router not initialized",
            )
        return router
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Server module not available",
        ) from None


@router.get("/cost-analysis")
async def get_cost_analysis() -> dict[str, Any]:
    """
    Get Claude API cost analysis for the current month.

    Returns monthly cost breakdown by model, task type, and day.
    Includes budget utilization and optimization recommendations.
    """
    tracker = _get_cost_tracker()
    monthly = await tracker.get_monthly_cost()
    recommendations = await tracker.get_cost_recommendations()
    return {
        **monthly,
        "recommendations": recommendations,
        "stats": tracker.get_stats(),
    }


@router.get("/cost-analysis/{optimization_id}")
async def get_optimization_cost(
    optimization_id: str,
) -> dict[str, Any]:
    """Get cost breakdown for a specific optimization run."""
    tracker = _get_cost_tracker()
    return await tracker.get_optimization_breakdown(optimization_id)


@router.get("/routing-stats")
async def get_routing_stats() -> dict[str, Any]:
    """
    Get task complexity routing statistics.

    Shows model usage breakdown, average complexity, and routing decisions.
    """
    router = _get_task_router()
    return router.get_routing_stats()


class MultiObjectiveRequest(BaseModel):
    """Request for multi-objective optimization."""

    objectives: list[dict[str, Any]] = Field(
        description="List of objectives: [{name, maximize, weight}]"
    )
    param_bounds: dict[str, list[float]] = Field(description="Parameter bounds: {name: [min, max]}")
    initial_params: dict[str, float] = Field(
        default_factory=dict,
        description="Initial parameter values",
    )
    population_size: int = Field(
        default=50,
        description="Population size for NSGA-II",
    )
    generations: int = Field(
        default=20,
        description="Number of generations",
    )


@router.post("/multi-objective-optimize")
async def multi_objective_optimize(
    request: MultiObjectiveRequest,
) -> dict[str, Any]:
    """
    Run multi-objective optimization using NSGA-II.

    Returns Pareto frontier of non-dominated solutions.
    """
    try:
        from coldpath.ai.multi_objective_optimizer import (
            MultiObjectiveOptimizer,
            Objective,
        )

        objectives = [
            Objective(
                name=obj["name"],
                maximize=obj.get("maximize", True),
                weight=obj.get("weight", 1.0),
            )
            for obj in request.objectives
        ]

        param_bounds = {k: (v[0], v[1]) for k, v in request.param_bounds.items()}

        optimizer = MultiObjectiveOptimizer(
            objectives=objectives,
            param_bounds=param_bounds,
            population_size=request.population_size,
            generations=request.generations,
        )

        # Use internal simulation for evaluation
        from coldpath.backtest.stress_test_engine import StressTestEngine

        engine = StressTestEngine()

        async def evaluate(params):
            result = engine._simulate_backtest(params, [])
            return result

        pareto = await optimizer.optimize(
            initial_params=request.initial_params or {},
            evaluate_fn=evaluate,
        )

        # Select best compromise
        best = optimizer.select_best_compromise(pareto)

        return {
            "pareto_frontier": [s.to_dict() for s in pareto],
            "pareto_size": len(pareto),
            "best_compromise": best.to_dict() if best else None,
        }

    except Exception as exc:
        logger.error("Multi-objective optimization failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class InteractionAnalysisRequest(BaseModel):
    """Request for parameter interaction analysis."""

    params: dict[str, Any] = Field(description="Strategy parameters to analyze")


@router.post("/analyze-interactions")
async def analyze_interactions(
    request: InteractionAnalysisRequest,
) -> dict[str, Any]:
    """
    Analyze parameter interactions (synergies, conflicts, dominances).
    """
    try:
        from coldpath.ai.interaction_analyzer import (
            ParameterInteractionAnalyzer,
        )

        analyzer = ParameterInteractionAnalyzer()
        report = await analyzer.analyze_interactions(request.params)
        return report.to_dict()

    except Exception as exc:
        logger.error("Interaction analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
