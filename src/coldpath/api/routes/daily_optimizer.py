"""
Daily Optimizer API Routes - Control endpoints for automated optimization.

Provides REST API for:
- Starting/stopping the daily optimizer
- Getting optimization status and results
- Managing optimization profiles
- Viewing optimization history
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from coldpath.backtest.daily_optimizer import (
    DEFAULT_PROFILES,
    DailyOptimizer,
    OptimizationFrequency,
    create_daily_optimizer,
)
from coldpath.backtest.profit_config import (
    SCENARIO_RECOMMENDATIONS,
    compare_configs,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/optimizer", tags=["Daily Optimizer"])


# Global optimizer instance
_optimizer: DailyOptimizer | None = None


class StartOptimizerRequest(BaseModel):
    """Request to start the daily optimizer."""

    profile_name: str = Field(
        default="profit_maximizer", description="Name of optimization profile to use"
    )
    frequency: str = Field(
        default="daily",
        description="Optimization frequency: hourly, every_6_hours, every_12_hours, daily, weekly",
    )
    schedule_time: str = Field(default="06:00", description="Time for daily runs (HH:MM format)")
    run_immediately: bool = Field(
        default=False, description="Run optimization immediately after starting"
    )
    quick_mode: bool = Field(
        default=True, description="Use quick mode (synthetic data) for faster optimization"
    )


class OptimizerStatusResponse(BaseModel):
    """Response with optimizer status."""

    running: bool
    profile: str | None = None
    frequency: str | None = None
    schedule_time: str | None = None
    last_run: str | None = None
    next_run: str | None = None
    best_score: float = 0.0
    total_runs: int = 0
    successful_runs: int = 0


class OptimizerResultResponse(BaseModel):
    """Response with optimization result."""

    timestamp: str
    profile_name: str
    success: bool
    best_params: dict[str, Any]
    best_score: float
    sharpe_ratio: float
    win_rate_pct: float
    max_drawdown_pct: float
    robustness_score: float | None = None
    is_robust: bool | None = None
    improvement_over_previous: float = 0.0


class ProfileListResponse(BaseModel):
    """Response with available profiles."""

    profiles: list[dict[str, Any]]
    default: str


class ScenarioListResponse(BaseModel):
    """Response with scenario recommendations."""

    scenarios: dict[str, dict[str, Any]]


async def _create_backtest_runner(quick_mode: bool = True):
    """Create a backtest runner function for optimization."""
    from coldpath.backtest.engine import BacktestConfig, BacktestEngine

    engine = BacktestEngine()

    async def run_backtest(params: dict[str, Any]) -> dict[str, float]:
        """Run a single backtest with given params."""
        try:
            now = datetime.utcnow()
            lookback_days = 7 if quick_mode else 30
            start_ts = int((now - timedelta(days=lookback_days)).timestamp() * 1000)
            end_ts = int(now.timestamp() * 1000)

            config = BacktestConfig(
                start_timestamp_ms=start_ts,
                end_timestamp_ms=end_ts,
                initial_capital_sol=params.get("initial_capital_sol", 1.0),
                max_position_sol=params.get("max_position_sol", 0.05),
                slippage_bps=params.get("slippage_bps", 300),
                default_stop_loss_pct=params.get("stop_loss_pct", 8.0),
                default_take_profit_pct=params.get("take_profit_pct", 20.0),
                min_liquidity_usd=params.get("min_liquidity_usd", 10000),
            )

            data_source = "synthetic" if quick_mode else "bitquery_ohlcv"
            result = await engine.run(
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                data_source=data_source,
                config=config,
            )

            return result.get("metrics", {})

        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            return {
                "sharpe_ratio": 0.0,
                "win_rate_pct": 0.0,
                "max_drawdown_pct": 100.0,
                "total_return_pct": -100.0,
                "profit_factor": 0.0,
                "profit_quality_score": 0.0,
            }

    return run_backtest


def get_optimizer() -> DailyOptimizer | None:
    """Get the global optimizer instance."""
    return _optimizer


async def _ensure_optimizer(quick_mode: bool = True) -> DailyOptimizer:
    """Ensure optimizer is initialized and return it."""
    global _optimizer

    if _optimizer is None:
        runner = await _create_backtest_runner(quick_mode=quick_mode)
        _optimizer = create_daily_optimizer(
            backtest_runner=runner,
            profile_name="profit_maximizer",
            frequency="daily",
            schedule_time="06:00",
            storage_path="./data/optimization_history",
        )
        logger.info("Daily optimizer initialized lazily")

    return _optimizer


@router.post("/start", response_model=dict[str, Any])
async def start_optimizer(request: StartOptimizerRequest):
    """Start the daily optimizer scheduler.

    This will begin running optimizations according to the specified schedule.
    """
    global _optimizer

    optimizer = await _ensure_optimizer(quick_mode=request.quick_mode)

    if optimizer._running:
        return {
            "status": "already_running",
            "message": "Optimizer is already running",
            "profile": optimizer.profile.name,
            "frequency": optimizer.frequency.value,
        }

    # Update profile if specified
    if request.profile_name in DEFAULT_PROFILES:
        optimizer.profile = DEFAULT_PROFILES[request.profile_name]

    # Update frequency
    freq_map = {
        "hourly": OptimizationFrequency.HOURLY,
        "every_6_hours": OptimizationFrequency.EVERY_6_HOURS,
        "every_12_hours": OptimizationFrequency.EVERY_12_HOURS,
        "daily": OptimizationFrequency.DAILY,
        "weekly": OptimizationFrequency.WEEKLY,
    }
    optimizer.frequency = freq_map.get(request.frequency, OptimizationFrequency.DAILY)
    optimizer.schedule_time = request.schedule_time

    # Start the optimizer
    await optimizer.start()

    # Run immediately if requested
    if request.run_immediately:
        asyncio.create_task(optimizer.run_optimization())

    return {
        "status": "started",
        "profile": optimizer.profile.name,
        "frequency": optimizer.frequency.value,
        "schedule_time": request.schedule_time,
        "run_immediately": request.run_immediately,
    }


@router.post("/stop", response_model=dict[str, Any])
async def stop_optimizer():
    """Stop the daily optimizer scheduler."""
    global _optimizer

    if _optimizer is None or not _optimizer._running:
        return {"status": "not_running", "message": "Optimizer is not running"}

    await _optimizer.stop()

    return {"status": "stopped", "message": "Optimizer stopped successfully"}


@router.get("/status", response_model=OptimizerStatusResponse)
async def get_optimizer_status():
    """Get the current status of the daily optimizer."""
    global _optimizer

    if _optimizer is None:
        return OptimizerStatusResponse(running=False)

    stats = _optimizer.get_statistics()

    # Calculate next run time
    next_run = None
    if _optimizer._running:
        try:
            next_run_dt = _optimizer._get_next_run_time()
            next_run = next_run_dt.isoformat()
        except Exception:
            pass

    return OptimizerStatusResponse(
        running=_optimizer._running,
        profile=stats.get("profile"),
        frequency=stats.get("frequency"),
        schedule_time=_optimizer.schedule_time,
        last_run=stats.get("last_run"),
        next_run=next_run,
        best_score=stats.get("best_score", 0.0),
        total_runs=stats.get("total_runs", 0),
        successful_runs=stats.get("successful_runs", 0),
    )


@router.post("/run-now", response_model=OptimizerResultResponse)
async def run_optimization_now(quick_mode: bool = True):
    """Run an optimization immediately (outside of schedule)."""
    global _optimizer

    optimizer = await _ensure_optimizer(quick_mode=quick_mode)

    result = await optimizer.run_optimization()

    return OptimizerResultResponse(
        timestamp=result.timestamp,
        profile_name=result.profile_name,
        success=result.success,
        best_params=result.best_params,
        best_score=result.best_score,
        sharpe_ratio=result.sharpe_ratio,
        win_rate_pct=result.win_rate_pct,
        max_drawdown_pct=result.max_drawdown_pct,
        robustness_score=result.robustness_score,
        is_robust=result.is_robust,
        improvement_over_previous=result.improvement_over_previous,
    )


@router.get("/best-params", response_model=dict[str, Any])
async def get_best_params():
    """Get the best parameters found so far."""
    global _optimizer

    if _optimizer is None:
        raise HTTPException(
            status_code=503, detail="Optimizer not initialized. Run an optimization first."
        )

    params = _optimizer.get_best_params()
    stats = _optimizer.get_statistics()

    return {
        "params": params,
        "best_score": stats.get("best_score", 0.0),
        "total_optimizations": stats.get("total_runs", 0),
        "avg_sharpe": stats.get("avg_sharpe", 0.0),
        "avg_win_rate": stats.get("avg_win_rate", 0.0),
        "last_updated": datetime.utcnow().isoformat(),
    }


@router.get("/history", response_model=list[OptimizerResultResponse])
async def get_optimization_history(limit: int = 20):
    """Get the optimization history."""
    global _optimizer

    if _optimizer is None:
        return []

    history = _optimizer.get_history(limit=limit)

    return [
        OptimizerResultResponse(
            timestamp=r.timestamp,
            profile_name=r.profile_name,
            success=r.success,
            best_params=r.best_params,
            best_score=r.best_score,
            sharpe_ratio=r.sharpe_ratio,
            win_rate_pct=r.win_rate_pct,
            max_drawdown_pct=r.max_drawdown_pct,
            robustness_score=r.robustness_score,
            is_robust=r.is_robust,
            improvement_over_previous=r.improvement_over_previous,
        )
        for r in history
    ]


@router.get("/profiles", response_model=ProfileListResponse)
async def list_profiles():
    """List available optimization profiles."""
    profiles = []
    for name, profile in DEFAULT_PROFILES.items():
        profiles.append(
            {
                "name": name,
                "risk_tolerance": profile.risk_tolerance,
                "primary_goal": profile.primary_goal,
                "max_iterations": profile.max_iterations,
                "max_time_minutes": profile.max_time_minutes,
                "enable_robustness_validation": profile.enable_robustness_validation,
                "quick_mode": profile.quick_mode,
            }
        )

    return ProfileListResponse(profiles=profiles, default="profit_maximizer")


@router.post("/profile/{profile_name}")
async def set_profile(profile_name: str):
    """Set the active optimization profile."""
    global _optimizer

    if _optimizer is None:
        raise HTTPException(
            status_code=503, detail="Optimizer not initialized. Run an optimization first."
        )

    if profile_name not in DEFAULT_PROFILES:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Profile '{profile_name}' not found. Available: {list(DEFAULT_PROFILES.keys())}"
            ),
        )

    _optimizer.set_profile(DEFAULT_PROFILES[profile_name])

    return {
        "status": "updated",
        "profile": profile_name,
    }


@router.get("/scenarios", response_model=ScenarioListResponse)
async def list_scenarios():
    """List scenario-based recommendations."""
    scenarios = {}
    for name, rec in SCENARIO_RECOMMENDATIONS.items():
        primary = rec.get("primary")
        scenarios[name] = {
            "recommended_profile": primary.name if primary else None,
            "description": rec.get("description", ""),
            "expected_sharpe": primary.expected_sharpe if primary else None,
            "expected_max_dd": primary.expected_max_dd if primary else None,
        }

    return ScenarioListResponse(scenarios=scenarios)


@router.get("/compare")
async def compare_all_configs():
    """Compare all profit configurations."""
    return {
        "comparison": compare_configs(),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/statistics")
async def get_detailed_statistics():
    """Get detailed optimizer statistics."""
    global _optimizer

    if _optimizer is None:
        return {
            "initialized": False,
            "running": False,
        }

    stats = _optimizer.get_statistics()
    stats["initialized"] = True
    stats["timestamp"] = datetime.utcnow().isoformat()

    return stats


@router.post("/run-all-profiles")
async def run_all_profiles(quick_mode: bool = True):
    """Run optimization for all profiles and compare results."""
    global _optimizer

    optimizer = await _ensure_optimizer(quick_mode=quick_mode)

    # This is a long-running operation
    results = await optimizer.run_all_profiles()

    # Find the best
    best_name = None
    best_score = -float("inf")
    summary = {}

    for name, result in results.items():
        summary[name] = {
            "success": result.success,
            "best_score": result.best_score,
            "sharpe_ratio": result.sharpe_ratio,
            "win_rate_pct": result.win_rate_pct,
            "is_robust": result.is_robust,
        }
        if result.best_score > best_score:
            best_score = result.best_score
            best_name = name

    return {
        "best_profile": best_name,
        "best_score": best_score,
        "summary": summary,
        "total_profiles_run": len(results),
    }
