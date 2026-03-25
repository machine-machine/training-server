"""
Backtest routes for running and retrieving backtests.

Endpoints:
- POST /backtests - Run a new backtest
- GET /backtests/{id} - Get backtest results
- GET /backtests - List recent backtests
"""

import logging
from datetime import datetime
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtests", tags=["backtests"])


# In-memory storage for backtests (replace with database in production)
_backtest_store: dict[str, dict[str, Any]] = {}


# Request/Response Models


class BacktestParams(BaseModel):
    """Parameters for backtest."""

    stop_loss_pct: float = Field(
        default=8.0, ge=0.0, le=100.0, description="Stop loss percentage (0-100)"
    )
    take_profit_pct: float = Field(
        default=15.0, ge=0.0, le=1000.0, description="Take profit percentage (0-1000)"
    )
    max_hold_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,  # 24 hours max
        description="Max hold time in minutes",
    )
    min_liquidity_usd: float = Field(default=10000, ge=0, description="Min liquidity USD")
    max_fdv_usd: float = Field(
        default=5000000,
        ge=0,
        le=1_000_000_000,  # 1B max
        description="Max FDV USD",
    )
    max_risk_score: float = Field(default=0.45, ge=0.0, le=1.0, description="Max risk score (0-1)")
    max_position_sol: float = Field(
        default=0.05, ge=0.0001, le=10000.0, description="Max position SOL"
    )
    slippage_bps: int = Field(
        default=300,
        ge=0,
        le=10000,  # 100% max
        description="Slippage tolerance in bps",
    )


class BacktestRequest(BaseModel):
    """Request to run a backtest."""

    baseline_params: BacktestParams = Field(..., description="Baseline strategy parameters")
    candidate_params: BacktestParams | None = Field(
        None, description="Candidate parameters to compare (if None, baseline only)"
    )
    initial_capital_sol: float = Field(default=1.0, ge=0.01, description="Initial capital in SOL")
    lookback_days: int = Field(default=7, ge=1, le=90, description="Days of historical data to use")
    quick_mode: bool = Field(
        default=True, description="Quick mode uses sampling for faster results"
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**31 - 1,
        description="Optional RNG seed for deterministic backtests (0 to 2^31-1)",
    )


class BacktestMetrics(BaseModel):
    """Metrics from a backtest run."""

    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    avg_drawdown_pct: float
    volatility_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    fill_rate_pct: float
    avg_slippage_bps: float
    expectancy_pct: float | None = None
    estimated_profit_factor: float | None = None
    profit_quality_score: float | None = None
    compute_backend: str | None = None


class BacktestComparison(BaseModel):
    """Comparison between baseline and candidate."""

    metric: str
    baseline: str
    candidate: str
    delta: str
    is_positive: bool


class BacktestResult(BaseModel):
    """Full backtest result."""

    id: str
    status: str  # "pending", "running", "completed", "failed"
    created_at: str
    completed_at: str | None = None
    progress: float | None = None  # 0.0 to 1.0
    baseline_metrics: BacktestMetrics | None = None
    candidate_metrics: BacktestMetrics | None = None
    comparisons: list[BacktestComparison] | None = None
    summary: str | None = None
    error: str | None = None


class BacktestListItem(BaseModel):
    """Summary for backtest list."""

    id: str
    status: str
    created_at: str
    summary: str | None = None


# Dependency injection


def get_metrics_engine():
    """Get metrics engine instance."""
    from coldpath.api.server import get_app_state

    return get_app_state().metrics_engine


# Endpoints


MetricsEngineDep = Annotated[Any, Depends(get_metrics_engine)]


@router.post("", response_model=BacktestResult)
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    metrics_engine: MetricsEngineDep,
):
    """Run a new backtest.

    In quick mode, returns immediately with pending status and runs
    backtest in background. In non-quick mode, waits for completion.
    """
    # Evict oldest entries when store exceeds 100
    if len(_backtest_store) > 100:
        sorted_ids = sorted(
            _backtest_store.keys(),
            key=lambda k: _backtest_store[k].get("created_at", ""),
        )
        for old_id in sorted_ids[: len(_backtest_store) - 100]:
            del _backtest_store[old_id]

    backtest_id = str(uuid4())
    created_at = datetime.utcnow().isoformat()

    # Store initial state
    _backtest_store[backtest_id] = {
        "id": backtest_id,
        "status": "pending",
        "created_at": created_at,
        "request": request.model_dump(),
    }

    if request.quick_mode:
        # Run in background
        background_tasks.add_task(
            _run_backtest_async,
            backtest_id,
            request,
            metrics_engine,
        )

        return BacktestResult(
            id=backtest_id,
            status="pending",
            created_at=created_at,
        )
    else:
        # Run synchronously
        await _run_backtest_async(backtest_id, request, metrics_engine)
        return _get_backtest_result(backtest_id)


@router.get("/{backtest_id}", response_model=BacktestResult)
async def get_backtest(backtest_id: str):
    """Get backtest results by ID."""
    if backtest_id not in _backtest_store:
        raise HTTPException(status_code=404, detail="Backtest not found")

    return _get_backtest_result(backtest_id)


@router.get("", response_model=list[BacktestListItem])
async def list_backtests(
    limit: int = 10,
    status: str | None = None,
):
    """List recent backtests."""
    items = []

    for backtest_id, data in sorted(
        _backtest_store.items(),
        key=lambda x: x[1].get("created_at", ""),
        reverse=True,
    ):
        if status and data.get("status") != status:
            continue

        items.append(
            BacktestListItem(
                id=backtest_id,
                status=data.get("status", "unknown"),
                created_at=data.get("created_at", ""),
                summary=data.get("summary"),
            )
        )

        if len(items) >= limit:
            break

    return items


async def _run_backtest_async(
    backtest_id: str,
    request: BacktestRequest,
    metrics_engine,
):
    """Run backtest asynchronously using the real BacktestEngine."""
    try:
        from coldpath.backtest.engine import BacktestConfig, BacktestEngine

        # Update status
        _backtest_store[backtest_id]["status"] = "running"
        _backtest_store[backtest_id]["progress"] = 0.0

        # Calculate timestamps
        now_ms = int(datetime.utcnow().timestamp() * 1000)
        lookback_ms = request.lookback_days * 24 * 60 * 60 * 1000
        start_timestamp = now_ms - lookback_ms
        end_timestamp = now_ms

        # Build configuration from request params
        baseline_config = BacktestConfig(
            start_timestamp_ms=start_timestamp,
            end_timestamp_ms=end_timestamp,
            initial_capital_sol=request.initial_capital_sol,
            max_position_sol=request.baseline_params.max_position_sol,
            slippage_bps=request.baseline_params.slippage_bps,
            default_stop_loss_pct=request.baseline_params.stop_loss_pct,
            default_take_profit_pct=request.baseline_params.take_profit_pct,
            min_liquidity_usd=request.baseline_params.min_liquidity_usd,
            random_seed=request.seed,
            use_live_signal_generation=True,
        )

        # Create engine with metrics
        engine = BacktestEngine()
        engine.metrics_engine = metrics_engine

        # Run baseline backtest
        logger.info(f"Running baseline backtest {backtest_id}")
        _backtest_store[backtest_id]["progress"] = 0.1

        try:
            # Use persisted local telemetry for full runs to reuse real-market history.
            data_source = "synthetic" if request.quick_mode else "sqlite_telemetry"
            baseline_result = await engine.run(
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                data_source=data_source,
                config=baseline_config,
            )
            baseline_metrics_dict = baseline_result.get("metrics", {})
        except Exception as engine_error:
            logger.warning(f"Engine backtest failed, falling back to mock: {engine_error}")
            # Fallback to mock data if engine fails
            baseline_trades = _generate_mock_trades(
                params=request.baseline_params,
                initial_capital=request.initial_capital_sol,
                num_trades=50 if request.quick_mode else 200,
                seed=request.seed,
            )
            baseline_metrics_obj = metrics_engine.calculate(
                trades=baseline_trades,
                initial_capital=request.initial_capital_sol,
            )
            baseline_metrics_dict = baseline_metrics_obj.to_dict()

        _backtest_store[backtest_id]["progress"] = 0.5

        # If candidate params provided, run comparison
        candidate_metrics_dict = None
        comparisons = None
        summary = None

        if request.candidate_params:
            logger.info(f"Running candidate backtest {backtest_id}")

            candidate_config = BacktestConfig(
                start_timestamp_ms=start_timestamp,
                end_timestamp_ms=end_timestamp,
                initial_capital_sol=request.initial_capital_sol,
                max_position_sol=request.candidate_params.max_position_sol,
                slippage_bps=request.candidate_params.slippage_bps,
                default_stop_loss_pct=request.candidate_params.stop_loss_pct,
                default_take_profit_pct=request.candidate_params.take_profit_pct,
                min_liquidity_usd=request.candidate_params.min_liquidity_usd,
                random_seed=request.seed,
                use_live_signal_generation=True,
            )

            try:
                candidate_result = await engine.run(
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    data_source=data_source,
                    config=candidate_config,
                )
                candidate_metrics_dict = candidate_result.get("metrics", {})
            except Exception as engine_error:
                logger.warning(
                    f"Candidate engine backtest failed, falling back to mock: {engine_error}"
                )
                candidate_trades = _generate_mock_trades(
                    params=request.candidate_params,
                    initial_capital=request.initial_capital_sol,
                    num_trades=50 if request.quick_mode else 200,
                    seed=request.seed,
                )
                candidate_metrics_obj = metrics_engine.calculate(
                    trades=candidate_trades,
                    initial_capital=request.initial_capital_sol,
                )
                candidate_metrics_dict = candidate_metrics_obj.to_dict()

            _backtest_store[backtest_id]["progress"] = 0.8

            # Generate comparisons
            comparisons = _generate_comparisons(baseline_metrics_dict, candidate_metrics_dict)

            # Create temporary objects for summary generation
            class MetricsProxy:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)

            summary = _generate_summary(
                MetricsProxy(baseline_metrics_dict), MetricsProxy(candidate_metrics_dict)
            )

        _backtest_store[backtest_id]["progress"] = 1.0

        # Update store with results
        _backtest_store[backtest_id].update(
            {
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "baseline_metrics": baseline_metrics_dict,
                "candidate_metrics": candidate_metrics_dict,
                "comparisons": [c.model_dump() for c in comparisons] if comparisons else None,
                "summary": summary,
            }
        )

        logger.info(f"Backtest {backtest_id} completed")

    except Exception as e:
        logger.error(f"Backtest {backtest_id} failed: {e}", exc_info=True)
        _backtest_store[backtest_id].update(
            {
                "status": "failed",
                "error": str(e),
                "progress": 0.0,
            }
        )


def _get_backtest_result(backtest_id: str) -> BacktestResult:
    """Convert stored data to BacktestResult."""
    data = _backtest_store[backtest_id]

    baseline_metrics = None
    if data.get("baseline_metrics"):
        baseline_metrics = BacktestMetrics(**data["baseline_metrics"])

    candidate_metrics = None
    if data.get("candidate_metrics"):
        candidate_metrics = BacktestMetrics(**data["candidate_metrics"])

    comparisons = None
    if data.get("comparisons"):
        comparisons = [BacktestComparison(**c) for c in data["comparisons"]]

    return BacktestResult(
        id=data["id"],
        status=data["status"],
        created_at=data["created_at"],
        completed_at=data.get("completed_at"),
        progress=data.get("progress"),
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        comparisons=comparisons,
        summary=data.get("summary"),
        error=data.get("error"),
    )


def _generate_mock_trades(
    params: BacktestParams,
    initial_capital: float,
    num_trades: int,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate mock trade data for backtesting.

    In production, this would use the actual backtest engine with
    historical data.
    """
    import random

    trades = []
    capital = initial_capital
    rng = random.Random(seed) if seed is not None else random

    # Adjust probabilities based on parameters
    # Tighter stops = higher win rate but lower avg win
    base_win_rate = 0.45
    stop_factor = (10 - params.stop_loss_pct) / 10  # Tighter stop = higher
    tp_factor = (params.take_profit_pct - 10) / 30  # Higher TP = lower wins

    win_rate = base_win_rate + (stop_factor * 0.1) - (tp_factor * 0.15)
    win_rate = max(0.3, min(0.7, win_rate))

    for _i in range(num_trades):
        is_win = rng.random() < win_rate

        if is_win:
            # Win: return between 5% and take_profit
            pnl_pct = rng.uniform(5, params.take_profit_pct)
        else:
            # Loss: return between 0 and -stop_loss
            pnl_pct = rng.uniform(-params.stop_loss_pct, -1)

        position_size = min(params.max_position_sol, capital * 0.1)
        pnl_sol = position_size * (pnl_pct / 100)
        capital += pnl_sol

        # Simulate slippage
        realized_slippage = rng.uniform(
            params.slippage_bps * 0.3,
            params.slippage_bps * 1.2,
        )

        # Simulate inclusion (higher slippage tolerance = higher inclusion)
        inclusion_prob = min(0.95, 0.7 + (params.slippage_bps / 2000))
        included = rng.random() < inclusion_prob

        trades.append(
            {
                "pnl_sol": pnl_sol if included else 0,
                "pnl_pct": pnl_pct if included else 0,
                "included": included,
                "realized_slippage_bps": realized_slippage if included else 0,
                "quoted_slippage_bps": params.slippage_bps,
                "position_size_sol": position_size,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    return trades


def _generate_comparisons(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[BacktestComparison]:
    """Generate metric comparisons."""
    comparisons = []

    key_metrics = [
        ("profit_quality_score", "Profit Quality", True, "{:.2f}"),
        ("expectancy_pct", "Expectancy", True, "{:.2f}%"),
        ("sharpe_ratio", "Sharpe", True, "{:.2f}"),
        ("max_drawdown_pct", "Max DD", False, "{:.1f}%"),
        ("total_return_pct", "Return", True, "{:.1f}%"),
        ("win_rate_pct", "Win Rate", True, "{:.1f}%"),
        ("profit_factor", "Profit Factor", True, "{:.2f}"),
        ("fill_rate_pct", "Fill Rate", True, "{:.1f}%"),
    ]

    for key, label, higher_is_better, fmt in key_metrics:
        b_val = baseline.get(key, 0)
        c_val = candidate.get(key, 0)
        delta = c_val - b_val

        # Format values
        b_str = fmt.format(b_val)
        c_str = fmt.format(c_val)

        # Format delta
        if delta > 0:
            d_str = f"+{fmt.format(delta)}"
        else:
            d_str = fmt.format(delta)

        # Determine if change is positive
        if higher_is_better:
            is_positive = delta > 0
        else:
            is_positive = delta < 0

        comparisons.append(
            BacktestComparison(
                metric=label,
                baseline=b_str,
                candidate=c_str,
                delta=d_str,
                is_positive=is_positive,
            )
        )

    return comparisons


def _generate_summary(baseline_metrics, candidate_metrics) -> str:
    """Generate natural language summary of comparison."""
    improvements = []
    degradations = []

    # Check key metrics
    if candidate_metrics.sharpe_ratio > baseline_metrics.sharpe_ratio:
        improvements.append("Sharpe ratio")
    elif candidate_metrics.sharpe_ratio < baseline_metrics.sharpe_ratio:
        degradations.append("Sharpe ratio")

    if candidate_metrics.max_drawdown_pct < baseline_metrics.max_drawdown_pct:
        improvements.append("max drawdown")
    elif candidate_metrics.max_drawdown_pct > baseline_metrics.max_drawdown_pct:
        degradations.append("max drawdown")

    if candidate_metrics.win_rate_pct > baseline_metrics.win_rate_pct:
        improvements.append("win rate")
    elif candidate_metrics.win_rate_pct < baseline_metrics.win_rate_pct:
        degradations.append("win rate")

    if candidate_metrics.total_return_pct > baseline_metrics.total_return_pct:
        improvements.append("total return")
    elif candidate_metrics.total_return_pct < baseline_metrics.total_return_pct:
        degradations.append("total return")

    if (
        getattr(candidate_metrics, "profit_quality_score", None) is not None
        and getattr(baseline_metrics, "profit_quality_score", None) is not None
    ):
        if candidate_metrics.profit_quality_score > baseline_metrics.profit_quality_score:
            improvements.append("profit quality")
        elif candidate_metrics.profit_quality_score < baseline_metrics.profit_quality_score:
            degradations.append("profit quality")

    # Build summary
    parts = []

    if improvements:
        parts.append(f"Candidate improves {', '.join(improvements)}")

    if degradations:
        parts.append(f"but reduces {', '.join(degradations)}")

    if not parts:
        return "No significant changes between baseline and candidate"

    return ". ".join(parts) + "."
