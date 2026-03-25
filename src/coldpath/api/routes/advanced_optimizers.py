"""
Advanced Optimizer API Routes - Endpoints for all optimization methods.

Provides REST API for:
- Unified optimizer interface
- Individual optimizer methods (Genetic, Pareto, RL, etc.)
- Ensemble optimization
- MEV/Slippage estimation
- Order book simulation
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/advanced", tags=["Advanced Optimizers"])


# ============================================================================
# Request/Response Models
# ============================================================================


class OptimizeRequest(BaseModel):
    """Request for optimization."""

    param_bounds: dict[str, list[float]] = Field(
        ...,
        description="Parameter bounds as {name: [min, max]}",
        example={"stop_loss_pct": [3.0, 15.0], "take_profit_pct": [8.0, 50.0]},
    )
    fixed_params: dict[str, Any] | None = Field(
        default=None,
        description="Fixed parameters",
    )
    method: str = Field(
        default="auto",
        description="Optimization method: auto, optuna, genetic, pareto, ensemble, rl, random",
    )
    max_iterations: int = Field(default=50, ge=10, le=500)
    max_time_minutes: float = Field(default=15.0, ge=1.0, le=60.0)
    quick_mode: bool = Field(default=True)


class OptimizeResponse(BaseModel):
    """Response from optimization."""

    best_params: dict[str, float]
    best_metrics: dict[str, float]
    best_score: float
    method: str
    actual_method: str | None
    iterations: int
    time_seconds: float
    convergence_achieved: bool
    warnings: list[str]
    recommendations: list[str]


class MEVEstimateRequest(BaseModel):
    """Request for MEV/slippage estimation."""

    trade_size_sol: float = Field(..., gt=0)
    pool_liquidity_usd: float = Field(..., gt=0)
    volatility_pct: float = Field(default=20.0, ge=0)
    is_memecoin: bool = Field(default=False)
    is_buy: bool = Field(default=True)


class MEVEstimateResponse(BaseModel):
    """Response from MEV estimation."""

    base_slippage_bps: float
    volatility_slippage_bps: float
    liquidity_slippage_bps: float
    frontrun_risk_bps: float
    sandwich_risk_bps: float
    mev_cost_bps: float
    priority_fee_bps: float
    total_slippage_bps: float
    total_cost_bps: float
    expected_fill_rate: float
    recommended_max_slippage_bps: float


class OrderBookSimRequest(BaseModel):
    """Request for order book simulation."""

    mid_price: float = Field(..., gt=0)
    liquidity_usd: float = Field(..., gt=0)
    trade_size_sol: float = Field(..., gt=0)
    side: str = Field(default="buy", pattern="^(buy|sell)$")
    volatility_pct: float = Field(default=20.0)
    is_memecoin: bool = Field(default=False)


class OrderBookSimResponse(BaseModel):
    """Response from order book simulation."""

    filled: bool
    fill_rate: float
    filled_size: float
    avg_price: float
    worst_price: float
    slippage_bps: float
    price_impact_pct: float
    levels_consumed: int


class GeneticOptimizeRequest(BaseModel):
    """Request for genetic algorithm optimization."""

    param_bounds: dict[str, list[float]]
    fixed_params: dict[str, Any] | None = None
    population_size: int = Field(default=50, ge=10, le=200)
    generations: int = Field(default=100, ge=10, le=500)
    crossover_rate: float = Field(default=0.8, ge=0, le=1)
    mutation_rate: float = Field(default=0.1, ge=0, le=1)
    quick_mode: bool = Field(default=True)


class ParetoOptimizeRequest(BaseModel):
    """Request for Pareto (NSGA-II) optimization."""

    param_bounds: dict[str, list[float]]
    fixed_params: dict[str, Any] | None = None
    objectives: list[str] | None = Field(
        default=None,
        description="Objectives to optimize: sharpe_ratio, max_drawdown_pct, win_rate_pct",
    )
    population_size: int = Field(default=100, ge=20, le=300)
    generations: int = Field(default=200, ge=20, le=500)
    quick_mode: bool = Field(default=True)


class ParetoOptimizeResponse(BaseModel):
    """Response from Pareto optimization."""

    pareto_front_size: int
    best_balanced_params: dict[str, float]
    best_balanced_metrics: dict[str, float]
    best_by_objective: dict[str, dict[str, Any]]
    generations: int
    time_seconds: float


class EnsembleOptimizeRequest(BaseModel):
    """Request for ensemble optimization."""

    param_bounds: dict[str, list[float]]
    fixed_params: dict[str, Any] | None = None
    methods: list[str] = Field(
        default=["optuna", "genetic"],
        description="Methods to use: optuna, genetic, pareto, random",
    )
    combination_method: str = Field(
        default="weighted",
        description="How to combine: weighted, median, best, voting",
    )
    quick_mode: bool = Field(default=True)


class EnsembleOptimizeResponse(BaseModel):
    """Response from ensemble optimization."""

    ensemble_params: dict[str, float]
    ensemble_score: float
    individual_results: list[dict[str, Any]]
    best_individual: str
    improvement_over_best: float


# ============================================================================
# Backtest Runner Helper
# ============================================================================


async def _create_backtest_runner(quick_mode: bool = True):
    """Create a backtest runner function."""
    from coldpath.backtest.engine import BacktestConfig, BacktestEngine

    engine = BacktestEngine()

    async def run_backtest(params: dict[str, Any]) -> dict[str, float]:
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
            }

    return run_backtest


# ============================================================================
# Unified Optimizer Endpoints
# ============================================================================


@router.post("/optimize", response_model=OptimizeResponse)
async def unified_optimize(
    request: OptimizeRequest,
    background_tasks: BackgroundTasks,
):
    """Run optimization with unified interface.

    Automatically selects best method or uses specified method.
    """
    from coldpath.backtest.unified_optimizer import (
        OptimizerType,
        UnifiedOptimizer,
        UnifiedOptimizerConfig,
    )

    # Convert param_bounds
    param_bounds = {k: tuple(v) for k, v in request.param_bounds.items()}

    # Get method
    method_map = {
        "auto": OptimizerType.AUTO,
        "optuna": OptimizerType.OPTUNA,
        "genetic": OptimizerType.GENETIC,
        "pareto": OptimizerType.PARETO,
        "ensemble": OptimizerType.ENSEMBLE,
        "rl": OptimizerType.RL,
        "random": OptimizerType.RANDOM,
    }
    method = method_map.get(request.method.lower(), OptimizerType.AUTO)

    config = UnifiedOptimizerConfig(
        max_iterations=request.max_iterations,
        max_time_minutes=request.max_time_minutes,
    )

    runner = await _create_backtest_runner(request.quick_mode)
    optimizer = UnifiedOptimizer(backtest_runner=runner, config=config)

    result = await optimizer.optimize(
        param_bounds=param_bounds,
        fixed_params=request.fixed_params,
        method=method,
    )

    return OptimizeResponse(
        best_params=result.best_params,
        best_metrics=result.best_metrics,
        best_score=result.best_score,
        method=result.method.value,
        actual_method=result.actual_method.value if result.actual_method else None,
        iterations=result.iterations,
        time_seconds=result.time_seconds,
        convergence_achieved=result.convergence_achieved,
        warnings=result.warnings,
        recommendations=result.recommendations,
    )


# ============================================================================
# Genetic Algorithm Endpoints
# ============================================================================


@router.post("/genetic", response_model=OptimizeResponse)
async def genetic_optimize(request: GeneticOptimizeRequest):
    """Run genetic algorithm optimization.

    Uses evolutionary approach with crossover and mutation.
    Good for exploring large parameter spaces.
    """
    from coldpath.backtest.genetic_optimizer import (
        GAConfig,
        GeneticOptimizer,
    )

    param_bounds = {k: tuple(v) for k, v in request.param_bounds.items()}

    config = GAConfig(
        population_size=request.population_size,
        max_generations=request.generations,
        crossover_rate=request.crossover_rate,
        mutation_rate=request.mutation_rate,
    )

    runner = await _create_backtest_runner(request.quick_mode)
    optimizer = GeneticOptimizer(backtest_runner=runner, config=config)

    result = await optimizer.optimize(
        param_bounds=param_bounds,
        fixed_params=request.fixed_params,
    )

    return OptimizeResponse(
        best_params=result.best_individual.genes,
        best_metrics=result.best_individual.metrics,
        best_score=result.best_individual.fitness,
        method="genetic",
        actual_method="genetic",
        iterations=result.generation,
        time_seconds=result.time_seconds,
        convergence_achieved=len(result.convergence_history) > 10,
        warnings=[],
        recommendations=[
            f"Population: {request.population_size}, Generations: {request.generations}"
        ],
    )


# ============================================================================
# Pareto (NSGA-II) Endpoints
# ============================================================================


@router.post("/pareto", response_model=ParetoOptimizeResponse)
async def pareto_optimize(request: ParetoOptimizeRequest):
    """Run multi-objective Pareto optimization (NSGA-II).

    Finds trade-off solutions for conflicting objectives.
    Returns a Pareto front of non-dominated solutions.
    """
    from coldpath.backtest.pareto_optimizer import (
        NSGA2Config,
        Objective,
        ParetoOptimizer,
    )

    param_bounds = {k: tuple(v) for k, v in request.param_bounds.items()}

    # Set up objectives
    if request.objectives:
        objectives = [
            Objective(obj, maximize=(obj != "max_drawdown_pct")) for obj in request.objectives
        ]
    else:
        objectives = [
            Objective("sharpe_ratio", maximize=True, weight=0.4),
            Objective("max_drawdown_pct", maximize=False, weight=0.3),
            Objective("win_rate_pct", maximize=True, weight=0.3),
        ]

    config = NSGA2Config(
        population_size=request.population_size,
        max_generations=request.generations,
    )

    runner = await _create_backtest_runner(request.quick_mode)
    optimizer = ParetoOptimizer(
        backtest_runner=runner,
        objectives=objectives,
        config=config,
    )

    result = await optimizer.optimize(
        param_bounds=param_bounds,
        fixed_params=request.fixed_params,
    )

    best_balanced = result.get_best_balanced()

    return ParetoOptimizeResponse(
        pareto_front_size=len(result.pareto_front),
        best_balanced_params=best_balanced.params,
        best_balanced_metrics=best_balanced.metrics,
        best_by_objective={
            k: {"params": v.params, "objectives": v.objectives}
            for k, v in result.best_by_objective.items()
        },
        generations=result.generation,
        time_seconds=result.time_seconds,
    )


# ============================================================================
# Ensemble Endpoints
# ============================================================================


@router.post("/ensemble", response_model=EnsembleOptimizeResponse)
async def ensemble_optimize(request: EnsembleOptimizeRequest):
    """Run ensemble optimization combining multiple methods.

    Runs multiple optimizers and combines their results
    for more robust parameter selection.
    """
    from coldpath.backtest.ensemble_optimizer import (
        EnsembleConfig,
        EnsembleMethod,
        EnsembleOptimizer,
        OptimizationMethod,
    )

    param_bounds = {k: tuple(v) for k, v in request.param_bounds.items()}

    # Map methods
    method_map = {
        "optuna": OptimizationMethod.OPTUNA,
        "genetic": OptimizationMethod.GENETIC,
        "pareto": OptimizationMethod.PARETO,
        "random": OptimizationMethod.RANDOM,
    }

    methods = [method_map.get(m, OptimizationMethod.OPTUNA) for m in request.methods]

    # Map combination method
    combo_map = {
        "weighted": EnsembleMethod.WEIGHTED_AVERAGE,
        "median": EnsembleMethod.MEDIAN,
        "best": EnsembleMethod.BEST_PERFORMER,
        "voting": EnsembleMethod.VOTING,
    }

    config = EnsembleConfig(
        methods=methods,
        ensemble_method=combo_map.get(request.combination_method, EnsembleMethod.WEIGHTED_AVERAGE),
    )

    runner = await _create_backtest_runner(request.quick_mode)
    optimizer = EnsembleOptimizer(backtest_runner=runner, config=config)

    result = await optimizer.optimize(
        param_bounds=param_bounds,
        fixed_params=request.fixed_params,
    )

    return EnsembleOptimizeResponse(
        ensemble_params=result.ensemble_params,
        ensemble_score=result.ensemble_score,
        individual_results=[r.to_dict() for r in result.individual_results],
        best_individual=result.best_individual.value,
        improvement_over_best=result.improvement_over_best,
    )


# ============================================================================
# MEV/Slippage Endpoints
# ============================================================================


@router.post("/mev-estimate", response_model=MEVEstimateResponse)
async def estimate_mev(request: MEVEstimateRequest):
    """Estimate MEV exposure and slippage for a trade.

    Calculates realistic trading costs including:
    - AMM slippage
    - Volatility impact
    - Front-running risk
    - Sandwich attack exposure
    - Priority fee recommendations
    """
    from coldpath.backtest.mev_slippage import MEVSlippageModeler

    modeler = MEVSlippageModeler()

    result = modeler.estimate_execution_cost(
        trade_size_sol=request.trade_size_sol,
        pool_liquidity_usd=request.pool_liquidity_usd,
        volatility_pct=request.volatility_pct,
        is_buy=request.is_buy,
        is_memecoin=request.is_memecoin,
    )

    return MEVEstimateResponse(
        base_slippage_bps=result.base_slippage_bps,
        volatility_slippage_bps=result.volatility_slippage_bps,
        liquidity_slippage_bps=result.liquidity_slippage_bps,
        frontrun_risk_bps=result.frontrun_risk_bps,
        sandwich_risk_bps=result.sandwich_risk_bps,
        mev_cost_bps=result.mev_cost_bps,
        priority_fee_bps=result.priority_fee_bps,
        total_slippage_bps=result.total_slippage_bps,
        total_cost_bps=result.total_cost_bps,
        expected_fill_rate=result.expected_fill_rate,
        recommended_max_slippage_bps=result.recommended_max_slippage_bps,
    )


# ============================================================================
# Order Book Simulation Endpoints
# ============================================================================


@router.post("/orderbook-sim", response_model=OrderBookSimResponse)
async def simulate_orderbook(request: OrderBookSimRequest):
    """Simulate order book execution.

    Models realistic fill dynamics including:
    - Price level depth
    - Partial fills
    - Slippage
    - Price impact
    """
    from coldpath.backtest.order_book_sim import OrderBookSimulator

    simulator = OrderBookSimulator()

    # Create order book
    book = simulator.create_book(
        mid_price=request.mid_price,
        liquidity_usd=request.liquidity_usd,
        volatility_pct=request.volatility_pct,
        is_memecoin=request.is_memecoin,
    )

    # Simulate market order
    result = simulator.simulate_market_order(
        book=book,
        size_sol=request.trade_size_sol,
        side=request.side,
    )

    return OrderBookSimResponse(
        filled=result.filled,
        fill_rate=result.fill_rate,
        filled_size=result.filled_size,
        avg_price=result.avg_price,
        worst_price=result.worst_price,
        slippage_bps=result.slippage_bps,
        price_impact_pct=result.price_impact_pct,
        levels_consumed=result.levels_consumed,
    )


# ============================================================================
# Info Endpoints
# ============================================================================


@router.get("/methods")
async def list_methods():
    """List available optimization methods."""
    return {
        "methods": [
            {
                "name": "auto",
                "description": "Automatically select best method based on problem",
                "recommended_for": "General use",
            },
            {
                "name": "optuna",
                "description": "Bayesian optimization with TPE sampler",
                "recommended_for": "Small parameter spaces (1-5 params)",
            },
            {
                "name": "genetic",
                "description": "Evolutionary algorithm with crossover/mutation",
                "recommended_for": "Large parameter spaces (5+ params)",
            },
            {
                "name": "pareto",
                "description": "Multi-objective NSGA-II for trade-off solutions",
                "recommended_for": "Multiple conflicting objectives",
            },
            {
                "name": "ensemble",
                "description": "Combines multiple methods for robustness",
                "recommended_for": "Critical optimizations",
            },
            {
                "name": "rl",
                "description": "Reinforcement learning for adaptive tuning",
                "recommended_for": "Learning from experience",
            },
            {
                "name": "random",
                "description": "Random search baseline",
                "recommended_for": "Quick exploration",
            },
        ]
    }


@router.get("/health")
async def health_check():
    """Check advanced optimizer health."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "available_methods": [
            "unified",
            "genetic",
            "pareto",
            "ensemble",
            "mev",
            "orderbook",
        ],
    }
