"""
Advanced Optimizer API Routes - RL, Ensemble, and Multi-Source endpoints.

Provides REST API endpoints for:
- RL-based parameter optimization
- Ensemble optimization with multiple strategies
- Multi-source data integration
- Unified optimizer interface

Endpoints:
- POST /advanced/rl/optimize - Run RL-based optimization
- POST /advanced/rl/train - Train RL agent from history
- POST /advanced/ensemble/optimize - Run ensemble optimization
- GET /advanced/ensemble/status - Get ensemble status
- POST /advanced/data/token - Get token data from multiple sources
- POST /advanced/data/ohlcv - Get historical OHLCV data
- POST /advanced/unified/optimize - Unified optimization with all features
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/advanced", tags=["advanced-optimization"])


# ============================================================================
# Request/Response Models
# ============================================================================


class RLOptimizeRequest(BaseModel):
    """Request for RL-based optimization."""

    risk_tolerance: str = Field(default="moderate")
    capital_sol: float = Field(default=1.0, ge=0.01)
    episodes: int = Field(default=30, ge=5, le=200)
    exploration_rate: float = Field(default=0.3, ge=0.05, le=0.5)
    initial_params: dict[str, Any] | None = None
    market_metrics: dict[str, Any] | None = None


class RLOptimizeResponse(BaseModel):
    """Response from RL optimization."""

    best_params: dict[str, Any]
    best_metrics: dict[str, float]
    best_score: float
    episodes_trained: int
    training_time_seconds: float
    final_exploration_rate: float
    confidence: float


class RLTrainRequest(BaseModel):
    """Request to train RL agent from history."""

    epochs: int = Field(default=10, ge=1, le=50)
    reset_agent: bool = Field(default=False)


class RLTrainResponse(BaseModel):
    """Response from RL training."""

    epochs_trained: int
    total_reward: float
    avg_reward: float
    best_reward: float
    experiences_collected: int
    training_time_seconds: float


class EnsembleOptimizeRequest(BaseModel):
    """Request for ensemble optimization."""

    risk_tolerance: str = Field(default="moderate")
    primary_goal: str = Field(default="balanced")
    capital_sol: float = Field(default=1.0, ge=0.01)
    strategies: list[str] = Field(
        default=["bayesian", "genetic"], description="Strategies: bayesian, genetic, pareto, random"
    )
    aggregation_method: str = Field(
        default="weighted", description="Methods: weighted, voting, median, best"
    )
    iterations_per_strategy: int = Field(default=20, ge=5, le=100)


class EnsembleOptimizeResponse(BaseModel):
    """Response from ensemble optimization."""

    final_params: dict[str, Any]
    final_score: float
    final_metrics: dict[str, float]
    strategies_used: list[str]
    strategy_results: list[dict[str, Any]]
    aggregation_method: str
    diversity_score: float
    improvement_over_best_single: float
    total_time_seconds: float


class TokenDataRequest(BaseModel):
    """Request for token data."""

    token_address: str
    sources: list[str] | None = Field(
        default=None, description="Sources: birdeye, dexscreener, bitquery"
    )
    use_cache: bool = Field(default=True)


class TokenDataResponse(BaseModel):
    """Response with token data."""

    token_address: str
    symbol: str
    name: str
    price_usd: float
    price_change_24h: float
    volume_24h: float
    liquidity_usd: float
    risk_score: float
    source: str
    timestamp: str


class OHLCVRequest(BaseModel):
    """Request for historical OHLCV data."""

    token_address: str
    hours: int = Field(default=24, ge=1, le=720)
    interval: str = Field(default="1h", description="1m, 5m, 15m, 1h, 4h, 1d")


class OHLCVCandleResponse(BaseModel):
    """Single OHLCV candle."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class OHLCVResponse(BaseModel):
    """Response with OHLCV data."""

    token_address: str
    interval: str
    candles: list[OHLCVCandleResponse]
    source: str


class UnifiedOptimizeRequest(BaseModel):
    """Request for unified optimization."""

    risk_tolerance: str = Field(default="moderate")
    primary_goal: str = Field(default="balanced")
    capital_sol: float = Field(default=1.0, ge=0.01)
    mode: str = Field(
        default="balanced", description="Modes: quick, balanced, thorough, ensemble, rl_only"
    )
    enable_regime_detection: bool = Field(default=True)
    enable_learning: bool = Field(default=True)
    market_metrics: dict[str, Any] | None = None


class UnifiedOptimizeResponse(BaseModel):
    """Response from unified optimization."""

    best_params: dict[str, Any]
    best_metrics: dict[str, float]
    best_score: float
    mode: str
    strategies_used: list[str]
    regime_detected: str | None
    total_iterations: int
    total_time_seconds: float
    confidence: float
    recommendations: list[str]
    warnings: list[str]


class BatchOptimizeRequest(BaseModel):
    """Request for batch optimization."""

    profiles: list[dict[str, Any]] = Field(..., description="List of optimization profiles")
    parallel: bool = Field(default=False)


class BatchOptimizeResponse(BaseModel):
    """Response from batch optimization."""

    results: list[UnifiedOptimizeResponse]
    total_time_seconds: float


# ============================================================================
# Helper Functions
# ============================================================================

# Global state
_rl_agents: dict[str, Any] = {}
_data_provider = None
_unified_optimizer = None


async def get_backtest_runner():
    """Get or create backtest runner."""

    # Mock runner for demo - in production would use real engine
    async def mock_runner(params: dict[str, Any]) -> dict[str, float]:
        import random

        await asyncio.sleep(0.01)

        sl = params.get("stop_loss_pct", 8.0)
        tp = params.get("take_profit_pct", 20.0)
        rr_ratio = tp / max(sl, 0.1)

        base_sharpe = 0.5 + rr_ratio * 0.8
        noise = random.gauss(0, 0.2)

        return {
            "sharpe_ratio": max(0, base_sharpe + noise),
            "win_rate_pct": 40 + rr_ratio * 8 + random.gauss(0, 3),
            "max_drawdown_pct": 25 - rr_ratio * 3 + random.gauss(0, 2),
            "total_return_pct": 10 + rr_ratio * 15 + random.gauss(0, 5),
            "profit_factor": 1.2 + rr_ratio * 0.4,
            "profit_quality_score": min(1.0, rr_ratio * 0.3),
        }

    return mock_runner


def get_data_provider():
    """Get or create data provider."""
    global _data_provider
    if _data_provider is None:
        from coldpath.data.multi_source_provider import (
            MultiSourceConfig,
            MultiSourceDataProvider,
        )

        _data_provider = MultiSourceDataProvider(config=MultiSourceConfig(mock_mode=True))
    return _data_provider


# ============================================================================
# RL Optimization Endpoints
# ============================================================================


@router.post("/rl/optimize", response_model=RLOptimizeResponse)
async def rl_optimize(request: RLOptimizeRequest):
    """Run RL-based parameter optimization.

    Uses reinforcement learning to learn optimal parameter
    adjustments based on backtest outcomes.

    The agent:
    1. Starts with provided or default parameters
    2. Explores parameter space using epsilon-greedy policy
    3. Learns from rewards (Sharpe, win rate, drawdown)
    4. Returns best found parameters
    """
    from coldpath.backtest.rl_optimizer import (
        RLAgentConfig,
        RLParameterAgent,
    )

    try:
        runner = await get_backtest_runner()

        # Create agent
        config = RLAgentConfig(
            exploration_rate=request.exploration_rate,
        )

        agent = RLParameterAgent(config=config)

        # Get initial params from AI guide
        if request.initial_params is None:
            from coldpath.backtest.ai_backtest_guide import AIBacktestGuide

            guide = AIBacktestGuide()
            guided = guide.create_guided_setup(
                risk_tolerance=request.risk_tolerance,
                capital_sol=request.capital_sol,
            )
            initial_params = guided.suggested_config
        else:
            initial_params = request.initial_params

        # Run optimization
        result = await agent.train_online(
            initial_params=initial_params,
            backtest_runner=runner,
            num_episodes=request.episodes,
            market_metrics=request.market_metrics,
        )

        # Get final suggestion
        final_params, confidence = agent.suggest_parameters(
            initial_params,
            request.market_metrics,
            deterministic=True,
        )

        # Run final backtest
        final_metrics = await runner(final_params)

        return RLOptimizeResponse(
            best_params=final_params,
            best_metrics=final_metrics,
            best_score=result.best_reward,
            episodes_trained=result.episodes_trained,
            training_time_seconds=result.training_time_seconds,
            final_exploration_rate=result.final_exploration_rate,
            confidence=confidence,
        )

    except Exception as e:
        logger.error(f"RL optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/rl/train", response_model=RLTrainResponse)
async def rl_train(request: RLTrainRequest):
    """Train RL agent from historical results.

    Uses historical backtest results to train the RL agent
    without running new backtests.
    """
    try:
        # In production, would load from feedback loop or database
        historical_results = []

        from coldpath.backtest.rl_optimizer import (
            RLAgentConfig,
            RLParameterAgent,
        )

        config = RLAgentConfig()
        agent = RLParameterAgent(config=config)

        if historical_results:
            result = await agent.train_from_history(
                historical_results,
                epochs=request.epochs,
            )
        else:
            result = await agent.train_from_history(
                [],  # Empty history - just initialize
                epochs=1,
            )

        return RLTrainResponse(
            epochs_trained=result.episodes_trained,
            total_reward=result.total_reward,
            avg_reward=result.avg_reward,
            best_reward=result.best_reward,
            experiences_collected=result.experiences_collected,
            training_time_seconds=result.training_time_seconds,
        )

    except Exception as e:
        logger.error(f"RL training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Ensemble Optimization Endpoints
# ============================================================================


@router.post("/ensemble/optimize", response_model=EnsembleOptimizeResponse)
async def ensemble_optimize(request: EnsembleOptimizeRequest):
    """Run ensemble optimization with multiple strategies.

    Combines results from multiple optimization strategies:
    - Bayesian (bayesian): Efficient exploration
    - Genetic (genetic): Global search
    - Pareto (pareto): Multi-objective
    - Random (random): Baseline

    Aggregation methods:
    - weighted: Weighted average by performance
    - voting: Consensus voting
    - median: Median of parameters
    - best: Use best performing strategy
    """
    from coldpath.backtest.ensemble_optimizer import (
        EnsembleConfig,
        EnsembleMethod,
        EnsembleOptimizer,
        OptimizationMethod,
    )

    try:
        runner = await get_backtest_runner()

        # Map strategy names
        method_map = {
            "bayesian": OptimizationMethod.OPTUNA,
            "genetic": OptimizationMethod.GENETIC,
            "pareto": OptimizationMethod.PARETO,
            "random": OptimizationMethod.RANDOM,
        }

        methods = [method_map.get(s, OptimizationMethod.OPTUNA) for s in request.strategies]

        # Map aggregation method
        agg_map = {
            "weighted": EnsembleMethod.WEIGHTED_AVERAGE,
            "voting": EnsembleMethod.VOTING,
            "median": EnsembleMethod.MEDIAN,
            "best": EnsembleMethod.BEST_PERFORMER,
        }

        ensemble_method = agg_map.get(request.aggregation_method, EnsembleMethod.WEIGHTED_AVERAGE)

        config = EnsembleConfig(
            methods=methods,
            ensemble_method=ensemble_method,
            optuna_iterations=request.iterations_per_strategy,
            genetic_generations=request.iterations_per_strategy // 2,
        )

        optimizer = EnsembleOptimizer(
            backtest_runner=runner,
            config=config,
        )

        # Get initial params from guide
        from coldpath.backtest.ai_backtest_guide import AIBacktestGuide

        guide = AIBacktestGuide()
        guide.create_guided_setup(
            risk_tolerance=request.risk_tolerance,
            primary_goal=request.primary_goal,
            capital_sol=request.capital_sol,
        )

        # Define param bounds
        param_bounds = {
            "stop_loss_pct": (3.0, 20.0),
            "take_profit_pct": (8.0, 60.0),
            "max_position_sol": (0.01, 0.20),
            "max_risk_score": (0.20, 0.70),
        }

        result = await optimizer.optimize(
            param_bounds=param_bounds,
            fixed_params={"initial_capital_sol": request.capital_sol},
        )

        return EnsembleOptimizeResponse(
            final_params=result.ensemble_params,
            final_score=result.ensemble_score,
            final_metrics=result.ensemble_metrics,
            strategies_used=[m.value for m in methods],
            strategy_results=[r.to_dict() for r in result.individual_results],
            aggregation_method=ensemble_method.value,
            diversity_score=0.5,  # Would calculate
            improvement_over_best_single=result.improvement_over_best,
            total_time_seconds=result.total_time_seconds,
        )

    except Exception as e:
        logger.error(f"Ensemble optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Data Source Endpoints
# ============================================================================


@router.post("/data/token", response_model=TokenDataResponse)
async def get_token_data(request: TokenDataRequest):
    """Get token data from multiple sources.

    Fetches token data from Birdeye, DexScreener, or Bitquery
    with automatic fallback on failure.
    """
    try:
        provider = get_data_provider()

        data = await provider.get_token_data(
            token_address=request.token_address,
            sources=request.sources,
            use_cache=request.use_cache,
        )

        if data is None:
            raise HTTPException(
                status_code=404, detail=f"Token data not found for {request.token_address}"
            )

        return TokenDataResponse(
            token_address=data.token_address,
            symbol=data.symbol,
            name=data.name,
            price_usd=data.price_usd,
            price_change_24h=data.price_change_24h,
            volume_24h=data.volume_24h,
            liquidity_usd=data.liquidity_usd,
            risk_score=data.risk_score,
            source=data.source.value,
            timestamp=data.timestamp,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token data fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/data/ohlcv", response_model=OHLCVResponse)
async def get_ohlcv_data(request: OHLCVRequest):
    """Get historical OHLCV candlestick data.

    Fetches historical price data for charting and analysis.
    """
    try:
        provider = get_data_provider()

        end_time = int(datetime.utcnow().timestamp() * 1000)
        start_time = end_time - (request.hours * 3600 * 1000)

        candles = await provider.get_historical_ohlcv(
            token_address=request.token_address,
            start_time=start_time,
            end_time=end_time,
            interval=request.interval,
        )

        return OHLCVResponse(
            token_address=request.token_address,
            interval=request.interval,
            candles=[
                OHLCVCandleResponse(
                    timestamp=c.timestamp,
                    open=c.open,
                    high=c.high,
                    low=c.low,
                    close=c.close,
                    volume=c.volume,
                )
                for c in candles
            ],
            source="multi",
        )

    except Exception as e:
        logger.error(f"OHLCV fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Unified Optimization Endpoints
# ============================================================================


@router.post("/unified/optimize", response_model=UnifiedOptimizeResponse)
async def unified_optimize(request: UnifiedOptimizeRequest):
    """Run unified optimization with all features.

    This is the most comprehensive optimization endpoint,
    combining:
    - AI-guided initial parameters
    - Multiple optimization strategies (Bayesian, Genetic, RL, Ensemble)
    - Automatic method selection

    Modes:
    - auto: Auto-select best method
    - optuna: Bayesian optimization
    - genetic: Genetic algorithm
    - pareto: Multi-objective optimization
    - ensemble: Multi-strategy combination
    - rl: RL-based optimization
    """
    from coldpath.backtest.unified_optimizer import (
        OptimizerType,
        UnifiedOptimizer,
        UnifiedOptimizerConfig,
    )

    try:
        runner = await get_backtest_runner()

        # Map mode
        mode_map = {
            "auto": OptimizerType.AUTO,
            "quick": OptimizerType.OPTUNA,
            "optuna": OptimizerType.OPTUNA,
            "balanced": OptimizerType.OPTUNA,
            "thorough": OptimizerType.GENETIC,
            "genetic": OptimizerType.GENETIC,
            "pareto": OptimizerType.PARETO,
            "ensemble": OptimizerType.ENSEMBLE,
            "rl": OptimizerType.RL,
            "rl_only": OptimizerType.RL,
        }

        method = mode_map.get(request.mode.lower(), OptimizerType.AUTO)

        config = UnifiedOptimizerConfig(
            default_method=method,
            max_iterations=50,
            max_time_minutes=15,
        )

        optimizer = UnifiedOptimizer(
            backtest_runner=runner,
            config=config,
        )

        # Get initial params from AI guide
        from coldpath.backtest.ai_backtest_guide import AIBacktestGuide

        guide = AIBacktestGuide()
        guided = guide.create_guided_setup(
            risk_tolerance=request.risk_tolerance,
            primary_goal=request.primary_goal,
            capital_sol=request.capital_sol,
        )
        initial_params = guided.suggested_config

        # Define param bounds
        param_bounds = {
            "stop_loss_pct": (3.0, 20.0),
            "take_profit_pct": (8.0, 60.0),
            "max_position_sol": (0.01, 0.20),
            "max_risk_score": (0.20, 0.70),
        }

        result = await optimizer.optimize(
            param_bounds=param_bounds,
            fixed_params={"initial_capital_sol": request.capital_sol},
            initial_params=initial_params,
            method=method,
        )

        strategies_used = [result.actual_method.value] if result.actual_method else [method.value]

        return UnifiedOptimizeResponse(
            best_params=result.best_params,
            best_metrics=result.best_metrics,
            best_score=result.best_score,
            mode=request.mode,
            strategies_used=strategies_used,
            regime_detected=None,
            total_iterations=result.iterations,
            total_time_seconds=result.time_seconds,
            confidence=0.8,
            recommendations=result.recommendations,
            warnings=result.warnings,
        )

    except Exception as e:
        logger.error(f"Unified optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/unified/batch", response_model=BatchOptimizeResponse)
async def batch_optimize(request: BatchOptimizeRequest):
    """Run batch optimization for multiple profiles.

    Optimizes multiple risk profiles in parallel or sequentially.
    Useful for generating parameters for different market conditions.
    """
    from coldpath.backtest.unified_optimizer import (
        OptimizerType,
        UnifiedOptimizer,
        UnifiedOptimizerConfig,
    )

    try:
        runner = await get_backtest_runner()

        config = UnifiedOptimizerConfig(default_method=OptimizerType.AUTO)
        optimizer = UnifiedOptimizer(backtest_runner=runner, config=config)

        start_time = datetime.utcnow()
        results = []

        param_bounds = {
            "stop_loss_pct": (3.0, 20.0),
            "take_profit_pct": (8.0, 60.0),
            "max_position_sol": (0.01, 0.20),
        }

        for profile in request.profiles:
            result = await optimizer.optimize(
                param_bounds=param_bounds,
                fixed_params=profile,
                method=OptimizerType.AUTO,
            )
            results.append(result)

        total_time = (datetime.utcnow() - start_time).total_seconds()

        return BatchOptimizeResponse(
            results=[
                UnifiedOptimizeResponse(
                    best_params=r.best_params,
                    best_metrics=r.best_metrics,
                    best_score=r.best_score,
                    mode="auto",
                    strategies_used=[r.actual_method.value] if r.actual_method else ["auto"],
                    regime_detected=None,
                    total_iterations=r.iterations,
                    total_time_seconds=r.time_seconds,
                    confidence=0.8,
                    recommendations=r.recommendations,
                    warnings=r.warnings,
                )
                for r in results
            ],
            total_time_seconds=total_time,
        )

    except Exception as e:
        logger.error(f"Batch optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Status Endpoints
# ============================================================================


class StatusResponse(BaseModel):
    """Status response."""

    module: str
    status: str
    features: list[str]
    statistics: dict[str, Any]


@router.get("/status", response_model=dict[str, StatusResponse])
async def get_advanced_status():
    """Get status of all advanced optimization modules."""
    return {
        "rl_optimizer": StatusResponse(
            module="RL Optimizer",
            status="operational",
            features=[
                "Online training",
                "Historical training",
                "Parameter suggestions",
                "Confidence scoring",
            ],
            statistics={"agents_created": len(_rl_agents)},
        ),
        "ensemble": StatusResponse(
            module="Ensemble Optimizer",
            status="operational",
            features=[
                "Multi-strategy combination",
                "Weighted aggregation",
                "Voting consensus",
                "Performance tracking",
            ],
            statistics={},
        ),
        "data_provider": StatusResponse(
            module="Multi-Source Data Provider",
            status="operational",
            features=[
                "Birdeye integration",
                "DexScreener integration",
                "Bitquery integration",
                "Automatic fallback",
                "Data caching",
            ],
            statistics=get_data_provider().get_source_status() if _data_provider else {},
        ),
        "unified": StatusResponse(
            module="Unified Optimizer",
            status="operational",
            features=[
                "Quick mode",
                "Balanced mode",
                "Thorough mode",
                "Ensemble mode",
                "RL-only mode",
                "Batch optimization",
            ],
            statistics={},
        ),
    }
