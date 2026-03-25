"""
AI Guided Backtest Routes - Intelligent setup and auto-optimization endpoints.

These endpoints provide:
- Guided setup wizard for optimal initial parameters
- Auto mode for hands-free parameter optimization
- Parameter analysis and improvement suggestions
- Session management for long-running optimizations

Endpoints:
- POST /guided/setup - Get AI-guided parameter recommendations
- POST /guided/quick-setup - Quick preset-based setup
- POST /guided/analyze - Analyze current parameters
- GET /guided/param-info/{param} - Get parameter explanation
- POST /auto/start - Start auto optimization session
- GET /auto/status/{session_id} - Get session status
- POST /auto/cancel/{session_id} - Cancel running session
- GET /auto/sessions - List all sessions
"""

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/guided", tags=["guided-backtest"])


# ============================================================================
# Request/Response Models
# ============================================================================


class GuidedSetupRequest(BaseModel):
    """Request for AI-guided setup."""

    risk_tolerance: str = Field(
        default="moderate", description="Risk tolerance: conservative, moderate, aggressive, degen"
    )
    primary_goal: str = Field(
        default="balanced",
        description=(
            "Optimization goal: max_sharpe, max_return, min_drawdown, max_win_rate, "
            "profit_quality, balanced"
        ),
    )
    capital_sol: float = Field(default=1.0, ge=0.01, description="Available capital in SOL")
    memecoin_focus: bool = Field(default=True, description="Whether focusing on memecoins")
    max_daily_trades: int = Field(default=20, ge=1, le=100, description="Maximum trades per day")
    preferred_hold_time_minutes: int = Field(
        default=30, ge=1, le=1440, description="Target hold time in minutes"
    )
    market_regime: str | None = Field(
        default=None,
        description=(
            "Current market regime: trending_up, trending_down, sideways, "
            "high_volatility, low_volatility, memecoin_season, risk_off"
        ),
    )


class ParameterRecommendationResponse(BaseModel):
    """A single parameter recommendation."""

    key: str
    label: str
    value: float
    min_value: float
    max_value: float
    unit: str
    rationale: str
    confidence: float
    can_auto_tune: bool


class GuidedSetupResponse(BaseModel):
    """Response from AI-guided setup."""

    recommendations: list[ParameterRecommendationResponse]
    suggested_config: dict[str, Any]
    expected_metrics: dict[str, float]
    validation_warnings: list[str]
    next_steps: list[str]


class QuickSetupRequest(BaseModel):
    """Request for quick preset setup."""

    preset: str = Field(
        default="balanced", description="Preset name: conservative, balanced, aggressive, degen"
    )
    capital_sol: float = Field(default=1.0, ge=0.01, description="Available capital in SOL")


class QuickSetupResponse(BaseModel):
    """Response from quick setup."""

    preset: str
    config: dict[str, Any]
    description: str


class AnalyzeParamsRequest(BaseModel):
    """Request to analyze current parameters."""

    current_config: dict[str, Any] = Field(
        ..., description="Current backtest configuration to analyze"
    )


class AnalyzeParamsResponse(BaseModel):
    """Response from parameter analysis."""

    issues: list[dict[str, Any]]
    suggestions: list[dict[str, Any]]
    optimization_potential: str


class ParamInfoResponse(BaseModel):
    """Response with parameter explanation."""

    name: str
    description: str
    impact: str
    typical_range: str
    pro_tip: str


class AutoModeStartRequest(BaseModel):
    """Request to start auto optimization."""

    initial_params: dict[str, Any] | None = Field(
        default=None, description="Starting parameters (optional)"
    )
    max_iterations: int = Field(
        default=50, ge=10, le=500, description="Maximum optimization iterations"
    )
    max_time_minutes: int = Field(default=15, ge=5, le=120, description="Maximum time in minutes")
    target_sharpe: float = Field(default=1.5, ge=0.5, description="Target Sharpe ratio")
    target_win_rate: float = Field(
        default=0.45, ge=0.3, le=0.8, description="Target win rate (0-1)"
    )
    target_max_drawdown: float = Field(
        default=25.0, ge=5.0, le=50.0, description="Target max drawdown %"
    )
    quick_mode: bool = Field(
        default=True, description="Use quick mode (synthetic data) for faster optimization"
    )


class AutoModeStatusResponse(BaseModel):
    """Response with auto mode status."""

    session_id: str
    state: str
    iteration: int
    max_iterations: int
    progress_pct: float
    best_score: float | None
    best_params: dict[str, Any] | None
    best_metrics: dict[str, float] | None
    elapsed_seconds: float


class AutoModeResultResponse(BaseModel):
    """Response with final auto mode result."""

    session_id: str
    state: str
    stopping_reason: str | None
    best_params: dict[str, Any]
    best_metrics: dict[str, float]
    best_score: float
    total_iterations: int
    total_time_seconds: float
    improvements_count: int
    pareto_front: list[dict[str, Any]]


class SessionListItem(BaseModel):
    """Session summary for listing."""

    session_id: str
    state: str
    progress_pct: float
    created_at: str


# ============================================================================
# Session Manager (Global State)
# ============================================================================

# Global session manager
_session_manager = None
_auto_results: dict[str, dict[str, Any]] = {}


def get_session_manager():
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        from coldpath.backtest.auto_backtest import AutoModeSessionManager

        _session_manager = AutoModeSessionManager()
    return _session_manager


# ============================================================================
# Guided Setup Endpoints
# ============================================================================


@router.post("/setup", response_model=GuidedSetupResponse)
async def guided_setup(request: GuidedSetupRequest):
    """Get AI-guided parameter recommendations.

    This endpoint analyzes your risk profile and goals to provide
    intelligent parameter recommendations that eliminate trial-and-error.

    Returns:
        - Recommended parameters with rationale
        - Expected metrics based on your profile
        - Validation warnings if any
        - Next steps for getting started
    """
    from coldpath.backtest.ai_backtest_guide import AIBacktestGuide

    try:
        guide = AIBacktestGuide()
        result = guide.create_guided_setup(
            risk_tolerance=request.risk_tolerance,
            primary_goal=request.primary_goal,
            capital_sol=request.capital_sol,
            memecoin_focus=request.memecoin_focus,
            max_daily_trades=request.max_daily_trades,
            preferred_hold_time_minutes=request.preferred_hold_time_minutes,
            market_regime=request.market_regime,
        )

        recommendations = [
            ParameterRecommendationResponse(
                key=r.key,
                label=r.label,
                value=r.value,
                min_value=r.min_value,
                max_value=r.max_value,
                unit=r.unit,
                rationale=r.rationale,
                confidence=r.confidence,
                can_auto_tune=r.can_auto_tune,
            )
            for r in result.recommendations
        ]

        return GuidedSetupResponse(
            recommendations=recommendations,
            suggested_config=result.suggested_config,
            expected_metrics=result.expected_metrics,
            validation_warnings=result.validation_warnings,
            next_steps=result.next_steps,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Guided setup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/quick-setup", response_model=QuickSetupResponse)
async def quick_setup(request: QuickSetupRequest):
    """Quick setup with predefined presets.

    Provides instant configuration based on common trading profiles:
    - conservative: Capital preservation, low risk
    - balanced: Moderate growth with reasonable risk
    - aggressive: Higher risk for higher returns
    - degen: Maximum risk for maximum potential

    Returns:
        Ready-to-use configuration dict
    """
    from coldpath.backtest.ai_backtest_guide import AIBacktestGuide

    preset_descriptions = {
        "conservative": (
            "Low-risk setup focused on capital preservation. "
            "Tight stops, high liquidity requirements."
        ),
        "balanced": "Balanced approach with moderate risk. Good for most traders starting out.",
        "aggressive": "Higher risk tolerance for better returns. Wider stops, larger positions.",
        "degen": "Maximum risk for memecoin hunting. Only for experienced traders.",
    }

    if request.preset not in preset_descriptions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown preset: {request.preset}. Use: conservative, balanced, aggressive, degen"
            ),
        )

    try:
        guide = AIBacktestGuide()
        config = guide.quick_setup(
            preset=request.preset,
            capital_sol=request.capital_sol,
        )

        return QuickSetupResponse(
            preset=request.preset,
            config=config,
            description=preset_descriptions[request.preset],
        )

    except Exception as e:
        logger.error(f"Quick setup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/analyze", response_model=AnalyzeParamsResponse)
async def analyze_params(request: AnalyzeParamsRequest):
    """Analyze current parameters and get improvement suggestions.

    Examines your current configuration for:
    - Risk/reward ratio issues
    - Position sizing problems
    - Liquidity filter concerns
    - Slippage tolerance issues

    Returns analysis with specific improvement suggestions.
    """
    from coldpath.backtest.ai_backtest_guide import AIBacktestGuide

    try:
        guide = AIBacktestGuide()
        analysis = guide.analyze_current_params(request.current_config)

        return AnalyzeParamsResponse(
            issues=analysis["issues"],
            suggestions=analysis["suggestions"],
            optimization_potential=analysis["optimization_potential"],
        )

    except Exception as e:
        logger.error(f"Parameter analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/param-info/{param_key}", response_model=ParamInfoResponse)
async def get_param_info(param_key: str):
    """Get detailed explanation of a parameter.

    Provides comprehensive information about a specific parameter:
    - What it does
    - How it impacts performance
    - Typical value ranges
    - Pro tips for optimal settings
    """
    from coldpath.backtest.ai_backtest_guide import AIBacktestGuide

    try:
        guide = AIBacktestGuide()
        info = guide.get_parameter_explanation(param_key)

        return ParamInfoResponse(
            name=info["name"],
            description=info["description"],
            impact=info["impact"],
            typical_range=info["typical_range"],
            pro_tip=info["pro_tip"],
        )

    except Exception:
        raise HTTPException(status_code=404, detail=f"Unknown parameter: {param_key}") from None


# ============================================================================
# Auto Mode Endpoints
# ============================================================================


async def _create_backtest_runner(quick_mode: bool = True):
    """Create a backtest runner function for auto mode."""
    from datetime import datetime, timedelta

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
            # Return default metrics on failure
            return {
                "sharpe_ratio": 0.0,
                "win_rate_pct": 0.0,
                "max_drawdown_pct": 100.0,
                "total_return_pct": -100.0,
                "profit_factor": 0.0,
                "profit_quality_score": 0.0,
            }

    return run_backtest


@router.post("/auto/start", response_model=AutoModeStatusResponse)
async def start_auto_mode(
    request: AutoModeStartRequest,
    background_tasks: BackgroundTasks,
):
    """Start automatic parameter optimization.

    Runs optimization in background to find optimal parameters
    without manual trial-and-error.

    The system will:
    1. Start with provided or default parameters
    2. Run iterative backtests
    3. Learn from results
    4. Converge on optimal parameters

    Returns session ID for tracking progress.
    """
    from coldpath.backtest.auto_backtest import AutoModeConfig

    session_id = str(uuid4())
    manager = get_session_manager()

    # Create config
    from coldpath.backtest.auto_backtest import OptimizationTarget

    config = AutoModeConfig(
        max_iterations=request.max_iterations,
        max_time_minutes=request.max_time_minutes,
        targets=OptimizationTarget(
            min_sharpe=request.target_sharpe,
            min_win_rate=request.target_win_rate,
            max_drawdown=request.target_max_drawdown,
        ),
    )

    # Create runner
    runner = await _create_backtest_runner(quick_mode=request.quick_mode)

    # Create session
    session = manager.create_session(
        session_id=session_id,
        backtest_runner=runner,
        config=config,
    )

    # Run in background
    async def run_optimization():
        try:
            result = await session.run(initial_params=request.initial_params)
            _auto_results[session_id] = result.to_dict()
        except Exception as e:
            logger.error(f"Auto optimization {session_id} failed: {e}")
            _auto_results[session_id] = {"error": str(e)}

    background_tasks.add_task(run_optimization)

    return AutoModeStatusResponse(
        session_id=session_id,
        state="running",
        iteration=0,
        max_iterations=request.max_iterations,
        progress_pct=0.0,
        best_score=None,
        best_params=None,
        best_metrics=None,
        elapsed_seconds=0.0,
    )


@router.get("/auto/status/{session_id}", response_model=AutoModeStatusResponse)
async def get_auto_status(session_id: str):
    """Get current status of auto optimization session.

    Returns:
    - Current iteration and progress
    - Best score found so far
    - Best parameters
    - Estimated metrics
    """
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    status = session.get_current_status()

    return AutoModeStatusResponse(
        session_id=session_id,
        state=status["state"],
        iteration=status["iteration"],
        max_iterations=status["max_iterations"],
        progress_pct=status["progress_pct"],
        best_score=status.get("best_score"),
        best_params=status.get("best_params"),
        best_metrics=None,  # Would need to track this
        elapsed_seconds=status["elapsed_seconds"],
    )


@router.get("/auto/result/{session_id}", response_model=AutoModeResultResponse)
async def get_auto_result(session_id: str):
    """Get final result of completed auto optimization.

    Only available after optimization has completed.
    Returns full results including:
    - Best parameters found
    - All metrics
    - Optimization history
    - Pareto front (multi-objective view)
    """
    if session_id not in _auto_results:
        raise HTTPException(
            status_code=404,
            detail="Result not found. Session may still be running or never existed.",
        )

    result = _auto_results[session_id]

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return AutoModeResultResponse(
        session_id=session_id,
        state=result["state"],
        stopping_reason=result.get("stopping_reason"),
        best_params=result["best_params"],
        best_metrics=result["best_metrics"],
        best_score=result["best_score"],
        total_iterations=result["total_iterations"],
        total_time_seconds=result["total_time_seconds"],
        improvements_count=result["improvements_count"],
        pareto_front=result.get("pareto_front", []),
    )


@router.post("/auto/cancel/{session_id}")
async def cancel_auto_mode(session_id: str):
    """Cancel a running auto optimization session."""
    manager = get_session_manager()

    if not manager.cancel_session(session_id):
        raise HTTPException(
            status_code=400, detail="Could not cancel session. It may not be running."
        )

    return {"status": "cancelled", "session_id": session_id}


@router.get("/auto/sessions", response_model=list[SessionListItem])
async def list_auto_sessions():
    """List all auto optimization sessions."""
    manager = get_session_manager()
    sessions = manager.get_all_sessions()

    items = []
    for session_id, status in sessions.items():
        items.append(
            SessionListItem(
                session_id=session_id,
                state=status["state"],
                progress_pct=status["progress_pct"],
                created_at=datetime.utcnow().isoformat(),  # Would track actual creation
            )
        )

    return items


@router.delete("/auto/sessions/cleanup")
async def cleanup_completed_sessions():
    """Remove all completed/failed sessions."""
    manager = get_session_manager()
    removed_count = manager.cleanup_completed()

    return {
        "status": "cleaned",
        "sessions_removed": removed_count,
    }


# ============================================================================
# Feedback Loop Integration
# ============================================================================


def _get_backtest_history() -> list[dict[str, Any]]:
    """Get the shared backtest history from app state."""
    try:
        from coldpath.api.server import get_app_state

        state = get_app_state()
        if state.backtest_history is None:
            state.backtest_history = []
        return state.backtest_history
    except Exception:
        return []


def _store_optimization_result(
    params: dict[str, Any],
    metrics: dict[str, float],
    risk_tolerance: str,
    score: float,
):
    """Store optimization result in shared history for learning."""
    try:
        from datetime import datetime

        from coldpath.api.server import get_app_state

        state = get_app_state()
        if state.backtest_history is None:
            state.backtest_history = []

        result = {
            "params": params,
            "metrics": metrics,
            "risk_tolerance": risk_tolerance,
            "score": score,
            "timestamp": datetime.utcnow().isoformat(),
        }

        state.backtest_history.append(result)

        # Limit history size
        if len(state.backtest_history) > 500:
            state.backtest_history = state.backtest_history[-300:]

        # Also feed to feedback loop if available
        if state.feedback_loop is not None:
            try:
                # Feed as outcome for model training
                state.feedback_loop.add_backtest_outcome(result)
            except Exception as e:
                logger.debug(f"Could not feed to feedback loop: {e}")

    except Exception as e:
        logger.warning(f"Failed to store optimization result: {e}")


def _get_feedback_loop():
    """Get the feedback loop from app state."""
    try:
        from coldpath.api.server import get_app_state

        state = get_app_state()
        return state.feedback_loop
    except Exception:
        return None


# ============================================================================
# Smart Optimization Endpoints (Phase 5)
# ============================================================================


class SmartOptimizeRequest(BaseModel):
    """Request for smart optimization."""

    risk_tolerance: str = Field(default="moderate")
    primary_goal: str = Field(default="balanced")
    capital_sol: float = Field(default=1.0, ge=0.01, le=100000.0, description="Capital in SOL")
    strategy: str = Field(default="balanced")  # fast, balanced, thorough, adaptive
    enable_regime_detection: bool = Field(default=True)
    enable_learning: bool = Field(default=True)
    quick_mode: bool = Field(default=True)
    # Market metrics for regime detection
    market_metrics: dict[str, Any] | None = Field(default=None)


class SmartOptimizeResponse(BaseModel):
    """Response from smart optimization."""

    best_params: dict[str, Any]
    best_metrics: dict[str, float]
    best_score: float
    detected_regime: str | None = None
    regime_adjustments: list[dict[str, Any]] = []
    optimization_iterations: int = 0
    optimization_time_seconds: float = 0.0
    improvement_over_baseline: float = 0.0
    learned_from_history: bool = False
    historical_samples_used: int = 0
    recommendations: list[str] = []
    warnings: list[str] = []


class QuickOptimizeRequest(BaseModel):
    """Request for quick optimization."""

    risk_tolerance: str = Field(default="moderate")
    capital_sol: float = Field(default=1.0, ge=0.01, le=100000.0, description="Capital in SOL")
    preset: str | None = Field(default=None)


class RegimeAnalyzeRequest(BaseModel):
    """Request for regime analysis."""

    market_metrics: dict[str, Any] = Field(..., description="Market metrics for regime detection")
    current_params: dict[str, Any] | None = Field(
        default=None, description="Optional params to adjust based on regime"
    )


class RegimeAnalyzeResponse(BaseModel):
    """Response from regime analysis."""

    primary_regime: str
    secondary_regime: str | None = None
    volatility_level: str
    trend_strength: str
    confidence: float
    risk_level: str
    recommended_actions: list[str]
    adjusted_params: dict[str, Any] | None = None


@router.post("/smart/optimize", response_model=SmartOptimizeResponse)
async def smart_optimize(request: SmartOptimizeRequest):
    """Full smart optimization with all AI components.

    This is the most intelligent optimization method, combining:
    - AI-guided initial parameters
    - Market regime detection and adjustment
    - Bayesian optimization
    - Learning from historical results

    Returns optimized parameters with full analysis.
    """
    from coldpath.backtest.market_regime import (
        create_metrics_from_market_data,
    )
    from coldpath.backtest.smart_orchestrator import (
        LearningMode,
        OptimizationStrategy,
        SmartBacktestOrchestrator,
        SmartOrchestratorConfig,
    )

    try:
        # Map strategy
        strategy_map = {
            "fast": OptimizationStrategy.FAST,
            "balanced": OptimizationStrategy.BALANCED,
            "thorough": OptimizationStrategy.THOROUGH,
            "adaptive": OptimizationStrategy.ADAPTIVE,
        }

        config = SmartOrchestratorConfig(
            optimization_strategy=strategy_map.get(request.strategy, OptimizationStrategy.BALANCED),
            enable_regime_detection=request.enable_regime_detection,
            learning_mode=(
                LearningMode.ACTIVE if request.enable_learning else LearningMode.DISABLED
            ),
        )

        # Create backtest runner
        runner = await _create_backtest_runner(quick_mode=request.quick_mode)

        # Get shared history for learning
        history = _get_backtest_history()

        # Create orchestrator with shared history
        orchestrator = SmartBacktestOrchestrator(
            backtest_runner=runner,
            config=config,
            historical_results=history,
            feedback_store=_get_feedback_loop(),
        )

        # Parse market metrics if provided
        market_metrics = None
        if request.market_metrics:
            market_metrics = create_metrics_from_market_data(
                price_data=request.market_metrics,
            )

        # Run smart optimization
        result = await orchestrator.smart_optimize(
            risk_tolerance=request.risk_tolerance,
            primary_goal=request.primary_goal,
            capital_sol=request.capital_sol,
            market_metrics=market_metrics,
        )

        # Store result for future learning and feedback loop
        _store_optimization_result(
            params=result.best_params,
            metrics=result.best_metrics,
            risk_tolerance=request.risk_tolerance,
            score=result.best_score,
        )

        return SmartOptimizeResponse(
            best_params=result.best_params,
            best_metrics=result.best_metrics,
            best_score=result.best_score,
            detected_regime=result.detected_regime,
            regime_adjustments=result.regime_adjustments,
            optimization_iterations=result.optimization_iterations,
            optimization_time_seconds=result.optimization_time_seconds,
            improvement_over_baseline=result.improvement_over_baseline,
            learned_from_history=result.learned_from_history,
            historical_samples_used=result.historical_samples_used,
            recommendations=result.recommendations,
            warnings=result.warnings,
        )

    except Exception as e:
        logger.error(f"Smart optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/smart/quick", response_model=SmartOptimizeResponse)
async def quick_optimize(request: QuickOptimizeRequest):
    """Quick optimization with preset parameters.

    Fastest way to get optimized parameters:
    - Uses preset based on risk tolerance
    - Runs brief optimization
    - Returns optimized parameters quickly
    """
    from coldpath.backtest.smart_orchestrator import (
        OptimizationStrategy,
        SmartBacktestOrchestrator,
        SmartOrchestratorConfig,
    )

    try:
        config = SmartOrchestratorConfig(
            optimization_strategy=OptimizationStrategy.FAST,
        )

        runner = await _create_backtest_runner(quick_mode=True)

        # Get shared history for learning
        history = _get_backtest_history()

        orchestrator = SmartBacktestOrchestrator(
            backtest_runner=runner,
            config=config,
            historical_results=history,
        )

        result = await orchestrator.quick_optimize(
            risk_tolerance=request.risk_tolerance,
            capital_sol=request.capital_sol,
            preset=request.preset,
        )

        # Store result for future learning
        _store_optimization_result(
            params=result.best_params,
            metrics=result.best_metrics,
            risk_tolerance=request.risk_tolerance,
            score=result.best_score,
        )

        return SmartOptimizeResponse(
            best_params=result.best_params,
            best_metrics=result.best_metrics,
            best_score=result.best_score,
            detected_regime=result.detected_regime,
            optimization_iterations=result.optimization_iterations,
            optimization_time_seconds=result.optimization_time_seconds,
            improvement_over_baseline=result.improvement_over_baseline,
            recommendations=result.recommendations,
            warnings=result.warnings,
        )

    except Exception as e:
        logger.error(f"Quick optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/smart/regime", response_model=RegimeAnalyzeResponse)
async def analyze_regime(request: RegimeAnalyzeRequest):
    """Analyze current market regime.

    Detects market conditions and provides:
    - Regime classification (trending, volatile, etc.)
    - Parameter adjustment recommendations
    - Risk assessment
    - Recommended actions
    """
    from coldpath.backtest.market_regime import (
        MarketRegimeDetector,
        create_metrics_from_market_data,
    )

    try:
        detector = MarketRegimeDetector()

        # Create metrics from request
        metrics = create_metrics_from_market_data(
            price_data=request.market_metrics,
        )

        # Analyze regime
        analysis = detector.analyze(metrics)

        # Adjust params if provided
        adjusted_params = None
        if request.current_params:
            adjusted_params = detector.adjust_parameters(request.current_params, analysis)

        return RegimeAnalyzeResponse(
            primary_regime=analysis.primary_regime.value,
            secondary_regime=(
                analysis.secondary_regime.value if analysis.secondary_regime else None
            ),
            volatility_level=analysis.volatility_level.value,
            trend_strength=analysis.trend_strength.value,
            confidence=analysis.confidence,
            risk_level=analysis.risk_level,
            recommended_actions=analysis.recommended_actions,
            adjusted_params=adjusted_params,
        )

    except Exception as e:
        logger.error(f"Regime analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Integration Status & History Endpoints
# ============================================================================


class IntegrationStatusResponse(BaseModel):
    """Response with integration status."""

    feedback_loop_connected: bool
    backtest_history_count: int
    feedback_loop_outcomes_count: int
    smart_orchestrator_configured: bool


class BacktestHistoryItem(BaseModel):
    """Single backtest history item."""

    timestamp: str
    risk_tolerance: str
    score: float
    sharpe_ratio: float | None = None
    win_rate_pct: float | None = None
    total_return_pct: float | None = None


class BacktestHistoryResponse(BaseModel):
    """Response with backtest history."""

    total_count: int
    items: list[BacktestHistoryItem]


@router.get("/integration/status", response_model=IntegrationStatusResponse)
async def get_integration_status():
    """Get status of AI guided backtest integrations.

    Shows:
    - Whether feedback loop is connected
    - Number of backtest results in history
    - Number of outcomes in feedback loop
    """
    feedback_loop = _get_feedback_loop()
    history = _get_backtest_history()

    feedback_outcomes = 0
    if feedback_loop is not None:
        try:
            feedback_outcomes = len(feedback_loop.get_backtest_outcomes())
        except Exception:
            pass

    return IntegrationStatusResponse(
        feedback_loop_connected=feedback_loop is not None,
        backtest_history_count=len(history),
        feedback_loop_outcomes_count=feedback_outcomes,
        smart_orchestrator_configured=True,
    )


@router.get("/integration/history", response_model=BacktestHistoryResponse)
async def get_backtest_history(limit: int = 50):
    """Get backtest optimization history.

    Shows recent optimization results stored for learning.
    """
    history = _get_backtest_history()

    # Sort by timestamp descending
    sorted_history = sorted(
        history,
        key=lambda x: x.get("timestamp", ""),
        reverse=True,
    )[:limit]

    items = []
    for item in sorted_history:
        metrics = item.get("metrics", {})
        items.append(
            BacktestHistoryItem(
                timestamp=item.get("timestamp", ""),
                risk_tolerance=item.get("risk_tolerance", "unknown"),
                score=item.get("score", 0),
                sharpe_ratio=metrics.get("sharpe_ratio"),
                win_rate_pct=metrics.get("win_rate_pct"),
                total_return_pct=metrics.get("total_return_pct"),
            )
        )

    return BacktestHistoryResponse(
        total_count=len(history),
        items=items,
    )


@router.delete("/integration/history")
async def clear_backtest_history():
    """Clear all stored backtest history.

    WARNING: This will reset the learning history.
    """
    try:
        from coldpath.api.server import get_app_state

        state = get_app_state()
        state.backtest_history = []

        # Also clear feedback loop backtest outcomes
        feedback_loop = _get_feedback_loop()
        if feedback_loop is not None:
            try:
                feedback_loop._backtest_outcomes = []
            except Exception:
                pass

        return {"status": "cleared", "message": "Backtest history cleared"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/integration/sync")
async def sync_to_feedback_loop():
    """Manually sync backtest history to feedback loop.

    Useful if feedback loop was not connected during optimizations.
    """
    feedback_loop = _get_feedback_loop()
    if feedback_loop is None:
        raise HTTPException(status_code=503, detail="Feedback loop not available")

    history = _get_backtest_history()
    synced = 0

    for item in history:
        try:
            feedback_loop.add_backtest_outcome(item)
            synced += 1
        except Exception as e:
            logger.warning(f"Failed to sync item: {e}")

    return {
        "status": "synced",
        "items_synced": synced,
        "total_items": len(history),
    }
