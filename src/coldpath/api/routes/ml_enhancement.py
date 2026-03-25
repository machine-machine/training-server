"""
ML Enhancement Agent API Routes.

Exposes the ML Enhancement Agent via REST API for integration
with the trading system dashboard and automation.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from coldpath.ai.ml_enhancement_agent import (
    EnhancementResult,
    MLEnhancementAgent,
    MLPrediction,
    ModelMetrics,
    TradeOutcome,
    get_ml_enhancement_agent,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml-enhancement", tags=["ml-enhancement"])


# =============================================================================
# Request/Response Models
# =============================================================================


class MLPredictionRequest(BaseModel):
    """Request model for prediction explanation."""

    score: float = Field(..., ge=0.0, le=1.0, description="ML score (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence (0-1)")
    signal: str = Field(..., description="Trading signal (BUY/SELL/HOLD)")
    fraud_score: float = Field(0.0, ge=0.0, le=1.0, description="Fraud score")
    regime: str = Field("unknown", description="Market regime")
    regime_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Regime confidence")
    token_data: dict[str, Any] | None = Field(None, description="Additional token context")


class ParameterSuggestionRequest(BaseModel):
    """Request model for parameter suggestions."""

    current_params: dict[str, Any] = Field(..., description="Current strategy parameters")
    metrics: dict[str, Any] = Field(..., description="Model performance metrics")
    recent_trades: list[dict[str, Any]] = Field(default_factory=list, description="Recent trades")


class AnomalyAnalysisRequest(BaseModel):
    """Request model for anomaly analysis."""

    anomaly_data: dict[str, Any] = Field(..., description="Detected anomaly details")
    context: dict[str, Any] | None = Field(None, description="Additional context")


class FeedbackSynthesisRequest(BaseModel):
    """Request model for feedback synthesis."""

    trades: list[dict[str, Any]] = Field(..., description="Trade outcomes")
    metrics: dict[str, Any] = Field(..., description="Model metrics")
    learning_insights: dict[str, Any] | None = Field(None, description="Additional insights")


class RegimeNarrativeRequest(BaseModel):
    """Request model for regime narrative."""

    regime: str = Field(..., description="Current market regime")
    regime_confidence: float = Field(..., ge=0.0, le=1.0, description="Regime confidence")
    market_data: dict[str, Any] = Field(..., description="Market data")
    transition_probability: float = Field(0.0, ge=0.0, le=1.0, description="Transition probability")


class PortfolioRiskRequest(BaseModel):
    """Request model for portfolio risk assessment."""

    positions: list[dict[str, Any]] = Field(..., description="Current positions")
    portfolio_value: float = Field(..., gt=0, description="Total portfolio value")
    market_conditions: dict[str, Any] = Field(..., description="Market conditions")
    ml_predictions: list[dict[str, Any]] | None = Field(None, description="ML predictions")


class ModelDriftRequest(BaseModel):
    """Request model for drift detection."""

    baseline_metrics: dict[str, Any] = Field(..., description="Baseline metrics")
    current_metrics: dict[str, Any] = Field(..., description="Current metrics")
    feature_distribution_shift: dict[str, float] | None = Field(None, description="Feature drift")
    prediction_distribution: dict[str, Any] | None = Field(
        None, description="Prediction distribution"
    )


class FeatureImportanceRequest(BaseModel):
    """Request model for feature importance analysis."""

    feature_importance: dict[str, float] = Field(..., description="Feature importance scores")
    performance_by_feature: dict[str, dict[str, float]] | None = Field(
        None, description="Performance by feature"
    )
    correlation_matrix: dict[str, dict[str, float]] | None = Field(
        None, description="Correlation matrix"
    )


class ImprovementCycleRequest(BaseModel):
    """Request model for continuous improvement cycle."""

    trades: list[dict[str, Any]] = Field(..., description="Recent trades")
    metrics: dict[str, Any] = Field(..., description="Model metrics")
    current_params: dict[str, Any] = Field(..., description="Current parameters")


class EnhancementResponse(BaseModel):
    """Response model for enhancement results."""

    enhancement_type: str = Field(..., description="Type of enhancement")
    success: bool = Field(..., description="Whether enhancement succeeded")
    content: str = Field(..., description="Raw LLM content")
    structured_data: dict[str, Any] | None = Field(None, description="Parsed JSON data")
    confidence: float = Field(0.0, description="Confidence in result")
    latency_ms: float = Field(0.0, description="Processing latency")
    tokens_used: int = Field(0, description="Tokens generated")
    error: str | None = Field(None, description="Error message if failed")
    timestamp: str = Field(..., description="ISO timestamp")


class AgentStatsResponse(BaseModel):
    """Response model for agent statistics."""

    total_enhancements: int = Field(..., description="Total enhancements performed")
    successful_enhancements: int = Field(..., description="Successful enhancements")
    success_rate: float = Field(..., description="Success rate (0-1)")
    avg_latency_ms: float = Field(..., description="Average latency")
    total_tokens: int = Field(..., description="Total tokens used")
    by_type: dict[str, dict[str, Any]] = Field(..., description="Stats by enhancement type")


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_trade_outcome(data: dict[str, Any]) -> TradeOutcome:
    """Parse a TradeOutcome from dict."""
    return TradeOutcome(
        token_symbol=data.get("token_symbol", "UNKNOWN"),
        action=data.get("action", "BUY"),
        entry_price=data.get("entry_price", 0.0),
        exit_price=data.get("exit_price"),
        pnl_pct=data.get("pnl_pct"),
        hold_time_seconds=data.get("hold_time_seconds", 0.0),
        ml_score=data.get("ml_score", 0.5),
        ml_confidence=data.get("ml_confidence", 0.5),
        ml_signal=data.get("ml_signal", "HOLD"),
        market_regime=data.get("market_regime", "unknown"),
        success=data.get("success", False),
        timestamp=datetime.fromisoformat(data["timestamp"])
        if "timestamp" in data
        else datetime.utcnow(),
    )


def _parse_model_metrics(data: dict[str, Any]) -> ModelMetrics:
    """Parse ModelMetrics from dict."""
    return ModelMetrics(
        total_trades=data.get("total_trades", 0),
        win_rate=data.get("win_rate", 0.0),
        avg_pnl_pct=data.get("avg_pnl_pct", 0.0),
        sharpe_ratio=data.get("sharpe_ratio", 0.0),
        max_drawdown=data.get("max_drawdown", 0.0),
        prediction_accuracy=data.get("prediction_accuracy", 0.0),
        calibration_error=data.get("calibration_error", 0.0),
        feature_drift_score=data.get("feature_drift_score", 0.0),
        regime_accuracy=data.get("regime_accuracy", 0.0),
        fraud_detection_rate=data.get("fraud_detection_rate", 0.0),
        false_positive_rate=data.get("false_positive_rate", 0.0),
    )


def _result_to_response(result: EnhancementResult) -> EnhancementResponse:
    """Convert EnhancementResult to response model."""
    return EnhancementResponse(
        enhancement_type=result.enhancement_type.value,
        success=result.success,
        content=result.content,
        structured_data=result.structured_data,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        tokens_used=result.tokens_used,
        error=result.error,
        timestamp=result.timestamp.isoformat(),
    )


def get_agent() -> MLEnhancementAgent:
    """Dependency to get the ML Enhancement Agent."""
    return get_ml_enhancement_agent()


# =============================================================================
# Routes
# =============================================================================


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ml-enhancement-agent"}


@router.get("/stats", response_model=AgentStatsResponse)
async def get_stats():
    """Get agent statistics."""
    agent = get_agent()
    return agent.get_stats()


@router.post("/explain", response_model=EnhancementResponse)
async def explain_prediction(request: MLPredictionRequest):
    """
    Explain an ML prediction in human-readable terms.

    Provides:
    - Summary of the prediction
    - Signal reasoning
    - Risk factors
    - Actionable recommendations
    """
    agent = get_agent()

    prediction = MLPrediction(
        score=request.score,
        confidence=request.confidence,
        signal=request.signal,
        fraud_score=request.fraud_score,
        regime=request.regime,
        regime_confidence=request.regime_confidence,
    )

    result = await agent.explain_prediction(prediction, request.token_data)
    return _result_to_response(result)


@router.post("/suggest-parameters", response_model=EnhancementResponse)
async def suggest_parameters(request: ParameterSuggestionRequest):
    """
    Suggest parameter optimizations based on performance data.

    Analyzes:
    - Current parameters vs performance
    - Recent trade patterns
    - Risk/reward tradeoffs
    """
    agent = get_agent()

    metrics = _parse_model_metrics(request.metrics)
    trades = [_parse_trade_outcome(t) for t in request.recent_trades]

    result = await agent.suggest_parameters(request.current_params, metrics, trades)
    return _result_to_response(result)


@router.post("/analyze-anomaly", response_model=EnhancementResponse)
async def analyze_anomaly(request: AnomalyAnalysisRequest):
    """
    Analyze an anomaly and provide root cause hypothesis.

    Provides:
    - Anomaly classification
    - Severity assessment
    - Root cause hypothesis
    - Recommended actions
    """
    agent = get_agent()

    result = await agent.analyze_anomaly(request.anomaly_data, request.context)
    return _result_to_response(result)


@router.post("/synthesize-feedback", response_model=EnhancementResponse)
async def synthesize_feedback(request: FeedbackSynthesisRequest):
    """
    Synthesize learning feedback from trade outcomes.

    Provides:
    - Pattern discoveries
    - Model improvement suggestions
    - Confidence adjustments
    - Learning focus recommendations
    """
    agent = get_agent()

    metrics = _parse_model_metrics(request.metrics)
    trades = [_parse_trade_outcome(t) for t in request.trades]

    result = await agent.synthesize_feedback(trades, metrics, request.learning_insights)
    return _result_to_response(result)


@router.post("/regime-narrative", response_model=EnhancementResponse)
async def generate_regime_narrative(request: RegimeNarrativeRequest):
    """
    Generate a narrative explanation of current market regime.

    Provides:
    - Regime characteristics
    - Trading implications
    - Regime stability assessment
    - Risk considerations
    """
    agent = get_agent()

    result = await agent.generate_regime_narrative(
        request.regime,
        request.regime_confidence,
        request.market_data,
        request.transition_probability,
    )
    return _result_to_response(result)


@router.post("/assess-risk", response_model=EnhancementResponse)
async def assess_portfolio_risk(request: PortfolioRiskRequest):
    """
    Assess portfolio risk and provide recommendations.

    Provides:
    - Overall risk assessment
    - Risk breakdown by type
    - Position recommendations
    - Stress test results
    """
    agent = get_agent()

    ml_predictions = None
    if request.ml_predictions:
        ml_predictions = [MLPrediction(**p) for p in request.ml_predictions]

    result = await agent.assess_portfolio_risk(
        request.positions,
        request.portfolio_value,
        request.market_conditions,
        ml_predictions,
    )
    return _result_to_response(result)


@router.post("/detect-drift", response_model=EnhancementResponse)
async def detect_model_drift(request: ModelDriftRequest):
    """
    Detect and analyze model drift.

    Provides:
    - Drift severity assessment
    - Performance drift analysis
    - Feature drift analysis
    - Recommended actions
    """
    agent = get_agent()

    baseline = _parse_model_metrics(request.baseline_metrics)
    current = _parse_model_metrics(request.current_metrics)

    result = await agent.detect_model_drift(
        baseline,
        current,
        request.feature_distribution_shift,
        request.prediction_distribution,
    )
    return _result_to_response(result)


@router.post("/analyze-features", response_model=EnhancementResponse)
async def analyze_feature_importance(request: FeatureImportanceRequest):
    """
    Analyze feature importance and suggest optimizations.

    Provides:
    - Top predictive features
    - Underperforming features
    - Redundant features
    - Feature engineering suggestions
    """
    agent = get_agent()

    result = await agent.optimize_feature_importance(
        request.feature_importance,
        request.performance_by_feature,
        request.correlation_matrix,
    )
    return _result_to_response(result)


@router.post("/improvement-cycle")
async def run_improvement_cycle(request: ImprovementCycleRequest):
    """
    Run a complete continuous improvement cycle.

    Combines:
    - Feedback synthesis
    - Parameter suggestions
    - Prioritized action list
    """
    agent = get_agent()

    metrics = _parse_model_metrics(request.metrics)
    trades = [_parse_trade_outcome(t) for t in request.trades]

    result = await agent.continuous_improvement_cycle(
        trades,
        metrics,
        request.current_params,
    )
    return result


@router.post("/reset-stats")
async def reset_stats():
    """Reset agent statistics."""
    global _ml_enhancement_agent
    _ml_enhancement_agent = None
    return {"status": "reset", "message": "Agent statistics reset"}


# =============================================================================
# Register with Main App
# =============================================================================


def register_ml_enhancement_routes(app):
    """Register ML enhancement routes with the FastAPI app."""
    app.include_router(router)
