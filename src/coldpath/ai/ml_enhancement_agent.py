"""
ML Enhancement Agent - FinGPT-powered ML trading system improvements.

This agent bridges local LLM (FinGPT/Ollama) with the ML trading infrastructure,
providing:
- ML prediction explainability and interpretability
- Strategy parameter optimization suggestions
- Anomaly detection with natural language reasoning
- Continuous learning feedback synthesis
- Market regime narrative generation
- Risk assessment narratives

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    ML Enhancement Agent                      │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │   FinGPT      │  │   ML Models   │  │  Trade Data   │   │
│  │   (Ollama)    │  │   (Ensemble)  │  │  (Outcomes)   │   │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘   │
│          │                  │                  │            │
│          └──────────────────┴──────────────────┘            │
│                             │                               │
│  ┌──────────────────────────┴──────────────────────────┐   │
│  │              Enhancement Capabilities               │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ • explain_prediction() - Human-readable ML insights │   │
│  │ • suggest_parameters() - Strategy optimization      │   │
│  │ • analyze_anomaly() - Unusual pattern reasoning     │   │
│  │ • synthesize_feedback() - Learning recommendations  │   │
│  │ • generate_regime_narrative() - Market context      │   │
│  │ • assess_portfolio_risk() - Risk narratives         │   │
│  │ • detect_model_drift() - Model health analysis      │   │
│  │ • optimize_feature_importance() - Feature insights  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from .fingpt_client import FinGPTClient, get_fingpt_client

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


class EnhancementType(Enum):
    """Types of ML enhancements available."""

    PREDICTION_EXPLANATION = "prediction_explanation"
    PARAMETER_SUGGESTION = "parameter_suggestion"
    ANOMALY_ANALYSIS = "anomaly_analysis"
    FEEDBACK_SYNTHESIS = "feedback_synthesis"
    REGIME_NARRATIVE = "regime_narrative"
    RISK_ASSESSMENT = "risk_assessment"
    MODEL_DRIFT = "model_drift"
    FEATURE_IMPORTANCE = "feature_importance"


@dataclass
class MLPrediction:
    """ML model prediction with context."""

    score: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    signal: str  # BUY/SELL/HOLD
    fraud_score: float = 0.0
    regime: str = "unknown"
    regime_confidence: float = 0.5
    features: np.ndarray | None = None
    feature_names: list[str] | None = None
    model_version: int = 1
    timestamp_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "confidence": self.confidence,
            "signal": self.signal,
            "fraud_score": self.fraud_score,
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "model_version": self.model_version,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class TradeOutcome:
    """Outcome of a trade for learning."""

    token_symbol: str
    action: str  # BUY/SELL
    entry_price: float
    exit_price: float | None
    pnl_pct: float | None
    hold_time_seconds: float
    ml_score: float
    ml_confidence: float
    ml_signal: str
    market_regime: str
    success: bool
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_symbol": self.token_symbol,
            "action": self.action,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl_pct": self.pnl_pct,
            "hold_time_seconds": self.hold_time_seconds,
            "ml_score": self.ml_score,
            "ml_confidence": self.ml_confidence,
            "ml_signal": self.ml_signal,
            "market_regime": self.market_regime,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ModelMetrics:
    """Current ML model performance metrics."""

    total_trades: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    prediction_accuracy: float = 0.0
    calibration_error: float = 0.0
    feature_drift_score: float = 0.0
    regime_accuracy: float = 0.0
    fraud_detection_rate: float = 0.0
    false_positive_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "avg_pnl_pct": self.avg_pnl_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "prediction_accuracy": self.prediction_accuracy,
            "calibration_error": self.calibration_error,
            "feature_drift_score": self.feature_drift_score,
            "regime_accuracy": self.regime_accuracy,
            "fraud_detection_rate": self.fraud_detection_rate,
            "false_positive_rate": self.false_positive_rate,
        }


@dataclass
class EnhancementResult:
    """Result of an ML enhancement operation."""

    enhancement_type: EnhancementType
    success: bool
    content: str
    structured_data: dict[str, Any] | None = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enhancement_type": self.enhancement_type.value,
            "success": self.success,
            "content": self.content,
            "structured_data": self.structured_data,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ParameterSuggestion:
    """Suggested parameter change."""

    parameter: str
    current_value: Any
    suggested_value: Any
    rationale: str
    expected_impact: str  # "positive", "neutral", "negative"
    confidence: float


@dataclass
class FeatureInsight:
    """Insight about a feature's importance."""

    feature_name: str
    importance_score: float
    current_behavior: str
    recommendation: str
    correlation_with_pnl: float


@dataclass
class AnomalyReport:
    """Report of detected anomaly."""

    anomaly_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_components: list[str]
    root_cause_hypothesis: str
    recommended_actions: list[str]
    confidence: float


# =============================================================================
# ML Enhancement Agent
# =============================================================================


class MLEnhancementAgent:
    """
    Agent that enhances ML trading systems using FinGPT.

    This agent provides natural language explanations, suggestions,
    and analysis for the ML trading infrastructure.
    """

    SYSTEM_PROMPT = """You are an expert ML Trading System Analyst with deep expertise in:
- Quantitative trading strategies and risk management
- Machine learning model interpretation and debugging
- Solana DEX ecosystem (Raydium, Jupiter, Orca, Meteora)
- Market microstructure and liquidity analysis
- Feature engineering for trading models
- Model drift detection and calibration

Your role is to:
1. Explain ML predictions in human-readable terms
2. Suggest parameter optimizations based on performance data
3. Analyze anomalies and provide root cause hypotheses
4. Synthesize learning feedback from trade outcomes
5. Generate market regime narratives
6. Assess portfolio risk with clear narratives
7. Detect and explain model drift
8. Optimize feature importance understanding

Always provide:
- Clear, actionable insights
- Confidence levels for your analysis
- Specific recommendations with rationale
- Risk considerations

Respond in JSON format when structured output is requested."""

    def __init__(
        self,
        fingpt_client: FinGPTClient | None = None,
        model: str = "llama3.2:3b",
        temperature: float = 0.3,  # Lower for more deterministic analysis
        cache_ttl_seconds: float = 60.0,
        max_history: int = 100,
    ):
        self.client = fingpt_client or get_fingpt_client()
        self.temperature = temperature
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_history = max_history

        # Enhancement history for learning
        self._history: list[EnhancementResult] = []
        self._cache: dict[str, tuple[float, Any]] = {}

        # Statistics
        self._total_enhancements = 0
        self._successful_enhancements = 0
        self._total_latency_ms = 0.0
        self._total_tokens = 0

    # =========================================================================
    # Core Enhancement Methods
    # =========================================================================

    async def explain_prediction(
        self,
        prediction: MLPrediction,
        token_data: dict[str, Any] | None = None,
    ) -> EnhancementResult:
        """
        Generate a human-readable explanation of an ML prediction.

        Args:
            prediction: The ML prediction to explain
            token_data: Additional token context (symbol, price, etc.)

        Returns:
            EnhancementResult with explanation
        """
        start_time = time.time()

        token_context = ""
        if token_data:
            token_context = f"""
Token Context:
- Symbol: {token_data.get("symbol", "Unknown")}
- Price: ${token_data.get("price_usd", 0):.6f}
- 24h Volume: ${token_data.get("volume_24h", 0):,.0f}
- Market Cap: ${token_data.get("market_cap", 0):,.0f}
- Liquidity: ${token_data.get("liquidity", 0):,.0f}
"""

        prompt = f"""Analyze and explain this ML trading prediction in human-readable terms.

ML Prediction:
- Score: {prediction.score:.3f} (0.0 = strong sell, 1.0 = strong buy)
- Confidence: {prediction.confidence:.3f}
- Signal: {prediction.signal}
- Fraud Score: {prediction.fraud_score:.3f} (0.0 = safe, 1.0 = high risk)
- Market Regime: {prediction.regime} (confidence: {prediction.regime_confidence:.2f})
{token_context}

Provide a clear explanation in JSON format:
{{
  "summary": "One-sentence summary of the prediction",
  "signal_reasoning": "Why the model generated this signal",
  "confidence_factors": ["factor1", "factor2", ...],
  "risk_factors": ["risk1", "risk2", ...],
  "fraud_analysis": "Analysis of fraud score implications",
  "regime_context": "How current market regime affects this prediction",
  "actionable_recommendation": "Specific action to take",
  "confidence_level": "high/medium/low"
}}"""

        response = await self.client.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.temperature,
            json_mode=True,
        )

        latency_ms = (time.time() - start_time) * 1000

        result = EnhancementResult(
            enhancement_type=EnhancementType.PREDICTION_EXPLANATION,
            success=not bool(response.error) and response.parsed_json is not None,
            content=response.content,
            structured_data=response.parsed_json,
            confidence=(
                response.parsed_json.get("confidence_level", "medium")
                if response.parsed_json
                else 0.0
            ),
            latency_ms=latency_ms,
            tokens_used=response.tokens_generated,
            error=response.error,
        )

        self._record_enhancement(result)
        return result

    async def suggest_parameters(
        self,
        current_params: dict[str, Any],
        metrics: ModelMetrics,
        recent_trades: list[TradeOutcome],
    ) -> EnhancementResult:
        """
        Suggest parameter optimizations based on performance data.

        Args:
            current_params: Current strategy parameters
            metrics: Model performance metrics
            recent_trades: Recent trade outcomes

        Returns:
            EnhancementResult with parameter suggestions
        """
        start_time = time.time()

        # Summarize recent trades
        trade_summary = self._summarize_trades(recent_trades)

        prompt = f"""Analyze current trading parameters and suggest optimizations.

Current Parameters:
{json.dumps(current_params, indent=2)}

Model Performance Metrics:
{json.dumps(metrics.to_dict(), indent=2)}

Recent Trade Summary (last {len(recent_trades)} trades):
{trade_summary}

Based on this data, suggest parameter optimizations in JSON format:
{{
  "overall_assessment": "Brief assessment of current parameter health",
  "suggestions": [
    {{
      "parameter": "parameter_name",
      "current_value": "current",
      "suggested_value": "suggested",
      "rationale": "Why this change would help",
      "expected_impact": "positive/neutral/negative",
      "confidence": 0.0-1.0
    }}
  ],
  "priority_order": ["most_important_param", ...],
  "risk_considerations": ["risk1", ...],
  "should_pause_trading": true/false,
  "pause_reason": "If should_pause_trading, explain why"
}}"""

        response = await self.client.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.temperature,
            json_mode=True,
        )

        latency_ms = (time.time() - start_time) * 1000

        result = EnhancementResult(
            enhancement_type=EnhancementType.PARAMETER_SUGGESTION,
            success=not bool(response.error) and response.parsed_json is not None,
            content=response.content,
            structured_data=response.parsed_json,
            confidence=sum(
                s.get("confidence", 0) for s in response.parsed_json.get("suggestions", [])
            )
            / max(1, len(response.parsed_json.get("suggestions", [])))
            if response.parsed_json
            else 0.0,
            latency_ms=latency_ms,
            tokens_used=response.tokens_generated,
            error=response.error,
        )

        self._record_enhancement(result)
        return result

    async def analyze_anomaly(
        self,
        anomaly_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EnhancementResult:
        """
        Analyze an anomaly and provide root cause hypothesis.

        Args:
            anomaly_data: Detected anomaly details
            context: Additional context (market state, recent events)

        Returns:
            EnhancementResult with anomaly analysis
        """
        start_time = time.time()

        context_str = ""
        if context:
            context_str = f"""
Additional Context:
{json.dumps(context, indent=2)}
"""

        prompt = f"""Analyze this detected anomaly and provide insights.

Anomaly Details:
{json.dumps(anomaly_data, indent=2)}
{context_str}

Provide analysis in JSON format:
{{
  "anomaly_type": "classification of the anomaly",
  "severity": "low/medium/high/critical",
  "description": "Clear description of what happened",
  "affected_components": ["component1", ...],
  "root_cause_hypothesis": "Most likely root cause",
  "contributing_factors": ["factor1", ...],
  "recommended_actions": [
    "action1 with expected outcome",
    ...
  ],
  "prevention_measures": ["measure1", ...],
  "confidence": 0.0-1.0,
  "requires_immediate_attention": true/false
}}"""

        response = await self.client.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.temperature,
            json_mode=True,
        )

        latency_ms = (time.time() - start_time) * 1000

        result = EnhancementResult(
            enhancement_type=EnhancementType.ANOMALY_ANALYSIS,
            success=not bool(response.error) and response.parsed_json is not None,
            content=response.content,
            structured_data=response.parsed_json,
            confidence=response.parsed_json.get("confidence", 0.5) if response.parsed_json else 0.0,
            latency_ms=latency_ms,
            tokens_used=response.tokens_generated,
            error=response.error,
        )

        self._record_enhancement(result)
        return result

    async def synthesize_feedback(
        self,
        trades: list[TradeOutcome],
        metrics: ModelMetrics,
        learning_insights: dict[str, Any] | None = None,
    ) -> EnhancementResult:
        """
        Synthesize learning feedback from trade outcomes.

        Args:
            trades: List of trade outcomes
            metrics: Current model metrics
            learning_insights: Additional learning insights

        Returns:
            EnhancementResult with synthesized feedback
        """
        start_time = time.time()

        # Aggregate trade statistics
        trade_stats = self._compute_trade_statistics(trades)

        insights_str = ""
        if learning_insights:
            insights_str = f"""
Additional Learning Insights:
{json.dumps(learning_insights, indent=2)}
"""

        prompt = f"""Synthesize learning feedback from recent trading activity.

Trade Statistics:
{json.dumps(trade_stats, indent=2)}

Model Metrics:
{json.dumps(metrics.to_dict(), indent=2)}
{insights_str}

Provide synthesized feedback in JSON format:
{{
  "overall_assessment": "Summary of trading performance",
  "what_worked": ["strategy/condition that worked", ...],
  "what_failed": ["strategy/condition that failed", ...],
  "pattern_discoveries": [
    {{
      "pattern": "description of discovered pattern",
      "evidence": "supporting evidence",
      "actionable": true/false
    }}
  ],
  "model_improvements": [
    {{
      "area": "model component to improve",
      "current_issue": "what's wrong",
      "suggested_fix": "how to fix it"
    }}
  ],
  "recommended_experiments": [
    "experiment1: description",
    ...
  ],
  "confidence_adjustment": {{
    "direction": "increase/decrease/maintain",
    "magnitude": 0.0-0.2,
    "rationale": "why"
  }},
  "next_learning_focus": "most important area to focus on",
  "confidence": 0.0-1.0
}}"""

        response = await self.client.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.temperature,
            json_mode=True,
        )

        latency_ms = (time.time() - start_time) * 1000

        result = EnhancementResult(
            enhancement_type=EnhancementType.FEEDBACK_SYNTHESIS,
            success=not bool(response.error) and response.parsed_json is not None,
            content=response.content,
            structured_data=response.parsed_json,
            confidence=response.parsed_json.get("confidence", 0.5) if response.parsed_json else 0.0,
            latency_ms=latency_ms,
            tokens_used=response.tokens_generated,
            error=response.error,
        )

        self._record_enhancement(result)
        return result

    async def generate_regime_narrative(
        self,
        regime: str,
        regime_confidence: float,
        market_data: dict[str, Any],
        transition_probability: float = 0.0,
    ) -> EnhancementResult:
        """
        Generate a narrative explanation of current market regime.

        Args:
            regime: Current detected regime
            regime_confidence: Confidence in regime classification
            market_data: Market data (prices, volumes, volatility)
            transition_probability: Probability of regime change

        Returns:
            EnhancementResult with regime narrative
        """
        start_time = time.time()

        prompt = f"""Generate a narrative explanation of the current market regime.

Regime: {regime}
Regime Confidence: {regime_confidence:.2f}
Transition Probability: {transition_probability:.2f}

Market Data:
{json.dumps(market_data, indent=2)}

Provide regime narrative in JSON format:
{{
  "regime_summary": "One-sentence regime description",
  "characteristics": ["characteristic1", ...],
  "driving_factors": ["what's driving this regime", ...],
  "expected_behavior": "What to expect in this regime",
  "trading_implications": {{
    "optimal_strategies": ["strategy1", ...],
    "avoid_strategies": ["strategy1", ...],
    "position_sizing": "aggressive/moderate/conservative",
    "holding_period": "short/medium/long"
  }},
  "regime_stability": {{
    "stable": true/false,
    "expected_duration": "hours/days/weeks",
    "transition_signals": ["signal to watch for", ...]
  }},
  "risk_assessment": "Current risk level and key concerns",
  "confidence": 0.0-1.0
}}"""

        response = await self.client.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.temperature,
            json_mode=True,
        )

        latency_ms = (time.time() - start_time) * 1000

        result = EnhancementResult(
            enhancement_type=EnhancementType.REGIME_NARRATIVE,
            success=not bool(response.error) and response.parsed_json is not None,
            content=response.content,
            structured_data=response.parsed_json,
            confidence=response.parsed_json.get("confidence", 0.5) if response.parsed_json else 0.0,
            latency_ms=latency_ms,
            tokens_used=response.tokens_generated,
            error=response.error,
        )

        self._record_enhancement(result)
        return result

    async def assess_portfolio_risk(
        self,
        positions: list[dict[str, Any]],
        portfolio_value: float,
        market_conditions: dict[str, Any],
        ml_predictions: list[MLPrediction] | None = None,
    ) -> EnhancementResult:
        """
        Generate a comprehensive portfolio risk assessment.

        Args:
            positions: Current portfolio positions
            portfolio_value: Total portfolio value
            market_conditions: Current market conditions
            ml_predictions: ML predictions for held tokens

        Returns:
            EnhancementResult with risk assessment
        """
        start_time = time.time()

        ml_context = ""
        if ml_predictions:
            ml_context = f"""
ML Predictions for Held Tokens:
{json.dumps([p.to_dict() for p in ml_predictions], indent=2)}
"""

        prompt = f"""Assess portfolio risk and provide actionable recommendations.

Portfolio Value: ${portfolio_value:,.2f}

Current Positions:
{json.dumps(positions, indent=2)}

Market Conditions:
{json.dumps(market_conditions, indent=2)}
{ml_context}

Provide risk assessment in JSON format:
{{
  "overall_risk_level": "low/medium/high/critical",
  "risk_score": 0-100,
  "risk_breakdown": {{
    "concentration_risk": {{
      "score": 0-100,
      "description": "Analysis of position concentration"
    }},
    "liquidity_risk": {{
      "score": 0-100,
      "description": "Analysis of liquidity exposure"
    }},
    "volatility_risk": {{
      "score": 0-100,
      "description": "Analysis of volatility exposure"
    }},
    "correlation_risk": {{
      "score": 0-100,
      "description": "Analysis of position correlation"
    }}
  }},
  "position_recommendations": [
    {{
      "symbol": "TOKEN",
      "action": "hold/reduce/increase/exit",
      "rationale": "Why",
      "urgency": "immediate/soon/monitor"
    }}
  ],
  "hedging_suggestions": ["suggestion1", ...],
  "stress_test_results": {{
    "best_case": "+X%",
    "expected": "+Y%",
    "worst_case": "-Z%"
  }},
  "key_warnings": ["warning1", ...],
  "confidence": 0.0-1.0
}}"""

        response = await self.client.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.temperature,
            json_mode=True,
        )

        latency_ms = (time.time() - start_time) * 1000

        result = EnhancementResult(
            enhancement_type=EnhancementType.RISK_ASSESSMENT,
            success=not bool(response.error) and response.parsed_json is not None,
            content=response.content,
            structured_data=response.parsed_json,
            confidence=response.parsed_json.get("confidence", 0.5) if response.parsed_json else 0.0,
            latency_ms=latency_ms,
            tokens_used=response.tokens_generated,
            error=response.error,
        )

        self._record_enhancement(result)
        return result

    async def detect_model_drift(
        self,
        baseline_metrics: ModelMetrics,
        current_metrics: ModelMetrics,
        feature_distribution_shift: dict[str, float] | None = None,
        prediction_distribution: dict[str, Any] | None = None,
    ) -> EnhancementResult:
        """
        Detect and analyze model drift.

        Args:
            baseline_metrics: Baseline model performance
            current_metrics: Current model performance
            feature_distribution_shift: Feature-level drift scores
            prediction_distribution: Distribution of recent predictions

        Returns:
            EnhancementResult with drift analysis
        """
        start_time = time.time()

        feature_drift_str = ""
        if feature_distribution_shift:
            feature_drift_str = f"""
Feature Distribution Shift (higher = more drift):
{json.dumps(feature_distribution_shift, indent=2)}
"""

        pred_dist_str = ""
        if prediction_distribution:
            pred_dist_str = f"""
Prediction Distribution:
{json.dumps(prediction_distribution, indent=2)}
"""

        prompt = f"""Analyze model drift between baseline and current performance.

Baseline Metrics:
{json.dumps(baseline_metrics.to_dict(), indent=2)}

Current Metrics:
{json.dumps(current_metrics.to_dict(), indent=2)}
{feature_drift_str}
{pred_dist_str}

Provide drift analysis in JSON format:
{{
  "drift_detected": true/false,
  "drift_severity": "none/minor/moderate/severe/critical",
  "drift_types": ["performance", "feature", "prediction", ...],
  "performance_drift": {{
    "win_rate_change": "+/-X%",
    "accuracy_change": "+/-X%",
    "calibration_change": "+/-X%",
    "interpretation": "What this means"
  }},
  "feature_drift_analysis": {{
    "most_affected_features": ["feature1", ...],
    "potential_causes": ["cause1", ...]
  }},
  "prediction_drift_analysis": {{
    "distribution_shift": "description",
    "bias_direction": "overconfident/underconfident/balanced"
  }},
  "recommended_actions": [
    {{
      "action": "retrain/recalibrate/adjust_thresholds",
      "rationale": "why",
      "urgency": "immediate/soon/scheduled",
      "expected_impact": "description"
    }}
  ],
  "should_trigger_retraining": true/false,
  "confidence": 0.0-1.0
}}"""

        response = await self.client.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.temperature,
            json_mode=True,
        )

        latency_ms = (time.time() - start_time) * 1000

        result = EnhancementResult(
            enhancement_type=EnhancementType.MODEL_DRIFT,
            success=not bool(response.error) and response.parsed_json is not None,
            content=response.content,
            structured_data=response.parsed_json,
            confidence=response.parsed_json.get("confidence", 0.5) if response.parsed_json else 0.0,
            latency_ms=latency_ms,
            tokens_used=response.tokens_generated,
            error=response.error,
        )

        self._record_enhancement(result)
        return result

    async def optimize_feature_importance(
        self,
        feature_importance: dict[str, float],
        performance_by_feature: dict[str, dict[str, float]] | None = None,
        correlation_matrix: dict[str, dict[str, float]] | None = None,
    ) -> EnhancementResult:
        """
        Analyze feature importance and suggest optimizations.

        Args:
            feature_importance: Current feature importance scores
            performance_by_feature: Performance metrics per feature
            correlation_matrix: Feature correlation matrix

        Returns:
            EnhancementResult with feature optimization suggestions
        """
        start_time = time.time()

        perf_str = ""
        if performance_by_feature:
            perf_str = f"""
Performance by Feature:
{json.dumps(performance_by_feature, indent=2)}
"""

        corr_str = ""
        if correlation_matrix:
            # Summarize correlations to avoid huge prompts
            high_corr = []
            for f1, correlations in correlation_matrix.items():
                for f2, corr in correlations.items():
                    if f1 < f2 and abs(corr) > 0.7:  # Avoid duplicates and low correlations
                        high_corr.append({"features": [f1, f2], "correlation": corr})
            if high_corr:
                corr_str = f"""
High Correlations (|r| > 0.7):
{json.dumps(high_corr[:10], indent=2)}  # Top 10
"""

        prompt = f"""Analyze feature importance and suggest optimizations.

Feature Importance Scores:
{json.dumps(feature_importance, indent=2)}
{perf_str}
{corr_str}

Provide feature analysis in JSON format:
{{
  "top_predictive_features": [
    {{
      "feature": "feature_name",
      "importance": 0.0-1.0,
      "interpretation": "Why this feature matters"
    }}
  ],
  "underperforming_features": [
    {{
      "feature": "feature_name",
      "issue": "What's wrong",
      "recommendation": "remove/transform/keep"
    }}
  ],
  "redundant_features": [
    {{
      "features": ["f1", "f2"],
      "correlation": 0.0-1.0,
      "recommendation": "which to keep"
    }}
  ],
  "missing_features": [
    {{
      "feature": "suggested_new_feature",
      "rationale": "Why it would help",
      "expected_impact": "high/medium/low"
    }}
  ],
  "feature_engineering_suggestions": [
    {{
      "transformation": "description",
      "target_features": ["f1", ...],
      "expected_benefit": "description"
    }}
  ],
  "confidence": 0.0-1.0
}}"""

        response = await self.client.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.temperature,
            json_mode=True,
        )

        latency_ms = (time.time() - start_time) * 1000

        result = EnhancementResult(
            enhancement_type=EnhancementType.FEATURE_IMPORTANCE,
            success=not bool(response.error) and response.parsed_json is not None,
            content=response.content,
            structured_data=response.parsed_json,
            confidence=response.parsed_json.get("confidence", 0.5) if response.parsed_json else 0.0,
            latency_ms=latency_ms,
            tokens_used=response.tokens_generated,
            error=response.error,
        )

        self._record_enhancement(result)
        return result

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def batch_explain_predictions(
        self,
        predictions: list[tuple[MLPrediction, dict[str, Any]]],
        max_concurrent: int = 5,
    ) -> list[EnhancementResult]:
        """
        Explain multiple predictions concurrently.

        Args:
            predictions: List of (prediction, token_data) tuples
            max_concurrent: Maximum concurrent explanations

        Returns:
            List of EnhancementResults
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def explain_with_limit(pred: MLPrediction, data: dict[str, Any]) -> EnhancementResult:
            async with semaphore:
                return await self.explain_prediction(pred, data)

        tasks = [explain_with_limit(pred, data) for pred, data in predictions]
        return await asyncio.gather(*tasks)

    # =========================================================================
    # Continuous Learning Integration
    # =========================================================================

    async def continuous_improvement_cycle(
        self,
        trades: list[TradeOutcome],
        metrics: ModelMetrics,
        current_params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run a complete continuous improvement cycle.

        This combines multiple enhancement types into a comprehensive
        analysis that can be used to improve the trading system.

        Args:
            trades: Recent trade outcomes
            metrics: Current model metrics
            current_params: Current strategy parameters

        Returns:
            Comprehensive improvement report
        """
        start_time = time.time()

        # Run all enhancements in parallel where possible
        feedback_task = self.synthesize_feedback(trades, metrics)
        params_task = self.suggest_parameters(current_params, metrics, trades)

        # Wait for results
        feedback_result, params_result = await asyncio.gather(feedback_task, params_task)

        cycle_time_ms = (time.time() - start_time) * 1000

        return {
            "cycle_timestamp": datetime.utcnow().isoformat(),
            "cycle_duration_ms": cycle_time_ms,
            "feedback_synthesis": feedback_result.to_dict(),
            "parameter_suggestions": params_result.to_dict(),
            "summary": self._generate_cycle_summary(feedback_result, params_result),
            "recommended_actions": self._prioritize_actions(feedback_result, params_result),
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _summarize_trades(self, trades: list[TradeOutcome]) -> str:
        """Generate a summary of recent trades."""
        if not trades:
            return "No recent trades"

        wins = [t for t in trades if t.success]
        losses = [t for t in trades if not t.success]

        avg_pnl = np.mean([t.pnl_pct for t in trades if t.pnl_pct is not None] or [0])
        avg_hold = np.mean([t.hold_time_seconds for t in trades])

        return f"""
Total Trades: {len(trades)}
Wins: {len(wins)} ({len(wins) / len(trades) * 100:.1f}%)
Losses: {len(losses)} ({len(losses) / len(trades) * 100:.1f}%)
Avg PnL: {avg_pnl * 100:.2f}%
Avg Hold Time: {avg_hold / 60:.1f} minutes
"""

    def _compute_trade_statistics(self, trades: list[TradeOutcome]) -> dict[str, Any]:
        """Compute detailed trade statistics."""
        if not trades:
            return {"total": 0}

        pnls = [t.pnl_pct for t in trades if t.pnl_pct is not None]

        # By signal type
        by_signal = {}
        for t in trades:
            if t.ml_signal not in by_signal:
                by_signal[t.ml_signal] = {"wins": 0, "losses": 0, "total": 0}
            by_signal[t.ml_signal]["total"] += 1
            if t.success:
                by_signal[t.ml_signal]["wins"] += 1
            else:
                by_signal[t.ml_signal]["losses"] += 1

        # By regime
        by_regime = {}
        for t in trades:
            if t.market_regime not in by_regime:
                by_regime[t.market_regime] = {"wins": 0, "losses": 0, "total": 0}
            by_regime[t.market_regime]["total"] += 1
            if t.success:
                by_regime[t.market_regime]["wins"] += 1
            else:
                by_regime[t.market_regime]["losses"] += 1

        # Confidence bins
        confidence_bins = {
            "high (0.8+)": {"wins": 0, "total": 0},
            "medium (0.5-0.8)": {"wins": 0, "total": 0},
            "low (<0.5)": {"wins": 0, "total": 0},
        }
        for t in trades:
            if t.ml_confidence >= 0.8:
                bin_key = "high (0.8+)"
            elif t.ml_confidence >= 0.5:
                bin_key = "medium (0.5-0.8)"
            else:
                bin_key = "low (<0.5)"
            confidence_bins[bin_key]["total"] += 1
            if t.success:
                confidence_bins[bin_key]["wins"] += 1

        return {
            "total": len(trades),
            "win_rate": len([t for t in trades if t.success]) / len(trades),
            "avg_pnl_pct": float(np.mean(pnls)) if pnls else 0.0,
            "median_pnl_pct": float(np.median(pnls)) if pnls else 0.0,
            "std_pnl_pct": float(np.std(pnls)) if pnls else 0.0,
            "by_signal": by_signal,
            "by_regime": by_regime,
            "by_confidence": confidence_bins,
        }

    def _generate_cycle_summary(
        self,
        feedback: EnhancementResult,
        params: EnhancementResult,
    ) -> str:
        """Generate a summary of the improvement cycle."""
        parts = []

        if feedback.success and feedback.structured_data:
            parts.append(f"Overall: {feedback.structured_data.get('overall_assessment', 'N/A')}")

        if params.success and params.structured_data:
            parts.append(f"Parameters: {params.structured_data.get('overall_assessment', 'N/A')}")

        return " | ".join(parts) if parts else "Cycle completed with errors"

    def _prioritize_actions(
        self,
        feedback: EnhancementResult,
        params: EnhancementResult,
    ) -> list[dict[str, Any]]:
        """Prioritize recommended actions."""
        actions = []

        # Check if pause is recommended
        if params.success and params.structured_data:
            if params.structured_data.get("should_pause_trading"):
                actions.append(
                    {
                        "priority": 1,
                        "action": "pause_trading",
                        "reason": params.structured_data.get(
                            "pause_reason", "Recommended by ML agent"
                        ),
                    }
                )

        # Add parameter suggestions
        if params.success and params.structured_data:
            for i, suggestion in enumerate(params.structured_data.get("suggestions", [])):
                actions.append(
                    {
                        "priority": 2 + i,
                        "action": f"adjust_{suggestion.get('parameter', 'param')}",
                        "current": suggestion.get("current_value"),
                        "suggested": suggestion.get("suggested_value"),
                        "rationale": suggestion.get("rationale"),
                    }
                )

        # Add feedback recommendations
        if feedback.success and feedback.structured_data:
            for improvement in feedback.structured_data.get("model_improvements", []):
                actions.append(
                    {
                        "priority": 10 + len(actions),
                        "action": "model_improvement",
                        "area": improvement.get("area"),
                        "fix": improvement.get("suggested_fix"),
                    }
                )

        return sorted(actions, key=lambda x: x["priority"])

    def _record_enhancement(self, result: EnhancementResult):
        """Record enhancement in history and update stats."""
        self._history.append(result)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

        self._total_enhancements += 1
        if result.success:
            self._successful_enhancements += 1
        self._total_latency_ms += result.latency_ms
        self._total_tokens += result.tokens_used

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        by_type = {}
        for result in self._history:
            t = result.enhancement_type.value
            if t not in by_type:
                by_type[t] = {"count": 0, "success": 0, "avg_latency_ms": 0.0}
            by_type[t]["count"] += 1
            if result.success:
                by_type[t]["success"] += 1
            by_type[t]["avg_latency_ms"] = (
                by_type[t]["avg_latency_ms"] * (by_type[t]["count"] - 1) + result.latency_ms
            ) / by_type[t]["count"]

        return {
            "total_enhancements": self._total_enhancements,
            "successful_enhancements": self._successful_enhancements,
            "success_rate": self._successful_enhancements / max(1, self._total_enhancements),
            "avg_latency_ms": self._total_latency_ms / max(1, self._total_enhancements),
            "total_tokens": self._total_tokens,
            "by_type": by_type,
        }

    async def close(self):
        """Close the underlying client."""
        await self.client.close()


# =============================================================================
# Singleton and Convenience Functions
# =============================================================================


_ml_enhancement_agent: MLEnhancementAgent | None = None


def get_ml_enhancement_agent() -> MLEnhancementAgent:
    """Get or create the singleton ML Enhancement Agent."""
    global _ml_enhancement_agent
    if _ml_enhancement_agent is None:
        _ml_enhancement_agent = MLEnhancementAgent()
    return _ml_enhancement_agent


async def explain_prediction(
    prediction: MLPrediction,
    token_data: dict[str, Any] | None = None,
) -> EnhancementResult:
    """Convenience function to explain a prediction."""
    agent = get_ml_enhancement_agent()
    return await agent.explain_prediction(prediction, token_data)


async def suggest_improvements(
    current_params: dict[str, Any],
    metrics: ModelMetrics,
    recent_trades: list[TradeOutcome],
) -> EnhancementResult:
    """Convenience function to get parameter suggestions."""
    agent = get_ml_enhancement_agent()
    return await agent.suggest_parameters(current_params, metrics, recent_trades)
