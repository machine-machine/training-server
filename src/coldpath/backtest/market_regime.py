"""
Market Regime Detector for dynamic parameter adjustment.

Detects current market conditions and recommends parameter adjustments
based on volatility, trend, liquidity, and risk sentiment.

Key Features:
- Real-time regime classification
- Multi-factor analysis (volatility, trend, volume, sentiment)
- Parameter adjustment recommendations
- Historical regime tracking
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime types."""

    TRENDING_UP = "trending_up"  # Strong upward momentum
    TRENDING_DOWN = "trending_down"  # Strong downward momentum
    SIDEWAYS = "sideways"  # Range-bound, low direction
    HIGH_VOLATILITY = "high_volatility"  # Large price swings
    LOW_VOLATILITY = "low_volatility"  # Compressed price action
    MEMECOIN_SEASON = "memecoin_season"  # High risk-on, speculative
    RISK_OFF = "risk_off"  # Flight to safety
    TRANSITIONAL = "transitional"  # Between regimes


class VolatilityLevel(Enum):
    """Volatility classification."""

    VERY_LOW = "very_low"  # < 2%
    LOW = "low"  # 2-5%
    MODERATE = "moderate"  # 5-10%
    HIGH = "high"  # 10-20%
    EXTREME = "extreme"  # > 20%


class TrendStrength(Enum):
    """Trend strength classification."""

    STRONG_UP = "strong_up"
    MODERATE_UP = "moderate_up"
    WEAK_UP = "weak_up"
    NEUTRAL = "neutral"
    WEAK_DOWN = "weak_down"
    MODERATE_DOWN = "moderate_down"
    STRONG_DOWN = "strong_down"


@dataclass
class MarketMetrics:
    """Current market metrics snapshot."""

    timestamp: str

    # Price metrics
    current_price: float
    price_change_1h: float = 0.0
    price_change_24h: float = 0.0
    price_change_7d: float = 0.0

    # Volatility
    volatility_1h: float = 0.0
    volatility_24h: float = 0.0
    atr_pct: float = 0.0  # Average True Range %

    # Volume
    volume_24h: float = 0.0
    volume_change_24h: float = 0.0
    volume_profile: str = "normal"  # low, normal, high, extreme

    # Liquidity
    total_liquidity_usd: float = 0.0
    liquidity_change_24h: float = 0.0

    # Market breadth
    advancing_tokens: int = 0
    declining_tokens: int = 0
    new_launches_24h: int = 0

    # Risk indicators
    avg_risk_score: float = 0.5
    rug_count_24h: int = 0

    # Sentiment (if available)
    fear_greed_index: float = 50.0
    social_sentiment: float = 0.0


@dataclass
class RegimeAdjustment:
    """Recommended parameter adjustment for a regime."""

    parameter: str
    adjustment_factor: float
    reason: str
    confidence: float


@dataclass
class RegimeAnalysis:
    """Full regime analysis result."""

    primary_regime: RegimeType
    secondary_regime: RegimeType | None
    volatility_level: VolatilityLevel
    trend_strength: TrendStrength

    confidence: float
    stability_score: float  # How stable is the regime (0-1)

    adjustments: list[RegimeAdjustment]
    risk_level: str  # low, moderate, high, extreme
    recommended_actions: list[str]

    metrics: MarketMetrics
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_regime": self.primary_regime.value,
            "secondary_regime": self.secondary_regime.value if self.secondary_regime else None,
            "volatility_level": self.volatility_level.value,
            "trend_strength": self.trend_strength.value,
            "confidence": self.confidence,
            "stability_score": self.stability_score,
            "adjustments": [
                {
                    "parameter": a.parameter,
                    "factor": a.adjustment_factor,
                    "reason": a.reason,
                    "confidence": a.confidence,
                }
                for a in self.adjustments
            ],
            "risk_level": self.risk_level,
            "recommended_actions": self.recommended_actions,
            "timestamp": self.timestamp,
        }


# Regime-specific parameter adjustments
REGIME_ADJUSTMENTS = {
    RegimeType.TRENDING_UP: {
        "take_profit_pct": (1.2, "Increase profit targets in uptrend"),
        "max_position_sol": (1.15, "Can size up in favorable conditions"),
        "max_hold_minutes": (1.3, "Hold longer to capture trend"),
        "stop_loss_pct": (1.1, "Slightly wider stops for trend noise"),
    },
    RegimeType.TRENDING_DOWN: {
        "stop_loss_pct": (0.75, "Tighter stops in downtrend"),
        "take_profit_pct": (0.7, "Lower targets in adverse conditions"),
        "max_position_sol": (0.7, "Reduce position size"),
        "min_liquidity_usd": (1.5, "Require higher liquidity"),
        "max_risk_score": (0.7, "More selective entry"),
    },
    RegimeType.HIGH_VOLATILITY: {
        "stop_loss_pct": (1.4, "Wider stops for volatility"),
        "take_profit_pct": (1.3, "Higher targets possible"),
        "slippage_bps": (1.4, "More slippage tolerance"),
        "max_position_sol": (0.75, "Smaller positions"),
        "max_risk_score": (0.8, "More risk-averse"),
    },
    RegimeType.LOW_VOLATILITY: {
        "take_profit_pct": (0.8, "Lower targets in low vol"),
        "max_hold_minutes": (0.7, "Shorter holds"),
        "slippage_bps": (0.8, "Less slippage expected"),
    },
    RegimeType.MEMECOIN_SEASON: {
        "take_profit_pct": (1.5, "Target larger gains"),
        "max_risk_score": (1.25, "Accept more risk"),
        "min_liquidity_usd": (0.6, "Lower liquidity OK"),
        "slippage_bps": (1.3, "More slippage tolerance"),
    },
    RegimeType.RISK_OFF: {
        "stop_loss_pct": (0.6, "Very tight stops"),
        "max_position_sol": (0.5, "Much smaller positions"),
        "min_liquidity_usd": (2.0, "Only high quality"),
        "max_risk_score": (0.5, "Very selective"),
        "take_profit_pct": (0.7, "Lower targets"),
    },
    RegimeType.SIDEWAYS: {
        "take_profit_pct": (0.85, "Lower targets in range"),
        "stop_loss_pct": (0.9, "Slightly tighter stops"),
        "max_hold_minutes": (0.8, "Shorter holds"),
    },
    RegimeType.TRANSITIONAL: {
        "max_position_sol": (0.8, "Reduce size during uncertainty"),
        "stop_loss_pct": (0.85, "Tight stops in transition"),
    },
}


class MarketRegimeDetector:
    """Detects market regimes and recommends parameter adjustments.

    Uses multi-factor analysis to classify market conditions:
    - Price momentum and trend
    - Volatility levels
    - Volume patterns
    - Risk sentiment indicators

    Usage:
        detector = MarketRegimeDetector()

        # Analyze current market
        analysis = detector.analyze(market_metrics)

        # Get adjusted parameters
        adjusted = detector.adjust_parameters(base_params, analysis)

        # Or use convenience method
        adjusted = detector.get_regime_adjusted_params(
            base_params,
            market_metrics,
        )
    """

    def __init__(
        self,
        volatility_thresholds: dict[str, float] | None = None,
        lookback_periods: int = 24,
    ):
        """Initialize the regime detector.

        Args:
            volatility_thresholds: Custom volatility thresholds
            lookback_periods: Hours to look back for trend analysis
        """
        self.volatility_thresholds = volatility_thresholds or {
            "very_low": 2.0,
            "low": 5.0,
            "moderate": 10.0,
            "high": 20.0,
        }
        self.lookback_periods = lookback_periods

        # Track regime history for stability analysis
        self._regime_history: list[tuple[datetime, RegimeType]] = []
        self._max_history = 100

    def analyze(
        self,
        metrics: MarketMetrics,
        include_history: bool = True,
    ) -> RegimeAnalysis:
        """Analyze market metrics to determine current regime.

        Args:
            metrics: Current market metrics
            include_history: Whether to use historical context

        Returns:
            RegimeAnalysis with regime classification and adjustments
        """
        # Determine volatility level
        vol_level = self._classify_volatility(metrics)

        # Determine trend strength
        trend = self._classify_trend(metrics)

        # Determine primary regime
        primary_regime = self._determine_regime(metrics, vol_level, trend)

        # Determine secondary regime (if any)
        secondary_regime = self._determine_secondary_regime(
            metrics, vol_level, trend, primary_regime
        )

        # Calculate confidence
        confidence = self._calculate_confidence(metrics, primary_regime)

        # Calculate stability
        stability = self._calculate_stability(primary_regime) if include_history else 0.5

        # Generate adjustments
        adjustments = self._generate_adjustments(primary_regime, secondary_regime)

        # Determine risk level
        risk_level = self._assess_risk_level(metrics, primary_regime)

        # Generate recommended actions
        actions = self._generate_actions(primary_regime, metrics)

        # Update history
        if include_history:
            self._regime_history.append((datetime.utcnow(), primary_regime))
            if len(self._regime_history) > self._max_history:
                self._regime_history.pop(0)

        return RegimeAnalysis(
            primary_regime=primary_regime,
            secondary_regime=secondary_regime,
            volatility_level=vol_level,
            trend_strength=trend,
            confidence=confidence,
            stability_score=stability,
            adjustments=adjustments,
            risk_level=risk_level,
            recommended_actions=actions,
            metrics=metrics,
        )

    def adjust_parameters(
        self,
        base_params: dict[str, Any],
        analysis: RegimeAnalysis,
    ) -> dict[str, Any]:
        """Apply regime adjustments to base parameters.

        Args:
            base_params: Original parameters
            analysis: Regime analysis result

        Returns:
            Adjusted parameters
        """
        adjusted = base_params.copy()

        for adj in analysis.adjustments:
            if adj.parameter in adjusted:
                original = adjusted[adj.parameter]
                if isinstance(original, (int, float)):
                    adjusted[adj.parameter] = original * adj.adjustment_factor

        return adjusted

    def get_regime_adjusted_params(
        self,
        base_params: dict[str, Any],
        metrics: MarketMetrics,
    ) -> tuple[dict[str, Any], RegimeAnalysis]:
        """Convenience method to get regime-adjusted parameters.

        Args:
            base_params: Original parameters
            metrics: Current market metrics

        Returns:
            Tuple of (adjusted_params, analysis)
        """
        analysis = self.analyze(metrics)
        adjusted = self.adjust_parameters(base_params, analysis)
        return adjusted, analysis

    def get_current_regime(self) -> RegimeType | None:
        """Get the most recent detected regime."""
        if self._regime_history:
            return self._regime_history[-1][1]
        return None

    def get_regime_distribution(self, hours: int = 24) -> dict[RegimeType, float]:
        """Get distribution of regimes over recent history.

        Args:
            hours: Number of hours to analyze

        Returns:
            Dict mapping regime to percentage of time
        """
        if not self._regime_history:
            return {}

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = [(t, r) for t, r in self._regime_history if t > cutoff]

        if not recent:
            return {}

        counts: dict[RegimeType, int] = {}
        for _, regime in recent:
            counts[regime] = counts.get(regime, 0) + 1

        total = len(recent)
        return {r: c / total for r, c in counts.items()}

    def _classify_volatility(self, metrics: MarketMetrics) -> VolatilityLevel:
        """Classify current volatility level."""
        vol = max(metrics.volatility_1h, metrics.volatility_24h, metrics.atr_pct)

        if vol < self.volatility_thresholds["very_low"]:
            return VolatilityLevel.VERY_LOW
        elif vol < self.volatility_thresholds["low"]:
            return VolatilityLevel.LOW
        elif vol < self.volatility_thresholds["moderate"]:
            return VolatilityLevel.MODERATE
        elif vol < self.volatility_thresholds["high"]:
            return VolatilityLevel.HIGH
        else:
            return VolatilityLevel.EXTREME

    def _classify_trend(self, metrics: MarketMetrics) -> TrendStrength:
        """Classify current trend strength."""
        # Combine short and medium term changes
        short_term = metrics.price_change_1h
        medium_term = metrics.price_change_24h

        # Weight towards medium term
        combined = short_term * 0.3 + medium_term * 0.7

        if combined > 15:
            return TrendStrength.STRONG_UP
        elif combined > 8:
            return TrendStrength.MODERATE_UP
        elif combined > 3:
            return TrendStrength.WEAK_UP
        elif combined < -15:
            return TrendStrength.STRONG_DOWN
        elif combined < -8:
            return TrendStrength.MODERATE_DOWN
        elif combined < -3:
            return TrendStrength.WEAK_DOWN
        else:
            return TrendStrength.NEUTRAL

    def _determine_regime(
        self,
        metrics: MarketMetrics,
        vol_level: VolatilityLevel,
        trend: TrendStrength,
    ) -> RegimeType:
        """Determine the primary market regime."""
        # Check for extreme conditions first

        # Risk-off detection
        if (
            metrics.fear_greed_index < 25
            or metrics.rug_count_24h > 10
            or (metrics.declining_tokens > metrics.advancing_tokens * 3)
        ):
            return RegimeType.RISK_OFF

        # Memecoin season detection
        if (
            metrics.new_launches_24h > 50
            or metrics.avg_risk_score > 0.6
            or metrics.social_sentiment > 0.7
        ):
            return RegimeType.MEMECOIN_SEASON

        # High volatility regime
        if vol_level in [VolatilityLevel.HIGH, VolatilityLevel.EXTREME]:
            # Can still have trend within high vol
            if trend in [TrendStrength.STRONG_UP, TrendStrength.MODERATE_UP]:
                return RegimeType.TRENDING_UP
            elif trend in [TrendStrength.STRONG_DOWN, TrendStrength.MODERATE_DOWN]:
                return RegimeType.TRENDING_DOWN
            return RegimeType.HIGH_VOLATILITY

        # Low volatility regime
        if vol_level in [VolatilityLevel.VERY_LOW, VolatilityLevel.LOW]:
            return RegimeType.LOW_VOLATILITY

        # Trend-based classification for moderate volatility
        if trend in [TrendStrength.STRONG_UP, TrendStrength.MODERATE_UP]:
            return RegimeType.TRENDING_UP
        elif trend in [TrendStrength.STRONG_DOWN, TrendStrength.MODERATE_DOWN]:
            return RegimeType.TRENDING_DOWN
        elif trend == TrendStrength.NEUTRAL:
            return RegimeType.SIDEWAYS

        # Default to transitional
        return RegimeType.TRANSITIONAL

    def _determine_secondary_regime(
        self,
        metrics: MarketMetrics,
        vol_level: VolatilityLevel,
        trend: TrendStrength,
        primary: RegimeType,
    ) -> RegimeType | None:
        """Determine secondary regime (if market shows multiple characteristics)."""
        # High vol + trend = secondary high vol
        if primary in [RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN]:
            if vol_level in [VolatilityLevel.HIGH, VolatilityLevel.EXTREME]:
                return RegimeType.HIGH_VOLATILITY

        # Memecoin season can coexist with other regimes
        if primary != RegimeType.MEMECOIN_SEASON:
            if metrics.new_launches_24h > 30:
                return RegimeType.MEMECOIN_SEASON

        return None

    def _calculate_confidence(
        self,
        metrics: MarketMetrics,
        regime: RegimeType,
    ) -> float:
        """Calculate confidence in regime classification."""
        base_confidence = 0.7

        # Higher confidence with more extreme values
        if metrics.volatility_24h > 20 or metrics.volatility_24h < 2:
            base_confidence += 0.1

        if abs(metrics.price_change_24h) > 15:
            base_confidence += 0.1

        # Lower confidence if metrics are sparse
        if metrics.volume_24h == 0:
            base_confidence -= 0.2

        return min(1.0, max(0.3, base_confidence))

    def _calculate_stability(self, current_regime: RegimeType) -> float:
        """Calculate regime stability based on history."""
        if len(self._regime_history) < 5:
            return 0.5  # Not enough history

        recent = [r for _, r in self._regime_history[-10:]]

        # Count how many times regime has changed
        changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])

        # Fewer changes = more stable
        stability = 1.0 - (changes / max(1, len(recent) - 1))

        # Bonus if current regime has been consistent
        same_count = sum(1 for r in recent[-5:] if r == current_regime)
        stability = (stability + same_count / 5) / 2

        return round(stability, 2)

    def _generate_adjustments(
        self,
        primary: RegimeType,
        secondary: RegimeType | None,
    ) -> list[RegimeAdjustment]:
        """Generate parameter adjustments for the detected regime."""
        adjustments = []

        # Get primary regime adjustments
        if primary in REGIME_ADJUSTMENTS:
            for param, (factor, reason) in REGIME_ADJUSTMENTS[primary].items():
                adjustments.append(
                    RegimeAdjustment(
                        parameter=param,
                        adjustment_factor=factor,
                        reason=reason,
                        confidence=0.8,
                    )
                )

        # Blend with secondary regime if present
        if secondary and secondary in REGIME_ADJUSTMENTS:
            secondary_adjustments = REGIME_ADJUSTMENTS[secondary]
            for param, (factor, reason) in secondary_adjustments.items():
                # Check if already in adjustments
                existing = next((a for a in adjustments if a.parameter == param), None)
                if existing:
                    # Blend the factors (geometric mean)
                    blended = math.sqrt(existing.adjustment_factor * factor)
                    existing.adjustment_factor = blended
                    existing.confidence *= 0.9
                else:
                    adjustments.append(
                        RegimeAdjustment(
                            parameter=param,
                            adjustment_factor=factor * 0.7,  # Reduced for secondary
                            reason=f"[Secondary] {reason}",
                            confidence=0.6,
                        )
                    )

        return adjustments

    def _assess_risk_level(
        self,
        metrics: MarketMetrics,
        regime: RegimeType,
    ) -> str:
        """Assess overall risk level."""
        risk_scores = {
            RegimeType.RISK_OFF: 4,
            RegimeType.HIGH_VOLATILITY: 4,
            RegimeType.TRANSITIONAL: 3,
            RegimeType.MEMECOIN_SEASON: 3,
            RegimeType.TRENDING_DOWN: 3,
            RegimeType.SIDEWAYS: 2,
            RegimeType.LOW_VOLATILITY: 1,
            RegimeType.TRENDING_UP: 2,
        }

        base_risk = risk_scores.get(regime, 2)

        # Adjust for specific metrics
        if metrics.rug_count_24h > 5:
            base_risk += 1
        if metrics.avg_risk_score > 0.6:
            base_risk += 1
        if metrics.volatility_24h > 25:
            base_risk += 1

        if base_risk >= 4:
            return "extreme"
        elif base_risk >= 3:
            return "high"
        elif base_risk >= 2:
            return "moderate"
        else:
            return "low"

    def _generate_actions(
        self,
        regime: RegimeType,
        metrics: MarketMetrics,
    ) -> list[str]:
        """Generate recommended actions based on regime."""
        actions = []

        if regime == RegimeType.RISK_OFF:
            actions.append("Consider reducing exposure significantly")
            actions.append("Raise liquidity requirements")
            actions.append("Tighten stop losses")

        elif regime == RegimeType.HIGH_VOLATILITY:
            actions.append("Reduce position sizes")
            actions.append("Widen stop losses to avoid whipsaws")
            actions.append("Be patient with entries")

        elif regime == RegimeType.MEMECOIN_SEASON:
            actions.append("Higher risk tolerance may be appropriate")
            actions.append("Target larger gains but respect stops")
            actions.append("Watch for sudden sentiment shifts")

        elif regime == RegimeType.TRENDING_UP:
            actions.append("Favor long positions")
            actions.append("Can extend hold times")
            actions.append("Trail stops to lock in gains")

        elif regime == RegimeType.TRENDING_DOWN:
            actions.append("Be selective with entries")
            actions.append("Quick profit taking")
            actions.append("Consider waiting for reversal")

        elif regime == RegimeType.SIDEWAYS:
            actions.append("Range trading strategies may work")
            actions.append("Quick in-and-out trades")
            actions.append("Avoid over-leveraging")

        if metrics.new_launches_24h > 20:
            actions.append(
                f"High new launch activity ({metrics.new_launches_24h} in 24h) - extra caution"
            )

        return actions


def create_metrics_from_market_data(
    price_data: dict[str, Any],
    volume_data: dict[str, Any] | None = None,
    sentiment_data: dict[str, Any] | None = None,
) -> MarketMetrics:
    """Create MarketMetrics from raw market data.

    Args:
        price_data: Dict with price, changes, volatility
        volume_data: Optional dict with volume info
        sentiment_data: Optional dict with sentiment info

    Returns:
        MarketMetrics instance
    """
    metrics = MarketMetrics(
        timestamp=datetime.utcnow().isoformat(),
        current_price=price_data.get("price", 0),
        price_change_1h=price_data.get("change_1h", 0),
        price_change_24h=price_data.get("change_24h", 0),
        price_change_7d=price_data.get("change_7d", 0),
        volatility_1h=price_data.get("volatility_1h", 0),
        volatility_24h=price_data.get("volatility_24h", 0),
        atr_pct=price_data.get("atr_pct", 0),
    )

    if volume_data:
        metrics.volume_24h = volume_data.get("volume_24h", 0)
        metrics.volume_change_24h = volume_data.get("volume_change_24h", 0)
        metrics.volume_profile = volume_data.get("volume_profile", "normal")
        metrics.total_liquidity_usd = volume_data.get("liquidity_usd", 0)

    if sentiment_data:
        metrics.fear_greed_index = sentiment_data.get("fear_greed", 50)
        metrics.social_sentiment = sentiment_data.get("social", 0)
        metrics.avg_risk_score = sentiment_data.get("avg_risk", 0.5)
        metrics.new_launches_24h = sentiment_data.get("new_launches", 0)
        metrics.rug_count_24h = sentiment_data.get("rug_count", 0)
        metrics.advancing_tokens = sentiment_data.get("advancing", 0)
        metrics.declining_tokens = sentiment_data.get("declining", 0)

    return metrics
