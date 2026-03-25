#
# mtf_confirmation.py
# 2DEXY ColdPath
#
# Multi-Timeframe Signal Confirmation
#
# Only trade when multiple timeframes align.
# Avoid trading against higher-timeframe trends.
#

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Trend(Enum):
    """Trend direction."""

    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0

    def to_signal(self) -> int:
        """Convert to signal value (-1, 0, 1)."""
        return self.value


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe."""

    timeframe: str  # "1m", "5m", "15m", "1h", "4h", "1d"
    trend: Trend
    momentum: float  # -1 to 1
    strength: float  # 0 to 1 (how strong the trend is)
    support_level: float | None = None
    resistance_level: float | None = None
    price_position: str = "middle"  # "above_resistance", "below_support", "middle"
    is_valid: bool = True


@dataclass
class Signal:
    """A trading signal."""

    is_buy: bool
    confidence: float  # 0 to 1
    mint: str
    timestamp_ms: int
    price: float | None = None
    size_sol: float | None = None


@dataclass
class ConfirmationResult:
    """Result of multi-timeframe confirmation."""

    is_confirmed: bool
    confidence_multiplier: float
    alignment_score: float  # -1 to 1 (negative = against trend)
    trend_alignment: dict[str, Trend]  # Per-timeframe trend
    conflicting_timeframes: list[str]
    recommendation: str
    risk_level: str  # "low", "medium", "high"


class MultiTimeframeAnalyzer:
    """
    Confirm signals across multiple timeframes.

    Only trade when trend alignment is favorable.
    The higher timeframes have more weight.

    Timeframe weights:
        1m: 0.10 (least important)
        5m: 0.15
        15m: 0.20
        1h: 0.25
        4h: 0.20
        1d: 0.10 (context only)

    Alignment scoring:
        - Signal direction matches trend: +weight
        - Signal against trend: -weight * 1.5 (penalty)
        - Neutral trend: 0

    Confirmation threshold: alignment_score >= 0.4
    """

    TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

    # Timeframe weights (must sum to 1.0)
    WEIGHTS: dict[str, float] = {
        "1m": 0.10,
        "5m": 0.15,
        "15m": 0.20,
        "1h": 0.25,
        "4h": 0.20,
        "1d": 0.10,
    }

    # Minimum data points needed per timeframe
    MIN_PERIODS: dict[str, int] = {
        "1m": 20,  # 20 minutes
        "5m": 24,  # 2 hours
        "15m": 32,  # 8 hours
        "1h": 24,  # 1 day
        "4h": 18,  # 3 days
        "1d": 7,  # 1 week
    }

    def __init__(
        self,
        confirmation_threshold: float = 0.4,
        penalty_multiplier: float = 1.5,
    ):
        self.confirmation_threshold = confirmation_threshold
        self.penalty_multiplier = penalty_multiplier

        # Cache of analyses per token
        self._analyses: dict[str, dict[str, TimeframeAnalysis]] = {}

    def analyze_timeframe(
        self,
        prices: list[float],
        volumes: list[float] | None = None,
        timeframe: str = "15m",
    ) -> TimeframeAnalysis:
        """
        Analyze a single timeframe.

        Uses simple trend detection:
        1. Moving average crossover
        2. Price momentum
        3. Support/resistance levels
        """
        if len(prices) < 5:
            return TimeframeAnalysis(
                timeframe=timeframe,
                trend=Trend.NEUTRAL,
                momentum=0.0,
                strength=0.0,
                is_valid=False,
            )

        n = len(prices)

        # Calculate simple moving averages
        short_period = min(5, n // 4)
        long_period = min(20, n // 2)

        short_ma = sum(prices[-short_period:]) / short_period
        long_ma = sum(prices[-long_period:]) / long_period

        # Determine trend
        if short_ma > long_ma * 1.01:  # 1% buffer
            trend = Trend.BULLISH
        elif short_ma < long_ma * 0.99:
            trend = Trend.BEARISH
        else:
            trend = Trend.NEUTRAL

        # Calculate momentum (rate of change)
        if len(prices) >= 2 and prices[-2] > 0:
            momentum = (prices[-1] - prices[-2]) / prices[-2]
            momentum = max(-1.0, min(1.0, momentum * 10))  # Scale and clamp
        else:
            momentum = 0.0

        # Calculate trend strength (how consistent is the direction)
        if n >= 10:
            changes = [
                1 if prices[i] > prices[i - 1] else (-1 if prices[i] < prices[i - 1] else 0)
                for i in range(1, n)
            ]
            # How many changes agree with trend?
            trend_val = trend.to_signal()
            if trend_val != 0:
                agreement = sum(1 for c in changes[-10:] if c == trend_val) / 10
                strength = agreement
            else:
                # No clear trend = low strength
                strength = sum(abs(c) for c in changes[-10:]) / 10
        else:
            strength = 0.5

        # Find support and resistance
        if n >= 20:
            recent = prices[-20:]
            support = min(recent)
            resistance = max(recent)
        else:
            support = min(prices)
            resistance = max(prices)

        # Price position
        current_price = prices[-1]
        if resistance > support:
            resistance - support
            if current_price >= resistance:
                price_position = "above_resistance"
            elif current_price <= support:
                price_position = "below_support"
            else:
                price_position = "middle"
        else:
            price_position = "middle"

        return TimeframeAnalysis(
            timeframe=timeframe,
            trend=trend,
            momentum=momentum,
            strength=strength,
            support_level=support,
            resistance_level=resistance,
            price_position=price_position,
            is_valid=True,
        )

    def analyze_all_timeframes(
        self,
        price_data: dict[str, list[float]],  # timeframe -> prices
        volume_data: dict[str, list[float]] | None = None,
        mint: str | None = None,
    ) -> dict[str, TimeframeAnalysis]:
        """
        Analyze all timeframes for a token.

        Args:
            price_data: Dict mapping timeframe to list of prices
            volume_data: Optional volume data per timeframe
            mint: Token identifier (for caching)
        """
        results = {}

        for tf in self.TIMEFRAMES:
            if tf in price_data and len(price_data[tf]) >= 5:
                prices = price_data[tf]
                volumes = volume_data.get(tf) if volume_data else None
                results[tf] = self.analyze_timeframe(prices, volumes, tf)
            else:
                results[tf] = TimeframeAnalysis(
                    timeframe=tf,
                    trend=Trend.NEUTRAL,
                    momentum=0.0,
                    strength=0.0,
                    is_valid=False,
                )

        # Cache if mint provided
        if mint:
            self._analyses[mint] = results

        return results

    def confirm_signal(
        self,
        signal: Signal,
        analyses: dict[str, TimeframeAnalysis] | None = None,
        mint: str | None = None,
    ) -> ConfirmationResult:
        """
        Check if signal is confirmed across timeframes.

        Args:
            signal: The trading signal to confirm
            analyses: Pre-computed analyses (or use cached)
            mint: Token to look up cached analyses

        Returns:
            ConfirmationResult with confirmation decision and confidence adjustment
        """
        # Get analyses
        if analyses is None and mint:
            analyses = self._analyses.get(mint, {})
        elif analyses is None:
            analyses = {}

        if not analyses:
            # No analyses available - neutral confirmation
            return ConfirmationResult(
                is_confirmed=True,  # Allow trade but no confidence boost
                confidence_multiplier=1.0,
                alignment_score=0.0,
                trend_alignment={},
                conflicting_timeframes=[],
                recommendation="No timeframe data - trade with caution",
                risk_level="high",
            )

        # Calculate alignment score
        alignment_score = 0.0
        trend_alignment = {}
        conflicting = []

        for tf, analysis in analyses.items():
            if not analysis.is_valid:
                continue

            weight = self.WEIGHTS.get(tf, 0.1)
            trend = analysis.trend
            trend_alignment[tf] = trend

            # Signal direction matches trend?
            if signal.is_buy:
                if trend == Trend.BULLISH:
                    alignment_score += weight * analysis.strength
                elif trend == Trend.BEARISH:
                    alignment_score -= weight * self.penalty_multiplier * analysis.strength
                    conflicting.append(tf)
            else:  # Sell signal
                if trend == Trend.BEARISH:
                    alignment_score += weight * analysis.strength
                elif trend == Trend.BULLISH:
                    alignment_score -= weight * self.penalty_multiplier * analysis.strength
                    conflicting.append(tf)

        # Determine confirmation
        is_confirmed = alignment_score >= self.confirmation_threshold

        # Calculate confidence multiplier
        # alignment_score ranges from about -0.6 to +0.6
        # Map to multiplier from 0.5 to 1.5
        confidence_mult = 0.5 + (alignment_score + 0.6) / 1.2  # 0.5 + 0 to 1
        confidence_mult = max(0.5, min(1.5, confidence_mult))

        # Determine risk level
        if alignment_score >= 0.5:
            risk_level = "low"
            recommendation = "Strong trend alignment - safe to trade"
        elif alignment_score >= 0.3:
            risk_level = "medium"
            recommendation = "Moderate alignment - trade with standard caution"
        elif alignment_score >= 0.0:
            risk_level = "medium"
            recommendation = "Weak alignment - consider reducing position size"
        else:
            risk_level = "high"
            recommendation = "Trading against trend - high risk, consider skipping"

        if conflicting:
            recommendation += f" (conflicts in: {', '.join(conflicting)})"

        return ConfirmationResult(
            is_confirmed=is_confirmed,
            confidence_multiplier=confidence_mult,
            alignment_score=alignment_score,
            trend_alignment=trend_alignment,
            conflicting_timeframes=conflicting,
            recommendation=recommendation,
            risk_level=risk_level,
        )

    def get_adjusted_confidence(
        self,
        base_confidence: float,
        signal: Signal,
        mint: str,
    ) -> float:
        """
        Get MTF-adjusted confidence for a signal.

        Use this to adjust signal confidence before making trading decisions.
        """
        result = self.confirm_signal(signal, mint=mint)

        # Apply confidence multiplier
        adjusted = base_confidence * result.confidence_multiplier

        # Apply alignment adjustment
        if result.alignment_score < 0:
            # Trading against trend - reduce confidence more
            adjusted *= 0.8

        return max(0.0, min(1.0, adjusted))

    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "tokens_analyzed": len(self._analyses),
            "confirmation_threshold": self.confirmation_threshold,
            "timeframe_weights": self.WEIGHTS,
        }

    def clear_cache(self, mint: str | None = None) -> None:
        """Clear analysis cache."""
        if mint:
            self._analyses.pop(mint, None)
        else:
            self._analyses.clear()


# Singleton instance
_analyzer: MultiTimeframeAnalyzer | None = None


def get_mtf_analyzer() -> MultiTimeframeAnalyzer:
    """Get the global MTF analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = MultiTimeframeAnalyzer()
    return _analyzer
