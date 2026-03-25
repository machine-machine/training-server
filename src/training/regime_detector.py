"""
Market Regime Detection module.

Classifies current market conditions to enable adaptive strategy parameters.
Different regimes require different trading approaches - aggressive during
meme seasons, defensive during bear markets, cautious during high MEV periods.

Regimes:
- MEME_SEASON: High volume, many new tokens, momentum working
- NORMAL: Average conditions
- BEAR_VOLATILE: Downtrend with high volatility
- HIGH_MEV: Elevated sandwich attack activity
- LOW_LIQUIDITY: Thin orderbooks, high slippage risk
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classifications."""
    MEME_SEASON = "MEME_SEASON"
    NORMAL = "NORMAL"
    BEAR_VOLATILE = "BEAR_VOLATILE"
    HIGH_MEV = "HIGH_MEV"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"


@dataclass
class MarketSnapshot:
    """Point-in-time market data for regime classification."""
    timestamp_ms: int

    # Volume metrics
    total_volume_24h_sol: float = 0.0
    avg_volume_7d_sol: float = 0.0
    volume_change_pct: float = 0.0  # vs 7d average

    # New token metrics
    new_tokens_24h: int = 0
    avg_new_tokens_7d: float = 0.0
    new_token_survival_rate: float = 0.0  # % surviving > 1 hour

    # Price momentum
    sol_price_usd: float = 0.0
    sol_price_change_24h_pct: float = 0.0
    sol_price_change_7d_pct: float = 0.0
    btc_dominance_pct: float = 0.0

    # Volatility metrics
    volatility_1h: float = 0.0  # Standard deviation of returns
    volatility_24h: float = 0.0
    avg_volatility_7d: float = 0.0

    # MEV metrics
    sandwich_attacks_1h: int = 0
    sandwich_attacks_24h: int = 0
    avg_sandwich_attacks_7d: float = 0.0
    jito_bundle_rate: float = 0.0  # % of txns using Jito

    # Liquidity metrics
    avg_pool_liquidity_sol: float = 0.0
    pools_below_10k_usd_pct: float = 0.0
    avg_slippage_500_sol_bps: int = 0  # Slippage for 500 SOL trade

    # Sentiment (optional - from external APIs)
    fear_greed_index: Optional[int] = None  # 0-100
    social_volume_change_pct: float = 0.0


@dataclass
class RegimeClassification:
    """Result of regime classification."""
    regime: MarketRegime
    confidence: float  # 0-1 confidence in classification
    timestamp_ms: int
    contributing_factors: List[str]  # Human-readable factors
    raw_scores: Dict[str, float]  # Score per regime
    recommended_adjustments: Dict[str, Any]  # Strategy adjustments


@dataclass
class RegimeHistory:
    """Track regime changes over time."""
    classifications: List[RegimeClassification] = field(default_factory=list)
    regime_durations: Dict[MarketRegime, int] = field(default_factory=dict)  # ms in each regime

    def add(self, classification: RegimeClassification):
        """Add a classification to history."""
        self.classifications.append(classification)
        # Keep only last 1000 classifications
        if len(self.classifications) > 1000:
            self.classifications = self.classifications[-1000:]

    def current_regime(self) -> Optional[MarketRegime]:
        """Get current regime."""
        if self.classifications:
            return self.classifications[-1].regime
        return None

    def regime_changed_recently(self, threshold_minutes: int = 60) -> bool:
        """Check if regime changed in the last N minutes."""
        if len(self.classifications) < 2:
            return False

        current = self.classifications[-1]
        threshold_ms = threshold_minutes * 60 * 1000

        for i in range(len(self.classifications) - 2, -1, -1):
            prev = self.classifications[i]
            age_ms = current.timestamp_ms - prev.timestamp_ms
            if age_ms > threshold_ms:
                return False
            if prev.regime != current.regime:
                return True

        return False

    def time_in_current_regime_ms(self) -> int:
        """Get time spent in current regime."""
        if not self.classifications:
            return 0

        current_regime = self.classifications[-1].regime
        total_ms = 0

        for i in range(len(self.classifications) - 1, 0, -1):
            if self.classifications[i].regime == current_regime:
                prev = self.classifications[i - 1]
                curr = self.classifications[i]
                total_ms += curr.timestamp_ms - prev.timestamp_ms
            else:
                break

        return total_ms


class RegimeDetector:
    """
    Classify market conditions into regimes for adaptive trading.

    Uses multiple signals:
    - Volume and new token activity
    - Price momentum and volatility
    - MEV activity levels
    - Liquidity conditions
    """

    # Thresholds for regime detection
    THRESHOLDS = {
        # MEME_SEASON thresholds
        "meme_volume_surge": 2.0,  # Volume > 2x 7d average
        "meme_new_tokens": 1.5,  # New tokens > 1.5x 7d average
        "meme_survival_rate": 0.3,  # > 30% tokens survive 1 hour

        # BEAR_VOLATILE thresholds
        "bear_sol_drop_7d": -15.0,  # SOL down > 15% in 7 days
        "bear_volatility_surge": 1.5,  # Volatility > 1.5x 7d average

        # HIGH_MEV thresholds
        "mev_sandwich_surge": 2.0,  # Sandwich attacks > 2x 7d average
        "mev_jito_rate": 0.6,  # > 60% using Jito

        # LOW_LIQUIDITY thresholds
        "low_liq_pools_pct": 0.5,  # > 50% pools below 10k USD
        "low_liq_slippage_bps": 300,  # > 300 bps slippage for 500 SOL
    }

    # Strategy adjustments per regime
    REGIME_ADJUSTMENTS = {
        MarketRegime.MEME_SEASON: {
            "min_confidence": 0.5,
            "take_profit_pct": 35.0,
            "stop_loss_pct": 12.0,
            "max_position_sol": 0.15,
            "max_hold_minutes": 60,
            "use_jito": False,  # Less critical
            "rationale": "Momentum is working - wider stops, larger positions for winners",
        },
        MarketRegime.NORMAL: {
            "min_confidence": 0.6,
            "take_profit_pct": 20.0,
            "stop_loss_pct": 8.0,
            "max_position_sol": 0.1,
            "max_hold_minutes": 30,
            "use_jito": True,
            "rationale": "Standard conditions - balanced risk/reward",
        },
        MarketRegime.BEAR_VOLATILE: {
            "min_confidence": 0.75,
            "take_profit_pct": 12.0,
            "stop_loss_pct": 5.0,
            "max_position_sol": 0.05,
            "max_hold_minutes": 15,
            "use_jito": True,
            "rationale": "Defensive mode - preserve capital, quick exits",
        },
        MarketRegime.HIGH_MEV: {
            "min_confidence": 0.65,
            "take_profit_pct": 18.0,
            "stop_loss_pct": 7.0,
            "max_position_sol": 0.08,
            "max_hold_minutes": 20,
            "use_jito": True,  # Critical!
            "slippage_bps": 400,  # Higher tolerance
            "rationale": "High MEV - must use Jito, smaller positions to reduce target size",
        },
        MarketRegime.LOW_LIQUIDITY: {
            "min_confidence": 0.7,
            "take_profit_pct": 15.0,
            "stop_loss_pct": 6.0,
            "max_position_sol": 0.03,
            "max_hold_minutes": 20,
            "use_jito": True,
            "min_liquidity_usd": 20000,  # Higher requirement
            "slippage_bps": 500,  # Higher tolerance
            "rationale": "Thin liquidity - very small positions, exit early",
        },
    }

    def __init__(self):
        self.history = RegimeHistory()
        self._last_snapshot: Optional[MarketSnapshot] = None

    def classify(self, snapshot: MarketSnapshot) -> RegimeClassification:
        """
        Classify current market conditions into a regime.

        Args:
            snapshot: Current market data snapshot

        Returns:
            RegimeClassification with regime, confidence, and adjustments
        """
        self._last_snapshot = snapshot

        # Calculate score for each regime
        scores = {
            MarketRegime.MEME_SEASON: self._score_meme_season(snapshot),
            MarketRegime.BEAR_VOLATILE: self._score_bear_volatile(snapshot),
            MarketRegime.HIGH_MEV: self._score_high_mev(snapshot),
            MarketRegime.LOW_LIQUIDITY: self._score_low_liquidity(snapshot),
        }

        # NORMAL is default - score is inverse of other regimes
        max_other = max(scores.values())
        scores[MarketRegime.NORMAL] = max(0, 1.0 - max_other)

        # Find regime with highest score
        regime = max(scores, key=lambda r: scores[r])
        confidence = scores[regime]

        # Gather contributing factors
        factors = self._get_contributing_factors(snapshot, regime)

        # Get recommended adjustments
        adjustments = self.REGIME_ADJUSTMENTS[regime].copy()

        classification = RegimeClassification(
            regime=regime,
            confidence=confidence,
            timestamp_ms=snapshot.timestamp_ms,
            contributing_factors=factors,
            raw_scores={r.value: s for r, s in scores.items()},
            recommended_adjustments=adjustments,
        )

        self.history.add(classification)

        logger.info(
            f"Regime classified: {regime.value} (confidence: {confidence:.2%})"
        )
        for factor in factors:
            logger.debug(f"  - {factor}")

        return classification

    def _score_meme_season(self, s: MarketSnapshot) -> float:
        """Score likelihood of MEME_SEASON regime."""
        score = 0.0
        weights = {"volume": 0.35, "new_tokens": 0.35, "survival": 0.15, "momentum": 0.15}

        # Volume surge
        if s.avg_volume_7d_sol > 0:
            volume_ratio = s.total_volume_24h_sol / s.avg_volume_7d_sol
            if volume_ratio >= self.THRESHOLDS["meme_volume_surge"]:
                score += weights["volume"]
            elif volume_ratio >= 1.5:
                score += weights["volume"] * 0.5

        # New token surge
        if s.avg_new_tokens_7d > 0:
            token_ratio = s.new_tokens_24h / s.avg_new_tokens_7d
            if token_ratio >= self.THRESHOLDS["meme_new_tokens"]:
                score += weights["new_tokens"]
            elif token_ratio >= 1.2:
                score += weights["new_tokens"] * 0.5

        # Token survival rate
        if s.new_token_survival_rate >= self.THRESHOLDS["meme_survival_rate"]:
            score += weights["survival"]
        elif s.new_token_survival_rate >= 0.2:
            score += weights["survival"] * 0.5

        # Positive momentum
        if s.sol_price_change_7d_pct > 5:
            score += weights["momentum"]
        elif s.sol_price_change_7d_pct > 0:
            score += weights["momentum"] * 0.5

        return min(1.0, score)

    def _score_bear_volatile(self, s: MarketSnapshot) -> float:
        """Score likelihood of BEAR_VOLATILE regime."""
        score = 0.0
        weights = {"price_drop": 0.4, "volatility": 0.4, "btc_dominance": 0.2}

        # SOL price dropping
        if s.sol_price_change_7d_pct <= self.THRESHOLDS["bear_sol_drop_7d"]:
            score += weights["price_drop"]
        elif s.sol_price_change_7d_pct <= -10:
            score += weights["price_drop"] * 0.7
        elif s.sol_price_change_7d_pct <= -5:
            score += weights["price_drop"] * 0.4

        # Elevated volatility
        if s.avg_volatility_7d > 0:
            vol_ratio = s.volatility_24h / s.avg_volatility_7d
            if vol_ratio >= self.THRESHOLDS["bear_volatility_surge"]:
                score += weights["volatility"]
            elif vol_ratio >= 1.2:
                score += weights["volatility"] * 0.5

        # BTC dominance rising (altcoin weakness)
        if s.btc_dominance_pct > 55:
            score += weights["btc_dominance"]
        elif s.btc_dominance_pct > 50:
            score += weights["btc_dominance"] * 0.5

        return min(1.0, score)

    def _score_high_mev(self, s: MarketSnapshot) -> float:
        """Score likelihood of HIGH_MEV regime."""
        score = 0.0
        weights = {"sandwich": 0.5, "jito_rate": 0.3, "short_term": 0.2}

        # Sandwich attack surge
        if s.avg_sandwich_attacks_7d > 0:
            sandwich_ratio = s.sandwich_attacks_24h / s.avg_sandwich_attacks_7d
            if sandwich_ratio >= self.THRESHOLDS["mev_sandwich_surge"]:
                score += weights["sandwich"]
            elif sandwich_ratio >= 1.5:
                score += weights["sandwich"] * 0.6

        # High Jito usage (indicates MEV awareness/protection needed)
        if s.jito_bundle_rate >= self.THRESHOLDS["mev_jito_rate"]:
            score += weights["jito_rate"]
        elif s.jito_bundle_rate >= 0.4:
            score += weights["jito_rate"] * 0.5

        # Short-term sandwich activity
        if s.avg_sandwich_attacks_7d > 0:
            hourly_rate = s.sandwich_attacks_1h * 24  # Annualize to daily
            if hourly_rate > s.avg_sandwich_attacks_7d * 2:
                score += weights["short_term"]

        return min(1.0, score)

    def _score_low_liquidity(self, s: MarketSnapshot) -> float:
        """Score likelihood of LOW_LIQUIDITY regime."""
        score = 0.0
        weights = {"pools": 0.4, "slippage": 0.4, "avg_liq": 0.2}

        # Many low-liquidity pools
        if s.pools_below_10k_usd_pct >= self.THRESHOLDS["low_liq_pools_pct"]:
            score += weights["pools"]
        elif s.pools_below_10k_usd_pct >= 0.35:
            score += weights["pools"] * 0.5

        # High slippage
        if s.avg_slippage_500_sol_bps >= self.THRESHOLDS["low_liq_slippage_bps"]:
            score += weights["slippage"]
        elif s.avg_slippage_500_sol_bps >= 200:
            score += weights["slippage"] * 0.5

        # Low average pool liquidity
        if s.avg_pool_liquidity_sol < 50:  # < 50 SOL average
            score += weights["avg_liq"]
        elif s.avg_pool_liquidity_sol < 100:
            score += weights["avg_liq"] * 0.5

        return min(1.0, score)

    def _get_contributing_factors(
        self,
        s: MarketSnapshot,
        regime: MarketRegime,
    ) -> List[str]:
        """Get human-readable factors contributing to classification."""
        factors = []

        if regime == MarketRegime.MEME_SEASON:
            if s.avg_volume_7d_sol > 0:
                ratio = s.total_volume_24h_sol / s.avg_volume_7d_sol
                factors.append(f"Volume {ratio:.1f}x 7d average")
            if s.avg_new_tokens_7d > 0:
                ratio = s.new_tokens_24h / s.avg_new_tokens_7d
                factors.append(f"New tokens {ratio:.1f}x 7d average")
            if s.new_token_survival_rate > 0:
                factors.append(f"Token survival rate: {s.new_token_survival_rate:.0%}")

        elif regime == MarketRegime.BEAR_VOLATILE:
            if s.sol_price_change_7d_pct < 0:
                factors.append(f"SOL {s.sol_price_change_7d_pct:+.1f}% (7d)")
            if s.avg_volatility_7d > 0:
                ratio = s.volatility_24h / s.avg_volatility_7d
                factors.append(f"Volatility {ratio:.1f}x 7d average")
            if s.btc_dominance_pct > 50:
                factors.append(f"BTC dominance: {s.btc_dominance_pct:.1f}%")

        elif regime == MarketRegime.HIGH_MEV:
            factors.append(f"Sandwich attacks (24h): {s.sandwich_attacks_24h}")
            factors.append(f"Jito bundle rate: {s.jito_bundle_rate:.0%}")

        elif regime == MarketRegime.LOW_LIQUIDITY:
            factors.append(f"Low-liq pools: {s.pools_below_10k_usd_pct:.0%}")
            factors.append(f"Avg slippage: {s.avg_slippage_500_sol_bps} bps")

        elif regime == MarketRegime.NORMAL:
            factors.append("No significant regime signals detected")

        return factors

    def get_current_regime(self) -> Optional[MarketRegime]:
        """Get the current market regime."""
        return self.history.current_regime()

    def get_recommended_adjustments(self) -> Dict[str, Any]:
        """Get strategy adjustments for current regime."""
        regime = self.get_current_regime()
        if regime:
            return self.REGIME_ADJUSTMENTS[regime].copy()
        return self.REGIME_ADJUSTMENTS[MarketRegime.NORMAL].copy()

    def regime_summary(self) -> str:
        """Get human-readable summary of current regime."""
        if not self.history.classifications:
            return "No regime classification yet"

        latest = self.history.classifications[-1]
        duration_ms = self.history.time_in_current_regime_ms()
        duration_hours = duration_ms / (3600 * 1000)

        lines = [
            f"Current Regime: {latest.regime.value}",
            f"Confidence: {latest.confidence:.0%}",
            f"Duration: {duration_hours:.1f} hours",
            "",
            "Contributing Factors:",
        ]
        for factor in latest.contributing_factors:
            lines.append(f"  - {factor}")

        lines.append("")
        lines.append("Recommended Adjustments:")
        for key, value in latest.recommended_adjustments.items():
            if key != "rationale":
                lines.append(f"  {key}: {value}")
        lines.append("")
        lines.append(f"Rationale: {latest.recommended_adjustments.get('rationale', 'N/A')}")

        return "\n".join(lines)


# Convenience function for quick regime check
def detect_regime(
    volume_24h_sol: float,
    avg_volume_7d_sol: float,
    new_tokens_24h: int,
    avg_new_tokens_7d: float,
    sol_price_change_7d_pct: float,
    volatility_24h: float,
    avg_volatility_7d: float,
    sandwich_attacks_24h: int = 0,
    avg_sandwich_attacks_7d: float = 0,
    jito_bundle_rate: float = 0.3,
    pools_below_10k_pct: float = 0.2,
    avg_slippage_bps: int = 100,
) -> Tuple[MarketRegime, float]:
    """
    Quick regime detection with minimal inputs.

    Returns:
        Tuple of (regime, confidence)
    """
    import time

    snapshot = MarketSnapshot(
        timestamp_ms=int(time.time() * 1000),
        total_volume_24h_sol=volume_24h_sol,
        avg_volume_7d_sol=avg_volume_7d_sol,
        new_tokens_24h=new_tokens_24h,
        avg_new_tokens_7d=avg_new_tokens_7d,
        sol_price_change_7d_pct=sol_price_change_7d_pct,
        volatility_24h=volatility_24h,
        avg_volatility_7d=avg_volatility_7d,
        sandwich_attacks_24h=sandwich_attacks_24h,
        avg_sandwich_attacks_7d=avg_sandwich_attacks_7d,
        jito_bundle_rate=jito_bundle_rate,
        pools_below_10k_usd_pct=pools_below_10k_pct,
        avg_slippage_500_sol_bps=avg_slippage_bps,
    )

    detector = RegimeDetector()
    result = detector.classify(snapshot)
    return result.regime, result.confidence
