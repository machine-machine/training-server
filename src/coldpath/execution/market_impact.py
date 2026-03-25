"""
Market Impact Prediction Model

Implements square-root market impact model (Almgren-Chriss style) for
predicting and minimizing execution costs.

Impact = sigma * sqrt(Q/V) * participation_rate

Where:
- sigma = volatility
- Q = trade size
- V = daily volume

Target: +5-10bps fill quality improvement
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ImpactResult:
    """Result of market impact prediction."""

    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float
    recommended_delay_seconds: float
    optimal_split_count: int
    confidence: float


@dataclass
class ImpactConfig:
    """Configuration for impact model."""

    temp_impact_coeff: float = 0.5
    perm_impact_coeff: float = 0.1
    max_participation: float = 0.1
    impact_threshold_bps: float = 50.0
    min_delay_between_splits: float = 1.0


class SquareRootImpactModel:
    """
    Square-root market impact model (Almgren-Chriss style).

    Predicts both temporary and permanent market impact for trade execution,
    and recommends optimal execution strategy.

    Usage:
        model = SquareRootImpactModel()
        result = model.predict_impact(
            trade_size_usd=1000.0,
            daily_volume_usd=50000.0,
            volatility=0.45,
        )
        if result.total_impact_bps > 50:
            # Split execution over time
            for i in range(result.optimal_split_count):
                execute_partial(...)
                time.sleep(result.recommended_delay_seconds)
    """

    def __init__(self, config: ImpactConfig | None = None):
        self.config = config or ImpactConfig()

        # Historical impact data for calibration (bounded)
        self._impact_history: deque = deque(maxlen=1000)

    def predict_impact(
        self,
        trade_size_usd: float,
        daily_volume_usd: float,
        volatility: float,
    ) -> ImpactResult:
        """
        Predict market impact in basis points.

        Args:
            trade_size_usd: Size of trade in USD
            daily_volume_usd: Daily volume of token in USD
            volatility: Realized volatility (0-1 scale, e.g., 0.45 = 45%)

        Returns:
            ImpactResult with impact predictions and execution recommendations
        """
        # Handle edge cases
        if daily_volume_usd <= 0:
            daily_volume_usd = 100_000  # Default assumption for unknown volume

        if volatility <= 0:
            volatility = 0.3  # Default assumption

        if trade_size_usd <= 0:
            return ImpactResult(
                temporary_impact_bps=0.0,
                permanent_impact_bps=0.0,
                total_impact_bps=0.0,
                recommended_delay_seconds=0.0,
                optimal_split_count=1,
                confidence=1.0,
            )

        # Calculate participation rate (capped at max)
        participation = min(trade_size_usd / daily_volume_usd, self.config.max_participation)
        sqrt_participation = math.sqrt(participation)

        # Temporary impact (recovers after trade)
        temp_impact = self.config.temp_impact_coeff * volatility * sqrt_participation

        # Permanent impact (price moves permanently)
        perm_impact = self.config.perm_impact_coeff * volatility * participation

        # Total in basis points
        total_bps = (temp_impact + perm_impact) * 10_000

        # Calculate optimal execution strategy
        optimal_splits, recommended_delay = self._calculate_optimal_execution(
            total_bps=total_bps,
            trade_size_usd=trade_size_usd,
            daily_volume_usd=daily_volume_usd,
        )

        # Confidence based on data quality
        confidence = self._calculate_confidence(
            daily_volume_usd=daily_volume_usd,
            volatility=volatility,
        )

        return ImpactResult(
            temporary_impact_bps=temp_impact * 10_000,
            permanent_impact_bps=perm_impact * 10_000,
            total_impact_bps=total_bps,
            recommended_delay_seconds=recommended_delay,
            optimal_split_count=optimal_splits,
            confidence=confidence,
        )

    def _calculate_optimal_execution(
        self,
        total_bps: float,
        trade_size_usd: float,
        daily_volume_usd: float,
    ) -> tuple[int, float]:
        """
        Calculate optimal number of splits and delay between them.

        Returns:
            (optimal_split_count, recommended_delay_seconds)
        """
        if total_bps <= self.config.impact_threshold_bps:
            # Low impact, no need to split
            return 1, 0.0

        # Split to keep each execution under threshold
        target_bps_per_split = self.config.impact_threshold_bps * 0.8
        splits_needed = math.ceil(total_bps / target_bps_per_split)

        # Cap at reasonable maximum
        optimal_splits = min(splits_needed, 10)

        # Delay between splits (more splits = longer delay)
        base_delay = self.config.min_delay_between_splits
        recommended_delay = base_delay * (1 + 0.5 * (optimal_splits - 1))

        return optimal_splits, recommended_delay

    def _calculate_confidence(
        self,
        daily_volume_usd: float,
        volatility: float,
    ) -> float:
        """
        Calculate confidence in the impact prediction.

        Higher confidence for:
        - Higher volume (more liquidity, more predictable)
        - Moderate volatility (extreme vol = less predictable)
        """
        # Volume confidence (higher = better)
        vol_confidence = min(1.0, daily_volume_usd / 100_000)

        # Volatility confidence (moderate = better)
        if volatility < 0.1:
            vol_conf = 0.7  # Too low, might be stale data
        elif volatility > 0.8:
            vol_conf = 0.6  # Too high, unpredictable
        else:
            vol_conf = 1.0  # Sweet spot

        # Historical calibration bonus
        history_bonus = min(0.2, len(self._impact_history) * 0.01)

        confidence = (vol_confidence * 0.4 + vol_conf * 0.4 + 0.2) + history_bonus
        return min(1.0, confidence)

    def record_actual_impact(
        self,
        predicted_impact_bps: float,
        actual_impact_bps: float,
        trade_size_usd: float,
        volatility: float,
    ) -> None:
        """
        Record actual impact for model calibration.

        Call this after trade execution to improve future predictions.
        """
        self._impact_history.append(
            (
                predicted_impact_bps,
                actual_impact_bps,
                trade_size_usd,
                volatility,
            )
        )

        # deque(maxlen=1000) handles eviction automatically

        # Recalibrate coefficients periodically
        if len(self._impact_history) % 100 == 0:
            self._recalibrate()

    def _recalibrate(self) -> None:
        """Recalibrate model coefficients based on historical data."""
        if len(self._impact_history) < 50:
            return

        # Simple linear regression to adjust temp coefficient
        predicted = np.array([h[0] for h in self._impact_history])
        actual = np.array([h[1] for h in self._impact_history])

        # Calculate bias (guard NaN from empty/corrupt data)
        bias = np.mean(actual - predicted)
        if not np.isfinite(bias):
            return

        # Adjust coefficient if significant bias
        if abs(bias) > 5:  # More than 5bps bias
            adjustment = 1 + (bias / 100) * 0.1  # 10% adjustment per 100bps bias
            new_coeff = self.config.temp_impact_coeff * adjustment

            # Clamp to reasonable range
            self.config.temp_impact_coeff = float(max(0.1, min(1.0, new_coeff)))

    def should_delay_execution(
        self,
        trade_size_usd: float,
        daily_volume_usd: float,
        volatility: float,
        max_acceptable_impact_bps: float = 50.0,
    ) -> tuple[bool, ImpactResult]:
        """
        Check if execution should be delayed or split.

        Args:
            trade_size_usd: Size of trade in USD
            daily_volume_usd: Daily volume of token in USD
            volatility: Realized volatility
            max_acceptable_impact_bps: Maximum impact threshold

        Returns:
            (should_delay, impact_result)
        """
        result = self.predict_impact(
            trade_size_usd=trade_size_usd,
            daily_volume_usd=daily_volume_usd,
            volatility=volatility,
        )

        should_delay = result.total_impact_bps > max_acceptable_impact_bps
        return should_delay, result

    def get_recommended_slippage(
        self,
        trade_size_usd: float,
        daily_volume_usd: float,
        volatility: float,
        base_slippage_bps: float = 50.0,
    ) -> float:
        """
        Get recommended slippage setting accounting for impact.

        Args:
            trade_size_usd: Size of trade in USD
            daily_volume_usd: Daily volume in USD
            volatility: Realized volatility
            base_slippage_bps: Base slippage setting

        Returns:
            Recommended slippage in basis points
        """
        result = self.predict_impact(
            trade_size_usd=trade_size_usd,
            daily_volume_usd=daily_volume_usd,
            volatility=volatility,
        )

        # Add impact to base slippage with buffer
        recommended = base_slippage_bps + result.total_impact_bps * 1.2

        # Cap at reasonable maximum
        return min(recommended, 500.0)


# Singleton instance for convenience
_default_model: SquareRootImpactModel | None = None


def get_impact_model() -> SquareRootImpactModel:
    """Get the default impact model instance."""
    global _default_model
    if _default_model is None:
        _default_model = SquareRootImpactModel()
    return _default_model
