#
# alpha_decay.py
# 2DEXY ColdPath
#
# Alpha Decay Measurement Module
# Measures how quickly trading signals decay after generation.
# Critical for non-priority access traders - signals become stale over time.
#

import logging
import math
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionDecision(Enum):
    """Decision on whether to execute based on signal freshness."""

    EXECUTE = "execute"  # Signal is fresh, execute normally
    EXECUTE_WITH_REDUCED_SIZE = "reduced"  # Signal is somewhat stale, reduce position
    SKIP = "skip"  # Signal has decayed too much, skip


@dataclass
class DecayResult:
    """Result of alpha decay analysis."""

    half_life_seconds: float  # Time for alpha to decay by 50%
    decay_rate: float  # Exponential decay constant (lambda)
    current_freshness: float  # 0-1, how fresh current signal is
    confidence_multiplier: float  # Multiplier for signal confidence
    recommendation: ExecutionDecision  # What to do with current signal
    samples_used: int  # Number of data points used
    r_squared: float  # Goodness of fit (0-1)


@dataclass
class SignalOutcome:
    """A signal and its eventual outcome for decay analysis."""

    signal_timestamp_ms: int
    execution_timestamp_ms: int
    signal_age_ms: int  # execution_timestamp - signal_timestamp
    pnl_percent: float  # Realized P&L as percentage
    win: bool  # Was this trade profitable?
    token_mint: str
    regime: str = "unknown"  # Market regime at signal time


@dataclass
class RegimeDecayConfig:
    """Decay configuration per market regime."""

    half_life_seconds: float = 30.0
    min_samples: int = 10
    skip_threshold: float = 1.0  # Skip if signal age > half_life * skip_threshold


class AlphaDecayAnalyzer:
    """
    Measure signal quality decay over time.

    Alpha decay follows an exponential model:
        alpha(t) = alpha_0 * exp(-lambda * t)

    Where:
        - alpha_0 is the initial alpha (signal quality at generation)
        - lambda is the decay rate
        - t is time since signal generation
        - half_life = ln(2) / lambda

    For non-priority access traders, signals typically decay faster
    because we're competing against faster participants.
    """

    # Default configurations per regime (class-level constant)
    _DEFAULT_REGIME_CONFIGS = {
        "trending": RegimeDecayConfig(half_life_seconds=45.0, skip_threshold=1.2),
        "ranging": RegimeDecayConfig(half_life_seconds=25.0, skip_threshold=0.8),
        "volatile": RegimeDecayConfig(half_life_seconds=15.0, skip_threshold=0.6),
        "low_liquidity": RegimeDecayConfig(half_life_seconds=10.0, skip_threshold=0.5),
        "high_volume": RegimeDecayConfig(half_life_seconds=35.0, skip_threshold=1.0),
        "unknown": RegimeDecayConfig(half_life_seconds=30.0, skip_threshold=1.0),
    }

    def __init__(
        self,
        min_samples_for_analysis: int = 20,
        default_half_life: float = 30.0,
        max_signal_age_hours: float = 24.0,
    ):
        """
        Initialize the alpha decay analyzer.

        Args:
            min_samples_for_analysis: Minimum samples needed for decay estimation
            default_half_life: Default half-life when not enough data
            max_signal_age_hours: Discard signals older than this
        """
        self.min_samples = min_samples_for_analysis
        self.default_half_life = default_half_life
        self.max_signal_age_hours = max_signal_age_hours

        # Storage for historical signal-outcome pairs
        self.signal_outcomes: list[SignalOutcome] = []

        # Cache for regime-specific decay rates (copy from class defaults)
        self._regime_configs: dict[str, RegimeDecayConfig] = dict(self._DEFAULT_REGIME_CONFIGS)

        # Last computed decay metrics
        self._last_analysis: DecayResult | None = None

    def record_outcome(
        self,
        signal_timestamp_ms: int,
        execution_timestamp_ms: int,
        pnl_percent: float,
        token_mint: str,
        regime: str = "unknown",
    ) -> None:
        """
        Record a signal outcome for decay analysis.

        Args:
            signal_timestamp_ms: When the signal was generated
            execution_timestamp_ms: When the trade was executed
            pnl_percent: Realized P&L as percentage
            token_mint: Token identifier
            regime: Market regime at signal time
        """
        signal_age_ms = execution_timestamp_ms - signal_timestamp_ms

        # Skip if signal is too old (outlier)
        max_age_ms = self.max_signal_age_hours * 3600 * 1000
        if signal_age_ms > max_age_ms or signal_age_ms < 0:
            return

        outcome = SignalOutcome(
            signal_timestamp_ms=signal_timestamp_ms,
            execution_timestamp_ms=execution_timestamp_ms,
            signal_age_ms=signal_age_ms,
            pnl_percent=pnl_percent,
            win=pnl_percent > 0,
            token_mint=token_mint,
            regime=regime,
        )

        self.signal_outcomes.append(outcome)

        # Keep only recent outcomes (last 1000)
        if len(self.signal_outcomes) > 1000:
            self.signal_outcomes = self.signal_outcomes[-1000:]

    def measure_half_life(
        self,
        regime: str | None = None,
        min_samples: int | None = None,
    ) -> DecayResult:
        """
        Measure the half-life of alpha decay from historical data.

        Groups trades by execution lag (signal age) and calculates
        average P&L per lag bucket, then fits an exponential decay curve.

        Args:
            regime: Optional regime filter
            min_samples: Minimum samples required

        Returns:
            DecayResult with half-life and fit quality
        """
        min_samples = min_samples or self.min_samples

        # Filter outcomes by regime if specified
        outcomes = self.signal_outcomes
        if regime:
            outcomes = [o for o in outcomes if o.regime == regime]

        if len(outcomes) < min_samples:
            logger.warning(
                f"Insufficient samples for decay analysis: {len(outcomes)} < {min_samples}"
            )
            return DecayResult(
                half_life_seconds=self.default_half_life,
                decay_rate=math.log(2) / self.default_half_life,
                current_freshness=1.0,
                confidence_multiplier=1.0,
                recommendation=ExecutionDecision.EXECUTE,
                samples_used=len(outcomes),
                r_squared=0.0,
            )

        # Group by signal age buckets (in seconds)
        bucket_size_seconds = 5.0  # 5-second buckets
        buckets: dict[int, list[float]] = {}

        for outcome in outcomes:
            bucket_key = int(outcome.signal_age_ms / 1000 / bucket_size_seconds)
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(outcome.pnl_percent)

        # Calculate average P&L per bucket
        bucket_data: list[tuple[float, float]] = []  # (time_seconds, avg_pnl)
        for bucket_key, pnls in buckets.items():
            if len(pnls) >= 3:  # Need at least 3 samples per bucket
                time_seconds = bucket_key * bucket_size_seconds
                avg_pnl = sum(pnls) / len(pnls)
                bucket_data.append((time_seconds, avg_pnl))

        if len(bucket_data) < 3:
            logger.warning(f"Insufficient buckets for decay fit: {len(bucket_data)}")
            return DecayResult(
                half_life_seconds=self.default_half_life,
                decay_rate=math.log(2) / self.default_half_life,
                current_freshness=1.0,
                confidence_multiplier=1.0,
                recommendation=ExecutionDecision.EXECUTE,
                samples_used=len(outcomes),
                r_squared=0.0,
            )

        # Fit exponential decay: pnl(t) = a * exp(-lambda * t)
        # Take log: log(pnl) = log(a) - lambda * t
        # Use linear regression on log-transformed data

        # Filter positive P&L values for log transform
        positive_data = [(t, p) for t, p in bucket_data if p > 0]

        if len(positive_data) < 3:
            # Fallback: use win rate instead of P&L
            bucket_wins: dict[int, tuple[int, int]] = {}
            for outcome in outcomes:
                bucket_key = int(outcome.signal_age_ms / 1000 / bucket_size_seconds)
                if bucket_key not in bucket_wins:
                    bucket_wins[bucket_key] = (0, 0)
                wins, total = bucket_wins[bucket_key]
                bucket_wins[bucket_key] = (wins + (1 if outcome.win else 0), total + 1)

            bucket_data = [
                (k * bucket_size_seconds, w / t if t > 0 else 0.5)
                for k, (w, t) in bucket_wins.items()
                if t >= 3
            ]

        if len(bucket_data) < 3:
            return DecayResult(
                half_life_seconds=self.default_half_life,
                decay_rate=math.log(2) / self.default_half_life,
                current_freshness=1.0,
                confidence_multiplier=1.0,
                recommendation=ExecutionDecision.EXECUTE,
                samples_used=len(outcomes),
                r_squared=0.0,
            )

        # Linear regression on log-transformed data
        n = len(bucket_data)
        sum_t = sum(t for t, _ in bucket_data)
        sum_log_p = sum(math.log(max(p, 0.001)) for _, p in bucket_data)
        sum_t_log_p = sum(t * math.log(max(p, 0.001)) for t, p in bucket_data)
        sum_t_sq = sum(t * t for t, _ in bucket_data)

        # Calculate slope (negative decay rate)
        slope = 0.0  # Initialize slope
        denominator = n * sum_t_sq - sum_t * sum_t
        if abs(denominator) < 0.0001:
            decay_rate = math.log(2) / self.default_half_life
        else:
            # slope = -lambda
            slope = (n * sum_t_log_p - sum_t * sum_log_p) / denominator
            decay_rate = max(-slope, 0.001)  # Ensure positive decay rate

        # Calculate R-squared
        mean_log_p = sum_log_p / n
        ss_tot = sum((math.log(max(p, 0.001)) - mean_log_p) ** 2 for _, p in bucket_data)
        intercept = (sum_log_p - slope * sum_t) / n
        ss_res = sum(
            (math.log(max(p, 0.001)) - (intercept + slope * t)) ** 2 for t, p in bucket_data
        )
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate half-life
        half_life = math.log(2) / decay_rate

        self._last_analysis = DecayResult(
            half_life_seconds=half_life,
            decay_rate=decay_rate,
            current_freshness=1.0,  # Will be updated for current signal
            confidence_multiplier=1.0,
            recommendation=ExecutionDecision.EXECUTE,
            samples_used=len(outcomes),
            r_squared=r_squared,
        )

        logger.info(
            f"Alpha decay analysis: half_life={half_life:.1f}s, "
            f"r²={r_squared:.3f}, samples={len(outcomes)}"
        )

        return self._last_analysis

    def should_execute(
        self,
        signal_age_ms: int,
        current_regime: str = "unknown",
    ) -> tuple[ExecutionDecision, float]:
        """
        Determine if a signal should still be executed based on its age.

        Args:
            signal_age_ms: Age of the signal in milliseconds
            current_regime: Current market regime

        Returns:
            Tuple of (ExecutionDecision, confidence_multiplier)
        """
        config = self._regime_configs.get(current_regime, self._regime_configs["unknown"])
        half_life = config.half_life_seconds

        # Refresh analysis if we have new data
        if len(self.signal_outcomes) >= self.min_samples:
            analysis = self.measure_half_life(regime=current_regime)
            if analysis.r_squared > 0.3:  # Only use if fit is reasonable
                half_life = analysis.half_life_seconds

        signal_age_seconds = signal_age_ms / 1000.0
        freshness_ratio = signal_age_seconds / half_life

        # Calculate confidence multiplier based on decay
        # alpha(t) = alpha_0 * exp(-lambda * t) = exp(-ln(2) * t / half_life)
        confidence_multiplier = math.exp(-math.log(2) * freshness_ratio)

        # Determine execution decision
        if freshness_ratio < 0.5:
            # Signal is fresh (< 50% of half-life)
            decision = ExecutionDecision.EXECUTE
        elif freshness_ratio < config.skip_threshold:
            # Signal is somewhat stale, reduce position
            decision = ExecutionDecision.EXECUTE_WITH_REDUCED_SIZE
        else:
            # Signal has decayed too much
            decision = ExecutionDecision.SKIP

        return decision, confidence_multiplier

    def get_position_size_multiplier(
        self,
        signal_age_ms: int,
        current_regime: str = "unknown",
    ) -> float:
        """
        Get the position size multiplier based on signal freshness.

        Use this to scale down positions for older signals.

        Args:
            signal_age_ms: Age of the signal in milliseconds
            current_regime: Current market regime

        Returns:
            Position size multiplier (0.0 to 1.0)
        """
        decision, confidence_mult = self.should_execute(signal_age_ms, current_regime)

        if decision == ExecutionDecision.SKIP:
            return 0.0
        elif decision == ExecutionDecision.EXECUTE_WITH_REDUCED_SIZE:
            # Reduce position more aggressively for stale signals
            return confidence_mult * 0.7
        else:
            # Full position for fresh signals, but still apply confidence
            return min(confidence_mult, 1.0)

    def update_regime_config(
        self,
        regime: str,
        half_life_seconds: float | None = None,
        skip_threshold: float | None = None,
    ) -> None:
        """Update decay configuration for a specific regime."""
        if regime not in self._regime_configs:
            self._regime_configs[regime] = RegimeDecayConfig()

        config = self._regime_configs[regime]
        if half_life_seconds is not None:
            config.half_life_seconds = half_life_seconds
        if skip_threshold is not None:
            config.skip_threshold = skip_threshold

    def get_stats(self) -> dict:
        """Get current statistics for monitoring."""
        regime_counts: dict[str, int] = {}
        for outcome in self.signal_outcomes:
            regime_counts[outcome.regime] = regime_counts.get(outcome.regime, 0) + 1

        return {
            "total_outcomes": len(self.signal_outcomes),
            "regime_distribution": regime_counts,
            "last_half_life": self._last_analysis.half_life_seconds
            if self._last_analysis
            else None,
            "last_r_squared": self._last_analysis.r_squared if self._last_analysis else None,
            "regime_configs": {
                regime: {
                    "half_life": config.half_life_seconds,
                    "skip_threshold": config.skip_threshold,
                }
                for regime, config in self._regime_configs.items()
            },
        }


# Singleton instance for global use
_analyzer: AlphaDecayAnalyzer | None = None


def get_alpha_decay_analyzer() -> AlphaDecayAnalyzer:
    """Get the global alpha decay analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = AlphaDecayAnalyzer()
    return _analyzer
