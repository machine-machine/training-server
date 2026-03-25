"""
Parameter effectiveness tracker for regime-specific learning.

Tracks which parameter values perform best under different market conditions,
using exponentially weighted moving averages to prioritize recent data.

Example learning:
  - stop_loss_pct=8 in bull regime  -> avg P&L +2.5%
  - stop_loss_pct=8 in bear regime  -> avg P&L -1.2%
  - stop_loss_pct=12 in bear regime -> avg P&L +0.8%
  => Use wider stop loss in bear markets

Usage:
    tracker = ParameterEffectivenessTracker()
    await tracker.update_parameter_score(
        "stop_loss_pct", 8.0, "bull", trade_outcome
    )
    best = await tracker.get_best_value("stop_loss_pct", "bull")
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParameterScore:
    """Effectiveness score for a parameter value in a specific regime."""

    parameter_name: str
    parameter_value: Any
    regime: str

    # Exponentially weighted moving average of P&L
    ewma_pnl_pct: float = 0.0
    ewma_pnl_sol: float = 0.0

    # Sample counts
    total_samples: int = 0
    live_samples: int = 0
    paper_samples: int = 0
    backtest_samples: int = 0

    # Win/loss tracking
    wins: int = 0
    losses: int = 0

    # Best/worst outcomes
    best_pnl_pct: float = float("-inf")
    worst_pnl_pct: float = float("inf")

    # Timestamps
    first_seen: datetime | None = None
    last_updated: datetime | None = None

    @property
    def win_rate(self) -> float:
        """Win rate for this parameter value in this regime."""
        total = self.wins + self.losses
        if total == 0:
            return 0.0
        return self.wins / total

    @property
    def has_sufficient_data(self) -> bool:
        """Whether we have enough data for reliable recommendations."""
        return self.total_samples >= 10

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "parameter_name": self.parameter_name,
            "parameter_value": self.parameter_value,
            "regime": self.regime,
            "ewma_pnl_pct": round(self.ewma_pnl_pct, 4),
            "ewma_pnl_sol": round(self.ewma_pnl_sol, 6),
            "total_samples": self.total_samples,
            "live_samples": self.live_samples,
            "paper_samples": self.paper_samples,
            "backtest_samples": self.backtest_samples,
            "win_rate": round(self.win_rate, 4),
            "best_pnl_pct": round(self.best_pnl_pct, 4)
            if self.best_pnl_pct != float("-inf")
            else None,
            "worst_pnl_pct": round(self.worst_pnl_pct, 4)
            if self.worst_pnl_pct != float("inf")
            else None,
            "first_seen": (self.first_seen.isoformat() if self.first_seen else None),
            "last_updated": (self.last_updated.isoformat() if self.last_updated else None),
        }


@dataclass
class ParameterRecommendation:
    """Recommendation for a parameter value."""

    parameter_name: str
    recommended_value: Any
    current_value: Any
    regime: str
    expected_pnl_pct: float
    confidence: float  # 0.0 to 1.0
    sample_count: int
    reason: str


class ParameterEffectivenessTracker:
    """
    Track which parameter values work best in which market conditions.

    Uses exponentially weighted moving averages (EWMA) with configurable
    decay to weight recent trades more heavily.

    Source weights:
    - LIVE trades: weight 1.0 (most trustworthy)
    - PAPER trades: weight 0.5 (realistic but no capital at risk)
    - BACKTEST trades: weight 0.3 (historical, may not reflect current)

    EWMA decay:
    - Half-life of 24 hours (configurable)
    - Recent trades have exponentially more influence
    """

    # Source weights for different trade origins
    SOURCE_WEIGHTS = {
        "live": 1.0,
        "paper": 0.5,
        "backtest": 0.3,
    }

    def __init__(
        self,
        ewma_halflife_hours: float = 24.0,
        min_samples_for_recommendation: int = 20,
    ) -> None:
        """
        Initialize the tracker.

        Args:
            ewma_halflife_hours: Half-life for exponential decay (hours).
            min_samples_for_recommendation: Minimum samples before recommending.
        """
        self.ewma_halflife_hours = ewma_halflife_hours
        self.min_samples = min_samples_for_recommendation

        # Compute decay factor per update
        # alpha such that weight is 0.5 after halflife hours
        # For per-trade updates, we approximate based on trade frequency
        self._alpha = 0.1  # Default smoothing factor; adjusted dynamically

        # Storage: {(param_name, str(param_value), regime): ParameterScore}
        self._scores: dict[tuple[str, str, str], ParameterScore] = {}

        # Index for fast lookup by parameter name
        self._param_index: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    def _score_key(
        self,
        parameter_name: str,
        parameter_value: Any,
        regime: str,
    ) -> tuple[str, str, str]:
        """Create a hashable key for a parameter score entry."""
        return (parameter_name, str(parameter_value), regime)

    def _get_or_create_score(
        self,
        parameter_name: str,
        parameter_value: Any,
        regime: str,
    ) -> ParameterScore:
        """Get existing score or create a new one."""
        key = self._score_key(parameter_name, parameter_value, regime)
        if key not in self._scores:
            self._scores[key] = ParameterScore(
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                regime=regime,
            )
            self._param_index[parameter_name].append(key)
        return self._scores[key]

    def _compute_weight(
        self,
        source: str,
        trade_timestamp: datetime,
    ) -> float:
        """
        Compute the combined weight for a trade update.

        Combines source weight with time decay.

        Args:
            source: Trade source ("live", "paper", "backtest").
            trade_timestamp: When the trade occurred.

        Returns:
            Combined weight factor.
        """
        source_weight = self.SOURCE_WEIGHTS.get(source, 0.1)

        # Time decay: 50% weight after halflife hours
        age_hours = (datetime.now() - trade_timestamp).total_seconds() / 3600.0
        if age_hours < 0:
            age_hours = 0

        if self.ewma_halflife_hours > 0:
            time_decay = math.pow(0.5, age_hours / self.ewma_halflife_hours)
        else:
            time_decay = 1.0

        return source_weight * time_decay

    async def update_parameter_score(
        self,
        parameter_name: str,
        parameter_value: Any,
        regime: str,
        outcome: Any,
    ) -> None:
        """
        Update effectiveness score for a parameter value in a regime.

        Uses EWMA with source-weighted and time-decayed updates.

        Args:
            parameter_name: Name of the parameter (e.g., "stop_loss_pct").
            parameter_value: The parameter value used in the trade.
            regime: Market regime during the trade.
            outcome: TradeOutcome object with pnl_pct, pnl_sol, source, timestamp.
        """
        score = self._get_or_create_score(parameter_name, parameter_value, regime)

        # Compute weight
        weight = self._compute_weight(outcome.source, outcome.timestamp)

        # Adaptive alpha: higher weight -> more influence on EWMA
        alpha = min(0.3, self._alpha * weight)

        # Update EWMA for P&L percentage
        if score.total_samples == 0:
            score.ewma_pnl_pct = outcome.pnl_pct
            score.ewma_pnl_sol = outcome.pnl_sol
        else:
            score.ewma_pnl_pct = alpha * outcome.pnl_pct + (1 - alpha) * score.ewma_pnl_pct
            score.ewma_pnl_sol = alpha * outcome.pnl_sol + (1 - alpha) * score.ewma_pnl_sol

        # Update counters
        score.total_samples += 1
        if outcome.source == "live":
            score.live_samples += 1
        elif outcome.source == "paper":
            score.paper_samples += 1
        elif outcome.source == "backtest":
            score.backtest_samples += 1

        # Win/loss
        if outcome.pnl_sol > 0:
            score.wins += 1
        else:
            score.losses += 1

        # Best/worst
        if outcome.pnl_pct > score.best_pnl_pct:
            score.best_pnl_pct = outcome.pnl_pct
        if outcome.pnl_pct < score.worst_pnl_pct:
            score.worst_pnl_pct = outcome.pnl_pct

        # Timestamps
        if score.first_seen is None:
            score.first_seen = outcome.timestamp
        score.last_updated = datetime.now()

    async def get_best_value(
        self,
        parameter_name: str,
        regime: str,
        min_samples: int | None = None,
    ) -> Any | None:
        """
        Get the best-performing value for a parameter in a regime.

        Args:
            parameter_name: Parameter to query.
            regime: Market regime.
            min_samples: Minimum sample count (uses default if None).

        Returns:
            Best parameter value, or None if insufficient data.
        """
        required = min_samples or self.min_samples

        candidates: list[tuple[Any, float, int]] = []

        for key in self._param_index.get(parameter_name, []):
            score = self._scores.get(key)
            if score is None:
                continue
            if score.regime != regime:
                continue
            if score.total_samples < required:
                continue

            candidates.append(
                (
                    score.parameter_value,
                    score.ewma_pnl_pct,
                    score.total_samples,
                )
            )

        if not candidates:
            return None

        # Sort by EWMA P&L (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_value_str = candidates[0][0]

        # Try to convert back to original type
        return self._try_parse_value(best_value_str)

    async def get_recommendations(
        self,
        current_params: dict[str, Any],
        regime: str,
    ) -> list[ParameterRecommendation]:
        """
        Get parameter change recommendations for the current regime.

        Compares current parameter values to the best-performing values
        and recommends changes where improvements are likely.

        Args:
            current_params: Current parameter values.
            regime: Current market regime.

        Returns:
            List of ParameterRecommendation objects.
        """
        recommendations: list[ParameterRecommendation] = []

        for param_name, current_value in current_params.items():
            # Get all scores for this parameter in this regime
            candidates: list[tuple[ParameterScore, float]] = []

            for key in self._param_index.get(param_name, []):
                score = self._scores.get(key)
                if score is None or score.regime != regime:
                    continue
                if score.total_samples < self.min_samples:
                    continue

                candidates.append((score, score.ewma_pnl_pct))

            if not candidates:
                continue

            # Sort by EWMA P&L (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_score = candidates[0][0]

            # Check if the best is different from current
            current_key = self._score_key(param_name, current_value, regime)
            current_score = self._scores.get(current_key)

            if current_score is None:
                # No data for current value, recommend the best
                confidence = min(1.0, best_score.total_samples / (self.min_samples * 2))
                recommendations.append(
                    ParameterRecommendation(
                        parameter_name=param_name,
                        recommended_value=self._try_parse_value(best_score.parameter_value),
                        current_value=current_value,
                        regime=regime,
                        expected_pnl_pct=best_score.ewma_pnl_pct,
                        confidence=confidence,
                        sample_count=best_score.total_samples,
                        reason=(
                            f"No data for current value in {regime} regime. "
                            f"Best observed: EWMA P&L {best_score.ewma_pnl_pct:.2f}% "
                            f"({best_score.total_samples} samples)"
                        ),
                    )
                )
                continue

            # Check if best is meaningfully better than current
            improvement = best_score.ewma_pnl_pct - current_score.ewma_pnl_pct
            if improvement > 0.5:  # At least 0.5% improvement
                confidence = min(
                    1.0,
                    (best_score.total_samples + current_score.total_samples)
                    / (self.min_samples * 4),
                )
                recommendations.append(
                    ParameterRecommendation(
                        parameter_name=param_name,
                        recommended_value=self._try_parse_value(best_score.parameter_value),
                        current_value=current_value,
                        regime=regime,
                        expected_pnl_pct=best_score.ewma_pnl_pct,
                        confidence=confidence,
                        sample_count=best_score.total_samples,
                        reason=(
                            f"Better value found: "
                            f"EWMA P&L {best_score.ewma_pnl_pct:.2f}% vs "
                            f"current {current_score.ewma_pnl_pct:.2f}% "
                            f"(+{improvement:.2f}%)"
                        ),
                    )
                )

        # Sort by expected improvement (descending)
        recommendations.sort(key=lambda r: r.expected_pnl_pct, reverse=True)
        return recommendations

    async def get_parameter_report(
        self,
        parameter_name: str,
    ) -> dict[str, Any]:
        """
        Get a full report for a parameter across all regimes and values.

        Args:
            parameter_name: The parameter to report on.

        Returns:
            Dict with breakdown by regime and value.
        """
        report: dict[str, Any] = {
            "parameter_name": parameter_name,
            "regimes": {},
        }

        for key in self._param_index.get(parameter_name, []):
            score = self._scores.get(key)
            if score is None:
                continue

            regime = score.regime
            if regime not in report["regimes"]:
                report["regimes"][regime] = {
                    "values": [],
                    "best_value": None,
                    "best_ewma_pnl_pct": float("-inf"),
                }

            report["regimes"][regime]["values"].append(score.to_dict())

            if score.ewma_pnl_pct > report["regimes"][regime]["best_ewma_pnl_pct"]:
                report["regimes"][regime]["best_ewma_pnl_pct"] = score.ewma_pnl_pct
                report["regimes"][regime]["best_value"] = score.parameter_value

        return report

    def _try_parse_value(self, value_str: str) -> Any:
        """Try to parse a string value back to its original type."""
        if value_str == "None":
            return None
        if value_str == "True":
            return True
        if value_str == "False":
            return False
        try:
            return int(value_str)
        except (ValueError, TypeError):
            pass
        try:
            return float(value_str)
        except (ValueError, TypeError):
            pass
        return value_str

    @property
    def total_scores(self) -> int:
        """Total number of score entries tracked."""
        return len(self._scores)

    @property
    def tracked_parameters(self) -> list[str]:
        """List of parameter names being tracked."""
        return list(self._param_index.keys())
