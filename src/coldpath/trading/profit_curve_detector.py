"""
Profit Curve Degradation Detector.

Detects when strategy performance is degrading by analyzing:
- Profit curve slope changes
- Rolling metrics degradation
- Regime shift detection
- Statistical significance tests
- Early warning signals

Key Features:
- Real-time profit curve monitoring
- Statistical degradation detection
- Early warning system for strategy decay
- Automatic alert generation
- Performance regime classification
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Severity level of degradation."""

    NONE = "none"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class DegradationType(Enum):
    """Type of performance degradation."""

    SLOPE_DECLINE = "slope_decline"
    VOLATILITY_INCREASE = "volatility_increase"
    DRAWDOWN_EXPANSION = "drawdown_expansion"
    WIN_RATE_DECAY = "win_rate_decay"
    SHARPE_COLLAPSE = "sharpe_collapse"
    CONSISTENCY_LOSS = "consistency_loss"
    REGIME_SHIFT = "regime_shift"


class AlertType(Enum):
    """Type of alert generated."""

    INFO = "info"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


@dataclass
class DegradationAlert:
    """Alert for performance degradation."""

    timestamp: str
    alert_type: AlertType
    degradation_type: DegradationType
    level: DegradationLevel
    message: str
    current_value: float
    threshold_value: float
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "alert_type": self.alert_type.value,
            "degradation_type": self.degradation_type.value,
            "level": self.level.value,
            "message": self.message,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "recommended_action": self.recommended_action,
        }


@dataclass
class CurveStatistics:
    """Statistics about the profit curve."""

    slope: float
    slope_acceleration: float
    volatility: float
    max_drawdown: float
    current_drawdown: float
    sharpe_estimate: float
    win_rate: float
    consistency_score: float
    r_squared: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "slope": self.slope,
            "slope_acceleration": self.slope_acceleration,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "sharpe_estimate": self.sharpe_estimate,
            "win_rate": self.win_rate,
            "consistency_score": self.consistency_score,
            "r_squared": self.r_squared,
        }


@dataclass
class DegradationResult:
    """Result of degradation analysis."""

    is_degrading: bool
    degradation_level: DegradationLevel
    degradation_types: list[DegradationType]
    current_stats: CurveStatistics
    baseline_stats: CurveStatistics | None
    alerts: list[DegradationAlert]
    slope_change_pct: float
    confidence: float
    recommended_actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_degrading": self.is_degrading,
            "degradation_level": self.degradation_level.value,
            "degradation_types": [d.value for d in self.degradation_types],
            "current_stats": self.current_stats.to_dict(),
            "baseline_stats": self.baseline_stats.to_dict() if self.baseline_stats else None,
            "alerts": [a.to_dict() for a in self.alerts],
            "slope_change_pct": self.slope_change_pct,
            "confidence": self.confidence,
            "recommended_actions": self.recommended_actions,
        }


@dataclass
class DegradationDetectorConfig:
    """Configuration for degradation detector."""

    lookback_periods: int = 100
    short_window: int = 20
    medium_window: int = 50
    long_window: int = 100

    slope_decline_threshold: float = -0.5
    slope_acceleration_threshold: float = -0.3
    volatility_increase_threshold: float = 0.5
    drawdown_expansion_threshold: float = 0.3
    win_rate_decay_threshold: float = -0.1
    sharpe_collapse_threshold: float = -0.5

    consistency_threshold: float = 0.5
    r_squared_threshold: float = 0.3

    alert_cooldown_periods: int = 10

    min_samples_for_detection: int = 30


class ProfitCurveDegradationDetector:
    """Detects profit curve degradation in real-time.

    Monitors strategy performance and detects when the profit curve
    shows signs of degradation, allowing for early intervention.

    Usage:
        detector = ProfitCurveDegradationDetector()

        for trade in trades:
            detector.add_trade(trade)

            if len(trades) % 10 == 0:
                result = detector.analyze()
                if result.is_degrading:
                    print(f"Warning: {result.degradation_level.value}")
    """

    def __init__(self, config: DegradationDetectorConfig | None = None):
        self.config = config or DegradationDetectorConfig()

        self._equity_curve: deque = deque(maxlen=self.config.lookback_periods)
        self._returns: deque = deque(maxlen=self.config.lookback_periods)
        self._trade_results: deque = deque(maxlen=self.config.lookback_periods)
        self._timestamps: deque = deque(maxlen=self.config.lookback_periods)

        self._baseline_stats: CurveStatistics | None = None
        self._last_alert_period: int = 0
        self._current_period: int = 0

        self._slope_history: deque = deque(maxlen=50)
        self._sharpe_history: deque = deque(maxlen=50)

    def add_equity_point(self, equity: float, timestamp: str | None = None) -> None:
        """Add an equity curve data point.

        Args:
            equity: Current equity value
            timestamp: Optional timestamp string
        """
        self._current_period += 1

        if len(self._equity_curve) > 0:
            prev_equity = self._equity_curve[-1]
            ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self._returns.append(ret)

        self._equity_curve.append(equity)
        self._timestamps.append(timestamp or datetime.utcnow().isoformat())

    def add_trade(self, pnl_pct: float, timestamp: str | None = None) -> None:
        """Add a trade result.

        Args:
            pnl_pct: Trade P&L as percentage
            timestamp: Optional timestamp string
        """
        self._current_period += 1
        self._trade_results.append(pnl_pct)
        self._timestamps.append(timestamp or datetime.utcnow().isoformat())

    def set_baseline(self, equity_curve: list[float]) -> None:
        """Set baseline statistics from historical data.

        Args:
            equity_curve: Historical equity curve
        """
        if len(equity_curve) < self.config.min_samples_for_detection:
            logger.warning("Insufficient data for baseline calculation")
            return

        self._baseline_stats = self._calculate_statistics(equity_curve)
        logger.info(f"Baseline set: slope={self._baseline_stats.slope:.4f}")

    def analyze(self) -> DegradationResult:
        """Analyze current profit curve for degradation.

        Returns:
            DegradationResult with analysis and alerts
        """
        if len(self._equity_curve) < self.config.min_samples_for_detection:
            return self._insufficient_data_result()

        equity_list = list(self._equity_curve)
        current_stats = self._calculate_statistics(equity_list)

        self._slope_history.append(current_stats.slope)

        degradation_types: list[DegradationType] = []
        alerts: list[DegradationAlert] = []

        slope_change = 0.0
        if self._baseline_stats:
            if self._baseline_stats.slope != 0:
                slope_change = (
                    (current_stats.slope - self._baseline_stats.slope)
                    / abs(self._baseline_stats.slope)
                    * 100
                )

            if slope_change < self.config.slope_decline_threshold * 100:
                degradation_types.append(DegradationType.SLOPE_DECLINE)
                alerts.append(
                    self._create_alert(
                        DegradationType.SLOPE_DECLINE,
                        current_stats.slope,
                        self._baseline_stats.slope,
                        f"Profit slope declined {abs(slope_change):.1f}%",
                    )
                )

            if current_stats.slope_acceleration < self.config.slope_acceleration_threshold:
                degradation_types.append(DegradationType.CONSISTENCY_LOSS)
                alerts.append(
                    self._create_alert(
                        DegradationType.CONSISTENCY_LOSS,
                        current_stats.slope_acceleration,
                        self.config.slope_acceleration_threshold,
                        "Negative slope acceleration detected",
                    )
                )

            vol_change = (
                (current_stats.volatility - self._baseline_stats.volatility)
                / self._baseline_stats.volatility
                if self._baseline_stats.volatility > 0
                else 0
            )
            if vol_change > self.config.volatility_increase_threshold:
                degradation_types.append(DegradationType.VOLATILITY_INCREASE)
                alerts.append(
                    self._create_alert(
                        DegradationType.VOLATILITY_INCREASE,
                        current_stats.volatility,
                        self._baseline_stats.volatility,
                        f"Volatility increased {vol_change * 100:.1f}%",
                    )
                )

            dd_change = current_stats.current_drawdown - self._baseline_stats.current_drawdown
            if dd_change > self.config.drawdown_expansion_threshold:
                degradation_types.append(DegradationType.DRAWDOWN_EXPANSION)
                alerts.append(
                    self._create_alert(
                        DegradationType.DRAWDOWN_EXPANSION,
                        current_stats.current_drawdown,
                        self._baseline_stats.current_drawdown,
                        f"Drawdown expanded by {dd_change * 100:.1f}%",
                    )
                )

            if self._baseline_stats.win_rate > 0:
                wr_change = current_stats.win_rate - self._baseline_stats.win_rate
                if wr_change < self.config.win_rate_decay_threshold:
                    degradation_types.append(DegradationType.WIN_RATE_DECAY)
                    alerts.append(
                        self._create_alert(
                            DegradationType.WIN_RATE_DECAY,
                            current_stats.win_rate,
                            self._baseline_stats.win_rate,
                            f"Win rate declined {abs(wr_change) * 100:.1f}%",
                        )
                    )

            if self._baseline_stats.sharpe_estimate > 0:
                sharpe_change = (
                    current_stats.sharpe_estimate - self._baseline_stats.sharpe_estimate
                ) / self._baseline_stats.sharpe_estimate
                if sharpe_change < self.config.sharpe_collapse_threshold:
                    degradation_types.append(DegradationType.SHARPE_COLLAPSE)
                    alerts.append(
                        self._create_alert(
                            DegradationType.SHARPE_COLLAPSE,
                            current_stats.sharpe_estimate,
                            self._baseline_stats.sharpe_estimate,
                            f"Sharpe collapsed {abs(sharpe_change) * 100:.1f}%",
                        )
                    )

        if current_stats.r_squared < self.config.r_squared_threshold:
            if DegradationType.CONSISTENCY_LOSS not in degradation_types:
                degradation_types.append(DegradationType.CONSISTENCY_LOSS)

        degradation_level = self._determine_degradation_level(degradation_types, current_stats)

        recommended_actions = self._generate_recommendations(degradation_types, degradation_level)

        is_degrading = degradation_level not in [DegradationLevel.NONE, DegradationLevel.MINIMAL]

        confidence = self._calculate_confidence(current_stats)

        return DegradationResult(
            is_degrading=is_degrading,
            degradation_level=degradation_level,
            degradation_types=degradation_types,
            current_stats=current_stats,
            baseline_stats=self._baseline_stats,
            alerts=alerts,
            slope_change_pct=round(slope_change, 2),
            confidence=confidence,
            recommended_actions=recommended_actions,
        )

    def _calculate_statistics(self, equity_curve: list[float]) -> CurveStatistics:
        """Calculate curve statistics."""
        if len(equity_curve) < 2:
            return CurveStatistics(
                slope=0,
                slope_acceleration=0,
                volatility=0,
                max_drawdown=0,
                current_drawdown=0,
                sharpe_estimate=0,
                win_rate=0,
                consistency_score=0,
                r_squared=0,
            )

        # Convert to numpy array for consistent handling
        equity_arr = np.array(equity_curve, dtype=float)
        returns = np.diff(equity_arr) / equity_arr[:-1]
        # Filter out zero division cases
        returns = returns[equity_arr[:-1] != 0]

        if len(returns) == 0:
            returns = np.array([0])

        x = np.arange(len(equity_curve))
        y = np.array(equity_curve)

        if len(x) > 1:
            slope, _ = np.polyfit(x, y, 1)
        else:
            slope = 0

        if len(equity_curve) >= 30:
            mid = len(equity_curve) // 2
            first_half = equity_curve[:mid]
            second_half = equity_curve[mid:]

            if len(first_half) > 1 and len(second_half) > 1:
                slope1, _ = np.polyfit(np.arange(len(first_half)), first_half, 1)
                slope2, _ = np.polyfit(np.arange(len(second_half)), second_half, 1)
                slope_acceleration = (slope2 - slope1) / abs(slope1) if slope1 != 0 else 0
            else:
                slope_acceleration = 0
        else:
            slope_acceleration = 0

        volatility = float(np.std(returns)) if len(returns) > 1 else 0

        peak = equity_curve[0]
        max_dd = 0
        current_dd = 0

        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            current_dd = dd

        mean_ret = np.mean(returns) if len(returns) > 0 else 0
        std_ret = np.std(returns) if len(returns) > 1 else 1
        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0

        # Use numpy operations for win rate calculation
        if len(self._trade_results) > 0:
            trades_arr = np.array(list(self._trade_results), dtype=float)
        else:
            trades_arr = np.asarray(returns, dtype=float).flatten()
        wins = int(np.sum(trades_arr > 0))
        win_rate = wins / len(trades_arr) if len(trades_arr) > 0 else 0

        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        y_pred = slope * x + (y_mean - slope * np.mean(x))
        ss_res = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Use numpy operations for consistency calculation
        returns_arr = np.asarray(returns, dtype=float).flatten()
        positive_returns = int(np.sum(returns_arr > 0))
        consistency = positive_returns / len(returns_arr) if len(returns_arr) > 0 else 0

        return CurveStatistics(
            slope=float(slope),
            slope_acceleration=float(slope_acceleration),
            volatility=float(volatility),
            max_drawdown=float(max_dd),
            current_drawdown=float(current_dd),
            sharpe_estimate=float(sharpe),
            win_rate=float(win_rate),
            consistency_score=float(consistency),
            r_squared=float(max(0, r_squared)),
        )

    def _determine_degradation_level(
        self,
        degradation_types: list[DegradationType],
        stats: CurveStatistics,
    ) -> DegradationLevel:
        """Determine overall degradation level."""
        if len(degradation_types) == 0:
            return DegradationLevel.NONE

        if len(degradation_types) >= 4:
            return DegradationLevel.CRITICAL

        if DegradationType.SHARPE_COLLAPSE in degradation_types:
            if stats.sharpe_estimate < 0:
                return DegradationLevel.CRITICAL
            return DegradationLevel.SEVERE

        if DegradationType.SLOPE_DECLINE in degradation_types:
            if stats.slope < 0:
                return DegradationLevel.SEVERE
            return DegradationLevel.MODERATE

        if DegradationType.DRAWDOWN_EXPANSION in degradation_types:
            if stats.current_drawdown > 0.25:
                return DegradationLevel.SEVERE
            return DegradationLevel.MODERATE

        if len(degradation_types) >= 2:
            return DegradationLevel.MODERATE

        return DegradationLevel.MINIMAL

    def _create_alert(
        self,
        degradation_type: DegradationType,
        current_value: float,
        threshold_value: float,
        message: str,
    ) -> DegradationAlert:
        """Create a degradation alert."""
        if degradation_type in [DegradationType.SHARPE_COLLAPSE, DegradationType.REGIME_SHIFT]:
            alert_type = AlertType.DANGER
            action = "Consider pausing strategy and reviewing parameters"
        elif degradation_type in [
            DegradationType.SLOPE_DECLINE,
            DegradationType.DRAWDOWN_EXPANSION,
        ]:
            alert_type = AlertType.WARNING
            action = "Monitor closely and prepare to reduce position sizes"
        else:
            alert_type = AlertType.WARNING
            action = "Monitor for continued degradation"

        return DegradationAlert(
            timestamp=datetime.utcnow().isoformat(),
            alert_type=alert_type,
            degradation_type=degradation_type,
            level=self._determine_degradation_level(
                [degradation_type],
                CurveStatistics(
                    slope=0,
                    slope_acceleration=0,
                    volatility=0,
                    max_drawdown=0,
                    current_drawdown=0,
                    sharpe_estimate=0,
                    win_rate=0,
                    consistency_score=0,
                    r_squared=0,
                ),
            ),
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            recommended_action=action,
        )

    def _generate_recommendations(
        self,
        degradation_types: list[DegradationType],
        level: DegradationLevel,
    ) -> list[str]:
        """Generate recommended actions based on degradation."""
        recommendations = []

        if level == DegradationLevel.CRITICAL:
            recommendations.append("STOP: Immediately halt new positions")
            recommendations.append("Review strategy for fundamental issues")
            recommendations.append("Consider full parameter recalibration")
        elif level == DegradationLevel.SEVERE:
            recommendations.append("Reduce position sizes by 50%")
            recommendations.append("Tighten stop-loss levels")
            recommendations.append("Review recent market regime changes")
        elif level == DegradationLevel.MODERATE:
            recommendations.append("Reduce position sizes by 25%")
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Consider parameter adjustments")
        elif level == DegradationLevel.MINIMAL:
            recommendations.append("Continue monitoring")
            recommendations.append("Document current performance")

        if DegradationType.VOLATILITY_INCREASE in degradation_types:
            recommendations.append("Widen stops or reduce position sizes due to volatility")

        if DegradationType.WIN_RATE_DECAY in degradation_types:
            recommendations.append("Review entry criteria and signal quality")

        return recommendations

    def _calculate_confidence(self, stats: CurveStatistics) -> float:
        """Calculate confidence in degradation assessment."""
        n_points = len(self._equity_curve)

        if n_points < self.config.min_samples_for_detection:
            return 0.3

        if n_points < self.config.short_window:
            confidence = 0.5
        elif n_points < self.config.medium_window:
            confidence = 0.7
        elif n_points < self.config.long_window:
            confidence = 0.85
        else:
            confidence = 0.95

        confidence *= stats.r_squared * 0.5 + 0.5

        return round(min(1.0, confidence), 2)

    def _insufficient_data_result(self) -> DegradationResult:
        """Return result for insufficient data."""
        return DegradationResult(
            is_degrading=False,
            degradation_level=DegradationLevel.NONE,
            degradation_types=[],
            current_stats=CurveStatistics(
                slope=0,
                slope_acceleration=0,
                volatility=0,
                max_drawdown=0,
                current_drawdown=0,
                sharpe_estimate=0,
                win_rate=0,
                consistency_score=0,
                r_squared=0,
            ),
            baseline_stats=self._baseline_stats,
            alerts=[],
            slope_change_pct=0,
            confidence=0.1,
            recommended_actions=["Collect more data before analysis"],
        )

    def get_current_equity(self) -> float | None:
        """Get the most recent equity value."""
        return self._equity_curve[-1] if self._equity_curve else None

    def get_statistics_summary(self) -> dict[str, Any]:
        """Get summary of current statistics."""
        if len(self._equity_curve) < 2:
            return {"status": "insufficient_data"}

        equity_list = list(self._equity_curve)
        stats = self._calculate_statistics(equity_list)

        return {
            "data_points": len(self._equity_curve),
            "current_equity": equity_list[-1],
            "starting_equity": equity_list[0],
            "total_return_pct": (equity_list[-1] - equity_list[0]) / equity_list[0] * 100
            if equity_list[0] > 0
            else 0,
            "statistics": stats.to_dict(),
            "baseline_set": self._baseline_stats is not None,
        }

    def reset(self) -> None:
        """Reset the detector state."""
        self._equity_curve.clear()
        self._returns.clear()
        self._trade_results.clear()
        self._timestamps.clear()
        self._slope_history.clear()
        self._sharpe_history.clear()
        self._current_period = 0
        self._last_alert_period = 0


def detect_profit_degradation(
    equity_curve: list[float],
    baseline_curve: list[float] | None = None,
) -> DegradationResult:
    """Simple function to detect profit curve degradation.

    Args:
        equity_curve: Current equity curve
        baseline_curve: Optional baseline curve for comparison

    Returns:
        DegradationResult with analysis
    """
    detector = ProfitCurveDegradationDetector()

    if baseline_curve:
        detector.set_baseline(baseline_curve)

    for equity in equity_curve:
        detector.add_equity_point(equity)

    return detector.analyze()
