"""
Adaptive Parameter Adjuster.

Dynamically adjusts trading parameters based on:
- Market regime detection
- Performance feedback
- Risk management constraints
- Volatility adjustments

Key Features:
- Real-time parameter adaptation
- Market regime awareness
- Risk-based adjustments
- Learning from trade history
- Smooth parameter transitions
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class AdjustmentTrigger(Enum):
    """What triggered the adjustment."""

    MARKET_REGIME = "market_regime"
    PERFORMANCE = "performance"
    RISK_MANAGEMENT = "risk_management"
    VOLATILITY = "volatility"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    DEGRADATION = "degradation"


class ParameterDirection(Enum):
    """Direction of parameter change."""

    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"


@dataclass
class ParameterAdjustment:
    """A single parameter adjustment."""

    parameter: str
    old_value: float
    new_value: float
    change_pct: float
    direction: ParameterDirection
    trigger: AdjustmentTrigger
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "change_pct": self.change_pct,
            "direction": self.direction.value,
            "trigger": self.trigger.value,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass
class AdjustmentResult:
    """Result of parameter adjustment."""

    adjustments: list[ParameterAdjustment]
    old_params: dict[str, Any]
    new_params: dict[str, Any]
    confidence: float
    summary: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adjustments": [a.to_dict() for a in self.adjustments],
            "old_params": self.old_params,
            "new_params": self.new_params,
            "confidence": self.confidence,
            "summary": self.summary,
            "warnings": self.warnings,
        }


@dataclass
class AdaptiveParameter:
    """Definition of an adaptive parameter."""

    name: str
    current_value: float
    min_value: float
    max_value: float
    default_value: float
    adjustment_sensitivity: float = 1.0
    adjustment_smoothing: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "current_value": self.current_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "default_value": self.default_value,
            "adjustment_sensitivity": self.adjustment_sensitivity,
        }


@dataclass
class AdaptiveAdjusterConfig:
    """Configuration for adaptive parameter adjuster."""

    max_adjustment_pct: float = 0.2
    min_adjustment_pct: float = 0.02
    adjustment_cooldown: int = 5
    smoothing_factor: float = 0.3

    volatility_high_threshold: float = 0.05
    volatility_low_threshold: float = 0.01

    performance_lookback: int = 20
    min_trades_for_adjustment: int = 10

    risk_adjustment_enabled: bool = True
    max_risk_increase_pct: float = 0.5


class AdaptiveParameterAdjuster:
    """Adaptive parameter adjustment system.

    Dynamically adjusts trading parameters based on market conditions
    and performance feedback.

    Usage:
        adjuster = AdaptiveParameterAdjuster()

        adjuster.register_parameter("stop_loss_pct", 8.0, 3.0, 20.0)
        adjuster.register_parameter("take_profit_pct", 20.0, 5.0, 60.0)

        result = adjuster.adjust_for_volatility(current_volatility=0.03)
        result = adjuster.adjust_for_performance(recent_trades)
    """

    def __init__(self, config: AdaptiveAdjusterConfig | None = None):
        self.config = config or AdaptiveAdjusterConfig()

        self._parameters: dict[str, AdaptiveParameter] = {}
        self._adjustment_history: deque = deque(maxlen=100)
        self._trade_history: deque = deque(maxlen=200)

        self._last_adjustment_period: int = 0
        self._current_period: int = 0
        self._consecutive_adjustments: dict[str, int] = {}

        self._performance_metrics: dict[str, float] = {}

    def register_parameter(
        self,
        name: str,
        default_value: float,
        min_value: float,
        max_value: float,
        sensitivity: float = 1.0,
        smoothing: float = 0.5,
    ) -> None:
        """Register a parameter for adaptive adjustment.

        Args:
            name: Parameter name
            default_value: Default value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            sensitivity: How responsive to adjustments (0-1)
            smoothing: How smooth the transitions are (0-1)
        """
        self._parameters[name] = AdaptiveParameter(
            name=name,
            current_value=default_value,
            min_value=min_value,
            max_value=max_value,
            default_value=default_value,
            adjustment_sensitivity=sensitivity,
            adjustment_smoothing=smoothing,
        )
        self._consecutive_adjustments[name] = 0
        logger.debug(f"Registered parameter: {name}={default_value}")

    def get_current_params(self) -> dict[str, float]:
        """Get current parameter values."""
        return {name: param.current_value for name, param in self._parameters.items()}

    def set_params(self, params: dict[str, float]) -> None:
        """Set parameter values directly."""
        for name, value in params.items():
            if name in self._parameters:
                param = self._parameters[name]
                param.current_value = np.clip(value, param.min_value, param.max_value)

    def adjust_for_volatility(
        self,
        current_volatility: float,
        custom_adjustments: dict[str, Callable] | None = None,
    ) -> AdjustmentResult:
        """Adjust parameters based on market volatility.

        Args:
            current_volatility: Current market volatility (as decimal)
            custom_adjustments: Optional custom adjustment functions

        Returns:
            AdjustmentResult with adjusted parameters
        """
        self._current_period += 1
        old_params = self.get_current_params()
        adjustments: list[ParameterAdjustment] = []
        warnings: list[str] = []

        if current_volatility > self.config.volatility_high_threshold:
            regime = "high_volatility"
        elif current_volatility < self.config.volatility_low_threshold:
            regime = "low_volatility"
        else:
            regime = "normal"

        for name, param in self._parameters.items():
            if name in self._consecutive_adjustments:
                if self._consecutive_adjustments[name] >= 3:
                    warnings.append(f"Skipping {name}: too many consecutive adjustments")
                    continue

            adjustment_factor = self._calculate_volatility_adjustment(
                name, current_volatility, regime
            )

            if custom_adjustments and name in custom_adjustments:
                adjustment_factor = custom_adjustments[name](current_volatility, adjustment_factor)

            if abs(adjustment_factor - 1.0) < 0.01:
                continue

            new_value = self._apply_adjustment(param, adjustment_factor)

            if new_value != param.current_value:
                adj = self._create_adjustment(
                    param,
                    new_value,
                    AdjustmentTrigger.VOLATILITY,
                    f"Volatility adjustment ({regime}): {current_volatility:.2%}",
                )
                adjustments.append(adj)
                param.current_value = new_value
                self._consecutive_adjustments[name] = self._consecutive_adjustments.get(name, 0) + 1
            else:
                self._consecutive_adjustments[name] = 0

        new_params = self.get_current_params()

        if adjustments:
            self._adjustment_history.extend(adjustments)

        return AdjustmentResult(
            adjustments=adjustments,
            old_params=old_params,
            new_params=new_params,
            confidence=self._calculate_confidence(),
            summary=f"Applied {len(adjustments)} volatility-based adjustments ({regime})",
            warnings=warnings,
        )

    def adjust_for_performance(
        self,
        recent_trades: list[dict[str, Any]],
        performance_metrics: dict[str, float] | None = None,
    ) -> AdjustmentResult:
        """Adjust parameters based on recent performance.

        Args:
            recent_trades: List of recent trade results
            performance_metrics: Optional pre-calculated metrics

        Returns:
            AdjustmentResult with adjusted parameters
        """
        self._current_period += 1
        old_params = self.get_current_params()
        adjustments: list[ParameterAdjustment] = []
        warnings: list[str] = []

        if len(recent_trades) < self.config.min_trades_for_adjustment:
            return AdjustmentResult(
                adjustments=[],
                old_params=old_params,
                new_params=old_params,
                confidence=0.0,
                summary="Insufficient trades for adjustment",
                warnings=["Need at least {self.config.min_trades_for_adjustment} trades"],
            )

        if performance_metrics is None:
            performance_metrics = self._calculate_performance_metrics(recent_trades)

        self._performance_metrics = performance_metrics
        self._trade_history.extend(recent_trades)

        for name, param in self._parameters.items():
            adjustment_factor = self._calculate_performance_adjustment(name, performance_metrics)

            if abs(adjustment_factor - 1.0) < 0.01:
                continue

            new_value = self._apply_adjustment(param, adjustment_factor)

            if new_value != param.current_value:
                adj = self._create_adjustment(
                    param,
                    new_value,
                    AdjustmentTrigger.PERFORMANCE,
                    (
                        f"Performance adjustment: "
                        f"win_rate={performance_metrics.get('win_rate', 0):.1%}"
                    ),
                )
                adjustments.append(adj)
                param.current_value = new_value

        new_params = self.get_current_params()

        if adjustments:
            self._adjustment_history.extend(adjustments)

        return AdjustmentResult(
            adjustments=adjustments,
            old_params=old_params,
            new_params=new_params,
            confidence=self._calculate_confidence(),
            summary=f"Applied {len(adjustments)} performance-based adjustments",
            warnings=warnings,
        )

    def adjust_for_degradation(
        self,
        degradation_level: str,
        degradation_types: list[str],
    ) -> AdjustmentResult:
        """Adjust parameters in response to strategy degradation.

        Args:
            degradation_level: Level of degradation (none/minimal/moderate/severe/critical)
            degradation_types: Types of degradation detected

        Returns:
            AdjustmentResult with adjusted parameters
        """
        self._current_period += 1
        old_params = self.get_current_params()
        adjustments: list[ParameterAdjustment] = []
        warnings: list[str] = []

        adjustment_multipliers = {
            "none": 1.0,
            "minimal": 0.95,
            "moderate": 0.85,
            "severe": 0.70,
            "critical": 0.50,
        }

        base_factor = adjustment_multipliers.get(degradation_level, 1.0)

        if base_factor >= 1.0:
            return AdjustmentResult(
                adjustments=[],
                old_params=old_params,
                new_params=old_params,
                confidence=0.5,
                summary="No degradation detected, no adjustments needed",
                warnings=[],
            )

        for name, param in self._parameters.items():
            param_factor = base_factor

            if "stop_loss" in name.lower():
                if "volatility_increase" in degradation_types:
                    param_factor *= 1.1
                if "drawdown_expansion" in degradation_types:
                    param_factor *= 0.9

            if "position" in name.lower() or "size" in name.lower():
                param_factor = min(param_factor, 0.7)

            if "take_profit" in name.lower():
                if "win_rate_decay" in degradation_types:
                    param_factor *= 0.9

            new_value = self._apply_adjustment(param, param_factor)

            if new_value != param.current_value:
                adj = self._create_adjustment(
                    param,
                    new_value,
                    AdjustmentTrigger.DEGRADATION,
                    f"Degradation response ({degradation_level})",
                )
                adjustments.append(adj)
                param.current_value = new_value

        new_params = self.get_current_params()

        if adjustments:
            self._adjustment_history.extend(adjustments)

        if degradation_level in ["severe", "critical"]:
            warnings.append(f"High degradation level: {degradation_level}")
            warnings.append("Consider pausing strategy or manual intervention")

        return AdjustmentResult(
            adjustments=adjustments,
            old_params=old_params,
            new_params=new_params,
            confidence=0.8 if degradation_level in ["severe", "critical"] else 0.6,
            summary=f"Applied {len(adjustments)} degradation response adjustments",
            warnings=warnings,
        )

    def adjust_for_risk(
        self,
        current_drawdown: float,
        max_drawdown: float,
        current_exposure: float,
        max_exposure: float,
    ) -> AdjustmentResult:
        """Adjust parameters for risk management.

        Args:
            current_drawdown: Current drawdown (as decimal)
            max_drawdown: Maximum allowed drawdown
            current_exposure: Current total exposure
            max_exposure: Maximum allowed exposure

        Returns:
            AdjustmentResult with adjusted parameters
        """
        self._current_period += 1
        old_params = self.get_current_params()
        adjustments: list[ParameterAdjustment] = []
        warnings: list[str] = []

        drawdown_ratio = current_drawdown / max_drawdown if max_drawdown > 0 else 0
        exposure_ratio = current_exposure / max_exposure if max_exposure > 0 else 0

        for name, param in self._parameters.items():
            risk_factor = 1.0

            if "position" in name.lower() or "size" in name.lower():
                if drawdown_ratio > 0.7:
                    risk_factor = 0.5
                    warnings.append("High drawdown: reducing position sizes")
                elif drawdown_ratio > 0.5:
                    risk_factor = 0.7

                if exposure_ratio > 0.8:
                    risk_factor = min(risk_factor, 0.6)

            if "stop_loss" in name.lower():
                if drawdown_ratio > 0.6:
                    risk_factor = 0.85

            if "take_profit" in name.lower():
                if drawdown_ratio > 0.6:
                    risk_factor = 0.9

            if abs(risk_factor - 1.0) < 0.01:
                continue

            new_value = self._apply_adjustment(param, risk_factor)

            if new_value != param.current_value:
                adj = self._create_adjustment(
                    param,
                    new_value,
                    AdjustmentTrigger.RISK_MANAGEMENT,
                    f"Risk adjustment: DD={current_drawdown:.1%}, Exp={current_exposure:.1%}",
                )
                adjustments.append(adj)
                param.current_value = new_value

        new_params = self.get_current_params()

        if adjustments:
            self._adjustment_history.extend(adjustments)

        return AdjustmentResult(
            adjustments=adjustments,
            old_params=old_params,
            new_params=new_params,
            confidence=0.9,
            summary=f"Applied {len(adjustments)} risk management adjustments",
            warnings=warnings,
        )

    def _calculate_volatility_adjustment(
        self,
        param_name: str,
        volatility: float,
        regime: str,
    ) -> float:
        """Calculate adjustment factor for a parameter based on volatility."""

        if "stop_loss" in param_name.lower():
            if regime == "high_volatility":
                return 1.2
            elif regime == "low_volatility":
                return 0.85
            return 1.0

        if "take_profit" in param_name.lower():
            if regime == "high_volatility":
                return 1.3
            elif regime == "low_volatility":
                return 0.9
            return 1.0

        if "position" in param_name.lower() or "size" in param_name.lower():
            if regime == "high_volatility":
                return 0.7
            elif regime == "low_volatility":
                return 1.1
            return 1.0

        if "slippage" in param_name.lower():
            if regime == "high_volatility":
                return 1.3
            return 1.0

        return 1.0

    def _calculate_performance_adjustment(
        self,
        param_name: str,
        metrics: dict[str, float],
    ) -> float:
        """Calculate adjustment factor based on performance metrics."""
        win_rate = metrics.get("win_rate", 0.5)
        sharpe = metrics.get("sharpe", 0)
        profit_factor = metrics.get("profit_factor", 1.0)

        factor = 1.0

        if "stop_loss" in param_name.lower():
            if win_rate < 0.4:
                factor = 0.9
            elif win_rate > 0.6:
                factor = 1.1

        if "take_profit" in param_name.lower():
            if profit_factor < 1.5:
                factor = 0.85
            elif profit_factor > 2.5:
                factor = 1.15

        if "position" in param_name.lower() or "size" in param_name.lower():
            if sharpe < 1.0:
                factor = 0.8
            elif sharpe > 2.0:
                factor = 1.1

        return factor

    def _calculate_performance_metrics(
        self,
        trades: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Calculate performance metrics from trades."""
        if not trades:
            return {}

        pnls = [t.get("pnl_pct", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0

        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0

        profit_factor = (
            min(sum(wins) / sum(losses), 999.0)
            if losses and sum(losses) != 0
            else (999.0 if wins else 0.0)
        )

        mean_pnl = np.mean(pnls) if pnls else 0
        std_pnl = np.std(pnls) if len(pnls) > 1 else 1
        sharpe = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0

        return {
            "win_rate": win_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "total_trades": len(trades),
        }

    def _apply_adjustment(
        self,
        param: AdaptiveParameter,
        factor: float,
    ) -> float:
        """Apply an adjustment factor to a parameter with smoothing."""
        max_adj = self.config.max_adjustment_pct
        factor = np.clip(factor, 1 - max_adj, 1 + max_adj)

        raw_new = param.current_value * factor

        smoothing = param.adjustment_smoothing * self.config.smoothing_factor
        smoothed = param.current_value * smoothing + raw_new * (1 - smoothing)

        return round(np.clip(smoothed, param.min_value, param.max_value), 4)

    def _create_adjustment(
        self,
        param: AdaptiveParameter,
        new_value: float,
        trigger: AdjustmentTrigger,
        reason: str,
    ) -> ParameterAdjustment:
        """Create a parameter adjustment record."""
        old_value = param.current_value
        change_pct = (new_value - old_value) / old_value * 100 if old_value != 0 else 0

        if change_pct > 0:
            direction = ParameterDirection.INCREASE
        elif change_pct < 0:
            direction = ParameterDirection.DECREASE
        else:
            direction = ParameterDirection.MAINTAIN

        return ParameterAdjustment(
            parameter=param.name,
            old_value=old_value,
            new_value=new_value,
            change_pct=round(change_pct, 2),
            direction=direction,
            trigger=trigger,
            reason=reason,
        )

    def _calculate_confidence(self) -> float:
        """Calculate confidence in adjustments."""
        base_confidence = 0.7

        trade_count = len(self._trade_history)
        if trade_count >= self.config.performance_lookback:
            base_confidence += 0.1
        if trade_count >= self.config.performance_lookback * 2:
            base_confidence += 0.1

        recent_adjustments = len(
            [
                a
                for a in self._adjustment_history
                if (datetime.utcnow() - datetime.fromisoformat(a.timestamp)).total_seconds() < 3600
            ]
        )
        if recent_adjustments > 10:
            base_confidence -= 0.1

        return min(1.0, max(0.3, base_confidence))

    def get_adjustment_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent adjustment history."""
        history = list(self._adjustment_history)[-limit:]
        return [a.to_dict() for a in history]

    def get_parameter_stats(self) -> dict[str, Any]:
        """Get statistics about parameter adjustments."""
        stats = {}

        for name, param in self._parameters.items():
            param_adjustments = [a for a in self._adjustment_history if a.parameter == name]

            increases = sum(
                1 for a in param_adjustments if a.direction == ParameterDirection.INCREASE
            )
            decreases = sum(
                1 for a in param_adjustments if a.direction == ParameterDirection.DECREASE
            )

            stats[name] = {
                "current_value": param.current_value,
                "min_value": param.min_value,
                "max_value": param.max_value,
                "default_value": param.default_value,
                "total_adjustments": len(param_adjustments),
                "increases": increases,
                "decreases": decreases,
            }

        return stats

    def reset_to_defaults(self) -> AdjustmentResult:
        """Reset all parameters to defaults."""
        old_params = self.get_current_params()
        adjustments: list[ParameterAdjustment] = []

        for _name, param in self._parameters.items():
            if param.current_value != param.default_value:
                adj = self._create_adjustment(
                    param, param.default_value, AdjustmentTrigger.MANUAL, "Reset to default value"
                )
                adjustments.append(adj)
                param.current_value = param.default_value

        new_params = self.get_current_params()

        return AdjustmentResult(
            adjustments=adjustments,
            old_params=old_params,
            new_params=new_params,
            confidence=1.0,
            summary=f"Reset {len(adjustments)} parameters to defaults",
        )


def create_default_adjuster() -> AdaptiveParameterAdjuster:
    """Create an adjuster with default trading parameters."""
    adjuster = AdaptiveParameterAdjuster()

    adjuster.register_parameter("stop_loss_pct", 8.0, 3.0, 20.0, sensitivity=0.8)
    adjuster.register_parameter("take_profit_pct", 20.0, 5.0, 60.0, sensitivity=0.8)
    adjuster.register_parameter("max_position_sol", 0.05, 0.01, 0.20, sensitivity=0.6)
    adjuster.register_parameter("min_liquidity_usd", 10000, 5000, 50000, sensitivity=0.4)
    adjuster.register_parameter("max_risk_score", 0.40, 0.20, 0.70, sensitivity=0.5)
    adjuster.register_parameter("slippage_bps", 300, 100, 600, sensitivity=0.3)

    return adjuster


def adjust_parameters_for_conditions(
    current_params: dict[str, float],
    volatility: float,
    win_rate: float,
    drawdown: float,
) -> dict[str, float]:
    """Simple function to adjust parameters for current conditions.

    Args:
        current_params: Current parameter values
        volatility: Current volatility (as decimal)
        win_rate: Recent win rate (0-1)
        drawdown: Current drawdown (as decimal)

    Returns:
        Adjusted parameter values
    """
    adjuster = create_default_adjuster()
    adjuster.set_params(current_params)

    if volatility > 0.03:
        adjuster.adjust_for_volatility(volatility)

    if drawdown > 0.15:
        adjuster.adjust_for_risk(drawdown, 0.25, 0.5, 1.0)

    return adjuster.get_current_params()
