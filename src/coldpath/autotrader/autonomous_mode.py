"""
Autonomous Mode - Self-tuning configuration for AutoTrader.

This module integrates with the AutoTraderCoordinator to provide:
- Daily P&L target tracking (0.8-1% target)
- Automatic mode switching (AGGRESSIVE/BALANCED/CONSERVATIVE)
- Self-tuning of confidence thresholds and position sizes
- Rolling statistics for continuous improvement
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coldpath.autotrader.coordinator import AutoTraderCoordinator

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode based on daily P&L performance."""

    AGGRESSIVE = "aggressive"  # Below target - need more profit
    BALANCED = "balanced"  # On target - stay the course
    CONSERVATIVE = "conservative"  # Above max - protect gains
    PAUSED = "paused"  # Hit daily loss limit or safety trigger


@dataclass
class TuningAdjustment:
    """A single tuning adjustment made by the autonomous mode."""

    timestamp: datetime
    parameter: str
    old_value: Any
    new_value: Any
    reason: str
    trigger_metrics: dict[str, float]


@dataclass
class AutonomousModeConfig:
    """Configuration for autonomous mode."""

    # Daily P&L targets (as percentage of capital)
    target_min_pct: float = 0.6  # Minimum acceptable
    target_optimal_pct: float = 0.8  # Target
    target_max_pct: float = 1.2  # Reduce risk above this

    # Position size multipliers by mode
    multiplier_aggressive: float = 1.3
    multiplier_balanced: float = 1.0
    multiplier_conservative: float = 0.6

    # Tuning settings
    tuning_cooldown_hours: float = 4.0
    min_trades_for_tuning: int = 30

    # Adjustment limits
    max_confidence_change: float = 0.05
    max_kelly_change: float = 0.05
    max_position_change_pct: float = 20.0

    # Rolling stats
    rolling_trade_window: int = 50


class AutonomousMode:
    """Self-tuning autonomous trading mode.

    Integrates with AutoTraderCoordinator to:
    1. Track daily P&L against targets
    2. Switch between AGGRESSIVE/BALANCED/CONSERVATIVE modes
    3. Auto-tune parameters based on performance
    4. Record adjustments for audit trail

    Usage:
        autonomous = AutonomousMode(coordinator)

        # After each trade, check for tuning
        adjustments = await autonomous.auto_tune_if_needed()

        # Get current mode and multiplier
        mode = autonomous.get_mode()
        multiplier = autonomous.get_position_size_multiplier()
    """

    def __init__(
        self,
        coordinator: "AutoTraderCoordinator",
        config: AutonomousModeConfig | None = None,
    ):
        self.coordinator = coordinator
        self.config = config or AutonomousModeConfig()

        # Tuning state
        self._tuning_history: deque = deque(maxlen=100)
        self._last_tuning_time: datetime | None = None

        # Daily tracking
        self._daily_start_pnl: float = 0.0
        self._current_date = datetime.now().date()

        # Rolling trade stats
        self._trade_outcomes: deque = deque(maxlen=self.config.rolling_trade_window)

    def reset_if_new_day(self, current_pnl_sol: float) -> bool:
        """Reset tracking at start of new day."""
        today = datetime.now().date()
        if today != self._current_date:
            self._daily_start_pnl = current_pnl_sol
            self._current_date = today
            logger.info(f"New trading day: {today}, starting P&L: {current_pnl_sol:.6f} SOL")
            return True
        return False

    def record_trade_outcome(
        self,
        pnl_pct: float,
        pnl_sol: float,
        is_win: bool,
    ):
        """Record a trade outcome for rolling statistics."""
        self._trade_outcomes.append(
            {
                "timestamp": datetime.now(),
                "pnl_pct": pnl_pct,
                "pnl_sol": pnl_sol,
                "is_win": is_win,
            }
        )

    def get_rolling_stats(self) -> dict[str, float]:
        """Calculate rolling statistics from recent trades."""
        if not self._trade_outcomes:
            return {
                "count": 0,
                "win_rate": 0.0,
                "avg_pnl_pct": 0.0,
                "total_pnl_sol": 0.0,
            }

        outcomes = list(self._trade_outcomes)
        wins = sum(1 for o in outcomes if o["is_win"])

        return {
            "count": len(outcomes),
            "win_rate": wins / len(outcomes) if outcomes else 0.0,
            "avg_pnl_pct": sum(o["pnl_pct"] for o in outcomes) / len(outcomes),
            "total_pnl_sol": sum(o["pnl_sol"] for o in outcomes),
        }

    def get_daily_pnl_pct(self) -> float:
        """Calculate current daily P&L as percentage of capital."""
        # Estimate capital from config
        # In production, this should track actual capital
        estimated_capital = 1.0  # SOL default

        current_pnl = self.coordinator.daily_metrics.total_pnl_sol
        daily_pnl = current_pnl - self._daily_start_pnl

        return (daily_pnl / estimated_capital) * 100

    def get_mode(self) -> TradingMode:
        """Get current trading mode based on daily P&L."""
        # Check if paused
        if self.coordinator.state.value == "paused":
            return TradingMode.PAUSED

        pct = self.get_daily_pnl_pct()

        if pct < self.config.target_min_pct:
            return TradingMode.AGGRESSIVE
        elif pct > self.config.target_max_pct:
            return TradingMode.CONSERVATIVE
        else:
            return TradingMode.BALANCED

    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on trading mode."""
        mode = self.get_mode()

        if mode == TradingMode.AGGRESSIVE:
            return self.config.multiplier_aggressive
        elif mode == TradingMode.CONSERVATIVE:
            return self.config.multiplier_conservative
        else:
            return self.config.multiplier_balanced

    def should_trade_more(self) -> bool:
        """Check if we should increase trading activity."""
        pct = self.get_daily_pnl_pct()
        trade_count = self.coordinator.daily_metrics.trade_count

        # Below target and have room for more trades
        if pct < self.config.target_min_pct and trade_count < 50:
            return True

        return False

    def should_reduce_risk(self) -> bool:
        """Check if we should reduce risk exposure."""
        return self.get_mode() == TradingMode.CONSERVATIVE

    def is_on_target(self) -> bool:
        """Check if daily P&L is within target range."""
        pct = self.get_daily_pnl_pct()
        return self.config.target_min_pct <= pct <= self.config.target_max_pct

    async def auto_tune_if_needed(self) -> dict[str, Any] | None:
        """Check if tuning is needed and perform it.

        Returns:
            Dictionary of adjustments made, or None if no tuning needed
        """
        now = datetime.now()

        # Check cooldown
        if self._last_tuning_time:
            hours_since = (now - self._last_tuning_time).total_seconds() / 3600
            if hours_since < self.config.tuning_cooldown_hours:
                return None

        # Need minimum trades for statistical significance
        total_trades = (
            self.coordinator.daily_metrics.win_count + self.coordinator.daily_metrics.loss_count
        )
        if total_trades < self.config.min_trades_for_tuning:
            return None

        tuning_needed = False
        adjustments: dict[str, Any] = {
            "action": None,
            "changes": {},
            "reason": None,
        }

        # Get current metrics
        daily_pnl_pct = self.get_daily_pnl_pct()
        rolling = self.get_rolling_stats()

        # Decision logic

        # Case 1: Below minimum P&L target
        if daily_pnl_pct < self.config.target_min_pct:
            tuning_needed = True
            adjustments["action"] = "increase_aggression"
            adjustments["reason"] = (
                f"Daily P&L {daily_pnl_pct:.2f}% below minimum {self.config.target_min_pct}%"
            )

            # Lower confidence threshold for more signals
            new_conf = max(
                0.45,
                self.coordinator.config.min_confidence_to_trade - self.config.max_confidence_change,
            )
            adjustments["changes"]["min_confidence_to_trade"] = new_conf

            # Increase Kelly fraction
            new_kelly = min(
                0.50, self.coordinator.config.kelly_fraction + self.config.max_kelly_change
            )
            adjustments["changes"]["kelly_fraction"] = new_kelly

        # Case 2: Win rate too low
        elif rolling["count"] >= 20 and rolling["win_rate"] < 0.40:
            tuning_needed = True
            adjustments["action"] = "tighten_filters"
            adjustments["reason"] = (
                f"Win rate {rolling['win_rate'] * 100:.1f}% too low "
                f"(last {rolling['count']} trades)"
            )

            # Raise confidence threshold
            new_conf = min(
                0.65,
                self.coordinator.config.min_confidence_to_trade + self.config.max_confidence_change,
            )
            adjustments["changes"]["min_confidence_to_trade"] = new_conf

            # Lower max fraud score tolerance
            new_fraud = max(0.30, self.coordinator.config.max_fraud_score - 0.05)
            adjustments["changes"]["max_fraud_score"] = new_fraud

        # Case 3: Win rate excellent - can take more risk
        elif (
            rolling["count"] >= 20
            and rolling["win_rate"] > 0.60
            and daily_pnl_pct < self.config.target_optimal_pct
        ):
            tuning_needed = True
            adjustments["action"] = "increase_size"
            adjustments["reason"] = (
                f"Win rate {rolling['win_rate'] * 100:.1f}% excellent, increasing size"
            )

            # Increase max position size
            new_pos = min(0.10, self.coordinator.config.max_position_sol * 1.2)
            adjustments["changes"]["max_position_sol"] = new_pos

        # Case 4: Above max target - reduce risk
        elif daily_pnl_pct > self.config.target_max_pct:
            tuning_needed = True
            adjustments["action"] = "reduce_risk"
            adjustments["reason"] = (
                f"Daily P&L {daily_pnl_pct:.2f}% above max {self.config.target_max_pct}%"
            )

            # Reduce position size
            new_pos = self.coordinator.config.max_position_sol * 0.8
            adjustments["changes"]["max_position_sol"] = new_pos

            # Raise confidence threshold
            new_conf = min(0.65, self.coordinator.config.min_confidence_to_trade + 0.03)
            adjustments["changes"]["min_confidence_to_trade"] = new_conf

        if not tuning_needed:
            return None

        # Apply adjustments
        self._apply_adjustments(adjustments)
        self._last_tuning_time = now

        return adjustments

    def _apply_adjustments(self, adjustments: dict[str, Any]):
        """Apply tuning adjustments to coordinator config."""
        action = adjustments.get("action")
        changes = adjustments.get("changes", {})

        for key, new_value in changes.items():
            if hasattr(self.coordinator.config, key):
                old_value = getattr(self.coordinator.config, key)
                setattr(self.coordinator.config, key, new_value)

                logger.info(f"🔄 Auto-tune ({action}): {key} = {old_value:.4f} → {new_value:.4f}")

                # Record in history
                self._tuning_history.append(
                    TuningAdjustment(
                        timestamp=datetime.now(),
                        parameter=key,
                        old_value=old_value,
                        new_value=new_value,
                        reason=adjustments.get("reason", ""),
                        trigger_metrics={
                            "daily_pnl_pct": self.get_daily_pnl_pct(),
                            "win_rate": self.coordinator.daily_metrics.win_rate,
                            "trade_count": self.coordinator.daily_metrics.trade_count,
                        },
                    )
                )

    def get_summary(self) -> dict[str, Any]:
        """Get summary of autonomous mode status."""
        return {
            "mode": self.get_mode().value,
            "position_multiplier": self.get_position_size_multiplier(),
            "daily_pnl_pct": round(self.get_daily_pnl_pct(), 2),
            "target_min_pct": self.config.target_min_pct,
            "target_optimal_pct": self.config.target_optimal_pct,
            "target_max_pct": self.config.target_max_pct,
            "on_target": self.is_on_target(),
            "should_trade_more": self.should_trade_more(),
            "should_reduce_risk": self.should_reduce_risk(),
            "rolling_stats": self.get_rolling_stats(),
            "tuning_history_count": len(self._tuning_history),
            "last_tuning": self._last_tuning_time.isoformat() if self._last_tuning_time else None,
        }

    def get_tuning_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent tuning history for audit."""
        history = list(self._tuning_history)[-limit:]
        return [
            {
                "timestamp": adj.timestamp.isoformat(),
                "parameter": adj.parameter,
                "old_value": adj.old_value,
                "new_value": adj.new_value,
                "reason": adj.reason,
                "trigger_metrics": adj.trigger_metrics,
            }
            for adj in history
        ]
