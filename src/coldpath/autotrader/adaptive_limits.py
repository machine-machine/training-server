"""
Adaptive Limits Manager - Dynamic trade limit scaling based on performance.

Automatically scales trading limits UP when performance is strong and DOWN
when performance is weak, with hard floors and ceilings for safety.

🎯 PROFITABILITY-FIRST PHILOSOPHY (nasr, Feb 2026):
═══════════════════════════════════════════════════
"Why stop trading when making money?"

The engine should CONTINUE trading as long as:
  1. Win rate is healthy (>45%)
  2. Daily P&L is positive
  3. Not hitting daily loss limit (the REAL safety)

Trade count limits are RAISED to 1000+ to let profitable sessions run.
The get_profitability_aware_ceiling() method can DYNAMICALLY raise the
ceiling even higher when the engine is printing money.

Key Features:
- Rolling win rate monitoring (last N trades)
- Dynamic limit scaling with configurable thresholds
- Hard safety floors and ceilings (now: 1000 ceiling, not 300)
- Regime-aware adjustments
- Action logging for audit trail
- 🆕 Profitability-aware ceiling: scale up when making money!
"""

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LimitAction(Enum):
    """Actions the adaptive manager can take."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HOLD = "hold"
    EMERGENCY_REDUCE = "emergency_reduce"


@dataclass
class AdaptiveLimitConfig:
    """Configuration for adaptive limit management."""

    # Window for rolling calculations
    rolling_window_size: int = 30  # Last N trades to consider

    # Scaling thresholds
    scale_up_win_rate: float = 0.58  # Win rate to scale UP
    scale_down_win_rate: float = 0.45  # Win rate to scale DOWN
    emergency_win_rate: float = 0.35  # Emergency reduce threshold

    # Scaling factors
    scale_up_factor: float = 1.15  # Multiply limits by this when scaling up
    scale_down_factor: float = 0.85  # Multiply limits by this when scaling down
    emergency_factor: float = 0.50  # Emergency reduce factor

    # Cooldown between adjustments
    min_adjustment_interval_seconds: int = 300  # 5 minutes between adjustments
    scale_up_cooldown_seconds: int = 600  # 10 minutes after scaling up
    scale_down_cooldown_seconds: int = 180  # 3 minutes after scaling down

    # Minimum trades before first adjustment
    min_trades_before_adjustment: int = 15

    # Enable/disable autonomous operation
    autonomous_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "rolling_window_size": self.rolling_window_size,
            "scale_up_win_rate": self.scale_up_win_rate,
            "scale_down_win_rate": self.scale_down_win_rate,
            "emergency_win_rate": self.emergency_win_rate,
            "scale_up_factor": self.scale_up_factor,
            "scale_down_factor": self.scale_down_factor,
            "emergency_factor": self.emergency_factor,
            "min_adjustment_interval_seconds": self.min_adjustment_interval_seconds,
            "scale_up_cooldown_seconds": self.scale_up_cooldown_seconds,
            "scale_down_cooldown_seconds": self.scale_down_cooldown_seconds,
            "min_trades_before_adjustment": self.min_trades_before_adjustment,
            "autonomous_enabled": self.autonomous_enabled,
        }


@dataclass
class LimitBounds:
    """Hard bounds for each limit parameter - EXTENDED for 0.8-1% daily P&L target.

    🎯 PROFITABILITY-FIRST PHILOSOPHY (nasr, 2026-02):
    ─────────────────────────────────────────────────────
    Why stop trading when making money? The engine should CONTINUE as long as:
    1. Win rate is healthy (>45%)
    2. Daily P&L is positive
    3. Not hitting daily loss limit

    The ceiling below is now HIGH (1000) to let the adaptive system work.
    The REAL safety is the daily loss limit, not an arbitrary trade count.

    TODO (Future Enhancement): Implement Profitability-Aware Ceiling
    - If daily_pnl > 0 AND win_rate > 0.50: ceiling = unlimited (or 5000)
    - If daily_pnl < 0 OR win_rate < 0.40: ceiling = 50 (slow down)
    - This makes the ceiling DYNAMIC based on actual performance

    For now, 1000 is a generous ceiling that shouldn't be hit in normal operation.
    """

    # max_daily_trades bounds - RAISED to let profitable trading continue
    min_daily_trades: int = 20
    max_daily_trades: int = 1000  # Raised from 300 -> Why stop a money printer?

    # max_concurrent_positions bounds - EXTENDED
    min_concurrent_positions: int = 2
    max_concurrent_positions: int = 20  # Was 10 -> 2x for more throughput

    # max_position_sol bounds - EXTENDED
    min_position_sol: float = 0.02
    max_position_sol: float = 0.25  # Was 0.20 -> allow larger positions

    # max_daily_volume_sol bounds - EXTENDED
    min_daily_volume_sol: float = 0.20
    max_daily_volume_sol: float = 10.0  # Was 5.0 -> 2x for extended runs

    # max_daily_loss_sol bounds (scaled with volume) - EXTENDED
    min_daily_loss_sol: float = 0.10
    max_daily_loss_sol: float = 1.0  # Was 0.50 -> 2x for larger accounts

    # min_confidence_to_trade bounds
    min_confidence_floor: float = 0.45
    min_confidence_ceiling: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_daily_trades": self.min_daily_trades,
            "max_daily_trades": self.max_daily_trades,
            "min_concurrent_positions": self.min_concurrent_positions,
            "max_concurrent_positions": self.max_concurrent_positions,
            "min_position_sol": self.min_position_sol,
            "max_position_sol": self.max_position_sol,
            "min_daily_volume_sol": self.min_daily_volume_sol,
            "max_daily_volume_sol": self.max_daily_volume_sol,
            "min_daily_loss_sol": self.min_daily_loss_sol,
            "max_daily_loss_sol": self.max_daily_loss_sol,
            "min_confidence_floor": self.min_confidence_floor,
            "min_confidence_ceiling": self.min_confidence_ceiling,
        }


@dataclass
class AdjustmentRecord:
    """Record of a limit adjustment for audit trail."""

    timestamp: datetime
    action: LimitAction
    trigger_win_rate: float
    old_values: dict[str, Any]
    new_values: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "trigger_win_rate": self.trigger_win_rate,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "reason": self.reason,
        }


class AdaptiveLimitsManager:
    """Manages dynamic scaling of trading limits based on performance.

    Usage:
        manager = AdaptiveLimitsManager()

        # Record each trade outcome
        manager.record_outcome(is_win=True, pnl_pct=5.2)

        # Check if adjustments are needed
        action = manager.evaluate(current_config)

        if action != LimitAction.HOLD:
            new_config = manager.apply_adjustment(current_config, action)
    """

    def __init__(
        self,
        config: AdaptiveLimitConfig | None = None,
        bounds: LimitBounds | None = None,
    ):
        self.config = config or AdaptiveLimitConfig()
        self.bounds = bounds or LimitBounds()

        # Rolling window of trade outcomes
        self._outcomes: deque = deque(maxlen=self.config.rolling_window_size)

        # Track total trades for minimum threshold
        self._total_trades: int = 0

        # Last adjustment timestamp
        self._last_adjustment: datetime | None = None
        self._last_action: LimitAction | None = None

        # Adjustment history for audit
        self._adjustment_history: list[AdjustmentRecord] = []
        self._max_history: int = 100

        # Current multiplier (starts at 1.0)
        self._current_multiplier: float = 1.0

    def record_outcome(self, is_win: bool, pnl_pct: float = 0.0) -> None:
        """Record a trade outcome for rolling calculations.

        Args:
            is_win: Whether the trade was profitable
            pnl_pct: P&L as percentage (for weighted calculations)
        """
        self._outcomes.append(
            {
                "is_win": is_win,
                "pnl_pct": pnl_pct,
                "timestamp": datetime.now(),
            }
        )
        self._total_trades += 1

    def get_rolling_win_rate(self) -> float:
        """Calculate rolling win rate from recent outcomes."""
        if not self._outcomes:
            return 0.5  # Default to neutral

        wins = sum(1 for o in self._outcomes if o["is_win"])
        return wins / len(self._outcomes)

    def get_rolling_pnl_avg(self) -> float:
        """Calculate average P&L from recent outcomes."""
        if not self._outcomes:
            return 0.0

        return sum(o["pnl_pct"] for o in self._outcomes) / len(self._outcomes)

    def get_sample_size(self) -> int:
        """Get number of outcomes in rolling window."""
        return len(self._outcomes)

    def get_profitability_aware_ceiling(
        self,
        current_daily_pnl_sol: float = 0.0,
        current_trade_count: int = 0,
    ) -> int:
        """Calculate a dynamic trade ceiling based on actual profitability.

        🎯 PROFITABILITY-FIRST LOGIC (nasr's insight):
        ─────────────────────────────────────────────────
        Why stop at an arbitrary number when making money?

        Returns a dynamic ceiling based on:
        - Current win rate (healthy = higher ceiling)
        - Daily P&L (profitable = higher ceiling)

        Args:
            current_daily_pnl_sol: Today's P&L in SOL
            current_trade_count: Trades executed today

        Returns:
            Dynamic ceiling for max_daily_trades
        """
        win_rate = self.get_rolling_win_rate()
        self.get_rolling_pnl_avg()
        sample_size = self.get_sample_size()

        # Not enough data? Use base ceiling
        if sample_size < 10:
            return self.bounds.max_daily_trades

        # 🚀 PROFITABLE + HEALTHY WIN RATE = UNLIMITED UPSIDE
        if current_daily_pnl_sol > 0 and win_rate > 0.50:
            # Making money and winning > 50%? Let it ride!
            # Scale ceiling based on profitability
            profit_multiplier = 1.0 + min(2.0, math.log1p(max(0, current_daily_pnl_sol) * 5))
            dynamic_ceiling = int(self.bounds.max_daily_trades * profit_multiplier)
            logger.info(
                f"💰 Profitability boost! Win rate {win_rate:.1%}, "
                f"P&L {current_daily_pnl_sol:.4f} SOL -> ceiling raised to {dynamic_ceiling}"
            )
            return min(dynamic_ceiling, 2000)  # Cap at 2000 (was 5000)

        # 😐 BREAKING EVEN = NORMAL CEILING
        if win_rate >= 0.45:
            return self.bounds.max_daily_trades

        # 📉 LOSING OR LOW WIN RATE = CONSTRICT
        if win_rate < 0.40 or current_daily_pnl_sol < 0:
            constricted_ceiling = max(50, current_trade_count + 10)
            logger.warning(
                f"⚠️ Constricting ceiling: Win rate {win_rate:.1%}, "
                f"P&L {current_daily_pnl_sol:.4f} SOL -> ceiling reduced to {constricted_ceiling}"
            )
            return constricted_ceiling

        return self.bounds.max_daily_trades

    def evaluate(self, current_config: dict[str, Any]) -> LimitAction:
        """Evaluate whether limits should be adjusted.

        Args:
            current_config: Current AutoTraderConfig as dict

        Returns:
            LimitAction indicating what should happen
        """
        # Check if autonomous mode is enabled
        if not self.config.autonomous_enabled:
            return LimitAction.HOLD

        # Check minimum trades threshold
        if self._total_trades < self.config.min_trades_before_adjustment:
            logger.debug(
                f"Skipping adjustment: only {self._total_trades} trades "
                f"(need {self.config.min_trades_before_adjustment})"
            )
            return LimitAction.HOLD

        # Check cooldown
        if self._is_in_cooldown():
            logger.debug("Skipping adjustment: in cooldown period")
            return LimitAction.HOLD

        # Get rolling metrics
        win_rate = self.get_rolling_win_rate()
        sample_size = self.get_sample_size()

        # Need minimum sample
        if sample_size < 10:
            return LimitAction.HOLD

        # Check for emergency
        if win_rate < self.config.emergency_win_rate:
            logger.warning(
                f"🚨 EMERGENCY: Win rate {win_rate:.1%} below emergency threshold "
                f"{self.config.emergency_win_rate:.1%}"
            )
            return LimitAction.EMERGENCY_REDUCE

        # Check for scale down
        if win_rate < self.config.scale_down_win_rate:
            logger.info(
                f"📉 Scaling down: Win rate {win_rate:.1%} below threshold "
                f"{self.config.scale_down_win_rate:.1%}"
            )
            return LimitAction.SCALE_DOWN

        # Check for scale up
        if win_rate > self.config.scale_up_win_rate:
            # Also check that we're profitable on average
            avg_pnl = self.get_rolling_pnl_avg()
            if avg_pnl > 0:
                logger.info(
                    f"📈 Scaling up: Win rate {win_rate:.1%} above threshold "
                    f"{self.config.scale_up_win_rate:.1%}, avg P&L {avg_pnl:.2f}%"
                )
                return LimitAction.SCALE_UP

        return LimitAction.HOLD

    def _is_in_cooldown(self) -> bool:
        """Check if we're in cooldown from last adjustment."""
        if self._last_adjustment is None:
            return False

        elapsed = (datetime.now() - self._last_adjustment).total_seconds()

        # Different cooldowns for different actions
        if self._last_action == LimitAction.SCALE_UP:
            cooldown = self.config.scale_up_cooldown_seconds
        elif self._last_action == LimitAction.SCALE_DOWN:
            cooldown = self.config.scale_down_cooldown_seconds
        else:
            cooldown = self.config.min_adjustment_interval_seconds

        return elapsed < cooldown

    def apply_adjustment(
        self,
        current_config: dict[str, Any],
        action: LimitAction,
        reason: str | None = None,
        daily_pnl_sol: float = 0.0,
        current_trade_count: int = 0,
    ) -> dict[str, Any]:
        """Apply a limit adjustment to the configuration.

        Args:
            current_config: Current configuration dict
            action: The adjustment action to apply
            reason: Optional reason for the adjustment
            daily_pnl_sol: Today's P&L in SOL (for profitability-aware ceiling)
            current_trade_count: Trades executed today (for profitability-aware ceiling)

        Returns:
            New configuration dict with adjusted values
        """
        if action == LimitAction.HOLD:
            return current_config.copy()

        # Determine scaling factor
        if action == LimitAction.SCALE_UP:
            factor = self.config.scale_up_factor
            action_reason = reason or "Win rate above threshold"
        elif action == LimitAction.SCALE_DOWN:
            factor = self.config.scale_down_factor
            action_reason = reason or "Win rate below threshold"
        elif action == LimitAction.EMERGENCY_REDUCE:
            factor = self.config.emergency_factor
            action_reason = reason or "Emergency: win rate critically low"
        else:
            return current_config.copy()

        old_values = {}
        new_config = current_config.copy()

        # Scale max_daily_trades (integer) with PROFITABILITY-AWARE ceiling
        old_daily_trades = current_config.get("max_daily_trades", 35)
        new_daily_trades = int(round(old_daily_trades * factor))

        # 🎯 Use profitability-aware ceiling instead of static bound
        dynamic_ceiling = self.get_profitability_aware_ceiling(daily_pnl_sol, current_trade_count)
        new_daily_trades = self._clamp(
            new_daily_trades,
            self.bounds.min_daily_trades,
            dynamic_ceiling,  # Dynamic ceiling based on actual performance!
        )
        if new_daily_trades != old_daily_trades:
            old_values["max_daily_trades"] = old_daily_trades
            new_config["max_daily_trades"] = new_daily_trades

        # Scale max_concurrent_positions (integer)
        old_concurrent = current_config.get("max_concurrent_positions", 6)
        new_concurrent = int(round(old_concurrent * factor))
        new_concurrent = self._clamp(
            new_concurrent,
            self.bounds.min_concurrent_positions,
            self.bounds.max_concurrent_positions,
        )
        if new_concurrent != old_concurrent:
            old_values["max_concurrent_positions"] = old_concurrent
            new_config["max_concurrent_positions"] = new_concurrent

        # Scale max_position_sol (float)
        old_position = current_config.get("max_position_sol", 0.08)
        new_position = old_position * factor
        new_position = self._clamp(
            new_position,
            self.bounds.min_position_sol,
            self.bounds.max_position_sol,
        )
        if abs(new_position - old_position) > 0.001:
            old_values["max_position_sol"] = old_position
            new_config["max_position_sol"] = round(new_position, 4)

        # Scale max_daily_volume_sol (float)
        old_volume = current_config.get("max_daily_volume_sol", 1.5)
        new_volume = old_volume * factor
        new_volume = self._clamp(
            new_volume,
            self.bounds.min_daily_volume_sol,
            self.bounds.max_daily_volume_sol,
        )
        if abs(new_volume - old_volume) > 0.01:
            old_values["max_daily_volume_sol"] = old_volume
            new_config["max_daily_volume_sol"] = round(new_volume, 2)

        # Scale max_daily_loss_sol proportionally with volume
        old_loss = current_config.get("max_daily_loss_sol", 0.20)
        volume_ratio = new_volume / old_volume if old_volume > 0 else 1.0
        new_loss = old_loss * volume_ratio
        new_loss = self._clamp(
            new_loss,
            self.bounds.min_daily_loss_sol,
            self.bounds.max_daily_loss_sol,
        )
        if abs(new_loss - old_loss) > 0.001:
            old_values["max_daily_loss_sol"] = old_loss
            new_config["max_daily_loss_sol"] = round(new_loss, 3)

        # Adjust min_confidence_to_trade (inverse relationship)
        # Scale UP -> lower confidence threshold (more aggressive)
        # Scale DOWN -> higher confidence threshold (more conservative)
        old_confidence = current_config.get("min_confidence_to_trade", 0.60)
        if action == LimitAction.SCALE_UP:
            confidence_factor = 1.0 / factor  # Inverse
        else:
            confidence_factor = factor
        new_confidence = old_confidence * confidence_factor
        new_confidence = self._clamp(
            new_confidence,
            self.bounds.min_confidence_floor,
            self.bounds.min_confidence_ceiling,
        )
        if abs(new_confidence - old_confidence) > 0.01:
            old_values["min_confidence_to_trade"] = old_confidence
            new_config["min_confidence_to_trade"] = round(new_confidence, 2)

        # Record the adjustment
        if old_values:
            record = AdjustmentRecord(
                timestamp=datetime.now(),
                action=action,
                trigger_win_rate=self.get_rolling_win_rate(),
                old_values=old_values,
                new_values={k: new_config[k] for k in old_values.keys()},
                reason=action_reason,
            )
            self._adjustment_history.append(record)

            # Trim history
            if len(self._adjustment_history) > self._max_history:
                self._adjustment_history = self._adjustment_history[-self._max_history :]

            # Update state
            self._last_adjustment = datetime.now()
            self._last_action = action

            # Update multiplier tracking
            if action == LimitAction.SCALE_UP:
                self._current_multiplier *= factor
            elif action in [LimitAction.SCALE_DOWN, LimitAction.EMERGENCY_REDUCE]:
                self._current_multiplier *= factor

            logger.info(
                f"✅ Applied {action.value}: {old_values} -> "
                f"{dict((k, new_config[k]) for k in old_values.keys())}"
            )

        return new_config

    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp a value between min and max."""
        return max(min_val, min(max_val, value))

    def get_adjustment_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent adjustment history."""
        return [r.to_dict() for r in self._adjustment_history[-limit:]]

    def get_status(self) -> dict[str, Any]:
        """Get current status of the adaptive manager."""
        return {
            "autonomous_enabled": self.config.autonomous_enabled,
            "total_trades": self._total_trades,
            "rolling_sample_size": self.get_sample_size(),
            "rolling_win_rate": self.get_rolling_win_rate(),
            "rolling_avg_pnl": self.get_rolling_pnl_avg(),
            "current_multiplier": round(self._current_multiplier, 3),
            "last_adjustment": (
                self._last_adjustment.isoformat() if self._last_adjustment else None
            ),
            "last_action": self._last_action.value if self._last_action else None,
            "in_cooldown": self._is_in_cooldown(),
            "adjustment_count": len(self._adjustment_history),
            "config": self.config.to_dict(),
            "bounds": self.bounds.to_dict(),
        }

    def reset(self) -> None:
        """Reset the manager state."""
        self._outcomes.clear()
        self._total_trades = 0
        self._last_adjustment = None
        self._last_action = None
        self._current_multiplier = 1.0
        logger.info("AdaptiveLimitsManager reset")

    def set_autonomous(self, enabled: bool) -> None:
        """Enable or disable autonomous operation."""
        self.config.autonomous_enabled = enabled
        logger.info(f"Autonomous limit management: {'enabled' if enabled else 'disabled'}")

    def force_adjustment(self, action: LimitAction, reason: str = "manual") -> dict[str, Any]:
        """Force an adjustment regardless of conditions.

        Use with caution - bypasses all safety checks.
        """
        logger.warning(f"⚠️ Forced adjustment: {action.value} - {reason}")
        # Bypass cooldown for manual adjustments
        self._last_adjustment = None
        return {
            "action": action.value,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
