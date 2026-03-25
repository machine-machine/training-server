"""
Success Rate Strategy Engine - Smart Performance-Based Parameter Adjustment.

🎯 CORE PHILOSOPHY (nasr, Feb 2026):
═══════════════════════════════════════════════════════════════════════════════
"When you're hot, PRESS THE ADVANTAGE. When you're cold, PROTECT CAPITAL."

This engine dynamically adjusts ALL trading parameters based on success rate,
streak analysis, and performance momentum - maximizing profitability while
protecting against drawdowns.

KEY INSIGHT: The market doesn't care about your feelings, but your RECENT
PERFORMANCE is the best predictor of NEAR-TERM results. A trader on a hot
streak should be MORE aggressive; a trader on a cold streak should pull back.

═══════════════════════════════════════════════════════════════════════════════

STRATEGY TIERS (based on rolling win rate + streak analysis):
───────────────────────────────────────────────────────────────────────────────

🔥 ON_FIRE    (Win Rate > 65% + 5+ win streak)
   → MAXIMUM AGGRESSION: Press every edge
   → Larger positions, tighter stops, extended targets
   → Lower confidence threshold, faster execution
   → Shorter cooldowns, higher slippage tolerance

😊 HOT        (Win Rate 55-65% + positive momentum)
   → INCREASED AGGRESSION: Lean into the edge
   → Moderate position increase, normal stops/targets
   → Slightly lower confidence threshold
   → Normal cooldowns

😐 NEUTRAL   (Win Rate 45-55%)
   → BASELINE: Standard parameters
   → No adjustments from defaults

😰 COLD       (Win Rate 35-45% + negative momentum)
   → DEFENSIVE: Protect capital
   → Smaller positions, wider stops, quicker exits
   → Higher confidence threshold
   → Longer cooldowns after losses

🚨 FREEZING   (Win Rate < 35% OR 5+ loss streak)
   → EMERGENCY DEFENSE: Minimal risk
   → Minimum position size
   → Maximum confidence threshold
   → Longest cooldowns
   → Consider pausing

═══════════════════════════════════════════════════════════════════════════════

PARAMETERS ADJUSTED BY SUCCESS RATE:
───────────────────────────────────────────────────────────────────────────────

| Parameter              | ON_FIRE | HOT  | NEUTRAL | COLD  | FREEZING |
|------------------------|---------|------|---------|-------|----------|
| Position Size          | 1.5x    | 1.2x | 1.0x    | 0.7x  | 0.4x     |
| Confidence Threshold   | -0.10   | -0.05| 0.00    | +0.10 | +0.15    |
| Stop Loss Width        | 0.8x    | 0.9x | 1.0x    | 1.2x  | 1.5x     |
| Take Profit Target     | 1.3x    | 1.15x| 1.0x    | 0.85x | 0.7x     |
| Cooldown After Win     | 0.5x    | 0.7x | 1.0x    | 1.0x  | 1.0x     |
| Cooldown After Loss    | 0.5x    | 0.8x | 1.0x    | 1.5x  | 2.0x     |
| Max Slippage           | 1.3x    | 1.1x | 1.0x    | 0.8x  | 0.6x     |
| Min Liquidity          | 0.8x    | 0.9x | 1.0x    | 1.2x  | 1.5x     |
| Trailing Stop Trigger  | -15%    | -10% | 0%      | +10%  | +20%     |

═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PerformanceTier(Enum):
    """Performance tier based on success rate and streak."""

    ON_FIRE = "on_fire"  # >65% win rate + hot streak
    HOT = "hot"  # 55-65% win rate + positive momentum
    NEUTRAL = "neutral"  # 45-55% win rate
    COLD = "cold"  # 35-45% win rate + negative momentum
    FREEZING = "freezing"  # <35% win rate OR cold streak


class AdjustmentType(Enum):
    """Type of parameter adjustment."""

    POSITION_SIZE = "position_size"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    COOLDOWN_WIN = "cooldown_win"
    COOLDOWN_LOSS = "cooldown_loss"
    MAX_SLIPPAGE = "max_slippage"
    MIN_LIQUIDITY = "min_liquidity"
    TRAILING_STOP = "trailing_stop"
    KELLY_FRACTION = "kelly_fraction"


@dataclass
class TierAdjustments:
    """Parameter adjustments for a performance tier."""

    position_size_multiplier: float = 1.0
    confidence_adjustment: float = 0.0  # Negative = lower threshold
    stop_loss_multiplier: float = 1.0  # >1 = wider stop
    take_profit_multiplier: float = 1.0  # >1 = higher target
    cooldown_win_multiplier: float = 1.0  # <1 = shorter cooldown
    cooldown_loss_multiplier: float = 1.0  # >1 = longer cooldown
    max_slippage_multiplier: float = 1.0  # >1 = allow more slippage
    min_liquidity_multiplier: float = 1.0  # >1 = require more liquidity
    trailing_stop_adjustment: float = 0.0  # % adjustment to trigger
    kelly_fraction_multiplier: float = 1.0  # >1 = use more Kelly

    def to_dict(self) -> dict[str, float]:
        return {
            "position_size_multiplier": self.position_size_multiplier,
            "confidence_adjustment": self.confidence_adjustment,
            "stop_loss_multiplier": self.stop_loss_multiplier,
            "take_profit_multiplier": self.take_profit_multiplier,
            "cooldown_win_multiplier": self.cooldown_win_multiplier,
            "cooldown_loss_multiplier": self.cooldown_loss_multiplier,
            "max_slippage_multiplier": self.max_slippage_multiplier,
            "min_liquidity_multiplier": self.min_liquidity_multiplier,
            "trailing_stop_adjustment": self.trailing_stop_adjustment,
            "kelly_fraction_multiplier": self.kelly_fraction_multiplier,
        }


# Define adjustments for each tier
TIER_ADJUSTMENTS = {
    PerformanceTier.ON_FIRE: TierAdjustments(
        position_size_multiplier=1.50,
        confidence_adjustment=-0.10,
        stop_loss_multiplier=0.80,
        take_profit_multiplier=1.30,
        cooldown_win_multiplier=0.50,
        cooldown_loss_multiplier=0.50,
        max_slippage_multiplier=1.30,
        min_liquidity_multiplier=0.80,
        trailing_stop_adjustment=-15.0,  # Start trailing earlier
        kelly_fraction_multiplier=1.50,
    ),
    PerformanceTier.HOT: TierAdjustments(
        position_size_multiplier=1.20,
        confidence_adjustment=-0.05,
        stop_loss_multiplier=0.90,
        take_profit_multiplier=1.15,
        cooldown_win_multiplier=0.70,
        cooldown_loss_multiplier=0.80,
        max_slippage_multiplier=1.10,
        min_liquidity_multiplier=0.90,
        trailing_stop_adjustment=-10.0,
        kelly_fraction_multiplier=1.25,
    ),
    PerformanceTier.NEUTRAL: TierAdjustments(
        position_size_multiplier=1.00,
        confidence_adjustment=0.00,
        stop_loss_multiplier=1.00,
        take_profit_multiplier=1.00,
        cooldown_win_multiplier=1.00,
        cooldown_loss_multiplier=1.00,
        max_slippage_multiplier=1.00,
        min_liquidity_multiplier=1.00,
        trailing_stop_adjustment=0.0,
        kelly_fraction_multiplier=1.00,
    ),
    PerformanceTier.COLD: TierAdjustments(
        position_size_multiplier=0.70,
        confidence_adjustment=+0.10,
        stop_loss_multiplier=1.20,
        take_profit_multiplier=0.85,
        cooldown_win_multiplier=1.00,
        cooldown_loss_multiplier=1.50,
        max_slippage_multiplier=0.80,
        min_liquidity_multiplier=1.20,
        trailing_stop_adjustment=+10.0,  # Start trailing later
        kelly_fraction_multiplier=0.70,
    ),
    PerformanceTier.FREEZING: TierAdjustments(
        position_size_multiplier=0.40,
        confidence_adjustment=+0.15,
        stop_loss_multiplier=1.50,
        take_profit_multiplier=0.70,
        cooldown_win_multiplier=1.00,
        cooldown_loss_multiplier=2.00,
        max_slippage_multiplier=0.60,
        min_liquidity_multiplier=1.50,
        trailing_stop_adjustment=+20.0,
        kelly_fraction_multiplier=0.40,
    ),
}


@dataclass
class StreakInfo:
    """Information about current win/loss streaks."""

    current_streak: int = 0  # Positive = wins, negative = losses
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    streak_type: str = "none"  # "win", "loss", "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_streak": self.current_streak,
            "longest_win_streak": self.longest_win_streak,
            "longest_loss_streak": self.longest_loss_streak,
            "streak_type": self.streak_type,
        }


@dataclass
class PerformanceMomentum:
    """Momentum indicators for performance."""

    short_term_trend: float = 0.0  # Last 5 trades: positive = improving
    medium_term_trend: float = 0.0  # Last 15 trades
    momentum_score: float = 0.0  # Combined momentum (-1 to 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "short_term_trend": self.short_term_trend,
            "medium_term_trend": self.medium_term_trend,
            "momentum_score": self.momentum_score,
        }


@dataclass
class SuccessRateConfig:
    """Configuration for success rate strategy."""

    # Window sizes
    rolling_window_size: int = 30  # Trades for win rate calculation
    momentum_short_window: int = 5  # Trades for short-term trend
    momentum_medium_window: int = 15  # Trades for medium-term trend

    # Tier thresholds (win rates)
    on_fire_threshold: float = 0.65
    hot_threshold: float = 0.55
    cold_threshold: float = 0.45
    freezing_threshold: float = 0.35

    # Streak thresholds
    hot_streak_threshold: int = 5  # Wins to consider "hot streak"
    cold_streak_threshold: int = 5  # Losses to consider "cold streak"

    # Minimum trades before adjusting
    min_trades_for_adjustment: int = 10

    # Momentum weights
    short_term_weight: float = 0.6
    medium_term_weight: float = 0.4

    # Enable/disable features
    streak_adjustments_enabled: bool = True
    momentum_adjustments_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "rolling_window_size": self.rolling_window_size,
            "momentum_short_window": self.momentum_short_window,
            "momentum_medium_window": self.momentum_medium_window,
            "on_fire_threshold": self.on_fire_threshold,
            "hot_threshold": self.hot_threshold,
            "cold_threshold": self.cold_threshold,
            "freezing_threshold": self.freezing_threshold,
            "hot_streak_threshold": self.hot_streak_threshold,
            "cold_streak_threshold": self.cold_streak_threshold,
            "min_trades_for_adjustment": self.min_trades_for_adjustment,
            "streak_adjustments_enabled": self.streak_adjustments_enabled,
            "momentum_adjustments_enabled": self.momentum_adjustments_enabled,
        }


@dataclass
class AdjustmentRecord:
    """Record of a strategy adjustment."""

    timestamp: datetime
    old_tier: PerformanceTier
    new_tier: PerformanceTier
    win_rate: float
    streak: StreakInfo
    momentum: PerformanceMomentum
    adjustments_applied: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "old_tier": self.old_tier.value,
            "new_tier": self.new_tier.value,
            "win_rate": self.win_rate,
            "streak": self.streak.to_dict(),
            "momentum": self.momentum.to_dict(),
            "adjustments_applied": self.adjustments_applied,
            "reason": self.reason,
        }


class SuccessRateStrategy:
    """
    Success Rate-Based Strategy Engine.

    Dynamically adjusts trading parameters based on recent performance,
    streak analysis, and momentum indicators.

    Usage:
        strategy = SuccessRateStrategy()

        # Record each trade outcome
        strategy.record_outcome(is_win=True, pnl_pct=5.2)

        # Get current tier and adjustments
        tier, adjustments = strategy.evaluate()

        # Apply to trading config
        new_position = base_position * adjustments.position_size_multiplier
        new_confidence = base_confidence + adjustments.confidence_adjustment
    """

    def __init__(self, config: SuccessRateConfig | None = None):
        self.config = config or SuccessRateConfig()

        # Rolling windows
        self._outcomes: deque = deque(maxlen=self.config.rolling_window_size)

        # Streak tracking
        self._streak = StreakInfo()

        # Current state
        self._current_tier = PerformanceTier.NEUTRAL
        self._total_trades = 0

        # Adjustment history
        self._adjustment_history: list[AdjustmentRecord] = []
        self._max_history = 100

    def record_outcome(self, is_win: bool, pnl_pct: float = 0.0) -> None:
        """Record a trade outcome.

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

        # Update streak
        self._update_streak(is_win)

    def _update_streak(self, is_win: bool) -> None:
        """Update streak tracking."""
        if is_win:
            if self._streak.streak_type == "win":
                self._streak.current_streak += 1
            else:
                self._streak.streak_type = "win"
                self._streak.current_streak = 1
            self._streak.longest_win_streak = max(
                self._streak.longest_win_streak, self._streak.current_streak
            )
        else:
            if self._streak.streak_type == "loss":
                self._streak.current_streak -= 1
            else:
                self._streak.streak_type = "loss"
                self._streak.current_streak = -1
            self._streak.longest_loss_streak = max(
                self._streak.longest_loss_streak, abs(self._streak.current_streak)
            )

    def get_win_rate(self) -> float:
        """Calculate current rolling win rate."""
        if not self._outcomes:
            return 0.5  # Neutral default
        wins = sum(1 for o in self._outcomes if o["is_win"])
        return wins / len(self._outcomes)

    def get_streak(self) -> StreakInfo:
        """Get current streak information."""
        return self._streak

    def get_momentum(self) -> PerformanceMomentum:
        """Calculate performance momentum."""
        outcomes = list(self._outcomes)

        if len(outcomes) < self.config.momentum_short_window:
            return PerformanceMomentum()

        # Short-term trend (last N trades)
        short = outcomes[-self.config.momentum_short_window :]
        short_wins = sum(1 for o in short if o["is_win"])
        short_trend = (short_wins / len(short)) - 0.5  # -0.5 to 0.5

        # Medium-term trend
        if len(outcomes) >= self.config.momentum_medium_window:
            medium = outcomes[-self.config.momentum_medium_window :]
            medium_wins = sum(1 for o in medium if o["is_win"])
            medium_trend = (medium_wins / len(medium)) - 0.5
        else:
            medium_trend = short_trend

        # Combined momentum score
        momentum_score = (
            short_trend * self.config.short_term_weight
            + medium_trend * self.config.medium_term_weight
        )

        return PerformanceMomentum(
            short_term_trend=short_trend,
            medium_term_trend=medium_trend,
            momentum_score=momentum_score,
        )

    def _determine_tier(
        self, win_rate: float, streak: StreakInfo, momentum: PerformanceMomentum
    ) -> PerformanceTier:
        """Determine performance tier based on win rate, streak, and momentum.

        🧠 SMART LOGIC (nasr's Philosophy):
        - Overall win rate provides a "reputation buffer"
        - Recent momentum (short-term) is weighted heavily
        - Streaks matter, but context matters more
        - NET PROFITABILITY matters: if you're still up, be less reactive

        A trader with 70% overall win rate gets more leeway on a cold streak
        than a trader with 40% win rate on the same streak.

        Key insight: "Why stop trading when you're still profitable?"
        """

        # Calculate "reputation buffer" - high win rate = more tolerance for drawdowns
        reputation_buffer = max(0, (win_rate - 0.50) * 2)  # 0 to ~0.3 for 50-65% win rates
        # Decay reputation buffer under negative momentum
        # — don't let past success mask current losses
        if momentum.momentum_score < -0.15:
            momentum_penalty = min(1.0, abs(momentum.momentum_score) * 3)
            reputation_buffer *= 1.0 - momentum_penalty
        effective_freezing_threshold = self.config.freezing_threshold - (reputation_buffer * 0.10)
        effective_cold_threshold = self.config.cold_threshold - (reputation_buffer * 0.05)

        # Calculate net position (wins - losses in rolling window)
        outcomes = list(self._outcomes)
        wins_in_window = sum(1 for o in outcomes if o["is_win"])
        losses_in_window = len(outcomes) - wins_in_window
        net_position = wins_in_window - losses_in_window
        net_profit_buffer = max(0, net_position / max(len(outcomes), 1))  # 0 to 1

        # Short-term win rate (last 5 trades) - most important signal
        short_term_win_rate = 0.5 + momentum.short_term_trend

        # 🚨 FREEZING: Critical conditions ONLY
        # - Very low overall win rate (below adjusted threshold)
        # - AND: No reputation or net profit buffer
        if win_rate < effective_freezing_threshold and net_profit_buffer < 0.1:
            return PerformanceTier.FREEZING

        # 5+ loss streak triggers FREEZING ONLY IF:
        # - Short-term is terrible (<20% last 5) AND
        # - Overall win rate is below 50% OR net position is negative
        if (
            streak.streak_type == "loss"
            and abs(streak.current_streak) >= self.config.cold_streak_threshold
            and short_term_win_rate < 0.20
            and (win_rate < 0.50 or net_position < 0)
        ):
            return PerformanceTier.FREEZING

        # 🔥 ON_FIRE: Exceptional conditions
        # - Very high win rate AND hot streak AND positive momentum
        if (
            win_rate >= self.config.on_fire_threshold
            and streak.streak_type == "win"
            and streak.current_streak >= self.config.hot_streak_threshold
            and momentum.momentum_score > 0
        ):
            return PerformanceTier.ON_FIRE

        # Also ON_FIRE if win rate >70% and short-term >80%
        if win_rate >= 0.70 and short_term_win_rate >= 0.80:
            return PerformanceTier.ON_FIRE

        # 😊 HOT: Good conditions
        # - High win rate (uses effective threshold for reputation buffer)
        if win_rate >= self.config.hot_threshold:
            return PerformanceTier.HOT

        # Also HOT if positive momentum and decent short-term
        if momentum.momentum_score > 0.15 and short_term_win_rate >= 0.60:
            return PerformanceTier.HOT

        # 😰 COLD: Concerning conditions
        # - Low win rate (uses effective threshold)
        if win_rate < effective_cold_threshold:
            return PerformanceTier.COLD

        # Cold streak with negative short-term momentum
        if (
            streak.streak_type == "loss"
            and abs(streak.current_streak) >= 3
            and short_term_win_rate < 0.35
        ):
            return PerformanceTier.COLD

        # Negative momentum with declining performance
        if momentum.momentum_score < -0.20 and short_term_win_rate < 0.40:
            return PerformanceTier.COLD

        # 😐 NEUTRAL: Default - balanced conditions
        return PerformanceTier.NEUTRAL

    def evaluate(self) -> tuple[PerformanceTier, TierAdjustments]:
        """
        Evaluate current performance and get adjustments.

        Returns:
            Tuple of (PerformanceTier, TierAdjustments)
        """
        win_rate = self.get_win_rate()
        streak = self.get_streak()
        momentum = self.get_momentum()

        # Determine tier
        new_tier = self._determine_tier(win_rate, streak, momentum)

        # Check if tier changed
        if new_tier != self._current_tier:
            self._record_tier_change(self._current_tier, new_tier, win_rate, streak, momentum)
            self._current_tier = new_tier

        # Get adjustments for tier
        adjustments = TIER_ADJUSTMENTS[new_tier]

        return new_tier, adjustments

    def _record_tier_change(
        self,
        old_tier: PerformanceTier,
        new_tier: PerformanceTier,
        win_rate: float,
        streak: StreakInfo,
        momentum: PerformanceMomentum,
    ) -> None:
        """Record a tier change for audit trail."""
        reason = self._get_tier_change_reason(old_tier, new_tier, win_rate, streak)

        record = AdjustmentRecord(
            timestamp=datetime.now(),
            old_tier=old_tier,
            new_tier=new_tier,
            win_rate=win_rate,
            streak=streak,
            momentum=momentum,
            adjustments_applied=TIER_ADJUSTMENTS[new_tier].to_dict(),
            reason=reason,
        )

        self._adjustment_history.append(record)

        # Trim history
        if len(self._adjustment_history) > self._max_history:
            self._adjustment_history = self._adjustment_history[-self._max_history :]

        # Log the change
        tier_emoji = {
            PerformanceTier.ON_FIRE: "🔥",
            PerformanceTier.HOT: "😊",
            PerformanceTier.NEUTRAL: "😐",
            PerformanceTier.COLD: "😰",
            PerformanceTier.FREEZING: "🚨",
        }

        logger.info(
            f"{tier_emoji[new_tier]} Strategy tier changed: {old_tier.value} → {new_tier.value} | "
            f"Win rate: {win_rate:.1%} | Streak: {streak.current_streak} | {reason}"
        )

    def _get_tier_change_reason(
        self,
        old_tier: PerformanceTier,
        new_tier: PerformanceTier,
        win_rate: float,
        streak: StreakInfo,
    ) -> str:
        """Generate human-readable reason for tier change."""
        if new_tier == PerformanceTier.ON_FIRE:
            return f"On fire! {win_rate:.0%} win rate, {streak.current_streak}-win streak"
        elif new_tier == PerformanceTier.HOT:
            return f"Heating up: {win_rate:.0%} win rate"
        elif new_tier == PerformanceTier.COLD:
            return f"Cooling down: {win_rate:.0%} win rate"
        elif new_tier == PerformanceTier.FREEZING:
            if streak.streak_type == "loss":
                return f"Emergency: {abs(streak.current_streak)}-loss streak"
            return f"Emergency: {win_rate:.0%} win rate"
        else:
            return f"Normalizing: {win_rate:.0%} win rate"

    def apply_to_config(
        self,
        base_config: dict[str, Any],
        adjustments: TierAdjustments | None = None,
    ) -> dict[str, Any]:
        """
        Apply success-rate adjustments to a configuration dict.

        Args:
            base_config: Base configuration to adjust
            adjustments: Adjustments to apply (uses current tier if None)

        Returns:
            New configuration with adjustments applied
        """
        if adjustments is None:
            _, adjustments = self.evaluate()

        new_config = base_config.copy()

        # Position size
        if "max_position_sol" in base_config:
            new_config["max_position_sol"] = round(
                base_config["max_position_sol"] * adjustments.position_size_multiplier, 4
            )

        # Confidence threshold
        if "min_confidence_to_trade" in base_config:
            new_config["min_confidence_to_trade"] = max(
                0.35,
                min(
                    0.75, base_config["min_confidence_to_trade"] + adjustments.confidence_adjustment
                ),
            )

        # Stop loss
        if "default_stop_loss_pct" in base_config:
            new_config["default_stop_loss_pct"] = round(
                base_config["default_stop_loss_pct"] * adjustments.stop_loss_multiplier, 1
            )

        # Take profit
        if "default_take_profit_pct" in base_config:
            new_config["default_take_profit_pct"] = round(
                base_config["default_take_profit_pct"] * adjustments.take_profit_multiplier, 1
            )

        # Cooldowns
        if "win_cooldown_seconds" in base_config:
            new_config["win_cooldown_seconds"] = max(
                10, int(base_config["win_cooldown_seconds"] * adjustments.cooldown_win_multiplier)
            )
        if "loss_cooldown_seconds" in base_config:
            new_config["loss_cooldown_seconds"] = max(
                30, int(base_config["loss_cooldown_seconds"] * adjustments.cooldown_loss_multiplier)
            )

        # Slippage
        if "max_slippage_bps" in base_config:
            new_config["max_slippage_bps"] = round(
                base_config["max_slippage_bps"] * adjustments.max_slippage_multiplier, 0
            )

        # Liquidity requirement
        if "min_liquidity_usd" in base_config:
            new_config["min_liquidity_usd"] = round(
                base_config["min_liquidity_usd"] * adjustments.min_liquidity_multiplier, 0
            )

        # Kelly fraction
        if "kelly_fraction" in base_config:
            new_config["kelly_fraction"] = min(
                0.50, base_config["kelly_fraction"] * adjustments.kelly_fraction_multiplier
            )

        # Trailing stop trigger
        if "trailing_stop_trigger_pct" in base_config:
            new_config["trailing_stop_trigger_pct"] = max(
                5.0, base_config["trailing_stop_trigger_pct"] + adjustments.trailing_stop_adjustment
            )

        return new_config

    def get_status(self) -> dict[str, Any]:
        """Get current status of the strategy engine."""
        tier, adjustments = self.evaluate()

        return {
            "current_tier": tier.value,
            "win_rate": self.get_win_rate(),
            "total_trades": self._total_trades,
            "rolling_sample_size": len(self._outcomes),
            "streak": self._streak.to_dict(),
            "momentum": self.get_momentum().to_dict(),
            "current_adjustments": adjustments.to_dict(),
            "config": self.config.to_dict(),
        }

    def get_adjustment_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent adjustment history."""
        return [r.to_dict() for r in self._adjustment_history[-limit:]]

    def reset(self) -> None:
        """Reset the strategy state."""
        self._outcomes.clear()
        self._streak = StreakInfo()
        self._current_tier = PerformanceTier.NEUTRAL
        self._total_trades = 0
        logger.info("SuccessRateStrategy reset")

    def should_pause(self) -> tuple[bool, str]:
        """
        Check if trading should pause based on performance.

        Returns:
            Tuple of (should_pause, reason)
        """
        tier, _ = self.evaluate()
        win_rate = self.get_win_rate()
        streak = self.get_streak()

        if tier == PerformanceTier.FREEZING:
            if streak.streak_type == "loss" and abs(streak.current_streak) >= 5:
                return True, f"🚨 PAUSE: {abs(streak.current_streak)}-loss streak - cooling off"
            if win_rate < 0.30:
                return True, f"🚨 PAUSE: Win rate {win_rate:.0%} critically low"

        if tier == PerformanceTier.COLD and win_rate < 0.38:
            return True, f"⚠️ PAUSE: Win rate {win_rate:.0%} too low - regrouping"

        return False, ""

    def get_trading_advice(self) -> str:
        """
        Get human-readable trading advice based on current tier.
        """
        tier, adjustments = self.evaluate()
        win_rate = self.get_win_rate()
        streak = self.get_streak()
        self.get_momentum()

        advice_map = {
            PerformanceTier.ON_FIRE: (
                f"🔥 ON FIRE! Win rate: {win_rate:.0%}, {streak.current_streak}-win streak. "
                f"Press the advantage! Larger positions "
                f"({adjustments.position_size_multiplier:.1f}x), tighter stops, let winners run. "
                f"Strike while hot!"
            ),
            PerformanceTier.HOT: (
                f"😊 Running hot at {win_rate:.0%} win rate. "
                f"Increase aggression slightly. Good time to take quality setups."
            ),
            PerformanceTier.NEUTRAL: (
                f"😐 Neutral zone: {win_rate:.0%} win rate. "
                f"Stick to the plan. Standard position sizes and risk management."
            ),
            PerformanceTier.COLD: (
                f"😰 Cooling off: {win_rate:.0%} win rate. "
                f"Reduce position sizes, be more selective. Focus on high-conviction trades only."
            ),
            PerformanceTier.FREEZING: (
                f"🚨 FREEZING! Win rate: {win_rate:.0%}, {abs(streak.current_streak)}-loss streak. "
                f"Minimum position sizes only. Consider pausing to reset. "
                f"Protect capital at all costs!"
            ),
        }

        return advice_map[tier]
