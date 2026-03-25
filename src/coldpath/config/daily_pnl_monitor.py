"""
Daily P&L Monitor - Track and manage daily P&L against targets.

Features:
- Target-based mode switching (AGGRESSIVE/BALANCED/CONSERVATIVE)
- Position size adjustments based on daily P&L
- Automatic risk reduction when above target
- Alerts when below minimum threshold
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode based on daily P&L performance."""

    AGGRESSIVE = "aggressive"  # Below target - need more profit
    BALANCED = "balanced"  # On target - stay the course
    CONSERVATIVE = "conservative"  # Above max - protect gains
    PAUSED = "paused"  # Hit daily loss limit


@dataclass
class DailyPnLMetrics:
    """Daily P&L metrics snapshot."""

    date: date
    start_pnl_sol: float
    current_pnl_sol: float
    daily_pnl_sol: float
    daily_pnl_pct: float
    trade_count: int
    win_count: int
    loss_count: int
    mode: TradingMode
    position_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "start_pnl_sol": self.start_pnl_sol,
            "current_pnl_sol": self.current_pnl_sol,
            "daily_pnl_sol": self.daily_pnl_sol,
            "daily_pnl_pct": self.daily_pnl_pct,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "mode": self.mode.value,
            "position_multiplier": self.position_multiplier,
        }


@dataclass
class DailyPnLMonitor:
    """Monitor and manage daily P&L against targets.

    Implements target-based trading mode switching:
    - Below 0.6%: AGGRESSIVE mode (increase trading, larger positions)
    - 0.6% - 1.2%: BALANCED mode (normal operation)
    - Above 1.2%: CONSERVATIVE mode (reduce risk, protect gains)

    Usage:
        monitor = DailyPnLMonitor(initial_capital_sol=1.0)

        # Update throughout the day
        monitor.update(current_pnl_sol=0.008, trade_count=15, win_count=9)

        # Get current mode and multiplier
        mode = monitor.get_mode()
        multiplier = monitor.get_position_size_multiplier()
    """

    # Target thresholds (as percentage of capital)
    TARGET_MIN_PCT = 0.6  # Minimum acceptable daily P&L
    TARGET_OPTIMAL_PCT = 0.8  # Target daily P&L
    TARGET_MAX_PCT = 1.2  # Reduce risk above this

    # Position size multipliers by mode
    MULTIPLIER_AGGRESSIVE = 1.3
    MULTIPLIER_BALANCED = 1.0
    MULTIPLIER_CONSERVATIVE = 0.6

    # Initial capital
    initial_capital_sol: float = 1.0

    # Daily tracking
    current_date: date = field(default_factory=date.today)
    daily_start_pnl_sol: float = 0.0
    current_pnl_sol: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0

    # History (last 30 days)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize tracking for current day."""
        self.current_date = date.today()

    def reset_if_new_day(self, current_pnl_sol: float) -> bool:
        """Reset tracking at start of new day.

        Args:
            current_pnl_sol: Current cumulative P&L

        Returns:
            True if day changed, False otherwise
        """
        today = date.today()
        if today != self.current_date:
            # Save previous day to history
            if self.trade_count > 0:
                self._save_to_history()

            # Reset for new day
            self.current_date = today
            self.daily_start_pnl_sol = current_pnl_sol
            self.current_pnl_sol = current_pnl_sol
            self.trade_count = 0
            self.win_count = 0
            self.loss_count = 0

            logger.info(f"New trading day: {today}, starting P&L: {current_pnl_sol:.6f} SOL")
            return True
        return False

    def update(
        self,
        current_pnl_sol: float,
        trade_count: int,
        win_count: int,
        loss_count: int | None = None,
    ) -> DailyPnLMetrics:
        """Update P&L tracking with latest values.

        Args:
            current_pnl_sol: Current cumulative P&L in SOL
            trade_count: Total trades today
            win_count: Winning trades today
            loss_count: Losing trades today (optional, derived if not provided)

        Returns:
            DailyPnLMetrics snapshot
        """
        self.reset_if_new_day(current_pnl_sol)

        self.current_pnl_sol = current_pnl_sol
        self.trade_count = trade_count
        self.win_count = win_count
        self.loss_count = loss_count if loss_count is not None else (trade_count - win_count)

        return self.get_metrics()

    def record_trade(self, pnl_sol: float, is_win: bool):
        """Record a single trade result.

        Args:
            pnl_sol: P&L from this trade in SOL
            is_win: Whether this was a winning trade
        """
        self.current_pnl_sol += pnl_sol
        self.trade_count += 1
        if is_win:
            self.win_count += 1
        else:
            self.loss_count += 1

    def get_daily_pnl_sol(self) -> float:
        """Get today's P&L in SOL."""
        return self.current_pnl_sol - self.daily_start_pnl_sol

    def get_daily_pnl_pct(self) -> float:
        """Get today's P&L as percentage of capital."""
        if self.initial_capital_sol <= 0:
            return 0.0
        daily_pnl = self.get_daily_pnl_sol()
        return (daily_pnl / self.initial_capital_sol) * 100

    def get_mode(self) -> TradingMode:
        """Get recommended trading mode based on daily P&L."""
        pct = self.get_daily_pnl_pct()

        if pct < self.TARGET_MIN_PCT:
            return TradingMode.AGGRESSIVE
        elif pct > self.TARGET_MAX_PCT:
            return TradingMode.CONSERVATIVE
        else:
            return TradingMode.BALANCED

    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on daily P&L.

        Returns:
            Multiplier to apply to base position size:
            - AGGRESSIVE: 1.3x (increase size to catch up)
            - BALANCED: 1.0x (normal)
            - CONSERVATIVE: 0.6x (reduce size to protect gains)
        """
        mode = self.get_mode()

        if mode == TradingMode.AGGRESSIVE:
            return self.MULTIPLIER_AGGRESSIVE
        elif mode == TradingMode.CONSERVATIVE:
            return self.MULTIPLIER_CONSERVATIVE
        else:
            return self.MULTIPLIER_BALANCED

    def get_metrics(self) -> DailyPnLMetrics:
        """Get current daily P&L metrics snapshot."""
        return DailyPnLMetrics(
            date=self.current_date,
            start_pnl_sol=self.daily_start_pnl_sol,
            current_pnl_sol=self.current_pnl_sol,
            daily_pnl_sol=self.get_daily_pnl_sol(),
            daily_pnl_pct=self.get_daily_pnl_pct(),
            trade_count=self.trade_count,
            win_count=self.win_count,
            loss_count=self.loss_count,
            mode=self.get_mode(),
            position_multiplier=self.get_position_size_multiplier(),
        )

    def should_trade_more(self) -> bool:
        """Check if we should increase trading activity."""
        pct = self.get_daily_pnl_pct()

        # Below target and have room for more trades
        if pct < self.TARGET_MIN_PCT and self.trade_count < 30:
            return True

        return False

    def should_reduce_risk(self) -> bool:
        """Check if we should reduce risk exposure."""
        return self.get_mode() == TradingMode.CONSERVATIVE

    def get_win_rate(self) -> float:
        """Get today's win rate."""
        if self.trade_count == 0:
            return 0.0
        return self.win_count / self.trade_count

    def is_on_target(self) -> bool:
        """Check if daily P&L is within target range."""
        pct = self.get_daily_pnl_pct()
        return self.TARGET_MIN_PCT <= pct <= self.TARGET_MAX_PCT

    def distance_to_target(self) -> float:
        """Get distance to optimal target (positive = above, negative = below)."""
        return self.get_daily_pnl_pct() - self.TARGET_OPTIMAL_PCT

    def _save_to_history(self):
        """Save current day to history."""
        metrics = self.get_metrics()
        self.history.append(metrics.to_dict())

        # Keep last 30 days
        if len(self.history) > 30:
            self.history = self.history[-30:]

    def get_summary(self) -> dict[str, Any]:
        """Get summary of daily P&L status."""
        metrics = self.get_metrics()

        return {
            "date": metrics.date.isoformat(),
            "daily_pnl_sol": f"{metrics.daily_pnl_sol:.6f}",
            "daily_pnl_pct": f"{metrics.daily_pnl_pct:.2f}%",
            "target_min": f"{self.TARGET_MIN_PCT}%",
            "target_optimal": f"{self.TARGET_OPTIMAL_PCT}%",
            "target_max": f"{self.TARGET_MAX_PCT}%",
            "mode": metrics.mode.value,
            "position_multiplier": f"{metrics.position_multiplier:.1f}x",
            "trade_count": metrics.trade_count,
            "win_rate": f"{self.get_win_rate() * 100:.1f}%",
            "on_target": self.is_on_target(),
            "distance_to_target": f"{self.distance_to_target():.2f}%",
        }

    def get_weekly_stats(self) -> dict[str, Any]:
        """Get statistics for the last 7 days."""
        if len(self.history) < 1:
            return {
                "days_traded": 0,
                "total_pnl_pct": 0.0,
                "avg_daily_pnl_pct": 0.0,
                "days_on_target": 0,
            }

        recent = self.history[-7:]  # Last 7 days

        total_pnl = sum(d.get("daily_pnl_pct", 0) for d in recent)
        days_on_target = sum(
            1
            for d in recent
            if self.TARGET_MIN_PCT <= d.get("daily_pnl_pct", 0) <= self.TARGET_MAX_PCT
        )

        return {
            "days_traded": len(recent),
            "total_pnl_pct": round(total_pnl, 2),
            "avg_daily_pnl_pct": round(total_pnl / len(recent), 2),
            "days_on_target": days_on_target,
            "on_target_rate": f"{days_on_target / len(recent) * 100:.0f}%",
        }
