"""
Operation Modes - Predefined trading modes with different risk/throughput profiles.

Each mode is a preset configuration optimized for different trading scenarios:
- SCOUT: Observation only, no trades (for data collection)
- CONSERVATIVE: Small positions, high confidence threshold (for capital preservation)
- BALANCED: Default behavior with moderate risk (for steady growth)
- AGGRESSIVE: Higher throughput, accept more risk (for maximizing gains)
- AUTONOMOUS: Full self-management with dynamic mode switching

The AutonomousModeManager handles automatic mode switching based on:
- Market regime detection
- Rolling performance metrics
- Daily P&L targets
- Risk limits
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OperationMode(Enum):
    """Trading operation modes."""

    SCOUT = "scout"  # Observation only, no trades
    CONSERVATIVE = "conservative"  # Small positions, high confidence
    BALANCED = "balanced"  # Default behavior
    AGGRESSIVE = "aggressive"  # Higher throughput, more risk
    AUTONOMOUS = "autonomous"  # Full self-management


class ModeTransitionReason(Enum):
    """Reasons for mode transitions."""

    MANUAL = "manual"
    REGIME_CHANGE = "regime_change"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    DAILY_TARGET_REACHED = "daily_target_reached"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    WIN_RATE_THRESHOLD = "win_rate_threshold"
    AUTO_OPTIMIZATION = "auto_optimization"
    SAFETY_TRIGGER = "safety_trigger"


@dataclass
class ModeConfig:
    """Configuration preset for a specific operation mode."""

    mode: OperationMode

    # Position sizing
    max_position_sol: float
    min_position_sol: float
    kelly_fraction: float

    # Limits
    max_daily_trades: int
    max_concurrent_positions: int
    max_daily_volume_sol: float
    max_daily_loss_sol: float

    # Confidence
    min_confidence_to_trade: float
    min_confidence_for_full_size: float

    # Behavior
    paper_mode: bool
    skip_learning: bool
    auto_pause_enabled: bool
    min_win_rate_threshold: float

    # Mode-specific settings
    description: str = ""
    risk_level: str = "medium"  # low, medium, high
    target_daily_pnl_pct: float = 0.8

    def to_config_dict(self) -> dict[str, Any]:
        """Convert to AutoTraderConfig-compatible dict."""
        return {
            "max_position_sol": self.max_position_sol,
            "min_position_sol": self.min_position_sol,
            "kelly_fraction": self.kelly_fraction,
            "max_daily_trades": self.max_daily_trades,
            "max_concurrent_positions": self.max_concurrent_positions,
            "max_daily_volume_sol": self.max_daily_volume_sol,
            "max_daily_loss_sol": self.max_daily_loss_sol,
            "min_confidence_to_trade": self.min_confidence_to_trade,
            "min_confidence_for_full_size": self.min_confidence_for_full_size,
            "paper_mode": self.paper_mode,
            "skip_learning": self.skip_learning,
            "auto_pause_enabled": self.auto_pause_enabled,
            "min_win_rate_threshold": self.min_win_rate_threshold,
        }

    def to_dict(self) -> dict[str, Any]:
        """Full serialization."""
        return {
            "mode": self.mode.value,
            "max_position_sol": self.max_position_sol,
            "min_position_sol": self.min_position_sol,
            "kelly_fraction": self.kelly_fraction,
            "max_daily_trades": self.max_daily_trades,
            "max_concurrent_positions": self.max_concurrent_positions,
            "max_daily_volume_sol": self.max_daily_volume_sol,
            "max_daily_loss_sol": self.max_daily_loss_sol,
            "min_confidence_to_trade": self.min_confidence_to_trade,
            "min_confidence_for_full_size": self.min_confidence_for_full_size,
            "paper_mode": self.paper_mode,
            "skip_learning": self.skip_learning,
            "auto_pause_enabled": self.auto_pause_enabled,
            "min_win_rate_threshold": self.min_win_rate_threshold,
            "description": self.description,
            "risk_level": self.risk_level,
            "target_daily_pnl_pct": self.target_daily_pnl_pct,
        }


# ============================================================================
# MODE PRESETS
# ============================================================================

MODE_PRESETS: dict[OperationMode, ModeConfig] = {
    OperationMode.SCOUT: ModeConfig(
        mode=OperationMode.SCOUT,
        description="Observation mode - collects data without trading",
        risk_level="none",
        # No actual trading
        max_position_sol=0.0,
        min_position_sol=0.0,
        kelly_fraction=0.0,
        max_daily_trades=0,
        max_concurrent_positions=0,
        max_daily_volume_sol=0.0,
        max_daily_loss_sol=0.0,
        min_confidence_to_trade=1.0,  # Impossible threshold
        min_confidence_for_full_size=1.0,
        paper_mode=True,
        skip_learning=False,
        auto_pause_enabled=False,
        min_win_rate_threshold=0.0,
        target_daily_pnl_pct=0.0,
    ),
    OperationMode.CONSERVATIVE: ModeConfig(
        mode=OperationMode.CONSERVATIVE,
        description="Conservative mode - capital preservation focus",
        risk_level="low",
        # Small, careful positions
        max_position_sol=0.03,
        min_position_sol=0.001,
        kelly_fraction=0.15,  # Very conservative Kelly
        max_daily_trades=15,
        max_concurrent_positions=3,
        max_daily_volume_sol=0.50,
        max_daily_loss_sol=0.08,
        min_confidence_to_trade=0.70,  # High bar
        min_confidence_for_full_size=0.85,
        paper_mode=False,
        skip_learning=False,
        auto_pause_enabled=True,
        min_win_rate_threshold=0.30,  # Pause if <30% win rate
        target_daily_pnl_pct=0.5,
    ),
    OperationMode.BALANCED: ModeConfig(
        mode=OperationMode.BALANCED,
        description="Balanced mode - steady growth with moderate risk",
        risk_level="medium",
        # Default scaled settings
        max_position_sol=0.08,
        min_position_sol=0.002,
        kelly_fraction=0.30,
        max_daily_trades=40,
        max_concurrent_positions=6,
        max_daily_volume_sol=1.50,
        max_daily_loss_sol=0.20,
        min_confidence_to_trade=0.55,
        min_confidence_for_full_size=0.75,
        paper_mode=False,
        skip_learning=False,
        auto_pause_enabled=True,
        min_win_rate_threshold=0.25,
        target_daily_pnl_pct=0.8,
    ),
    OperationMode.AGGRESSIVE: ModeConfig(
        mode=OperationMode.AGGRESSIVE,
        description="Aggressive mode - maximize gains with higher risk",
        risk_level="high",
        # Larger positions, more trades
        max_position_sol=0.15,
        min_position_sol=0.005,
        kelly_fraction=0.45,  # More aggressive Kelly
        max_daily_trades=60,
        max_concurrent_positions=8,
        max_daily_volume_sol=3.0,
        max_daily_loss_sol=0.40,
        min_confidence_to_trade=0.50,  # Lower bar
        min_confidence_for_full_size=0.70,
        paper_mode=False,
        skip_learning=False,
        auto_pause_enabled=True,
        min_win_rate_threshold=0.20,  # Allow lower win rate
        target_daily_pnl_pct=1.2,
    ),
    OperationMode.AUTONOMOUS: ModeConfig(
        mode=OperationMode.AUTONOMOUS,
        description="Autonomous mode - full self-management with dynamic switching",
        risk_level="adaptive",
        # Start balanced, can scale up/down
        max_position_sol=0.10,
        min_position_sol=0.002,
        kelly_fraction=0.35,
        max_daily_trades=50,
        max_concurrent_positions=6,
        max_daily_volume_sol=2.0,
        max_daily_loss_sol=0.25,
        min_confidence_to_trade=0.55,
        min_confidence_for_full_size=0.75,
        paper_mode=False,
        skip_learning=False,
        auto_pause_enabled=True,
        min_win_rate_threshold=0.25,
        target_daily_pnl_pct=1.0,
    ),
}


@dataclass
class ModeTransition:
    """Record of a mode transition."""

    timestamp: datetime
    from_mode: OperationMode
    to_mode: OperationMode
    reason: ModeTransitionReason
    details: str
    performance_snapshot: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "from_mode": self.from_mode.value,
            "to_mode": self.to_mode.value,
            "reason": self.reason.value,
            "details": self.details,
            "performance_snapshot": self.performance_snapshot,
        }


class AutonomousModeManager:
    """Manages operation mode selection and automatic transitions.

    Features:
    - Mode presets for different risk profiles
    - Automatic mode switching based on performance
    - Regime-aware mode recommendations
    - Transition logging for audit trail
    - Safety constraints on autonomous operation
    """

    def __init__(
        self,
        initial_mode: OperationMode = OperationMode.BALANCED,
        autonomous_enabled: bool = False,
        min_trades_before_switch: int = 30,
        min_time_between_switches_seconds: int = 600,  # 10 minutes
    ):
        self.current_mode = initial_mode
        self.autonomous_enabled = autonomous_enabled
        self.min_trades_before_switch = min_trades_before_switch
        self.min_time_between_switches = min_time_between_switches_seconds

        # Transition tracking
        self._transition_history: list[ModeTransition] = []
        self._last_transition: datetime | None = None
        self._max_history: int = 100

        # Performance tracking for mode decisions
        self._trades_since_last_switch: int = 0
        self._total_trades: int = 0

        # Regime-aware switching thresholds
        self._regime_mode_map = {
            "BULL": OperationMode.AGGRESSIVE,
            "BEAR": OperationMode.CONSERVATIVE,
            "CHOP": OperationMode.BALANCED,
            "MEV_HEAVY": OperationMode.CONSERVATIVE,
        }

        # Performance thresholds for auto-switching
        self._performance_thresholds = {
            "win_rate_high": 0.60,  # Switch to aggressive
            "win_rate_low": 0.40,  # Switch to conservative
            "pnl_high_pct": 1.2,  # Daily P&L % to consider aggressive
            "pnl_low_pct": 0.3,  # Daily P&L % to consider conservative
            "drawdown_high_pct": 15.0,  # Max DD before conservative
        }

    def get_current_config(self) -> ModeConfig:
        """Get the configuration for the current mode."""
        return MODE_PRESETS[self.current_mode]

    def set_mode(
        self,
        new_mode: OperationMode,
        reason: ModeTransitionReason = ModeTransitionReason.MANUAL,
        details: str = "",
        performance_snapshot: dict[str, float] | None = None,
        force: bool = False,
    ) -> bool:
        """Switch to a new operation mode.

        Args:
            new_mode: The mode to switch to
            reason: Why the switch is happening
            details: Additional details
            performance_snapshot: Current performance metrics
            force: Bypass cooldown checks

        Returns:
            True if switched, False if blocked
        """
        if new_mode == self.current_mode:
            return False

        # Check cooldown unless forced
        if not force and self._is_in_cooldown():
            logger.debug(
                f"Mode switch blocked: in cooldown "
                f"({self._seconds_until_available():.0f}s remaining)"
            )
            return False

        # Check minimum trades unless forced
        if not force and self._trades_since_last_switch < self.min_trades_before_switch:
            logger.debug(
                f"Mode switch blocked: only {self._trades_since_last_switch} trades "
                f"(need {self.min_trades_before_switch})"
            )
            return False

        # Record transition
        old_mode = self.current_mode
        transition = ModeTransition(
            timestamp=datetime.now(),
            from_mode=old_mode,
            to_mode=new_mode,
            reason=reason,
            details=details or f"Switched from {old_mode.value} to {new_mode.value}",
            performance_snapshot=performance_snapshot or {},
        )
        self._transition_history.append(transition)

        # Trim history
        if len(self._transition_history) > self._max_history:
            self._transition_history = self._transition_history[-self._max_history :]

        # Update state
        self.current_mode = new_mode
        self._last_transition = datetime.now()
        self._trades_since_last_switch = 0

        logger.info(
            f"🔄 Mode switch: {old_mode.value} -> {new_mode.value} (reason: {reason.value})"
        )

        return True

    def record_trade(self) -> None:
        """Record that a trade occurred."""
        self._trades_since_last_switch += 1
        self._total_trades += 1

    def evaluate_auto_switch(
        self,
        win_rate: float,
        daily_pnl_pct: float,
        max_drawdown_pct: float,
        current_regime: str | None = None,
        daily_trades: int = 0,
    ) -> OperationMode | None:
        """Evaluate if automatic mode switch is warranted.

        Args:
            win_rate: Rolling win rate (0-1)
            daily_pnl_pct: Daily P&L as percentage
            max_drawdown_pct: Current max drawdown percentage
            current_regime: Market regime (BULL, BEAR, CHOP, MEV_HEAVY)
            daily_trades: Number of trades today

        Returns:
            Recommended mode, or None if no switch needed
        """
        if not self.autonomous_enabled:
            return None

        if self.current_mode == OperationMode.AUTONOMOUS:
            # In fully autonomous mode, we can switch between non-scout modes
            pass
        elif self.current_mode == OperationMode.SCOUT:
            # Scout mode doesn't auto-switch
            return None

        # Safety check: high drawdown -> always suggest conservative
        if max_drawdown_pct > self._performance_thresholds["drawdown_high_pct"]:
            logger.warning(f"⚠️ High drawdown ({max_drawdown_pct:.1f}%) - recommending conservative")
            return OperationMode.CONSERVATIVE

        # Regime-based recommendation (highest priority)
        if current_regime and current_regime in self._regime_mode_map:
            regime_mode = self._regime_mode_map[current_regime]
            if regime_mode != self.current_mode:
                logger.info(f"📊 Regime {current_regime} suggests {regime_mode.value} mode")
                # Don't auto-switch based solely on regime, just log

        # Performance-based switching
        if win_rate > self._performance_thresholds["win_rate_high"]:
            if daily_pnl_pct > self._performance_thresholds["pnl_high_pct"]:
                # Excellent performance - go aggressive
                if self.current_mode in [OperationMode.CONSERVATIVE, OperationMode.BALANCED]:
                    return OperationMode.AGGRESSIVE

        elif win_rate < self._performance_thresholds["win_rate_low"]:
            # Poor performance - go conservative
            if self.current_mode in [OperationMode.BALANCED, OperationMode.AGGRESSIVE]:
                return OperationMode.CONSERVATIVE

        # Daily P&L based
        if daily_pnl_pct < self._performance_thresholds["pnl_low_pct"]:
            if self.current_mode == OperationMode.AGGRESSIVE:
                return OperationMode.BALANCED

        # Daily target reached - consider scaling down risk
        target = MODE_PRESETS[self.current_mode].target_daily_pnl_pct
        if daily_pnl_pct >= target * 1.5:  # 150% of target
            if self.current_mode == OperationMode.AGGRESSIVE:
                logger.info(f"🎯 Daily target exceeded ({daily_pnl_pct:.1f}% vs {target:.1f}%)")
                # Could suggest balanced to lock in gains

        return None

    def get_recommended_mode(
        self,
        win_rate: float,
        daily_pnl_pct: float,
        risk_tolerance: str = "medium",
    ) -> OperationMode:
        """Get recommended mode based on current conditions.

        This is a recommendation only - does not trigger switch.
        """
        if risk_tolerance == "low":
            return OperationMode.CONSERVATIVE
        elif risk_tolerance == "high":
            if win_rate > 0.55 and daily_pnl_pct > 0.8:
                return OperationMode.AGGRESSIVE
            return OperationMode.BALANCED
        else:  # medium
            if win_rate > 0.55:
                return OperationMode.AGGRESSIVE
            elif win_rate < 0.45:
                return OperationMode.CONSERVATIVE
            return OperationMode.BALANCED

    def _is_in_cooldown(self) -> bool:
        """Check if we're in cooldown from last switch."""
        if self._last_transition is None:
            return False

        elapsed = (datetime.now() - self._last_transition).total_seconds()
        return elapsed < self.min_time_between_switches

    def _seconds_until_available(self) -> float:
        """Seconds until mode switch is available."""
        if self._last_transition is None:
            return 0.0

        elapsed = (datetime.now() - self._last_transition).total_seconds()
        remaining = self.min_time_between_switches - elapsed
        return max(0.0, remaining)

    def get_transition_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent mode transition history."""
        return [t.to_dict() for t in self._transition_history[-limit:]]

    def get_status(self) -> dict[str, Any]:
        """Get current mode manager status."""
        return {
            "current_mode": self.current_mode.value,
            "autonomous_enabled": self.autonomous_enabled,
            "trades_since_last_switch": self._trades_since_last_switch,
            "total_trades": self._total_trades,
            "in_cooldown": self._is_in_cooldown(),
            "seconds_until_available": self._seconds_until_available(),
            "transition_count": len(self._transition_history),
            "current_config": self.get_current_config().to_dict(),
        }

    def set_autonomous(self, enabled: bool) -> None:
        """Enable or disable autonomous mode switching."""
        self.autonomous_enabled = enabled
        logger.info(f"Autonomous mode switching: {'enabled' if enabled else 'disabled'}")

    def reset(self) -> None:
        """Reset mode manager state."""
        self._trades_since_last_switch = 0
        self._total_trades = 0
        self._last_transition = None
        self.current_mode = OperationMode.BALANCED
        logger.info("Mode manager reset to BALANCED")


def get_mode_config(mode: OperationMode) -> ModeConfig:
    """Get the configuration preset for a mode."""
    return MODE_PRESETS[mode]


def list_available_modes() -> list[dict[str, Any]]:
    """List all available operation modes with their configs."""
    return [
        {
            "mode": mode.value,
            "description": config.description,
            "risk_level": config.risk_level,
            "target_daily_pnl_pct": config.target_daily_pnl_pct,
            "max_daily_trades": config.max_daily_trades,
            "max_position_sol": config.max_position_sol,
        }
        for mode, config in MODE_PRESETS.items()
    ]
