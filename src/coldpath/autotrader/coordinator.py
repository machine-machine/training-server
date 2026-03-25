"""
AutoTrader Coordinator - Autonomous trading with continuous learning.

Implements a state machine for autonomous trading:
STOPPED -> STARTING -> LEARNING -> TRADING -> PAUSED/DEGRADED

Features:
- Multi-armed bandit for slippage selection
- Fraud model integration for risk filtering
- Position sizing with Kelly criterion
- Daily limits and risk management
- Continuous learning from trade outcomes
"""

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from ..calibration.bias_calibrator import BiasCalibrator
    from ..learning.outcome_tracker import OutcomeTracker
    from ..learning.profitability_learner import ProfitabilityLearner
    from ..learning.regime_detector import RegimeDetector
    from ..storage import DatabaseManager
    from ..training.bandit import BanditTrainer
    from ..training.fraud_model import FraudModel

from ..config.verified_settings import VerifiedSettingsStore
from ..execution.market_impact import SquareRootImpactModel

# GLM Week 1-4: Advanced features
from ..metrics.alpha_decay import AlphaDecayAnalyzer
from ..models.ml_trade_context import (
    DeployerRisk,
    MarketRegime,
    MLPaperTradingConfig,
    MLTradeContext,
)
from ..onchain.whale_tracker import WhaleWalletTracker
from ..signals.time_filters import TimePatternFilter
from ..trading.success_rate_strategy import (
    PerformanceTier,
    SuccessRateStrategy,
)
from .adaptive_limits import (
    AdaptiveLimitConfig,
    AdaptiveLimitsManager,
    LimitAction,
    LimitBounds,
)
from .operation_modes import (
    AutonomousModeManager,
    ModeTransitionReason,
    OperationMode,
    get_mode_config,
    list_available_modes,
)

logger = logging.getLogger(__name__)


class AutoTraderState(Enum):
    """AutoTrader state machine states."""

    STOPPED = "stopped"
    STARTING = "starting"
    LEARNING = "learning"
    TRADING = "trading"
    PAUSED = "paused"
    DEGRADED = "degraded"


@dataclass
class AutoTraderConfig:
    """Configuration for AutoTrader."""

    # Learning requirements (relaxed to avoid chicken-and-egg: can't trade without
    # history, can't build history without trading)
    # Trade immediately; learning-only gates were stalling long runs
    min_observations_to_trade: int = 0
    learning_duration_hours: int = 0
    min_win_rate_to_trade: float = 0.0

    # Paper mode controls *execution semantics* (paper fills / safe mode).
    # Learning/trading state transitions are controlled by `skip_learning`.
    # Default skip_learning=True to avoid blocking on first startup (chicken-and-egg)
    paper_mode: bool = True
    skip_learning: bool = True

    # Confidence thresholds
    # SCALED: 55% threshold for more trade opportunities
    min_confidence_to_trade: float = 0.01
    min_confidence_for_full_size: float = 0.60

    # Position sizing - SCALED UP for extended operation
    max_position_sol: float = 0.08  # Was 0.02 -> 4x increase
    min_position_sol: float = 0.002  # Was 0.0015 -> slight increase
    kelly_fraction: float = 0.30  # Was 0.25 -> slightly more aggressive

    # Daily limits - SCALED UP for extended operation
    # Target: 200+ trades/day for 0.8-1% daily P&L
    max_daily_volume_sol: float = 1_000.0
    max_daily_loss_sol: float = 50.0
    max_daily_trades: int = 1_500
    max_concurrent_positions: int = 10  # keep concurrency high

    # Auto-pause safety monitoring
    # If win rate drops below threshold after min_trades_for_pause, auto-pause for review
    auto_pause_enabled: bool = False
    min_trades_for_pause: int = 1_000_000  # effectively disabled
    min_win_rate_threshold: float = 0.0  # no auto-pause on win rate

    # Risk filters
    max_fraud_score: float = 0.4
    min_liquidity_usd: float = 10000
    max_fdv_usd: float = 5_000_000
    min_holder_count: int = 50
    min_token_age_seconds: int = 0
    min_volume_24h_usd: float = 0.0
    max_top_holder_pct: float | None = None
    max_volatility_1h: float | None = None

    # Slippage settings
    default_slippage_bps: int = 300
    max_slippage_bps: int = 1_000

    # Timing
    signal_expiry_seconds: int = 30
    cooldown_after_loss_seconds: int = 0
    # 0 disables timeouts (safe default)
    decision_timeout_ms: int = 0

    # UI-propagated settings
    ml_strategy: str = "adaptive"  # momentum/mean_reversion/breakout/sentiment/hybrid/adaptive
    risk_mode: str = "balanced"  # conservative/balanced/aggressive
    ensemble_enabled: bool = False
    trailing_stop_enabled: bool = False
    trailing_stop_percent: float = 5.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutoTraderConfig":
        """Create from dictionary."""
        defaults = cls()
        return cls(
            min_observations_to_trade=data.get(
                "min_observations_to_trade",
                defaults.min_observations_to_trade,
            ),
            learning_duration_hours=data.get(
                "learning_duration_hours",
                defaults.learning_duration_hours,
            ),
            min_win_rate_to_trade=data.get(
                "min_win_rate_to_trade",
                defaults.min_win_rate_to_trade,
            ),
            paper_mode=data.get("paper_mode", defaults.paper_mode),
            skip_learning=data.get("skip_learning", defaults.skip_learning),
            min_confidence_to_trade=data.get(
                "min_confidence_to_trade",
                defaults.min_confidence_to_trade,
            ),
            min_confidence_for_full_size=data.get(
                "min_confidence_for_full_size",
                defaults.min_confidence_for_full_size,
            ),
            max_position_sol=data.get("max_position_sol", defaults.max_position_sol),
            min_position_sol=data.get("min_position_sol", defaults.min_position_sol),
            kelly_fraction=data.get("kelly_fraction", defaults.kelly_fraction),
            max_daily_volume_sol=data.get(
                "max_daily_volume_sol",
                defaults.max_daily_volume_sol,
            ),
            max_daily_loss_sol=data.get("max_daily_loss_sol", defaults.max_daily_loss_sol),
            max_daily_trades=data.get("max_daily_trades", defaults.max_daily_trades),
            max_concurrent_positions=data.get(
                "max_concurrent_positions",
                defaults.max_concurrent_positions,
            ),
            auto_pause_enabled=data.get("auto_pause_enabled", defaults.auto_pause_enabled),
            min_trades_for_pause=data.get(
                "min_trades_for_pause",
                defaults.min_trades_for_pause,
            ),
            min_win_rate_threshold=data.get(
                "min_win_rate_threshold",
                defaults.min_win_rate_threshold,
            ),
            max_fraud_score=data.get("max_fraud_score", defaults.max_fraud_score),
            min_liquidity_usd=data.get("min_liquidity_usd", defaults.min_liquidity_usd),
            max_fdv_usd=data.get("max_fdv_usd", defaults.max_fdv_usd),
            min_holder_count=data.get("min_holder_count", defaults.min_holder_count),
            min_token_age_seconds=data.get(
                "min_token_age_seconds",
                defaults.min_token_age_seconds,
            ),
            min_volume_24h_usd=data.get(
                "min_volume_24h_usd",
                defaults.min_volume_24h_usd,
            ),
            max_top_holder_pct=data.get("max_top_holder_pct", defaults.max_top_holder_pct),
            max_volatility_1h=data.get("max_volatility_1h", defaults.max_volatility_1h),
            default_slippage_bps=data.get(
                "default_slippage_bps",
                defaults.default_slippage_bps,
            ),
            max_slippage_bps=data.get("max_slippage_bps", defaults.max_slippage_bps),
            signal_expiry_seconds=data.get(
                "signal_expiry_seconds",
                defaults.signal_expiry_seconds,
            ),
            cooldown_after_loss_seconds=data.get(
                "cooldown_after_loss_seconds",
                defaults.cooldown_after_loss_seconds,
            ),
            decision_timeout_ms=data.get("decision_timeout_ms", defaults.decision_timeout_ms),
            ml_strategy=data.get("ml_strategy", defaults.ml_strategy),
            risk_mode=data.get("risk_mode", defaults.risk_mode),
            ensemble_enabled=data.get("ensemble_enabled", defaults.ensemble_enabled),
            trailing_stop_enabled=data.get(
                "trailing_stop_enabled",
                defaults.trailing_stop_enabled,
            ),
            trailing_stop_percent=data.get(
                "trailing_stop_percent",
                defaults.trailing_stop_percent,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "min_observations_to_trade": self.min_observations_to_trade,
            "learning_duration_hours": self.learning_duration_hours,
            "min_win_rate_to_trade": self.min_win_rate_to_trade,
            "paper_mode": self.paper_mode,
            "skip_learning": self.skip_learning,
            "min_confidence_to_trade": self.min_confidence_to_trade,
            "min_confidence_for_full_size": self.min_confidence_for_full_size,
            "max_position_sol": self.max_position_sol,
            "min_position_sol": self.min_position_sol,
            "kelly_fraction": self.kelly_fraction,
            "max_daily_volume_sol": self.max_daily_volume_sol,
            "max_daily_loss_sol": self.max_daily_loss_sol,
            "max_daily_trades": self.max_daily_trades,
            "max_concurrent_positions": self.max_concurrent_positions,
            "auto_pause_enabled": self.auto_pause_enabled,
            "min_trades_for_pause": self.min_trades_for_pause,
            "min_win_rate_threshold": self.min_win_rate_threshold,
            "max_fraud_score": self.max_fraud_score,
            "min_liquidity_usd": self.min_liquidity_usd,
            "max_fdv_usd": self.max_fdv_usd,
            "min_holder_count": self.min_holder_count,
            "min_token_age_seconds": self.min_token_age_seconds,
            "min_volume_24h_usd": self.min_volume_24h_usd,
            "max_top_holder_pct": self.max_top_holder_pct,
            "max_volatility_1h": self.max_volatility_1h,
            "default_slippage_bps": self.default_slippage_bps,
            "max_slippage_bps": self.max_slippage_bps,
            "signal_expiry_seconds": self.signal_expiry_seconds,
            "cooldown_after_loss_seconds": self.cooldown_after_loss_seconds,
            "decision_timeout_ms": self.decision_timeout_ms,
            "ml_strategy": self.ml_strategy,
            "risk_mode": self.risk_mode,
            "ensemble_enabled": self.ensemble_enabled,
            "trailing_stop_enabled": self.trailing_stop_enabled,
            "trailing_stop_percent": self.trailing_stop_percent,
        }


@dataclass
class TradingCandidate:
    """A token candidate for trading."""

    token_mint: str
    token_symbol: str
    pool_address: str

    # Market data
    price_usd: float
    liquidity_usd: float
    fdv_usd: float
    volume_24h_usd: float
    holder_count: int
    age_seconds: int

    # Token metadata
    mint_authority_enabled: bool = False
    freeze_authority_enabled: bool = False
    lp_burn_pct: float = 0.0
    top_holder_pct: float = 0.0

    # Volatility metrics
    volatility_1h: float = 0.0
    price_change_1h: float = 0.0

    # Discovery metadata
    discovery_timestamp: datetime = field(default_factory=datetime.now)
    discovery_source: str = "scanner"


@dataclass
class TradingSignal:
    """A signal to execute a trade."""

    signal_id: str
    timestamp: datetime
    token_mint: str
    token_symbol: str
    pool_address: str

    # Trade parameters
    action: str  # "BUY" or "SELL"
    position_size_sol: float
    slippage_bps: int
    expected_price: float

    # Confidence and reasoning
    confidence: float
    fraud_score: float
    selected_arm: str  # Bandit arm selected

    # Expiry
    expires_at: datetime

    # Filters passed
    filters_passed: dict[str, bool] = field(default_factory=dict)

    # Feature snapshot
    feature_snapshot: dict[str, float] = field(default_factory=dict)

    # ML Trade Context for paper fill simulation
    ml_context: MLTradeContext | None = None

    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


@dataclass
class TradeOutcome:
    """Outcome of an executed trade."""

    signal_id: str
    executed_at: datetime
    pnl_sol: float
    pnl_pct: float
    slippage_realized_bps: int
    included: bool
    execution_latency_ms: int


@dataclass
class DailyMetrics:
    """Daily trading metrics for limit tracking."""

    date: datetime
    total_volume_sol: float = 0.0
    total_pnl_sol: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    max_drawdown_sol: float = 0.0
    active_positions: int = 0

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0


@dataclass
class LearningMetrics:
    """Metrics collected during learning phase."""

    observations: int = 0
    paper_trades: int = 0
    paper_wins: int = 0
    avg_slippage_bps: float = 0.0
    avg_latency_ms: float = 0.0
    inclusion_rate: float = 0.0
    start_time: datetime | None = None

    @property
    def learning_duration_hours(self) -> float:
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds() / 3600

    @property
    def paper_win_rate(self) -> float:
        return self.paper_wins / self.paper_trades if self.paper_trades > 0 else 0.0


# ===========================================
# Phase 4: Enhanced AutoTrader Autonomy
# ===========================================


class AutonomousMode:
    """Self-tuning autonomous trading mode.

    Monitors performance and auto-adjusts parameters to hit
    daily P&L targets (0.8-1% optimal, 0.6% minimum).

    Features:
    - 4-hour tuning cooldown to avoid over-adjustment
    - Adjusts confidence thresholds based on win rate
    - Tunes Kelly fraction based on P&L performance
    - Tracks all tuning adjustments for audit
    """

    TARGET_MIN_PNL_PCT = 0.6  # 0.6% minimum daily P&L
    TARGET_OPTIMAL_PNL_PCT = 0.8  # 0.8% target
    MIN_TRADES_FOR_TUNING = 30  # Minimum trades before tuning

    def __init__(self, coordinator: "AutoTraderCoordinator"):
        self.coordinator = coordinator
        self.tuning_history: deque = deque(maxlen=100)
        self.last_tuning_time: datetime | None = None
        self.tuning_cooldown_hours = 4
        self.total_tuning_events = 0

    async def auto_tune_if_needed(self) -> dict[str, Any] | None:
        """Check if tuning is needed and perform it.

        Returns:
            Dict of adjustments made, or None if no tuning needed
        """
        # Cooldown check
        now = datetime.now()
        if self.last_tuning_time:
            hours_since = (now - self.last_tuning_time).total_seconds() / 3600
            if hours_since < self.tuning_cooldown_hours:
                logger.debug(
                    f"Auto-tune cooldown: {hours_since:.1f}h since last tune "
                    f"(need {self.tuning_cooldown_hours}h)"
                )
                return None

        # Need sufficient trades for statistical significance
        total_trades = (
            self.coordinator.daily_metrics.win_count + self.coordinator.daily_metrics.loss_count
        )
        if total_trades < self.MIN_TRADES_FOR_TUNING:
            logger.debug(
                f"Auto-tune: Need {self.MIN_TRADES_FOR_TUNING} trades, have {total_trades}"
            )
            return None

        # Get current performance metrics
        current_pnl_pct = self.coordinator._calculate_daily_pnl_pct()
        win_rate = self.coordinator.daily_metrics.win_rate

        tuning_needed = False
        adjustments: dict[str, Any] = {}

        # Scenario 1: P&L below minimum target → increase aggression
        if current_pnl_pct < self.TARGET_MIN_PNL_PCT:
            tuning_needed = True
            adjustments["action"] = "increase_aggression"
            adjustments["min_confidence_to_trade"] = max(
                0.45, self.coordinator.config.min_confidence_to_trade - 0.05
            )
            adjustments["kelly_fraction"] = min(0.50, self.coordinator.config.kelly_fraction + 0.05)
            logger.info(
                f"🔥 Auto-tune: P&L {current_pnl_pct:.2f}% below {self.TARGET_MIN_PNL_PCT}% "
                f"- increasing aggression"
            )

        # Scenario 2: Win rate too low → tighten filters
        if win_rate < 0.40:
            tuning_needed = True
            adjustments["action"] = "tighten_filters"
            adjustments["min_confidence_to_trade"] = min(
                0.65, self.coordinator.config.min_confidence_to_trade + 0.05
            )
            adjustments["max_fraud_score"] = max(
                0.30, self.coordinator.config.max_fraud_score - 0.05
            )
            logger.info(f"🔒 Auto-tune: Win rate {win_rate:.0%} below 40% - tightening filters")

        # Scenario 3: Win rate excellent but P&L not maxed → increase size
        if win_rate > 0.60 and current_pnl_pct < 1.0:
            tuning_needed = True
            adjustments["action"] = "increase_size"
            adjustments["max_position_sol"] = min(
                0.15, self.coordinator.config.max_position_sol * 1.2
            )
            logger.info(
                f"📈 Auto-tune: Win rate {win_rate:.0%} excellent - increasing position size"
            )

        if not tuning_needed:
            logger.debug(
                f"Auto-tune: No adjustments needed (P&L={current_pnl_pct:.2f}%, "
                f"win_rate={win_rate:.0%})"
            )
            return None

        # Apply adjustments
        self._apply_adjustments(adjustments)
        self.last_tuning_time = now
        self.total_tuning_events += 1

        return adjustments

    def _apply_adjustments(self, adjustments: dict[str, Any]) -> None:
        """Apply tuning adjustments to coordinator config."""
        action = adjustments.pop("action", "unknown")

        for key, value in adjustments.items():
            if hasattr(self.coordinator.config, key):
                old_value = getattr(self.coordinator.config, key)
                setattr(self.coordinator.config, key, value)
                logger.info(f"⚙️ Auto-tune applied: {key} = {old_value} → {value} (action={action})")
            else:
                logger.warning(f"Auto-tune: Unknown config key '{key}'")

        # Record for audit trail
        self.tuning_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "adjustments": adjustments.copy(),
                "trigger_metrics": {
                    "pnl_pct": self.coordinator._calculate_daily_pnl_pct(),
                    "win_rate": self.coordinator.daily_metrics.win_rate,
                    "trade_count": (
                        self.coordinator.daily_metrics.win_count
                        + self.coordinator.daily_metrics.loss_count
                    ),
                },
            }
        )

    def get_tuning_status(self) -> dict[str, Any]:
        """Get current tuning status and history."""
        return {
            "total_tuning_events": self.total_tuning_events,
            "last_tuning_time": self.last_tuning_time.isoformat()
            if self.last_tuning_time
            else None,
            "cooldown_hours": self.tuning_cooldown_hours,
            "min_trades_for_tuning": self.MIN_TRADES_FOR_TUNING,
            "recent_adjustments": list(self.tuning_history)[-10:],  # Last 10
        }


class DailyPnLMonitor:
    """Monitor and manage daily P&L against 0.8-1% targets.

    Tracks daily performance and provides:
    - Trading mode recommendations (AGGRESSIVE/BALANCED/CONSERVATIVE)
    - Position size multipliers based on daily P&L
    - Day-over-day tracking
    """

    TARGET_MIN = 0.6  # 0.6% minimum acceptable
    TARGET_OPTIMAL = 0.8  # 0.8% target
    TARGET_MAX = 1.2  # Reduce risk above 1.2%

    def __init__(self, initial_capital_sol: float = 1.0):
        self.initial_capital = initial_capital_sol
        self.daily_start_pnl = 0.0
        self.current_date = datetime.now().date()
        self.peak_pnl_pct = 0.0
        self.trough_pnl_pct = 0.0

    def reset_if_new_day(self, current_pnl_sol: float) -> bool:
        """Reset tracking at start of new day.

        Returns:
            True if a new day was detected and reset occurred
        """
        today = datetime.now().date()
        if today != self.current_date:
            self.daily_start_pnl = current_pnl_sol
            self.current_date = today
            self.peak_pnl_pct = 0.0
            self.trough_pnl_pct = 0.0
            logger.info(
                f"📅 Daily P&L monitor reset for {today} | Starting P&L: {current_pnl_sol:.4f} SOL"
            )
            return True
        return False

    def get_daily_pnl_pct(self, current_pnl_sol: float) -> float:
        """Get today's P&L as percentage of capital."""
        daily_pnl = current_pnl_sol - self.daily_start_pnl
        if self.initial_capital <= 0:
            return 0.0
        return (daily_pnl / self.initial_capital) * 100

    def get_mode(self, current_pnl_sol: float) -> str:
        """Get recommended trading mode based on daily P&L.

        Returns:
            "AGGRESSIVE" - below target, need more trades
            "CONSERVATIVE" - above max, protect gains
            "BALANCED" - in target range
        """
        pct = self.get_daily_pnl_pct(current_pnl_sol)

        # Track peak and trough
        if pct > self.peak_pnl_pct:
            self.peak_pnl_pct = pct
        if pct < self.trough_pnl_pct:
            self.trough_pnl_pct = pct

        if pct < self.TARGET_MIN:
            return "AGGRESSIVE"
        elif pct > self.TARGET_MAX:
            return "CONSERVATIVE"
        else:
            return "BALANCED"

    def get_position_size_multiplier(self, current_pnl_sol: float) -> float:
        """Get position size multiplier based on daily P&L.

        Returns:
            1.2 if below target (increase size)
            0.6 if above max (reduce size)
            1.0 if in target range
        """
        pct = self.get_daily_pnl_pct(current_pnl_sol)

        if pct < self.TARGET_MIN:
            # Below target - increase size to catch up
            return 1.2
        elif pct > self.TARGET_MAX:
            # Above max - reduce size to protect gains
            return 0.6
        else:
            # In target range - normal sizing
            return 1.0

    def get_status(self, current_pnl_sol: float) -> dict[str, Any]:
        """Get full P&L monitor status."""
        pct = self.get_daily_pnl_pct(current_pnl_sol)
        mode = self.get_mode(current_pnl_sol)

        return {
            "daily_pnl_pct": round(pct, 3),
            "daily_pnl_sol": current_pnl_sol - self.daily_start_pnl,
            "mode": mode,
            "position_multiplier": self.get_position_size_multiplier(current_pnl_sol),
            "target_min": self.TARGET_MIN,
            "target_optimal": self.TARGET_OPTIMAL,
            "target_max": self.TARGET_MAX,
            "peak_pnl_pct": round(self.peak_pnl_pct, 3),
            "trough_pnl_pct": round(self.trough_pnl_pct, 3),
            "initial_capital_sol": self.initial_capital,
            "current_date": str(self.current_date),
            "is_on_target": self.TARGET_MIN <= pct <= self.TARGET_MAX,
        }


class AutoTraderCoordinator:
    """Autonomous trading coordinator with continuous learning.

    State Machine:
    - STOPPED: Not running, no trading
    - STARTING: Initializing, loading models
    - LEARNING: Observing market, collecting data, paper trading
    - TRADING: Actively generating and executing signals
    - PAUSED: Temporarily paused (user requested or circuit breaker)
    - DEGRADED: Running with reduced functionality (model issues)
    """

    def __init__(
        self,
        config: AutoTraderConfig,
        bandit: Optional["BanditTrainer"] = None,
        fraud_model: Optional["FraudModel"] = None,
        bias_calibrator: Optional["BiasCalibrator"] = None,
        db: Optional["DatabaseManager"] = None,
        outcome_tracker: Optional["OutcomeTracker"] = None,
        regime_detector: Optional["RegimeDetector"] = None,
        profitability_learner: Optional["ProfitabilityLearner"] = None,
        adaptive_limits_config: AdaptiveLimitConfig | None = None,
        adaptive_limits_bounds: LimitBounds | None = None,
    ):
        self.config = config
        self.bandit = bandit
        self.fraud_model = fraud_model
        self.bias_calibrator = bias_calibrator
        self.db = db
        self.outcome_tracker = outcome_tracker
        self.regime_detector = regime_detector
        self.profitability_learner = profitability_learner

        # Adaptive limits manager for autonomous scaling
        self.adaptive_limits = AdaptiveLimitsManager(
            config=adaptive_limits_config,
            bounds=adaptive_limits_bounds,
        )

        # Success rate strategy engine for smart parameter adjustment
        # Based on nasr's insight: "Why stop trading when you're making money?"
        self.success_rate_strategy = SuccessRateStrategy()
        self._last_strategy_tier = PerformanceTier.NEUTRAL
        # FIXED: Store original config for tier multiplier application
        # Prevents compounding drift when apply_to_config() runs repeatedly
        self._base_config_for_strategy: dict[str, Any] | None = None

        # Operation mode manager for mode switching
        self.mode_manager = AutonomousModeManager(
            initial_mode=OperationMode.BALANCED,
            autonomous_enabled=False,  # Start with manual mode control
        )

        # Phase 4: Autonomous self-tuning and P&L monitoring
        self.autonomous_tuner = AutonomousMode(self)
        self.pnl_monitor = DailyPnLMonitor(
            initial_capital_sol=config.max_position_sol * config.max_concurrent_positions
        )

        # GLM Week 1-4: Initialize advanced models
        self.alpha_decay = AlphaDecayAnalyzer()
        self.market_impact_model = SquareRootImpactModel()
        self.whale_tracker = WhaleWalletTracker()
        self.time_filter = TimePatternFilter()

        # ML Paper Trading configuration
        self.ml_config = MLPaperTradingConfig()

        # State
        self._state = AutoTraderState.STOPPED
        self._state_changed_at = datetime.now()

        # Metrics
        self.daily_metrics = DailyMetrics(date=datetime.now())
        self.learning_metrics = LearningMetrics()

        # Signal tracking
        self._pending_signals: dict[str, TradingSignal] = {}
        self._signal_history: deque = deque(maxlen=500)

        # Rolling trade outcome history for Kelly criterion computation
        self._trade_outcome_history: deque = deque(maxlen=50)

        # Cooldown tracking
        self._last_loss_time: datetime | None = None
        self._consecutive_losses: int = 0

        # Position tracking
        self._active_positions: dict[str, dict[str, Any]] = {}

        # Activity timestamps
        self.last_candidate_at: datetime | None = None
        self.last_signal_at: datetime | None = None
        self.last_outcome_at: datetime | None = None

        # Cached regime state for efficiency
        self._cached_regime: MarketRegime | None = None
        self._cached_regime_confidence: float = 0.5
        self._regime_cache_time: datetime | None = None
        self._regime_cache_ttl_seconds: int = 30
        self._outcome_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._hotpath_connected: bool = False
        self._execution_mode: str = self._derive_execution_mode()

    @property
    def state(self) -> AutoTraderState:
        return self._state

    def _set_state(self, new_state: AutoTraderState):
        """Transition to a new state."""
        old_state = self._state
        old_execution_mode = self._execution_mode
        self._state = new_state
        self._state_changed_at = datetime.now()
        self._execution_mode = self._derive_execution_mode()
        logger.info(
            "AutoTrader state: %s -> %s | execution_mode=%s",
            old_state.value,
            new_state.value,
            self._execution_mode,
        )
        self._log_execution_mode_transition(old_execution_mode, self._execution_mode)

    def set_hotpath_connected(self, connected: bool) -> None:
        """Track HotPath connectivity for execution-mode logging."""
        old_connected = self._hotpath_connected
        old_execution_mode = self._execution_mode
        self._hotpath_connected = connected
        self._execution_mode = self._derive_execution_mode()

        if old_connected != connected:
            logger.info(
                "HotPath connectivity updated: connected=%s state=%s execution_mode=%s",
                connected,
                self._state.value,
                self._execution_mode,
            )

        self._log_execution_mode_transition(old_execution_mode, self._execution_mode)

    def _derive_execution_mode(self) -> str:
        """Derive the active execution mode label exposed to logs and status."""
        if self._state == AutoTraderState.TRADING:
            if self.config.paper_mode:
                return "shadow_trading" if self._hotpath_connected else "paper_trading"
            return "live_trading"
        return self._state.value

    def _log_execution_mode_transition(self, old_mode: str, new_mode: str) -> None:
        """Emit explicit mode-transition logs for paper/shadow execution."""
        if old_mode == new_mode:
            return

        if old_mode == "learning" and new_mode == "paper_trading":
            logger.info("Mode transition: learning → paper_trading")
        elif old_mode == "paper_trading" and new_mode == "shadow_trading":
            logger.info("Mode transition: paper_trading → shadow_trading")
        elif old_mode == "shadow_trading" and new_mode == "paper_trading":
            logger.info(
                "Mode transition blocked: from=paper_trading, to=shadow_trading, "
                "reason=hotpath_disconnected"
            )

    def register_outcome_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback invoked after each finalized trade outcome."""
        if callback in self._outcome_callbacks:
            return
        self._outcome_callbacks.append(callback)

    async def start(self):
        """Start the AutoTrader."""
        if self._state != AutoTraderState.STOPPED:
            logger.warning(f"Cannot start from state {self._state.value}")
            return

        self._set_state(AutoTraderState.STARTING)

        # Initialize components
        try:
            # Reset daily metrics if new day
            if self.daily_metrics.date.date() != datetime.now().date():
                self.daily_metrics = DailyMetrics(date=datetime.now())

            # Check if we have enough data to start trading immediately
            if await self._check_ready_to_trade():
                self._set_state(AutoTraderState.TRADING)
                logger.info("AutoTrader started in TRADING mode")
            else:
                self.learning_metrics.start_time = datetime.now()
                self._set_state(AutoTraderState.LEARNING)
                logger.info("AutoTrader started in LEARNING mode")

        except Exception as e:
            logger.error("Failed to start AutoTrader: %s", e, exc_info=True)
            self._set_state(AutoTraderState.DEGRADED)

    async def stop(self):
        """Stop the AutoTrader."""
        logger.info("Stopping AutoTrader...")
        self._set_state(AutoTraderState.STOPPED)

        # Clear pending signals
        self._pending_signals.clear()

    async def pause(self, reason: str = "user_requested"):
        """Pause trading."""
        logger.info(f"Pausing AutoTrader: {reason}")
        self._set_state(AutoTraderState.PAUSED)

    async def resume(self):
        """Resume trading after pause."""
        if self._state != AutoTraderState.PAUSED:
            return

        if await self._check_ready_to_trade():
            self._set_state(AutoTraderState.TRADING)
        else:
            self._set_state(AutoTraderState.LEARNING)

    async def _check_ready_to_trade(self) -> bool:
        """Check if we have enough data to start trading."""
        bandit_pulls = self.bandit.total_pulls if self.bandit is not None else 0
        bandit_confidence = 0.0
        if self.bandit is not None and self.bandit.total_pulls > 0:
            bandit_confidence = self.bandit.get_recommendation().get("confidence", 0.0)

        # Explicit opt-out of learning requirements
        if self.config.skip_learning:
            logger.info(
                "Mode check: from=starting, to=paper, cond=skip_learning=true "
                f"bandit_pulls={bandit_pulls} "
                f"min_observations={self.config.min_observations_to_trade} "
                f"bandit_confidence={bandit_confidence:.4f} "
                f"min_confidence={self.config.min_confidence_to_trade:.4f}, ok=true"
            )
            return True

        # Paper mode: Allow trading even without bandit (will use default strategy)
        # This fixes the chicken-and-egg problem where we need trades to build bandit history
        if self.config.paper_mode and self.bandit is None:
            logger.info(
                "Mode check: from=starting, to=paper, cond=paper_mode=true bandit_available=false, "
                "ok=true (using default strategy)"
            )
            return True

        # Check bandit observations
        if self.bandit is None:
            logger.info(
                "Mode check: from=starting, to=paper, cond=skip_learning=false "
                "bandit_available=false, ok=false"
            )
            return False

        if self.bandit.total_pulls < self.config.min_observations_to_trade:
            logger.info(
                "Mode check: from=starting, to=paper, cond=skip_learning=false "
                f"bandit_pulls={bandit_pulls} "
                f"min_observations={self.config.min_observations_to_trade} "
                f"bandit_confidence={bandit_confidence:.4f} "
                f"min_confidence={self.config.min_confidence_to_trade:.4f}, ok=false"
            )
            return False

        # Check win rate
        recommendation = self.bandit.get_recommendation()
        if recommendation.get("confidence", 0) < self.config.min_confidence_to_trade:
            logger.info(
                "Mode check: from=starting, to=paper, cond=skip_learning=false "
                f"bandit_pulls={bandit_pulls} "
                f"min_observations={self.config.min_observations_to_trade} "
                f"bandit_confidence={recommendation.get('confidence', 0.0):.4f} "
                f"min_confidence={self.config.min_confidence_to_trade:.4f}, ok=false"
            )
            return False

        logger.info(
            "Mode check: from=starting, to=paper, cond=skip_learning=false "
            f"bandit_pulls={bandit_pulls} min_observations={self.config.min_observations_to_trade} "
            f"bandit_confidence={recommendation.get('confidence', 0.0):.4f} "
            f"min_confidence={self.config.min_confidence_to_trade:.4f}, ok=true"
        )
        return True

    async def process_candidate(self, candidate: TradingCandidate) -> TradingSignal | None:
        """Process a trading candidate and potentially generate a signal.

        Args:
            candidate: Token candidate to evaluate

        Returns:
            TradingSignal if the candidate passes all filters and we're in TRADING mode,
            None otherwise (or in LEARNING mode, records observation only)
        """
        self.last_candidate_at = datetime.now()
        logger.debug(
            f"Processing candidate: mint={candidate.token_mint} price_usd={candidate.price_usd} "
            f"state={self._state.value}"
        )
        self._cleanup_expired_signals()

        # Phase 4: Check for new day and reset P&L monitor
        self.pnl_monitor.reset_if_new_day(self.daily_metrics.total_pnl_sol)

        # Phase 4: Auto-tune if needed (only in TRADING mode)
        if self._state == AutoTraderState.TRADING:
            tuning_result = await self.autonomous_tuner.auto_tune_if_needed()
            if tuning_result:
                logger.info(f"🔄 Auto-tuning applied: {tuning_result.get('action', 'unknown')}")

        if self.config.decision_timeout_ms <= 0:
            return await self._process_candidate_inner(candidate)

        timeout_s = max(self.config.decision_timeout_ms, 50) / 1000.0
        try:
            return await asyncio.wait_for(
                self._process_candidate_inner(candidate),
                timeout=timeout_s,
            )
        except TimeoutError:
            logger.warning("Candidate processing timed out; skipping")
            await self._record_skip(candidate, "timeout")
            return None

    def _cleanup_expired_signals(self):
        """Remove expired signals from _pending_signals."""
        expired = [sid for sid, sig in self._pending_signals.items() if sig.is_expired]
        for sid in expired:
            del self._pending_signals[sid]
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired pending signals")

    async def _process_candidate_inner(self, candidate: TradingCandidate) -> TradingSignal | None:
        # 0. Validate candidate data
        candidate_validation = self._validate_candidate(candidate)
        if not candidate_validation["passed"]:
            logger.info(
                "Candidate rejected by validation: "
                f"mint={candidate.token_mint} reason={candidate_validation['reason']} "
                f"price_usd={candidate.price_usd}"
            )
            await self._record_skip(candidate, candidate_validation["reason"])
            return None

        # 1. Check daily limits
        limits_check = self._check_daily_limits()
        if not limits_check["passed"]:
            logger.debug(f"Daily limits exceeded: {limits_check['reason']}")
            await self._record_skip(candidate, limits_check["reason"])
            return None

        # 2. Evaluate fraud risk
        fraud_score = await self._evaluate_fraud_risk(candidate)
        if fraud_score > self.config.max_fraud_score:
            logger.info(
                "Candidate rejected by fraud screen: mint=%s fraud_score=%.4f max_allowed=%.4f",
                candidate.token_mint,
                fraud_score,
                self.config.max_fraud_score,
            )
            await self._record_skip(candidate, "high_risk")
            return None

        confidence = self._calculate_confidence(candidate, fraud_score)

        if self._state in (AutoTraderState.LEARNING, AutoTraderState.TRADING):
            self._record_learning_observation(
                candidate,
                fraud_score,
                confidence,
                allow_transition=self._state == AutoTraderState.LEARNING,
            )

        # 3. Apply basic filters
        filters = self._apply_filters(candidate)
        if not all(filters.values()):
            failed = [k for k, v in filters.items() if not v]
            logger.info(
                "Candidate rejected by filters: mint=%s failed=%s liquidity_usd=%.2f "
                "holder_count=%s age_seconds=%s volume_24h_usd=%.2f mint_authority=%s "
                "freeze_authority=%s",
                candidate.token_mint,
                ",".join(failed),
                candidate.liquidity_usd,
                candidate.holder_count,
                candidate.age_seconds,
                candidate.volume_24h_usd,
                candidate.mint_authority_enabled,
                candidate.freeze_authority_enabled,
            )
            await self._record_skip(candidate, "filter_reject")
            return None

        # 4. Select slippage via bandit
        slippage_bps, arm_name = self._select_slippage(candidate)
        slippage_bps = self._adjust_slippage(slippage_bps, candidate)

        # In LEARNING mode, observe only
        if self._state == AutoTraderState.LEARNING:
            await self._record_skip(candidate, "learning")
            return None

        # 6. Calculate position size
        position_size = self._calculate_position_size(candidate, fraud_score)
        if position_size < self.config.min_position_sol:
            await self._record_skip(candidate, "position_too_small")
            return None

        # In TRADING mode, check confidence threshold
        if confidence < self.config.min_confidence_to_trade:
            logger.debug(f"Confidence too low: {confidence:.2%}")
            await self._record_skip(candidate, "low_confidence")
            return None

        # Check cooldown
        if self._in_cooldown():
            logger.debug("In cooldown after loss")
            await self._record_skip(candidate, "cooldown")
            return None

        # Generate ML context for paper fill simulation
        ml_context = None
        if self.ml_config.enable_ml_context:
            try:
                ml_context = await self._generate_ml_context(candidate, fraud_score)
            except Exception as e:
                logger.warning(f"Failed to generate ML context: {e}")

        # Generate signal
        signal = TradingSignal(
            signal_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            token_mint=candidate.token_mint,
            token_symbol=candidate.token_symbol,
            pool_address=candidate.pool_address,
            action="BUY",
            position_size_sol=position_size,
            slippage_bps=slippage_bps,
            expected_price=candidate.price_usd,
            confidence=confidence,
            fraud_score=fraud_score,
            selected_arm=arm_name,
            expires_at=datetime.now() + timedelta(seconds=self.config.signal_expiry_seconds),
            filters_passed=filters,
            feature_snapshot=self._candidate_features(candidate),
            ml_context=ml_context,
        )
        self.last_signal_at = datetime.now()

        # Track signal
        self._pending_signals[signal.signal_id] = signal
        self._signal_history.append(signal)

        # Persist signal to database
        if self.db:
            try:
                await self.db.insert_trading_signal(
                    {
                        "signal_id": signal.signal_id,
                        "timestamp_ms": int(signal.timestamp.timestamp() * 1000),
                        "token_mint": signal.token_mint,
                        "token_symbol": signal.token_symbol,
                        "pool_address": signal.pool_address,
                        "action": signal.action,
                        "position_size_sol": signal.position_size_sol,
                        "slippage_bps": signal.slippage_bps,
                        "expected_price": signal.expected_price,
                        "confidence": signal.confidence,
                        "fraud_score": signal.fraud_score,
                        "status": "pending",
                        "expires_at_ms": int(signal.expires_at.timestamp() * 1000)
                        if signal.expires_at
                        else None,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to persist signal to database: {e}")

        logger.info(
            f"Generated signal: {candidate.token_symbol} "
            f"size={position_size:.3f} SOL, "
            f"confidence={confidence:.2%}, "
            f"slippage={slippage_bps}bps"
        )

        return signal

    def _validate_candidate(self, candidate: TradingCandidate) -> dict[str, Any]:
        """Validate candidate data before scoring."""
        if not candidate.token_mint:
            return {"passed": False, "reason": "invalid_mint"}
        if candidate.price_usd <= 0:
            return {"passed": False, "reason": "invalid_price"}
        if candidate.liquidity_usd <= 0:
            return {"passed": False, "reason": "invalid_liquidity"}
        if candidate.volume_24h_usd < 0:
            return {"passed": False, "reason": "invalid_volume"}
        if candidate.holder_count < 0:
            return {"passed": False, "reason": "invalid_holders"}
        return {"passed": True, "reason": None}

    def _check_daily_limits(self) -> dict[str, Any]:
        """Check if daily limits are exceeded."""
        # Reset if new day
        if self.daily_metrics.date.date() != datetime.now().date():
            self.daily_metrics = DailyMetrics(date=datetime.now())

        if self.daily_metrics.total_volume_sol >= self.config.max_daily_volume_sol:
            return {"passed": False, "reason": "daily_volume_exceeded"}

        if -self.daily_metrics.total_pnl_sol >= self.config.max_daily_loss_sol:
            return {"passed": False, "reason": "daily_loss_exceeded"}

        if self.daily_metrics.trade_count >= self.config.max_daily_trades:
            return {"passed": False, "reason": "daily_trades_exceeded"}

        if self.daily_metrics.active_positions >= self.config.max_concurrent_positions:
            return {"passed": False, "reason": "max_positions_exceeded"}

        return {"passed": True, "reason": None}

    async def _evaluate_fraud_risk(self, candidate: TradingCandidate) -> float:
        """Evaluate fraud/rug risk for a candidate."""
        if self.fraud_model is None:
            # Fallback to heuristic
            return self._heuristic_fraud_score(candidate)

        try:
            from ..training.fraud_model import TokenFeatures

            features = TokenFeatures(
                liquidity_usd=candidate.liquidity_usd,
                fdv_usd=candidate.fdv_usd,
                holder_count=candidate.holder_count,
                top_holder_pct=candidate.top_holder_pct,
                mint_authority_enabled=candidate.mint_authority_enabled,
                freeze_authority_enabled=candidate.freeze_authority_enabled,
                lp_burn_pct=candidate.lp_burn_pct,
                age_seconds=candidate.age_seconds,
                volume_24h=candidate.volume_24h_usd,
                price_change_1h=candidate.price_change_1h,
            )

            return self.fraud_model.predict_proba(features)

        except Exception as e:
            logger.warning(f"Fraud model prediction failed: {e}")
            return self._heuristic_fraud_score(candidate)

    def _heuristic_fraud_score(self, candidate: TradingCandidate) -> float:
        """Fallback heuristic fraud score."""
        score = 0.0

        if candidate.mint_authority_enabled:
            score += 0.3
        if candidate.freeze_authority_enabled:
            score += 0.2
        if candidate.liquidity_usd < 5000:
            score += 0.15
        if candidate.holder_count < 50:
            score += 0.1
        if candidate.top_holder_pct > 50:
            score += 0.15
        if candidate.lp_burn_pct < 50:
            score += 0.1

        return min(1.0, score)

    def _apply_filters(self, candidate: TradingCandidate) -> dict[str, bool]:
        """Apply basic trading filters."""
        filters = {
            "liquidity": candidate.liquidity_usd >= self.config.min_liquidity_usd,
            "fdv": candidate.fdv_usd <= self.config.max_fdv_usd,
            "holders": candidate.holder_count >= self.config.min_holder_count,
            "age": candidate.age_seconds >= self.config.min_token_age_seconds,
            "volume": candidate.volume_24h_usd >= self.config.min_volume_24h_usd,
            "mint_authority": not candidate.mint_authority_enabled,
            "freeze_authority": not candidate.freeze_authority_enabled,
        }

        if self.config.max_top_holder_pct is not None:
            filters["top_holder_pct"] = candidate.top_holder_pct <= self.config.max_top_holder_pct

        if self.config.max_volatility_1h is not None:
            filters["volatility_1h"] = candidate.volatility_1h <= self.config.max_volatility_1h

        return filters

    def _select_slippage(self, candidate: TradingCandidate) -> tuple[int, str]:
        """Select slippage using bandit."""
        if self.bandit is None:
            return self.config.default_slippage_bps, "default"

        # Get arm from bandit
        arm = self.bandit.select_arm()

        # Apply limits
        slippage = min(arm.slippage_bps, self.config.max_slippage_bps)

        return slippage, arm.name

    def _adjust_slippage(
        self,
        slippage_bps: int,
        candidate: TradingCandidate,
        position_size_sol: float = 0.0,
    ) -> int:
        """Adjust slippage based on liquidity, impact model, and time-of-day.

        Volatility is handled INSIDE the impact model only — not applied as a
        separate outer multiplier — to prevent double-counting.
        """
        liquidity_factor = 1.0
        if candidate.liquidity_usd > 0:
            liquidity_factor = min(2.0, max(0.5, 50000 / candidate.liquidity_usd))

        # GLM Week 3: Market Impact adjustment — use actual position size
        trade_size_usd = max(position_size_sol, 0.01) * 150.0
        impact_result = self.market_impact_model.predict_impact(
            trade_size_usd=trade_size_usd,
            daily_volume_usd=max(candidate.volume_24h_usd, 1.0),
            volatility=candidate.volatility_1h / 100.0,
        )
        # Only add impact that exceeds 10% of base — avoids inflating small trades
        extra_impact = max(0.0, impact_result.total_impact_bps - slippage_bps * 0.1)
        impact_adjusted = float(slippage_bps) + extra_impact

        # GLM Week 3: Time-of-day slippage adjustment
        time_adj_slippage = self.time_filter.get_slippage_adjustment(
            base_slippage_bps=impact_adjusted
        )

        # Liquidity factor only — volatility is already in the impact model
        adjusted = int(time_adj_slippage * liquidity_factor)
        return max(1, min(adjusted, self.config.max_slippage_bps))

    def _calculate_position_size(
        self,
        candidate: TradingCandidate,
        fraud_score: float,
    ) -> float:
        """Calculate position size using rolling Kelly criterion.

        Uses real trade outcomes (last 50 trades) to compute actual win_rate,
        avg_win, avg_loss. Falls back to conservative defaults when < 10
        historical trades exist.
        """
        # Compute rolling stats from actual trade outcomes
        win_rate, avg_win, avg_loss = self._get_rolling_trade_stats()

        # Kelly criterion: f* = (p * b - q) / b
        # where p = win_rate, q = 1-p, b = avg_win / avg_loss
        if avg_loss > 0 and avg_win > 0:
            b = avg_win / avg_loss
            kelly = (win_rate * b - (1 - win_rate)) / b
        else:
            kelly = 0

        # Apply half-Kelly for safety
        position = self.config.max_position_sol * self.config.kelly_fraction * max(0, kelly)

        # Scale down by fraud score
        position *= 1 - fraud_score

        # Scale down by liquidity
        if candidate.liquidity_usd > 0:
            liquidity_scale = min(
                1.0, candidate.liquidity_usd / max(self.config.min_liquidity_usd, 1)
            )
            position *= liquidity_scale

        # Apply survivorship discount
        if self.bias_calibrator:
            discount = self.bias_calibrator.get_coefficients().survivorship_discount
            position *= 1 - discount

        # Apply risk_mode multiplier
        risk_multipliers = {"conservative": 0.5, "balanced": 1.0, "aggressive": 1.5}
        position *= risk_multipliers.get(self.config.risk_mode, 1.0)

        # GLM Week 3: Time-of-day position sizing
        position = self.time_filter.get_position_size_adjustment(base_size=position)

        # GLM Week 1: Alpha decay — reduce position for stale signals
        signal_age_ms = int((datetime.now() - candidate.discovery_timestamp).total_seconds() * 1000)
        decay_mult = self.alpha_decay.get_position_size_multiplier(signal_age_ms)
        position *= decay_mult

        # Keep natural zero when edge is weak; do not force minimum size.
        if position <= 0:
            return 0.0

        # Clamp to risk limits and remaining daily budget
        remaining_budget = self.config.max_daily_volume_sol - self.daily_metrics.total_volume_sol
        if remaining_budget <= 0:
            return 0.0
        position = min(position, self.config.max_position_sol, remaining_budget)

        # If the edge is positive but Kelly is conservative, enforce a minimum
        # executable lot so valid opportunities are not dropped systematically.
        if 0 < position < self.config.min_position_sol:
            if remaining_budget >= self.config.min_position_sol:
                return self.config.min_position_sol
            return 0.0

        return position

    def _get_rolling_trade_stats(self) -> tuple:
        """Compute rolling win_rate, avg_win, avg_loss from recent trade outcomes.

        Returns (win_rate, avg_win, avg_loss) using the last 50 trades from
        outcome_tracker. Falls back to conservative defaults when insufficient
        history exists.
        """
        # Default conservative values
        default_win_rate = 0.5
        default_avg_win = 10.0  # 10%
        default_avg_loss = 5.0  # 5%

        if not self.outcome_tracker:
            return default_win_rate, default_avg_win, default_avg_loss

        # Get recent outcomes from the rolling deque
        recent = list(self._trade_outcome_history)
        if len(recent) < 10:
            # Also try bandit arm stats as secondary source
            if self.bandit and self.bandit.total_pulls > 0:
                best_arm = max(self.bandit.arms.values(), key=lambda a: a.mean_reward)
                if best_arm.count > 10 and best_arm.recent_rewards:
                    wins = sum(1 for r in best_arm.recent_rewards if r > 0)
                    win_rate = wins / len(best_arm.recent_rewards)
                    return win_rate, default_avg_win, default_avg_loss
            return default_win_rate, default_avg_win, default_avg_loss

        wins = [o["pnl_pct"] for o in recent if o["pnl_pct"] > 0]
        losses = [abs(o["pnl_pct"]) for o in recent if o["pnl_pct"] <= 0]

        win_rate = len(wins) / len(recent) if recent else default_win_rate
        avg_win = (sum(wins) / len(wins)) if wins else default_avg_win
        avg_loss = (sum(losses) / len(losses)) if losses else default_avg_loss

        return win_rate, avg_win, avg_loss

    def _calculate_confidence(
        self,
        candidate: TradingCandidate,
        fraud_score: float,
    ) -> float:
        """Calculate overall trading confidence using additive weighted model.

        Previous multiplicative approach (start at 0.5, multiply by 5 discount
        factors) collapsed to near-zero for almost all candidates. This additive
        model produces usable confidence scores that distribute across [0, 1].
        """
        # --- Component 1: Bandit confidence (weight=0.30) ---
        bandit_confidence = 0.5
        if self.bandit and self.bandit.total_pulls > 0:
            rec = self.bandit.get_recommendation()
            bandit_confidence = rec.get("confidence", 0.5)

        # --- Component 2: Fraud/safety confidence (weight=0.25) ---
        # fraud_score 0 = safe → 1.0, fraud_score 1 = scam → 0.0
        fraud_confidence = max(0.0, min(1.0, 1.0 - fraud_score))

        # --- Component 3: Liquidity & market quality (weight=0.20) ---
        liq_factor = min(1.0, candidate.liquidity_usd / 100_000)
        holder_factor = 1.0 if candidate.top_holder_pct <= 30 else 0.7
        vol_liq_factor = 1.0
        if candidate.liquidity_usd > 0:
            ratio = candidate.volume_24h_usd / candidate.liquidity_usd
            vol_liq_factor = min(1.0, max(0.3, ratio))
        market_quality = liq_factor * 0.4 + holder_factor * 0.3 + vol_liq_factor * 0.3

        # --- Component 4: Bias calibration (weight=0.15) ---
        bias_calibration = 1.0
        if self.bias_calibrator:
            discount = self.bias_calibrator.get_coefficients().survivorship_discount
            bias_calibration = max(0.0, min(1.0, 1.0 - discount))

        # --- Component 5: Recency factor (weight=0.10) ---
        # Newer tokens with healthy age get full score; very new = risky
        age_hours = candidate.age_seconds / 3600.0
        if age_hours < 0.5:
            recency_factor = 0.3  # Very new, risky
        elif age_hours < 6:
            recency_factor = 0.6 + 0.4 * min(1.0, age_hours / 6.0)
        else:
            recency_factor = 1.0

        # --- Component 6: Signal freshness via alpha decay (GLM Week 1) ---
        signal_age_ms = int((datetime.now() - candidate.discovery_timestamp).total_seconds() * 1000)
        decay_mult = self.alpha_decay.get_position_size_multiplier(signal_age_ms)
        freshness_score = max(0.0, min(1.0, decay_mult))

        # --- Component 7: Time-of-day boost (GLM Week 3) ---
        # Use the raw confidence_boost from the pattern rather than the
        # smoothed adjust_confidence() which always returns ~0.94-1.0
        time_pattern = self.time_filter.get_adjustment()
        time_score = max(0.0, min(1.0, time_pattern.confidence_boost))

        # --- Weighted sum ---
        # 6 active components, all clamped to [0, 1], weights sum to 1.0
        confidence = (
            0.25 * bandit_confidence
            + 0.20 * fraud_confidence
            + 0.20 * market_quality
            + 0.10 * bias_calibration
            + 0.10 * recency_factor
            + 0.05 * freshness_score
            + 0.10 * time_score
        )

        return max(0.0, min(1.0, confidence))

    def _candidate_features(self, candidate: TradingCandidate) -> dict[str, float]:
        """Map candidate to learning feature dictionary."""
        return {
            "liquidity_usd": candidate.liquidity_usd,
            "volume_24h_usd": candidate.volume_24h_usd,
            "holder_count": candidate.holder_count,
            "top_holder_pct": candidate.top_holder_pct,
            "price_change_1h": candidate.price_change_1h,
            "volatility_1h": candidate.volatility_1h,
            "age_seconds": candidate.age_seconds,
            "fdv_usd": candidate.fdv_usd,
            "lp_burn_pct": candidate.lp_burn_pct,
            "mint_authority_enabled": float(candidate.mint_authority_enabled),
            "freeze_authority_enabled": float(candidate.freeze_authority_enabled),
            "source_priority": 0.0,
            "time_of_day_hour": datetime.now().hour,
        }

    async def _generate_ml_context(
        self,
        candidate: TradingCandidate,
        fraud_score: float,
    ) -> MLTradeContext:
        """Generate ML trade context for paper fill simulation.

        Integrates outputs from:
        - FraudModel: fraud_score for risk adjustment
        - RegimeDetector: market regime and adjustments
        - ProfitabilityLearner: confidence calibration

        Args:
            candidate: The trading candidate being evaluated
            fraud_score: Pre-computed fraud score from FraudModel

        Returns:
            MLTradeContext with all ML-derived parameters
        """
        # 1. Get market regime from RegimeDetector
        regime = MarketRegime.CHOP
        regime_confidence = 0.5
        time_in_regime = 0
        transition_probability = 0.25

        if self.regime_detector is not None:
            try:
                # Check if we have a cached regime that's still valid
                now = datetime.now()
                if (
                    self._regime_cache_time is not None
                    and self._cached_regime is not None
                    and (now - self._regime_cache_time).total_seconds()
                    < self._regime_cache_ttl_seconds
                ):
                    regime = self._cached_regime
                    regime_confidence = self._cached_regime_confidence
                else:
                    # Build regime features from market data
                    from ..learning.regime_detector import RegimeFeatures

                    regime_features = RegimeFeatures(
                        volatility=candidate.volatility_1h,
                        momentum=candidate.price_change_1h / 100.0
                        if candidate.price_change_1h
                        else 0.0,
                        mev_activity=0.0,  # Would need MEV data from HotPath
                        liquidity_delta=0.0,  # Would need historical liquidity
                        volume_ratio=1.0,  # Would need volume comparison
                    )

                    # Get regime prediction
                    predicted_regime = self.regime_detector.predict(regime_features)
                    regime = MarketRegime.from_str(predicted_regime.value)

                    # Get regime state for additional metrics
                    regime_state = self.regime_detector.get_regime_state()
                    regime_confidence = regime_state.confidence
                    time_in_regime = regime_state.time_in_regime
                    transition_probability = regime_state.transition_probability

                    # Cache the regime
                    self._cached_regime = regime
                    self._cached_regime_confidence = regime_confidence
                    self._regime_cache_time = now

            except Exception as e:
                logger.warning(f"Regime detection failed: {e}, using default CHOP regime")

        # 2. Get profitability confidence from ProfitabilityLearner
        profitability_confidence = 0.5

        if self.profitability_learner is not None and self.profitability_learner.is_trained:
            try:
                features = self._candidate_features(candidate)
                feature_array = np.array([list(features.values())])
                _, calibrated = self.profitability_learner.predict(feature_array)
                profitability_confidence = float(calibrated[0])
            except Exception as e:
                logger.warning(f"Profitability prediction failed: {e}")

        # 3. Get deployer risk if available
        deployer_risk = None
        if self.fraud_model is not None:
            try:
                # Check if fraud model has deployer tracker
                if hasattr(self.fraud_model, "deployer_tracker"):
                    tracker = self.fraud_model.deployer_tracker
                    if hasattr(tracker, "get_deployer_for_token"):
                        deployer_wallet = tracker.get_deployer_for_token(candidate.token_mint)
                        if deployer_wallet:
                            profile = tracker.get_profile(deployer_wallet)
                            if profile:
                                deployer_risk = DeployerRisk(
                                    previous_tokens=profile.total_tokens_deployed,
                                    rug_count=profile.tokens_rugged,
                                    rug_rate=profile.rug_rate,
                                    avg_token_lifespan_hours=profile.avg_token_lifespan_seconds
                                    / 3600,
                                )
            except Exception as e:
                logger.debug(f"Could not get deployer risk: {e}")

        # 4. Create the ML trade context
        ml_context = MLTradeContext.create(
            fraud_score=fraud_score,
            regime=regime,
            regime_confidence=regime_confidence,
            profitability_confidence=profitability_confidence,
        )

        # Set deployer risk if available
        if deployer_risk:
            ml_context.deployer_risk = deployer_risk

        # Set additional fields
        ml_context.time_in_regime = time_in_regime
        ml_context.transition_probability = transition_probability

        # Log context creation at debug level
        logger.debug(
            f"ML context for {candidate.token_symbol}: "
            f"fraud={fraud_score:.2f}, regime={regime.value}, "
            f"regime_conf={regime_confidence:.2f}, "
            f"prof_conf={profitability_confidence:.2f}"
        )

        return ml_context

    def update_ml_config(self, config: MLPaperTradingConfig) -> None:
        """Update ML paper trading configuration."""
        self.ml_config = config
        logger.info(
            f"Updated ML paper trading config: enable_ml_context={config.enable_ml_context}"
        )

    async def _record_skip(self, candidate: TradingCandidate, reason: str) -> None:
        if not self.outcome_tracker:
            return

        from ..learning.outcome_tracker import SkipReason

        reason_map = {
            "daily_volume_exceeded": SkipReason.DAILY_LIMIT,
            "daily_loss_exceeded": SkipReason.DAILY_LIMIT,
            "daily_trades_exceeded": SkipReason.DAILY_LIMIT,
            "max_positions_exceeded": SkipReason.DAILY_LIMIT,
            "high_risk": SkipReason.HIGH_RISK,
            "filter_reject": SkipReason.UNKNOWN,
            "low_confidence": SkipReason.LOW_CONFIDENCE,
            "position_too_small": SkipReason.POSITION_TOO_SMALL,
            "cooldown": SkipReason.TIMEOUT,
            "timeout": SkipReason.TIMEOUT,
            "learning": SkipReason.MANUAL,
        }
        skip_reason = reason_map.get(reason, SkipReason.UNKNOWN)

        await self.outcome_tracker.record_skip(
            mint=candidate.token_mint,
            pool=candidate.pool_address,
            features=self._candidate_features(candidate),
            skip_reason=skip_reason,
            current_price=candidate.price_usd,
        )

        try:
            from .training_integration import get_training_integration

            training = get_training_integration()
            training.record_outcome(
                is_win=False,
                pnl_pct=0.0,
                pnl_sol=0.0,
                mint=candidate.token_mint,
                metadata={"source": "scan_outcome", "skip_reason": skip_reason.value},
            )
        except Exception as e:
            logger.debug(f"Training integration not available for skip outcome: {e}")

    def _record_learning_observation(
        self,
        candidate: TradingCandidate,
        fraud_score: float,
        confidence: float,
        allow_transition: bool = True,
    ):
        """Record an observation and optionally update learning-only state."""
        self.learning_metrics.observations += 1
        logger.info(
            f"Learning observation appended: mint={candidate.token_mint} "
            f"observations={self.learning_metrics.observations} confidence={confidence:.4f} "
            f"fraud_score={fraud_score:.4f} state={self._state.value}"
        )

        if not allow_transition:
            return

        # Simulate a paper trade
        self.learning_metrics.paper_trades += 1

        # Estimate if this would have been a win (simplified)
        if confidence > 0.6 and fraud_score < 0.3:
            self.learning_metrics.paper_wins += 1

        # Check if ready to transition to trading
        should_transition, reason, details = self._get_learning_transition_decision()
        logger.info(
            "Mode check: from=learning, to=paper, cond=%s, ok=%s",
            details,
            should_transition,
        )
        if should_transition:
            self._set_state(AutoTraderState.TRADING)
            logger.info(
                f"Transitioning to TRADING mode after {self.learning_metrics.observations} "
                "observations"
            )
        else:
            logger.info(
                "Mode transition blocked: from=learning, to=paper, reason=%s",
                reason,
            )

    def _should_transition_to_trading(self) -> bool:
        """Check if we should transition from LEARNING to TRADING.

        Uses OR logic: transition if ANY condition is met, or if skip_learning
        is configured. This avoids the chicken-and-egg problem where the system
        can't trade without history and can't build history without trading.
        """
        should_transition, _, _ = self._get_learning_transition_decision()
        return should_transition

    def _get_learning_transition_decision(self) -> tuple[bool, str, str]:
        """Explain why the learning-to-trading gate passes or blocks."""
        trained = bool(
            self.profitability_learner is not None and self.profitability_learner.is_trained
        )
        has_enough_observations = (
            self.learning_metrics.observations >= self.config.min_observations_to_trade
        )
        has_enough_duration = (
            self.learning_metrics.learning_duration_hours >= self.config.learning_duration_hours
        )
        has_good_win_rate = (
            self.learning_metrics.paper_win_rate >= self.config.min_win_rate_to_trade
            and self.learning_metrics.paper_trades >= 5
        )

        details = (
            f"observations={self.learning_metrics.observations}/"
            f"{self.config.min_observations_to_trade} "
            f"duration_hours={self.learning_metrics.learning_duration_hours:.2f}/"
            f"{self.config.learning_duration_hours} "
            f"paper_win_rate={self.learning_metrics.paper_win_rate:.4f}/"
            f"{self.config.min_win_rate_to_trade:.4f} "
            f"paper_trades={self.learning_metrics.paper_trades} "
            f"trained={trained} hotpath_connected={self._hotpath_connected} "
            f"paper_mode={self.config.paper_mode} skip_learning={self.config.skip_learning}"
        )

        if self.config.skip_learning:
            return True, "skip_learning_enabled", details
        if has_enough_observations:
            return True, "observations_threshold_met", details
        if has_enough_duration:
            return True, "learning_duration_met", details
        if has_good_win_rate:
            return True, "paper_win_rate_met", details

        reasons: list[str] = []
        if not has_enough_observations:
            reasons.append("observations_below_threshold")
        if not has_enough_duration:
            reasons.append("learning_duration_below_threshold")
        if not has_good_win_rate:
            reasons.append("paper_win_rate_below_threshold")
        if not trained:
            reasons.append("model_not_trained_yet")

        return False, ",".join(reasons), details

    def _in_cooldown(self) -> bool:
        """Check if we're in cooldown after a loss."""
        if self._last_loss_time is None:
            return False

        cooldown_end = self._last_loss_time + timedelta(
            seconds=self.config.cooldown_after_loss_seconds
        )
        return datetime.now() < cooldown_end

    async def record_trade_outcome(
        self,
        signal_id: str,
        pnl_sol: float,
        pnl_pct: float,
        slippage_realized_bps: int,
        included: bool = True,
        execution_latency_ms: int = 150,
        execution_details: dict | None = None,
    ):
        """Record the outcome of an executed trade with complete execution details.

        Updates bandit with reward and tracks metrics for continuous learning.

        Args:
            signal_id: Unique signal identifier
            pnl_sol: Realized P&L in SOL
            pnl_pct: Realized P&L as percentage
            slippage_realized_bps: Actual slippage in basis points
            included: Whether trade was included on-chain
            execution_latency_ms: Execution time in milliseconds
            execution_details: Optional dict with fee and execution metadata:
                - base_fee_lamports: Base transaction fee
                - priority_fee_lamports: Priority fee
                - jito_tip_lamports: Jito tip
                - dex_fee_lamports: DEX swap fee (calculated)
                - tx_signature: Blockchain transaction signature
                - slot: Blockchain slot number
                - execution_time_ms: Execution time
                - amount_out_lamports: Actual fill amount
        """
        self.last_outcome_at = datetime.now()
        # Find the signal
        signal = self._pending_signals.pop(signal_id, None)
        if signal is None:
            logger.warning(f"Signal {signal_id} not found")
            return

        # Normalize pnl_pct to percentage points (10 = +10%).
        normalized_pnl_pct = self._normalize_pnl_pct_points(pnl_pct)
        normalized_pnl_sol = pnl_sol

        # HotPath IPC can report zero PnL for open/incomplete lifecycle paths.
        # Use a deterministic execution-aware proxy so learning and aggregation
        # don't collapse to all-zero outcomes.
        if included and abs(normalized_pnl_pct) < 1e-9 and abs(normalized_pnl_sol) < 1e-12:
            normalized_pnl_pct = self._estimate_proxy_pnl_pct(
                signal=signal,
                slippage_realized_bps=slippage_realized_bps,
                execution_details=execution_details,
            )
            normalized_pnl_sol = signal.position_size_sol * (normalized_pnl_pct / 100.0)

        outcome = TradeOutcome(
            signal_id=signal_id,
            executed_at=datetime.now(),
            pnl_sol=normalized_pnl_sol,
            pnl_pct=normalized_pnl_pct,
            slippage_realized_bps=slippage_realized_bps,
            included=included,
            execution_latency_ms=execution_latency_ms,
        )

        # Update daily metrics
        self.daily_metrics.trade_count += 1
        self.daily_metrics.total_volume_sol += signal.position_size_sol
        self.daily_metrics.total_pnl_sol += normalized_pnl_sol

        if normalized_pnl_sol > 0:
            self.daily_metrics.win_count += 1
            self._consecutive_losses = 0
        else:
            self.daily_metrics.loss_count += 1
            self._consecutive_losses += 1
            self._last_loss_time = datetime.now()

        # Update max drawdown
        if self.daily_metrics.total_pnl_sol < -self.daily_metrics.max_drawdown_sol:
            self.daily_metrics.max_drawdown_sol = -self.daily_metrics.total_pnl_sol

        # Track outcome for rolling Kelly criterion
        self._trade_outcome_history.append(
            {
                "pnl_pct": normalized_pnl_pct,
                "pnl_sol": normalized_pnl_sol,
                "slippage_bps": slippage_realized_bps,
                "timestamp": datetime.now(),
            }
        )

        # Feed back to models for self-calibration
        try:
            # Alpha decay: record signal age vs outcome
            signal_age_ms = (
                int((outcome.executed_at - signal.timestamp).total_seconds() * 1000)
                if hasattr(signal, "timestamp") and signal.timestamp
                else 0
            )
            if signal_age_ms > 0:
                self.alpha_decay.record_outcome(
                    signal_timestamp_ms=int(signal.timestamp.timestamp() * 1000)
                    if hasattr(signal, "timestamp") and signal.timestamp
                    else 0,
                    execution_timestamp_ms=int(outcome.executed_at.timestamp() * 1000),
                    pnl_percent=normalized_pnl_pct,
                    token_mint=signal.token_mint,
                )

            # Market impact: record predicted vs actual for recalibration
            self.market_impact_model.record_actual_impact(
                predicted_impact_bps=signal.slippage_bps * 0.3,  # rough estimate of impact portion
                actual_impact_bps=float(slippage_realized_bps),
                trade_size_usd=signal.position_size_sol * 150.0,
                volatility=signal.feature_snapshot.get("volatility_1h", 30.0) / 100.0
                if hasattr(signal, "feature_snapshot") and signal.feature_snapshot
                else 0.3,
            )

            # Time filter: update session stats
            session = self.time_filter.get_session(datetime.now().hour)
            self.time_filter.update_session_stats(
                session=session,
                win_rate=1.0 if normalized_pnl_sol > 0 else 0.0,
                avg_volume_mult=1.0,
            )
        except Exception as e:
            logger.debug(f"Model feedback error (non-fatal): {e}")

        # Update bandit
        if self.bandit and signal.selected_arm in self.bandit.arms:
            reward = self._calculate_reward(outcome, signal)
            self.bandit.update(signal.selected_arm, reward, success=normalized_pnl_sol > 0)

        # Check if we should pause due to losses (skip in paper mode)
        if not self.config.paper_mode:
            if -self.daily_metrics.total_pnl_sol >= self.config.max_daily_loss_sol:
                await self.pause("daily_loss_limit")

            if self._consecutive_losses >= 5:
                await self.pause("consecutive_losses")

        # Auto-pause safety: check win rate after sufficient trades
        if self.config.auto_pause_enabled:
            total_trades = self.daily_metrics.win_count + self.daily_metrics.loss_count
            if total_trades >= self.config.min_trades_for_pause:
                current_win_rate = self.daily_metrics.win_rate
                if current_win_rate < self.config.min_win_rate_threshold:
                    logger.warning(
                        f"🚨 AUTO-PAUSE: Win rate {current_win_rate:.1%} below threshold "
                        f"{self.config.min_win_rate_threshold:.1%} after {total_trades} trades"
                    )
                    await self.pause(f"low_win_rate_{current_win_rate:.1%}")
                    # Log for monitoring
                    logger.info(
                        f"Auto-pause triggered - Review model performance. "
                        f"Win rate: {current_win_rate:.1%}, Trades: {total_trades}, "
                        f"Wins: {self.daily_metrics.win_count}, "
                        f"Losses: {self.daily_metrics.loss_count}"
                    )

        # Update signal status in database
        if self.db:
            try:
                await self.db.update_trading_signal(
                    signal_id,
                    {
                        "status": "completed",
                        "completed_at_ms": int(datetime.now().timestamp() * 1000),
                        "pnl_sol": normalized_pnl_sol,
                        "pnl_pct": normalized_pnl_pct,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to update signal status in database: {e}")

            # Store complete trade record with execution details
            try:
                trade_record = {
                    "id": signal_id,
                    "timestamp_ms": int(datetime.now().timestamp() * 1000),
                    "mint": signal.token_mint,
                    "symbol": signal.token_symbol,
                    "side": signal.action.lower(),
                    "amount_sol": signal.position_size_sol,
                    "amount_tokens": 0,  # Would need calculation from amount_out
                    "price": signal.expected_price,
                    "slippage_bps": slippage_realized_bps,
                    "quoted_slippage_bps": signal.slippage_bps,
                    "pnl_sol": normalized_pnl_sol,
                    "pnl_pct": normalized_pnl_pct,
                    "execution_mode": "live" if included else "paper",
                    "included": 1 if included else 0,
                    "latency_ms": execution_latency_ms,
                }

                # Add execution details if provided
                if execution_details:
                    trade_record.update(
                        {
                            "base_fee_lamports": execution_details.get("base_fee_lamports"),
                            "priority_fee_lamports": execution_details.get("priority_fee_lamports"),
                            "jito_tip_lamports": execution_details.get("jito_tip_lamports"),
                            "dex_fee_lamports": execution_details.get("dex_fee_lamports"),
                            "tx_signature": execution_details.get("tx_signature"),
                            "slot": execution_details.get("slot"),
                            "execution_time_ms": execution_details.get("execution_time_ms"),
                            "amount_out_lamports": execution_details.get("amount_out_lamports"),
                        }
                    )

                await self.db.insert_trade(trade_record)
                logger.debug(f"Stored complete trade record for {signal_id}")
            except Exception as e:
                logger.error(f"Failed to store trade record: {e}")

        logger.info(
            f"Trade outcome recorded: {signal.token_symbol} "
            f"PnL={normalized_pnl_pct:+.2f}%, slippage={slippage_realized_bps}bps"
        )

        if self.outcome_tracker:
            from ..learning.outcome_tracker import TradeResult

            trade_result = TradeResult(
                entry_price=signal.expected_price,
                pnl_pct=normalized_pnl_pct,
                pnl_sol=normalized_pnl_sol,
                slippage_bps=slippage_realized_bps,
                execution_mode="live" if included else "paper",
            )
            await self.outcome_tracker.record_trade(
                mint=signal.token_mint,
                pool=signal.pool_address,
                features=signal.feature_snapshot,
                profitability_score=0.0,
                confidence=signal.confidence,
                expected_return_pct=0.0,
                trade_result=trade_result,
            )

        # Feed outcome to profitability learner to close the learning loop
        if self.profitability_learner:
            try:
                self.profitability_learner.record_outcome(
                    features=signal.feature_snapshot,
                    pnl_pct=normalized_pnl_pct,
                    confidence=signal.confidence,
                )
            except Exception as e:
                logger.warning(f"Failed to update profitability learner: {e}")

        # Record outcome for adaptive limits manager
        is_win = normalized_pnl_sol > 0
        self.adaptive_limits.record_outcome(is_win=is_win, pnl_pct=normalized_pnl_pct)

        # Record outcome for success rate strategy engine
        # This enables smart parameter adjustment based on performance
        self.success_rate_strategy.record_outcome(is_win=is_win, pnl_pct=normalized_pnl_pct)

        # Record to training pipeline monitor for auto-training triggers
        try:
            from .training_integration import get_training_integration

            training = get_training_integration()
            training.record_outcome(
                is_win=is_win,
                pnl_pct=normalized_pnl_pct,
                pnl_sol=normalized_pnl_sol,
                mint=signal.token_mint,
                signal_id=signal_id,
            )
        except Exception as e:
            logger.debug(f"Training integration not available: {e}")

        # Record trade for mode manager
        self.mode_manager.record_trade()

        # Check and apply adaptive limit adjustments
        await self._check_adaptive_limits()

        # Notify observers (e.g., online learner ingestion) with a compact outcome payload.
        source = "live" if execution_details is not None else "paper"
        outcome_event = {
            "signal_id": signal_id,
            "timestamp_ms": int(datetime.now().timestamp() * 1000),
            "source": source,
            "included": included,
            "is_profitable": is_win,
            "pnl_pct": normalized_pnl_pct,
            "fraud_score": signal.fraud_score,
            "regime": (
                signal.ml_context.regime.value
                if signal.ml_context is not None
                else MarketRegime.CHOP.value
            ),
            "regime_confidence": (
                signal.ml_context.regime_confidence if signal.ml_context is not None else 0.5
            ),
            "features": dict(signal.feature_snapshot),
        }
        for callback in self._outcome_callbacks:
            try:
                callback(outcome_event)
            except Exception as e:
                logger.warning(f"Outcome callback error: {e}")

    @staticmethod
    def _normalize_pnl_pct_points(pnl_pct: float) -> float:
        """Normalize pnl_pct to percentage points.

        Accepts either decimal returns (0.10 = 10%) or pct points (10 = 10%).
        """
        if pnl_pct is None or not np.isfinite(pnl_pct):
            return 0.0
        return pnl_pct * 100.0 if abs(pnl_pct) <= 1.5 else pnl_pct

    def _estimate_proxy_pnl_pct(
        self,
        signal: TradingSignal,
        slippage_realized_bps: int,
        execution_details: dict | None,
    ) -> float:
        """Deterministic fallback PnL model when execution path reports zero PnL.

        The proxy combines confidence-derived edge minus execution costs.
        """
        # Edge is bounded to avoid unrealistic outcomes.
        confidence_edge_pct = (signal.confidence - self.config.min_confidence_to_trade) * 12.0
        confidence_edge_pct = max(-3.0, min(3.0, confidence_edge_pct))

        # Slippage is in bps -> convert to percentage points.
        execution_cost_pct = max(0.0, slippage_realized_bps) / 100.0

        fee_cost_pct = 0.0
        if execution_details and signal.position_size_sol > 0:
            total_fee_lamports = (
                int(execution_details.get("base_fee_lamports", 0) or 0)
                + int(execution_details.get("priority_fee_lamports", 0) or 0)
                + int(execution_details.get("jito_tip_lamports", 0) or 0)
                + int(execution_details.get("dex_fee_lamports", 0) or 0)
            )
            total_fee_sol = total_fee_lamports / 1_000_000_000.0
            fee_cost_pct = (total_fee_sol / signal.position_size_sol) * 100.0

        regime_bonus_pct = 0.0
        if signal.ml_context:
            if signal.ml_context.is_regime_stable():
                regime_bonus_pct += 0.25
            regime_bonus_pct -= signal.ml_context.survivorship_discount * 100.0 * 0.05

        return confidence_edge_pct + regime_bonus_pct - execution_cost_pct - fee_cost_pct

    async def _check_adaptive_limits(self) -> None:
        """Check and apply adaptive limit adjustments if needed.

        Also triggers:
        - Auto-save of config when performance exceeds targets
        - Mode evaluation for autonomous mode switching
        """
        # Only check in TRADING mode (not LEARNING)
        if self._state != AutoTraderState.TRADING:
            return

        # Check for auto-save opportunity first
        if self._should_auto_save_config():
            self._auto_save_successful_config()

        # Evaluate if adjustment is needed
        action = self.adaptive_limits.evaluate(self.config.to_dict())

        if action != LimitAction.HOLD:
            # Apply the adjustment with profitability-aware ceiling
            new_config_dict = self.adaptive_limits.apply_adjustment(
                self.config.to_dict(),
                action,
                daily_pnl_sol=self.daily_metrics.total_pnl_sol,
                current_trade_count=self.daily_metrics.trade_count,
            )

            # Update the config
            self.config = AutoTraderConfig.from_dict(new_config_dict)

            logger.info(
                f"🔄 Adaptive limits applied: {action.value} | "
                f"New limits: trades={self.config.max_daily_trades}, "
                f"positions={self.config.max_concurrent_positions}, "
                f"max_pos={self.config.max_position_sol} SOL, "
                f"confidence={self.config.min_confidence_to_trade:.0%}"
            )

            # If emergency reduce, also pause trading briefly
            if action == LimitAction.EMERGENCY_REDUCE:
                logger.warning("🚨 Emergency limit reduction - considering pause")
                # Could auto-pause here if needed

        # 🎯 Apply success rate strategy adjustments
        # Smart parameter tuning based on performance tier
        await self._apply_success_rate_strategy()

        # Evaluate mode switch (if autonomous enabled)
        await self._evaluate_mode_switch()

    async def _apply_success_rate_strategy(self) -> None:
        """Apply success rate strategy adjustments to trading parameters.

        This implements nasr's insight: "Why stop trading when making money?"
        - When HOT: Increase aggression (larger positions, lower thresholds)
        - When COLD: Decrease aggression (smaller positions, higher thresholds)
        - When FREEZING: Minimum risk only
        """
        # Evaluate current performance tier
        tier, adjustments = self.success_rate_strategy.evaluate()

        # Only apply if tier changed or periodically (every 10 trades)
        if tier != self._last_strategy_tier or self.daily_metrics.trade_count % 10 == 0:
            if tier != self._last_strategy_tier:
                logger.info(
                    f"🎯 Strategy tier: {self._last_strategy_tier.value} → {tier.value} | "
                    f"Win rate: {self.success_rate_strategy.get_win_rate():.0%}"
                )
                self._last_strategy_tier = tier

            # FIXED: Apply multipliers against the ORIGINAL base config, not the
            # already-adjusted one. This prevents compounding drift
            # (e.g., HOT 1.2x applied 5 times = 2.49x instead of 1.2x).
            if self._base_config_for_strategy is None:
                self._base_config_for_strategy = self.config.to_dict()
            base_config = self._base_config_for_strategy
            new_config = self.success_rate_strategy.apply_to_config(base_config, adjustments)

            # Only update if there are meaningful changes
            changes = []
            if (
                abs(new_config.get("max_position_sol", 0) - base_config.get("max_position_sol", 0))
                > 0.001
            ):
                changes.append(
                    f"position: {base_config['max_position_sol']:.3f} → "
                    f"{new_config['max_position_sol']:.3f}"
                )
            if (
                abs(
                    new_config.get("min_confidence_to_trade", 0)
                    - base_config.get("min_confidence_to_trade", 0)
                )
                > 0.01
            ):
                changes.append(
                    f"confidence: {base_config['min_confidence_to_trade']:.0%} → "
                    f"{new_config['min_confidence_to_trade']:.0%}"
                )

            if changes:
                self.config = AutoTraderConfig.from_dict(new_config)
                logger.info(f"📊 Strategy adjustments applied: {', '.join(changes)}")

            # Check if we should pause
            should_pause, reason = self.success_rate_strategy.should_pause()
            if should_pause:
                logger.warning(f"⚠️ {reason}")
                # Note: Actual pausing is left to the existing auto-pause logic
                # This just provides a signal

        # Log trading advice periodically
        if self.daily_metrics.trade_count % 20 == 0:
            advice = self.success_rate_strategy.get_trading_advice()
            logger.info(f"💡 Strategy: {advice[:150]}...")

    def _calculate_daily_pnl_pct(self) -> float:
        """Calculate daily P&L as percentage of deployed capital."""
        # Estimate daily capital based on max position * typical concurrent positions
        estimated_capital = self.config.max_position_sol * self.config.max_concurrent_positions
        if estimated_capital <= 0:
            return 0.0
        return (self.daily_metrics.total_pnl_sol / estimated_capital) * 100

    def _should_auto_save_config(self) -> bool:
        """Check if current config should be auto-saved.

        Auto-save when:
        - Win rate >= 55%
        - Daily P&L >= 0.8%
        - At least 20 trades
        - Max drawdown <= 15%
        """
        if self.daily_metrics.trade_count < 20:
            return False

        win_rate = self.daily_metrics.win_rate
        pnl_pct = self._calculate_daily_pnl_pct()
        max_dd = self.daily_metrics.max_drawdown_sol

        # Estimate capital for drawdown percentage
        estimated_capital = self.config.max_position_sol * self.config.max_concurrent_positions
        max_dd_pct = (max_dd / estimated_capital * 100) if estimated_capital > 0 else 100

        return win_rate >= 0.55 and pnl_pct >= 0.8 and max_dd_pct <= 15.0

    def _auto_save_successful_config(self) -> str | None:
        """Auto-save current config when performance exceeds targets.

        Returns:
            settings_id if saved, None if not saved
        """
        if not self._should_auto_save_config():
            return None

        try:
            store = VerifiedSettingsStore()

            # Calculate metrics for saving
            pnl_pct = self._calculate_daily_pnl_pct()
            estimated_capital = self.config.max_position_sol * self.config.max_concurrent_positions
            max_dd_pct = (
                (self.daily_metrics.max_drawdown_sol / estimated_capital * 100)
                if estimated_capital > 0
                else 0
            )

            # Calculate profit factor
            wins = self._trade_outcome_history and [
                o for o in self._trade_outcome_history if o["pnl_pct"] > 0
            ]
            losses = self._trade_outcome_history and [
                o for o in self._trade_outcome_history if o["pnl_pct"] <= 0
            ]
            total_wins = sum(o["pnl_pct"] for o in wins) if wins else 0
            total_losses = abs(sum(o["pnl_pct"] for o in losses)) if losses else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0

            # Estimate Sharpe (simplified)
            avg_pnl = self.adaptive_limits.get_rolling_pnl_avg()
            sharpe = avg_pnl / 5.0 if avg_pnl else 0  # Rough estimate

            settings_id = store.save_verified(
                settings=self.config.to_dict(),
                metrics={
                    "win_rate_pct": self.daily_metrics.win_rate * 100,
                    "daily_pnl_pct": pnl_pct,
                    "max_drawdown_pct": max_dd_pct,
                    "total_trades": self.daily_metrics.trade_count,
                    "sharpe_ratio": sharpe,
                    "profit_factor": profit_factor,
                    "avg_win_pct": total_wins / len(wins) if wins else 0,
                    "avg_loss_pct": total_losses / len(losses) if losses else 0,
                    "fill_rate_pct": 100.0,  # Would need actual fill rate
                },
                verified_by="live_auto_save",
                notes=f"Auto-saved: {self.daily_metrics.trade_count} trades, "
                f"{self.daily_metrics.win_rate:.0%} win rate, "
                f"{pnl_pct:.2f}% daily P&L",
                tags=["auto_saved", f"trades_{self.daily_metrics.trade_count}"],
            )

            if settings_id:
                logger.info(
                    f"💾 Auto-saved successful config: {settings_id} | "
                    f"win_rate={self.daily_metrics.win_rate:.0%}, "
                    f"daily_pnl={pnl_pct:.2f}%"
                )

            return settings_id

        except Exception as e:
            logger.warning(f"Failed to auto-save config: {e}")
            return None

    def load_best_verified_config(self) -> bool:
        """Load the best verified configuration from the store.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            store = VerifiedSettingsStore()
            best = store.get_best()

            if best is None:
                logger.info("No verified settings found to load")
                return False

            # Load the settings
            self.config = AutoTraderConfig.from_dict(best.settings)

            logger.info(
                f"✅ Loaded best verified config: {best.settings_id} | "
                f"win_rate={best.win_rate_pct:.1f}%, "
                f"daily_pnl={best.daily_pnl_pct:.2f}%"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to load best verified config: {e}")
            return False

    # ===========================================
    # Operation Mode Management
    # ===========================================

    def set_operation_mode(
        self,
        mode: OperationMode,
        reason: ModeTransitionReason = ModeTransitionReason.MANUAL,
        details: str = "",
        force: bool = False,
    ) -> bool:
        """Switch to a different operation mode.

        Args:
            mode: The target operation mode
            reason: Why the switch is happening
            details: Additional details about the switch
            force: Bypass cooldown and minimum trade checks

        Returns:
            True if mode was switched, False if blocked
        """
        # Record performance snapshot
        performance = {
            "win_rate": self.daily_metrics.win_rate,
            "daily_pnl_pct": self._calculate_daily_pnl_pct(),
            "trade_count": self.daily_metrics.trade_count,
            "rolling_win_rate": self.adaptive_limits.get_rolling_win_rate(),
        }

        # Attempt mode switch
        success = self.mode_manager.set_mode(
            new_mode=mode,
            reason=reason,
            details=details,
            performance_snapshot=performance,
            force=force,
        )

        if success:
            # Apply the new mode's configuration
            mode_config = get_mode_config(mode)
            config_dict = mode_config.to_config_dict()

            # Update config with mode preset (preserve some user settings)
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            logger.info(
                f"🎯 Operation mode changed to {mode.value} | "
                f"max_trades={self.config.max_daily_trades}, "
                f"max_pos={self.config.max_position_sol} SOL, "
                f"min_conf={self.config.min_confidence_to_trade:.0%}"
            )

        return success

    def get_operation_mode(self) -> OperationMode:
        """Get the current operation mode."""
        return self.mode_manager.current_mode

    def enable_autonomous_mode_switching(self, enabled: bool) -> None:
        """Enable or disable automatic mode switching.

        When enabled, the system will automatically switch between
        operation modes based on performance and market conditions.
        """
        self.mode_manager.set_autonomous(enabled)
        logger.info(f"🤖 Autonomous mode switching {'enabled' if enabled else 'disabled'}")

    async def _evaluate_mode_switch(self) -> None:
        """Evaluate if an automatic mode switch is warranted.

        Called after each trade outcome in TRADING mode.
        """
        if not self.mode_manager.autonomous_enabled:
            return

        # Get current metrics
        win_rate = self.adaptive_limits.get_rolling_win_rate()
        daily_pnl_pct = self._calculate_daily_pnl_pct()

        # Calculate max drawdown percentage
        estimated_capital = self.config.max_position_sol * self.config.max_concurrent_positions
        max_dd_pct = (
            (self.daily_metrics.max_drawdown_sol / estimated_capital * 100)
            if estimated_capital > 0
            else 0
        )

        # Get current regime if available
        current_regime = None
        if self._cached_regime:
            current_regime = self._cached_regime.value

        # Evaluate if switch is recommended
        recommended_mode = self.mode_manager.evaluate_auto_switch(
            win_rate=win_rate,
            daily_pnl_pct=daily_pnl_pct,
            max_drawdown_pct=max_dd_pct,
            current_regime=current_regime,
            daily_trades=self.daily_metrics.trade_count,
        )

        if recommended_mode and recommended_mode != self.mode_manager.current_mode:
            # Auto-switch to recommended mode
            self.set_operation_mode(
                mode=recommended_mode,
                reason=ModeTransitionReason.AUTO_OPTIMIZATION,
                details=f"Auto-switch based on win_rate={win_rate:.0%}, pnl={daily_pnl_pct:.2f}%",
                force=False,
            )

    def get_mode_status(self) -> dict[str, Any]:
        """Get detailed operation mode status."""
        return {
            "mode_manager": self.mode_manager.get_status(),
            "available_modes": list_available_modes(),
        }

    def _calculate_reward(self, outcome: TradeOutcome, signal: TradingSignal) -> float:
        """Calculate reward for bandit update."""
        # Base reward from PnL
        pnl_reward = np.tanh(outcome.pnl_pct / 10)

        # Penalty for non-inclusion
        if not outcome.included:
            return -0.5

        # Slippage efficiency bonus
        if signal.slippage_bps > 0:
            slippage_efficiency = (
                signal.slippage_bps - outcome.slippage_realized_bps
            ) / signal.slippage_bps
            slippage_bonus = slippage_efficiency * 0.2
        else:
            slippage_bonus = 0

        return pnl_reward + slippage_bonus

    def get_status(self) -> dict[str, Any]:
        """Get current AutoTrader status."""
        return {
            "state": self._state.value,
            "execution_mode": self._execution_mode,
            "state_changed_at": self._state_changed_at.isoformat(),
            "config": self.config.to_dict(),
            "last_status_at": datetime.now().isoformat(),
            "last_candidate_at": self.last_candidate_at.isoformat()
            if self.last_candidate_at
            else None,
            "last_signal_at": self.last_signal_at.isoformat() if self.last_signal_at else None,
            "last_outcome_at": self.last_outcome_at.isoformat() if self.last_outcome_at else None,
            "daily_metrics": {
                "date": self.daily_metrics.date.isoformat(),
                "volume_sol": self.daily_metrics.total_volume_sol,
                "pnl_sol": self.daily_metrics.total_pnl_sol,
                "pnl_pct": self._calculate_daily_pnl_pct(),
                "trade_count": self.daily_metrics.trade_count,
                "win_rate": self.daily_metrics.win_rate,
                "max_drawdown_sol": self.daily_metrics.max_drawdown_sol,
            },
            "learning_metrics": {
                "observations": self.learning_metrics.observations,
                "paper_trades": self.learning_metrics.paper_trades,
                "paper_win_rate": self.learning_metrics.paper_win_rate,
                "duration_hours": self.learning_metrics.learning_duration_hours,
            },
            "adaptive_limits": self.adaptive_limits.get_status(),
            "operation_mode": self.mode_manager.get_status(),
            "autonomous_tuner": self.autonomous_tuner.get_tuning_status(),
            "pnl_monitor": self.pnl_monitor.get_status(self.daily_metrics.total_pnl_sol),
            "pending_signals": len(self._pending_signals),
            "in_cooldown": self._in_cooldown(),
            "consecutive_losses": self._consecutive_losses,
        }

    def to_state(self) -> dict[str, Any]:
        """Export state for persistence."""
        return {
            "state": self._state.value,
            "execution_mode": self._execution_mode,
            "config": self.config.to_dict(),
            "daily_metrics": {
                "date": self.daily_metrics.date.isoformat(),
                "total_volume_sol": self.daily_metrics.total_volume_sol,
                "total_pnl_sol": self.daily_metrics.total_pnl_sol,
                "trade_count": self.daily_metrics.trade_count,
                "win_count": self.daily_metrics.win_count,
                "loss_count": self.daily_metrics.loss_count,
            },
            "learning_metrics": {
                "observations": self.learning_metrics.observations,
                "paper_trades": self.learning_metrics.paper_trades,
                "paper_wins": self.learning_metrics.paper_wins,
                "start_time": self.learning_metrics.start_time.isoformat()
                if self.learning_metrics.start_time
                else None,
            },
        }

    def load_state(self, state: dict[str, Any]):
        """Load state from persistence."""
        if "config" in state:
            loaded_config = AutoTraderConfig.from_dict(state["config"])
            # Preserve runtime-selected execution semantics and learning policy so
            # stale persisted artifacts cannot silently override current startup mode.
            loaded_config.paper_mode = self.config.paper_mode
            loaded_config.skip_learning = self.config.skip_learning
            loaded_config.min_token_age_seconds = self.config.min_token_age_seconds
            self.config = loaded_config

        if "daily_metrics" in state:
            dm = state["daily_metrics"]
            self.daily_metrics = DailyMetrics(
                date=datetime.fromisoformat(dm["date"])
                if isinstance(dm["date"], str)
                else dm["date"],
                total_volume_sol=dm.get("total_volume_sol", 0),
                total_pnl_sol=dm.get("total_pnl_sol", 0),
                trade_count=dm.get("trade_count", 0),
                win_count=dm.get("win_count", 0),
                loss_count=dm.get("loss_count", 0),
            )

        if "learning_metrics" in state:
            lm = state["learning_metrics"]
            self.learning_metrics.observations = lm.get("observations", 0)
            self.learning_metrics.paper_trades = lm.get("paper_trades", 0)
            self.learning_metrics.paper_wins = lm.get("paper_wins", 0)
            if lm.get("start_time"):
                self.learning_metrics.start_time = datetime.fromisoformat(lm["start_time"])

        self._execution_mode = state.get("execution_mode", self._derive_execution_mode())
        logger.info(f"Loaded AutoTrader state: {self.learning_metrics.observations} observations")
