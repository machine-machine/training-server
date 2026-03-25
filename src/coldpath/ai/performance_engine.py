"""
Performance feedback engine for real-time learning from trade outcomes.

Records every trade outcome with full context, computes parameter effectiveness
scores, tracks regime-specific performance, detects drift, and triggers
optimization when degradation is detected.

Usage:
    engine = PerformanceFeedbackEngine(parameter_store, drift_detector)
    await engine.initialize()

    result = await engine.process_trade_outcome(trade_outcome)
    if result.optimization_triggered:
        # Drift was detected, optimization has been queued
        pass
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Enhanced trade outcome with full context for learning."""

    trade_id: str
    timestamp: datetime
    params_version: int  # Which parameter version was used
    params_snapshot: dict[str, Any]  # Frozen copy of params at trade time

    # Outcome metrics
    pnl_sol: float
    pnl_pct: float
    fees_sol: float
    slippage_bps: int

    # Entry/exit details
    entry_price: float
    exit_price: float
    exit_reason: str  # "take_profit", "stop_loss", "trailing_stop", "timeout", "manual"
    hold_duration_seconds: int

    # Market context
    regime: str  # "bull", "bear", "chop", "mev_heavy"
    volatility: float
    liquidity_at_entry: float

    # Attribution
    ml_confidence: float | None = None
    source: str = "live"  # "live", "paper", "backtest"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "params_version": self.params_version,
            "params_snapshot": self.params_snapshot,
            "pnl_sol": self.pnl_sol,
            "pnl_pct": self.pnl_pct,
            "fees_sol": self.fees_sol,
            "slippage_bps": self.slippage_bps,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "hold_duration_seconds": self.hold_duration_seconds,
            "regime": self.regime,
            "volatility": self.volatility,
            "liquidity_at_entry": self.liquidity_at_entry,
            "ml_confidence": self.ml_confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradeOutcome":
        """Deserialize from dictionary."""
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            ts = datetime.now()

        return cls(
            trade_id=data["trade_id"],
            timestamp=ts,
            params_version=data.get("params_version", 0),
            params_snapshot=data.get("params_snapshot", {}),
            pnl_sol=data["pnl_sol"],
            pnl_pct=data["pnl_pct"],
            fees_sol=data.get("fees_sol", 0.0),
            slippage_bps=data.get("slippage_bps", 0),
            entry_price=data["entry_price"],
            exit_price=data["exit_price"],
            exit_reason=data.get("exit_reason", "unknown"),
            hold_duration_seconds=data.get("hold_duration_seconds", 0),
            regime=data.get("regime", "unknown"),
            volatility=data.get("volatility", 0.0),
            liquidity_at_entry=data.get("liquidity_at_entry", 0.0),
            ml_confidence=data.get("ml_confidence"),
            source=data.get("source", "live"),
        )


@dataclass
class FeedbackResult:
    """Result of processing a trade outcome."""

    trade_id: str
    processed: bool
    parameter_scores_updated: bool
    drift_checked: bool
    drift_detected: bool
    optimization_triggered: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimePerformance:
    """Performance metrics for a specific market regime."""

    regime: str
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl_sol: float = 0.0
    total_pnl_pct: float = 0.0
    total_fees_sol: float = 0.0
    avg_hold_duration_seconds: float = 0.0
    avg_slippage_bps: float = 0.0
    max_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0

    @property
    def win_rate(self) -> float:
        """Win rate as a ratio (0.0 to 1.0)."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_pnl_sol(self) -> float:
        """Average P&L per trade in SOL."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl_sol / self.total_trades

    @property
    def avg_pnl_pct(self) -> float:
        """Average P&L per trade as percentage."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl_pct / self.total_trades

    @property
    def net_pnl_sol(self) -> float:
        """Net P&L after fees."""
        return self.total_pnl_sol - self.total_fees_sol

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "regime": self.regime,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl_sol": round(self.total_pnl_sol, 6),
            "avg_pnl_sol": round(self.avg_pnl_sol, 6),
            "avg_pnl_pct": round(self.avg_pnl_pct, 4),
            "net_pnl_sol": round(self.net_pnl_sol, 6),
            "total_fees_sol": round(self.total_fees_sol, 6),
            "avg_hold_duration_seconds": round(self.avg_hold_duration_seconds, 1),
            "avg_slippage_bps": round(self.avg_slippage_bps, 1),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_losses": self.max_consecutive_losses,
        }


class PerformanceFeedbackEngine:
    """
    Real-time learning from trade outcomes.

    Responsibilities:
    - Record every trade outcome with full context
    - Compute parameter effectiveness scores
    - Track regime-specific performance
    - Detect performance drift
    - Trigger optimization when degradation is detected

    Data flow:
    1. Trade executed by HotPath
    2. Outcome reported via API or IPC
    3. This engine processes the outcome:
       a. Updates regime-specific performance
       b. Updates parameter effectiveness (via tracker)
       c. Checks for drift
       d. Queues optimization if degradation found
    """

    # Source weights for different trade origins
    SOURCE_WEIGHTS = {
        "live": 1.0,
        "paper": 0.5,
        "backtest": 0.3,
    }

    def __init__(
        self,
        parameter_store: Any = None,
        drift_detector: Any = None,
        parameter_tracker: Any = None,
        max_outcome_history: int = 10000,
        drift_check_interval: int = 50,
    ) -> None:
        """
        Initialize the performance feedback engine.

        Args:
            parameter_store: ParameterStore for version tracking.
            drift_detector: DriftDetector for degradation detection.
            parameter_tracker: ParameterEffectivenessTracker for learning.
            max_outcome_history: Maximum number of outcomes to retain.
            drift_check_interval: Check drift every N trades.
        """
        self.parameter_store = parameter_store
        self.drift_detector = drift_detector
        self.parameter_tracker = parameter_tracker
        self.max_outcome_history = max_outcome_history
        self.drift_check_interval = drift_check_interval

        # Trade outcome history (deque for efficient bounded history)
        self._outcomes: deque[TradeOutcome] = deque(maxlen=max_outcome_history)

        # Regime-specific performance tracking
        self._regime_performance: dict[str, RegimePerformance] = {}

        # Tracking consecutive losses (global)
        self._consecutive_losses: int = 0
        self._max_consecutive_losses: int = 0

        # Drift detection state
        self._trades_since_drift_check: int = 0

        # Callbacks
        self._on_drift_detected: list[Callable[[dict[str, Any]], None]] = []
        self._on_optimization_needed: list[Callable[[str, dict[str, Any]], None]] = []

        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the engine."""
        self._initialized = True
        logger.info("PerformanceFeedbackEngine initialized")

    async def process_trade_outcome(
        self,
        trade: TradeOutcome,
    ) -> FeedbackResult:
        """
        Process a completed trade and update all learning systems.

        Actions:
        1. Store outcome with attribution
        2. Update regime-specific performance
        3. Update parameter effectiveness scores
        4. Check for drift (every N trades)
        5. Trigger optimization if degradation detected

        Args:
            trade: The completed trade outcome.

        Returns:
            FeedbackResult with processing details.
        """
        async with self._lock:
            # 1. Store outcome
            self._outcomes.append(trade)

            # 2. Update regime performance
            self._update_regime_performance(trade)

            # 3. Update parameter effectiveness
            param_scores_updated = False
            if self.parameter_tracker:
                try:
                    await self._update_parameter_scores(trade)
                    param_scores_updated = True
                except Exception as exc:
                    logger.warning(
                        "Failed to update parameter scores for trade %s: %s",
                        trade.trade_id,
                        exc,
                    )

            # 4. Check for drift periodically
            self._trades_since_drift_check += 1
            drift_checked = False
            drift_detected = False
            optimization_triggered = False

            if self._trades_since_drift_check >= self.drift_check_interval and self.drift_detector:
                drift_checked = True
                self._trades_since_drift_check = 0

                try:
                    drift_report = await self.drift_detector.detect_drift(
                        outcomes=list(self._outcomes)
                    )
                    if drift_report.severity in ("moderate", "severe"):
                        drift_detected = True
                        logger.warning(
                            "Drift detected: %s (severity: %s)",
                            drift_report.description,
                            drift_report.severity,
                        )
                        for cb in self._on_drift_detected:
                            try:
                                cb(
                                    {
                                        "severity": drift_report.severity,
                                        "description": drift_report.description,
                                        "sharpe_degradation": drift_report.sharpe_degradation,
                                    }
                                )
                            except Exception:
                                pass

                        # 5. Trigger optimization if severe
                        if drift_report.severity == "severe":
                            optimization_triggered = True
                            for cb in self._on_optimization_needed:
                                try:
                                    cb(
                                        "drift_detected",
                                        {
                                            "drift_severity": drift_report.severity,
                                            "sharpe_degradation": (drift_report.sharpe_degradation),
                                            "win_rate_degradation": (
                                                drift_report.win_rate_degradation
                                            ),
                                        },
                                    )
                                except Exception:
                                    pass

                except Exception as exc:
                    logger.warning("Drift check failed: %s", exc)

            return FeedbackResult(
                trade_id=trade.trade_id,
                processed=True,
                parameter_scores_updated=param_scores_updated,
                drift_checked=drift_checked,
                drift_detected=drift_detected,
                optimization_triggered=optimization_triggered,
                details={
                    "regime": trade.regime,
                    "pnl_sol": trade.pnl_sol,
                    "source": trade.source,
                    "source_weight": self.SOURCE_WEIGHTS.get(trade.source, 0.1),
                    "total_outcomes": len(self._outcomes),
                },
            )

    def _update_regime_performance(self, trade: TradeOutcome) -> None:
        """Update regime-specific performance counters."""
        regime = trade.regime
        if regime not in self._regime_performance:
            self._regime_performance[regime] = RegimePerformance(regime=regime)

        perf = self._regime_performance[regime]
        perf.total_trades += 1
        perf.total_pnl_sol += trade.pnl_sol
        perf.total_pnl_pct += trade.pnl_pct
        perf.total_fees_sol += trade.fees_sol

        if trade.pnl_sol > 0:
            perf.winning_trades += 1
            perf.consecutive_losses = 0
            self._consecutive_losses = 0
        else:
            perf.consecutive_losses += 1
            self._consecutive_losses += 1
            if perf.consecutive_losses > perf.max_consecutive_losses:
                perf.max_consecutive_losses = perf.consecutive_losses
            if self._consecutive_losses > self._max_consecutive_losses:
                self._max_consecutive_losses = self._consecutive_losses

        # Update rolling averages
        n = perf.total_trades
        perf.avg_hold_duration_seconds = (
            perf.avg_hold_duration_seconds * (n - 1) + trade.hold_duration_seconds
        ) / n
        perf.avg_slippage_bps = (perf.avg_slippage_bps * (n - 1) + trade.slippage_bps) / n

    async def _update_parameter_scores(
        self,
        trade: TradeOutcome,
    ) -> None:
        """Update parameter effectiveness scores."""
        if not self.parameter_tracker:
            return

        # Update scores for each parameter in the snapshot
        for param_name, param_value in trade.params_snapshot.items():
            await self.parameter_tracker.update_parameter_score(
                parameter_name=param_name,
                parameter_value=param_value,
                regime=trade.regime,
                outcome=trade,
            )

    async def get_regime_performance(
        self,
        regime: str | None = None,
    ) -> dict[str, Any]:
        """
        Get performance metrics for a regime or all regimes.

        Args:
            regime: Specific regime to query (None for all).

        Returns:
            Dict with regime performance data.
        """
        if regime:
            perf = self._regime_performance.get(regime)
            if perf:
                return perf.to_dict()
            return {"regime": regime, "total_trades": 0}

        return {name: perf.to_dict() for name, perf in self._regime_performance.items()}

    async def get_current_regime(self) -> str:
        """
        Determine the current market regime from recent trades.

        Uses the most common regime in the last 20 trades.

        Returns:
            Regime string (e.g., "bull", "bear", "chop", "mev_heavy").
        """
        if not self._outcomes:
            return "unknown"

        recent = list(self._outcomes)[-20:]
        regime_counts: dict[str, int] = defaultdict(int)
        for outcome in recent:
            regime_counts[outcome.regime] += 1

        if not regime_counts:
            return "unknown"

        return max(regime_counts, key=regime_counts.get)  # type: ignore[arg-type]

    async def analyze(self) -> dict[str, Any]:
        """
        Analyze overall performance data for optimization.

        Returns a comprehensive performance report suitable for the
        AIStrategyOrchestrator's ANALYZING stage.
        """
        outcomes = list(self._outcomes)

        if not outcomes:
            return {
                "total_trades": 0,
                "sufficient_data": False,
            }

        pnls = [o.pnl_pct for o in outcomes]
        sol_pnls = [o.pnl_sol for o in outcomes]
        win_count = sum(1 for p in sol_pnls if p > 0)

        # Compute basic Sharpe approximation
        import math

        mean_pnl = sum(pnls) / len(pnls) if pnls else 0.0
        if len(pnls) > 1:
            variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
            std_pnl = math.sqrt(variance) if variance > 0 else 0.001
        else:
            std_pnl = 0.001

        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

        # Compute drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in sol_pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        return {
            "total_trades": len(outcomes),
            "sufficient_data": len(outcomes) >= 20,
            "win_rate": win_count / len(outcomes) if outcomes else 0.0,
            "total_pnl_sol": sum(sol_pnls),
            "avg_pnl_pct": mean_pnl,
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_sol": round(max_dd, 6),
            "consecutive_losses": self._consecutive_losses,
            "max_consecutive_losses": self._max_consecutive_losses,
            "regime_breakdown": {
                name: perf.to_dict() for name, perf in self._regime_performance.items()
            },
            "recent_trend": self._compute_recent_trend(outcomes),
        }

    def _compute_recent_trend(
        self,
        outcomes: list[TradeOutcome],
        window: int = 10,
    ) -> dict[str, Any]:
        """Compute recent performance trend."""
        if len(outcomes) < window:
            return {"trend": "insufficient_data", "window": window}

        recent = outcomes[-window:]
        older = (
            outcomes[-2 * window : -window]
            if len(outcomes) >= 2 * window
            else outcomes[: len(outcomes) - window]
        )

        if not older:
            return {"trend": "insufficient_data", "window": window}

        recent_avg = sum(o.pnl_pct for o in recent) / len(recent)
        older_avg = sum(o.pnl_pct for o in older) / len(older)

        recent_wr = sum(1 for o in recent if o.pnl_sol > 0) / len(recent)
        older_wr = sum(1 for o in older if o.pnl_sol > 0) / len(older)

        if recent_avg > older_avg * 1.1:
            trend = "improving"
        elif recent_avg < older_avg * 0.9:
            trend = "degrading"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "window": window,
            "recent_avg_pnl_pct": round(recent_avg, 4),
            "older_avg_pnl_pct": round(older_avg, 4),
            "recent_win_rate": round(recent_wr, 4),
            "older_win_rate": round(older_wr, 4),
        }

    def get_outcomes(
        self,
        limit: int | None = None,
        source: str | None = None,
        regime: str | None = None,
    ) -> list[TradeOutcome]:
        """
        Get trade outcomes with optional filtering.

        Args:
            limit: Maximum number of outcomes to return (most recent).
            source: Filter by source ("live", "paper", "backtest").
            regime: Filter by regime.

        Returns:
            List of TradeOutcome matching filters.
        """
        outcomes = list(self._outcomes)

        if source:
            outcomes = [o for o in outcomes if o.source == source]
        if regime:
            outcomes = [o for o in outcomes if o.regime == regime]
        if limit:
            outcomes = outcomes[-limit:]

        return outcomes

    def on_drift_detected(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Register a callback for drift detection events."""
        self._on_drift_detected.append(callback)

    def on_optimization_needed(
        self,
        callback: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """Register a callback for optimization trigger events."""
        self._on_optimization_needed.append(callback)

    @property
    def total_outcomes(self) -> int:
        """Total number of stored outcomes."""
        return len(self._outcomes)

    @property
    def consecutive_losses(self) -> int:
        """Current consecutive loss streak."""
        return self._consecutive_losses
