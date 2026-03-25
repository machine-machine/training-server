"""
Shadow Trading Metrics

Tracks and analyzes shadow trading performance including:
- P&L metrics (gross, net, fees)
- Risk metrics (Sharpe, Sortino, drawdown)
- Honeypot analysis
- Exit reason breakdown
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShadowTradeRecord:
    """Record of a closed shadow trade."""

    # Identifiers
    trade_id: str
    mint: str
    pool_address: str

    # Timing
    entry_time_ms: int
    exit_time_ms: int
    hold_duration_ms: int

    # Prices
    entry_price: float
    exit_price: float
    peak_price: float
    trough_price: float

    # P&L
    notional_sol: float
    gross_pnl_sol: float
    net_pnl_sol: float
    net_pnl_bps: int
    peak_pnl_bps: int
    trough_pnl_bps: int

    # Costs
    total_fees_sol: float
    total_slippage_sol: float
    entry_slippage_bps: int
    exit_slippage_bps: int

    # Exit
    exit_reason: str

    # Honeypot
    honeypot_score: float
    was_honeypot: bool
    honeypot_indicators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "mint": self.mint,
            "pool_address": self.pool_address,
            "entry_time_ms": self.entry_time_ms,
            "exit_time_ms": self.exit_time_ms,
            "hold_duration_ms": self.hold_duration_ms,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "peak_price": self.peak_price,
            "trough_price": self.trough_price,
            "notional_sol": self.notional_sol,
            "gross_pnl_sol": self.gross_pnl_sol,
            "net_pnl_sol": self.net_pnl_sol,
            "net_pnl_bps": self.net_pnl_bps,
            "peak_pnl_bps": self.peak_pnl_bps,
            "trough_pnl_bps": self.trough_pnl_bps,
            "total_fees_sol": self.total_fees_sol,
            "total_slippage_sol": self.total_slippage_sol,
            "entry_slippage_bps": self.entry_slippage_bps,
            "exit_slippage_bps": self.exit_slippage_bps,
            "exit_reason": self.exit_reason,
            "honeypot_score": self.honeypot_score,
            "was_honeypot": self.was_honeypot,
            "honeypot_indicators": self.honeypot_indicators,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShadowTradeRecord":
        """Create from dictionary."""
        return cls(
            trade_id=data["trade_id"],
            mint=data["mint"],
            pool_address=data.get("pool_address", ""),
            entry_time_ms=data["entry_time_ms"],
            exit_time_ms=data["exit_time_ms"],
            hold_duration_ms=data.get(
                "hold_duration_ms", data["exit_time_ms"] - data["entry_time_ms"]
            ),
            entry_price=data["entry_price"],
            exit_price=data["exit_price"],
            peak_price=data.get("peak_price", data["exit_price"]),
            trough_price=data.get("trough_price", data["exit_price"]),
            notional_sol=data["notional_sol"],
            gross_pnl_sol=data.get("gross_pnl_sol", data.get("net_pnl_sol", 0)),
            net_pnl_sol=data["net_pnl_sol"],
            net_pnl_bps=data["net_pnl_bps"],
            peak_pnl_bps=data.get("peak_pnl_bps", 0),
            trough_pnl_bps=data.get("trough_pnl_bps", 0),
            total_fees_sol=data.get("total_fees_sol", 0),
            total_slippage_sol=data.get("total_slippage_sol", 0),
            entry_slippage_bps=data.get("entry_slippage_bps", 0),
            exit_slippage_bps=data.get("exit_slippage_bps", 0),
            exit_reason=data["exit_reason"],
            honeypot_score=data.get("honeypot_score", 0),
            was_honeypot=data.get("was_honeypot", False),
            honeypot_indicators=data.get("honeypot_indicators", []),
        )


@dataclass
class PerformanceReport:
    """Complete performance report for shadow trading."""

    # Period
    generated_at: str
    period_start: str
    period_end: str
    trade_count: int

    # P&L
    gross_pnl_sol: float
    net_pnl_sol: float
    total_pnl_bps: int
    avg_trade_pnl_bps: float

    # Costs
    total_fees_sol: float
    total_slippage_sol: float
    fee_drag_bps: int

    # Win/Loss
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_win_bps: float
    avg_loss_bps: float
    profit_factor: float
    expectancy_bps: float

    # Risk
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_consecutive_losses: int
    volatility_bps: float
    calmar_ratio: float

    # Honeypot
    honeypot_trades: int
    honeypot_loss_sol: float
    honeypot_rate_pct: float
    honeypot_indicators_freq: dict[str, int]

    # Exit analysis
    exits_by_reason: dict[str, dict[str, Any]]
    avg_hold_time_ms: int

    # Slippage
    avg_entry_slippage_bps: float
    avg_exit_slippage_bps: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_at": self.generated_at,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "trade_count": self.trade_count,
            "gross_pnl_sol": self.gross_pnl_sol,
            "net_pnl_sol": self.net_pnl_sol,
            "total_pnl_bps": self.total_pnl_bps,
            "avg_trade_pnl_bps": self.avg_trade_pnl_bps,
            "total_fees_sol": self.total_fees_sol,
            "total_slippage_sol": self.total_slippage_sol,
            "fee_drag_bps": self.fee_drag_bps,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": self.win_rate_pct,
            "avg_win_bps": self.avg_win_bps,
            "avg_loss_bps": self.avg_loss_bps,
            "profit_factor": self.profit_factor,
            "expectancy_bps": self.expectancy_bps,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_consecutive_losses": self.max_consecutive_losses,
            "volatility_bps": self.volatility_bps,
            "calmar_ratio": self.calmar_ratio,
            "honeypot_trades": self.honeypot_trades,
            "honeypot_loss_sol": self.honeypot_loss_sol,
            "honeypot_rate_pct": self.honeypot_rate_pct,
            "honeypot_indicators_freq": self.honeypot_indicators_freq,
            "exits_by_reason": self.exits_by_reason,
            "avg_hold_time_ms": self.avg_hold_time_ms,
            "avg_entry_slippage_bps": self.avg_entry_slippage_bps,
            "avg_exit_slippage_bps": self.avg_exit_slippage_bps,
        }


class ShadowMetrics:
    """
    Aggregates and analyzes shadow trading metrics.

    Maintains rolling window and all-time statistics.
    """

    def __init__(
        self,
        rolling_window_size: int = 100,
        starting_equity: float = 1.0,
        annualization_factor: float = 365 * 100,
    ):
        self.rolling_window_size = rolling_window_size
        self.starting_equity = starting_equity
        self.annualization_factor = annualization_factor

        self.trades: deque = deque(maxlen=rolling_window_size)
        self.all_trades: list[ShadowTradeRecord] = []

        # Running totals
        self.total_gross_pnl = 0.0
        self.total_net_pnl = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        self.honeypot_count = 0
        self.honeypot_loss = 0.0

    def record_trade(self, trade: ShadowTradeRecord) -> None:
        """Record a new trade."""
        self.trades.append(trade)
        self.all_trades.append(trade)

        self.total_gross_pnl += trade.gross_pnl_sol
        self.total_net_pnl += trade.net_pnl_sol
        self.total_fees += trade.total_fees_sol
        self.total_slippage += trade.total_slippage_sol

        if trade.net_pnl_sol > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        if trade.was_honeypot:
            self.honeypot_count += 1
            if trade.net_pnl_sol < 0:
                self.honeypot_loss += abs(trade.net_pnl_sol)

        logger.debug(
            f"Recorded trade {trade.trade_id}: {trade.net_pnl_bps}bps, "
            f"honeypot={trade.was_honeypot}"
        )

    def generate_report(self, window: str = "rolling") -> PerformanceReport:
        """
        Generate performance report.

        Args:
            window: "rolling" for recent trades, "all" for all-time
        """
        if window == "rolling":
            trades = list(self.trades)
        else:
            trades = self.all_trades

        if not trades:
            return self._empty_report()

        return self._calculate_report(trades)

    def _calculate_report(self, trades: list[ShadowTradeRecord]) -> PerformanceReport:
        """Calculate all metrics for given trades."""
        # Timing
        period_start = datetime.fromtimestamp(trades[0].entry_time_ms / 1000).isoformat()
        period_end = datetime.fromtimestamp(trades[-1].exit_time_ms / 1000).isoformat()

        # P&L
        gross_pnl = sum(t.gross_pnl_sol for t in trades)
        net_pnl = sum(t.net_pnl_sol for t in trades)
        total_notional = sum(t.notional_sol for t in trades)
        total_pnl_bps = int(net_pnl / total_notional * 10000) if total_notional > 0 else 0
        avg_pnl_bps = sum(t.net_pnl_bps for t in trades) / len(trades)

        # Costs
        total_fees = sum(t.total_fees_sol for t in trades)
        total_slippage = sum(t.total_slippage_sol for t in trades)
        fee_drag = int(total_fees / total_notional * 10000) if total_notional > 0 else 0

        # Win/Loss
        winners = [t for t in trades if t.net_pnl_sol > 0]
        losers = [t for t in trades if t.net_pnl_sol <= 0]
        win_rate = len(winners) / len(trades) * 100

        avg_win = sum(t.net_pnl_bps for t in winners) / len(winners) if winners else 0.0
        avg_loss = sum(t.net_pnl_bps for t in losers) / len(losers) if losers else 0.0

        gross_wins = sum(t.net_pnl_sol for t in winners)
        gross_losses = sum(abs(t.net_pnl_sol) for t in losers)
        profit_factor = (
            gross_wins / gross_losses if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)
        )

        expectancy = (win_rate / 100) * avg_win + (1 - win_rate / 100) * avg_loss

        # Risk metrics
        returns = [t.net_pnl_bps / 10000 for t in trades]
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)
        max_dd = self._calculate_max_drawdown(trades)
        max_consec = self._calculate_max_consecutive_losses(trades)
        volatility = np.std(returns) * 10000 if len(returns) > 1 else 0

        calmar = sharpe / max(max_dd, 0.01) if max_dd > 0 else 0

        # Honeypot
        honeypots = [t for t in trades if t.was_honeypot]
        hp_loss = sum(abs(t.net_pnl_sol) for t in honeypots if t.net_pnl_sol < 0)
        hp_rate = len(honeypots) / len(trades) * 100

        indicator_freq: dict[str, int] = {}
        for t in honeypots:
            for ind in t.honeypot_indicators:
                indicator_freq[ind] = indicator_freq.get(ind, 0) + 1

        # Exit analysis
        exits: dict[str, dict[str, Any]] = {}
        for t in trades:
            reason = t.exit_reason
            if reason not in exits:
                exits[reason] = {"count": 0, "total_pnl": 0.0, "wins": 0}
            exits[reason]["count"] += 1
            exits[reason]["total_pnl"] += t.net_pnl_sol
            if t.net_pnl_sol > 0:
                exits[reason]["wins"] += 1

        for _reason, stats in exits.items():
            if stats["count"] > 0:
                stats["avg_pnl_bps"] = stats["total_pnl"] / stats["count"] * 10000
                stats["win_rate"] = stats["wins"] / stats["count"] * 100

        avg_hold = sum(t.hold_duration_ms for t in trades) // len(trades)

        # Slippage
        avg_entry_slip = sum(t.entry_slippage_bps for t in trades) / len(trades)
        avg_exit_slip = sum(t.exit_slippage_bps for t in trades) / len(trades)

        return PerformanceReport(
            generated_at=datetime.now().isoformat(),
            period_start=period_start,
            period_end=period_end,
            trade_count=len(trades),
            gross_pnl_sol=gross_pnl,
            net_pnl_sol=net_pnl,
            total_pnl_bps=total_pnl_bps,
            avg_trade_pnl_bps=avg_pnl_bps,
            total_fees_sol=total_fees,
            total_slippage_sol=total_slippage,
            fee_drag_bps=fee_drag,
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate_pct=win_rate,
            avg_win_bps=avg_win,
            avg_loss_bps=avg_loss,
            profit_factor=profit_factor,
            expectancy_bps=expectancy,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            max_consecutive_losses=max_consec,
            volatility_bps=volatility,
            calmar_ratio=calmar,
            honeypot_trades=len(honeypots),
            honeypot_loss_sol=hp_loss,
            honeypot_rate_pct=hp_rate,
            honeypot_indicators_freq=indicator_freq,
            exits_by_reason=exits,
            avg_hold_time_ms=avg_hold,
            avg_entry_slippage_bps=avg_entry_slip,
            avg_exit_slippage_bps=avg_exit_slip,
        )

    def _calculate_sharpe(self, returns: list[float]) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return float((mean / std) * np.sqrt(self.annualization_factor))

    def _calculate_sortino(self, returns: list[float]) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        mean = np.mean(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return 999.0
        downside_std = np.std(downside)
        if downside_std == 0:
            return 999.0
        return float((mean / downside_std) * np.sqrt(self.annualization_factor))

    def _calculate_max_drawdown(self, trades: list[ShadowTradeRecord]) -> float:
        """Calculate maximum drawdown percentage."""
        equity = self.starting_equity
        peak = equity
        max_dd = 0.0

        for t in trades:
            equity += t.net_pnl_sol
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        return max_dd * 100

    def _calculate_max_consecutive_losses(self, trades: list[ShadowTradeRecord]) -> int:
        """Calculate maximum consecutive losing trades."""
        max_streak = 0
        current_streak = 0

        for t in trades:
            if t.net_pnl_sol <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _empty_report(self) -> PerformanceReport:
        """Return empty report."""
        now = datetime.now().isoformat()
        return PerformanceReport(
            generated_at=now,
            period_start=now,
            period_end=now,
            trade_count=0,
            gross_pnl_sol=0.0,
            net_pnl_sol=0.0,
            total_pnl_bps=0,
            avg_trade_pnl_bps=0.0,
            total_fees_sol=0.0,
            total_slippage_sol=0.0,
            fee_drag_bps=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            avg_win_bps=0.0,
            avg_loss_bps=0.0,
            profit_factor=0.0,
            expectancy_bps=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            max_consecutive_losses=0,
            volatility_bps=0.0,
            calmar_ratio=0.0,
            honeypot_trades=0,
            honeypot_loss_sol=0.0,
            honeypot_rate_pct=0.0,
            honeypot_indicators_freq={},
            exits_by_reason={},
            avg_hold_time_ms=0,
            avg_entry_slippage_bps=0.0,
            avg_exit_slippage_bps=0.0,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get quick summary statistics."""
        total = self.winning_trades + self.losing_trades
        return {
            "total_trades": total,
            "win_rate_pct": self.winning_trades / total * 100 if total > 0 else 0,
            "total_net_pnl_sol": self.total_net_pnl,
            "total_fees_sol": self.total_fees,
            "honeypot_count": self.honeypot_count,
            "honeypot_loss_sol": self.honeypot_loss,
        }
