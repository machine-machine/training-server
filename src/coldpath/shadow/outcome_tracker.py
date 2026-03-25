"""
Shadow Trade Outcome Tracker

Tracks theoretical shadow trades and computes outcomes
by matching entry decisions with observed prices.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..storage import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class ShadowOutcome:
    """
    Computed outcome for a shadow trade.

    Created by matching shadow entry with observed exit prices.
    """

    # Trade identification
    shadow_id: str
    mint: str
    pool_address: str

    # Entry details (from shadow log)
    entry_time_ms: int
    entry_price: float
    entry_honeypot_score: float
    notional_sol: float

    # Computed exit (from price observation)
    exit_time_ms: int
    exit_price: float
    exit_reason: str

    # Computed P&L
    gross_pnl_sol: float
    net_pnl_sol: float
    net_pnl_bps: int

    # Price history during hold
    peak_price: float
    trough_price: float
    peak_pnl_bps: int
    trough_pnl_bps: int

    # Observed data
    observed_sell_tax_bps: int | None = None
    observed_liquidity_change_pct: float | None = None

    # Classification
    was_honeypot: bool = False
    honeypot_indicators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "shadow_id": self.shadow_id,
            "mint": self.mint,
            "pool_address": self.pool_address,
            "entry_time_ms": self.entry_time_ms,
            "entry_price": self.entry_price,
            "entry_honeypot_score": self.entry_honeypot_score,
            "notional_sol": self.notional_sol,
            "exit_time_ms": self.exit_time_ms,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "gross_pnl_sol": self.gross_pnl_sol,
            "net_pnl_sol": self.net_pnl_sol,
            "net_pnl_bps": self.net_pnl_bps,
            "peak_price": self.peak_price,
            "trough_price": self.trough_price,
            "peak_pnl_bps": self.peak_pnl_bps,
            "trough_pnl_bps": self.trough_pnl_bps,
            "observed_sell_tax_bps": self.observed_sell_tax_bps,
            "observed_liquidity_change_pct": self.observed_liquidity_change_pct,
            "was_honeypot": self.was_honeypot,
            "honeypot_indicators": self.honeypot_indicators,
        }


@dataclass
class PendingShadowTrade:
    """A shadow trade awaiting outcome computation."""

    shadow_id: str
    mint: str
    pool_address: str
    entry_time_ms: int
    entry_price: float
    entry_honeypot_score: float
    notional_sol: float
    entry_liquidity_usd: float

    # Exit parameters
    take_profit_bps: int
    stop_loss_bps: int
    trailing_stop_bps: int | None
    max_hold_ms: int

    # Price tracking
    peak_price: float = 0.0
    trough_price: float = float("inf")
    prices_observed: list[tuple] = field(default_factory=list)  # (timestamp_ms, price)


class OutcomeTracker:
    """
    Tracks shadow trades and computes outcomes from observed prices.

    Flow:
    1. HotPath logs shadow entry -> stored as PendingShadowTrade
    2. Price observations stream in
    3. For each price, check if exit conditions are met
    4. When exit triggered, compute full outcome
    """

    def __init__(
        self,
        db: Optional["DatabaseManager"] = None,
        fee_estimate_bps: int = 55,  # Jito-adjusted round-trip fees (25bps pool + 30bps MEV)
        slippage_estimate_bps: int = 100,  # Jito-adjusted round-trip slippage
    ):
        self.db = db
        self.fee_estimate_bps = fee_estimate_bps
        self.slippage_estimate_bps = slippage_estimate_bps

        # Pending trades awaiting outcome
        self.pending: dict[str, PendingShadowTrade] = {}

        # Computed outcomes (bounded to prevent unbounded memory growth)
        self.outcomes: list[ShadowOutcome] = []
        self._max_outcomes: int = 5000

    def register_shadow_entry(
        self,
        shadow_id: str,
        mint: str,
        pool_address: str,
        entry_price: float,
        honeypot_score: float,
        notional_sol: float,
        liquidity_usd: float,
        take_profit_bps: int = 500,
        stop_loss_bps: int = -1500,
        trailing_stop_bps: int | None = 300,
        max_hold_ms: int = 300_000,
    ) -> None:
        """
        Register a new shadow trade entry.

        Called when HotPath logs a shadow trade.
        """
        now_ms = int(time.time() * 1000)

        pending = PendingShadowTrade(
            shadow_id=shadow_id,
            mint=mint,
            pool_address=pool_address,
            entry_time_ms=now_ms,
            entry_price=entry_price,
            entry_honeypot_score=honeypot_score,
            notional_sol=notional_sol,
            entry_liquidity_usd=liquidity_usd,
            take_profit_bps=take_profit_bps,
            stop_loss_bps=stop_loss_bps,
            trailing_stop_bps=trailing_stop_bps,
            max_hold_ms=max_hold_ms,
            peak_price=entry_price,
            trough_price=entry_price,
        )

        self.pending[shadow_id] = pending

        logger.info(
            f"Registered shadow entry {shadow_id} for {mint} "
            f"at {entry_price}, hp_score={honeypot_score:.2f}"
        )

    def process_price_update(
        self,
        mint: str,
        current_price: float,
        current_liquidity_usd: float | None = None,
    ) -> list[ShadowOutcome]:
        """
        Process a price update for a mint.

        Checks all pending trades for this mint and triggers exits as needed.
        Returns list of newly computed outcomes.
        """
        now_ms = int(time.time() * 1000)
        new_outcomes: list[ShadowOutcome] = []
        to_remove: list[str] = []

        for shadow_id, pending in self.pending.items():
            if pending.mint != mint:
                continue

            # Update price tracking
            pending.prices_observed.append((now_ms, current_price))

            if current_price > pending.peak_price:
                pending.peak_price = current_price
            if current_price < pending.trough_price:
                pending.trough_price = current_price

            # Check exit conditions
            exit_reason = self._check_exit_conditions(
                pending, current_price, now_ms, current_liquidity_usd
            )

            if exit_reason:
                outcome = self._compute_outcome(
                    pending, current_price, now_ms, exit_reason, current_liquidity_usd
                )
                self.outcomes.append(outcome)
                new_outcomes.append(outcome)
                to_remove.append(shadow_id)

                logger.info(
                    f"Shadow trade {shadow_id} exited: {exit_reason}, pnl={outcome.net_pnl_bps}bps"
                )

        # Remove completed trades
        for shadow_id in to_remove:
            del self.pending[shadow_id]

        # Prune oldest outcomes to prevent unbounded memory growth
        if len(self.outcomes) > self._max_outcomes:
            self.outcomes = self.outcomes[-self._max_outcomes :]

        return new_outcomes

    def _check_exit_conditions(
        self,
        pending: PendingShadowTrade,
        current_price: float,
        now_ms: int,
        current_liquidity: float | None,
    ) -> str | None:
        """Check if exit conditions are met."""

        # Calculate current P&L
        pnl_bps = int((current_price - pending.entry_price) / pending.entry_price * 10000)

        # Take profit
        if pnl_bps >= pending.take_profit_bps:
            return "take_profit"

        # Stop loss
        if pnl_bps <= pending.stop_loss_bps:
            return "stop_loss"

        # Trailing stop
        if pending.trailing_stop_bps:
            peak_pnl_bps = int(
                (pending.peak_price - pending.entry_price) / pending.entry_price * 10000
            )
            # Only activate if we've been profitable
            if peak_pnl_bps > 100:
                from_peak_bps = int(
                    (current_price - pending.peak_price) / pending.peak_price * 10000
                )
                if from_peak_bps <= -pending.trailing_stop_bps:
                    return "trailing_stop"

        # Timeout
        hold_time = now_ms - pending.entry_time_ms
        if hold_time >= pending.max_hold_ms:
            return "timeout"

        # Liquidity drain (potential rug)
        if current_liquidity is not None and pending.entry_liquidity_usd > 0:
            liq_change = (
                (pending.entry_liquidity_usd - current_liquidity)
                / pending.entry_liquidity_usd
                * 100
            )
            if liq_change > 80:  # 80% liquidity drain
                return "liquidity_drained"

        # Rapid price crash (potential rug)
        if pnl_bps <= -8000:  # -80%
            return "honeypot_detected"

        return None

    def _compute_outcome(
        self,
        pending: PendingShadowTrade,
        exit_price: float,
        exit_time_ms: int,
        exit_reason: str,
        exit_liquidity: float | None,
    ) -> ShadowOutcome:
        """Compute full outcome for a shadow trade."""

        # Calculate P&L
        quantity = pending.notional_sol / pending.entry_price
        gross_pnl = (exit_price - pending.entry_price) * quantity

        # Estimate costs
        total_cost_bps = self.fee_estimate_bps + self.slippage_estimate_bps
        cost_sol = pending.notional_sol * total_cost_bps / 10000
        net_pnl = gross_pnl - cost_sol

        net_pnl_bps = int(net_pnl / pending.notional_sol * 10000)

        # Peak/trough P&L
        peak_pnl_bps = int((pending.peak_price - pending.entry_price) / pending.entry_price * 10000)
        trough_pnl_bps = int(
            (pending.trough_price - pending.entry_price) / pending.entry_price * 10000
        )

        # Honeypot classification
        was_honeypot = False
        honeypot_indicators: list[str] = []

        if exit_reason in ("honeypot_detected", "liquidity_drained"):
            was_honeypot = True

        # Check for honeypot indicators
        if net_pnl_bps <= -5000:  # Lost > 50%
            if exit_time_ms - pending.entry_time_ms < 60_000:  # Within 1 minute
                was_honeypot = True
                honeypot_indicators.append("rapid_crash")

        if pending.entry_honeypot_score > 0.6 and net_pnl_bps < -500:
            was_honeypot = True
            honeypot_indicators.append("high_initial_score")

        if exit_liquidity is not None and pending.entry_liquidity_usd > 0:
            liq_change = (
                (pending.entry_liquidity_usd - exit_liquidity) / pending.entry_liquidity_usd * 100
            )
            if liq_change > 50:
                honeypot_indicators.append(f"liquidity_drain_{int(liq_change)}pct")

        return ShadowOutcome(
            shadow_id=pending.shadow_id,
            mint=pending.mint,
            pool_address=pending.pool_address,
            entry_time_ms=pending.entry_time_ms,
            entry_price=pending.entry_price,
            entry_honeypot_score=pending.entry_honeypot_score,
            notional_sol=pending.notional_sol,
            exit_time_ms=exit_time_ms,
            exit_price=exit_price,
            exit_reason=exit_reason,
            gross_pnl_sol=gross_pnl,
            net_pnl_sol=net_pnl,
            net_pnl_bps=net_pnl_bps,
            peak_price=pending.peak_price,
            trough_price=pending.trough_price,
            peak_pnl_bps=peak_pnl_bps,
            trough_pnl_bps=trough_pnl_bps,
            observed_liquidity_change_pct=(
                (pending.entry_liquidity_usd - exit_liquidity) / pending.entry_liquidity_usd * 100
                if exit_liquidity is not None and pending.entry_liquidity_usd > 0
                else None
            ),
            was_honeypot=was_honeypot,
            honeypot_indicators=honeypot_indicators,
        )

    def force_close_all(self, current_prices: dict[str, float]) -> list[ShadowOutcome]:
        """Force close all pending trades (e.g., on shutdown)."""
        now_ms = int(time.time() * 1000)
        outcomes: list[ShadowOutcome] = []

        for _shadow_id, pending in list(self.pending.items()):
            price = current_prices.get(pending.mint, pending.entry_price)
            outcome = self._compute_outcome(pending, price, now_ms, "forced_close", None)
            self.outcomes.append(outcome)
            outcomes.append(outcome)

        self.pending.clear()
        return outcomes

    def get_pending_count(self) -> int:
        """Get number of pending trades."""
        return len(self.pending)

    def get_outcomes(self, limit: int | None = None) -> list[ShadowOutcome]:
        """Get computed outcomes."""
        if limit:
            return self.outcomes[-limit:]
        return self.outcomes

    def get_pending_summary(self) -> dict[str, Any]:
        """Get summary of pending trades."""
        if not self.pending:
            return {"count": 0, "mints": [], "oldest_age_ms": 0}

        now_ms = int(time.time() * 1000)
        ages = [now_ms - p.entry_time_ms for p in self.pending.values()]

        return {
            "count": len(self.pending),
            "mints": list(set(p.mint for p in self.pending.values())),
            "oldest_age_ms": max(ages) if ages else 0,
            "avg_age_ms": sum(ages) // len(ages) if ages else 0,
        }

    def export_for_optimization(self) -> list[dict[str, Any]]:
        """Export outcomes in format suitable for optimizer."""
        return [
            {
                "trade_id": o.shadow_id,
                "mint": o.mint,
                "net_pnl_bps": o.net_pnl_bps,
                "peak_pnl_bps": o.peak_pnl_bps,
                "trough_pnl_bps": o.trough_pnl_bps,
                "honeypot_score": o.entry_honeypot_score,
                "was_honeypot": o.was_honeypot,
                "exit_reason": o.exit_reason,
                "total_fee_bps": self.fee_estimate_bps + self.slippage_estimate_bps,
                "hold_duration_ms": o.exit_time_ms - o.entry_time_ms,
            }
            for o in self.outcomes
        ]
