"""
Whale Wallet Tracking Integration

Track known profitable wallets for signal confirmation.
Smart money following can significantly improve trade timing.

Target: +5-10% win rate improvement
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class WalletLabel(Enum):
    """Classification of whale wallet behavior."""

    SNIPER = "sniper"  # Fast entry/exit, high win rate
    HOLDER = "holder"  # Long-term holds
    DUMP_DETECTOR = "dump"  # Often sells before dumps
    RUG_PULLER = "rug"  # Associated with rugs
    SMART_MONEY = "smart"  # Consistently profitable
    WHALE = "whale"  # Large holder, unknown strategy
    BOT = "bot"  # Automated trading


@dataclass
class WhaleSignal:
    """Signal from whale wallet activity."""

    wallet: str
    label: WalletLabel
    action: str  # "buy", "sell", "transfer"
    token: str
    amount_sol: float
    amount_usd: float
    timestamp_ms: int
    confidence: float
    tx_signature: str | None = None

    @property
    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        if self.action == "buy":
            return self.label in [
                WalletLabel.SMART_MONEY,
                WalletLabel.SNIPER,
                WalletLabel.WHALE,
            ]
        if self.action == "sell":
            return self.label in [
                WalletLabel.RUG_PULLER,
                WalletLabel.DUMP_DETECTOR,
            ]
        return False

    @property
    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        if self.action == "sell":
            return self.label in [
                WalletLabel.SMART_MONEY,
                WalletLabel.SNIPER,
                WalletLabel.WHALE,
            ]
        if self.action == "buy":
            return self.label in [
                WalletLabel.RUG_PULLER,
            ]
        return False


@dataclass
class Transfer:
    """Represents a token transfer."""

    from_address: str
    to_address: str
    token: str
    amount: float
    amount_sol: float
    timestamp_ms: int
    tx_signature: str


@dataclass
class WhaleTrackerConfig:
    """Configuration for whale tracker."""

    max_signals_per_token: int = 100
    signal_ttl_seconds: int = 3600  # 1 hour
    min_confidence: float = 0.3
    whale_list_path: str | None = None


class WhaleWalletTracker:
    """
    Track known profitable wallets for signal confirmation.

    Uses pre-compiled list of labeled wallets to detect smart money
    movements and incorporate into trading decisions.

    Usage:
        tracker = WhaleWalletTracker()
        tracker.load_whale_list("whale_wallets.json")

        # Check for whale activity
        signal = tracker.detect_whale_entry(token, recent_transfers)
        if signal and signal.is_bullish:
            confidence *= 1.1  # Boost confidence

        # Get overall sentiment
        sentiment = tracker.get_whale_sentiment(token)
    """

    # Default known whale wallets (example data)
    DEFAULT_WHALES: dict[str, str] = {
        # These would be replaced with real data
        # "wallet_address": "label"
    }

    def __init__(self, config: WhaleTrackerConfig | None = None):
        self.config = config or WhaleTrackerConfig()

        # Labeled wallets: address -> label
        self.labeled_wallets: dict[str, WalletLabel] = {}

        # Recent signals by token
        self.recent_signals: dict[str, list[WhaleSignal]] = defaultdict(list)

        # Performance tracking per wallet
        self.wallet_performance: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "total_pnl": 0.0}
        )

        # Load default whales
        for address, label in self.DEFAULT_WHALES.items():
            self.labeled_wallets[address] = WalletLabel(label)

        # Load from file if specified
        if self.config.whale_list_path:
            self.load_whale_list(self.config.whale_list_path)

    def load_whale_list(self, path: str) -> int:
        """
        Load pre-compiled list of whale wallets.

        Args:
            path: Path to JSON file with wallet labels

        Returns:
            Number of wallets loaded
        """
        if not os.path.exists(path):
            return 0

        with open(path) as f:
            data = json.load(f)

        count = 0
        for wallet, label_str in data.items():
            try:
                label = WalletLabel(label_str.lower())
                self.labeled_wallets[wallet] = label
                count += 1
            except ValueError:
                # Unknown label, skip
                pass

        return count

    def add_whale_wallet(
        self,
        address: str,
        label: WalletLabel,
    ) -> None:
        """Add a whale wallet to tracking."""
        self.labeled_wallets[address] = label

    def is_tracked_wallet(self, address: str) -> bool:
        """Check if address is a tracked whale."""
        return address in self.labeled_wallets

    def get_wallet_label(self, address: str) -> WalletLabel | None:
        """Get label for a wallet address."""
        return self.labeled_wallets.get(address)

    def detect_whale_entry(
        self,
        token: str,
        recent_transfers: list[Transfer],
    ) -> WhaleSignal | None:
        """
        Check if any tracked whale just entered this token.

        Args:
            token: Token mint address
            recent_transfers: Recent transfers for this token

        Returns:
            WhaleSignal if whale detected, None otherwise
        """
        for transfer in recent_transfers:
            if transfer.to_address in self.labeled_wallets:
                label = self.labeled_wallets[transfer.to_address]

                # Smart money or sniper entering = bullish signal
                if label in [WalletLabel.SMART_MONEY, WalletLabel.SNIPER, WalletLabel.WHALE]:
                    confidence = self._calculate_signal_confidence(
                        label=label,
                        amount_sol=transfer.amount_sol,
                    )

                    signal = WhaleSignal(
                        wallet=transfer.to_address,
                        label=label,
                        action="buy",
                        token=token,
                        amount_sol=transfer.amount_sol,
                        amount_usd=transfer.amount_sol * 150,  # Approximate SOL price
                        timestamp_ms=transfer.timestamp_ms,
                        confidence=confidence,
                        tx_signature=transfer.tx_signature,
                    )

                    self._record_signal(token, signal)
                    return signal

        return None

    def detect_whale_exit(
        self,
        token: str,
        recent_transfers: list[Transfer],
    ) -> WhaleSignal | None:
        """
        Check if any tracked whale is exiting this token.

        Args:
            token: Token mint address
            recent_transfers: Recent transfers for this token

        Returns:
            WhaleSignal if whale exit detected, None otherwise
        """
        for transfer in recent_transfers:
            if transfer.from_address in self.labeled_wallets:
                label = self.labeled_wallets[transfer.from_address]

                # Smart money or sniper exiting = bearish signal
                if label in [WalletLabel.SMART_MONEY, WalletLabel.SNIPER]:
                    confidence = self._calculate_signal_confidence(
                        label=label,
                        amount_sol=transfer.amount_sol,
                    )

                    signal = WhaleSignal(
                        wallet=transfer.from_address,
                        label=label,
                        action="sell",
                        token=token,
                        amount_sol=transfer.amount_sol,
                        amount_usd=transfer.amount_sol * 150,
                        timestamp_ms=transfer.timestamp_ms,
                        confidence=confidence,
                        tx_signature=transfer.tx_signature,
                    )

                    self._record_signal(token, signal)
                    return signal

        return None

    def _calculate_signal_confidence(
        self,
        label: WalletLabel,
        amount_sol: float,
    ) -> float:
        """
        Calculate confidence for a whale signal.

        Higher confidence for:
        - SMART_MONEY over other labels
        - Larger position sizes
        - Wallets with good track record
        """
        # Base confidence by label
        label_confidence = {
            WalletLabel.SMART_MONEY: 0.8,
            WalletLabel.SNIPER: 0.6,
            WalletLabel.WHALE: 0.5,
            WalletLabel.DUMP_DETECTOR: 0.4,
            WalletLabel.HOLDER: 0.3,
            WalletLabel.RUG_PULLER: 0.2,
            WalletLabel.BOT: 0.3,
        }

        base = label_confidence.get(label, 0.3)

        # Size bonus (larger = more significant)
        if amount_sol >= 100:
            size_bonus = 0.15
        elif amount_sol >= 50:
            size_bonus = 0.10
        elif amount_sol >= 10:
            size_bonus = 0.05
        else:
            size_bonus = 0.0

        return min(1.0, base + size_bonus)

    def _record_signal(self, token: str, signal: WhaleSignal) -> None:
        """Record signal for sentiment tracking."""
        self.recent_signals[token].append(signal)

        # Prune old signals
        self._prune_old_signals(token)

    def _prune_old_signals(self, token: str) -> None:
        """Remove signals older than TTL and clean empty token entries."""
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        ttl_ms = self.config.signal_ttl_seconds * 1000

        self.recent_signals[token] = [
            s for s in self.recent_signals[token] if now_ms - s.timestamp_ms < ttl_ms
        ][: self.config.max_signals_per_token]

        # Remove empty token entries to prevent dict bloat
        if not self.recent_signals[token]:
            del self.recent_signals[token]

    def get_whale_sentiment(self, token: str) -> float:
        """
        Aggregate whale sentiment for a token.

        Returns:
            -1.0 (bearish) to +1.0 (bullish)
        """
        self._prune_old_signals(token)

        signals = self.recent_signals.get(token, [])
        if not signals:
            return 0.0

        sentiment = 0.0
        total_weight = 0.0

        for signal in signals:
            weight = signal.confidence

            if signal.action == "buy":
                if signal.label in [WalletLabel.SMART_MONEY, WalletLabel.SNIPER]:
                    sentiment += weight * 1.0
                elif signal.label == WalletLabel.RUG_PULLER:
                    sentiment -= weight * 0.5  # Rug puller buying = suspicious
            elif signal.action == "sell":
                if signal.label in [WalletLabel.SMART_MONEY, WalletLabel.SNIPER]:
                    sentiment -= weight * 1.0
                elif signal.label == WalletLabel.DUMP_DETECTOR:
                    sentiment -= weight * 0.8

            total_weight += weight

        if total_weight == 0:
            return 0.0

        return max(-1.0, min(1.0, sentiment / total_weight))

    def get_tracked_whales_for_token(self, token: str) -> list[WhaleSignal]:
        """Get all tracked whale activity for a token."""
        self._prune_old_signals(token)
        return list(self.recent_signals.get(token, []))

    _MAX_WALLET_PERF_ENTRIES = 10_000

    def record_whale_performance(
        self,
        wallet: str,
        trade_pnl_pct: float,
        was_win: bool,
    ) -> None:
        """
        Record performance of a whale wallet trade.

        Used to improve confidence scoring over time.
        """
        # Evict oldest entry if at capacity
        if (
            wallet not in self.wallet_performance
            and len(self.wallet_performance) >= self._MAX_WALLET_PERF_ENTRIES
        ):
            oldest = next(iter(self.wallet_performance))
            del self.wallet_performance[oldest]

        perf = self.wallet_performance[wallet]
        perf["trades"] += 1
        perf["total_pnl"] += trade_pnl_pct
        if was_win:
            perf["wins"] += 1

    def get_whale_win_rate(self, wallet: str) -> float:
        """Get historical win rate for a whale wallet."""
        perf = self.wallet_performance.get(wallet)
        if not perf or perf["trades"] == 0:
            return 0.5  # Unknown, assume neutral

        return perf["wins"] / perf["trades"]

    def get_confidence_adjustment(
        self,
        token: str,
        base_confidence: float,
    ) -> float:
        """
        Adjust confidence based on whale sentiment.

        Args:
            token: Token mint address
            base_confidence: Original confidence score

        Returns:
            Adjusted confidence (0-1)
        """
        sentiment = self.get_whale_sentiment(token)

        # Boost if positive whale sentiment
        if sentiment > 0.3:
            adjustment = 1.0 + sentiment * 0.2  # Up to +20% boost
        elif sentiment < -0.3:
            adjustment = 1.0 + sentiment * 0.15  # Up to -15% penalty
        else:
            adjustment = 1.0  # No adjustment

        return max(0.0, min(1.0, base_confidence * adjustment))

    def should_follow_whale(
        self,
        token: str,
        signal: WhaleSignal | None = None,
    ) -> tuple[bool, str]:
        """
        Determine if we should follow whale activity.

        Returns:
            (should_follow, reason)
        """
        signals = self.get_tracked_whales_for_token(token)

        if not signals:
            return False, "No whale activity"

        # Check for recent smart money entry
        smart_money_buys = [
            s for s in signals if s.label == WalletLabel.SMART_MONEY and s.action == "buy"
        ]

        if smart_money_buys:
            return True, f"Smart money bought ({len(smart_money_buys)} entries)"

        # Check for snipers
        sniper_buys = [s for s in signals if s.label == WalletLabel.SNIPER and s.action == "buy"]

        if len(sniper_buys) >= 2:
            return True, f"Multiple snipers entered ({len(sniper_buys)})"

        # Check sentiment
        sentiment = self.get_whale_sentiment(token)
        if sentiment > 0.5:
            return True, f"Strong bullish sentiment ({sentiment:.2f})"

        return False, f"Mixed signals (sentiment: {sentiment:.2f})"


# Singleton instance
_default_tracker: WhaleWalletTracker | None = None


def get_whale_tracker() -> WhaleWalletTracker:
    """Get the default whale tracker instance."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = WhaleWalletTracker()
    return _default_tracker
