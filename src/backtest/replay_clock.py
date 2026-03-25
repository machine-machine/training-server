"""
Replay clock for backtesting.

Enforces no-leakage by only exposing data available at simulation time.
"""

from dataclasses import dataclass
from typing import Optional


class ReplayClock:
    """
    Manages simulation time during backtesting.

    Ensures that strategy code cannot access future data.
    """

    def __init__(self, start_timestamp_ms: int, end_timestamp_ms: int):
        self.start_timestamp_ms = start_timestamp_ms
        self.end_timestamp_ms = end_timestamp_ms
        self.current_timestamp_ms = start_timestamp_ms

        # Alpha window: how far in the future we assume our signal is valid
        # Typically set to expected latency
        self.alpha_window_ms = 200

    def advance_to(self, timestamp_ms: int):
        """Advance clock to a new timestamp."""
        if timestamp_ms < self.current_timestamp_ms:
            raise ValueError(
                f"Cannot go back in time: {timestamp_ms} < {self.current_timestamp_ms}"
            )
        if timestamp_ms > self.end_timestamp_ms:
            raise ValueError(f"Cannot advance past end: {timestamp_ms}")

        self.current_timestamp_ms = timestamp_ms

    def is_available(self, data_timestamp_ms: int) -> bool:
        """Check if data at given timestamp is available now."""
        # Data is available if it's from before current time
        return data_timestamp_ms <= self.current_timestamp_ms

    def is_stale(self, data_timestamp_ms: int) -> bool:
        """Check if data is too old to be useful."""
        age_ms = self.current_timestamp_ms - data_timestamp_ms
        return age_ms > self.alpha_window_ms

    @property
    def now(self) -> int:
        """Current simulation timestamp."""
        return self.current_timestamp_ms

    @property
    def progress(self) -> float:
        """Fraction of backtest completed (0-1)."""
        total = self.end_timestamp_ms - self.start_timestamp_ms
        if total == 0:
            return 1.0
        elapsed = self.current_timestamp_ms - self.start_timestamp_ms
        return elapsed / total


class NoLeakageValidator:
    """
    Validates that features don't leak future information.

    Used during training data preparation.
    """

    def __init__(self, clock: ReplayClock):
        self.clock = clock

    def validate_feature(
        self,
        feature_name: str,
        feature_timestamp_ms: int,
        usage_timestamp_ms: int,
    ) -> bool:
        """
        Validate that a feature doesn't leak.

        Returns True if feature is valid (no leakage).
        """
        # Feature must be from before usage time
        if feature_timestamp_ms > usage_timestamp_ms:
            return False

        return True

    def validate_label(
        self,
        label_timestamp_ms: int,
        feature_timestamp_ms: int,
        min_gap_ms: int = 0,
    ) -> bool:
        """
        Validate that label doesn't leak into features.

        Label must be strictly after all features, with optional gap.
        """
        return label_timestamp_ms >= feature_timestamp_ms + min_gap_ms
