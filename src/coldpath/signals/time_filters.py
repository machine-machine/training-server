"""
Time-of-Day Pattern Filter

Adjust strategy based on historical time patterns.
Memecoins often have predictable patterns based on trading activity cycles.

Target: +5% win rate improvement
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum


class MarketSession(Enum):
    """Trading sessions based on geographic activity."""

    ASIAN = "asian"  # UTC 0-8
    EUROPEAN = "european"  # UTC 8-16
    US = "us"  # UTC 14-22
    QUIET = "quiet"  # Low activity periods


@dataclass
class TimePattern:
    """Time-based pattern adjustments."""

    hour: int
    day_of_week: int  # 0=Monday, 6=Sunday
    session: MarketSession
    win_rate_adjustment: float
    volume_multiplier: float
    slippage_adjustment: float
    confidence_boost: float


@dataclass
class TimeFilterConfig:
    """Configuration for time pattern filter."""

    enable_session_adjustment: bool = True
    enable_weekend_penalty: bool = True
    enable_hour_patterns: bool = True
    min_confidence_boost: float = 0.8
    max_confidence_boost: float = 1.2


class TimePatternFilter:
    """
    Adjust strategy based on historical time patterns.

    Memecoins on Solana show predictable patterns:
    - US evening hours: High activity, good liquidity
    - Asian hours: Lower Solana activity
    - Weekends: Lower liquidity, higher slippage

    Usage:
        filter = TimePatternFilter()

        # Get current adjustment
        pattern = filter.get_adjustment()
        adjusted_confidence = filter.adjust_confidence(0.6)

        # Check if good time to trade
        if filter.is_favorable_time():
            proceed_with_trade()
    """

    # Pre-computed patterns from backtesting
    # Format: (hour_utc, day_of_week) -> (win_rate_adj, volume_mult, slippage_adj)
    # These are starting values - should be calibrated with actual data
    PATTERNS: dict[tuple[int, int], tuple[float, float, float]] = {
        # Sunday (day 6)
        (0, 6): (1.10, 1.3, 1.0),  # Sunday night - high activity starts
        (1, 6): (1.15, 1.4, 0.95),
        (2, 6): (1.10, 1.2, 1.0),
        (3, 6): (1.05, 1.1, 1.0),
        (12, 6): (0.85, 0.7, 1.2),  # Sunday noon - quiet
        # Monday (day 0)
        (0, 0): (1.05, 1.2, 1.0),
        (1, 0): (1.10, 1.3, 0.95),
        (14, 0): (1.0, 1.0, 1.0),  # US morning
        (18, 0): (1.05, 1.1, 0.98),
        (20, 0): (1.08, 1.15, 0.97),
        # Tuesday-Friday (days 1-4)
        (0, 1): (1.05, 1.15, 1.0),
        (1, 1): (1.08, 1.25, 0.98),
        (6, 1): (0.90, 0.8, 1.1),  # Asian morning
        (10, 1): (0.85, 0.7, 1.15),
        (14, 1): (1.0, 1.0, 1.0),
        (18, 1): (1.05, 1.1, 0.98),
        (21, 1): (1.02, 1.05, 1.0),
        # Wednesday (day 2) - often highest volume
        (18, 2): (1.10, 1.2, 0.95),
        (20, 2): (1.12, 1.25, 0.93),
        # Friday (day 4) - weekend approaching
        (20, 4): (0.95, 0.9, 1.05),
        (22, 4): (0.90, 0.8, 1.1),
        # Saturday (day 5)
        (0, 5): (0.95, 0.85, 1.1),
        (12, 5): (0.80, 0.6, 1.25),  # Saturday noon - very quiet
        (20, 5): (0.90, 0.75, 1.15),
    }

    # Session definitions (UTC hours)
    SESSION_HOURS = {
        MarketSession.ASIAN: (0, 8),
        MarketSession.EUROPEAN: (8, 16),
        MarketSession.US: (14, 22),
        MarketSession.QUIET: (22, 24),
    }

    def __init__(self, config: TimeFilterConfig | None = None):
        self.config = config or TimeFilterConfig()
        self._session_stats: dict[MarketSession, dict[str, float]] = {
            MarketSession.ASIAN: {"win_rate": 0.48, "avg_volume": 0.7},
            MarketSession.EUROPEAN: {"win_rate": 0.52, "avg_volume": 1.0},
            MarketSession.US: {"win_rate": 0.58, "avg_volume": 1.3},
            MarketSession.QUIET: {"win_rate": 0.45, "avg_volume": 0.5},
        }

    def get_session(self, hour: int) -> MarketSession:
        """Determine market session from hour."""
        for session, (start, end) in self.SESSION_HOURS.items():
            if start <= hour < end:
                return session
        return MarketSession.QUIET

    def get_adjustment(self, dt: datetime | None = None) -> TimePattern:
        """
        Get time-based adjustments for current time.

        Args:
            dt: Datetime to check (defaults to now UTC)

        Returns:
            TimePattern with adjustments
        """
        if dt is None:
            dt = datetime.now(UTC)

        hour = dt.hour
        day = dt.weekday()
        session = self.get_session(hour)

        # Check exact match first
        key = (hour, day)
        if key in self.PATTERNS:
            win_adj, vol_mult, slip_adj = self.PATTERNS[key]
        else:
            # Fall back to hour-only pattern
            hour_pattern = None
            for (h, _), pattern in self.PATTERNS.items():
                if h == hour:
                    hour_pattern = pattern
                    break

            if hour_pattern:
                win_adj, vol_mult, slip_adj = hour_pattern
                # Reduce adjustment for non-exact match
                win_adj = 1.0 + (win_adj - 1.0) * 0.7
                vol_mult = 1.0 + (vol_mult - 1.0) * 0.7
                slip_adj = 1.0 + (slip_adj - 1.0) * 0.7
            else:
                # Default: no adjustment
                win_adj, vol_mult, slip_adj = 1.0, 1.0, 1.0

        # Apply weekend penalty if enabled
        if self.config.enable_weekend_penalty and day >= 5:  # Sat=5, Sun=6
            win_adj *= 0.95
            vol_mult *= 0.85
            slip_adj *= 1.1

        # Session-based adjustments
        if self.config.enable_session_adjustment:
            session_stats = self._session_stats.get(session, {})
            session_win_rate = session_stats.get("win_rate", 0.5)
            session_win_adj = session_win_rate / 0.5  # Normalize to 1.0 baseline
            win_adj *= (1.0 + session_win_adj) / 2

        # Calculate confidence boost
        confidence_boost = self._calculate_confidence_boost(win_adj, vol_mult)
        confidence_boost = max(
            self.config.min_confidence_boost,
            min(self.config.max_confidence_boost, confidence_boost),
        )

        return TimePattern(
            hour=hour,
            day_of_week=day,
            session=session,
            win_rate_adjustment=win_adj,
            volume_multiplier=vol_mult,
            slippage_adjustment=slip_adj,
            confidence_boost=confidence_boost,
        )

    def _calculate_confidence_boost(self, win_adj: float, vol_mult: float) -> float:
        """Calculate confidence boost from time factors."""
        # Combine win rate adjustment and volume
        # Higher win rate + higher volume = more confidence
        score = (win_adj + vol_mult) / 2
        return score

    def adjust_confidence(
        self,
        base_confidence: float,
        dt: datetime | None = None,
    ) -> float:
        """
        Adjust signal confidence based on time patterns.

        Args:
            base_confidence: Original confidence (0-1)
            dt: Datetime to check

        Returns:
            Adjusted confidence (0-1)
        """
        if not self.config.enable_hour_patterns:
            return base_confidence

        pattern = self.get_adjustment(dt)

        # Apply smoothing to avoid wild swings
        adjusted = base_confidence * (0.7 + 0.3 * pattern.confidence_boost)

        return max(0.0, min(1.0, adjusted))

    def is_favorable_time(self, dt: datetime | None = None) -> bool:
        """
        Check if current time is favorable for trading.

        Args:
            dt: Datetime to check

        Returns:
            True if time is favorable
        """
        pattern = self.get_adjustment(dt)

        # Favorable if win rate adjustment > 1.0
        return pattern.win_rate_adjustment >= 1.0

    def get_slippage_adjustment(
        self,
        base_slippage_bps: float,
        dt: datetime | None = None,
    ) -> float:
        """
        Get adjusted slippage based on time.

        Args:
            base_slippage_bps: Base slippage in basis points
            dt: Datetime to check

        Returns:
            Adjusted slippage in basis points
        """
        pattern = self.get_adjustment(dt)
        return base_slippage_bps * pattern.slippage_adjustment

    def get_position_size_adjustment(
        self,
        base_size: float,
        dt: datetime | None = None,
    ) -> float:
        """
        Adjust position size based on time.

        Args:
            base_size: Base position size
            dt: Datetime to check

        Returns:
            Adjusted position size
        """
        pattern = self.get_adjustment(dt)

        # Scale position by win rate and volume
        # Lower volume = smaller position (liquidity concerns)
        adjustment = pattern.win_rate_adjustment * pattern.volume_multiplier
        adjustment = adjustment**0.5  # Square root to reduce extremes

        return base_size * min(1.5, max(0.5, adjustment))

    def should_reduce_risk(self, dt: datetime | None = None) -> bool:
        """
        Check if we should reduce risk at current time.

        Args:
            dt: Datetime to check

        Returns:
            True if risk should be reduced
        """
        pattern = self.get_adjustment(dt)

        # Reduce risk if:
        # - Low win rate period
        # - Low volume (liquidity risk)
        # - Weekend

        return pattern.win_rate_adjustment < 0.9 or pattern.volume_multiplier < 0.7

    def get_best_trading_hours(self, day: int) -> list[int]:
        """
        Get best trading hours for a given day.

        Args:
            day: Day of week (0=Monday, 6=Sunday)

        Returns:
            List of best hours (UTC)
        """
        good_hours = []

        for (h, d), (win_adj, vol_mult, _) in self.PATTERNS.items():
            if d == day and win_adj >= 1.05 and vol_mult >= 1.0:
                good_hours.append(h)

        return sorted(set(good_hours))

    def update_session_stats(
        self,
        session: MarketSession,
        win_rate: float,
        avg_volume_mult: float,
    ) -> None:
        """
        Update session statistics from live data.

        Args:
            session: Market session
            win_rate: Observed win rate
            avg_volume_mult: Average volume multiplier
        """
        self._session_stats[session] = {
            "win_rate": win_rate,
            "avg_volume": avg_volume_mult,
        }

    def get_summary(self, dt: datetime | None = None) -> dict:
        """
        Get a summary of time-based adjustments.

        Args:
            dt: Datetime to check

        Returns:
            Dictionary with adjustment summary
        """
        pattern = self.get_adjustment(dt)

        return {
            "hour_utc": pattern.hour,
            "day_of_week": pattern.day_of_week,
            "session": pattern.session.value,
            "is_favorable": self.is_favorable_time(dt),
            "win_rate_adj": round(pattern.win_rate_adjustment, 3),
            "volume_mult": round(pattern.volume_multiplier, 3),
            "slippage_adj": round(pattern.slippage_adjustment, 3),
            "confidence_boost": round(pattern.confidence_boost, 3),
            "should_reduce_risk": self.should_reduce_risk(dt),
        }


# Singleton instance
_default_filter: TimePatternFilter | None = None


def get_time_filter() -> TimePatternFilter:
    """Get the default time pattern filter instance."""
    global _default_filter
    if _default_filter is None:
        _default_filter = TimePatternFilter()
    return _default_filter
