#
# volatility_sizing.py
# 2DEXY ColdPath
#
# Volatility-Scaled Position Sizing
#
# Scales position sizes so that portfolio volatility contribution is constant.
# A 0.1 SOL position in a 50% vol token has same risk as 0.5 SOL in a 10% vol token.
#

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class VolatilityConfig:
    """Configuration for volatility-scaled sizing."""

    # Target daily portfolio volatility (15% = moderate risk)
    target_daily_volatility: float = 0.15

    # Maximum position as percentage of capital (10% max)
    max_position_pct: float = 0.10

    # Minimum position as percentage of capital (0.1% min)
    min_position_pct: float = 0.001

    # Lookback periods for volatility calculation
    vol_lookback_periods: int = 20

    # Minimum volatility floor (don't scale below this)
    min_volatility_floor: float = 0.05  # 5% annualized

    # Maximum volatility cap (treat higher vols as this)
    max_volatility_cap: float = 2.0  # 200% annualized

    # Kelly fraction to use as base before vol adjustment
    base_kelly_fraction: float = 0.25

    # Smoothing factor for vol updates (0-1, higher = faster adaptation)
    vol_smoothing: float = 0.1


@dataclass
class PricePoint:
    """A price observation for volatility calculation."""

    timestamp_ms: int
    price: float


@dataclass
class VolatilityResult:
    """Result of volatility calculation."""

    realized_volatility: float  # Annualized realized volatility
    daily_volatility: float  # Daily volatility
    price_change_pct: float  # Price change over lookback
    samples_used: int  # Number of data points
    is_valid: bool  # Is calculation reliable?


class VolatilityCalculator:
    """
    Calculate realized volatility from price history.

    Uses log returns and standard deviation, annualized.
    """

    # Trading periods per day (for 15-min candles: 96)
    PERIODS_PER_DAY = 96
    # Trading days per year
    TRADING_DAYS_PER_YEAR = 365

    def __init__(self, lookback_periods: int = 20):
        self.lookback = lookback_periods
        self.price_history: deque[PricePoint] = deque(maxlen=lookback_periods + 1)

    def add_price(self, price: float, timestamp_ms: int | None = None) -> None:
        """Add a price observation."""
        if timestamp_ms is None:
            timestamp_ms = int(datetime.now(UTC).timestamp() * 1000)

        self.price_history.append(PricePoint(timestamp_ms=timestamp_ms, price=price))

    def calculate(self) -> VolatilityResult:
        """
        Calculate realized volatility from price history.

        Returns annualized volatility.
        """
        prices = list(self.price_history)

        if len(prices) < 3:
            return VolatilityResult(
                realized_volatility=0.0,
                daily_volatility=0.0,
                price_change_pct=0.0,
                samples_used=len(prices),
                is_valid=False,
            )

        # Calculate log returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i].price > 0 and prices[i - 1].price > 0:
                log_return = math.log(prices[i].price / prices[i - 1].price)
                returns.append(log_return)

        if len(returns) < 2:
            return VolatilityResult(
                realized_volatility=0.0,
                daily_volatility=0.0,
                price_change_pct=0.0,
                samples_used=len(prices),
                is_valid=False,
            )

        # Calculate standard deviation of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_return = math.sqrt(variance)

        # Annualize: multiply by sqrt(periods_per_year)
        # For 15-min candles: 96 periods/day * 365 days = 35,040 periods/year
        periods_per_year = self.PERIODS_PER_DAY * self.TRADING_DAYS_PER_YEAR
        annualized_vol = std_return * math.sqrt(periods_per_year)

        # Daily volatility
        daily_vol = annualized_vol / math.sqrt(self.TRADING_DAYS_PER_YEAR)

        # Price change over lookback
        price_change = (
            (prices[-1].price - prices[0].price) / prices[0].price if prices[0].price > 0 else 0.0
        )

        return VolatilityResult(
            realized_volatility=annualized_vol,
            daily_volatility=daily_vol,
            price_change_pct=price_change * 100,
            samples_used=len(prices),
            is_valid=True,
        )

    def reset(self) -> None:
        """Clear price history."""
        self.price_history.clear()


class VolatilityScaledSizing:
    """
    Position sizing based on realized volatility.

    Core principle: Scale positions so that portfolio volatility
    contribution is constant regardless of asset volatility.

    Formula:
        position = capital * min(adjusted_kelly, max_pct)

    Where:
        adjusted_kelly = base_kelly * (target_vol / current_vol)

    This means:
    - High volatility assets get SMALLER positions
    - Low volatility assets get LARGER positions
    - Both contribute same risk to portfolio
    """

    def __init__(self, config: VolatilityConfig | None = None):
        self.config = config or VolatilityConfig()

        # Per-token volatility calculators
        self._vol_calculators: dict[str, VolatilityCalculator] = {}

        # Cached volatility estimates (for tokens without price history)
        self._vol_cache: dict[str, float] = {}

        # Default volatility for unknown tokens (50% annualized = typical memecoin)
        self._default_volatility = 0.50

    def _get_calculator(self, token_mint: str) -> VolatilityCalculator:
        """Get or create volatility calculator for a token."""
        if token_mint not in self._vol_calculators:
            self._vol_calculators[token_mint] = VolatilityCalculator(
                lookback_periods=self.config.vol_lookback_periods
            )
        return self._vol_calculators[token_mint]

    def update_price(
        self,
        token_mint: str,
        price: float,
        timestamp_ms: int | None = None,
    ) -> None:
        """Update price observation for a token."""
        calc = self._get_calculator(token_mint)
        calc.add_price(price, timestamp_ms)

        # Update cache
        result = calc.calculate()
        if result.is_valid:
            # Smooth update
            old_vol = self._vol_cache.get(token_mint, result.realized_volatility)
            smoothed = (
                old_vol * (1 - self.config.vol_smoothing)
                + result.realized_volatility * self.config.vol_smoothing
            )
            self._vol_cache[token_mint] = smoothed

    def set_volatility_override(self, token_mint: str, volatility: float) -> None:
        """Manually set volatility for a token (e.g., from enrichment data)."""
        self._vol_cache[token_mint] = max(
            self.config.min_volatility_floor, min(volatility, self.config.max_volatility_cap)
        )

    def get_volatility(self, token_mint: str) -> float:
        """Get current volatility estimate for a token."""
        vol = self._vol_cache.get(token_mint, self._default_volatility)
        return max(self.config.min_volatility_floor, min(vol, self.config.max_volatility_cap))

    def calculate_position(
        self,
        capital_sol: float,
        token_mint: str,
        current_volatility: float | None = None,
        base_kelly_fraction: float | None = None,
    ) -> float:
        """
        Calculate position size in SOL for a token.

        Args:
            capital_sol: Available capital in SOL
            token_mint: Token identifier
            current_volatility: Override volatility (optional)
            base_kelly_fraction: Override Kelly fraction (optional)

        Returns:
            Position size in SOL
        """
        # Get volatility
        if current_volatility is not None:
            vol = current_volatility
        else:
            vol = self.get_volatility(token_mint)

        # Clamp volatility
        vol = max(self.config.min_volatility_floor, min(vol, self.config.max_volatility_cap))

        # Get base Kelly fraction
        kelly = base_kelly_fraction or self.config.base_kelly_fraction

        # Volatility scalar: how much to adjust position
        # If vol > target: reduce position
        # If vol < target: increase position (but cap at max_pct)
        vol_scalar = self.config.target_daily_volatility / vol

        # Apply scalar to Kelly fraction
        adjusted_kelly = kelly * vol_scalar

        # Calculate raw position
        raw_position = capital_sol * adjusted_kelly

        # Apply min/max bounds
        max_position = capital_sol * self.config.max_position_pct
        min_position = capital_sol * self.config.min_position_pct

        position = max(min_position, min(raw_position, max_position))

        logger.debug(
            f"Vol sizing for {token_mint[:8]}: "
            f"vol={vol:.1%}, scalar={vol_scalar:.2f}, "
            f"kelly={kelly:.2f} -> adjusted={adjusted_kelly:.3f}, "
            f"position={position:.4f} SOL"
        )

        return position

    def calculate_position_usd(
        self,
        capital_usd: float,
        token_mint: str,
        current_volatility: float | None = None,
    ) -> float:
        """Calculate position size in USD."""
        # Same logic, just different unit
        vol = current_volatility or self.get_volatility(token_mint)
        vol = max(self.config.min_volatility_floor, min(vol, self.config.max_volatility_cap))

        vol_scalar = self.config.target_daily_volatility / vol
        adjusted_kelly = self.config.base_kelly_fraction * vol_scalar

        raw_position = capital_usd * adjusted_kelly
        max_position = capital_usd * self.config.max_position_pct
        min_position = capital_usd * self.config.min_position_pct

        return max(min_position, min(raw_position, max_position))

    def get_risk_adjusted_size(
        self,
        base_size_sol: float,
        token_mint: str,
        current_volatility: float | None = None,
    ) -> float:
        """
        Adjust an existing position size based on volatility.

        Use this when you have a base position size from another
        calculation (e.g., Kelly) and want to vol-adjust it.
        """
        vol = current_volatility or self.get_volatility(token_mint)
        vol = max(self.config.min_volatility_floor, min(vol, self.config.max_volatility_cap))

        vol_scalar = self.config.target_daily_volatility / vol

        return base_size_sol * vol_scalar

    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "tokens_tracked": len(self._vol_calculators),
            "volatility_cache": {mint[:8]: f"{vol:.1%}" for mint, vol in self._vol_cache.items()},
            "config": {
                "target_daily_vol": f"{self.config.target_daily_volatility:.1%}",
                "max_position_pct": f"{self.config.max_position_pct:.1%}",
                "base_kelly": self.config.base_kelly_fraction,
            },
        }

    def reset(self, token_mint: str | None = None) -> None:
        """Reset volatility calculator(s)."""
        if token_mint:
            if token_mint in self._vol_calculators:
                self._vol_calculators[token_mint].reset()
            self._vol_cache.pop(token_mint, None)
        else:
            self._vol_calculators.clear()
            self._vol_cache.clear()


# Singleton instance
_sizing: VolatilityScaledSizing | None = None


def get_volatility_sizing() -> VolatilityScaledSizing:
    """Get the global volatility sizing instance."""
    global _sizing
    if _sizing is None:
        _sizing = VolatilityScaledSizing()
    return _sizing
