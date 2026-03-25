"""
Order Book Simulator - Depth-aware execution simulation.

Simulates realistic order book dynamics for accurate fill estimation:
- Price levels with depth
- Partial fills for large orders
- Queue position simulation
- Dynamic spread modeling
- Market impact over time

Usage:
    from coldpath.backtest.order_book_sim import OrderBookSimulator

    sim = OrderBookSimulator()

    # Create order book for a pool
    book = sim.create_book(
        mid_price=0.001,  # SOL per token
        liquidity_usd=50000,
        spread_bps=100,
    )

    # Simulate market buy
    result = sim.simulate_market_order(
        book=book,
        size_sol=0.1,
        side="buy",
    )

    print(f"Fill rate: {result.fill_rate:.1%}")
    print(f"Average price: {result.avg_price:.6f}")
    print(f"Slippage: {result.slippage_bps:.1f} bps")
"""

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"


@dataclass
class PriceLevel:
    """A single price level in the order book."""

    price: float
    size: float  # Total size at this level
    orders: int = 1  # Number of orders at this level

    @property
    def is_empty(self) -> bool:
        return self.size <= 0


@dataclass
class OrderBook:
    """Simulated order book."""

    symbol: str
    mid_price: float
    bids: list[PriceLevel] = field(default_factory=list)  # Sorted descending
    asks: list[PriceLevel] = field(default_factory=list)  # Sorted ascending
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def best_bid(self) -> PriceLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> PriceLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return 0.0

    @property
    def spread_bps(self) -> float:
        if self.mid_price > 0 and self.spread > 0:
            return (self.spread / self.mid_price) * 10000
        return 0.0

    @property
    def bid_depth(self) -> float:
        return sum(level.size for level in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(level.size for level in self.asks)

    @property
    def total_depth(self) -> float:
        return self.bid_depth + self.ask_depth


@dataclass
class FillResult:
    """Result of order execution simulation."""

    filled: bool
    fill_rate: float  # 0-1
    filled_size: float
    unfilled_size: float
    avg_price: float
    worst_price: float
    slippage_bps: float
    price_impact_pct: float
    levels_consumed: int
    partial_fills: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "filled": self.filled,
            "fill_rate": self.fill_rate,
            "filled_size": self.filled_size,
            "unfilled_size": self.unfilled_size,
            "avg_price": self.avg_price,
            "worst_price": self.worst_price,
            "slippage_bps": self.slippage_bps,
            "price_impact_pct": self.price_impact_pct,
            "levels_consumed": self.levels_consumed,
        }


@dataclass
class OrderBookConfig:
    """Configuration for order book simulation."""

    # Price level configuration
    num_levels: int = 20  # Number of price levels per side
    level_spacing_bps: float = 10.0  # Basis points between levels

    # Depth distribution (power law)
    depth_alpha: float = 1.5  # Power law exponent
    base_level_size: float = 0.1  # Base size at best level (in SOL)

    # Spread modeling
    base_spread_bps: float = 50.0  # Minimum spread
    spread_volatility_factor: float = 2.0  # Extra spread per 10% volatility

    # Order dynamics
    order_count_range: tuple[int, int] = (1, 5)  # Orders per level
    size_variance: float = 0.3  # Variance in order sizes

    # Simulation settings
    random_seed: int | None = 42


class OrderBookSimulator:
    """Simulate order book dynamics for realistic execution.

    Provides:
    - Order book creation with realistic depth profiles
    - Market order execution simulation
    - Limit order queue simulation
    - Price impact estimation
    - Dynamic spread modeling

    Example:
        sim = OrderBookSimulator()

        # Create order book
        book = sim.create_book(
            mid_price=0.001,
            liquidity_usd=50000,
            volatility_pct=25.0,
        )

        # Simulate market buy
        result = sim.simulate_market_order(book, size_sol=0.1, side="buy")

        # Check execution quality
        if result.slippage_bps > 200:
            print("Warning: High slippage expected!")
    """

    def __init__(self, config: OrderBookConfig | None = None):
        """Initialize the order book simulator.

        Args:
            config: Simulation configuration
        """
        self.config = config or OrderBookConfig()
        self._rng = random.Random(self.config.random_seed)

    def create_book(
        self,
        mid_price: float,
        liquidity_usd: float,
        volatility_pct: float = 20.0,
        is_memecoin: bool = False,
        symbol: str = "UNKNOWN",
    ) -> OrderBook:
        """Create a realistic order book.

        Args:
            mid_price: Current mid price
            liquidity_usd: Total pool liquidity in USD
            volatility_pct: Current volatility percentage
            is_memecoin: Whether this is a memecoin
            symbol: Token symbol

        Returns:
            OrderBook with populated levels
        """
        # Calculate spread (wider for volatile/memecoins)
        spread_bps = self.config.base_spread_bps
        spread_bps += volatility_pct / 10 * self.config.spread_volatility_factor
        if is_memecoin:
            spread_bps *= 1.5

        spread = mid_price * spread_bps / 10000

        # Calculate depth
        sol_price = 150.0
        total_depth_sol = liquidity_usd / sol_price
        depth_per_side = total_depth_sol / 2

        # Create bids (below mid price)
        bids = []
        for i in range(self.config.num_levels):
            level_price = (
                mid_price - spread / 2 - i * mid_price * self.config.level_spacing_bps / 10000
            )
            level_size = self._calculate_level_size(i, depth_per_side)

            bids.append(
                PriceLevel(
                    price=max(0, level_price),
                    size=level_size,
                    orders=self._rng.randint(*self.config.order_count_range),
                )
            )

        # Create asks (above mid price)
        asks = []
        for i in range(self.config.num_levels):
            level_price = (
                mid_price + spread / 2 + i * mid_price * self.config.level_spacing_bps / 10000
            )
            level_size = self._calculate_level_size(i, depth_per_side)

            asks.append(
                PriceLevel(
                    price=level_price,
                    size=level_size,
                    orders=self._rng.randint(*self.config.order_count_range),
                )
            )

        return OrderBook(
            symbol=symbol,
            mid_price=mid_price,
            bids=bids,
            asks=asks,
        )

    def _calculate_level_size(self, level_index: int, total_depth: float) -> float:
        """Calculate size at a price level using power law distribution."""
        # Power law: size_i = base / (i + 1)^alpha
        base = self.config.base_level_size

        if level_index == 0:
            return base * self._rng.uniform(0.8, 1.2)

        size = base / ((level_index + 1) ** self.config.depth_alpha)

        # Add variance
        size *= self._rng.uniform(1 - self.config.size_variance, 1 + self.config.size_variance)

        return max(0.001, size)

    def simulate_market_order(
        self,
        book: OrderBook,
        size_sol: float,
        side: str,
        slippage_tolerance_bps: float = 500.0,
    ) -> FillResult:
        """Simulate market order execution.

        Args:
            book: Order book to execute against
            size_sol: Order size in SOL
            side: "buy" or "sell"
            slippage_tolerance_bps: Maximum acceptable slippage

        Returns:
            FillResult with execution details
        """
        levels = book.asks if side == "buy" else book.bids
        if not levels:
            return FillResult(
                filled=False,
                fill_rate=0.0,
                filled_size=0.0,
                unfilled_size=size_sol,
                avg_price=0.0,
                worst_price=0.0,
                slippage_bps=0.0,
                price_impact_pct=0.0,
                levels_consumed=0,
            )

        remaining = size_sol
        total_cost = 0.0
        total_filled = 0.0
        worst_price = 0.0
        levels_consumed = 0
        partial_fills = []

        for level in levels:
            if remaining <= 0:
                break

            levels_consumed += 1

            # Calculate how much we can fill at this level
            # For buy: we spend SOL to get tokens
            # level.size is in tokens, we need to convert
            tokens_available = level.size
            sol_cost_at_level = tokens_available * level.price

            if sol_cost_at_level >= remaining:
                # Partial fill at this level
                tokens_filled = remaining / level.price
                total_filled += tokens_filled
                total_cost += remaining
                worst_price = level.price

                partial_fills.append(
                    {
                        "price": level.price,
                        "tokens": tokens_filled,
                        "sol": remaining,
                    }
                )

                remaining = 0
            else:
                # Fill entire level
                total_filled += tokens_available
                total_cost += sol_cost_at_level
                worst_price = level.price
                remaining -= sol_cost_at_level

                partial_fills.append(
                    {
                        "price": level.price,
                        "tokens": tokens_available,
                        "sol": sol_cost_at_level,
                    }
                )

        # Calculate results
        fill_rate = (size_sol - remaining) / size_sol if size_sol > 0 else 0.0
        avg_price = total_cost / total_filled if total_filled > 0 else 0.0

        # Calculate slippage vs mid price
        if book.mid_price > 0:
            slippage_bps = ((worst_price - book.mid_price) / book.mid_price) * 10000
            if side == "sell":
                slippage_bps = ((book.mid_price - worst_price) / book.mid_price) * 10000
        else:
            slippage_bps = 0.0

        # Price impact
        price_impact = abs(slippage_bps) / 100

        return FillResult(
            filled=remaining <= 0,
            fill_rate=fill_rate,
            filled_size=size_sol - remaining,
            unfilled_size=remaining,
            avg_price=avg_price,
            worst_price=worst_price,
            slippage_bps=slippage_bps,
            price_impact_pct=price_impact,
            levels_consumed=levels_consumed,
            partial_fills=partial_fills,
        )

    def simulate_limit_order(
        self,
        book: OrderBook,
        size_sol: float,
        side: str,
        limit_price: float,
        time_in_force_seconds: float = 30.0,
    ) -> FillResult:
        """Simulate limit order execution.

        Args:
            book: Order book
            size_sol: Order size in SOL
            side: "buy" or "sell"
            limit_price: Limit price
            time_in_force_seconds: How long the order is active

        Returns:
            FillResult with execution details
        """
        levels = book.asks if side == "buy" else book.bids

        # Find levels that satisfy limit price
        fillable_levels = []
        for level in levels:
            if side == "buy" and level.price <= limit_price:
                fillable_levels.append(level)
            elif side == "sell" and level.price >= limit_price:
                fillable_levels.append(level)

        if not fillable_levels:
            return FillResult(
                filled=False,
                fill_rate=0.0,
                filled_size=0.0,
                unfilled_size=size_sol,
                avg_price=limit_price,
                worst_price=limit_price,
                slippage_bps=0.0,
                price_impact_pct=0.0,
                levels_consumed=0,
            )

        # Simulate execution through fillable levels
        remaining = size_sol
        total_cost = 0.0
        total_filled = 0.0
        worst_price = 0.0
        levels_consumed = 0
        partial_fills = []

        for level in fillable_levels:
            if remaining <= 0:
                break

            levels_consumed += 1

            tokens_available = level.size
            sol_cost_at_level = tokens_available * level.price

            if sol_cost_at_level >= remaining:
                tokens_filled = remaining / level.price
                total_filled += tokens_filled
                total_cost += remaining
                worst_price = level.price
                partial_fills.append(
                    {
                        "price": level.price,
                        "tokens": tokens_filled,
                        "sol": remaining,
                    }
                )
                remaining = 0
            else:
                total_filled += tokens_available
                total_cost += sol_cost_at_level
                worst_price = level.price
                remaining -= sol_cost_at_level
                partial_fills.append(
                    {
                        "price": level.price,
                        "tokens": tokens_available,
                        "sol": sol_cost_at_level,
                    }
                )

        fill_rate = (size_sol - remaining) / size_sol if size_sol > 0 else 0.0
        avg_price = total_cost / total_filled if total_filled > 0 else limit_price

        if book.mid_price > 0:
            slippage_bps = ((worst_price - book.mid_price) / book.mid_price) * 10000
            if side == "sell":
                slippage_bps = ((book.mid_price - worst_price) / book.mid_price) * 10000
        else:
            slippage_bps = 0.0

        return FillResult(
            filled=remaining <= 0,
            fill_rate=fill_rate,
            filled_size=size_sol - remaining,
            unfilled_size=remaining,
            avg_price=avg_price,
            worst_price=worst_price,
            slippage_bps=slippage_bps,
            price_impact_pct=abs(slippage_bps) / 100,
            levels_consumed=levels_consumed,
            partial_fills=partial_fills,
        )

    def estimate_price_impact(
        self,
        book: OrderBook,
        size_sol: float,
        side: str,
    ) -> dict[str, float]:
        """Estimate price impact without full simulation.

        Args:
            book: Order book
            size_sol: Trade size
            side: "buy" or "sell"

        Returns:
            Dict with impact estimates
        """
        levels = book.asks if side == "buy" else book.bids

        total_depth = sum(level.size * level.price for level in levels)
        depth_ratio = size_sol / total_depth if total_depth > 0 else 1.0

        # Estimate using square root model
        # impact = spread/2 + sigma * sqrt(size/ADV)
        spread_impact = book.spread_bps / 2
        depth_impact = math.sqrt(depth_ratio) * 100  # Convert to bps

        total_impact = spread_impact + depth_impact

        return {
            "spread_impact_bps": spread_impact,
            "depth_impact_bps": depth_impact,
            "total_impact_bps": total_impact,
            "depth_ratio": depth_ratio,
            "estimated_fill_rate": max(0.5, 1.0 - depth_ratio),
        }

    def update_book(
        self,
        book: OrderBook,
        trade_size_sol: float,
        side: str,
        volatility_pct: float = 20.0,
    ) -> OrderBook:
        """Update order book after a trade (simulate other traders).

        Args:
            book: Current order book
            trade_size_sol: Size of executed trade
            side: Side of executed trade
            volatility_pct: Current volatility

        Returns:
            Updated order book
        """
        # Create updated book
        new_bids = [
            PriceLevel(price=level.price, size=level.size, orders=level.orders)
            for level in book.bids
        ]
        new_asks = [
            PriceLevel(price=level.price, size=level.size, orders=level.orders)
            for level in book.asks
        ]

        # Adjust levels based on trade
        if side == "buy":
            # Buy trade consumed asks, shift mid price up slightly
            price_shift = book.mid_price * volatility_pct / 1000 * self._rng.uniform(0.5, 1.5)
            new_mid = book.mid_price + price_shift

            # Reduce ask depth
            for level in new_asks:
                level.size *= self._rng.uniform(0.9, 1.0)
        else:
            # Sell trade consumed bids, shift mid price down
            price_shift = book.mid_price * volatility_pct / 1000 * self._rng.uniform(0.5, 1.5)
            new_mid = book.mid_price - price_shift

            # Reduce bid depth
            for level in new_bids:
                level.size *= self._rng.uniform(0.9, 1.0)

        return OrderBook(
            symbol=book.symbol,
            mid_price=new_mid,
            bids=new_bids,
            asks=new_asks,
        )


def create_book_from_liquidity(
    liquidity_usd: float,
    current_price: float,
    volatility_pct: float = 20.0,
    is_memecoin: bool = False,
) -> OrderBook:
    """Convenience function to create order book from liquidity.

    Args:
        liquidity_usd: Pool liquidity in USD
        current_price: Current token price
        volatility_pct: Current volatility
        is_memecoin: Is this a memecoin

    Returns:
        Populated OrderBook
    """
    sim = OrderBookSimulator()
    return sim.create_book(
        mid_price=current_price,
        liquidity_usd=liquidity_usd,
        volatility_pct=volatility_pct,
        is_memecoin=is_memecoin,
    )
