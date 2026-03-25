"""
MEV (Maximal Extractable Value) Event Generator

Generates realistic MEV events including:
- Sandwich attacks (frontrun + backrun)
- JIT (Just-In-Time) liquidity attacks
- Backrunning liquidations
- DEX arbitrage
- Cross-DEX arbitrage
- Batching opportunities
- Priority gas auctions

MEV is a dominant force in Solana DEX trading - it affects:
- Price impact estimation
- Slippage calculation
- Trade execution timing
- Profit opportunities

This generator models:
- MEV bot competition
- Realistic profitability thresholds
- Latency-aware execution
- Gas fee dynamics
- Transaction ordering

Architecture:
- Event queue with timestamps
- MEV opportunity detection
- Bot reaction simulation
- Profit/loss tracking

Usage:
    mev_generator = MEVEventGenerator(
        mev_bot_count=10,
        avg_latency_ms=5,
    )

    # Generate MEV events for a trade
    mev_events = mev_generator.generate_for_trade(
        trade_price=1.0,
        trade_size=1000,
        order_book_snapshot=order_book,
    )
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MEVEventType(Enum):
    """Types of MEV events."""

    SANDWICH = "sandwich"  # Frontrun + backrun
    JIT_LIQUIDITY = "jit_liquidity"  # Just-In-Time liquidity provision
    BACKRUN = "backrun"  # Backrunning liquidations
    DEX_ARBITRAGE = "dex_arbitrage"  # Cross-DEX arbitrage
    BATCHING = "batching"  # Transaction batching
    PRIORITY_AUCTION = "priority_auction"  # Priority gas auctions


class MEVBot:
    """MEV bot with specific behavior."""

    def __init__(
        self,
        bot_id: int,
        latency_ms: float,
        capital_usd: float = 100_000,
        gas_budget_sol: float = 0.1,
        success_rate: float = 0.9,
    ):
        self.bot_id = bot_id
        self.latency_ms = latency_ms
        self.capital_usd = capital_usd
        self.gas_budget_sol = gas_budget_sol
        self.success_rate = success_rate

        # Track bot performance
        self.trades_attempted = 0
        self.trades_succeeded = 0
        self.total_profit_usd = 0.0
        self.total_gas_spent_sol = 0.0


@dataclass
class MEVEvent:
    """A single MEV event."""

    event_id: int
    event_type: MEVEventType
    bot_id: int

    # Timing
    timestamp: float
    latency_ms: float

    # Trade details
    trigger_trade_id: int | None  # Trade that triggered MEV
    target_price: float
    target_size: float

    # MEV-specific details
    frontrun_size: float = 0.0
    backrun_size: float = 0.0
    profit_usd: float = 0.0
    gas_cost_sol: float = 0.0

    # Status
    success: bool = False
    error_reason: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderBookSnapshot:
    """Order book snapshot for MEV opportunity detection."""

    timestamp: float
    best_bid: float
    best_ask: float
    mid_price: float
    bid_depth: float
    ask_depth: float
    spread: float
    spread_pct: float
    volume_24h: float = 0.0


@dataclass
class Trade:
    """A trade that may trigger MEV."""

    trade_id: int
    timestamp: float
    price: float
    size: float
    side: str  # "buy" or "sell"
    is_limit: bool = False


class MEVEventGenerator:
    """Generate realistic MEV events for trading simulation."""

    def __init__(
        self,
        mev_bot_count: int = 10,
        avg_latency_ms: float = 5.0,
        latency_std_ms: float = 2.0,
        sol_price_usd: float = 100.0,
    ):
        """Initialize MEV generator.

        Args:
            mev_bot_count: Number of MEV bots
            avg_latency_ms: Average bot latency
            latency_std_ms: Latency standard deviation
            sol_price_usd: Current SOL price in USD
        """
        self.mev_bots = [
            MEVBot(
                bot_id=i,
                latency_ms=np.random.normal(avg_latency_ms, latency_std_ms),
                capital_usd=np.random.uniform(50_000, 500_000),
                gas_budget_sol=np.random.uniform(0.05, 0.5),
                success_rate=np.random.uniform(0.8, 0.95),
            )
            for i in range(mev_bot_count)
        ]

        self.sol_price_usd = sol_price_usd

        # Event tracking
        self.next_event_id = 0
        self.events: list[MEVEvent] = []

        # Statistics
        self.sandwich_count = 0
        self.jit_count = 0
        self.backrun_count = 0
        self.arb_count = 0

    def detect_sandwich_opportunity(
        self,
        trade: Trade,
        order_book: OrderBookSnapshot,
    ) -> float | None:
        """Detect sandwich attack opportunity.

        Args:
            trade: The triggering trade
            order_book: Order book snapshot

        Returns:
            Estimated profit in USD, or None if no opportunity
        """
        # Sandwich attacks are profitable when:
        # 1. Trade is large (significant price impact)
        # 2. Spread is wide (allows for profit)
        # 3. Order book is thin (easy to move price)

        price_impact = self._estimate_price_impact(trade, order_book)

        # Need sufficient price impact
        if price_impact < 0.01:  # < 1% impact = not profitable
            return None

        # Estimate frontrun and backrun sizes
        # Typical: 50% of trade size
        trade.size * 0.5
        trade.size * 0.5

        # Estimate profit
        # Profit ≈ price_impact * trade_size - gas costs
        gross_profit = price_impact * trade.size * order_book.mid_price

        # Gas costs (simplified)
        gas_cost_sol = 0.01
        gas_cost_usd = gas_cost_sol * self.sol_price_usd

        net_profit = gross_profit - gas_cost_usd

        if net_profit > 10.0:  # Minimum $10 profit
            return net_profit

        return None

    def detect_jit_opportunity(
        self,
        trade: Trade,
        order_book: OrderBookSnapshot,
    ) -> float | None:
        """Detect JIT liquidity opportunity.

        Args:
            trade: The triggering trade
            order_book: Order book snapshot

        Returns:
            Estimated profit in USD, or None if no opportunity
        """
        # JIT liquidity is profitable when:
        # 1. Large trade is incoming (can see in mempool)
        # 2. Order book is thin on that side
        # 3. Liquidity rewards are high

        if trade.side == "buy":
            # Thin asks = JIT opportunity
            if order_book.ask_depth < trade.size * 2:
                liquidity_needed = trade.size - order_book.ask_depth
                liquidity_reward = liquidity_needed * 0.001  # 0.1% fee

                gas_cost_sol = 0.005
                gas_cost_usd = gas_cost_sol * self.sol_price_usd

                profit = liquidity_reward * order_book.mid_price - gas_cost_usd

                if profit > 5.0:
                    return profit

        return None

    def detect_backrun_opportunity(
        self,
        trade: Trade,
        order_book: OrderBookSnapshot,
    ) -> float | None:
        """Detect backrun opportunity (e.g., liquidation).

        Args:
            trade: The triggering trade
            order_book: Order book snapshot

        Returns:
            Estimated profit in USD, or None if no opportunity
        """
        # Backrunning is common for:
        # 1. Liquidations (large sell-offs)
        # 2. Large trades that move price significantly

        price_impact = self._estimate_price_impact(trade, order_book)

        if price_impact > 0.05:  # > 5% impact = liquidation candidate
            # Backrun by buying the dip
            backrun_size = trade.size * 0.3

            expected_rebound = price_impact * 0.5  # Expect 50% rebound
            gross_profit = expected_rebound * backrun_size * order_book.mid_price

            gas_cost_sol = 0.008
            gas_cost_usd = gas_cost_sol * self.sol_price_usd

            profit = gross_profit - gas_cost_usd

            if profit > 10.0:
                return profit

        return None

    def detect_arbitrage_opportunity(
        self,
        trade: Trade,
        order_book: OrderBookSnapshot,
        cross_dex_price: float,
    ) -> float | None:
        """Detect cross-DEX arbitrage opportunity.

        Args:
            trade: The triggering trade
            order_book: Current order book
            cross_dex_price: Price on another DEX

        Returns:
            Estimated profit in USD, or None if no opportunity
        """
        # Arbitrage exists when prices differ significantly
        price_diff_pct = abs(order_book.mid_price - cross_dex_price) / order_book.mid_price

        if price_diff_pct > 0.01:  # > 1% price difference
            trade_size = min(trade.size, 1000)  # Max 1000 tokens
            gross_profit = price_diff_pct * trade_size * order_book.mid_price

            gas_cost_sol = 0.015  # Higher for cross-DEX
            gas_cost_usd = gas_cost_sol * self.sol_price_usd

            profit = gross_profit - gas_cost_usd

            if profit > 20.0:
                return profit

        return None

    def generate_for_trade(
        self,
        trade: Trade,
        order_book: OrderBookSnapshot,
        cross_dex_price: float | None = None,
    ) -> list[MEVEvent]:
        """Generate MEV events for a trade.

        Args:
            trade: The triggering trade
            order_book: Order book snapshot
            cross_dex_price: Optional cross-DEX price for arbitrage

        Returns:
            List of MEV events
        """
        events = []

        # Check for various MEV opportunities
        sandwich_profit = self.detect_sandwich_opportunity(trade, order_book)
        jit_profit = self.detect_jit_opportunity(trade, order_book)
        backrun_profit = self.detect_backrun_opportunity(trade, order_book)
        arb_profit = self.detect_arbitrage_opportunity(
            trade,
            order_book,
            cross_dex_price or order_book.mid_price,
        )

        # Generate events based on opportunities
        if sandwich_profit is not None:
            self.sandwich_count += 1
            event = self._create_sandwich_event(
                trade,
                order_book,
                sandwich_profit,
            )
            events.append(event)

        if jit_profit is not None:
            self.jit_count += 1
            event = self._create_jit_event(
                trade,
                order_book,
                jit_profit,
            )
            events.append(event)

        if backrun_profit is not None:
            self.backrun_count += 1
            event = self._create_backrun_event(
                trade,
                order_book,
                backrun_profit,
            )
            events.append(event)

        if arb_profit is not None:
            self.arb_count += 1
            event = self._create_arbitrage_event(
                trade,
                order_book,
                arb_profit,
            )
            events.append(event)

        return events

    def _create_sandwich_event(
        self,
        trade: Trade,
        order_book: OrderBookSnapshot,
        profit_usd: float,
    ) -> MEVEvent:
        """Create a sandwich attack event."""
        bot = self._select_best_bot(profit_usd)

        event = MEVEvent(
            event_id=self.next_event_id,
            event_type=MEVEventType.SANDWICH,
            bot_id=bot.bot_id,
            timestamp=trade.timestamp + bot.latency_ms / 1000.0,
            latency_ms=bot.latency_ms,
            trigger_trade_id=trade.trade_id,
            target_price=order_book.mid_price,
            target_size=trade.size,
            frontrun_size=trade.size * 0.5,
            backrun_size=trade.size * 0.5,
            profit_usd=profit_usd,
            gas_cost_sol=0.01,
            success=np.random.random() < bot.success_rate,
            metadata={
                "price_impact": self._estimate_price_impact(trade, order_book),
                "spread": order_book.spread_pct,
            },
        )

        self.next_event_id += 1
        self.events.append(event)

        return event

    def _create_jit_event(
        self,
        trade: Trade,
        order_book: OrderBookSnapshot,
        profit_usd: float,
    ) -> MEVEvent:
        """Create a JIT liquidity event."""
        bot = self._select_best_bot(profit_usd)

        liquidity_size = trade.size - order_book.ask_depth if trade.side == "buy" else 0

        event = MEVEvent(
            event_id=self.next_event_id,
            event_type=MEVEventType.JIT_LIQUIDITY,
            bot_id=bot.bot_id,
            timestamp=trade.timestamp + bot.latency_ms / 1000.0,
            latency_ms=bot.latency_ms,
            trigger_trade_id=trade.trade_id,
            target_price=order_book.mid_price,
            target_size=liquidity_size,
            profit_usd=profit_usd,
            gas_cost_sol=0.005,
            success=np.random.random() < bot.success_rate,
            metadata={
                "liquidity_provided": liquidity_size,
                "fee_rate": 0.001,
            },
        )

        self.next_event_id += 1
        self.events.append(event)

        return event

    def _create_backrun_event(
        self,
        trade: Trade,
        order_book: OrderBookSnapshot,
        profit_usd: float,
    ) -> MEVEvent:
        """Create a backrun event."""
        bot = self._select_best_bot(profit_usd)

        event = MEVEvent(
            event_id=self.next_event_id,
            event_type=MEVEventType.BACKRUN,
            bot_id=bot.bot_id,
            timestamp=trade.timestamp + bot.latency_ms / 1000.0,
            latency_ms=bot.latency_ms,
            trigger_trade_id=trade.trade_id,
            target_price=order_book.mid_price * 0.95,  # Buy the dip
            target_size=trade.size * 0.3,
            profit_usd=profit_usd,
            gas_cost_sol=0.008,
            success=np.random.random() < bot.success_rate,
            metadata={
                "price_impact": self._estimate_price_impact(trade, order_book),
                "expected_rebound": self._estimate_price_impact(trade, order_book) * 0.5,
            },
        )

        self.next_event_id += 1
        self.events.append(event)

        return event

    def _create_arbitrage_event(
        self,
        trade: Trade,
        order_book: OrderBookSnapshot,
        profit_usd: float,
    ) -> MEVEvent:
        """Create an arbitrage event."""
        bot = self._select_best_bot(profit_usd)

        event = MEVEvent(
            event_id=self.next_event_id,
            event_type=MEVEventType.DEX_ARBITRAGE,
            bot_id=bot.bot_id,
            timestamp=trade.timestamp + bot.latency_ms / 1000.0,
            latency_ms=bot.latency_ms,
            trigger_trade_id=trade.trade_id,
            target_price=order_book.mid_price,
            target_size=min(trade.size, 1000),
            profit_usd=profit_usd,
            gas_cost_sol=0.015,
            success=np.random.random() < bot.success_rate,
            metadata={
                "cross_dex_price": order_book.mid_price * 1.01,  # 1% higher
            },
        )

        self.next_event_id += 1
        self.events.append(event)

        return event

    def _select_best_bot(self, profit_usd: float) -> MEVBot:
        """Select the best bot for an MEV opportunity."""
        # Sort by latency (fastest first), filter by capital
        capable_bots = [b for b in self.mev_bots if b.capital_usd >= profit_usd * 10]

        if not capable_bots:
            return self.mev_bots[0]

        # Return fastest capable bot
        return min(capable_bots, key=lambda b: b.latency_ms)

    def _estimate_price_impact(
        self,
        trade: Trade,
        order_book: OrderBookSnapshot,
    ) -> float:
        """Estimate price impact of a trade."""
        # Simplified price impact model
        # Impact ∝ trade_size / depth

        if trade.side == "buy":
            depth = order_book.ask_depth
        else:
            depth = order_book.bid_depth

        if depth < trade.size:
            # Trade exceeds available depth - high impact
            return 0.1 + (trade.size / depth - 1.0) * 0.05
        else:
            # Normal impact
            return (trade.size / depth) * 0.02

    def get_statistics(self) -> dict[str, Any]:
        """Get MEV statistics."""
        successful_events = [e for e in self.events if e.success]

        return {
            "total_events": len(self.events),
            "successful_events": len(successful_events),
            "success_rate": len(successful_events) / len(self.events) if self.events else 0.0,
            "sandwich_count": self.sandwich_count,
            "jit_count": self.jit_count,
            "backrun_count": self.backrun_count,
            "arb_count": self.arb_count,
            "total_profit_usd": sum(e.profit_usd for e in successful_events),
            "avg_profit_usd": (
                np.mean([e.profit_usd for e in successful_events]) if successful_events else 0.0
            ),
            "total_gas_sol": sum(e.gas_cost_sol for e in self.events),
        }


if __name__ == "__main__":
    logger.info("Testing MEV event generator")

    # Create generator
    generator = MEVEventGenerator(
        mev_bot_count=10,
        avg_latency_ms=5.0,
    )

    # Create a sample order book
    order_book = OrderBookSnapshot(
        timestamp=0.0,
        best_bid=0.999,
        best_ask=1.001,
        mid_price=1.0,
        bid_depth=10000.0,
        ask_depth=5000.0,  # Thin asks = MEV opportunity
        spread=0.002,
        spread_pct=0.002,
        volume_24h=1_000_000,
    )

    # Create a large trade (triggers MEV)
    trade = Trade(
        trade_id=1,
        timestamp=0.0,
        price=1.0,
        size=2000,  # Large trade
        side="buy",
    )

    # Generate MEV events
    mev_events = generator.generate_for_trade(trade, order_book)

    print(f"\n=== Generated {len(mev_events)} MEV events ===")
    for event in mev_events:
        print(f"\nEvent {event.event_id}:")
        print(f"  Type: {event.event_type.value}")
        print(f"  Bot: {event.bot_id}")
        print(f"  Latency: {event.latency_ms:.2f}ms")
        print(f"  Profit: ${event.profit_usd:.2f}")
        print(f"  Success: {event.success}")

    # Print statistics
    stats = generator.get_statistics()
    print("\n=== MEV Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
