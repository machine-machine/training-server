"""
Order Book Simulator for Realistic Market Micro-Structure

Generates realistic order book dynamics including:
- Limit order placement (Poisson process)
- Market orders (clustering)
- Bid-ask spread dynamics
- Liquidity provision competition
- Order cancellation
- Iceberg orders
- Cross-market arbitrage

This simulator captures the reality of Solana DEX order books:
- MEV bots competing for arbitrage
- Liquidity providers competing for fees
- Snipers exploiting new token launches
- Market makers adjusting quotes

Architecture:
- Event-driven simulation
- Multiple market participants (MMs, snipers, retail)
- Realistic order types (limit, market, IOC, FOK)
- Latency-aware execution

Usage:
    simulator = OrderBookSimulator(
        initial_price=1.0,
        tick_size=0.0001,
        liquidity_usd=100_000,
    )

    # Simulate 1 hour of trading
    order_book_events = simulator.simulate(
        duration_minutes=60,
        participants=100,
        mev_bot_count=5,
    )

    # Extract features for ML training
    features = simulator.extract_features(order_book_events)
"""

import heapq
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""

    LIMIT = "limit"
    MARKET = "market"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


class OrderSide(Enum):
    """Order sides."""

    BID = "bid"
    ASK = "ask"


class ParticipantType(Enum):
    """Types of market participants."""

    MARKET_MAKER = "market_maker"
    RETAIL = "retail"
    SNIPER = "sniper"
    MEV_BOT = "mev_bot"
    ARBITRAGEUR = "arbitrageur"


@dataclass
class Order:
    """Order in the order book."""

    order_id: int
    side: OrderSide
    order_type: OrderType
    price: float
    size: float
    participant_id: int
    participant_type: ParticipantType
    timestamp: float

    # Remaining quantity (for partial fills)
    remaining: float = field(init=False)

    # Order metadata
    is_iceberg: bool = False
    is_hidden: bool = False
    iceberg_quantity: float | None = None

    def __post_init__(self):
        self.remaining = self.size

    def __lt__(self, other: "Order") -> bool:
        """Heap comparison: best bid (max) or best ask (min)."""
        if self.side == OrderSide.BID:
            return self.price > other.price  # Max-heap for bids
        else:
            return self.price < other.price  # Min-heap for asks


@dataclass
class Trade:
    """Executed trade."""

    trade_id: int
    price: float
    size: float
    bid_order: Order
    ask_order: Order
    timestamp: float
    is_mev: bool = False
    mev_type: str | None = None


@dataclass
class OrderBookEvent:
    """Event in the order book."""

    event_type: str  # "order_placed", "order_cancelled", "trade", "order_modified"
    timestamp: float
    order: Order | None = None
    trade: Trade | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book state."""

    timestamp: float
    bids: list[tuple[float, float]]  # [(price, size), ...] sorted desc
    asks: list[tuple[float, float]]  # [(price, size), ...] sorted asc
    best_bid: float
    best_ask: float
    spread: float
    total_volume: float
    last_trade_price: float


class Participant:
    """Market participant with specific behavior."""

    def __init__(
        self,
        participant_id: int,
        participant_type: ParticipantType,
        latency_ms: float = 10.0,
    ):
        self.participant_id = participant_id
        self.participant_type = participant_type
        self.latency_ms = latency_ms
        self.last_order_time = 0.0

    def should_place_order(
        self,
        current_time: float,
        order_book: "OrderBookSimulator",
    ) -> bool:
        """Decide whether to place an order."""
        time_since_last = current_time - self.last_order_time

        if self.participant_type == ParticipantType.MARKET_MAKER:
            # MMs place orders frequently to maintain quotes
            return np.random.random() < 0.1 and time_since_last > 0.1

        elif self.participant_type == ParticipantType.RETAIL:
            # Retail orders are less frequent
            return np.random.random() < 0.01 and time_since_last > 1.0

        elif self.participant_type == ParticipantType.SNIPER:
            # Snipers react to new tokens or arbitrage
            return order_book.has_arbitrage_opportunity() and time_since_last > 0.01

        elif self.participant_type == ParticipantType.MEV_BOT:
            # MEV bots react aggressively to opportunities
            return order_book.has_mev_opportunity() and time_since_last > 0.001

        return False

    def generate_order(
        self,
        current_time: float,
        order_book: "OrderBookSimulator",
    ) -> Order | None:
        """Generate an order based on participant type."""
        best_bid, best_ask = order_book.get_best_bid_ask()
        mid_price = (best_bid + best_ask) / 2.0

        if self.participant_type == ParticipantType.MARKET_MAKER:
            # Place limit orders on both sides
            side = np.random.choice([OrderSide.BID, OrderSide.ASK])

            if side == OrderSide.BID:
                # Bid below best bid
                price = best_bid * (1.0 - np.random.uniform(0.0001, 0.001))
            else:
                # Ask above best ask
                price = best_ask * (1.0 + np.random.uniform(0.0001, 0.001))

            size = np.random.uniform(10, 1000)
            order_type = OrderType.LIMIT

        elif self.participant_type == ParticipantType.RETAIL:
            # Mostly market orders
            order_type = np.random.choice(
                [OrderType.MARKET, OrderType.LIMIT],
                p=[0.7, 0.3],
            )

            if order_type == OrderType.MARKET:
                side = np.random.choice([OrderSide.BID, OrderSide.ASK])
                size = np.random.uniform(1, 100)
                price = mid_price  # Ignored for market orders
            else:
                side = np.random.choice([OrderSide.BID, OrderSide.ASK])
                if side == OrderSide.BID:
                    price = best_bid * np.random.uniform(0.9, 1.0)
                else:
                    price = best_ask * np.random.uniform(1.0, 1.1)
                size = np.random.uniform(1, 100)

        elif self.participant_type == ParticipantType.SNIPER:
            # Aggressive market orders for arbitrage
            order_type = OrderType.MARKET
            side = OrderSide.BID if order_book.has_pump_opportunity() else OrderSide.ASK
            size = np.random.uniform(100, 1000)
            price = mid_price

        elif self.participant_type == ParticipantType.MEV_BOT:
            # Aggressive orders for MEV
            order_type = OrderType.MARKET
            side = np.random.choice([OrderSide.BID, OrderSide.ASK])
            size = np.random.uniform(100, 5000)
            price = mid_price

        else:
            return None

        order = Order(
            order_id=int(current_time * 1_000_000) + self.participant_id,
            side=side,
            order_type=order_type,
            price=price,
            size=size,
            participant_id=self.participant_id,
            participant_type=self.participant_type,
            timestamp=current_time,
        )

        self.last_order_time = current_time
        return order


class OrderBookSimulator:
    """Realistic order book simulator for DEX trading."""

    def __init__(
        self,
        initial_price: float = 1.0,
        tick_size: float = 0.0001,
        liquidity_usd: float = 100_000.0,
        n_price_levels: int = 10,
    ):
        """Initialize order book.

        Args:
            initial_price: Starting price
            tick_size: Minimum price increment
            liquidity_usd: Initial liquidity in USD
            n_price_levels: Number of price levels to track
        """
        self.initial_price = initial_price
        self.tick_size = tick_size
        self.n_price_levels = n_price_levels

        # Order book: heaps for bids and asks
        self.bids: list[Order] = []  # Max-heap
        self.asks: list[Order] = []  # Min-heap

        # Order lookup by ID
        self.orders: dict[int, Order] = {}

        # Participants
        self.participants: list[Participant] = []
        self.next_participant_id = 0

        # Simulation state
        self.current_time = 0.0
        self.next_order_id = 0
        self.trade_id = 0

        # Event log
        self.events: list[OrderBookEvent] = []

        # Trade history
        self.trades: list[Trade] = []
        self.last_trade_price = initial_price

        # Order book snapshots
        self.snapshots: list[OrderBookSnapshot] = []

        # Initialize order book with liquidity
        self._initialize_liquidity(liquidity_usd)

    def _initialize_liquidity(self, liquidity_usd: float):
        """Initialize order book with initial liquidity."""
        # Place symmetric limit orders around mid price
        for i in range(1, self.n_price_levels + 1):
            bid_price = self.initial_price * (1.0 - i * 0.001)
            ask_price = self.initial_price * (1.0 + i * 0.001)

            # Distribute liquidity evenly
            bid_size = liquidity_usd / (self.n_price_levels * bid_price * 2)
            ask_size = liquidity_usd / (self.n_price_levels * ask_price * 2)

            bid_order = Order(
                order_id=self.next_order_id,
                side=OrderSide.BID,
                order_type=OrderType.LIMIT,
                price=bid_price,
                size=bid_size,
                participant_id=-1,  # Initial liquidity
                participant_type=ParticipantType.MARKET_MAKER,
                timestamp=0.0,
            )
            self.next_order_id += 1

            ask_order = Order(
                order_id=self.next_order_id,
                side=OrderSide.ASK,
                order_type=OrderType.LIMIT,
                price=ask_price,
                size=ask_size,
                participant_id=-1,
                participant_type=ParticipantType.MARKET_MAKER,
                timestamp=0.0,
            )
            self.next_order_id += 1

            heapq.heappush(self.bids, bid_order)
            heapq.heappush(self.asks, ask_order)
            self.orders[bid_order.order_id] = bid_order
            self.orders[ask_order.order_id] = ask_order

        # Take initial snapshot
        self._take_snapshot()

    def add_participant(
        self,
        participant_type: ParticipantType,
        latency_ms: float = 10.0,
    ) -> Participant:
        """Add a market participant."""
        participant = Participant(
            participant_id=self.next_participant_id,
            participant_type=participant_type,
            latency_ms=latency_ms,
        )
        self.participants.append(participant)
        self.next_participant_id += 1
        return participant

    def get_best_bid_ask(self) -> tuple[float, float]:
        """Get current best bid and ask prices."""
        best_bid = self.bids[0].price if self.bids else self.initial_price * 0.99
        best_ask = self.asks[0].price if self.asks else self.initial_price * 1.01
        return best_bid, best_ask

    def get_spread(self) -> float:
        """Get current bid-ask spread."""
        best_bid, best_ask = self.get_best_bid_ask()
        return best_ask - best_bid

    def has_arbitrage_opportunity(self) -> bool:
        """Check for arbitrage opportunities."""
        best_bid, best_ask = self.get_best_bid_ask()
        return best_bid > best_ask

    def has_pump_opportunity(self) -> bool:
        """Check for pump opportunity (rapid price increase)."""
        if len(self.trades) < 10:
            return False

        recent_trades = self.trades[-10:]
        avg_price = np.mean([t.price for t in recent_trades])
        momentum = recent_trades[-1].price / avg_price - 1.0

        return momentum > 0.05  # 5% momentum

    def has_mev_opportunity(self) -> bool:
        """Check for MEV opportunities."""
        # Large spread = MEV opportunity
        spread = self.get_spread()
        mid_price = sum(self.get_best_bid_ask()) / 2.0
        relative_spread = spread / mid_price

        return relative_spread > 0.01  # 1% spread

    def place_order(self, order: Order) -> list[Trade]:
        """Place an order and return filled trades."""
        self.events.append(
            OrderBookEvent(
                event_type="order_placed",
                timestamp=self.current_time,
                order=order,
            )
        )

        trades = []

        # Market orders: match against opposite side
        if order.order_type == OrderType.MARKET:
            book = self.asks if order.side == OrderSide.BID else self.bids
            while book and order.remaining > 0.001:
                best_order = book[0]

                # Check if prices match
                if order.side == OrderSide.BID and order.price < best_order.price:
                    break
                if order.side == OrderSide.ASK and order.price > best_order.price:
                    break

                # Fill trade
                fill_size = min(order.remaining, best_order.remaining)
                trade_price = best_order.price

                trade = Trade(
                    trade_id=self.trade_id,
                    price=trade_price,
                    size=fill_size,
                    bid_order=order if order.side == OrderSide.BID else best_order,
                    ask_order=best_order if order.side == OrderSide.ASK else order,
                    timestamp=self.current_time,
                    is_mev=best_order.participant_type == ParticipantType.MEV_BOT,
                    mev_type=(
                        "sandwich"
                        if best_order.participant_type == ParticipantType.MEV_BOT
                        else None
                    ),
                )
                self.trade_id += 1

                trades.append(trade)
                self.trades.append(trade)
                self.last_trade_price = trade_price

                self.events.append(
                    OrderBookEvent(
                        event_type="trade",
                        timestamp=self.current_time,
                        trade=trade,
                    )
                )

                # Update order quantities
                order.remaining -= fill_size
                best_order.remaining -= fill_size

                # Remove fully filled orders from book
                if best_order.remaining < 0.001:
                    heapq.heappop(book)
                    del self.orders[best_order.order_id]

            # If IOC or FOK order, cancel remaining
            if order.order_type in [OrderType.IOC, OrderType.FOK]:
                if order.order_type == OrderType.FOK and order.remaining > 0.001:
                    # Cancel entire trade (FOK)
                    for trade in trades:
                        self.trades.remove(trade)
                    return []

                # Cancel remaining (IOC)
                order.remaining = 0.0
                return trades

        # Limit orders: add to book
        else:
            book = self.bids if order.side == OrderSide.BID else self.asks
            heapq.heappush(book, order)
            self.orders[order.order_id] = order

        return trades

    def simulate(
        self,
        duration_minutes: float = 60.0,
        time_step_seconds: float = 1.0,
        n_market_makers: int = 5,
        n_retail: int = 50,
        n_snipers: int = 10,
        n_mev_bots: int = 5,
    ) -> list[OrderBookEvent]:
        """Run order book simulation.

        Args:
            duration_minutes: Simulation duration in minutes
            time_step_seconds: Time step size in seconds
            n_market_makers: Number of market makers
            n_retail: Number of retail participants
            n_snipers: Number of snipers
            n_mev_bots: Number of MEV bots

        Returns:
            List of order book events
        """
        # Add participants
        for _ in range(n_market_makers):
            self.add_participant(
                ParticipantType.MARKET_MAKER,
                latency_ms=np.random.uniform(5, 20),
            )

        for _ in range(n_retail):
            self.add_participant(
                ParticipantType.RETAIL,
                latency_ms=np.random.uniform(100, 500),
            )

        for _ in range(n_snipers):
            self.add_participant(
                ParticipantType.SNIPER,
                latency_ms=np.random.uniform(10, 50),
            )

        for _ in range(n_mev_bots):
            self.add_participant(
                ParticipantType.MEV_BOT,
                latency_ms=np.random.uniform(1, 10),
            )

        # Simulation loop
        n_steps = int(duration_minutes * 60 / time_step_seconds)
        for step in range(n_steps):
            self.current_time = step * time_step_seconds

            # Each participant decides whether to place an order
            for participant in self.participants:
                if participant.should_place_order(self.current_time, self):
                    order = participant.generate_order(self.current_time, self)
                    if order:
                        self.place_order(order)

            # Random order cancellations
            if np.random.random() < 0.05:  # 5% chance per step
                self._cancel_random_order()

            # Take snapshot periodically
            if step % 60 == 0:  # Every minute
                self._take_snapshot()

        logger.info(f"Simulation complete: {len(self.events)} events, {len(self.trades)} trades")
        logger.info(f"MEV trades: {sum(1 for t in self.trades if t.is_mev)}")

        return self.events

    def _cancel_random_order(self):
        """Cancel a random order."""
        if not self.orders:
            return

        order_id = np.random.choice(list(self.orders.keys()))
        order = self.orders[order_id]

        # Remove from book
        if order.side == OrderSide.BID:
            self.bids = [o for o in self.bids if o.order_id != order_id]
            heapq.heapify(self.bids)
        else:
            self.asks = [o for o in self.asks if o.order_id != order_id]
            heapq.heapify(self.asks)

        del self.orders[order_id]

        self.events.append(
            OrderBookEvent(
                event_type="order_cancelled",
                timestamp=self.current_time,
                order=order,
            )
        )

    def _take_snapshot(self):
        """Take a snapshot of the order book."""
        # Get bids and asks (sorted)
        bids = []
        temp_bids = []
        while self.bids and len(bids) < self.n_price_levels:
            order = heapq.heappop(self.bids)
            bids.append((order.price, order.remaining))
            temp_bids.append(order)

        asks = []
        temp_asks = []
        while self.asks and len(asks) < self.n_price_levels:
            order = heapq.heappop(self.asks)
            asks.append((order.price, order.remaining))
            temp_asks.append(order)

        # Put orders back
        for order in temp_bids:
            heapq.heappush(self.bids, order)
        for order in temp_asks:
            heapq.heappush(self.asks, order)

        best_bid = bids[0][0] if bids else self.initial_price * 0.99
        best_ask = asks[0][0] if asks else self.initial_price * 1.01

        snapshot = OrderBookSnapshot(
            timestamp=self.current_time,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=best_ask - best_bid,
            total_volume=sum(t.size for t in self.trades),
            last_trade_price=self.last_trade_price,
        )

        self.snapshots.append(snapshot)

    def extract_features(self) -> np.ndarray:
        """Extract ML features from order book history.

        Returns:
            features: [n_snapshots, n_features] feature matrix
        """
        if not self.snapshots:
            return np.array([])

        features = []
        for i, snapshot in enumerate(self.snapshots):
            feature_vector = []

            # 1. Price features
            mid_price = (snapshot.best_bid + snapshot.best_ask) / 2.0
            feature_vector.append(mid_price)
            feature_vector.append(snapshot.spread)
            feature_vector.append(snapshot.spread / mid_price)

            # 2. Depth features
            bid_depth = sum(size for _, size in snapshot.bids)
            ask_depth = sum(size for _, size in snapshot.asks)
            feature_vector.append(bid_depth)
            feature_vector.append(ask_depth)
            feature_vector.append(bid_depth / ask_depth if ask_depth > 0 else 1.0)

            # 3. Order imbalance
            feature_vector.append((bid_depth - ask_depth) / (bid_depth + ask_depth))

            # 4. Volatility (recent trades)
            if i >= 10:
                recent_prices = [s.last_trade_price for s in self.snapshots[i - 10 : i]]
                volatility = np.std(recent_prices) / np.mean(recent_prices)
            else:
                volatility = 0.0
            feature_vector.append(volatility)

            # 5. Trading activity
            recent_trades = [
                t for t in self.trades if t.timestamp > self.snapshots[i].timestamp - 60
            ]
            feature_vector.append(len(recent_trades))
            feature_vector.append(sum(t.size for t in recent_trades))

            # 6. MEV activity
            recent_mev = [t for t in recent_trades if t.is_mev]
            feature_vector.append(len(recent_mev))
            feature_vector.append(len(recent_mev) / len(recent_trades) if recent_trades else 0.0)

            # 7. Momentum
            if i >= 10:
                momentum = (
                    snapshot.last_trade_price - self.snapshots[i - 10].last_trade_price
                ) / self.snapshots[i - 10].last_trade_price
            else:
                momentum = 0.0
            feature_vector.append(momentum)

            features.append(feature_vector)

        return np.array(features)


if __name__ == "__main__":
    logger.info("Starting order book simulation")

    # Create simulator
    simulator = OrderBookSimulator(
        initial_price=1.0,
        tick_size=0.0001,
        liquidity_usd=100_000,
    )

    # Run simulation
    events = simulator.simulate(
        duration_minutes=60,
        n_market_makers=5,
        n_retail=50,
        n_snipers=10,
        n_mev_bots=5,
    )

    # Extract features
    features = simulator.extract_features()
    n_features = features.shape[1] if len(features) > 0 else 0
    logger.info(f"Extracted {len(features)} snapshots with {n_features} features")

    # Print statistics
    print("\n=== Order Book Statistics ===")
    print(f"Total events: {len(events)}")
    print(f"Total trades: {len(simulator.trades)}")
    print(f"MEV trades: {sum(1 for t in simulator.trades if t.is_mev)}")
    print(f"Snapshots: {len(simulator.snapshots)}")

    if simulator.snapshots:
        final = simulator.snapshots[-1]
        print(f"Final price: {final.last_trade_price:.4f}")
        print(f"Final spread: {final.spread:.4f}")
        print(f"Total volume: {final.total_volume:.2f}")
