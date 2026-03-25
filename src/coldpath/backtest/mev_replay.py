"""
Historical MEV Replay Engine for Sniper Performance Testing.

Replays historical sandwich attacks to model realistic slippage and
execution quality for token sniping strategies.

Key features:
- Historical MEV pattern analysis from Dune Analytics
- Sandwich attack probability modeling by trade size
- Realistic slippage estimation with MEV impact
- Monte Carlo simulation with historical MEV distributions
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from .dune_provider import DuneDataProvider

logger = logging.getLogger(__name__)


class MEVType(Enum):
    """Types of MEV attacks."""

    SANDWICH = "sandwich"
    FRONTRUN = "frontrun"
    BACKRUN = "backrun"
    ARBITRAGE = "arbitrage"


@dataclass
class MEVEvent:
    """A historical MEV event."""

    timestamp_ms: int
    mev_type: MEVType
    token_mint: str
    pool_address: str
    victim_tx: str
    victim_amount_sol: float
    profit_sol: float
    price_impact_bps: int
    searcher_address: str
    block_slot: int

    @property
    def profit_rate(self) -> float:
        """Calculate profit as percentage of victim amount."""
        if self.victim_amount_sol > 0:
            return (self.profit_sol / self.victim_amount_sol) * 100
        return 0.0


@dataclass
class MEVProfile:
    """Statistical profile of MEV activity for a time period."""

    # Basic stats
    total_events: int
    total_victim_volume_sol: float
    total_mev_profit_sol: float

    # Attack probability by trade size (buckets in SOL)
    attack_prob_by_size: dict[str, float]  # "0-0.1", "0.1-1", "1-10", "10+"

    # Price impact distribution (percentiles)
    impact_percentiles: dict[str, int]  # "p50", "p75", "p90", "p95", "p99"

    # Profit distribution
    avg_profit_sol: float
    max_profit_sol: float
    profit_percentile_95: float

    # Time patterns
    hourly_attack_frequency: dict[int, float]  # hour -> attacks per hour

    def get_attack_probability(self, trade_size_sol: float) -> float:
        """Get probability of being sandwiched based on trade size."""
        if trade_size_sol < 0.1:
            return self.attack_prob_by_size.get("0-0.1", 0.0)
        elif trade_size_sol < 1.0:
            return self.attack_prob_by_size.get("0.1-1", 0.0)
        elif trade_size_sol < 10.0:
            return self.attack_prob_by_size.get("1-10", 0.0)
        else:
            return self.attack_prob_by_size.get("10+", 0.0)

    def sample_impact_bps(self) -> int:
        """Sample a price impact from the historical distribution."""
        # Use weighted random based on percentiles
        r = np.random.random()
        if r < 0.50:
            return self.impact_percentiles.get("p50", 100)
        elif r < 0.75:
            return self.impact_percentiles.get("p75", 200)
        elif r < 0.90:
            return self.impact_percentiles.get("p90", 350)
        elif r < 0.95:
            return self.impact_percentiles.get("p95", 500)
        else:
            return self.impact_percentiles.get("p99", 800)


@dataclass
class MEVReplayConfig:
    """Configuration for MEV replay simulation."""

    # Dune API config
    dune_api_key: str | None = None

    # Simulation parameters
    include_sandwich: bool = True
    include_frontrun: bool = True
    use_historical_distribution: bool = True

    # Default probabilities if no historical data
    default_sandwich_prob: float = 0.15
    default_frontrun_prob: float = 0.05
    default_impact_bps: int = 150

    # Trade size thresholds (SOL) for increased MEV attention
    high_attention_threshold_sol: float = 1.0
    very_high_attention_threshold_sol: float = 10.0

    # Time-based adjustments
    apply_time_patterns: bool = True

    @classmethod
    def from_env(cls) -> "MEVReplayConfig":
        import os

        return cls(
            dune_api_key=os.getenv("DUNE_API_KEY"),
        )


class MEVReplayEngine:
    """
    Replays historical MEV patterns for realistic backtesting.

    Uses historical data from Dune Analytics to:
    1. Model probability of being sandwiched by trade size
    2. Sample realistic price impacts from historical distribution
    3. Apply time-of-day patterns (more MEV during high activity)
    4. Monte Carlo simulation with MEV uncertainty
    """

    def __init__(
        self,
        config: MEVReplayConfig | None = None,
        dune_provider: DuneDataProvider | None = None,
    ):
        self.config = config or MEVReplayConfig()
        self.dune_provider = dune_provider
        self._mev_profile: MEVProfile | None = None
        self._historical_events: list[MEVEvent] = []

    async def load_historical_data(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        mints: list[str] | None = None,
    ) -> MEVProfile:
        """Load and analyze historical MEV data.

        Args:
            start_timestamp_ms: Start of analysis period
            end_timestamp_ms: End of analysis period
            mints: Optional filter for specific tokens

        Returns:
            MEVProfile with statistical analysis
        """
        if not self.dune_provider:
            if self.config.dune_api_key:
                self.dune_provider = DuneDataProvider(api_key=self.config.dune_api_key)
            else:
                logger.warning("No Dune API key - using default MEV profile")
                return self._default_profile()

        # Fetch MEV events from Dune
        events = []
        async for event in self.dune_provider.stream_events(
            start_timestamp_ms, end_timestamp_ms, "dune_mev", mints
        ):
            if event.event_type == "mev_sandwich":
                mev_event = MEVEvent(
                    timestamp_ms=event.timestamp_ms,
                    mev_type=MEVType.SANDWICH,
                    token_mint=event.mint,
                    pool_address=event.pool,
                    victim_tx=event.data.get("victim_tx", ""),
                    victim_amount_sol=float(event.data.get("victim_amount_sol", 0)),
                    profit_sol=float(event.data.get("profit_sol", 0)),
                    price_impact_bps=int(event.data.get("price_impact_bps", 0)),
                    searcher_address=event.data.get("searcher", ""),
                    block_slot=0,
                )
                events.append(mev_event)

        self._historical_events = events
        self._mev_profile = self._analyze_events(events)

        logger.info(
            f"Loaded {len(events)} MEV events, "
            f"total victim volume: {self._mev_profile.total_victim_volume_sol:.2f} SOL"
        )

        return self._mev_profile

    def _analyze_events(self, events: list[MEVEvent]) -> MEVProfile:
        """Analyze historical MEV events to build statistical profile."""
        if not events:
            return self._default_profile()

        # Calculate basic stats
        total_victim_volume = sum(e.victim_amount_sol for e in events)
        total_profit = sum(e.profit_sol for e in events)

        # Attack probability by trade size
        size_buckets = {"0-0.1": [], "0.1-1": [], "1-10": [], "10+": []}
        for e in events:
            size = e.victim_amount_sol
            if size < 0.1:
                size_buckets["0-0.1"].append(1)
            elif size < 1.0:
                size_buckets["0.1-1"].append(1)
            elif size < 10.0:
                size_buckets["1-10"].append(1)
            else:
                size_buckets["10+"].append(1)

        # Estimate probability based on frequency
        # This is simplified - ideally we'd have total trades in each bucket
        total_events = len(events)
        attack_prob = {}
        for bucket, attacks in size_buckets.items():
            # Higher buckets have higher attack probability
            base_prob = len(attacks) / max(total_events, 1)
            attack_prob[bucket] = min(0.5, base_prob * 2)  # Cap at 50%

        # Price impact distribution
        impacts = sorted([e.price_impact_bps for e in events])
        n = len(impacts)
        impact_percentiles = {
            "p50": impacts[int(n * 0.50)] if n > 0 else 100,
            "p75": impacts[int(n * 0.75)] if n > 0 else 200,
            "p90": impacts[int(n * 0.90)] if n > 0 else 350,
            "p95": impacts[int(n * 0.95)] if n > 0 else 500,
            "p99": impacts[min(int(n * 0.99), n - 1)] if n > 0 else 800,
        }

        # Profit stats
        profits = [e.profit_sol for e in events]
        avg_profit = sum(profits) / len(profits) if profits else 0
        max_profit = max(profits) if profits else 0
        profits_sorted = sorted(profits)
        profit_p95 = profits_sorted[int(len(profits_sorted) * 0.95)] if profits_sorted else 0

        # Hourly patterns
        hourly_counts: dict[int, int] = {h: 0 for h in range(24)}
        for e in events:
            hour = (e.timestamp_ms // 3600000) % 24
            hourly_counts[hour] += 1

        total_hours = (
            max(1, (events[-1].timestamp_ms - events[0].timestamp_ms) / 3600000) if events else 1
        )
        hourly_freq = {h: count / total_hours for h, count in hourly_counts.items()}

        return MEVProfile(
            total_events=len(events),
            total_victim_volume_sol=total_victim_volume,
            total_mev_profit_sol=total_profit,
            attack_prob_by_size=attack_prob,
            impact_percentiles=impact_percentiles,
            avg_profit_sol=avg_profit,
            max_profit_sol=max_profit,
            profit_percentile_95=profit_p95,
            hourly_attack_frequency=hourly_freq,
        )

    def _default_profile(self) -> MEVProfile:
        """Return default MEV profile when no historical data available."""
        return MEVProfile(
            total_events=0,
            total_victim_volume_sol=0,
            total_mev_profit_sol=0,
            attack_prob_by_size={
                "0-0.1": 0.05,
                "0.1-1": 0.10,
                "1-10": 0.20,
                "10+": 0.35,
            },
            impact_percentiles={
                "p50": 100,
                "p75": 200,
                "p90": 350,
                "p95": 500,
                "p99": 800,
            },
            avg_profit_sol=0.01,
            max_profit_sol=0.5,
            profit_percentile_95=0.1,
            hourly_attack_frequency={h: 1.0 for h in range(24)},
        )

    def simulate_trade_impact(
        self,
        trade_size_sol: float,
        timestamp_ms: int | None = None,
        base_slippage_bps: int = 0,
    ) -> tuple[int, bool, float]:
        """
        Simulate realistic trade impact including potential MEV.

        Args:
            trade_size_sol: Size of trade in SOL
            timestamp_ms: Trade timestamp (for time-pattern adjustment)
            base_slippage_bps: Base AMM slippage before MEV

        Returns:
            Tuple of (total_slippage_bps, was_sandwiched, mev_loss_sol)
        """
        profile = self._mev_profile or self._default_profile()

        # Get base attack probability
        attack_prob = profile.get_attack_probability(trade_size_sol)

        # Apply time-of-day adjustment
        if timestamp_ms and self.config.apply_time_patterns:
            hour = (timestamp_ms // 3600000) % 24
            hour_factor = profile.hourly_attack_frequency.get(hour, 1.0)
            avg_factor = sum(profile.hourly_attack_frequency.values()) / 24
            if avg_factor > 0:
                attack_prob *= hour_factor / avg_factor

        # Determine if sandwiched
        was_sandwiched = np.random.random() < attack_prob

        if not was_sandwiched:
            return base_slippage_bps, False, 0.0

        # Sample price impact from distribution
        mev_impact_bps = profile.sample_impact_bps()

        # Calculate MEV loss
        mev_loss_sol = trade_size_sol * mev_impact_bps / 10000

        total_slippage = base_slippage_bps + mev_impact_bps

        return total_slippage, True, mev_loss_sol

    def simulate_batch_trades(
        self,
        trade_sizes_sol: np.ndarray,
        timestamps_ms: np.ndarray | None = None,
        base_slippages_bps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate MEV impact for a batch of trades (vectorized).

        Args:
            trade_sizes_sol: Array of trade sizes
            timestamps_ms: Array of timestamps
            base_slippages_bps: Array of base slippages

        Returns:
            Tuple of (total_slippages, sandwiched_flags, mev_losses)
        """
        n = len(trade_sizes_sol)

        if base_slippages_bps is None:
            base_slippages_bps = np.zeros(n)

        total_slippages = np.zeros(n)
        sandwiched = np.zeros(n, dtype=bool)
        mev_losses = np.zeros(n)

        for i in range(n):
            ts = timestamps_ms[i] if timestamps_ms is not None else None
            slippage, was_sandwiched, loss = self.simulate_trade_impact(
                trade_sizes_sol[i], ts, int(base_slippages_bps[i])
            )
            total_slippages[i] = slippage
            sandwiched[i] = was_sandwiched
            mev_losses[i] = loss

        return total_slippages, sandwiched, mev_losses


@dataclass
class MEVAwareBacktestResult:
    """Results from MEV-aware backtesting."""

    # Base metrics
    total_trades: int
    total_pnl_sol: float
    win_rate: float

    # MEV impact
    trades_sandwiched: int
    total_mev_loss_sol: float
    avg_slippage_bps: float
    avg_mev_impact_bps: float

    # Adjusted metrics
    pnl_without_mev: float  # Hypothetical PnL if no MEV
    mev_cost_ratio: float  # MEV loss as % of volume

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "total_pnl_sol": self.total_pnl_sol,
            "win_rate": self.win_rate,
            "trades_sandwiched": self.trades_sandwiched,
            "sandwich_rate": self.trades_sandwiched / max(1, self.total_trades),
            "total_mev_loss_sol": self.total_mev_loss_sol,
            "avg_slippage_bps": self.avg_slippage_bps,
            "avg_mev_impact_bps": self.avg_mev_impact_bps,
            "pnl_without_mev": self.pnl_without_mev,
            "mev_cost_ratio": self.mev_cost_ratio,
        }


class MEVAwareBacktester:
    """
    Backtester with realistic MEV impact modeling.

    Combines vectorized backtesting with historical MEV patterns
    for accurate sniper performance estimation.
    """

    def __init__(
        self,
        mev_engine: MEVReplayEngine | None = None,
        config: MEVReplayConfig | None = None,
    ):
        self.mev_engine = mev_engine or MEVReplayEngine(config)
        self.config = config or MEVReplayConfig()

    async def run_mev_aware_backtest(
        self,
        # OHLCV data
        prices: np.ndarray,
        volumes: np.ndarray,
        timestamps: np.ndarray,
        # Signals (from vectorized backtester)
        entry_signals: np.ndarray,
        # Config
        initial_capital_sol: float = 10.0,
        position_size_pct: float = 0.1,
        take_profit_pct: float = 50.0,
        stop_loss_pct: float = 10.0,
    ) -> MEVAwareBacktestResult:
        """
        Run backtest with realistic MEV impact.

        Args:
            prices: Price array (n_tokens x n_timepoints or n_timepoints)
            volumes: Volume array
            timestamps: Timestamp array (ms)
            entry_signals: Boolean entry signals
            initial_capital_sol: Starting capital
            position_size_pct: Position size as % of capital
            take_profit_pct: Take profit threshold
            stop_loss_pct: Stop loss threshold

        Returns:
            MEVAwareBacktestResult with MEV impact analysis
        """
        # Ensure MEV profile is loaded
        if self.mev_engine._mev_profile is None:
            # Use full timestamp range
            await self.mev_engine.load_historical_data(
                int(timestamps.min()),
                int(timestamps.max()),
            )

        # Reshape if 1D
        if len(prices.shape) == 1:
            prices = prices.reshape(1, -1)
            entry_signals = entry_signals.reshape(1, -1)

        n_tokens, n_timepoints = prices.shape

        # Track trades
        trades = []
        capital = initial_capital_sol

        for i in range(n_tokens):
            t = 0
            while t < n_timepoints:
                if entry_signals[i, t]:
                    entry_price = prices[i, t]
                    if entry_price <= 0:
                        t += 1
                        continue

                    trade_size = capital * position_size_pct
                    if trade_size < 0.001:
                        t += 1
                        continue

                    # Simulate MEV on entry
                    entry_slippage, entry_sandwiched, entry_mev_loss = (
                        self.mev_engine.simulate_trade_impact(
                            trade_size,
                            int(timestamps[t]) if t < len(timestamps) else None,
                            base_slippage_bps=0,
                        )
                    )

                    # Adjust entry price for slippage
                    entry_price_adj = entry_price * (1 + entry_slippage / 10000)
                    tokens = (trade_size - entry_mev_loss) / entry_price_adj

                    # Find exit
                    exit_t = t + 1
                    exit_reason = "max_hold"
                    max_hold = 60  # 60 periods

                    while exit_t < min(t + max_hold, n_timepoints):
                        current_price = prices[i, exit_t]
                        if current_price <= 0:
                            exit_t += 1
                            continue

                        pnl_pct = (current_price - entry_price_adj) / entry_price_adj * 100

                        if pnl_pct >= take_profit_pct:
                            exit_reason = "take_profit"
                            break
                        elif pnl_pct <= -stop_loss_pct:
                            exit_reason = "stop_loss"
                            break

                        exit_t += 1

                    # Get exit price
                    if exit_t < n_timepoints:
                        exit_price = prices[i, exit_t]
                    else:
                        exit_price = prices[i, -1]

                    # Simulate MEV on exit
                    exit_value = tokens * exit_price
                    exit_slippage, exit_sandwiched, exit_mev_loss = (
                        self.mev_engine.simulate_trade_impact(
                            exit_value,
                            int(timestamps[exit_t]) if exit_t < len(timestamps) else None,
                            base_slippage_bps=0,
                        )
                    )

                    # Final exit value after MEV
                    exit_price_adj = exit_price * (1 - exit_slippage / 10000)
                    final_value = tokens * exit_price_adj - exit_mev_loss

                    # Record trade
                    trade_pnl = final_value - trade_size
                    hypothetical_pnl = tokens * exit_price - trade_size  # Without MEV

                    trades.append(
                        {
                            "entry_t": t,
                            "exit_t": exit_t,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "trade_size": trade_size,
                            "pnl": trade_pnl,
                            "pnl_without_mev": hypothetical_pnl,
                            "entry_sandwiched": entry_sandwiched,
                            "exit_sandwiched": exit_sandwiched,
                            "entry_slippage_bps": entry_slippage,
                            "exit_slippage_bps": exit_slippage,
                            "total_mev_loss": entry_mev_loss + exit_mev_loss,
                            "exit_reason": exit_reason,
                        }
                    )

                    capital += trade_pnl
                    t = exit_t + 1
                else:
                    t += 1

        # Calculate results
        if not trades:
            return MEVAwareBacktestResult(
                total_trades=0,
                total_pnl_sol=0,
                win_rate=0,
                trades_sandwiched=0,
                total_mev_loss_sol=0,
                avg_slippage_bps=0,
                avg_mev_impact_bps=0,
                pnl_without_mev=0,
                mev_cost_ratio=0,
            )

        total_pnl = sum(t["pnl"] for t in trades)
        pnl_without_mev = sum(t["pnl_without_mev"] for t in trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        sandwiched = sum(1 for t in trades if t["entry_sandwiched"] or t["exit_sandwiched"])
        total_mev_loss = sum(t["total_mev_loss"] for t in trades)
        total_volume = sum(t["trade_size"] for t in trades)

        avg_entry_slip = np.mean([t["entry_slippage_bps"] for t in trades])
        avg_exit_slip = np.mean([t["exit_slippage_bps"] for t in trades])
        avg_slippage = (avg_entry_slip + avg_exit_slip) / 2

        # MEV impact = difference between actual and hypothetical slippage
        mev_impact = (pnl_without_mev - total_pnl) / max(len(trades), 1)
        avg_mev_impact_bps = (
            (mev_impact / (total_volume / len(trades))) * 10000 if total_volume > 0 else 0
        )

        return MEVAwareBacktestResult(
            total_trades=len(trades),
            total_pnl_sol=total_pnl,
            win_rate=wins / len(trades),
            trades_sandwiched=sandwiched,
            total_mev_loss_sol=total_mev_loss,
            avg_slippage_bps=avg_slippage,
            avg_mev_impact_bps=avg_mev_impact_bps,
            pnl_without_mev=pnl_without_mev,
            mev_cost_ratio=total_mev_loss / total_volume if total_volume > 0 else 0,
        )


# Convenience function
async def run_mev_aware_backtest(
    prices: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray,
    entry_signals: np.ndarray,
    dune_api_key: str | None = None,
    **kwargs,
) -> MEVAwareBacktestResult:
    """
    Run MEV-aware backtest with default configuration.

    Example:
        results = await run_mev_aware_backtest(
            prices=df['close'].values,
            volumes=df['volume'].values,
            timestamps=df['timestamp'].values,
            entry_signals=signals,
        )
        print(f"PnL: {results.total_pnl_sol:.4f} SOL")
        print(f"MEV Loss: {results.total_mev_loss_sol:.4f} SOL")
    """
    config = MEVReplayConfig(dune_api_key=dune_api_key)
    backtester = MEVAwareBacktester(config=config)

    return await backtester.run_mev_aware_backtest(
        prices=prices,
        volumes=volumes,
        timestamps=timestamps,
        entry_signals=entry_signals,
        **kwargs,
    )
