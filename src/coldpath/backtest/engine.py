"""
Backtest engine.

Walk-forward backtesting with no-leakage guarantees.
Supports pluggable strategies and realistic fill simulation.

Enhanced with:
- 4-model ensemble integration (Isolation Forest, LSTM, XGBoost, Kelly)
- Kelly Criterion position sizing
- Rug detection with veto logic
"""

import inspect
import json
import logging
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator

import numpy as np

from ..autotrader.coordinator import (
    AutoTraderConfig,
    AutoTraderCoordinator,
)
from ..autotrader.coordinator import (
    TradingCandidate as LiveTradingCandidate,
)
from ..learning.ensemble import EnsembleDecision, EnsembleSignal, ModelEnsemble
from ..learning.feature_engineering import FeatureEngineer
from ..models.ml_trade_context import MLTradeContext
from ..trading.advanced_orchestrator import (
    AdvancedOrchestratorConfig,
    AdvancedStrategyOrchestrator,
)
from ..trading.kelly_criterion import KellyCriterion
from ..validation.data_quality import DataQualityValidator
from .data_provider import BitqueryDataProvider, LocalTelemetryDataProvider, MarketEvent
from .dune_provider import DuneDataProvider
from .fill_simulator import FillSimulator
from .metrics import MetricsEngine
from .replay_clock import ReplayClock
from .synthetic_scenarios import (
    MarketEvent as ScenarioMarketEvent,
)
from .synthetic_scenarios import (
    ScenarioEngine,
    create_default_scenario_engine,
)

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    """A trading signal from the strategy."""

    type: SignalType
    mint: str
    timestamp_ms: int
    signal_id: str | None = None
    confidence: float = 1.0
    target_amount_sol: float | None = None
    reason: str | None = None
    expected_price: float | None = None
    slippage_bps: int | None = None
    ml_context: dict[str, Any] | None = None


@dataclass
class Position:
    """An open position."""

    mint: str
    entry_price: float
    amount_tokens: float
    amount_sol: float
    entry_timestamp_ms: int
    current_price: float = 0.0
    high_price: float = 0.0  # For trailing stop
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None

    @property
    def unrealized_pnl_sol(self) -> float:
        """Calculate unrealized PnL in SOL."""
        if self.entry_price == 0:
            return 0
        current_value = self.amount_tokens * self.current_price
        return current_value - self.amount_sol

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized PnL percentage."""
        if self.amount_sol == 0:
            return 0
        return (self.unrealized_pnl_sol / self.amount_sol) * 100

    def update_price(self, price: float):
        """Update current price and track high."""
        self.current_price = price
        if price > self.high_price:
            self.high_price = price

    def should_stop_loss(self) -> bool:
        """Check if stop loss should trigger."""
        if self.stop_loss_pct is None or self.entry_price == 0:
            return False
        pnl_pct = (self.current_price - self.entry_price) / self.entry_price * 100
        return pnl_pct <= -self.stop_loss_pct

    def should_take_profit(self) -> bool:
        """Check if take profit should trigger."""
        if self.take_profit_pct is None or self.entry_price == 0:
            return False
        pnl_pct = (self.current_price - self.entry_price) / self.entry_price * 100
        return pnl_pct >= self.take_profit_pct


@dataclass
class BacktestConfig:
    """Backtest configuration."""

    start_timestamp_ms: int
    end_timestamp_ms: int
    data_source: str = "bitquery_ohlcv"
    initial_capital_sol: float = 10.0
    max_position_sol: float = 0.15  # TUNED: Increased from 0.10 for high-conviction trades
    max_positions: int = 5
    slippage_bps: int = 100  # Jito bundles achieve 0.5-1.5% real slippage
    default_stop_loss_pct: float = 10.0  # TUNED: Increased from 8% for memecoin volatility
    default_take_profit_pct: float = 20.0  # TUNED: Increased from 15% to let winners run
    min_liquidity_usd: float = 10000.0
    random_seed: int | None = None
    use_live_signal_generation: bool = False

    # Ensemble and Kelly settings
    use_ensemble: bool = False
    use_kelly_sizing: bool = False
    kelly_safety_factor: float = 0.50  # Half-Kelly: industry standard for algo trading
    kelly_max_position_pct: float = 20.0  # TUNED: Allow high-conviction positions (was 15%)
    rug_veto_threshold: float = 0.80  # Raised: reduce false positives on volatile tokens
    rug_exit_threshold: float = 0.90  # Only hard-exit on clear rugs
    min_confidence_for_trade: float = 0.35  # Let more signals through, rely on rug veto for safety

    # Target metrics (for validation)
    target_win_rate: float = 0.50  # Lowered from 0.55: 50%+ with good R/R is profitable
    target_sharpe: float = 1.5  # Lowered from 2.0: more realistic for memecoins
    target_max_drawdown: float = 30.0  # Raised from 25%: memecoins are volatile

    # CAIA-style adversarial evaluation
    enable_adversarial_scenarios: bool = False
    adversarial_seed: int | None = 42
    honeypot_probability: float = 0.10
    sandwich_probability: float = 0.15
    rug_probability: float = 0.05
    flash_crash_probability: float = 0.03
    whale_dump_probability: float = 0.05
    catastrophic_loss_trade_pct: float = 10.0
    catastrophic_drawdown_pct: float = 25.0

    # End-to-end orchestration (SAPT + semantic + mean-field)
    enable_advanced_orchestration: bool = True
    enable_sapt_spatial_model: bool = True
    enable_semantic_fragmentation_model: bool = True
    enable_mean_field_stablecoin_model: bool = True
    orchestration_min_samples: int = 5
    orchestration_refit_interval: int = 8
    orchestration_max_assets: int = 8
    orchestration_stablecoin_symbols_csv: str = "USDC,USDT,DAI,PYUSD,FDUSD,USDE"
    orchestration_max_confidence_multiplier: float = 1.15
    orchestration_min_confidence_multiplier: float = 0.55
    orchestration_stressed_regime_multiplier: float = 0.80
    orchestration_parity_boost_scale: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_timestamp": self.start_timestamp_ms,
            "end_timestamp": self.end_timestamp_ms,
            "data_source": self.data_source,
            "initial_capital_sol": self.initial_capital_sol,
            "max_position_sol": self.max_position_sol,
            "max_positions": self.max_positions,
            "slippage_bps": self.slippage_bps,
            "random_seed": self.random_seed,
            "use_live_signal_generation": self.use_live_signal_generation,
            "use_ensemble": self.use_ensemble,
            "use_kelly_sizing": self.use_kelly_sizing,
            "kelly_safety_factor": self.kelly_safety_factor,
            "target_win_rate": self.target_win_rate,
            "target_sharpe": self.target_sharpe,
            "enable_adversarial_scenarios": self.enable_adversarial_scenarios,
            "adversarial_seed": self.adversarial_seed,
            "honeypot_probability": self.honeypot_probability,
            "sandwich_probability": self.sandwich_probability,
            "rug_probability": self.rug_probability,
            "flash_crash_probability": self.flash_crash_probability,
            "whale_dump_probability": self.whale_dump_probability,
            "catastrophic_loss_trade_pct": self.catastrophic_loss_trade_pct,
            "catastrophic_drawdown_pct": self.catastrophic_drawdown_pct,
            "enable_advanced_orchestration": self.enable_advanced_orchestration,
            "enable_sapt_spatial_model": self.enable_sapt_spatial_model,
            "enable_semantic_fragmentation_model": self.enable_semantic_fragmentation_model,
            "enable_mean_field_stablecoin_model": self.enable_mean_field_stablecoin_model,
            "orchestration_min_samples": self.orchestration_min_samples,
            "orchestration_refit_interval": self.orchestration_refit_interval,
            "orchestration_max_assets": self.orchestration_max_assets,
            "orchestration_stablecoin_symbols_csv": self.orchestration_stablecoin_symbols_csv,
            "orchestration_max_confidence_multiplier": self.orchestration_max_confidence_multiplier,
            "orchestration_min_confidence_multiplier": self.orchestration_min_confidence_multiplier,
            "orchestration_stressed_regime_multiplier": (
                self.orchestration_stressed_regime_multiplier
            ),
            "orchestration_parity_boost_scale": self.orchestration_parity_boost_scale,
        }


class Strategy(ABC):
    """Base class for trading strategies."""

    @abstractmethod
    def on_event(
        self,
        event: MarketEvent,
        positions: dict[str, Position],
        capital: float,
    ) -> Signal | None:
        """Process a market event and return a signal if any."""
        pass

    def on_position_opened(self, position: Position):  # noqa: B027
        """Optional hook called when a position is opened. Override in subclass if needed."""

    def on_position_closed(self, position: Position, pnl: float):  # noqa: B027
        """Optional hook called when a position is closed. Override in subclass if needed."""


class MomentumStrategy(Strategy):
    """Simple momentum-based strategy for testing."""

    def __init__(
        self,
        min_volume_change_pct: float = 50.0,
        min_price_change_pct: float = 5.0,
        max_age_hours: float = 1.0,
    ):
        self.min_volume_change_pct = min_volume_change_pct
        self.min_price_change_pct = min_price_change_pct
        self.max_age_hours = max_age_hours
        self.price_history: dict[str, deque] = {}
        self._max_history_mints = 200

    def reset(self) -> None:
        """Clear accumulated price history."""
        self.price_history.clear()

    def on_event(
        self,
        event: MarketEvent,
        positions: dict[str, Position],
        capital: float,
    ) -> Signal | None:
        """Generate signal based on momentum indicators."""
        if event.event_type != "ohlcv":
            return None

        mint = event.mint
        data = event.data

        if mint not in self.price_history:
            if len(self.price_history) >= self._max_history_mints:
                oldest = next(iter(self.price_history))
                del self.price_history[oldest]
            self.price_history[mint] = deque(maxlen=10)
        self.price_history[mint].append(data.get("close", 0))

        # Need at least 3 data points
        if len(self.price_history[mint]) < 3:
            return None

        prices = self.price_history[mint]
        current_price = prices[-1]
        prev_price = prices[-2]

        if prev_price == 0:
            return None

        price_change_pct = (current_price - prev_price) / prev_price * 100

        # Check for buy signal
        if price_change_pct >= self.min_price_change_pct:
            if mint not in positions:
                return Signal(
                    type=SignalType.BUY,
                    mint=mint,
                    timestamp_ms=event.timestamp_ms,
                    confidence=min(1.0, price_change_pct / 20),
                    reason=f"Momentum: {price_change_pct:.1f}% price increase",
                )

        # Check for sell signal (position exits)
        elif price_change_pct <= -self.min_price_change_pct / 2:
            if mint in positions:
                return Signal(
                    type=SignalType.SELL,
                    mint=mint,
                    timestamp_ms=event.timestamp_ms,
                    reason=f"Momentum reversal: {price_change_pct:.1f}%",
                )

        return None


class LiveAutoTraderStrategy(Strategy):
    """Backtest entry strategy that reuses the live AutoTrader decision path."""

    def __init__(
        self,
        coordinator: AutoTraderCoordinator | None = None,
        coordinator_config: AutoTraderConfig | None = None,
    ):
        self.coordinator = coordinator or AutoTraderCoordinator(
            config=coordinator_config
            or AutoTraderConfig(
                paper_mode=True,
                skip_learning=True,
            )
        )
        self._started = False

    async def _ensure_started(self) -> None:
        if self._started:
            return
        await self.coordinator.start()
        self._started = True

    def reset(self) -> None:
        self._started = False

    async def on_event(
        self,
        event: MarketEvent,
        positions: dict[str, Position],
        capital: float,
    ) -> Signal | None:
        if event.event_type != "ohlcv" or event.mint in positions:
            return None

        await self._ensure_started()

        data = event.data or {}
        price = float(data.get("close", data.get("price", 0.0)) or 0.0)
        liquidity_usd = float(
            data.get("liquidity_usd", data.get("liquidity_sol", 0.0) * max(price, 1.0)) or 0.0
        )
        volume_24h_usd = float(
            data.get("volume_24h_usd", data.get("volume_24h", data.get("volume", 0.0))) or 0.0
        )

        candidate = LiveTradingCandidate(
            token_mint=event.mint,
            token_symbol=str(data.get("symbol") or event.mint[:8]),
            pool_address=str(event.pool or f"backtest:{event.mint}"),
            price_usd=max(price, 0.0001),
            liquidity_usd=max(liquidity_usd, 0.0),
            fdv_usd=float(data.get("fdv_usd", 0.0) or 0.0),
            volume_24h_usd=max(volume_24h_usd, 0.0),
            holder_count=int(data.get("holder_count", 0) or 0),
            age_seconds=int(data.get("age_seconds", 0) or 0),
            mint_authority_enabled=bool(data.get("mint_authority_enabled", False)),
            freeze_authority_enabled=bool(data.get("freeze_authority_enabled", False)),
            lp_burn_pct=float(data.get("lp_burn_pct", data.get("lp_lock_pct", 0.0)) or 0.0),
            top_holder_pct=float(data.get("top_holder_pct", 0.0) or 0.0),
            volatility_1h=float(data.get("volatility_1h", data.get("volatility_5m", 0.0)) or 0.0),
            price_change_1h=float(
                data.get("price_change_1h", data.get("price_momentum_30s", 0.0)) or 0.0
            ),
            discovery_timestamp=datetime.now(),
            discovery_source=str(data.get("source") or "backtest"),
        )

        live_signal = await self.coordinator.process_candidate(candidate)
        if live_signal is None:
            return None

        ml_context_payload = (
            live_signal.ml_context.to_dict() if live_signal.ml_context is not None else None
        )
        if ml_context_payload is not None:
            round_trip = MLTradeContext.from_dict(ml_context_payload)
            ml_context_payload = round_trip.to_dict() if round_trip is not None else None

        return Signal(
            type=SignalType.BUY,
            mint=event.mint,
            timestamp_ms=event.timestamp_ms,
            signal_id=live_signal.signal_id,
            confidence=live_signal.confidence,
            target_amount_sol=live_signal.position_size_sol,
            reason=live_signal.selected_arm,
            expected_price=live_signal.expected_price,
            slippage_bps=live_signal.slippage_bps,
            ml_context=ml_context_payload,
        )


@dataclass
class BacktestBatchStats:
    """Batch submission counters mirrored from the live HotPath path."""

    accepted_count: int = 0
    rejected_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
        }


class BacktestEngine:
    """Walk-forward backtesting engine with pluggable strategies.

    Enhanced with:
    - Model ensemble integration for signal generation
    - Kelly Criterion for position sizing
    - Rug detection with automatic veto
    """

    def __init__(
        self,
        bitquery_api_key: str | None = None,
        strategy: Strategy | None = None,
        bitquery_provider: BitqueryDataProvider | None = None,
        dune_provider: DuneDataProvider | None = None,
        data_validator: DataQualityValidator | None = None,
        validation_batch_size: int = 1000,
        ensemble: ModelEnsemble | None = None,
        kelly_calculator: KellyCriterion | None = None,
        feature_engineer: FeatureEngineer | None = None,
    ):
        self.bitquery_provider = bitquery_provider or BitqueryDataProvider(bitquery_api_key)
        self.dune_provider = dune_provider
        self.local_telemetry_provider = LocalTelemetryDataProvider()
        self.data_validator = data_validator or DataQualityValidator()
        self.validation_batch_size = validation_batch_size
        self.fill_simulator = FillSimulator()
        self.metrics_engine = MetricsEngine()
        self.strategy = strategy or MomentumStrategy()
        self._active_strategy: Strategy = self.strategy

        # Ensemble and Kelly integration
        self.ensemble = ensemble
        self.kelly = kelly_calculator
        self.feature_engineer = feature_engineer or FeatureEngineer()

        # Price sequence buffer for LSTM (per mint) — deque auto-evicts in O(1)
        self._price_sequences: dict[str, deque] = {}
        self._sequence_length = 60
        self._max_price_sequence_mints = 500  # Prevent unbounded growth
        self._batch_stats = BacktestBatchStats()

    def set_ensemble(self, ensemble: ModelEnsemble) -> None:
        """Set the model ensemble for signal generation."""
        self.ensemble = ensemble

    def set_kelly(self, kelly: KellyCriterion) -> None:
        """Set the Kelly Criterion calculator."""
        self.kelly = kelly

    def reset_state(self) -> None:
        """Reset internal state for clean backtest isolation.

        Call between consecutive backtests to prevent state leakage
        and free memory from accumulated price sequences.
        """
        self._price_sequences.clear()
        self._batch_stats = BacktestBatchStats()
        self._active_strategy = self.strategy
        if hasattr(self._active_strategy, "reset"):
            self._active_strategy.reset()

    def _extract_features_from_event(self, event: MarketEvent) -> np.ndarray:
        """Extract 50 features from a market event."""
        data = event.data

        raw_features = {
            # Liquidity
            "pool_tvl_sol": data.get("liquidity_sol", data.get("liquidity_usd", 0)),
            "pool_age_seconds": data.get("age_seconds", 0),
            "lp_lock_percentage": data.get("lp_burn_pct", data.get("lp_lock_pct", 0)),
            "lp_concentration": data.get("lp_concentration", 0),
            "slippage_1pct": data.get("slippage_1pct", 0),
            "slippage_5pct": data.get("slippage_5pct", 0),
            # Holder
            "holder_count_unique": data.get("holder_count", 0),
            "top_10_holder_concentration": data.get("top_holder_pct", 0),
            "holder_growth_velocity": data.get("holder_growth_rate", 0),
            "mint_authority_revoked": not data.get("mint_authority_enabled", True),
            "token_freezeable": data.get("freeze_authority_enabled", False),
            # Price & Volume
            "price_momentum_30s": data.get("price_change_1h", 0),
            "price_momentum_5m": data.get("price_change_24h", 0),
            "volatility_5m": data.get("volatility_1h", 0),
            "buy_volume_ratio": data.get("buy_volume_ratio", 0.5),
            # Risk
            "rug_pull_ml_score": data.get("goplus_risk_score", 0),
            # Legacy compatibility
            "liquidity_usd": data.get("liquidity_usd", 0),
            "volume_24h_usd": data.get("volume_24h", 0),
            "fdv_usd": data.get("fdv_usd", 0),
        }

        fs = self.feature_engineer.extract(raw_features)
        return fs.features

    def _update_price_sequence(self, mint: str, event: MarketEvent) -> np.ndarray | None:
        """Update price sequence buffer for LSTM and return if ready."""
        if mint not in self._price_sequences:
            if len(self._price_sequences) >= self._max_price_sequence_mints:
                oldest_mint = next(iter(self._price_sequences))
                del self._price_sequences[oldest_mint]
            self._price_sequences[mint] = deque(maxlen=self._sequence_length)

        data = event.data
        price = data.get("close", data.get("price", 0))
        volume = data.get("volume", data.get("volume_24h", 0))
        volatility = data.get("volatility_1h", 0)

        # Simple feature vector for LSTM
        seq_features = np.array(
            [
                price,
                volume,
                volatility,
                data.get("price_change_1h", 0),
                data.get("buy_volume_ratio", 0.5),
            ]
        )

        self._price_sequences[mint].append(seq_features)

        # Return sequence if we have enough data (deque auto-evicts old entries)
        if len(self._price_sequences[mint]) >= self._sequence_length:
            return np.array(self._price_sequences[mint])

        return None

    def _get_ensemble_decision(
        self,
        event: MarketEvent,
        capital: float,
        config: BacktestConfig,
    ) -> EnsembleDecision | None:
        """Get trading decision from ensemble."""
        if self.ensemble is None:
            return None

        mint = event.mint

        # Extract features
        features_50 = self._extract_features_from_event(event)

        # Get price sequence
        price_sequence = self._update_price_sequence(mint, event)

        # Get current price
        current_price = event.data.get("close", event.data.get("price", 1.0))

        if price_sequence is not None:
            # Full evaluation with LSTM
            decision = self.ensemble.evaluate(
                features_50=features_50,
                price_sequence=price_sequence,
                current_price=current_price,
                capital=capital,
            )
        else:
            # Simplified evaluation without LSTM
            decision = self.ensemble.evaluate_simple(
                features_50=features_50,
                capital=capital,
            )

        return decision

    def _ensemble_to_signal(
        self, decision: EnsembleDecision, mint: str, timestamp_ms: int
    ) -> Signal | None:
        """Convert ensemble decision to trading signal."""
        if decision.signal == EnsembleSignal.EXIT:
            return Signal(
                type=SignalType.SELL,
                mint=mint,
                timestamp_ms=timestamp_ms,
                confidence=decision.confidence,
                reason="Rug risk EXIT",
            )
        elif decision.signal == EnsembleSignal.STRONG_SELL:
            return Signal(
                type=SignalType.SELL,
                mint=mint,
                timestamp_ms=timestamp_ms,
                confidence=decision.confidence,
                reason="Strong SELL signal",
            )
        elif decision.signal == EnsembleSignal.SELL:
            return Signal(
                type=SignalType.SELL,
                mint=mint,
                timestamp_ms=timestamp_ms,
                confidence=decision.confidence,
                reason="SELL signal",
            )
        elif decision.signal in [EnsembleSignal.BUY, EnsembleSignal.STRONG_BUY]:
            return Signal(
                type=SignalType.BUY,
                mint=mint,
                timestamp_ms=timestamp_ms,
                confidence=decision.confidence,
                target_amount_sol=decision.position_size,
                reason=(
                    f"{'Strong ' if decision.signal == EnsembleSignal.STRONG_BUY else ''}BUY signal"
                ),
            )

        return None  # HOLD

    async def _resolve_strategy(self, config: BacktestConfig) -> Strategy:
        """Select the signal generation path for this run."""
        if config.use_live_signal_generation:
            return LiveAutoTraderStrategy(
                coordinator_config=AutoTraderConfig(
                    paper_mode=True,
                    skip_learning=True,
                    max_position_sol=config.max_position_sol,
                )
            )
        return self.strategy

    def _submit_backtest_signal(
        self,
        signal: Signal,
        event: MarketEvent,
        config: BacktestConfig,
    ) -> bool:
        """Simulate the live HotPath batch submission boundary using JSON payloads."""
        expected_price = signal.expected_price
        if expected_price is None:
            expected_price = float(event.data.get("close", event.data.get("price", 0.0)) or 0.0)

        if expected_price <= 0:
            self._batch_stats.rejected_count += 1
            return False

        ml_context_payload = signal.ml_context
        if ml_context_payload is not None:
            try:
                serialized = json.dumps(ml_context_payload, sort_keys=True)
                restored = json.loads(serialized)
                round_trip = MLTradeContext.from_dict(restored)
                ml_context_payload = round_trip.to_dict() if round_trip is not None else None
            except Exception as exc:
                logger.warning("Backtest rejected malformed ML context: %s", exc)
                self._batch_stats.rejected_count += 1
                return False

        signal.expected_price = expected_price
        signal.ml_context = ml_context_payload
        signal.slippage_bps = signal.slippage_bps or config.slippage_bps
        self._batch_stats.accepted_count += 1
        return True

    async def run(
        self,
        start_timestamp: int,
        end_timestamp: int,
        data_source: str = "bitquery_ohlcv",
        config: BacktestConfig | None = None,
        validate_data: bool = False,
    ) -> dict[str, Any]:
        """Run a backtest.

        Args:
            start_timestamp: Start time in milliseconds.
            end_timestamp: End time in milliseconds.
            data_source: Data source to use (bitquery_ohlcv, bitquery_trades,
                dune_ohlcv, dune_trades, dune_mev, dune_liquidity, synthetic, sqlite_telemetry).
            config: Optional backtest configuration.
            validate_data: Whether to validate streamed data quality.

        Returns:
            Dictionary with backtest results.
        """
        if config is None:
            config = BacktestConfig(
                start_timestamp_ms=start_timestamp,
                end_timestamp_ms=end_timestamp,
                data_source=data_source,
            )

        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
            logger.info("Backtest seed set: %s", config.random_seed)

        self._batch_stats = BacktestBatchStats()
        strategy = await self._resolve_strategy(config)
        self._active_strategy = strategy
        if hasattr(strategy, "reset"):
            strategy.reset()

        logger.info(
            f"Starting backtest from {start_timestamp} to {end_timestamp} using {data_source}"
        )

        # Initialize replay clock
        clock = ReplayClock(start_timestamp, end_timestamp)

        # Track state
        trades: list[dict] = []
        capital_sol = config.initial_capital_sol
        positions: dict[str, Position] = {}
        equity_curve: list[dict] = []
        events_processed = 0

        provider, normalized_source = self._select_provider(data_source)
        try:
            return await self._run_backtest_loop(
                provider,
                normalized_source,
                config,
                strategy,
                clock,
                trades,
                capital_sol,
                positions,
                equity_curve,
                events_processed,
                validate_data,
                start_timestamp,
                end_timestamp,
                data_source,
            )
        finally:
            # Ensure provider is cleaned up even if backtest throws
            if hasattr(provider, "close"):
                try:
                    await provider.close()
                except Exception:
                    pass
            self._price_sequences.clear()

    async def _run_backtest_loop(
        self,
        provider,
        normalized_source,
        config,
        strategy,
        clock,
        trades,
        capital_sol,
        positions,
        equity_curve,
        events_processed,
        validate_data,
        start_timestamp,
        end_timestamp,
        data_source,
    ):
        """Inner backtest loop, extracted for try/finally cleanup wrapper."""
        scenario_engine = self._build_scenario_engine(config)
        advanced_orchestrator = self._build_advanced_orchestrator(config)
        orchestration_decisions: list[dict[str, Any]] = []
        data_quality_reports: list[dict[str, Any]] = []
        validation_buffer: list[dict[str, Any]] = []

        event_stream = provider.stream_events(start_timestamp, end_timestamp, normalized_source)

        if scenario_engine is not None:
            event_stream = self._scenario_compatible_stream(event_stream)
            event_stream = scenario_engine.wrap_stream(event_stream)

        # Stream market events
        async for event in event_stream:
            clock.advance_to(event.timestamp_ms)
            events_processed += 1

            if advanced_orchestrator is not None:
                advanced_orchestrator.process_event(event)

            if validate_data and event.event_type in {"ohlcv", "trade"}:
                validation_buffer.append(self._build_validation_record(event))
                if len(validation_buffer) >= self.validation_batch_size:
                    self._validate_records(validation_buffer, data_quality_reports)
                    validation_buffer = []

            # Update existing positions with new prices
            if event.event_type == "ohlcv" and event.mint in positions:
                positions[event.mint].update_price(event.data.get("close", 0))

            # Check stop loss / take profit for all positions
            for mint, pos in list(positions.items()):
                if pos.should_stop_loss():
                    trade = self._close_position(
                        pos, clock.current_timestamp_ms, "stop_loss", config
                    )
                    if trade:
                        trades.append(trade)
                        if trade.get("included", False):
                            capital_sol += trade.get("pnl_sol", 0.0)
                        del positions[mint]
                        self._active_strategy.on_position_closed(pos, trade.get("pnl_sol", 0.0))

                elif pos.should_take_profit():
                    trade = self._close_position(
                        pos, clock.current_timestamp_ms, "take_profit", config
                    )
                    if trade:
                        trades.append(trade)
                        if trade.get("included", False):
                            capital_sol += trade.get("pnl_sol", 0.0)
                        del positions[mint]
                        self._active_strategy.on_position_closed(pos, trade.get("pnl_sol", 0.0))

            # Get signal from strategy or ensemble
            signal = None
            ensemble_decision = None

            if config.use_live_signal_generation:
                signal_or_coro = strategy.on_event(event, positions, capital_sol)
                signal = (
                    await signal_or_coro if inspect.isawaitable(signal_or_coro) else signal_or_coro
                )
            elif config.use_ensemble and self.ensemble is not None:
                # Use ensemble for signal generation
                ensemble_decision = self._get_ensemble_decision(event, capital_sol, config)
                if ensemble_decision is not None:
                    signal = self._ensemble_to_signal(
                        ensemble_decision, event.mint, event.timestamp_ms
                    )
            else:
                # Use traditional strategy
                signal_or_coro = strategy.on_event(event, positions, capital_sol)
                signal = (
                    await signal_or_coro if inspect.isawaitable(signal_or_coro) else signal_or_coro
                )

            if signal:
                if not self._submit_backtest_signal(signal, event, config):
                    continue
                orchestration_decision = None
                if advanced_orchestrator is not None:
                    signal, decision = advanced_orchestrator.adjust_signal(signal, event)
                    orchestration_decision = decision.to_dict()

                trade = self._process_signal(
                    signal,
                    positions,
                    capital_sol,
                    config,
                    event,
                    ensemble_decision=ensemble_decision,
                )
                if trade:
                    if orchestration_decision is not None:
                        trade["orchestration"] = orchestration_decision
                        orchestration_decisions.append(orchestration_decision)
                    trades.append(trade)
                    if trade.get("included", False):
                        if signal.type == SignalType.BUY:
                            capital_sol -= trade.get("amount_sol", 0.0)
                        else:
                            capital_sol += trade.get("amount_sol", 0.0) + trade.get("pnl_sol", 0.0)

                    # Record trade result for Kelly adaptation
                    if self.kelly is not None and trade.get("pnl_pct") is not None:
                        self.kelly.record_trade_result(trade["pnl_pct"])

            # Record equity curve (every 100 events)
            if events_processed % 100 == 0:
                total_equity = capital_sol + sum(
                    p.amount_sol + p.unrealized_pnl_sol for p in positions.values()
                )
                equity_curve.append(
                    {
                        "timestamp_ms": clock.current_timestamp_ms,
                        "equity_sol": total_equity,
                        "cash_sol": capital_sol,
                        "positions_value": total_equity - capital_sol,
                        "num_positions": len(positions),
                    }
                )

        # Close remaining positions at end
        for _mint, pos in list(positions.items()):
            trade = self._close_position(pos, end_timestamp, "backtest_end", config)
            if trade:
                trades.append(trade)
                if trade.get("included", False):
                    capital_sol += trade.get("pnl_sol", 0.0)

        # Record final equity point
        final_equity = capital_sol + sum(
            p.amount_sol + p.unrealized_pnl_sol for p in positions.values()
        )
        if not equity_curve or equity_curve[-1]["timestamp_ms"] != end_timestamp:
            equity_curve.append(
                {
                    "timestamp_ms": end_timestamp,
                    "equity_sol": final_equity,
                    "cash_sol": capital_sol,
                    "positions_value": final_equity - capital_sol,
                    "num_positions": len(positions),
                }
            )

        # Calculate metrics
        metrics = self.metrics_engine.calculate(trades, config.initial_capital_sol)
        trajectory_risk = self._compute_trajectory_risk(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=config.initial_capital_sol,
            config=config,
        )

        logger.info(
            f"Backtest complete: {events_processed} events, {len(trades)} trades, "
            f"return: {metrics.total_return_pct:.1f}%, "
            f"sharpe: {metrics.sharpe_ratio:.2f}"
        )
        logger.info(
            "Batch backtest: %s accepted, %s rejected",
            self._batch_stats.accepted_count,
            self._batch_stats.rejected_count,
        )

        if validate_data and validation_buffer:
            self._validate_records(validation_buffer, data_quality_reports)

        result = {
            "config": config.to_dict(),
            "trades": trades,
            "metrics": metrics.to_dict(),
            "batch_stats": self._batch_stats.to_dict(),
            "equity_curve": equity_curve,
            "trajectory_risk": trajectory_risk,
            "summary": {
                "events_processed": events_processed,
                "final_capital": capital_sol,
                "total_return_pct": metrics.total_return_pct,
            },
        }

        if validate_data:
            result["data_quality"] = {
                "reports": data_quality_reports,
                "overall_passed": all(r.get("passed") for r in data_quality_reports),
            }

        if scenario_engine is not None:
            result["adversarial_evaluation"] = {
                "enabled": True,
                "scenario_summary": self._serialize_scenario_stats(scenario_engine),
                "trajectory_risk": trajectory_risk,
            }
        else:
            result["adversarial_evaluation"] = {"enabled": False}

        if advanced_orchestrator is not None:
            result["orchestration"] = {
                "enabled": True,
                "summary": advanced_orchestrator.summary(),
                "decisions": orchestration_decisions,
            }
        else:
            result["orchestration"] = {"enabled": False}

        return result

    def _build_advanced_orchestrator(
        self,
        config: BacktestConfig,
    ) -> AdvancedStrategyOrchestrator | None:
        """Create unified orchestration layer for advanced models."""
        if not config.enable_advanced_orchestration:
            return None

        symbols = tuple(
            s.strip().upper()
            for s in config.orchestration_stablecoin_symbols_csv.split(",")
            if s.strip()
        )
        orchestrator_config = AdvancedOrchestratorConfig(
            enable_spatial=config.enable_sapt_spatial_model,
            enable_semantic=config.enable_semantic_fragmentation_model,
            enable_mean_field=config.enable_mean_field_stablecoin_model,
            min_samples_for_spatial=config.orchestration_min_samples,
            spatial_refit_interval=config.orchestration_refit_interval,
            max_assets_for_spatial=config.orchestration_max_assets,
            max_confidence_multiplier=config.orchestration_max_confidence_multiplier,
            min_confidence_multiplier=config.orchestration_min_confidence_multiplier,
            stressed_regime_multiplier=config.orchestration_stressed_regime_multiplier,
            parity_boost_scale=config.orchestration_parity_boost_scale,
            stablecoin_symbols=symbols or AdvancedOrchestratorConfig().stablecoin_symbols,
        )
        return AdvancedStrategyOrchestrator(config=orchestrator_config)

    def _build_scenario_engine(self, config: BacktestConfig) -> ScenarioEngine | None:
        """Create the adversarial scenario engine when configured."""
        if not config.enable_adversarial_scenarios:
            return None

        engine = create_default_scenario_engine(
            honeypot_probability=config.honeypot_probability,
            sandwich_probability=config.sandwich_probability,
            rug_probability=config.rug_probability,
            seed=config.adversarial_seed,
        )

        # Optional scenarios beyond default CAIA set.
        if config.flash_crash_probability > 0:
            from .synthetic_scenarios import FlashCrashScenario, ScenarioConfig

            engine.add_scenario(
                FlashCrashScenario(),
                ScenarioConfig(probability=config.flash_crash_probability),
            )
        if config.whale_dump_probability > 0:
            from .synthetic_scenarios import ScenarioConfig, WhaleDumpScenario

            engine.add_scenario(
                WhaleDumpScenario(),
                ScenarioConfig(probability=config.whale_dump_probability),
            )

        return engine

    async def _scenario_compatible_stream(
        self,
        event_stream: AsyncIterator[MarketEvent],
    ) -> AsyncIterator[ScenarioMarketEvent]:
        """Convert provider events into scenario-compatible events with copy()."""
        async for event in event_stream:
            if isinstance(event, ScenarioMarketEvent):
                yield event
                continue

            yield ScenarioMarketEvent(
                event_type=event.event_type,
                mint=event.mint,
                pool=getattr(event, "pool", ""),
                timestamp_ms=event.timestamp_ms,
                data=dict(event.data or {}),
            )

    def _serialize_scenario_stats(self, scenario_engine: ScenarioEngine) -> dict[str, Any]:
        """Serialize scenario injection stats with aggregate counters."""
        stats = scenario_engine.get_stats()
        by_scenario: dict[str, Any] = {}
        total_triggers = 0
        total_modified = 0
        for scenario_name, scenario_stats in stats.items():
            by_scenario[scenario_name] = {
                "triggers": scenario_stats.triggers,
                "events_modified": scenario_stats.events_modified,
                "tokens_affected": len(scenario_stats.tokens_affected),
            }
            total_triggers += scenario_stats.triggers
            total_modified += scenario_stats.events_modified

        return {
            "total_triggers": total_triggers,
            "total_events_modified": total_modified,
            "by_scenario": by_scenario,
        }

    def _compute_trajectory_risk(
        self,
        trades: list[dict[str, Any]],
        equity_curve: list[dict[str, Any]],
        initial_capital: float,
        config: BacktestConfig,
    ) -> dict[str, Any]:
        """Compute CAIA-style trajectory-level risk metrics."""
        if initial_capital <= 0:
            return {
                "catastrophic_trade_count": 0,
                "catastrophic_trade_rate": 0.0,
                "worst_trade_loss_sol": 0.0,
                "worst_trade_loss_pct_of_capital": 0.0,
                "worst_episode_loss_pct": 0.0,
                "catastrophic_drawdown_event": False,
            }

        sell_trades = [t for t in trades if t.get("type") == "sell" and t.get("included", True)]
        losses = [float(t.get("pnl_sol", 0.0)) for t in sell_trades]
        worst_trade_loss_sol = min(losses) if losses else 0.0
        worst_trade_loss_pct = abs(worst_trade_loss_sol) / initial_capital * 100

        catastrophic_loss_threshold_sol = initial_capital * (
            config.catastrophic_loss_trade_pct / 100.0
        )
        catastrophic_trade_count = sum(
            1 for loss in losses if loss <= -catastrophic_loss_threshold_sol
        )
        catastrophic_trade_rate = (
            catastrophic_trade_count / len(sell_trades) * 100 if sell_trades else 0.0
        )

        peak_equity = initial_capital
        worst_episode_loss_pct = 0.0
        for point in equity_curve:
            equity_value = float(point.get("equity_sol", initial_capital))
            if equity_value > peak_equity:
                peak_equity = equity_value
            if peak_equity > 0:
                dd_pct = (peak_equity - equity_value) / peak_equity * 100
                worst_episode_loss_pct = max(worst_episode_loss_pct, dd_pct)

        return {
            "catastrophic_trade_count": catastrophic_trade_count,
            "catastrophic_trade_rate": catastrophic_trade_rate,
            "worst_trade_loss_sol": worst_trade_loss_sol,
            "worst_trade_loss_pct_of_capital": worst_trade_loss_pct,
            "worst_episode_loss_pct": worst_episode_loss_pct,
            "catastrophic_drawdown_event": (
                worst_episode_loss_pct >= config.catastrophic_drawdown_pct
            ),
        }

    def _select_provider(self, data_source: str) -> tuple[Any, str]:
        bitquery_sources = {"bitquery_ohlcv", "bitquery_trades", "synthetic"}
        dune_sources = {
            "dune_ohlcv",
            "dune_trades",
            "dune_mev",
            "dune_liquidity",
            "dune_dex_trades",
            "dune_launches",
        }

        if data_source in bitquery_sources:
            return self.bitquery_provider, data_source

        if data_source in dune_sources:
            if data_source == "dune_dex_trades":
                normalized_source = "dune_ohlcv"
            else:
                normalized_source = data_source

            if not self.dune_provider:
                self.dune_provider = DuneDataProvider()

            return self.dune_provider, normalized_source

        if data_source == "sqlite_telemetry":
            return self.local_telemetry_provider, data_source

        raise ValueError(f"Unknown data source: {data_source}")

    def _build_validation_record(self, event: MarketEvent) -> dict[str, Any]:
        data = event.data
        amount = self._extract_amount(data)
        price = self._extract_price(data)
        return {
            "timestamp_ms": event.timestamp_ms,
            "mint": event.mint,
            "side": data.get("side") or "buy",
            "amount_sol": amount,
            "price": price,
        }

    def _extract_amount(self, data: dict[str, Any]) -> float:
        for key in ("amount_sol", "volume_sol", "amount_tokens", "volume"):
            value = data.get(key)
            if value is not None:
                return float(value)
        return 0.0

    def _extract_price(self, data: dict[str, Any]) -> float:
        for key in ("price", "close"):
            value = data.get(key)
            if value is not None:
                return float(value)
        return 0.0

    def _validate_records(
        self,
        records: list[dict[str, Any]],
        reports: list[dict[str, Any]],
    ) -> None:
        if not records:
            return

        import pandas as pd

        df = pd.DataFrame(records)
        report = self.data_validator.validate_dataframe(df)
        reports.append(report.to_dict())
        if not report.passed:
            logger.warning(report.summary())

    def _process_signal(
        self,
        signal: Signal,
        positions: dict[str, Position],
        capital: float,
        config: BacktestConfig,
        event: MarketEvent,
        ensemble_decision: EnsembleDecision | None = None,
    ) -> dict | None:
        """Process a trading signal."""
        if signal.type == SignalType.BUY:
            trade = self._open_position(signal, capital, config, event, ensemble_decision)
            # CRITICAL FIX: Add position to positions dict after successful buy
            if trade and trade.get("included", False):
                price = trade.get("price", 0)
                position = Position(
                    mint=signal.mint,
                    entry_price=price,
                    amount_tokens=trade.get("amount_tokens", 0),
                    amount_sol=trade.get("amount_sol", 0),
                    entry_timestamp_ms=signal.timestamp_ms,
                    current_price=price,
                    high_price=price,
                    stop_loss_pct=config.default_stop_loss_pct,
                    take_profit_pct=config.default_take_profit_pct,
                )
                positions[signal.mint] = position
                if hasattr(self._active_strategy, "on_position_opened"):
                    self._active_strategy.on_position_opened(position)
            return trade
        elif signal.type == SignalType.SELL:
            if signal.mint in positions:
                pos = positions.pop(signal.mint)
                return self._close_position(pos, signal.timestamp_ms, "signal", config)
        return None

    def _open_position(
        self,
        signal: Signal,
        capital: float,
        config: BacktestConfig,
        event: MarketEvent,
        ensemble_decision: EnsembleDecision | None = None,
    ) -> dict | None:
        """Open a new position with optional Kelly sizing."""
        # Calculate position size
        if config.use_kelly_sizing and ensemble_decision is not None:
            # Use ensemble-calculated position size
            amount_sol = ensemble_decision.position_size
        elif config.use_kelly_sizing and self.kelly is not None:
            # Use Kelly calculator directly
            kelly_result = self.kelly.calculate_from_signal(
                signal_confidence=signal.confidence,
                signal_expected_return=10.0,  # Assume 10% expected return
                capital=capital,
            )
            amount_sol = kelly_result.position_size
        elif signal.target_amount_sol is not None:
            # Use signal's specified amount
            amount_sol = signal.target_amount_sol
        else:
            # Default: 10% of capital scaled by confidence
            amount_sol = capital * 0.1 * signal.confidence

        # Apply maximum position limit
        amount_sol = min(amount_sol, config.max_position_sol)

        if amount_sol < 0.001:  # Minimum trade size
            return None

        # Get market data
        price = event.data.get("close", 0)
        liquidity_sol = event.data.get("liquidity_sol", 10000)
        liquidity_tokens = event.data.get("liquidity_tokens", 10000000)

        if price == 0:
            return None

        # Simulate fill
        quoted_slippage_bps = signal.slippage_bps or config.slippage_bps
        fill = self.fill_simulator.simulate_buy(
            amount_in_sol=amount_sol,
            liquidity_sol=liquidity_sol,
            liquidity_tokens=liquidity_tokens,
            quoted_slippage_bps=quoted_slippage_bps,
        )

        if not fill.filled:
            return {
                "type": "buy",
                "mint": signal.mint,
                "signal_id": signal.signal_id,
                "timestamp_ms": signal.timestamp_ms,
                "amount_sol": amount_sol,
                "included": False,
                "failure_reason": fill.failure_reason,
                "ml_context": signal.ml_context,
                "quoted_slippage_bps": quoted_slippage_bps,
            }

        # Create trade record
        return {
            "type": "buy",
            "mint": signal.mint,
            "signal_id": signal.signal_id,
            "timestamp_ms": signal.timestamp_ms,
            "amount_sol": amount_sol,
            "amount_tokens": fill.amount_out,
            "price": fill.fill_price,
            "expected_price": signal.expected_price,
            "slippage_bps": fill.slippage_bps,
            "quoted_slippage_bps": quoted_slippage_bps,
            "included": True,
            "reason": signal.reason,
            "ml_context": signal.ml_context,
        }

    def _close_position(
        self,
        position: Position,
        timestamp_ms: int,
        reason: str,
        config: BacktestConfig,
    ) -> dict | None:
        """Close an existing position."""
        # Use latest observed price so PnL reflects market movement.
        reference_price = (
            position.current_price if position.current_price > 0 else position.entry_price
        )
        liquidity_tokens = max(position.amount_tokens * 100.0, 1.0)
        liquidity_sol = max(liquidity_tokens * reference_price, 1e-9)

        # Simulate sell
        fill = self.fill_simulator.simulate_sell(
            amount_tokens=position.amount_tokens,
            liquidity_sol=liquidity_sol,
            liquidity_tokens=liquidity_tokens,
            quoted_slippage_bps=config.slippage_bps,
        )

        if not fill.filled:
            return {
                "type": "sell",
                "mint": position.mint,
                "timestamp_ms": timestamp_ms,
                "amount_tokens": position.amount_tokens,
                "included": False,
                "failure_reason": fill.failure_reason,
            }

        # Calculate PnL
        exit_value = fill.amount_out or 0
        pnl_sol = exit_value - position.amount_sol
        pnl_pct = (pnl_sol / position.amount_sol * 100) if position.amount_sol > 0 else 0

        return {
            "type": "sell",
            "mint": position.mint,
            "timestamp_ms": timestamp_ms,
            "amount_tokens": position.amount_tokens,
            "amount_sol": exit_value,
            "entry_price": position.entry_price,
            "exit_price": fill.fill_price,
            "slippage_bps": fill.slippage_bps,
            "pnl_sol": pnl_sol,
            "pnl_pct": pnl_pct,
            "included": True,
            "reason": reason,
            "hold_time_ms": timestamp_ms - position.entry_timestamp_ms,
        }
