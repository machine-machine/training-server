"""
Synthetic Scenario Injection for Backtesting.

Injects synthetic adverse scenarios into market event streams to test
strategy robustness against:
- Delayed honeypots (token becomes untradeable after initial window)
- MEV sandwich attacks
- LP rug pulls
- Flash crashes
- Liquidity drain

This enables testing edge cases that may be rare in historical data.
"""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of synthetic scenarios."""

    DELAYED_HONEYPOT = "delayed_honeypot"
    GRADUATED_HONEYPOT = "graduated_honeypot"
    SANDWICH_ATTACK = "sandwich_attack"
    LP_RUG = "lp_rug"
    FLASH_CRASH = "flash_crash"
    LIQUIDITY_DRAIN = "liquidity_drain"
    WHALE_DUMP = "whale_dump"
    COORDINATED_DUMP = "coordinated_dump"


@dataclass
class MarketEvent:
    """A single market event in the backtest stream."""

    event_type: str  # "trade", "pool_update", "price_update", etc.
    mint: str
    pool: str
    timestamp_ms: int
    data: dict[str, Any]

    def copy(self) -> "MarketEvent":
        """Create a copy of this event."""
        return MarketEvent(
            event_type=self.event_type,
            mint=self.mint,
            pool=self.pool,
            timestamp_ms=self.timestamp_ms,
            data=self.data.copy(),
        )


@dataclass
class ScenarioConfig:
    """Configuration for a scenario injector."""

    probability: float  # Probability of injection per qualifying event
    enabled: bool = True
    min_events_before_trigger: int = 0  # Minimum events before scenario can trigger
    max_triggers_per_token: int = 1  # Maximum times to trigger per token


@dataclass
class ScenarioStats:
    """Statistics for scenario injection."""

    scenario_type: ScenarioType
    triggers: int = 0
    events_modified: int = 0
    tokens_affected: set[str] = field(default_factory=set)


class ScenarioInjector(ABC):
    """Base class for scenario injection.

    Subclasses implement specific adverse scenarios that modify
    market events to simulate attacks, rugs, etc.
    """

    def __init__(self, config: ScenarioConfig | None = None):
        self.config = config or ScenarioConfig(probability=0.1)
        self.stats = ScenarioStats(scenario_type=self.scenario_type)
        self._trigger_count: dict[str, int] = {}

    @property
    @abstractmethod
    def scenario_type(self) -> ScenarioType:
        """Return the type of scenario this injector handles."""
        pass

    @abstractmethod
    def should_trigger(self, event: MarketEvent) -> bool:
        """Determine if this scenario should trigger on the given event."""
        pass

    @abstractmethod
    def modify_event(self, event: MarketEvent) -> MarketEvent:
        """Modify the event to inject the scenario."""
        pass

    def can_trigger(self, event: MarketEvent) -> bool:
        """Check if we can still trigger for this token."""
        token_triggers = self._trigger_count.get(event.mint, 0)
        return token_triggers < self.config.max_triggers_per_token

    def record_trigger(self, event: MarketEvent) -> None:
        """Record that we triggered for this token."""
        self._trigger_count[event.mint] = self._trigger_count.get(event.mint, 0) + 1
        self.stats.triggers += 1
        self.stats.tokens_affected.add(event.mint)

    def reset(self) -> None:
        """Reset state for new backtest run."""
        self._trigger_count.clear()
        self.stats = ScenarioStats(scenario_type=self.scenario_type)


class DelayedHoneypotScenario(ScenarioInjector):
    """Simulate a token that becomes a honeypot after initial open trading.

    Pattern:
    1. Token trades normally for `open_window_minutes`
    2. After window expires, sells become blocked with high probability
    """

    def __init__(
        self,
        open_window_minutes: int = 10,
        restriction_probability: float = 0.99,
        config: ScenarioConfig | None = None,
    ):
        super().__init__(config)
        self.open_window_minutes = open_window_minutes
        self.restriction_probability = restriction_probability
        self._token_launch_times: dict[str, int] = {}
        self._triggered_tokens: set[str] = set()

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.DELAYED_HONEYPOT

    def should_trigger(self, event: MarketEvent) -> bool:
        mint = event.mint

        if mint in self._triggered_tokens:
            return True  # Already triggered, keep blocking

        # Track launch time
        if mint not in self._token_launch_times:
            self._token_launch_times[mint] = event.timestamp_ms
            return False

        # Check if window expired
        elapsed_ms = event.timestamp_ms - self._token_launch_times[mint]
        window_ms = self.open_window_minutes * 60 * 1000

        if elapsed_ms > window_ms:
            if random.random() < self.restriction_probability:
                self._triggered_tokens.add(mint)
                self.record_trigger(event)
                return True

        return False

    def modify_event(self, event: MarketEvent) -> MarketEvent:
        modified = event.copy()
        modified.data["sell_blocked"] = True
        modified.data["price_impact_bps"] = 9500  # 95% impact
        modified.data["unique_sellers_5min"] = 0
        modified.data["honeypot_type"] = "delayed"
        modified.data["scenario_injected"] = self.scenario_type.value
        self.stats.events_modified += 1
        return modified


class GraduatedHoneypotScenario(ScenarioInjector):
    """Simulate graduated restriction - small sells work, large sells blocked."""

    def __init__(
        self,
        small_threshold_pct: float = 1.0,  # 1% of position
        small_tax_bps: int = 200,  # 2% tax for small
        large_tax_bps: int = 9000,  # 90% tax for large
        config: ScenarioConfig | None = None,
    ):
        super().__init__(config)
        self.small_threshold_pct = small_threshold_pct
        self.small_tax_bps = small_tax_bps
        self.large_tax_bps = large_tax_bps
        self._affected_tokens: set[str] = set()

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.GRADUATED_HONEYPOT

    def should_trigger(self, event: MarketEvent) -> bool:
        # Trigger on sell events
        if event.event_type != "trade":
            return False
        if event.data.get("is_buy", True):
            return False
        if event.mint in self._affected_tokens:
            return True
        if not self.can_trigger(event):
            return False
        if random.random() < self.config.probability:
            self._affected_tokens.add(event.mint)
            self.record_trigger(event)
            return True
        return False

    def modify_event(self, event: MarketEvent) -> MarketEvent:
        modified = event.copy()

        # Check sell size
        sell_pct = modified.data.get("sell_pct_of_position", 5.0)

        if sell_pct <= self.small_threshold_pct:
            modified.data["tax_bps"] = self.small_tax_bps
        else:
            modified.data["tax_bps"] = self.large_tax_bps
            modified.data["price_impact_bps"] = self.large_tax_bps

        modified.data["graduated_restriction"] = True
        modified.data["scenario_injected"] = self.scenario_type.value
        self.stats.events_modified += 1
        return modified


class SandwichAttackScenario(ScenarioInjector):
    """Simulate MEV sandwich attacks on trades."""

    def __init__(
        self,
        probability: float = 0.15,
        slippage_penalty_bps: int = 200,
        frontrun_size_multiplier: float = 2.0,
        config: ScenarioConfig | None = None,
    ):
        if config is None:
            config = ScenarioConfig(probability=probability)
        super().__init__(config)
        self.slippage_penalty_bps = slippage_penalty_bps
        self.frontrun_size_multiplier = frontrun_size_multiplier

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.SANDWICH_ATTACK

    def should_trigger(self, event: MarketEvent) -> bool:
        if event.event_type != "trade":
            return False
        if not self.can_trigger(event):
            return False
        # Larger trades more likely to be sandwiched
        trade_size = event.data.get("amount_sol", 0.1)
        size_factor = min(2.0, trade_size / 0.5)  # Larger = more likely
        if random.random() < self.config.probability * size_factor:
            self.record_trigger(event)
            return True
        return False

    def modify_event(self, event: MarketEvent) -> MarketEvent:
        modified = event.copy()
        current_slippage = modified.data.get("slippage_bps", 0)
        modified.data["slippage_bps"] = current_slippage + self.slippage_penalty_bps
        modified.data["mev_sandwiched"] = True
        modified.data["frontrun_size"] = (
            modified.data.get("amount_sol", 0.1) * self.frontrun_size_multiplier
        )
        modified.data["scenario_injected"] = self.scenario_type.value
        self.stats.events_modified += 1
        return modified


class LPRugScenario(ScenarioInjector):
    """Simulate LP removal (rug pull)."""

    def __init__(
        self,
        accumulation_minutes: int = 30,
        removal_pct: float = 0.95,
        price_crash_pct: float = 0.95,
        config: ScenarioConfig | None = None,
    ):
        super().__init__(config)
        self.accumulation_minutes = accumulation_minutes
        self.removal_pct = removal_pct
        self.price_crash_pct = price_crash_pct
        self._token_launch_times: dict[str, int] = {}
        self._triggered_tokens: set[str] = set()

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.LP_RUG

    def should_trigger(self, event: MarketEvent) -> bool:
        mint = event.mint

        if mint in self._triggered_tokens:
            return True  # Already triggered

        if mint not in self._token_launch_times:
            self._token_launch_times[mint] = event.timestamp_ms
            return False

        if not self.can_trigger(event):
            return False

        elapsed_ms = event.timestamp_ms - self._token_launch_times[mint]
        window_ms = self.accumulation_minutes * 60 * 1000

        if elapsed_ms > window_ms:
            if random.random() < self.config.probability:
                self._triggered_tokens.add(mint)
                self.record_trigger(event)
                return True

        return False

    def modify_event(self, event: MarketEvent) -> MarketEvent:
        modified = event.copy()
        current_liquidity = modified.data.get("liquidity_usd", 10000)
        modified.data["liquidity_usd"] = current_liquidity * (1 - self.removal_pct)
        modified.data["lp_removed"] = True
        modified.data["lp_removal_pct"] = self.removal_pct * 100

        # Price crash
        current_price = modified.data.get("price", 1.0)
        modified.data["price"] = current_price * (1 - self.price_crash_pct)
        modified.data["price_change_1h"] = -self.price_crash_pct * 100

        modified.data["scenario_injected"] = self.scenario_type.value
        self.stats.events_modified += 1
        return modified


class FlashCrashScenario(ScenarioInjector):
    """Simulate sudden flash crash and recovery."""

    def __init__(
        self,
        crash_depth_pct: float = 0.50,
        recovery_time_minutes: int = 5,
        config: ScenarioConfig | None = None,
    ):
        super().__init__(config)
        self.crash_depth_pct = crash_depth_pct
        self.recovery_time_minutes = recovery_time_minutes
        self._crash_events: dict[str, tuple[int, float]] = {}
        # mint -> (crash_time, pre_crash_price)

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.FLASH_CRASH

    def should_trigger(self, event: MarketEvent) -> bool:
        mint = event.mint

        # Check if in recovery phase
        if mint in self._crash_events:
            crash_time, _ = self._crash_events[mint]
            elapsed_ms = event.timestamp_ms - crash_time
            if elapsed_ms < self.recovery_time_minutes * 60 * 1000:
                return True
            else:
                del self._crash_events[mint]
                return False

        if not self.can_trigger(event):
            return False

        if random.random() < self.config.probability:
            pre_price = event.data.get("price", 1.0)
            self._crash_events[mint] = (event.timestamp_ms, pre_price)
            self.record_trigger(event)
            return True

        return False

    def modify_event(self, event: MarketEvent) -> MarketEvent:
        modified = event.copy()
        mint = event.mint

        if mint in self._crash_events:
            crash_time, pre_price = self._crash_events[mint]
            elapsed_ms = event.timestamp_ms - crash_time
            recovery_ms = self.recovery_time_minutes * 60 * 1000

            # Calculate recovery progress (0 = just crashed, 1 = fully recovered)
            recovery_progress = min(1.0, elapsed_ms / recovery_ms)

            # Price follows V-shape
            if recovery_progress < 0.5:
                # Still crashing
                crash_progress = recovery_progress * 2  # 0 to 1 during first half
                price_factor = 1.0 - (self.crash_depth_pct * crash_progress)
            else:
                # Recovering
                recover_progress = (recovery_progress - 0.5) * 2  # 0 to 1 during second half
                min_price_factor = 1.0 - self.crash_depth_pct
                price_factor = min_price_factor + (self.crash_depth_pct * recover_progress)

            modified.data["price"] = pre_price * price_factor
            modified.data["flash_crash"] = True
            modified.data["recovery_progress"] = recovery_progress

        modified.data["scenario_injected"] = self.scenario_type.value
        self.stats.events_modified += 1
        return modified


class WhaleDumpScenario(ScenarioInjector):
    """Simulate large whale dumping position."""

    def __init__(
        self,
        dump_pct_of_supply: float = 10.0,
        price_impact_per_pct: int = 200,  # bps per 1% of supply
        config: ScenarioConfig | None = None,
    ):
        super().__init__(config)
        self.dump_pct_of_supply = dump_pct_of_supply
        self.price_impact_per_pct = price_impact_per_pct

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.WHALE_DUMP

    def should_trigger(self, event: MarketEvent) -> bool:
        if event.event_type != "trade":
            return False
        if not self.can_trigger(event):
            return False
        if random.random() < self.config.probability:
            self.record_trigger(event)
            return True
        return False

    def modify_event(self, event: MarketEvent) -> MarketEvent:
        modified = event.copy()

        # Calculate price impact
        impact_bps = int(self.dump_pct_of_supply * self.price_impact_per_pct)

        current_price = modified.data.get("price", 1.0)
        modified.data["price"] = current_price * (1 - impact_bps / 10000)
        modified.data["whale_dump"] = True
        modified.data["dump_pct"] = self.dump_pct_of_supply
        modified.data["price_impact_bps"] = impact_bps
        modified.data["large_sell_count"] = modified.data.get("large_sell_count", 0) + 1

        modified.data["scenario_injected"] = self.scenario_type.value
        self.stats.events_modified += 1
        return modified


class ScenarioEngine:
    """Manages multiple scenario injectors.

    Wraps event streams to inject synthetic scenarios based on
    configured probabilities and rules.

    Example:
        engine = ScenarioEngine()
        engine.add_scenario(
            DelayedHoneypotScenario(open_window_minutes=10),
            ScenarioConfig(probability=0.10),
        )
        engine.add_scenario(
            SandwichAttackScenario(),
            ScenarioConfig(probability=0.15),
        )

        async for event in engine.wrap_stream(data_stream):
            process(event)
    """

    def __init__(self, seed: int | None = None):
        self.scenarios: list[tuple[ScenarioInjector, ScenarioConfig]] = []
        self._seed = seed
        if seed is not None:
            random.seed(seed)

    def add_scenario(
        self,
        injector: ScenarioInjector,
        config: ScenarioConfig | None = None,
    ) -> None:
        """Add a scenario injector."""
        if config is not None:
            injector.config = config
        self.scenarios.append((injector, injector.config))
        logger.info(f"Added scenario: {injector.scenario_type.value}")

    def remove_scenario(self, scenario_type: ScenarioType) -> None:
        """Remove scenarios of a given type."""
        self.scenarios = [
            (inj, cfg) for inj, cfg in self.scenarios if inj.scenario_type != scenario_type
        ]

    def reset(self) -> None:
        """Reset all scenario states."""
        for injector, _ in self.scenarios:
            injector.reset()
        if self._seed is not None:
            random.seed(self._seed)

    async def wrap_stream(
        self,
        event_stream: AsyncIterator[MarketEvent],
    ) -> AsyncIterator[MarketEvent]:
        """Wrap event stream with scenario injection.

        Args:
            event_stream: Original market event stream

        Yields:
            Events with scenarios injected based on configuration
        """
        async for event in event_stream:
            modified_event = event

            for injector, config in self.scenarios:
                if not config.enabled:
                    continue

                if random.random() < config.probability:
                    if injector.should_trigger(modified_event):
                        modified_event = injector.modify_event(modified_event)

            yield modified_event

    def process_event(self, event: MarketEvent) -> MarketEvent:
        """Synchronously process a single event.

        Useful for non-async backtests.
        """
        modified_event = event

        for injector, config in self.scenarios:
            if not config.enabled:
                continue

            if random.random() < config.probability:
                if injector.should_trigger(modified_event):
                    modified_event = injector.modify_event(modified_event)

        return modified_event

    def get_stats(self) -> dict[str, ScenarioStats]:
        """Get statistics for all scenarios."""
        return {injector.scenario_type.value: injector.stats for injector, _ in self.scenarios}

    def get_summary(self) -> str:
        """Get human-readable summary of injection stats."""
        lines = ["Scenario Injection Summary:"]
        total_triggers = 0
        total_modified = 0

        for injector, _config in self.scenarios:
            stats = injector.stats
            total_triggers += stats.triggers
            total_modified += stats.events_modified
            lines.append(
                f"  {stats.scenario_type.value}: "
                f"{stats.triggers} triggers, "
                f"{stats.events_modified} events modified, "
                f"{len(stats.tokens_affected)} tokens affected"
            )

        lines.append(f"  Total: {total_triggers} triggers, {total_modified} events modified")
        return "\n".join(lines)


def create_default_scenario_engine(
    honeypot_probability: float = 0.10,
    sandwich_probability: float = 0.15,
    rug_probability: float = 0.05,
    seed: int | None = None,
) -> ScenarioEngine:
    """Create a scenario engine with default scenarios.

    Args:
        honeypot_probability: Probability of delayed honeypot
        sandwich_probability: Probability of sandwich attack
        rug_probability: Probability of LP rug
        seed: Random seed for reproducibility

    Returns:
        Configured ScenarioEngine
    """
    engine = ScenarioEngine(seed=seed)

    engine.add_scenario(
        DelayedHoneypotScenario(open_window_minutes=10),
        ScenarioConfig(probability=honeypot_probability),
    )

    engine.add_scenario(
        SandwichAttackScenario(slippage_penalty_bps=200),
        ScenarioConfig(probability=sandwich_probability),
    )

    engine.add_scenario(
        LPRugScenario(accumulation_minutes=30),
        ScenarioConfig(probability=rug_probability),
    )

    return engine
