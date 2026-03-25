"""
Solana Slot Simulator - Models transaction inclusion and failure dynamics.

Based on Helius MEV Report and empirical Solana network data:
- Base failure rate: ~40% (varies with congestion)
- Priority fee impact: Higher fee → higher inclusion probability
- Congestion modeling: Network state affects success rate
- Slippage failure: Quote staleness × trade size
- Slot timing: ~400ms average, variable by leader

Integration: Backtest engine uses for realistic fill simulation.
"""

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CongestionLevel(Enum):
    """Network congestion levels."""

    LOW = "low"  # <30% capacity, ~20% failure
    NORMAL = "normal"  # 30-60% capacity, ~40% failure
    HIGH = "high"  # 60-80% capacity, ~60% failure
    EXTREME = "extreme"  # >80% capacity, ~80% failure


class FailureReason(Enum):
    """Reasons for transaction failure."""

    SLOT_MISSED = "slot_missed"  # Didn't land in target slot
    INSUFFICIENT_FEE = "insufficient_fee"  # Outbid by others
    SLIPPAGE_EXCEEDED = "slippage_exceeded"
    QUOTE_STALE = "quote_stale"  # Quote expired
    ACCOUNT_LOCKED = "account_locked"  # Concurrent tx conflict
    SIMULATION_FAILED = "simulation_failed"
    NETWORK_ERROR = "network_error"
    MEV_FRONTRUN = "mev_frontrun"  # Frontrun by MEV bot


@dataclass
class SlotConfig:
    """Configuration for slot simulation."""

    # Base failure rates by congestion
    failure_rates: dict[str, float] = field(
        default_factory=lambda: {
            "low": 0.20,
            "normal": 0.40,
            "high": 0.60,
            "extreme": 0.80,
        }
    )

    # Slot timing parameters (milliseconds)
    slot_duration_mean_ms: int = 400
    slot_duration_std_ms: int = 50
    leader_variance_ms: int = 100  # Additional variance by leader

    # Priority fee parameters (lamports)
    base_fee_lamports: int = 5000
    priority_fee_percentiles: dict[int, int] = field(
        default_factory=lambda: {
            25: 1000,  # 25th percentile
            50: 5000,  # Median
            75: 25000,  # 75th percentile
            90: 100000,  # 90th percentile
            99: 500000,  # Top 1%
        }
    )

    # MEV parameters
    mev_frontrun_probability: float = 0.15  # Without Jito
    sandwich_probability: float = 0.15

    # Quote staleness parameters
    quote_staleness_threshold_ms: int = 1000  # Quote > 1s old = stale
    staleness_failure_rate: float = 0.30  # Additional failure from stale quotes

    # Trade size impact
    large_trade_threshold_sol: float = 1.0
    large_trade_failure_boost: float = 0.10

    # Retry parameters
    max_retries: int = 3
    retry_delay_ms: int = 500


@dataclass
class TransactionResult:
    """Result of a transaction attempt."""

    success: bool
    slot: int | None
    signature: str | None
    failure_reason: FailureReason | None
    attempts: int
    total_latency_ms: float
    priority_fee_paid: int
    effective_price: float | None
    slippage_bps: int | None
    congestion_level: CongestionLevel

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "slot": self.slot,
            "signature": self.signature,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "attempts": self.attempts,
            "total_latency_ms": self.total_latency_ms,
            "priority_fee_paid": self.priority_fee_paid,
            "effective_price": self.effective_price,
            "slippage_bps": self.slippage_bps,
            "congestion_level": self.congestion_level.value,
        }


@dataclass
class NetworkState:
    """Current network state for simulation."""

    congestion: CongestionLevel
    current_slot: int
    tps: float  # Transactions per second
    priority_fee_median: int
    mev_activity_level: float  # 0-1, higher = more MEV

    @classmethod
    def random(cls, seed: int | None = None) -> "NetworkState":
        """Generate random network state."""
        if seed:
            random.seed(seed)

        congestion_weights = [0.3, 0.4, 0.2, 0.1]  # Most time in low/normal
        congestion = random.choices(
            list(CongestionLevel),
            weights=congestion_weights,
        )[0]

        tps_by_congestion = {
            CongestionLevel.LOW: random.uniform(2000, 3000),
            CongestionLevel.NORMAL: random.uniform(3000, 4000),
            CongestionLevel.HIGH: random.uniform(4000, 5000),
            CongestionLevel.EXTREME: random.uniform(5000, 6500),
        }

        fee_by_congestion = {
            CongestionLevel.LOW: random.randint(1000, 5000),
            CongestionLevel.NORMAL: random.randint(5000, 25000),
            CongestionLevel.HIGH: random.randint(25000, 100000),
            CongestionLevel.EXTREME: random.randint(100000, 500000),
        }

        return cls(
            congestion=congestion,
            current_slot=random.randint(200000000, 300000000),
            tps=tps_by_congestion[congestion],
            priority_fee_median=fee_by_congestion[congestion],
            mev_activity_level=random.uniform(0.1, 0.4),
        )


class SlotSimulator:
    """Simulate Solana transaction inclusion dynamics.

    Models realistic transaction success/failure based on:
    - Network congestion levels
    - Priority fee competition
    - MEV activity
    - Quote staleness
    - Trade size effects

    Example:
        simulator = SlotSimulator()
        result = simulator.simulate_transaction(
            priority_fee=10000,
            trade_size_sol=0.5,
            quote_age_ms=500,
        )
        if result.success:
            print(f"Landed in slot {result.slot}")
    """

    def __init__(self, config: SlotConfig | None = None, seed: int | None = None):
        self.config = config or SlotConfig()
        self.rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

        # State tracking
        self._current_state: NetworkState | None = None
        self._transaction_history: deque[TransactionResult] = deque(maxlen=10_000)

    def set_network_state(self, state: NetworkState):
        """Set the current network state."""
        self._current_state = state

    def simulate_transaction(
        self,
        priority_fee: int,
        trade_size_sol: float,
        quote_age_ms: int,
        expected_price: float,
        slippage_tolerance_bps: int,
        use_jito: bool = False,
        network_state: NetworkState | None = None,
    ) -> TransactionResult:
        """Simulate a single transaction attempt.

        Args:
            priority_fee: Priority fee in lamports
            trade_size_sol: Trade size in SOL
            quote_age_ms: Age of price quote in milliseconds
            expected_price: Expected execution price
            slippage_tolerance_bps: Maximum acceptable slippage in bps
            use_jito: Whether using Jito bundle
            network_state: Optional network state override

        Returns:
            TransactionResult with success/failure details
        """
        state = network_state or self._current_state or NetworkState.random()

        # Calculate base failure probability
        base_failure = self.config.failure_rates[state.congestion.value]

        # Adjust for priority fee
        fee_adjustment = self._calculate_fee_adjustment(priority_fee, state.priority_fee_median)
        failure_prob = base_failure * (1 - fee_adjustment)

        # Adjust for quote staleness
        if quote_age_ms > self.config.quote_staleness_threshold_ms:
            failure_prob += self.config.staleness_failure_rate * (
                quote_age_ms / self.config.quote_staleness_threshold_ms - 1
            )

        # Adjust for trade size
        if trade_size_sol > self.config.large_trade_threshold_sol:
            failure_prob += self.config.large_trade_failure_boost * (
                trade_size_sol / self.config.large_trade_threshold_sol
            )

        # Jito reduces failure and MEV
        if use_jito:
            failure_prob *= 0.7  # 30% improvement
            mev_prob = self.config.mev_frontrun_probability * 0.1  # 90% reduction
        else:
            mev_prob = self.config.mev_frontrun_probability * state.mev_activity_level

        # Clamp probability
        failure_prob = min(0.95, max(0.05, failure_prob))

        # Simulate latency
        slot_latency = self._np_rng.normal(
            self.config.slot_duration_mean_ms,
            self.config.slot_duration_std_ms,
        )
        slot_latency = max(100, slot_latency)  # Minimum 100ms

        # Determine outcome
        roll = self.rng.random()

        if roll < failure_prob:
            # Transaction failed
            failure_reason = self._determine_failure_reason(
                quote_age_ms, priority_fee, state, mev_prob
            )

            result = TransactionResult(
                success=False,
                slot=None,
                signature=None,
                failure_reason=failure_reason,
                attempts=1,
                total_latency_ms=slot_latency,
                priority_fee_paid=0,
                effective_price=None,
                slippage_bps=None,
                congestion_level=state.congestion,
            )
        else:
            # Transaction succeeded
            effective_price, slippage = self._calculate_execution_price(
                expected_price, trade_size_sol, state, mev_prob
            )

            # Check if slippage exceeds tolerance
            if abs(slippage) > slippage_tolerance_bps:
                result = TransactionResult(
                    success=False,
                    slot=state.current_slot + 1,
                    signature=None,
                    failure_reason=FailureReason.SLIPPAGE_EXCEEDED,
                    attempts=1,
                    total_latency_ms=slot_latency,
                    priority_fee_paid=priority_fee,
                    effective_price=effective_price,
                    slippage_bps=slippage,
                    congestion_level=state.congestion,
                )
            else:
                result = TransactionResult(
                    success=True,
                    slot=state.current_slot + 1,
                    signature=f"sim_{self.rng.randint(0, 2**32):x}",
                    failure_reason=None,
                    attempts=1,
                    total_latency_ms=slot_latency,
                    priority_fee_paid=priority_fee,
                    effective_price=effective_price,
                    slippage_bps=slippage,
                    congestion_level=state.congestion,
                )

        self._transaction_history.append(result)
        return result

    def simulate_with_retry(
        self,
        priority_fee: int,
        trade_size_sol: float,
        quote_age_ms: int,
        expected_price: float,
        slippage_tolerance_bps: int,
        use_jito: bool = False,
        max_retries: int | None = None,
    ) -> TransactionResult:
        """Simulate transaction with retries.

        Args:
            priority_fee: Initial priority fee
            trade_size_sol: Trade size
            quote_age_ms: Initial quote age
            expected_price: Expected price
            slippage_tolerance_bps: Slippage tolerance
            use_jito: Whether to use Jito
            max_retries: Max retry attempts

        Returns:
            Final TransactionResult
        """
        max_retries = max_retries or self.config.max_retries
        total_attempts = 0
        total_latency = 0.0
        current_quote_age = quote_age_ms

        for attempt in range(max_retries + 1):
            total_attempts += 1

            result = self.simulate_transaction(
                priority_fee=priority_fee * (1 + attempt * 0.5),  # Increase fee on retry
                trade_size_sol=trade_size_sol,
                quote_age_ms=current_quote_age,
                expected_price=expected_price,
                slippage_tolerance_bps=slippage_tolerance_bps,
                use_jito=use_jito,
            )

            total_latency += result.total_latency_ms

            if result.success:
                result.attempts = total_attempts
                result.total_latency_ms = total_latency
                return result

            # Update quote age for next attempt
            current_quote_age += self.config.retry_delay_ms + int(result.total_latency_ms)

        # All retries exhausted
        result.attempts = total_attempts
        result.total_latency_ms = total_latency
        return result

    def _calculate_fee_adjustment(
        self,
        priority_fee: int,
        median_fee: int,
    ) -> float:
        """Calculate success probability adjustment based on fee.

        Higher fee relative to median = higher success probability.
        """
        if median_fee == 0:
            return 0.5

        fee_ratio = priority_fee / median_fee

        # Sigmoid-like curve: ratio of 2x median gives ~0.3 boost
        adjustment = 0.5 / (1 + np.exp(-2 * (fee_ratio - 1)))

        return min(0.5, adjustment)

    def _calculate_execution_price(
        self,
        expected_price: float,
        trade_size_sol: float,
        state: NetworkState,
        mev_prob: float,
    ) -> tuple[float, int]:
        """Calculate actual execution price and slippage.

        Returns:
            Tuple of (effective_price, slippage_bps)
        """
        # Base price impact from trade size
        size_impact_pct = 0.1 * np.sqrt(trade_size_sol)  # Sqrt for diminishing impact

        # Congestion impact
        congestion_impact = {
            CongestionLevel.LOW: 0.05,
            CongestionLevel.NORMAL: 0.10,
            CongestionLevel.HIGH: 0.20,
            CongestionLevel.EXTREME: 0.40,
        }
        congestion_pct = congestion_impact[state.congestion]

        # MEV impact (frontrunning)
        if self.rng.random() < mev_prob:
            mev_pct = self.rng.uniform(0.5, 2.0)  # 0.5-2% MEV extraction
        else:
            mev_pct = 0.0

        # Total slippage (always negative for buys = worse price)
        total_slippage_pct = size_impact_pct + congestion_pct + mev_pct
        # Add some randomness
        total_slippage_pct *= self._np_rng.uniform(0.5, 1.5)

        # Calculate effective price (assuming buy = pay more)
        effective_price = expected_price * (1 + total_slippage_pct / 100)
        slippage_bps = int(total_slippage_pct * 100)

        return effective_price, slippage_bps

    def _determine_failure_reason(
        self,
        quote_age_ms: int,
        priority_fee: int,
        state: NetworkState,
        mev_prob: float,
    ) -> FailureReason:
        """Determine the most likely failure reason."""
        reasons_probs = []

        # Quote staleness
        if quote_age_ms > self.config.quote_staleness_threshold_ms:
            reasons_probs.append((FailureReason.QUOTE_STALE, 0.3))

        # Low fee
        if priority_fee < state.priority_fee_median * 0.5:
            reasons_probs.append((FailureReason.INSUFFICIENT_FEE, 0.3))

        # MEV
        if self.rng.random() < mev_prob:
            reasons_probs.append((FailureReason.MEV_FRONTRUN, 0.2))

        # General failures
        reasons_probs.extend(
            [
                (FailureReason.SLOT_MISSED, 0.25),
                (FailureReason.ACCOUNT_LOCKED, 0.1),
                (FailureReason.SIMULATION_FAILED, 0.1),
                (FailureReason.NETWORK_ERROR, 0.05),
            ]
        )

        # Normalize and sample
        total = sum(p for _, p in reasons_probs)
        reasons_probs = [(r, p / total) for r, p in reasons_probs]

        roll = self.rng.random()
        cumulative = 0.0
        for reason, prob in reasons_probs:
            cumulative += prob
            if roll < cumulative:
                return reason

        return FailureReason.SLOT_MISSED

    def get_statistics(self) -> dict[str, Any]:
        """Get simulation statistics."""
        if not self._transaction_history:
            return {"total_transactions": 0}

        successes = sum(1 for r in self._transaction_history if r.success)
        total = len(self._transaction_history)

        slippages = [
            r.slippage_bps for r in self._transaction_history if r.slippage_bps is not None
        ]

        latencies = [r.total_latency_ms for r in self._transaction_history]

        failure_reasons = {}
        for r in self._transaction_history:
            if r.failure_reason:
                key = r.failure_reason.value
                failure_reasons[key] = failure_reasons.get(key, 0) + 1

        return {
            "total_transactions": total,
            "success_rate": successes / total if total > 0 else 0,
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "avg_slippage_bps": np.mean(slippages) if slippages else 0,
            "max_slippage_bps": max(slippages) if slippages else 0,
            "failure_reasons": failure_reasons,
        }

    def reset(self):
        """Reset transaction history."""
        self._transaction_history.clear()


def estimate_optimal_priority_fee(
    target_success_rate: float,
    congestion: CongestionLevel,
    config: SlotConfig | None = None,
) -> int:
    """Estimate optimal priority fee for target success rate.

    Args:
        target_success_rate: Desired success probability (0-1)
        congestion: Current network congestion
        config: Slot configuration

    Returns:
        Recommended priority fee in lamports
    """
    config = config or SlotConfig()

    base_failure = config.failure_rates[congestion.value]
    target_failure = 1 - target_success_rate

    if target_failure >= base_failure:
        # Base fee is sufficient
        return config.base_fee_lamports

    # Need to boost fee to reduce failure rate
    # Inverse of fee_adjustment calculation
    required_adjustment = (base_failure - target_failure) / base_failure
    required_adjustment = min(0.5, required_adjustment)

    # Inverse sigmoid: ratio = 1 - ln((0.5/adj) - 1) / 2
    if required_adjustment >= 0.5:
        fee_ratio = 3.0  # Cap at 3x median
    else:
        fee_ratio = 1 + 0.5 * np.log((0.5 / required_adjustment) - 1)
        fee_ratio = max(1.0, min(5.0, fee_ratio))

    # Get median fee for congestion
    median_fees = {
        CongestionLevel.LOW: 5000,
        CongestionLevel.NORMAL: 25000,
        CongestionLevel.HIGH: 100000,
        CongestionLevel.EXTREME: 300000,
    }

    return int(median_fees[congestion] * fee_ratio)
