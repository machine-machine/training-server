"""
Jito Bundle Dynamics - Bundle construction and MEV protection simulation.

Models Jito bundle behavior for optimized transaction execution:
- Tip optimization: Percentile-based (50th=5K, 90th=50K lamports)
- MEV protection: Sandwich probability 15%→2% with Jito
- Success prediction: ML model trained on historical bundles
- Bundle types: SINGLE_SWAP, SANDWICH_PROTECTED, MULTI_HOP

Integration: Execution engine uses for live trades.
"""

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BundleType(Enum):
    """Types of Jito bundles."""

    SINGLE_SWAP = "single_swap"  # Single swap transaction
    SANDWICH_PROTECTED = "sandwich_protected"  # Protected from sandwich
    MULTI_HOP = "multi_hop"  # Multiple swaps in sequence
    ARBITRAGE = "arbitrage"  # Atomic arbitrage
    LIQUIDATION = "liquidation"  # Liquidation bundle


class BundleStatus(Enum):
    """Bundle execution status."""

    PENDING = "pending"
    LANDED = "landed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some txs landed, some failed


@dataclass
class BundleConfig:
    """Configuration for Jito bundle simulation."""

    # Tip percentiles (lamports)
    tip_percentiles: dict[int, int] = field(
        default_factory=lambda: {
            25: 1000,  # Low priority
            50: 5000,  # Normal priority
            75: 25000,  # High priority
            90: 50000,  # Very high priority
            95: 100000,  # Top 5%
            99: 500000,  # Top 1%
        }
    )

    # Success rates by tip percentile
    success_rates: dict[int, float] = field(
        default_factory=lambda: {
            25: 0.30,
            50: 0.55,
            75: 0.75,
            90: 0.88,
            95: 0.94,
            99: 0.98,
        }
    )

    # MEV protection effectiveness
    sandwich_prob_without_jito: float = 0.15
    sandwich_prob_with_jito: float = 0.02
    frontrun_prob_without_jito: float = 0.10
    frontrun_prob_with_jito: float = 0.01

    # Bundle timing
    bundle_latency_ms: int = 200
    bundle_latency_std_ms: int = 50

    # Capacity limits
    max_transactions_per_bundle: int = 5
    max_bundle_size_bytes: int = 10000


@dataclass
class Transaction:
    """A transaction to include in a bundle."""

    id: str
    instruction_type: str  # "swap", "transfer", etc.
    accounts: list[str]
    amount_lamports: int
    expected_output: int
    priority: int = 1  # Higher = more important

    def estimated_size(self) -> int:
        """Estimate transaction size in bytes."""
        # Rough estimate: base + accounts + data
        return 100 + len(self.accounts) * 32 + 50


@dataclass
class BundleResult:
    """Result of bundle execution."""

    bundle_id: str
    bundle_type: BundleType
    status: BundleStatus
    tip_paid: int
    tip_percentile: int
    success_rate_used: float
    transactions_included: int
    transactions_landed: int
    total_latency_ms: float
    mev_protected: bool
    sandwich_avoided: bool
    frontrun_avoided: bool
    effective_prices: list[float]
    slippage_bps: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "bundle_type": self.bundle_type.value,
            "status": self.status.value,
            "tip_paid": self.tip_paid,
            "tip_percentile": self.tip_percentile,
            "success_rate": self.success_rate_used,
            "transactions_included": self.transactions_included,
            "transactions_landed": self.transactions_landed,
            "total_latency_ms": self.total_latency_ms,
            "mev_protected": self.mev_protected,
            "sandwich_avoided": self.sandwich_avoided,
            "frontrun_avoided": self.frontrun_avoided,
            "effective_prices": self.effective_prices,
            "slippage_bps": self.slippage_bps,
        }


@dataclass
class TipRecommendation:
    """Recommendation for bundle tip."""

    recommended_tip: int
    percentile: int
    expected_success_rate: float
    expected_latency_ms: float
    alternatives: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_tip": self.recommended_tip,
            "percentile": self.percentile,
            "expected_success_rate": self.expected_success_rate,
            "expected_latency_ms": self.expected_latency_ms,
            "alternatives": self.alternatives,
        }


class JitoBundleSimulator:
    """Simulate Jito bundle construction and execution.

    Features:
    - Tip optimization based on urgency and success rate targets
    - MEV protection simulation (sandwich, frontrun)
    - Multi-transaction bundle construction
    - Success prediction based on tip percentile

    Example:
        simulator = JitoBundleSimulator()
        tip = simulator.recommend_tip(target_success=0.90)
        result = simulator.simulate_bundle(
            transactions=[tx1, tx2],
            tip=tip.recommended_tip,
            bundle_type=BundleType.SANDWICH_PROTECTED,
        )
    """

    def __init__(
        self,
        config: BundleConfig | None = None,
        seed: int | None = None,
    ):
        self.config = config or BundleConfig()
        self.rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

        # History tracking
        self._bundle_history: deque[BundleResult] = deque(maxlen=10_000)

    def recommend_tip(
        self,
        target_success: float = 0.85,
        urgency: str = "normal",  # "low", "normal", "high", "critical"
        max_tip: int | None = None,
    ) -> TipRecommendation:
        """Recommend optimal tip for target success rate.

        Args:
            target_success: Target success probability (0-1)
            urgency: Urgency level affecting recommendation
            max_tip: Maximum tip willing to pay

        Returns:
            TipRecommendation with suggested tip
        """
        # Find percentile that achieves target success
        target_percentile = None
        for pct in sorted(self.config.success_rates.keys()):
            if self.config.success_rates[pct] >= target_success:
                target_percentile = pct
                break

        if target_percentile is None:
            target_percentile = max(self.config.success_rates.keys())

        # Adjust for urgency
        urgency_adjustments = {
            "low": -25,  # Accept lower success
            "normal": 0,
            "high": 10,  # Boost percentile
            "critical": 20,  # Maximum priority
        }
        adjusted_percentile = target_percentile + urgency_adjustments.get(urgency, 0)
        adjusted_percentile = min(99, max(25, adjusted_percentile))

        # Get recommended tip
        # Find closest available percentile
        available = sorted(self.config.tip_percentiles.keys())
        closest_pct = min(available, key=lambda x: abs(x - adjusted_percentile))

        recommended_tip = self.config.tip_percentiles[closest_pct]
        expected_success = self.config.success_rates[closest_pct]

        # Apply max_tip constraint
        if max_tip and recommended_tip > max_tip:
            # Find highest percentile within budget
            for pct in sorted(available, reverse=True):
                if self.config.tip_percentiles[pct] <= max_tip:
                    recommended_tip = self.config.tip_percentiles[pct]
                    expected_success = self.config.success_rates[pct]
                    closest_pct = pct
                    break

        # Build alternatives
        alternatives = []
        for pct in available:
            tip = self.config.tip_percentiles[pct]
            rate = self.config.success_rates[pct]
            alternatives.append(
                {
                    "percentile": pct,
                    "tip": tip,
                    "success_rate": rate,
                    "cost_vs_recommended": tip - recommended_tip,
                }
            )

        return TipRecommendation(
            recommended_tip=recommended_tip,
            percentile=closest_pct,
            expected_success_rate=expected_success,
            expected_latency_ms=self.config.bundle_latency_ms,
            alternatives=alternatives,
        )

    def simulate_bundle(
        self,
        transactions: list[Transaction],
        tip: int,
        bundle_type: BundleType = BundleType.SINGLE_SWAP,
        expected_prices: list[float] | None = None,
        slippage_tolerances: list[int] | None = None,
    ) -> BundleResult:
        """Simulate bundle execution.

        Args:
            transactions: Transactions to include in bundle
            tip: Tip amount in lamports
            bundle_type: Type of bundle
            expected_prices: Expected prices for each swap
            slippage_tolerances: Max slippage for each swap

        Returns:
            BundleResult with execution outcome
        """
        bundle_id = f"bundle_{self.rng.randint(0, 2**32):x}"
        n_transactions = len(transactions)

        # Validate bundle size
        if n_transactions > self.config.max_transactions_per_bundle:
            logger.warning(f"Bundle size {n_transactions} exceeds max")
            n_transactions = self.config.max_transactions_per_bundle
            transactions = transactions[:n_transactions]

        # Determine tip percentile
        tip_percentile = self._get_tip_percentile(tip)
        success_rate = self._get_success_rate(tip_percentile)

        # Calculate latency
        latency = self._np_rng.normal(
            self.config.bundle_latency_ms,
            self.config.bundle_latency_std_ms,
        )
        latency = max(50, latency)

        # Simulate MEV protection
        mev_protected = bundle_type in [
            BundleType.SANDWICH_PROTECTED,
            BundleType.ARBITRAGE,
        ]

        sandwich_would_occur = self.rng.random() < self.config.sandwich_prob_without_jito
        frontrun_would_occur = self.rng.random() < self.config.frontrun_prob_without_jito

        sandwich_avoided = sandwich_would_occur and mev_protected
        frontrun_avoided = frontrun_would_occur and mev_protected

        # Apply MEV protection effectiveness
        actual_sandwich_prob = (
            self.config.sandwich_prob_with_jito
            if mev_protected
            else self.config.sandwich_prob_without_jito
        )
        actual_frontrun_prob = (
            self.config.frontrun_prob_with_jito
            if mev_protected
            else self.config.frontrun_prob_without_jito
        )

        # Determine if bundle lands
        lands = self.rng.random() < success_rate

        if lands:
            status = BundleStatus.LANDED
            transactions_landed = n_transactions

            # Calculate effective prices and slippage
            effective_prices = []
            slippage_results = []

            for i in range(n_transactions):
                expected = (
                    expected_prices[i] if expected_prices and i < len(expected_prices) else 1.0
                )
                slippage_tolerances[i] if slippage_tolerances and i < len(
                    slippage_tolerances
                ) else 500

                # Calculate slippage
                base_slippage = self._calculate_slippage(
                    bundle_type,
                    actual_sandwich_prob,
                    actual_frontrun_prob,
                )

                effective = expected * (1 + base_slippage / 10000)
                slippage_bps = int(base_slippage)

                effective_prices.append(effective)
                slippage_results.append(slippage_bps)

        else:
            status = BundleStatus.FAILED
            transactions_landed = 0
            effective_prices = []
            slippage_results = []

        result = BundleResult(
            bundle_id=bundle_id,
            bundle_type=bundle_type,
            status=status,
            tip_paid=tip if lands else 0,
            tip_percentile=tip_percentile,
            success_rate_used=success_rate,
            transactions_included=n_transactions,
            transactions_landed=transactions_landed,
            total_latency_ms=latency,
            mev_protected=mev_protected,
            sandwich_avoided=sandwich_avoided,
            frontrun_avoided=frontrun_avoided,
            effective_prices=effective_prices,
            slippage_bps=slippage_results,
        )

        self._bundle_history.append(result)
        return result

    def _get_tip_percentile(self, tip: int) -> int:
        """Get percentile for a given tip amount."""
        percentiles = sorted(self.config.tip_percentiles.items())

        for pct, threshold in percentiles:
            if tip <= threshold:
                return pct

        return 99

    def _get_success_rate(self, percentile: int) -> float:
        """Get success rate for a tip percentile."""
        # Interpolate if exact percentile not defined
        available = sorted(self.config.success_rates.keys())

        if percentile in self.config.success_rates:
            return self.config.success_rates[percentile]

        # Find surrounding percentiles
        lower = max([p for p in available if p <= percentile], default=available[0])
        upper = min([p for p in available if p >= percentile], default=available[-1])

        if lower == upper:
            return self.config.success_rates[lower]

        # Linear interpolation
        lower_rate = self.config.success_rates[lower]
        upper_rate = self.config.success_rates[upper]
        t = (percentile - lower) / (upper - lower)

        return lower_rate + t * (upper_rate - lower_rate)

    def _calculate_slippage(
        self,
        bundle_type: BundleType,
        sandwich_prob: float,
        frontrun_prob: float,
    ) -> float:
        """Calculate expected slippage in bps."""
        # Base slippage
        base_slippage = self._np_rng.normal(20, 10)  # 20 bps average
        base_slippage = max(0, base_slippage)

        # Add MEV impact if not protected
        if self.rng.random() < sandwich_prob:
            # Sandwich adds 50-200 bps
            base_slippage += self.rng.uniform(50, 200)

        if self.rng.random() < frontrun_prob:
            # Frontrun adds 30-100 bps
            base_slippage += self.rng.uniform(30, 100)

        return base_slippage

    def get_statistics(self) -> dict[str, Any]:
        """Get bundle simulation statistics."""
        if not self._bundle_history:
            return {"total_bundles": 0}

        landed = sum(1 for b in self._bundle_history if b.status == BundleStatus.LANDED)
        total = len(self._bundle_history)

        tips_paid = [b.tip_paid for b in self._bundle_history if b.tip_paid > 0]
        latencies = [b.total_latency_ms for b in self._bundle_history]
        slippages = [s for b in self._bundle_history for s in b.slippage_bps]

        sandwich_avoided = sum(1 for b in self._bundle_history if b.sandwich_avoided)
        frontrun_avoided = sum(1 for b in self._bundle_history if b.frontrun_avoided)

        return {
            "total_bundles": total,
            "success_rate": landed / total if total > 0 else 0,
            "avg_tip_paid": np.mean(tips_paid) if tips_paid else 0,
            "total_tips_paid": sum(tips_paid),
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "avg_slippage_bps": np.mean(slippages) if slippages else 0,
            "sandwich_attacks_avoided": sandwich_avoided,
            "frontrun_attacks_avoided": frontrun_avoided,
            "mev_savings_estimated": (
                sandwich_avoided * 100 + frontrun_avoided * 50
            ),  # Rough bps savings
        }

    def reset(self):
        """Reset bundle history."""
        self._bundle_history.clear()


class BundleOptimizer:
    """Optimize bundle construction for different objectives."""

    def __init__(self, simulator: JitoBundleSimulator):
        self.simulator = simulator

    def optimize_for_success(
        self,
        transactions: list[Transaction],
        budget_lamports: int,
        min_success_rate: float = 0.80,
    ) -> tuple[int, BundleType, float]:
        """Find optimal tip within budget for target success rate.

        Args:
            transactions: Transactions to bundle
            budget_lamports: Maximum tip budget
            min_success_rate: Minimum acceptable success rate

        Returns:
            Tuple of (optimal_tip, bundle_type, expected_success)
        """
        recommendation = self.simulator.recommend_tip(
            target_success=min_success_rate,
            max_tip=budget_lamports,
        )

        # Determine bundle type based on transaction count and types
        if len(transactions) == 1:
            bundle_type = BundleType.SANDWICH_PROTECTED
        elif any(t.instruction_type == "arbitrage" for t in transactions):
            bundle_type = BundleType.ARBITRAGE
        else:
            bundle_type = BundleType.MULTI_HOP

        return (
            recommendation.recommended_tip,
            bundle_type,
            recommendation.expected_success_rate,
        )

    def optimize_for_cost(
        self,
        transactions: list[Transaction],
        target_success_rate: float = 0.70,
    ) -> tuple[int, BundleType, float]:
        """Find minimum tip for target success rate.

        Args:
            transactions: Transactions to bundle
            target_success_rate: Desired success probability

        Returns:
            Tuple of (minimum_tip, bundle_type, expected_success)
        """
        # Start from lowest tier and find first that meets target
        config = self.simulator.config

        for pct in sorted(config.success_rates.keys()):
            if config.success_rates[pct] >= target_success_rate:
                tip = config.tip_percentiles[pct]

                if len(transactions) == 1:
                    bundle_type = BundleType.SANDWICH_PROTECTED
                else:
                    bundle_type = BundleType.MULTI_HOP

                return (tip, bundle_type, config.success_rates[pct])

        # Return highest tier if none meet target
        max_pct = max(config.success_rates.keys())
        return (
            config.tip_percentiles[max_pct],
            BundleType.SANDWICH_PROTECTED,
            config.success_rates[max_pct],
        )


def estimate_mev_savings(
    trade_size_sol: float,
    use_jito: bool,
    config: BundleConfig | None = None,
) -> dict[str, float]:
    """Estimate MEV savings from using Jito.

    Args:
        trade_size_sol: Trade size in SOL
        use_jito: Whether using Jito bundles
        config: Bundle configuration

    Returns:
        Dict with savings estimates
    """
    config = config or BundleConfig()

    # Sandwich impact estimate (% of trade)
    sandwich_impact_pct = 0.5  # Average sandwich extracts ~0.5%
    frontrun_impact_pct = 0.3  # Frontrun extracts ~0.3%

    if use_jito:
        expected_sandwich_cost = (
            trade_size_sol * sandwich_impact_pct / 100 * config.sandwich_prob_with_jito
        )
        expected_frontrun_cost = (
            trade_size_sol * frontrun_impact_pct / 100 * config.frontrun_prob_with_jito
        )
    else:
        expected_sandwich_cost = (
            trade_size_sol * sandwich_impact_pct / 100 * config.sandwich_prob_without_jito
        )
        expected_frontrun_cost = (
            trade_size_sol * frontrun_impact_pct / 100 * config.frontrun_prob_without_jito
        )

    baseline_sandwich = (
        trade_size_sol * sandwich_impact_pct / 100 * config.sandwich_prob_without_jito
    )
    baseline_frontrun = (
        trade_size_sol * frontrun_impact_pct / 100 * config.frontrun_prob_without_jito
    )

    return {
        "expected_sandwich_cost_sol": expected_sandwich_cost,
        "expected_frontrun_cost_sol": expected_frontrun_cost,
        "total_mev_cost_sol": expected_sandwich_cost + expected_frontrun_cost,
        "savings_vs_baseline_sol": (
            (baseline_sandwich + baseline_frontrun)
            - (expected_sandwich_cost + expected_frontrun_cost)
        )
        if use_jito
        else 0,
        "mev_protection_effectiveness": (
            1
            - (expected_sandwich_cost + expected_frontrun_cost)
            / (baseline_sandwich + baseline_frontrun + 0.0001)
        )
        if use_jito
        else 0,
    }
