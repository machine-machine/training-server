"""
Market condition perturbations for Monte Carlo simulation.

Implements realistic stochastic perturbations for:
- Slippage (execution price impact)
- Latency (transaction delays)
- Failure Rate (transaction inclusion probability)
- MEV Impact (frontrunning/sandwich attacks)
- Liquidity (available pool liquidity)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PerturbationType(Enum):
    """Types of market perturbations."""

    SLIPPAGE = "slippage"
    LATENCY = "latency"
    FAILURE_RATE = "failure_rate"
    MEV_IMPACT = "mev_impact"
    LIQUIDITY = "liquidity"


@dataclass
class PerturbationConfig:
    """Configuration for Monte Carlo perturbations."""

    # Slippage: Normal distribution
    slippage_mean_bps: float = 0.0  # Mean additional slippage
    slippage_std_bps: float = 50.0  # Std dev of slippage

    # Latency: Exponential distribution
    latency_mean_ms: float = 50.0  # Mean latency
    latency_max_ms: float = 500.0  # Max latency cap

    # Failure Rate: Uniform distribution
    failure_rate_min: float = 0.30  # Minimum failure rate
    failure_rate_max: float = 0.50  # Maximum failure rate
    failure_base_prob: float = 0.85  # Base inclusion probability

    # MEV Impact: Beta distribution
    mev_alpha: float = 2.0  # Beta alpha parameter
    mev_beta: float = 5.0  # Beta beta parameter
    mev_max_bps: float = 200.0  # Maximum MEV impact

    # Liquidity: Normal distribution
    liquidity_std_pct: float = 15.0  # Std dev as percentage

    # Random seed for reproducibility
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slippage": {"mean_bps": self.slippage_mean_bps, "std_bps": self.slippage_std_bps},
            "latency": {"mean_ms": self.latency_mean_ms, "max_ms": self.latency_max_ms},
            "failure_rate": {"min": self.failure_rate_min, "max": self.failure_rate_max},
            "mev": {"alpha": self.mev_alpha, "beta": self.mev_beta, "max_bps": self.mev_max_bps},
            "liquidity": {"std_pct": self.liquidity_std_pct},
        }


@dataclass
class PerturbationResult:
    """Result of applying perturbations to a single trade."""

    # Applied perturbations
    slippage_bps: float = 0.0
    latency_ms: float = 0.0
    included: bool = True
    mev_impact_bps: float = 0.0
    liquidity_multiplier: float = 1.0

    # Adjusted trade values
    adjusted_entry_price: float = 0.0
    adjusted_exit_price: float = 0.0
    adjusted_pnl_pct: float = 0.0

    def total_slippage_bps(self) -> float:
        """Total slippage including MEV."""
        return self.slippage_bps + self.mev_impact_bps


class BasePerturbation(ABC):
    """Base class for market perturbations."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    @abstractmethod
    def apply(self, value: float, context: dict[str, Any]) -> float:
        """Apply perturbation to a value."""
        pass


class SlippagePerturbation(BasePerturbation):
    """Slippage perturbation with normal distribution.

    Slippage increases with:
    - Trade size relative to liquidity
    - Market volatility
    """

    def __init__(
        self,
        rng: np.random.Generator,
        mean_bps: float = 0.0,
        std_bps: float = 50.0,
    ):
        super().__init__(rng)
        self.mean_bps = mean_bps
        self.std_bps = std_bps

    def apply(self, value: float, context: dict[str, Any]) -> float:
        """Apply slippage perturbation to execution price.

        Args:
            value: Original price
            context: Contains 'is_buy', 'trade_size', 'liquidity', 'volatility'

        Returns:
            Price adjusted for slippage
        """
        # Base random slippage
        base_slippage_bps = self.rng.normal(self.mean_bps, self.std_bps)

        # Scale by trade size / liquidity
        trade_size = context.get("trade_size", 0)
        liquidity = context.get("liquidity", 1)
        size_factor = 1.0 + (trade_size / max(liquidity, 1e-10))

        # Scale by volatility
        volatility = context.get("volatility", 0)
        vol_factor = 1.0 + volatility / 100.0

        # Total slippage
        total_slippage_bps = abs(base_slippage_bps) * size_factor * vol_factor

        # Apply direction
        is_buy = context.get("is_buy", True)
        direction = 1 if is_buy else -1

        adjusted_price = value * (1 + direction * total_slippage_bps / 10000)

        return adjusted_price


class LatencyPerturbation(BasePerturbation):
    """Latency perturbation with exponential distribution.

    Models network and block confirmation delays.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        mean_ms: float = 50.0,
        max_ms: float = 500.0,
    ):
        super().__init__(rng)
        self.mean_ms = mean_ms
        self.max_ms = max_ms

    def apply(self, value: float, context: dict[str, Any]) -> float:
        """Generate latency perturbation.

        Args:
            value: Base latency (ignored, generates new)
            context: May contain 'congestion_level'

        Returns:
            Latency in milliseconds
        """
        # Base exponential latency
        latency = self.rng.exponential(self.mean_ms)

        # Scale by network congestion
        congestion = context.get("congestion_level", 1.0)
        latency *= congestion

        # Cap at maximum
        return min(latency, self.max_ms)


class FailureRatePerturbation(BasePerturbation):
    """Transaction failure/inclusion perturbation.

    Models transaction not landing on-chain due to:
    - Network congestion
    - Insufficient priority fee
    - Slippage tolerance exceeded
    """

    def __init__(
        self,
        rng: np.random.Generator,
        failure_rate_min: float = 0.30,
        failure_rate_max: float = 0.50,
        base_prob: float = 0.85,
    ):
        super().__init__(rng)
        self.failure_rate_min = failure_rate_min
        self.failure_rate_max = failure_rate_max
        self.base_prob = base_prob

    def apply(self, value: float, context: dict[str, Any]) -> float:
        """Determine if transaction is included.

        Args:
            value: Ignored
            context: May contain 'priority_fee', 'slippage_tolerance'

        Returns:
            1.0 if included, 0.0 if failed
        """
        # Sample failure rate from uniform
        failure_rate = self.rng.uniform(self.failure_rate_min, self.failure_rate_max)

        # Adjust for priority fee
        priority_fee = context.get("priority_fee", 0)
        if priority_fee > 0:
            # Higher fee = lower failure rate
            failure_rate *= max(0.5, 1.0 - priority_fee / 10000)

        # Effective inclusion probability
        inclusion_prob = self.base_prob * (1 - failure_rate)

        # Random inclusion
        return 1.0 if self.rng.random() < inclusion_prob else 0.0


class MEVImpactPerturbation(BasePerturbation):
    """MEV (frontrunning/sandwich) impact perturbation.

    Uses Beta distribution to model MEV attacks:
    - Most transactions have low MEV impact
    - Some have significant impact (heavy tail)
    """

    def __init__(
        self,
        rng: np.random.Generator,
        alpha: float = 2.0,
        beta: float = 5.0,
        max_bps: float = 200.0,
    ):
        super().__init__(rng)
        self.alpha = alpha
        self.beta = beta
        self.max_bps = max_bps

    def apply(self, value: float, context: dict[str, Any]) -> float:
        """Apply MEV impact to price.

        Args:
            value: Original execution price
            context: May contain 'trade_size', 'is_popular_token', 'is_buy'

        Returns:
            Price adjusted for MEV impact
        """
        # Sample MEV impact from Beta distribution
        mev_factor = self.rng.beta(self.alpha, self.beta)
        mev_bps = mev_factor * self.max_bps

        # Higher for popular tokens
        if context.get("is_popular_token", False):
            mev_bps *= 1.5

        # Higher for larger trades
        trade_size = context.get("trade_size", 0)
        if trade_size > 1000:  # > $1000
            mev_bps *= 1.0 + np.log10(trade_size / 1000)

        # Apply impact (always unfavorable)
        is_buy = context.get("is_buy", True)
        direction = 1 if is_buy else -1

        adjusted_price = value * (1 + direction * mev_bps / 10000)

        return adjusted_price


class LiquidityPerturbation(BasePerturbation):
    """Liquidity perturbation with normal distribution.

    Models fluctuations in pool liquidity that affect
    execution quality and price impact.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        std_pct: float = 15.0,
    ):
        super().__init__(rng)
        self.std_pct = std_pct

    def apply(self, value: float, context: dict[str, Any]) -> float:
        """Apply liquidity perturbation.

        Args:
            value: Original liquidity
            context: May contain 'time_of_day', 'is_weekend'

        Returns:
            Perturbed liquidity value
        """
        # Base normal perturbation
        multiplier = 1.0 + self.rng.normal(0, self.std_pct / 100)

        # Lower liquidity during off-hours
        if context.get("is_weekend", False):
            multiplier *= 0.85

        # Ensure positive
        multiplier = max(0.1, multiplier)

        return value * multiplier


class Perturbator:
    """Applies all perturbations to trades for Monte Carlo simulation."""

    def __init__(self, config: PerturbationConfig | None = None):
        self.config = config or PerturbationConfig()

        # Initialize RNG
        seed = self.config.seed
        self.rng = np.random.default_rng(seed)

        # Create perturbation instances
        self.slippage = SlippagePerturbation(
            self.rng,
            self.config.slippage_mean_bps,
            self.config.slippage_std_bps,
        )
        self.latency = LatencyPerturbation(
            self.rng,
            self.config.latency_mean_ms,
            self.config.latency_max_ms,
        )
        self.failure = FailureRatePerturbation(
            self.rng,
            self.config.failure_rate_min,
            self.config.failure_rate_max,
            self.config.failure_base_prob,
        )
        self.mev = MEVImpactPerturbation(
            self.rng,
            self.config.mev_alpha,
            self.config.mev_beta,
            self.config.mev_max_bps,
        )
        self.liquidity = LiquidityPerturbation(
            self.rng,
            self.config.liquidity_std_pct,
        )

    def perturb_trade(
        self,
        entry_price: float,
        exit_price: float,
        trade_size: float,
        liquidity: float,
        volatility: float = 0.0,
        is_popular_token: bool = False,
        priority_fee: float = 0.0,
    ) -> PerturbationResult:
        """Apply all perturbations to a single trade.

        Args:
            entry_price: Original entry price
            exit_price: Original exit price
            trade_size: Trade size in USD
            liquidity: Pool liquidity in USD
            volatility: Market volatility (0-100)
            is_popular_token: Whether token is popular (more MEV)
            priority_fee: Priority fee in lamports

        Returns:
            PerturbationResult with all adjustments
        """
        result = PerturbationResult()

        # Build context
        entry_context = {
            "trade_size": trade_size,
            "liquidity": liquidity,
            "volatility": volatility,
            "is_buy": True,
            "is_popular_token": is_popular_token,
            "priority_fee": priority_fee,
        }

        exit_context = {
            **entry_context,
            "is_buy": False,
        }

        # Check inclusion first
        included = self.failure.apply(0, entry_context)
        result.included = included > 0.5

        if not result.included:
            # Transaction failed - no execution
            return result

        # Apply perturbations to entry
        perturbed_liquidity = self.liquidity.apply(liquidity, entry_context)
        result.liquidity_multiplier = perturbed_liquidity / max(liquidity, 1e-10)

        # Update context with perturbed liquidity
        entry_context["liquidity"] = perturbed_liquidity
        exit_context["liquidity"] = perturbed_liquidity

        # Entry price perturbations
        entry_after_slippage = self.slippage.apply(entry_price, entry_context)
        entry_after_mev = self.mev.apply(entry_after_slippage, entry_context)
        result.adjusted_entry_price = entry_after_mev

        # Calculate entry slippage
        result.slippage_bps = abs(entry_after_slippage - entry_price) / entry_price * 10000
        result.mev_impact_bps = (
            abs(entry_after_mev - entry_after_slippage) / entry_after_slippage * 10000
        )

        # Exit price perturbations
        exit_after_slippage = self.slippage.apply(exit_price, exit_context)
        exit_after_mev = self.mev.apply(exit_after_slippage, exit_context)
        result.adjusted_exit_price = exit_after_mev

        # Add exit slippage to total
        exit_slippage_bps = abs(exit_after_slippage - exit_price) / exit_price * 10000
        exit_mev_bps = abs(exit_after_mev - exit_after_slippage) / exit_after_slippage * 10000
        result.slippage_bps += exit_slippage_bps
        result.mev_impact_bps += exit_mev_bps

        # Calculate adjusted PnL
        (exit_price - entry_price) / entry_price * 100
        adjusted_pnl_pct = (
            (result.adjusted_exit_price - result.adjusted_entry_price)
            / result.adjusted_entry_price
            * 100
        )
        result.adjusted_pnl_pct = adjusted_pnl_pct

        # Latency
        result.latency_ms = self.latency.apply(0, entry_context)

        return result

    def perturb_trades_batch(
        self,
        trades: list[dict[str, Any]],
    ) -> list[PerturbationResult]:
        """Apply perturbations to a batch of trades.

        Args:
            trades: List of trade dictionaries with keys:
                    entry_price, exit_price, trade_size, liquidity, etc.

        Returns:
            List of PerturbationResults
        """
        results = []
        for trade in trades:
            result = self.perturb_trade(
                entry_price=trade.get("entry_price", 0),
                exit_price=trade.get("exit_price", 0),
                trade_size=trade.get("trade_size", 0),
                liquidity=trade.get("liquidity", 10000),
                volatility=trade.get("volatility", 0),
                is_popular_token=trade.get("is_popular_token", False),
                priority_fee=trade.get("priority_fee", 0),
            )
            results.append(result)
        return results

    def reset_seed(self, seed: int):
        """Reset the random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)
        # Recreate perturbation instances with new RNG
        self.__init__(self.config)
        self.config.seed = seed
