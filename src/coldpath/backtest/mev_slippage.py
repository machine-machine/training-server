"""
MEV and Slippage Modeler - Realistic trading cost simulation.

Models real-world trading costs on Solana DEXs:
- Slippage based on trade size vs liquidity
- MEV (Maximal Extractable Value) exposure
- Front-running risk
- Priority fee impacts
- Market impact estimation

This makes backtests more realistic by accounting for costs that
simple fixed-slippage models miss.

Usage:
    from coldpath.backtest.mev_slippage import MEVSlippageModeler

    modeler = MEVSlippageModeler()

    # Estimate total execution cost
    result = modeler.estimate_execution_cost(
        trade_size_sol=0.1,
        pool_liquidity_usd=50000,
        volatility_pct=25.0,
        is_buy=True,
    )

    print(f"Total slippage: {result.total_slippage_bps:.1f} bps")
    print(f"MEV exposure: {result.mev_cost_bps:.1f} bps")
"""

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TradeType(Enum):
    """Type of trade."""

    BUY = "buy"
    SELL = "sell"


class MarketCondition(Enum):
    """Current market condition."""

    NORMAL = "normal"
    VOLATILE = "volatile"
    EXTREME = "extreme"
    MEMECOIN_MANIA = "memecoin_mania"


@dataclass
class LiquidityProfile:
    """Liquidity characteristics of a pool."""

    total_liquidity_usd: float
    token_reserve: float
    sol_reserve: float
    fee_bps: int = 300  # 3% typical for memecoins

    @property
    def price(self) -> float:
        """Current price (SOL per token)."""
        if self.token_reserve > 0:
            return self.sol_reserve / self.token_reserve
        return 0.0

    def get_output_amount(self, input_sol: float) -> float:
        """Calculate output tokens for input SOL (constant product AMM)."""
        # x * y = k
        # output = (y * input) / (x + input)
        k = self.sol_reserve * self.token_reserve
        new_sol = self.sol_reserve + input_sol * (1 - self.fee_bps / 10000)
        new_token = k / new_sol
        return self.token_reserve - new_token

    def get_price_impact(self, input_sol: float) -> float:
        """Calculate price impact as percentage."""
        if self.sol_reserve == 0:
            return 100.0
        impact = (input_sol / self.sol_reserve) * 100
        return min(impact, 100.0)


@dataclass
class ExecutionCostResult:
    """Result of execution cost estimation."""

    # Slippage components (in basis points)
    base_slippage_bps: float  # AMM slippage from trade size
    volatility_slippage_bps: float  # Additional slippage from volatility
    liquidity_slippage_bps: float  # Additional for low liquidity

    # MEV components
    frontrun_risk_bps: float  # Expected loss from front-running
    sandwich_risk_bps: float  # Expected loss from sandwich attacks
    mev_cost_bps: float  # Total MEV exposure

    # Priority/fees
    priority_fee_bps: float  # Recommended priority fee
    jito_tip_bps: float  # Jito tip for MEV protection

    # Totals
    total_slippage_bps: float
    total_cost_bps: float

    # Execution quality
    expected_fill_rate: float  # 0-1 probability of fill
    recommended_max_slippage_bps: float  # Max slippage to set
    execution_confidence: float  # 0-1 confidence in estimate

    # Breakdown
    cost_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_slippage_bps": self.base_slippage_bps,
            "volatility_slippage_bps": self.volatility_slippage_bps,
            "liquidity_slippage_bps": self.liquidity_slippage_bps,
            "frontrun_risk_bps": self.frontrun_risk_bps,
            "sandwich_risk_bps": self.sandwich_risk_bps,
            "mev_cost_bps": self.mev_cost_bps,
            "priority_fee_bps": self.priority_fee_bps,
            "total_slippage_bps": self.total_slippage_bps,
            "total_cost_bps": self.total_cost_bps,
            "expected_fill_rate": self.expected_fill_rate,
            "execution_confidence": self.execution_confidence,
        }


@dataclass
class MEVSlippageConfig:
    """Configuration for MEV/slippage modeling."""

    # Base slippage model
    base_slippage_bps: float = 50  # Minimum slippage even for small trades
    slippage_multiplier: float = 1.0  # Overall multiplier for calibration

    # Volatility adjustment
    volatility_base_bps: float = 10  # Additional bps per 1% volatility
    extreme_volatility_threshold: float = 50.0  # % volatility for extreme

    # Liquidity adjustment
    min_liquidity_usd: float = 10000  # Below this = very high slippage
    liquidity_sensitivity: float = 0.5  # How much liquidity affects slippage

    # MEV parameters
    frontrun_base_risk_bps: float = 5  # Base front-running risk
    sandwich_base_risk_bps: float = 10  # Base sandwich attack risk
    mev_multiplier_memecoin: float = 3.0  # Extra MEV risk for memecoins
    mev_multiplier_volatile: float = 2.0  # Extra MEV risk in volatile markets

    # Priority fees
    base_priority_fee_bps: float = 5  # Normal priority fee
    high_priority_fee_bps: float = 20  # High priority for fast fills
    jito_tip_base_bps: float = 10  # Jito tip for MEV protection

    # Fill rate model
    base_fill_rate: float = 0.95  # Base probability of fill
    low_liquidity_fill_penalty: float = 0.1  # Penalty for low liquidity

    # SOL price for USD<->SOL conversions (was hardcoded at 150.0)
    sol_price_usd: float = 150.0  # Update this to current market price

    # Random seed for reproducibility
    random_seed: int | None = 42


class MEVSlippageModeler:
    """Model realistic trading costs including MEV and slippage.

    Provides estimates for:
    - AMM slippage based on trade size
    - Market impact
    - MEV exposure (front-running, sandwich attacks)
    - Priority fee recommendations
    - Fill rate estimation

    Example:
        modeler = MEVSlippageModeler()

        result = modeler.estimate_execution_cost(
            trade_size_sol=0.1,
            pool_liquidity_usd=50000,
            volatility_pct=25.0,
            is_memecoin=True,
        )

        # Use in backtest
        effective_price = requested_price * (1 + result.total_cost_bps / 10000)
    """

    def __init__(self, config: MEVSlippageConfig | None = None):
        """Initialize the MEV/slippage modeler.

        Args:
            config: Configuration parameters
        """
        self.config = config or MEVSlippageConfig()
        self._rng = random.Random(self.config.random_seed)

    def estimate_execution_cost(
        self,
        trade_size_sol: float,
        pool_liquidity_usd: float,
        volatility_pct: float = 20.0,
        is_buy: bool = True,
        is_memecoin: bool = False,
        market_condition: MarketCondition | None = None,
        pending_tx_count: int = 100,  # Network congestion
    ) -> ExecutionCostResult:
        """Estimate total execution cost for a trade.

        Args:
            trade_size_sol: Size of trade in SOL
            pool_liquidity_usd: Pool liquidity in USD
            volatility_pct: Current volatility percentage
            is_buy: True for buy, False for sell
            is_memecoin: Whether this is a memecoin
            market_condition: Current market condition
            pending_tx_count: Number of pending transactions

        Returns:
            ExecutionCostResult with detailed cost breakdown
        """
        # Determine market condition
        if market_condition is None:
            if volatility_pct > 50:
                market_condition = MarketCondition.EXTREME
            elif volatility_pct > 30 or is_memecoin:
                market_condition = MarketCondition.VOLATILE
            elif is_memecoin and volatility_pct > 20:
                market_condition = MarketCondition.MEMECOIN_MANIA
            else:
                market_condition = MarketCondition.NORMAL

        # 1. Base slippage (AMM mechanics)
        base_slippage = self._calculate_base_slippage(trade_size_sol, pool_liquidity_usd)

        # 2. Volatility slippage
        vol_slippage = self._calculate_volatility_slippage(volatility_pct, market_condition)

        # 3. Liquidity slippage
        liq_slippage = self._calculate_liquidity_slippage(pool_liquidity_usd, trade_size_sol)

        # 4. MEV costs
        frontrun_risk, sandwich_risk = self._calculate_mev_risk(
            trade_size_sol,
            pool_liquidity_usd,
            is_memecoin,
            market_condition,
        )
        mev_cost = frontrun_risk + sandwich_risk

        # 5. Priority fees
        priority_fee, jito_tip = self._calculate_fees(
            is_memecoin,
            market_condition,
            pending_tx_count,
        )

        # Total slippage (without fees)
        total_slippage = (
            base_slippage + vol_slippage + liq_slippage + mev_cost
        ) * self.config.slippage_multiplier

        # Total cost (including fees)
        total_cost = total_slippage + priority_fee + jito_tip

        # Fill rate estimation
        fill_rate = self._estimate_fill_rate(
            pool_liquidity_usd,
            trade_size_sol,
            volatility_pct,
            market_condition,
        )

        # Recommended max slippage (for transaction)
        recommended_max = total_slippage * 1.5  # 50% buffer

        # Execution confidence
        confidence = self._calculate_confidence(
            pool_liquidity_usd,
            volatility_pct,
            market_condition,
        )

        return ExecutionCostResult(
            base_slippage_bps=base_slippage,
            volatility_slippage_bps=vol_slippage,
            liquidity_slippage_bps=liq_slippage,
            frontrun_risk_bps=frontrun_risk,
            sandwich_risk_bps=sandwich_risk,
            mev_cost_bps=mev_cost,
            priority_fee_bps=priority_fee,
            jito_tip_bps=jito_tip,
            total_slippage_bps=total_slippage,
            total_cost_bps=total_cost,
            expected_fill_rate=fill_rate,
            recommended_max_slippage_bps=recommended_max,
            execution_confidence=confidence,
            cost_breakdown={
                "amm_slippage": base_slippage,
                "volatility_impact": vol_slippage,
                "liquidity_impact": liq_slippage,
                "mev_frontrun": frontrun_risk,
                "mev_sandwich": sandwich_risk,
                "priority_fee": priority_fee,
                "jito_tip": jito_tip,
            },
        )

    def _calculate_base_slippage(
        self,
        trade_size_sol: float,
        liquidity_usd: float,
    ) -> float:
        """Calculate base AMM slippage."""
        # Convert liquidity to SOL equivalent using configurable SOL price
        sol_price = self.config.sol_price_usd
        liquidity_sol = liquidity_usd / sol_price

        if liquidity_sol <= 0:
            return 1000.0  # 10% max slippage

        # Price impact = trade_size / liquidity * 100 (in percent)
        # Convert to bps (* 100)
        impact_pct = (trade_size_sol / liquidity_sol) * 100
        impact_bps = impact_pct * 100

        # Add base slippage
        total = self.config.base_slippage_bps + impact_bps

        # Cap at reasonable maximum
        return min(total, 5000.0)  # Max 50% slippage

    def _calculate_volatility_slippage(
        self,
        volatility_pct: float,
        condition: MarketCondition,
    ) -> float:
        """Calculate additional slippage from volatility."""
        # Linear relationship with volatility
        base = volatility_pct * self.config.volatility_base_bps

        # Extra for extreme conditions
        if condition == MarketCondition.EXTREME:
            base *= 2.0
        elif condition == MarketCondition.MEMECOIN_MANIA:
            base *= 1.5

        return base

    def _calculate_liquidity_slippage(
        self,
        liquidity_usd: float,
        trade_size_sol: float,
    ) -> float:
        """Calculate additional slippage for low liquidity."""
        if liquidity_usd >= 100000:
            return 0.0  # No penalty for high liquidity

        if liquidity_usd < self.config.min_liquidity_usd:
            # Very low liquidity - high penalty
            return 200.0  # 2% extra slippage

        # Graduated penalty
        ratio = liquidity_usd / self.config.min_liquidity_usd
        penalty = (1 - ratio) * 100 * self.config.liquidity_sensitivity

        return penalty

    def _calculate_mev_risk(
        self,
        trade_size_sol: float,
        liquidity_usd: float,
        is_memecoin: bool,
        condition: MarketCondition,
    ) -> tuple[float, float]:
        """Calculate MEV exposure (front-running, sandwich attacks)."""
        # Base risks
        frontrun = self.config.frontrun_base_risk_bps
        sandwich = self.config.sandwich_base_risk_bps

        # Scale by trade size (larger trades = more attractive target)
        size_factor = min(trade_size_sol / 0.5, 3.0)  # Cap at 3x
        frontrun *= size_factor
        sandwich *= size_factor

        # Extra for memecoins (more MEV bots watching)
        if is_memecoin:
            frontrun *= self.config.mev_multiplier_memecoin
            sandwich *= self.config.mev_multiplier_memecoin

        # Extra for volatile markets
        if condition in [MarketCondition.VOLATILE, MarketCondition.EXTREME]:
            frontrun *= self.config.mev_multiplier_volatile
            sandwich *= self.config.mev_multiplier_volatile

        # Low liquidity = more MEV opportunity
        if liquidity_usd < 50000:
            frontrun *= 1.5
            sandwich *= 1.5

        return frontrun, sandwich

    def _calculate_fees(
        self,
        is_memecoin: bool,
        condition: MarketCondition,
        pending_tx_count: int,
    ) -> tuple[float, float]:
        """Calculate priority fees and Jito tips."""
        # Base priority fee
        if pending_tx_count > 500 or condition == MarketCondition.EXTREME:
            priority_fee = self.config.high_priority_fee_bps
        else:
            priority_fee = self.config.base_priority_fee_bps

        # Jito tip (for MEV protection)
        jito_tip = self.config.jito_tip_base_bps

        if is_memecoin:
            jito_tip *= 2.0  # More protection needed

        return priority_fee, jito_tip

    def _estimate_fill_rate(
        self,
        liquidity_usd: float,
        trade_size_sol: float,
        volatility_pct: float,
        condition: MarketCondition,
    ) -> float:
        """Estimate probability of successful fill."""
        rate = self.config.base_fill_rate

        # Penalty for low liquidity
        if liquidity_usd < self.config.min_liquidity_usd:
            rate -= self.config.low_liquidity_fill_penalty

        # Penalty for volatility
        if volatility_pct > 30:
            rate -= 0.05
        if volatility_pct > 50:
            rate -= 0.10

        # Penalty for extreme conditions
        if condition == MarketCondition.EXTREME:
            rate -= 0.10

        return max(0.5, min(1.0, rate))  # Clamp between 50% and 100%

    def _calculate_confidence(
        self,
        liquidity_usd: float,
        volatility_pct: float,
        condition: MarketCondition,
    ) -> float:
        """Calculate confidence in the estimate."""
        # Start with high confidence
        confidence = 0.9

        # Lower confidence for edge cases
        if liquidity_usd < self.config.min_liquidity_usd:
            confidence -= 0.2

        if volatility_pct > 40:
            confidence -= 0.1

        if condition == MarketCondition.MEMECOIN_MANIA:
            confidence -= 0.15

        return max(0.5, confidence)

    def simulate_execution(
        self,
        trade_size_sol: float,
        pool_liquidity_usd: float,
        requested_slippage_bps: float,
        volatility_pct: float = 20.0,
        is_memecoin: bool = False,
    ) -> tuple[bool, float, str]:
        """Simulate trade execution with realistic outcomes.

        Args:
            trade_size_sol: Trade size in SOL
            pool_liquidity_usd: Pool liquidity in USD
            requested_slippage_bps: User's max slippage setting
            volatility_pct: Current volatility
            is_memecoin: Is this a memecoin trade

        Returns:
            Tuple of (success, actual_slippage_bps, reason)
        """
        cost = self.estimate_execution_cost(
            trade_size_sol=trade_size_sol,
            pool_liquidity_usd=pool_liquidity_usd,
            volatility_pct=volatility_pct,
            is_memecoin=is_memecoin,
        )

        # Check if fill is likely
        if self._rng.random() > cost.expected_fill_rate:
            return False, 0, "no_fill"

        # Check if within slippage tolerance
        if cost.total_slippage_bps > requested_slippage_bps:
            if self._rng.random() < 0.3:  # 30% chance of still filling at higher slippage
                actual = requested_slippage_bps + self._rng.uniform(0, 50)
                return True, actual, "high_slippage_fill"
            return False, cost.total_slippage_bps, "slippage_exceeded"

        # Successful fill with some variance
        variance = self._rng.gauss(0, cost.total_slippage_bps * 0.1)
        actual = max(0, cost.total_slippage_bps + variance)

        return True, actual, "success"

    def get_optimal_slippage_setting(
        self,
        trade_size_sol: float,
        pool_liquidity_usd: float,
        volatility_pct: float = 20.0,
        is_memecoin: bool = False,
        aggressiveness: str = "balanced",  # conservative, balanced, aggressive
    ) -> dict[str, Any]:
        """Get optimal slippage setting for a trade.

        Args:
            trade_size_sol: Trade size in SOL
            pool_liquidity_usd: Pool liquidity
            volatility_pct: Current volatility
            is_memecoin: Is this a memecoin
            aggressiveness: How aggressive to be with fills

        Returns:
            Dict with recommended settings
        """
        cost = self.estimate_execution_cost(
            trade_size_sol=trade_size_sol,
            pool_liquidity_usd=pool_liquidity_usd,
            volatility_pct=volatility_pct,
            is_memecoin=is_memecoin,
        )

        base_recommended = cost.recommended_max_slippage_bps

        if aggressiveness == "conservative":
            # Lower slippage, accept more failed fills
            recommended = base_recommended * 0.8
            expected_fill_rate = cost.expected_fill_rate * 0.85
        elif aggressiveness == "aggressive":
            # Higher slippage, prioritize fills
            recommended = base_recommended * 1.3
            expected_fill_rate = min(1.0, cost.expected_fill_rate * 1.1)
        else:  # balanced
            recommended = base_recommended
            expected_fill_rate = cost.expected_fill_rate

        return {
            "recommended_slippage_bps": recommended,
            "expected_fill_rate": expected_fill_rate,
            "mev_protection_recommended": cost.mev_cost_bps > 20,
            "use_jito": is_memecoin or cost.mev_cost_bps > 15,
            "total_expected_cost_bps": cost.total_cost_bps,
            "aggressiveness": aggressiveness,
        }
