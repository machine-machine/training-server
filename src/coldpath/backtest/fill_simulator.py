"""
Fill simulator for backtesting.

Models realistic fills based on AMM mechanics.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FillResult:
    """Result of a simulated fill."""

    filled: bool
    fill_price: float | None
    amount_out: float | None
    slippage_bps: int | None
    failure_reason: str | None


class FillSimulator:
    """
    Simulates trade fills using AMM mechanics.

    Uses constant product formula with MEV penalty.
    """

    def __init__(
        self,
        mev_penalty_bps: float = 30.0,  # Jito bundles drastically reduce MEV extraction
        inclusion_prob: float = 0.95,  # Jito achieves 95%+ inclusion rate
        pool_fee_bps: float = 25.0,  # Raydium standard fee tier
    ):
        self.mev_penalty_bps = mev_penalty_bps
        self.inclusion_prob = inclusion_prob
        self.pool_fee_bps = pool_fee_bps

    def simulate_buy(
        self,
        amount_in_sol: float,
        liquidity_sol: float,
        liquidity_tokens: float,
        quoted_slippage_bps: int,
    ) -> FillResult:
        """
        Simulate a buy order using constant product formula.

        x * y = k
        """
        # Check inclusion
        if np.random.random() > self.inclusion_prob:
            return FillResult(
                filled=False,
                fill_price=None,
                amount_out=None,
                slippage_bps=None,
                failure_reason="Transaction not included",
            )

        # Guard against invalid liquidity values (FIXED: prevent division by zero/inf)
        if liquidity_tokens <= 0 or liquidity_sol <= 0:
            return FillResult(
                filled=False,
                fill_price=None,
                amount_out=None,
                slippage_bps=None,
                failure_reason="Insufficient liquidity",
            )

        # Constant product AMM (apply pool fee on input)
        amount_in_after_fee = amount_in_sol * (1 - self.pool_fee_bps / 10000)
        k = liquidity_sol * liquidity_tokens
        new_sol = liquidity_sol + amount_in_after_fee
        new_tokens = k / new_sol
        tokens_out = liquidity_tokens - new_tokens

        # Guard against zero tokens out
        if tokens_out <= 0:
            return FillResult(
                filled=False,
                fill_price=None,
                amount_out=None,
                slippage_bps=None,
                failure_reason="Trade too large for liquidity",
            )

        # Calculate actual price
        actual_price = amount_in_sol / tokens_out
        spot_price = liquidity_sol / liquidity_tokens

        # Price impact in bps
        price_impact_bps = int(abs(actual_price / spot_price - 1) * 10000)

        # Add MEV penalty
        mev_impact = self._sample_mev_penalty()
        total_slippage_bps = price_impact_bps + mev_impact

        if total_slippage_bps > quoted_slippage_bps:
            return FillResult(
                filled=False,
                fill_price=None,
                amount_out=None,
                slippage_bps=None,
                failure_reason="Slippage exceeded",
            )

        # Adjust tokens out for MEV
        mev_factor = 1 - mev_impact / 10000
        tokens_out_after_mev = tokens_out * mev_factor
        final_price = amount_in_sol / tokens_out_after_mev if tokens_out_after_mev > 0 else 0

        return FillResult(
            filled=True,
            fill_price=final_price,
            amount_out=tokens_out_after_mev,
            slippage_bps=total_slippage_bps,
            failure_reason=None,
        )

    def simulate_sell(
        self,
        amount_tokens: float,
        liquidity_sol: float,
        liquidity_tokens: float,
        quoted_slippage_bps: int,
    ) -> FillResult:
        """Simulate a sell order."""
        # Check inclusion
        if np.random.random() > self.inclusion_prob:
            return FillResult(
                filled=False,
                fill_price=None,
                amount_out=None,
                slippage_bps=None,
                failure_reason="Transaction not included",
            )

        # Guard against invalid liquidity values (FIXED: prevent division by zero/inf)
        if liquidity_tokens <= 0 or liquidity_sol <= 0:
            return FillResult(
                filled=False,
                fill_price=None,
                amount_out=None,
                slippage_bps=None,
                failure_reason="Insufficient liquidity",
            )

        # Constant product AMM (apply pool fee on input)
        amount_in_after_fee = amount_tokens * (1 - self.pool_fee_bps / 10000)
        k = liquidity_sol * liquidity_tokens
        new_tokens = liquidity_tokens + amount_in_after_fee
        new_sol = k / new_tokens
        sol_out = liquidity_sol - new_sol

        # Guard against zero SOL out
        if sol_out <= 0:
            return FillResult(
                filled=False,
                fill_price=None,
                amount_out=None,
                slippage_bps=None,
                failure_reason="Trade too large for liquidity",
            )

        # Calculate actual price
        actual_price = amount_tokens / sol_out
        spot_price = liquidity_tokens / liquidity_sol

        # Price impact in bps
        price_impact_bps = int(abs(actual_price / spot_price - 1) * 10000) if spot_price > 0 else 0

        # Add MEV penalty
        mev_impact = self._sample_mev_penalty()
        total_slippage_bps = price_impact_bps + mev_impact

        if total_slippage_bps > quoted_slippage_bps:
            return FillResult(
                filled=False,
                fill_price=None,
                amount_out=None,
                slippage_bps=None,
                failure_reason="Slippage exceeded",
            )

        # Adjust SOL out for MEV
        mev_factor = 1 - mev_impact / 10000
        sol_out_after_mev = sol_out * mev_factor
        final_price = amount_tokens / sol_out_after_mev if sol_out_after_mev > 0 else 0

        return FillResult(
            filled=True,
            fill_price=final_price,
            amount_out=sol_out_after_mev,
            slippage_bps=total_slippage_bps,
            failure_reason=None,
        )

    def _sample_mev_penalty(self) -> int:
        """Sample MEV penalty from exponential distribution."""
        # Exponential with mean = mev_penalty_bps
        penalty = np.random.exponential(self.mev_penalty_bps)
        return int(min(penalty, 500))  # Cap at 5%
