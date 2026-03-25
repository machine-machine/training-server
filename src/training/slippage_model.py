"""
Slippage prediction model.

Learns the mapping from quoted slippage to realized slippage.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from scipy.optimize import minimize


@dataclass
class SlippageObservation:
    """A single slippage observation from a trade."""
    quoted_slippage_bps: float
    realized_slippage_bps: float
    liquidity_usd: float
    volume_usd: float
    volatility: float
    latency_ms: int


class SlippageModel:
    """
    Linear model for slippage prediction.

    realized = base + quoted * quote_coef + liquidity_coef / liquidity + volatility * vol_coef + mev_penalty
    """

    def __init__(self):
        # Default parameters
        self.base_bps = 50.0
        self.quote_coef = 1.2  # Multiplier on quoted slippage
        self.liquidity_coef = 10000.0  # Inverse relationship
        self.volatility_coef = 100.0
        self.mev_penalty_bps = 100.0
        self.latency_coef = 0.1  # bps per ms

    def predict(self, obs: SlippageObservation) -> float:
        """Predict realized slippage in bps."""
        predicted = (
            self.base_bps
            + obs.quoted_slippage_bps * self.quote_coef
            + self.liquidity_coef / max(obs.liquidity_usd, 1000)
            + obs.volatility * self.volatility_coef
            + self.mev_penalty_bps
            + obs.latency_ms * self.latency_coef
        )
        return max(0, predicted)

    def fit(self, observations: List[SlippageObservation]):
        """Fit model parameters to historical data."""
        if len(observations) < 10:
            return  # Not enough data

        def objective(params):
            self.base_bps = params[0]
            self.quote_coef = params[1]
            self.liquidity_coef = params[2]
            self.volatility_coef = params[3]
            self.mev_penalty_bps = params[4]
            self.latency_coef = params[5]

            errors = []
            for obs in observations:
                predicted = self.predict(obs)
                error = (predicted - obs.realized_slippage_bps) ** 2
                errors.append(error)

            return np.mean(errors)

        # Initial params
        x0 = [
            self.base_bps,
            self.quote_coef,
            self.liquidity_coef,
            self.volatility_coef,
            self.mev_penalty_bps,
            self.latency_coef,
        ]

        # Bounds
        bounds = [
            (0, 200),      # base_bps
            (0.5, 3.0),    # quote_coef
            (0, 50000),    # liquidity_coef
            (0, 500),      # volatility_coef
            (0, 500),      # mev_penalty_bps
            (0, 1.0),      # latency_coef
        ]

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

        # Update params
        self.base_bps = result.x[0]
        self.quote_coef = result.x[1]
        self.liquidity_coef = result.x[2]
        self.volatility_coef = result.x[3]
        self.mev_penalty_bps = result.x[4]
        self.latency_coef = result.x[5]

    def to_params(self) -> dict:
        """Export model parameters."""
        return {
            "slippage_base_bps": self.base_bps,
            "slippage_quote_coef": self.quote_coef,
            "slippage_liquidity_coef": self.liquidity_coef,
            "slippage_volatility_coef": self.volatility_coef,
            "mev_penalty_bps": self.mev_penalty_bps,
            "slippage_latency_coef": self.latency_coef,
        }
