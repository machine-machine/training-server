"""
Monte Carlo robustness testing for trading strategies.

Generates confidence intervals and prevents overfitting through
stochastic perturbation of market conditions.
"""

from .numba_mc import (
    MonteCarloConfig,
    MonteCarloEngine,
    MonteCarloResult,
    RiskMetrics,
    compute_probability_of_ruin,
    compute_sharpe_confidence_interval,
    compute_var_cvar,
    run_monte_carlo_simulation,
)
from .perturbation import (
    FailureRatePerturbation,
    LatencyPerturbation,
    LiquidityPerturbation,
    MEVImpactPerturbation,
    PerturbationConfig,
    PerturbationResult,
    Perturbator,
    SlippagePerturbation,
)

__all__ = [
    # Perturbations
    "PerturbationConfig",
    "PerturbationResult",
    "Perturbator",
    "SlippagePerturbation",
    "LatencyPerturbation",
    "FailureRatePerturbation",
    "MEVImpactPerturbation",
    "LiquidityPerturbation",
    # Monte Carlo Engine
    "MonteCarloEngine",
    "MonteCarloConfig",
    "MonteCarloResult",
    "RiskMetrics",
    "run_monte_carlo_simulation",
    "compute_var_cvar",
    "compute_probability_of_ruin",
    "compute_sharpe_confidence_interval",
]
