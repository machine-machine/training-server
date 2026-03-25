"""
Monte Carlo robustness testing for trading strategies.

Generates confidence intervals and prevents overfitting through
stochastic perturbation of market conditions.
"""

from .perturbation import (
    PerturbationConfig,
    PerturbationResult,
    Perturbator,
    SlippagePerturbation,
    LatencyPerturbation,
    FailureRatePerturbation,
    MEVImpactPerturbation,
    LiquidityPerturbation,
)

from .numba_mc import (
    MonteCarloEngine,
    MonteCarloConfig,
    MonteCarloResult,
    RiskMetrics,
    run_monte_carlo_simulation,
    compute_var_cvar,
    compute_probability_of_ruin,
    compute_sharpe_confidence_interval,
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
