"""
Optuna-based hyperparameter optimization for backtesting.

Provides Bayesian optimization with walk-forward cross-validation
to prevent overfitting and maximize trading performance.
"""

from .optuna_optimizer import (
    CVFold,
    ObjectiveMetric,
    OptimizationConfig,
    OptimizationResult,
    OptunaOptimizer,
    WalkForwardCV,
)
from .search_space import (
    FULL_SEARCH_SPACE,
    RISK_SEARCH_SPACE,
    SIGNAL_SEARCH_SPACE,
    TRADING_SEARCH_SPACE,
    ParameterDefinition,
    ParameterType,
    SearchSpace,
)

__all__ = [
    # Search Space
    "SearchSpace",
    "ParameterDefinition",
    "ParameterType",
    "TRADING_SEARCH_SPACE",
    "RISK_SEARCH_SPACE",
    "SIGNAL_SEARCH_SPACE",
    "FULL_SEARCH_SPACE",
    # Optimizer
    "OptunaOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "WalkForwardCV",
    "CVFold",
    "ObjectiveMetric",
]
