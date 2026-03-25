"""Shadow Trading Optimizer Module

Provides parameter optimization for shadow trading strategies including:
- ShadowOptimizer: Main optimizer class for parameter tuning
- ParameterTuner: Core optimization algorithms
- ShadowMetrics: Metrics tracking and analysis
- OutcomeTracker: Track shadow trade outcomes
"""

from .metrics import (
    PerformanceReport,
    ShadowMetrics,
    ShadowTradeRecord,
)
from .optimizer import (
    OptimizationResult,
    ShadowOptimizer,
    ShadowOptimizerConfig,
)
from .outcome_tracker import (
    OutcomeTracker,
    ShadowOutcome,
)
from .parameter_tuner import (
    ParameterTuner,
    TradingParams,
    TuningResult,
)

__all__ = [
    # Optimizer
    "ShadowOptimizer",
    "ShadowOptimizerConfig",
    "OptimizationResult",
    # Tuner
    "ParameterTuner",
    "TuningResult",
    "TradingParams",
    # Metrics
    "ShadowMetrics",
    "ShadowTradeRecord",
    "PerformanceReport",
    # Outcome
    "OutcomeTracker",
    "ShadowOutcome",
]
