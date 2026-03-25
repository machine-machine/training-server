"""
Experiments module for A/B testing and experimentation.

Components:
- ABTestFramework: Full experiment management
- Traffic allocation and stopping rules
"""

from .ab_testing import (
    ABTestFramework,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    StoppingRule,
)

__all__ = [
    "ABTestFramework",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentStatus",
    "StoppingRule",
]
