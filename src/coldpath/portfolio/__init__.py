"""
Portfolio management module for multi-strategy optimization.

Components:
- MultiStrategyPortfolio: Portfolio optimization across strategies
- Risk parity allocation
- Correlation analysis
- Hedging optimization
- Correlation-aware position limits
"""

from .correlation_limits import (
    CorrelationAwareLimiter,
    CorrelationConfig,
    ExposureResult,
    Position,
    TokenCategory,
    get_correlation_limiter,
)
from .multi_strategy import (
    AllocationResult,
    CorrelationAnalyzer,
    MultiStrategyPortfolio,
    PortfolioConfig,
    RiskParityAllocator,
    StrategyAllocation,
)
from .spatial_pricing import (
    SpatialArbitragePricingModel,
    SpatialPricingConfig,
    SpatialPricingResult,
)

__all__ = [
    "MultiStrategyPortfolio",
    "PortfolioConfig",
    "StrategyAllocation",
    "AllocationResult",
    "RiskParityAllocator",
    "CorrelationAnalyzer",
    "SpatialArbitragePricingModel",
    "SpatialPricingConfig",
    "SpatialPricingResult",
    # Correlation limits
    "CorrelationAwareLimiter",
    "CorrelationConfig",
    "Position",
    "ExposureResult",
    "TokenCategory",
    "get_correlation_limiter",
]
