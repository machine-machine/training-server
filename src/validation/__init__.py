"""
Centralized validation utilities for ColdPath.

This module provides reusable validation functions for numeric inputs,
data quality checks, and plausibility validation across the ML pipeline.

# Usage
```python
from coldpath.validation import NumericValidator, ValidationBounds

validator = NumericValidator()
validated = validator.validate_positive(price, "price")

bounds = ValidationBounds.trading()
size = bounds.clamp_position_size(raw_size)
```

# Future Upgrades
- Add domain-specific validators (token, pool, etc.)
- Add async validation for I/O-bound checks
- Add validation caching for repeated checks
- Add validation metrics/telemetry
"""

from .numeric import (
    NumericValidator,
    ValidationError,
    safe_divide,
    safe_log,
    clamp,
    is_valid_float,
)

from .bounds import (
    ValidationBounds,
    TradingBounds,
    CalibrationBounds,
    MLModelBounds,
)

from .drift_detector import (
    DriftDetector,
    DriftDetectorConfig,
    DriftAlert,
    DriftType,
    DriftSeverity,
    DriftState,
    PredictionRecord,
)

__all__ = [
    # Numeric validation
    "NumericValidator",
    "ValidationError",
    "safe_divide",
    "safe_log",
    "clamp",
    "is_valid_float",
    # Bounds
    "ValidationBounds",
    "TradingBounds",
    "CalibrationBounds",
    "MLModelBounds",
    # Drift detection
    "DriftDetector",
    "DriftDetectorConfig",
    "DriftAlert",
    "DriftType",
    "DriftSeverity",
    "DriftState",
    "PredictionRecord",
]
