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

from .bounds import (
    CalibrationBounds,
    MLModelBounds,
    TradingBounds,
    ValidationBounds,
)
from .data_splitting import (
    SplitConfig,
    get_recommended_split_for_model,
    time_aware_split,
    time_series_cv_split,
    validate_no_temporal_leakage,
)
from .drift_detector import (
    DriftAlert,
    DriftDetector,
    DriftDetectorConfig,
    DriftSeverity,
    DriftState,
    DriftType,
    PredictionRecord,
)
from .numeric import (
    NumericValidator,
    ValidationError,
    clamp,
    is_valid_float,
    safe_divide,
    safe_log,
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
    # Data splitting
    "time_aware_split",
    "time_series_cv_split",
    "validate_no_temporal_leakage",
    "get_recommended_split_for_model",
    "SplitConfig",
]
