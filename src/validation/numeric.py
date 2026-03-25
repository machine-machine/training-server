"""
Numeric validation utilities for safe calculations.

Provides functions for validating and sanitizing numeric inputs
to prevent NaN propagation, division by zero, and overflow issues.
"""

import math
import logging
from typing import Optional, TypeVar, Union

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T', float, np.ndarray)


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


def is_valid_float(value: float, allow_zero: bool = False) -> bool:
    """Check if a float is valid (not NaN, not Inf, optionally not zero).

    Args:
        value: The value to check
        allow_zero: Whether to allow zero values

    Returns:
        True if the value is valid, False otherwise
    """
    if value is None:
        return False
    if np.isnan(value) or np.isinf(value):
        return False
    if not allow_zero and value == 0.0:
        return False
    return True


def safe_divide(
    numerator: T,
    denominator: T,
    default: T = 0.0,
    log_warning: bool = False,
) -> T:
    """Safe division with fallback for invalid denominators.

    Handles:
    - Division by zero
    - NaN/Inf inputs
    - Array operations

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value if division fails
        log_warning: Whether to log a warning on fallback

    Returns:
        The division result or default value
    """
    # Handle numpy arrays
    if isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray):
        num = np.asarray(numerator)
        den = np.asarray(denominator)
        # Create mask for valid divisions
        valid_mask = (den != 0) & np.isfinite(num) & np.isfinite(den)
        result = np.where(valid_mask, num / np.where(den != 0, den, 1), default)
        return result

    # Handle scalar values
    if not is_valid_float(denominator, allow_zero=False):
        if log_warning:
            logger.warning(f"safe_divide: invalid denominator {denominator}, using default {default}")
        return default

    if not is_valid_float(numerator, allow_zero=True):
        if log_warning:
            logger.warning(f"safe_divide: invalid numerator {numerator}, using default {default}")
        return default

    result = numerator / denominator

    if not is_valid_float(result, allow_zero=True):
        if log_warning:
            logger.warning(f"safe_divide: result {result} is invalid, using default {default}")
        return default

    return result


def safe_log(
    value: T,
    base: Optional[float] = None,
    default: float = 0.0,
    epsilon: float = 1e-10,
) -> T:
    """Safe logarithm with protection against invalid inputs.

    Args:
        value: The value to take logarithm of
        base: The logarithm base (None for natural log)
        default: Default value for invalid inputs
        epsilon: Small value to add to zero inputs

    Returns:
        The logarithm or default value
    """
    if isinstance(value, np.ndarray):
        # Ensure positive values for log
        safe_value = np.where(value <= 0, epsilon, value)
        if base is None:
            return np.log(safe_value)
        return np.log(safe_value) / np.log(base)

    # Scalar case
    if not is_valid_float(value, allow_zero=True) or value <= 0:
        return default

    if base is None:
        return math.log(max(value, epsilon))
    return math.log(max(value, epsilon), base)


def clamp(
    value: T,
    min_val: float,
    max_val: float,
    default: Optional[float] = None,
) -> T:
    """Clamp value to range, with optional default for invalid inputs.

    Args:
        value: The value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        default: Default value for NaN/Inf inputs (uses min_val if None)

    Returns:
        The clamped value
    """
    if isinstance(value, np.ndarray):
        # Handle invalid values in arrays
        result = np.clip(value, min_val, max_val)
        if default is not None:
            result = np.where(np.isfinite(value), result, default)
        return result

    # Scalar case
    if not is_valid_float(value, allow_zero=True):
        return default if default is not None else min_val

    return max(min_val, min(max_val, value))


class NumericValidator:
    """Reusable numeric validation with detailed error reporting.

    This class provides validation methods that can be extended
    for domain-specific validation needs.

    # Extensibility
    Subclass and override methods for custom validation logic:

    ```python
    class TradingValidator(NumericValidator):
        def validate_position_size(self, size: float) -> float:
            validated = self.validate_positive(size, "position_size")
            if validated > 100.0:
                raise ValidationError("Position size exceeds maximum")
            return validated
    ```
    """

    def __init__(self, strict: bool = False):
        """Initialize validator.

        Args:
            strict: If True, raise exceptions. If False, log warnings and sanitize.
        """
        self.strict = strict

    def validate_finite(self, value: float, name: str = "value") -> float:
        """Validate that value is finite (not NaN or Inf)."""
        if np.isnan(value):
            msg = f"{name} is NaN"
            if self.strict:
                raise ValidationError(msg)
            logger.warning(msg)
            return 0.0

        if np.isinf(value):
            msg = f"{name} is infinite"
            if self.strict:
                raise ValidationError(msg)
            logger.warning(msg)
            return 0.0

        return value

    def validate_positive(self, value: float, name: str = "value") -> float:
        """Validate that value is positive (> 0)."""
        value = self.validate_finite(value, name)
        if value <= 0:
            msg = f"{name} must be positive, got {value}"
            if self.strict:
                raise ValidationError(msg)
            logger.warning(msg)
            return 1e-10  # Small positive epsilon

        return value

    def validate_non_negative(self, value: float, name: str = "value") -> float:
        """Validate that value is non-negative (>= 0)."""
        value = self.validate_finite(value, name)
        if value < 0:
            msg = f"{name} must be non-negative, got {value}"
            if self.strict:
                raise ValidationError(msg)
            logger.warning(msg)
            return 0.0

        return value

    def validate_probability(self, value: float, name: str = "probability") -> float:
        """Validate that value is a valid probability [0, 1]."""
        value = self.validate_finite(value, name)
        if value < 0.0 or value > 1.0:
            msg = f"{name} must be in [0, 1], got {value}"
            if self.strict:
                raise ValidationError(msg)
            logger.warning(msg)
            return clamp(value, 0.0, 1.0)

        return value

    def validate_percentage(self, value: float, name: str = "percentage") -> float:
        """Validate that value is a valid percentage [0, 100]."""
        value = self.validate_finite(value, name)
        if value < 0.0 or value > 100.0:
            msg = f"{name} must be in [0, 100], got {value}"
            if self.strict:
                raise ValidationError(msg)
            logger.warning(msg)
            return clamp(value, 0.0, 100.0)

        return value

    def validate_in_range(
        self,
        value: float,
        min_val: float,
        max_val: float,
        name: str = "value",
    ) -> float:
        """Validate that value is within a specific range."""
        value = self.validate_finite(value, name)
        if value < min_val:
            msg = f"{name} ({value}) is below minimum ({min_val})"
            if self.strict:
                raise ValidationError(msg)
            logger.warning(msg)
            return min_val

        if value > max_val:
            msg = f"{name} ({value}) exceeds maximum ({max_val})"
            if self.strict:
                raise ValidationError(msg)
            logger.warning(msg)
            return max_val

        return value

    def validate_array(
        self,
        arr: np.ndarray,
        name: str = "array",
        min_length: int = 1,
        allow_nan: bool = False,
    ) -> np.ndarray:
        """Validate numpy array for common issues."""
        if arr is None:
            msg = f"{name} is None"
            if self.strict:
                raise ValidationError(msg)
            logger.warning(msg)
            return np.array([])

        if len(arr) < min_length:
            msg = f"{name} must have at least {min_length} elements, has {len(arr)}"
            if self.strict:
                raise ValidationError(msg)
            logger.warning(msg)

        if not allow_nan and np.any(np.isnan(arr)):
            nan_count = np.sum(np.isnan(arr))
            msg = f"{name} contains {nan_count} NaN values"
            if self.strict:
                raise ValidationError(msg)
            logger.warning(msg)
            # Replace NaN with zeros
            arr = np.nan_to_num(arr, nan=0.0)

        if np.any(np.isinf(arr)):
            inf_count = np.sum(np.isinf(arr))
            msg = f"{name} contains {inf_count} infinite values"
            if self.strict:
                raise ValidationError(msg)
            logger.warning(msg)
            # Replace Inf with large finite values
            arr = np.nan_to_num(arr, posinf=1e10, neginf=-1e10)

        return arr
