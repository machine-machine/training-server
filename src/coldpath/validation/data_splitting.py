"""
Time-aware data splitting utilities for ML training.

Provides standardized train/val/test splitting that avoids temporal leakage
in time-series data (trading outcomes, price history, etc.).

# Why Time-Aware Splitting?
Random train_test_split() shuffles data, which can cause:
- Training on future data to predict past data (temporal leakage)
- Overly optimistic validation metrics
- Models that don't generalize to real-time inference

# Usage
```python
from coldpath.validation import time_aware_split

# Simple split
X_train, X_val, X_test, y_train, y_val, y_test = time_aware_split(
    X, y, train_frac=0.7, val_frac=0.15
)

# With timestamps for explicit ordering
X_train, X_val, X_test = time_aware_split(X, timestamps=ts)
```
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SplitConfig:
    """Configuration for time-aware data splitting."""

    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    shuffle: bool = False  # Default: no shuffle for time-series data
    stratify: bool = False  # Stratification doesn't work well with time series
    random_seed: int = 42

    def __post_init__(self):
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")


def time_aware_split(
    X: NDArray,
    y: NDArray | None = None,
    timestamps: NDArray | None = None,
    config: SplitConfig | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray | None, NDArray | None, NDArray | None]:
    """
    Split data into train/val/test sets in time order.

    This is the recommended splitting method for trading ML models to avoid
    temporal leakage.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Optional labels (n_samples,)
        timestamps: Optional timestamps for explicit ordering. If None, uses
                   current row order (assumes data is already time-sorted)
        config: SplitConfig with train/val/test fractions

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)

    Example:
        >>> X = np.random.randn(1000, 50)
        >>> y = np.random.randint(0, 2, 1000)
        >>> X_train, X_val, X_test, y_train, y_val, y_test = time_aware_split(X, y)
        >>> X_train.shape[0]  # ~700 samples
        700
    """
    config = config or SplitConfig()
    n_samples = X.shape[0]

    # Sort by timestamp if provided
    if timestamps is not None:
        sort_idx = np.argsort(timestamps)
        X = X[sort_idx]
        if y is not None:
            y = y[sort_idx]

    # Calculate split indices
    train_end = int(n_samples * config.train_fraction)
    val_end = train_end + int(n_samples * config.val_fraction)

    # Split data
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    if y is not None:
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
    else:
        y_train = y_val = y_test = None

    return X_train, X_val, X_test, y_train, y_val, y_test


def time_series_cv_split(
    n_samples: int,
    n_splits: int = 5,
) -> list[tuple[NDArray, NDArray]]:
    """
    Generate time-series cross-validation indices.

    Uses sklearn's TimeSeriesSplit under the hood but returns index arrays
    for consistency with the rest of the codebase.

    Args:
        n_samples: Total number of samples
        n_splits: Number of CV splits

    Returns:
        List of (train_indices, val_indices) tuples

    Example:
        >>> splits = time_series_cv_split(1000, n_splits=5)
        >>> len(splits)
        5
        >>> splits[0][0].shape[0] < splits[-1][0].shape[0]  # Training grows each fold
        True
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    indices = []

    for train_idx, val_idx in tscv.split(np.arange(n_samples)):
        indices.append((train_idx, val_idx))

    return indices


def validate_no_temporal_leakage(
    train_timestamps: NDArray,
    test_timestamps: NDArray,
) -> bool:
    """
    Validate that no temporal leakage exists between train and test sets.

    Args:
        train_timestamps: Timestamps of training samples
        test_timestamps: Timestamps of test samples

    Returns:
        True if no leakage (all train timestamps < all test timestamps)

    Raises:
        ValueError: If temporal leakage is detected
    """
    if len(train_timestamps) == 0 or len(test_timestamps) == 0:
        return True

    max_train = np.max(train_timestamps)
    min_test = np.min(test_timestamps)

    if max_train >= min_test:
        leakage_count = np.sum(train_timestamps >= min_test)
        raise ValueError(
            f"Temporal leakage detected: {leakage_count} training samples "
            f"have timestamps >= earliest test sample. "
            f"Max train: {max_train}, Min test: {min_test}"
        )

    return True


def get_recommended_split_for_model(model_type: str) -> SplitConfig:
    """
    Get recommended split configuration for a model type.

    Args:
        model_type: One of "profitability", "fraud", "slippage", "regime"

    Returns:
        SplitConfig with recommended settings
    """
    configs = {
        "profitability": SplitConfig(
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            shuffle=False,
        ),
        "fraud": SplitConfig(
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            shuffle=False,  # Fraud patterns evolve over time
        ),
        "slippage": SplitConfig(
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            shuffle=False,
        ),
        "regime": SplitConfig(
            train_fraction=0.60,  # Regime detection needs more test data
            val_fraction=0.20,
            test_fraction=0.20,
            shuffle=False,
        ),
    }

    if model_type not in configs:
        raise ValueError(
            f"Unknown model type: {model_type}. Expected one of {list(configs.keys())}"
        )

    return configs[model_type]
