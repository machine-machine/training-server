"""
Synthetic Data Validation and Quality Checks

Ensures generated synthetic data has realistic statistical properties
and is suitable for training ML models.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from .synthetic_data import SyntheticDataset

logger = logging.getLogger(__name__)


def sanitize_features(matrix: np.ndarray, clip_value: float = 1e10) -> np.ndarray:
    """Replace NaN/Inf with safe defaults and clip extreme values.

    This should be called before training to prevent crashes from invalid data.

    Args:
        matrix: Feature matrix to sanitize
        clip_value: Maximum absolute value to clip to (default 1e10)

    Returns:
        Sanitized copy of the matrix
    """
    # Replace NaN with 0
    sanitized = np.nan_to_num(matrix, nan=0.0, posinf=clip_value, neginf=-clip_value)
    # Clip to prevent extreme values
    sanitized = np.clip(sanitized, -clip_value, clip_value)
    return sanitized


@dataclass
class ValidationResult:
    """Result of data validation."""

    passed: bool
    warnings: list[str]
    errors: list[str]
    metrics: dict[str, float]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "warnings": self.warnings,
            "errors": self.errors,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
        }


class SyntheticDataValidator:
    """Validate synthetic datasets for quality and realism."""

    def __init__(
        self,
        min_samples: int = 500,
        max_rug_rate: float = 0.15,
        min_class_balance: float = 0.05,
    ):
        self.min_samples = min_samples
        self.max_rug_rate = max_rug_rate
        self.min_class_balance = min_class_balance

    def validate(self, dataset: SyntheticDataset) -> ValidationResult:
        """Run comprehensive validation on synthetic dataset."""
        warnings = []
        errors = []
        metrics = {}
        recommendations = []

        # 1. Size validation
        if len(dataset) < self.min_samples:
            errors.append(f"Dataset too small: {len(dataset)} < {self.min_samples}")

        # 2. Price validation
        price_metrics = self._validate_prices(dataset.prices)
        metrics.update(price_metrics)

        if price_metrics.get("negative_prices", 0) > 0:
            errors.append("Found negative prices in dataset")

        if price_metrics.get("price_volatility", 0) > 2.0:
            warnings.append("Very high price volatility (>200%)")

        # 3. Feature validation
        feature_metrics = self._validate_features(dataset.features_50)
        metrics.update(feature_metrics)

        if feature_metrics.get("nan_count", 0) > 0:
            errors.append(f"Found {feature_metrics['nan_count']} NaN values in features")

        if feature_metrics.get("inf_count", 0) > 0:
            errors.append(f"Found {feature_metrics['inf_count']} Inf values in features")

        # 4. Label validation
        label_metrics = self._validate_labels(
            signal_labels=dataset.signal_labels,
            rug_labels=dataset.rug_labels,
        )
        metrics.update(label_metrics)

        # Check class balance
        for class_name, fraction in label_metrics.get("signal_distribution", {}).items():
            if fraction < self.min_class_balance:
                warnings.append(f"Low {class_name} signal fraction: {fraction:.1%}")

        # Check rug rate
        rug_rate = label_metrics.get("rug_rate", 0)
        if rug_rate > self.max_rug_rate:
            warnings.append(f"High rug rate: {rug_rate:.1%} > {self.max_rug_rate:.1%}")

        # 5. Statistical properties
        stat_metrics = self._validate_statistics(dataset)
        metrics.update(stat_metrics)

        # 6. Recommendations
        if stat_metrics.get("kurtosis", 0) > 10:
            recommendations.append("Consider smoothing extreme price movements")

        if label_metrics.get("signal_distribution", {}).get("hold", 0) > 0.8:
            recommendations.append("Increase signal threshold to get more BUY/SELL labels")

        passed = len(errors) == 0

        return ValidationResult(
            passed=passed,
            warnings=warnings,
            errors=errors,
            metrics=metrics,
            recommendations=recommendations,
        )

    def _validate_prices(self, prices: np.ndarray) -> dict[str, float]:
        """Validate price series."""
        returns = np.diff(prices) / prices[:-1]

        return {
            "price_min": float(np.min(prices)),
            "price_max": float(np.max(prices)),
            "price_mean": float(np.mean(prices)),
            "price_std": float(np.std(prices)),
            "price_volatility": float(np.std(returns)),
            "negative_prices": int(np.sum(prices < 0)),
            "zero_prices": int(np.sum(prices == 0)),
            "max_return": float(np.max(returns)),
            "min_return": float(np.min(returns)),
        }

    def _validate_features(self, features: np.ndarray) -> dict[str, float]:
        """Validate feature matrix."""
        return {
            "feature_shape": features.shape,
            "nan_count": int(np.sum(np.isnan(features))),
            "inf_count": int(np.sum(np.isinf(features))),
            "feature_mean": float(np.mean(features)),
            "feature_std": float(np.std(features)),
            "feature_min": float(np.min(features)),
            "feature_max": float(np.max(features)),
        }

    def _validate_labels(
        self,
        signal_labels: np.ndarray,
        rug_labels: np.ndarray,
    ) -> dict[str, Any]:
        """Validate label distributions."""
        n_total = len(signal_labels)

        signal_dist = {
            "sell": float(np.sum(signal_labels == 0) / n_total),
            "hold": float(np.sum(signal_labels == 1) / n_total),
            "buy": float(np.sum(signal_labels == 2) / n_total),
        }

        rug_rate = float(np.sum(rug_labels == 1) / n_total)

        return {
            "signal_distribution": signal_dist,
            "rug_rate": rug_rate,
            "rug_count": int(np.sum(rug_labels == 1)),
            "clean_count": int(np.sum(rug_labels == 0)),
        }

    def _validate_statistics(self, dataset: SyntheticDataset) -> dict[str, float]:
        """Validate statistical properties."""
        returns = np.diff(dataset.prices) / dataset.prices[:-1]

        # Normality test
        _, p_value = stats.normaltest(returns)

        return {
            "returns_mean": float(np.mean(returns)),
            "returns_std": float(np.std(returns)),
            "returns_skew": float(stats.skew(returns)),
            "returns_kurtosis": float(stats.kurtosis(returns)),
            "normality_p_value": float(p_value),
            "autocorrelation": float(np.corrcoef(returns[:-1], returns[1:])[0, 1]),
        }

    def compare_to_real_data(
        self,
        synthetic: SyntheticDataset,
        real_prices: np.ndarray,
    ) -> dict[str, Any]:
        """Compare synthetic data to real market data."""
        synth_returns = np.diff(synthetic.prices) / synthetic.prices[:-1]
        real_returns = np.diff(real_prices) / real_prices[:-1]

        comparison = {
            "volatility_ratio": float(np.std(synth_returns) / np.std(real_returns)),
            "mean_return_diff": float(np.mean(synth_returns) - np.mean(real_returns)),
            "skew_diff": float(stats.skew(synth_returns) - stats.skew(real_returns)),
            "kurtosis_diff": float(stats.kurtosis(synth_returns) - stats.kurtosis(real_returns)),
        }

        # KS test for distribution similarity
        ks_stat, ks_pvalue = stats.ks_2samp(synth_returns, real_returns)
        comparison["ks_statistic"] = float(ks_stat)
        comparison["ks_pvalue"] = float(ks_pvalue)

        return comparison
