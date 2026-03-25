"""
Validation utilities for synthetic training datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .synthetic_data import SyntheticDataset


@dataclass
class ValidationResult:
    passed: bool
    score: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class SyntheticDataValidator:
    """Basic sanity checks for generated synthetic datasets."""

    def validate(self, dataset: SyntheticDataset) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        n = len(dataset)
        if n == 0:
            errors.append("dataset_empty")
            return ValidationResult(passed=False, score=0.0, errors=errors, warnings=warnings)

        if dataset.features_50.shape[1] != 50:
            errors.append("invalid_feature_width")

        if len(dataset.signal_labels) != n:
            errors.append("signal_label_length_mismatch")
        if len(dataset.future_returns_t15) != n:
            errors.append("future_return_length_mismatch")
        if len(dataset.rug_labels) != n:
            errors.append("rug_label_length_mismatch")
        if len(dataset.price_sequences) == 0:
            errors.append("missing_price_sequences")

        if not np.isfinite(dataset.features_50).all():
            errors.append("non_finite_features")

        label_rate = float(np.mean(dataset.signal_labels))
        if label_rate < 0.05 or label_rate > 0.95:
            warnings.append("label_imbalance")

        rug_rate = float(np.mean(dataset.rug_labels))
        if rug_rate <= 0.0:
            warnings.append("no_rug_examples")

        # Score favors balanced labels and finite, shape-valid arrays.
        balance_score = 1.0 - abs(label_rate - 0.5) * 2.0
        quality_score = max(0.0, min(1.0, balance_score))
        if errors:
            quality_score = 0.0

        return ValidationResult(
            passed=len(errors) == 0,
            score=quality_score,
            errors=errors,
            warnings=warnings,
        )
