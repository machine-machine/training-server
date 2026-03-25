"""
Student Distiller - Knowledge distillation from 50-feature teacher to 7-feature student.

Produces a logistic regression model compatible with HotPath's MlRuntime,
using teacher soft labels as training targets.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..learning.feature_engineering import FeatureIndex
from .teacher_trainer import TeacherResult

logger = logging.getLogger(__name__)

# 7 HotPath-compatible features
HOTPATH_FEATURES = [
    "liquidity_usd",
    "fdv_usd",
    "volume_24h",
    "holder_count",
    "top_holder_pct",
    "fraud_score",
    "token_age_hours",
]

# Mapping from HotPath feature name to ColdPath FeatureIndex
FEATURE_MAP = {
    "liquidity_usd": FeatureIndex.POOL_TVL_SOL,  # idx 0
    "fdv_usd": FeatureIndex.TOTAL_SUPPLY,  # idx 12 (proxy)
    "volume_24h": FeatureIndex.VOLUME_ACCELERATION,  # idx 26 (proxy)
    "holder_count": FeatureIndex.HOLDER_COUNT_UNIQUE,  # idx 15
    "top_holder_pct": FeatureIndex.TOP_10_HOLDER_CONCENTRATION,  # idx 14
    "fraud_score": FeatureIndex.RUG_PULL_ML_SCORE,  # idx 42
    "token_age_hours": FeatureIndex.POOL_AGE_SECONDS,  # idx 1 (÷3600)
}


@dataclass
class FeatureTransformSpec:
    """Specification for a feature transform matching proto FeatureTransform."""

    feature_name: str
    transform_type: str  # "log", "zscore", "none"
    param1: float = 0.0  # mean (zscore) or 0 (log)
    param2: float = 0.0  # std (zscore) or 0 (log)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "transform_type": self.transform_type,
            "param1": self.param1,
            "param2": self.param2,
        }


@dataclass
class PlattCalibration:
    """Platt scaling calibration parameters."""

    temperature: float = 1.0
    platt_a: float = 0.0  # sigmoid: P = 1 / (1 + exp(A*f + B))
    platt_b: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
        }


@dataclass
class StudentConfig:
    """Configuration for student distillation."""

    # Validation thresholds
    min_correlation: float = 0.85
    max_calibration_error: float = 0.05

    # Training
    validation_split: float = 0.2
    random_state: int = 42

    # Confidence threshold for HotPath inference
    confidence_threshold: float = 0.5

    # Transform preferences per feature
    # log: good for skewed distributions (liquidity, volume, fdv)
    # zscore: good for roughly normal distributions
    default_transforms: Dict[str, str] = field(default_factory=lambda: {
        "liquidity_usd": "log",
        "fdv_usd": "log",
        "volume_24h": "log",
        "holder_count": "log",
        "top_holder_pct": "zscore",
        "fraud_score": "none",
        "token_age_hours": "log",
    })


@dataclass
class StudentResult:
    """Result of student distillation."""

    weights: np.ndarray  # 7 logistic regression coefficients
    bias: float  # Intercept term
    transforms: List[FeatureTransformSpec]
    calibration: PlattCalibration
    metrics: Dict[str, float]
    feature_names: List[str]
    confidence_threshold: float


class StudentDistiller:
    """Distill teacher ensemble into 7-feature logistic regression student."""

    def __init__(self, config: Optional[StudentConfig] = None):
        self.config = config or StudentConfig()

    def distill(self, teacher: TeacherResult) -> Optional[StudentResult]:
        """Distill teacher to student model.

        Args:
            teacher: TeacherResult with trained model, features, and soft labels.

        Returns:
            StudentResult or None if quality checks fail.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import log_loss

        # 1. Extract 7-feature subset from full 50-feature matrix
        X_student, unit_conversion_applied = self._extract_student_features(
            teacher.features
        )

        if X_student is None:
            logger.error("Failed to extract student features")
            return None

        # 2. Compute per-feature transforms and store params
        transforms, X_transformed = self._compute_transforms(X_student)

        # 3. Split into train/validation
        n_samples = len(X_transformed)
        indices = np.arange(n_samples)
        train_idx, val_idx = train_test_split(
            indices,
            test_size=self.config.validation_split,
            random_state=self.config.random_state,
        )

        X_train = X_transformed[train_idx]
        X_val = X_transformed[val_idx]
        soft_train = teacher.soft_labels[train_idx]
        soft_val = teacher.soft_labels[val_idx]
        hard_val = teacher.labels[val_idx]

        # 4. Fit LogisticRegression using soft labels
        # Use sample_weight = abs(soft_label - 0.5) to emphasize confident examples
        sample_weights = np.abs(soft_train - 0.5) * 2  # Scale to [0, 1]
        sample_weights = np.clip(sample_weights, 0.01, 1.0)  # Avoid zero weights

        # Convert soft labels to hard for sklearn (it doesn't support soft targets directly)
        # But weight samples by teacher confidence
        hard_train = (soft_train > 0.5).astype(int)

        model = LogisticRegression(
            max_iter=1000,
            random_state=self.config.random_state,
            C=1.0,
        )
        model.fit(X_train, hard_train, sample_weight=sample_weights)

        # Get student predictions on validation set
        student_probs = model.predict_proba(X_val)[:, 1]

        # 5. Platt scaling on validation set
        calibration = self._platt_scale(student_probs, hard_val)

        # 6. Quality metrics
        # Student vs teacher correlation
        correlation = np.corrcoef(student_probs, soft_val)[0, 1]

        # Calibration error (ECE)
        calibration_error = self._expected_calibration_error(
            student_probs, hard_val, n_bins=10
        )

        # Agreement rate
        student_binary = (student_probs > 0.5).astype(int)
        teacher_binary = (soft_val > 0.5).astype(int)
        agreement = np.mean(student_binary == teacher_binary)

        # Validation accuracy against hard labels
        accuracy = np.mean(student_binary == hard_val)

        metrics = {
            "correlation": float(correlation),
            "calibration_error": float(calibration_error),
            "agreement": float(agreement),
            "accuracy": float(accuracy),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "teacher_auc": float(teacher.metrics.auc_roc),
        }

        logger.info(
            f"Student distilled: correlation={correlation:.3f}, "
            f"calibration_error={calibration_error:.4f}, "
            f"agreement={agreement:.3f}, accuracy={accuracy:.3f}"
        )

        # 7. Validation gate
        if correlation < self.config.min_correlation:
            logger.warning(
                f"Student rejected: correlation {correlation:.3f} < "
                f"{self.config.min_correlation}"
            )
            return None

        if calibration_error > self.config.max_calibration_error:
            logger.warning(
                f"Student rejected: calibration_error {calibration_error:.4f} > "
                f"{self.config.max_calibration_error}"
            )
            return None

        return StudentResult(
            weights=model.coef_[0],
            bias=float(model.intercept_[0]),
            transforms=transforms,
            calibration=calibration,
            metrics=metrics,
            feature_names=HOTPATH_FEATURES,
            confidence_threshold=self.config.confidence_threshold,
        )

    def _extract_student_features(
        self, full_features: np.ndarray
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Extract 7-feature subset from 50-feature matrix with unit conversions."""
        n_samples = full_features.shape[0]
        n_student = len(HOTPATH_FEATURES)
        X_student = np.zeros((n_samples, n_student))

        for i, feat_name in enumerate(HOTPATH_FEATURES):
            source_idx = int(FEATURE_MAP[feat_name])

            if source_idx >= full_features.shape[1]:
                logger.warning(
                    f"Feature index {source_idx} out of bounds for {feat_name}"
                )
                return None, False

            values = full_features[:, source_idx].copy()

            # Unit conversions
            if feat_name == "token_age_hours":
                # POOL_AGE_SECONDS -> hours
                values = values / 3600.0

            X_student[:, i] = values

        return X_student, True

    def _compute_transforms(
        self, X: np.ndarray
    ) -> Tuple[List[FeatureTransformSpec], np.ndarray]:
        """Compute per-feature transforms and return transformed data."""
        transforms = []
        X_out = np.zeros_like(X)

        for i, feat_name in enumerate(HOTPATH_FEATURES):
            transform_type = self.config.default_transforms.get(feat_name, "none")
            col = X[:, i].copy()

            if transform_type == "log":
                # Log1p transform (handles zeros)
                col_transformed = np.log1p(np.abs(col))
                transforms.append(FeatureTransformSpec(
                    feature_name=feat_name,
                    transform_type="log",
                    param1=0.0,
                    param2=0.0,
                ))
                X_out[:, i] = col_transformed

            elif transform_type == "zscore":
                mean = float(np.mean(col))
                std = float(np.std(col))
                if std < 1e-10:
                    std = 1.0
                col_transformed = (col - mean) / std
                transforms.append(FeatureTransformSpec(
                    feature_name=feat_name,
                    transform_type="zscore",
                    param1=mean,
                    param2=std,
                ))
                X_out[:, i] = col_transformed

            else:
                # No transform
                transforms.append(FeatureTransformSpec(
                    feature_name=feat_name,
                    transform_type="none",
                    param1=0.0,
                    param2=0.0,
                ))
                X_out[:, i] = col

        return transforms, X_out

    def _platt_scale(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> PlattCalibration:
        """Fit Platt scaling: P(y=1|f) = 1 / (1 + exp(A*f + B))."""
        from sklearn.linear_model import LogisticRegression

        # Fit logistic regression on raw probabilities
        platt_model = LogisticRegression(max_iter=1000)
        platt_model.fit(probs.reshape(-1, 1), labels)

        # Extract A and B: sigmoid(A*f + B)
        platt_a = float(platt_model.coef_[0][0])
        platt_b = float(platt_model.intercept_[0])

        # Temperature from Platt coefficients
        temperature = 1.0 / max(abs(platt_a), 1e-6)

        return PlattCalibration(
            temperature=temperature,
            platt_a=platt_a,
            platt_b=platt_b,
        )

    def _expected_calibration_error(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(probs)

        for i in range(n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if i == n_bins - 1:
                mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])

            count = np.sum(mask)
            if count == 0:
                continue

            avg_confidence = np.mean(probs[mask])
            avg_accuracy = np.mean(labels[mask])
            ece += (count / total) * abs(avg_confidence - avg_accuracy)

        return ece
