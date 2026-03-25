"""
Profitability Learner - Train models to predict token profitability.

Uses LightGBM for fast inference with optional linear fallback.
Implements isotonic calibration for probability estimation.

M4 Pro Optimizations:
- MPS detection for unified memory architecture
- Optimal thread count for 10+ performance cores
- LightGBM force_col_wise for cache efficiency
- Scaled batch sizes for 48GB RAM
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

try:
    from sklearn.calibration import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import (
        cross_val_score,
        train_test_split,  # noqa: F401
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# Import centralized M4 Pro optimization utilities
from ..utils.m4_pro_optimizer import (
    ComputeDevice,
)
from ..utils.m4_pro_optimizer import (
    detect_cuda as _detect_cuda,
)
from ..utils.m4_pro_optimizer import (
    detect_mps as _detect_mps,
)
from ..utils.m4_pro_optimizer import (
    detect_optimal_device as _detect_optimal_device,
)
from ..utils.m4_pro_optimizer import (
    get_optimal_thread_count as _get_optimal_thread_count,
)

# Cached values for performance
_CUDA_AVAILABLE = _detect_cuda()
_MPS_AVAILABLE = _detect_mps()
# Convert ComputeDevice enum to string for backward compatibility
_detected_device = _detect_optimal_device()
_OPTIMAL_DEVICE = (
    _detected_device.value if isinstance(_detected_device, ComputeDevice) else str(_detected_device)
)
_OPTIMAL_THREADS = _get_optimal_thread_count()

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from ..validation.feature_audit import FeatureAuditor  # noqa: E402
from .feature_engineering import (  # noqa: E402
    FeatureEngineer,
    FeatureIndex,
    balance_dataset,
    compute_sample_weights,
)
from .mutual_information import MIConfig, MutualInformationSelector, SelectionMethod  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model type
    use_lightgbm: bool = True
    use_xgboost: bool = True
    use_linear_fallback: bool = True
    use_ensemble: bool = True

    # Ensemble weights
    ensemble_weights: dict[str, float] = field(
        default_factory=lambda: {
            "linear": 0.4,
            "lightgbm": 0.3,
            "xgboost": 0.3,
        }
    )

    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    n_cv_folds: int = 5

    # LightGBM parameters
    lgb_num_leaves: int = 64  # Increased from 31 for better accuracy
    lgb_max_depth: int = 8  # Increased from 5 for complex trading signals
    lgb_learning_rate: float = 0.1
    lgb_n_estimators: int = 200  # Increased, early stopping will prevent overfitting
    lgb_min_child_samples: int = 15  # Reduced from 20 for better signal capture
    lgb_reg_alpha: float = 0.1
    lgb_reg_lambda: float = 0.1

    # XGBoost parameters
    xgb_max_depth: int = 8  # Increased from 6 for complex trading signals
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 200  # Increased, early stopping will prevent overfitting
    xgb_min_child_weight: int = 1
    xgb_gamma: float = 0.0
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0

    # Early stopping parameters (20-30% faster training, prevents overfitting)
    early_stopping_rounds: int = 50
    use_early_stopping: bool = True

    # Sample weighting
    traded_weight: float = 1.0
    counterfactual_weight: float = 0.5
    balance_classes: bool = True
    max_class_ratio: float = 3.0

    # Minimum requirements
    min_samples: int = 100
    min_positive_samples: int = 20

    # Calibration
    use_isotonic_calibration: bool = True
    calibration_bins: int = 10

    # MI Feature Selection (reduces overfitting, improves generalization)
    use_mi_selection: bool = True
    mi_n_features: int = (
        35  # Select top 35 features from 50 (MI selector drops low-signal features)
    )
    mi_method: str = "mrmr"  # "mrmr", "jmi", or "cmim"
    mi_n_bins: int = 10  # Discretization bins for MI calculation

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "use_lightgbm": self.use_lightgbm,
            "use_xgboost": self.use_xgboost,
            "use_linear_fallback": self.use_linear_fallback,
            "use_ensemble": self.use_ensemble,
            "ensemble_weights": self.ensemble_weights,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "n_cv_folds": self.n_cv_folds,
            "lgb_num_leaves": self.lgb_num_leaves,
            "lgb_max_depth": self.lgb_max_depth,
            "lgb_learning_rate": self.lgb_learning_rate,
            "lgb_n_estimators": self.lgb_n_estimators,
            "xgb_max_depth": self.xgb_max_depth,
            "xgb_learning_rate": self.xgb_learning_rate,
            "xgb_n_estimators": self.xgb_n_estimators,
            "min_samples": self.min_samples,
            "min_positive_samples": self.min_positive_samples,
            "use_mi_selection": self.use_mi_selection,
            "mi_n_features": self.mi_n_features,
            "mi_method": self.mi_method,
        }


@dataclass
class TrainingMetrics:
    """Metrics from model training."""

    training_samples: int = 0
    test_samples: int = 0
    positive_samples: int = 0
    negative_samples: int = 0

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0

    cv_accuracy_mean: float = 0.0
    cv_accuracy_std: float = 0.0

    training_time_seconds: float = 0.0
    model_type: str = "unknown"

    feature_importance: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "training_samples": self.training_samples,
            "test_samples": self.test_samples,
            "positive_samples": self.positive_samples,
            "negative_samples": self.negative_samples,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "cv_accuracy_mean": self.cv_accuracy_mean,
            "cv_accuracy_std": self.cv_accuracy_std,
            "training_time_seconds": self.training_time_seconds,
            "model_type": self.model_type,
            "feature_importance": self.feature_importance,
        }


@dataclass
class LinearModelWeights:
    """Weights for linear model."""

    coefficients: list[float] = field(default_factory=lambda: [0.0] * 50)
    bias: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "coefficients": self.coefficients,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinearModelWeights:
        """Create from dictionary."""
        return cls(
            coefficients=data.get("coefficients", [0.0] * 50),
            bias=data.get("bias", 0.0),
        )


@dataclass
class LightGBMWeights:
    """Weights/config for LightGBM model."""

    model_json: str = ""
    learning_rate: float = 0.1
    base_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "model_json": self.model_json,
            "learning_rate": self.learning_rate,
            "base_score": self.base_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LightGBMWeights:
        """Create from dictionary."""
        return cls(
            model_json=data.get("model_json", ""),
            learning_rate=data.get("learning_rate", 0.1),
            base_score=data.get("base_score", 0.0),
        )


@dataclass
class XGBoostWeights:
    """Weights/config for XGBoost model."""

    model_json: str = ""
    learning_rate: float = 0.1
    base_score: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "model_json": self.model_json,
            "learning_rate": self.learning_rate,
            "base_score": self.base_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> XGBoostWeights:
        """Create from dictionary."""
        return cls(
            model_json=data.get("model_json", ""),
            learning_rate=data.get("learning_rate", 0.1),
            base_score=data.get("base_score", 0.5),
        )


@dataclass
class EnsembleWeights:
    """Weights for model ensemble."""

    linear: float = 0.4
    lightgbm: float = 0.3
    xgboost: float = 0.3

    def to_dict(self) -> dict[str, float]:
        """Export to dictionary."""
        return {
            "linear": self.linear,
            "lightgbm": self.lightgbm,
            "xgboost": self.xgboost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> EnsembleWeights:
        """Create from dictionary."""
        return cls(
            linear=data.get("linear", 0.4),
            lightgbm=data.get("lightgbm", 0.3),
            xgboost=data.get("xgboost", 0.3),
        )

    def normalize(self) -> EnsembleWeights:
        """Normalize weights to sum to 1.

        Returns equal weights (1/3 each) as fallback if total is invalid.
        Prevents division-by-zero crashes during training.
        """
        total = self.linear + self.lightgbm + self.xgboost
        if total > 0:
            return EnsembleWeights(
                linear=self.linear / total,
                lightgbm=self.lightgbm / total,
                xgboost=self.xgboost / total,
            )
        # Fallback to equal weights if total is invalid (prevents crashes)
        logger.warning("EnsembleWeights.normalize: total=%s <= 0, using equal weights", total)
        return EnsembleWeights(linear=1 / 3, lightgbm=1 / 3, xgboost=1 / 3)


class ProfitabilityLearner:
    """Train and manage profitability prediction models."""

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.feature_engineer = FeatureEngineer()

        # Models
        self._lgb_model = None
        self._xgb_model = None
        self._linear_model = None
        self._calibrator = None

        # State
        self._is_trained = False
        self._last_metrics: TrainingMetrics | None = None
        self._linear_weights: LinearModelWeights | None = None
        self._lgb_weights: LightGBMWeights | None = None
        self._xgb_weights: XGBoostWeights | None = None
        self._ensemble_weights: EnsembleWeights | None = None
        self._isotonic_points: list[tuple[float, float]] = []

        # MI Feature Selection
        self._mi_selector: MutualInformationSelector | None = None
        self._selected_feature_indices: np.ndarray | None = None
        self._selected_feature_names: list[str] | None = None

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained

    def record_outcome(
        self,
        features: dict[str, float],
        pnl_pct: float,
        confidence: float,
    ) -> None:
        """Record a trade outcome for incremental learning.

        Stores the outcome in a buffer. When enough outcomes accumulate,
        they can be used for retraining via `train()`.
        """
        if not hasattr(self, "_outcome_buffer"):
            self._outcome_buffer: list[dict[str, Any]] = []

        self._outcome_buffer.append(
            {
                "features": features,
                "pnl_pct": pnl_pct,
                "confidence": confidence,
                "profitable": pnl_pct > 0,
            }
        )

    def train(
        self,
        outcomes: list[dict[str, Any]],
        feature_engineer: FeatureEngineer | None = None,
    ) -> TrainingMetrics:
        """Train the profitability model.

        Args:
            outcomes: List of scan outcome dictionaries
            feature_engineer: Optional pre-fitted feature engineer

        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for training")

        start_time = datetime.now()
        metrics = TrainingMetrics()

        # Use provided or create new feature engineer
        if feature_engineer:
            self.feature_engineer = feature_engineer

        # Convert outcomes to features
        import pandas as pd

        df = pd.DataFrame(outcomes)
        metrics.training_samples = len(df)

        # Check minimum samples
        if len(df) < self.config.min_samples:
            logger.warning(f"Insufficient samples: {len(df)} < {self.config.min_samples}")
            return metrics

        if "pnl_pct" in df.columns:
            df["label"] = (df["pnl_pct"].fillna(0) > 0).astype(int)
        elif "was_profitable_counterfactual" in df.columns:
            df["label"] = df["was_profitable_counterfactual"].fillna(False).astype(int)

        audit = FeatureAuditor().audit(df, label_col="label")
        if audit.leakage_detected:
            logger.warning(f"Feature leakage detected: {audit.leaking_features}")
        elif audit.warnings:
            logger.warning(f"Feature audit warnings: {audit.warnings}")

        # CRITICAL: Split data BEFORE fitting feature engineer to prevent data leakage
        # The feature engineer computes normalization params (mean, std) which must
        # only be derived from training data, not test data

        # First, split the raw dataframe
        from sklearn.model_selection import train_test_split as sklearn_split

        train_indices, test_indices = sklearn_split(
            np.arange(len(df)),
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df["label"],
        )

        df_train = df.iloc[train_indices].reset_index(drop=True)
        df_test = df.iloc[test_indices].reset_index(drop=True)
        outcomes_train = [outcomes[i] for i in train_indices]
        outcomes_test = [outcomes[i] for i in test_indices]

        # Fit feature engineer ONLY on training data
        self.feature_engineer.fit(df_train)

        # Extract features from train and test separately
        X_train, y_train = self.feature_engineer.extract_batch(df_train)
        X_test, y_test = self.feature_engineer.extract_batch(df_test)

        # Apply MI feature selection (reduces overfitting, improves generalization)
        if self.config.use_mi_selection and X_train.shape[1] > self.config.mi_n_features:
            logger.info(
                f"Applying MI feature selection: {X_train.shape[1]} -> "
                f"{self.config.mi_n_features} features"
            )

            # Initialize MI selector
            mi_config = MIConfig(
                n_bins=self.config.mi_n_bins,
                min_features=self.config.mi_n_features,
                max_features=self.config.mi_n_features,
            )
            self._mi_selector = MutualInformationSelector(mi_config)

            # Map method string to enum
            method_map = {
                "mrmr": SelectionMethod.MRMR,
                "jmi": SelectionMethod.JMI,
                "cmim": SelectionMethod.CMIM,
            }
            mi_method = method_map.get(self.config.mi_method.lower(), SelectionMethod.MRMR)

            # Get feature names from FeatureIndex enum
            feature_names = [f.name for f in FeatureIndex]

            # Select features using training data only
            selection_result = self._mi_selector.select_features(
                X_train,
                y_train,
                feature_names=feature_names,
                method=mi_method,
            )

            # Store selected feature indices and names
            self._selected_feature_indices = np.array(selection_result.selected_indices)
            self._selected_feature_names = selection_result.selected_features

            top5 = ", ".join(self._selected_feature_names[:5])
            logger.info(
                f"Selected {len(self._selected_feature_indices)} features via "
                f"{self.config.mi_method}: {top5}..."
            )

            # Apply selection to train and test data
            X_train = X_train[:, self._selected_feature_indices]
            X_test = X_test[:, self._selected_feature_indices]
        else:
            self._selected_feature_indices = None
            self._selected_feature_names = None

        # Compute sample weights for train and test
        w_train = compute_sample_weights(
            outcomes_train,
            traded_weight=self.config.traded_weight,
            counterfactual_weight=self.config.counterfactual_weight,
        )
        compute_sample_weights(
            outcomes_test,
            traded_weight=self.config.traded_weight,
            counterfactual_weight=self.config.counterfactual_weight,
        )

        # Balance training data if configured (only balance train, not test)
        if self.config.balance_classes:
            X_train, y_train, w_train = balance_dataset(
                X_train, y_train, w_train, max_ratio=self.config.max_class_ratio
            )

        metrics.positive_samples = int(y_train.sum()) + int(y_test.sum())
        metrics.negative_samples = len(y_train) + len(y_test) - metrics.positive_samples

        pos_samples = int(y_train.sum())
        min_required = self.config.min_positive_samples
        if pos_samples < min_required:
            logger.warning(
                f"Insufficient positive samples in training: {pos_samples} < {min_required}"
            )
            return metrics

        metrics.test_samples = len(y_test)

        # Train models based on configuration
        trained_models = []

        # Always train linear as fallback
        if self.config.use_linear_fallback:
            self._train_linear(X_train, y_train, w_train)
            trained_models.append("linear")

        # Train LightGBM if available and configured
        if LIGHTGBM_AVAILABLE and self.config.use_lightgbm:
            self._train_lightgbm(X_train, y_train, w_train, X_test, y_test)
            trained_models.append("lightgbm")

        # Train XGBoost if available and configured
        if XGBOOST_AVAILABLE and self.config.use_xgboost:
            self._train_xgboost(X_train, y_train, w_train, X_test, y_test)
            trained_models.append("xgboost")

        if not trained_models:
            logger.error("No model type available for training")
            return metrics

        # Set model type based on what was trained
        if self.config.use_ensemble and len(trained_models) > 1:
            metrics.model_type = "ensemble"
            # Set ensemble weights
            self._ensemble_weights = EnsembleWeights(
                linear=self.config.ensemble_weights.get("linear", 0.4)
                if "linear" in trained_models
                else 0.0,
                lightgbm=self.config.ensemble_weights.get("lightgbm", 0.3)
                if "lightgbm" in trained_models
                else 0.0,
                xgboost=self.config.ensemble_weights.get("xgboost", 0.3)
                if "xgboost" in trained_models
                else 0.0,
            ).normalize()
        elif "xgboost" in trained_models:
            metrics.model_type = "xgboost"
        elif "lightgbm" in trained_models:
            metrics.model_type = "lightgbm"
        else:
            metrics.model_type = "linear"

        # Get predictions using ensemble or best model
        y_pred_proba = self._get_ensemble_predictions(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        metrics.accuracy = accuracy_score(y_test, y_pred)
        metrics.precision = precision_score(y_test, y_pred, zero_division=0)
        metrics.recall = recall_score(y_test, y_pred, zero_division=0)
        metrics.f1_score = f1_score(y_test, y_pred, zero_division=0)

        try:
            metrics.auc_roc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            metrics.auc_roc = 0.5

        # Cross-validation (FIXED: use X_train, y_train to prevent data leakage)
        # Using full X, y would leak test data into CV evaluation
        cv_scores = cross_val_score(
            self._lgb_model if self._lgb_model else self._linear_model,
            X_train,
            y_train,
            cv=self.config.n_cv_folds,
            scoring="accuracy",
        )
        metrics.cv_accuracy_mean = cv_scores.mean()
        metrics.cv_accuracy_std = cv_scores.std()

        # Feature importance
        feature_names = self.feature_engineer.get_feature_importance_names()
        if self._lgb_model:
            importance = self._lgb_model.feature_importances_
        else:
            importance = np.abs(self._linear_model.coef_[0])

        metrics.feature_importance = {
            name: float(imp) for name, imp in zip(feature_names, importance, strict=False)
        }

        # Calibration
        if self.config.use_isotonic_calibration:
            self._train_calibrator(X_test, y_test)

        # Extract weights for export
        self._extract_weights()

        metrics.training_time_seconds = (datetime.now() - start_time).total_seconds()
        self._last_metrics = metrics
        self._is_trained = True

        logger.info(
            f"Model trained: {metrics.model_type}, "
            f"accuracy={metrics.accuracy:.3f}, "
            f"AUC={metrics.auc_roc:.3f}, "
            f"samples={metrics.training_samples}"
        )

        return metrics

    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        weights: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ):
        """Train LightGBM model with optional early stopping.

        M4 Pro Optimizations:
        - n_jobs=-1 uses all cores for parallel training
        - force_col_wise=True improves cache efficiency on Apple Silicon
        - max_bin=255 balanced for M4 Pro's unified memory
        """
        self._lgb_model = lgb.LGBMClassifier(
            num_leaves=self.config.lgb_num_leaves,
            max_depth=self.config.lgb_max_depth,
            learning_rate=self.config.lgb_learning_rate,
            n_estimators=self.config.lgb_n_estimators,
            min_child_samples=self.config.lgb_min_child_samples,
            reg_alpha=self.config.lgb_reg_alpha,
            reg_lambda=self.config.lgb_reg_lambda,
            random_state=self.config.random_state,
            # Device selection: CUDA > MPS (via CPU) > CPU
            # Note: LightGBM doesn't support MPS directly, but CPU is fast on M4 Pro
            device="gpu" if _CUDA_AVAILABLE else "cpu",
            # M4 Pro optimization: use all cores
            n_jobs=_OPTIMAL_THREADS,
            # Apple Silicon optimization: force_col_wise improves cache efficiency
            force_col_wise=_MPS_AVAILABLE,
            # Balanced binning for M4 Pro's unified memory
            max_bin=255,
            verbose=-1,
        )

        # Use early stopping if validation data provided and enabled
        if self.config.use_early_stopping and X_val is not None and y_val is not None:
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=self.config.early_stopping_rounds, verbose=False
                ),
                lgb.log_evaluation(period=0),  # Suppress iteration logging
            ]
            self._lgb_model.fit(
                X_train,
                y_train,
                sample_weight=weights,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
            logger.debug(f"LightGBM stopped at iteration {self._lgb_model.best_iteration_}")
        else:
            self._lgb_model.fit(X_train, y_train, sample_weight=weights)

    def _train_linear(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        weights: np.ndarray,
    ):
        """Train logistic regression model."""
        self._linear_model = LogisticRegression(
            max_iter=1000,
            random_state=self.config.random_state,
        )
        self._linear_model.fit(X_train, y_train, sample_weight=weights)

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        weights: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ):
        """Train XGBoost model with optional early stopping.

        M4 Pro Optimizations:
        - n_jobs uses all cores for parallel histogram construction
        - tree_method="hist" is fastest on CPU (no CUDA on Mac)
        - max_bin=256 balanced for M4 Pro's unified memory
        """
        self._xgb_model = xgb.XGBClassifier(
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            n_estimators=self.config.xgb_n_estimators,
            min_child_weight=self.config.xgb_min_child_weight,
            gamma=self.config.xgb_gamma,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            reg_alpha=self.config.xgb_reg_alpha,
            reg_lambda=self.config.xgb_reg_lambda,
            random_state=self.config.random_state,
            eval_metric="logloss",
            # hist is fastest on CPU; gpu_hist requires CUDA
            tree_method="gpu_hist" if _CUDA_AVAILABLE else "hist",
            device="cuda" if _CUDA_AVAILABLE else "cpu",
            # M4 Pro optimization: use all cores for histogram construction
            n_jobs=_OPTIMAL_THREADS,
            # Balanced binning for M4 Pro's unified memory
            max_bin=256,
            verbosity=0,
            early_stopping_rounds=self.config.early_stopping_rounds
            if self.config.use_early_stopping
            else None,
        )

        # Use early stopping if validation data provided and enabled
        if self.config.use_early_stopping and X_val is not None and y_val is not None:
            self._xgb_model.fit(
                X_train,
                y_train,
                sample_weight=weights,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            logger.debug(f"XGBoost stopped at iteration {self._xgb_model.best_iteration}")
        else:
            self._xgb_model.fit(X_train, y_train, sample_weight=weights)

    def _get_ensemble_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from ensemble or best available model."""
        predictions = []
        weights = []

        # Get linear predictions
        if self._linear_model is not None:
            linear_pred = self._linear_model.predict_proba(X)[:, 1]
            predictions.append(linear_pred)
            w = self._ensemble_weights.linear if self._ensemble_weights else 0.4
            weights.append(w)

        # Get LightGBM predictions (must use predict_proba for probabilities)
        if self._lgb_model is not None:
            lgb_pred = self._lgb_model.predict_proba(X)[:, 1]
            predictions.append(lgb_pred)
            w = self._ensemble_weights.lightgbm if self._ensemble_weights else 0.3
            weights.append(w)

        # Get XGBoost predictions
        if self._xgb_model is not None:
            xgb_pred = self._xgb_model.predict_proba(X)[:, 1]
            predictions.append(xgb_pred)
            w = self._ensemble_weights.xgboost if self._ensemble_weights else 0.3
            weights.append(w)

        if not predictions:
            raise RuntimeError("No models available for prediction")

        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        ensemble_pred = np.zeros(len(X))
        for pred, w in zip(predictions, weights, strict=False):
            ensemble_pred += w * pred

        return ensemble_pred

    def _train_calibrator(self, X_test: np.ndarray, y_test: np.ndarray):
        """Train isotonic calibrator."""
        if self._lgb_model:
            raw_proba = self._lgb_model.predict_proba(X_test)[:, 1]
        else:
            raw_proba = self._linear_model.predict_proba(X_test)[:, 1]

        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(raw_proba, y_test)

        # Extract calibration points for Hot Path
        # Use deciles of the raw probability distribution
        percentiles = np.linspace(0, 100, self.config.calibration_bins + 1)
        raw_points = np.percentile(raw_proba, percentiles)
        calibrated_points = self._calibrator.predict(raw_points)

        self._isotonic_points = [
            (float(r), float(c)) for r, c in zip(raw_points, calibrated_points, strict=False)
        ]

    def _extract_weights(self):
        """Extract model weights for export to Hot Path."""
        # Linear weights
        if self._linear_model:
            self._linear_weights = LinearModelWeights(
                coefficients=self._linear_model.coef_[0].tolist(),
                bias=float(self._linear_model.intercept_[0]),
            )
        elif self._lgb_model:
            # Use feature importance as pseudo-coefficients for linear fallback
            importance = self._lgb_model.feature_importances_
            normalized = importance / (importance.sum() + 1e-10)
            self._linear_weights = LinearModelWeights(
                coefficients=normalized.tolist(),
                bias=0.0,
            )
        elif self._xgb_model:
            # Use XGBoost feature importance for linear fallback
            importance = self._xgb_model.feature_importances_
            normalized = importance / (importance.sum() + 1e-10)
            self._linear_weights = LinearModelWeights(
                coefficients=normalized.tolist(),
                bias=0.0,
            )

        # LightGBM weights
        if self._lgb_model:
            model_json = self._lgb_model.booster_.dump_model()
            self._lgb_weights = LightGBMWeights(
                model_json=json.dumps(model_json),
                learning_rate=self.config.lgb_learning_rate,
                base_score=0.0,
            )

        # XGBoost weights
        if self._xgb_model:
            # XGBoost can export to JSON format
            model_json = self._xgb_model.get_booster().save_raw(raw_format="json").decode("utf-8")
            self._xgb_weights = XGBoostWeights(
                model_json=model_json,
                learning_rate=self.config.xgb_learning_rate,
                base_score=0.5,  # XGBoost default base score
            )

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict profitability probability.

        Args:
            features: Feature matrix (N x num_features)

        Returns:
            Tuple of (raw probabilities, calibrated probabilities)
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        # Apply MI feature selection if enabled
        if self._selected_feature_indices is not None:
            features = features[:, self._selected_feature_indices]

        if self._lgb_model:
            raw_proba = self._lgb_model.predict_proba(features)[:, 1]
        else:
            raw_proba = self._linear_model.predict_proba(features)[:, 1]

        if self._calibrator:
            calibrated_proba = self._calibrator.predict(raw_proba)
        else:
            calibrated_proba = raw_proba

        return raw_proba, calibrated_proba

    def get_export_data(self) -> dict[str, Any]:
        """Get model data for export to Hot Path."""
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        # Determine model type
        if self._ensemble_weights and (self._lgb_model or self._xgb_model):
            model_type = "ensemble"
        elif self._xgb_model:
            model_type = "xgboost"
        elif self._lgb_model:
            model_type = "lightgbm"
        else:
            model_type = "linear"

        export = {
            "model_type": model_type,
            "version": int(datetime.now().timestamp()),
            "trained_at_ms": int(datetime.now().timestamp() * 1000),
        }

        # Linear weights (always included as fallback)
        if self._linear_weights:
            export["linear"] = self._linear_weights.to_dict()

        # LightGBM weights
        if self._lgb_weights:
            export["lightgbm"] = self._lgb_weights.to_dict()

        # XGBoost weights
        if self._xgb_weights:
            export["xgboost"] = self._xgb_weights.to_dict()

        # Ensemble weights
        if self._ensemble_weights:
            export["ensemble_weights"] = self._ensemble_weights.to_dict()

        # Calibration points
        if self._isotonic_points:
            export["isotonic_points"] = self._isotonic_points

        # Normalization parameters
        export["normalization"] = self.feature_engineer.export_params()

        # MI Feature Selection indices (for Hot Path to apply same selection)
        if self._selected_feature_indices is not None:
            export["mi_selection"] = {
                "indices": self._selected_feature_indices.tolist(),
                "feature_names": self._selected_feature_names,
                "method": self.config.mi_method,
                "n_features": len(self._selected_feature_indices),
            }

        # Metrics
        if self._last_metrics:
            export["metrics"] = self._last_metrics.to_dict()

        return export

    def load_from_export(self, data: dict[str, Any]):
        """Load model from exported data."""
        if "normalization" in data:
            self.feature_engineer.import_params(data["normalization"])

        if "linear" in data:
            self._linear_weights = LinearModelWeights.from_dict(data["linear"])

        if "lightgbm" in data:
            self._lgb_weights = LightGBMWeights.from_dict(data["lightgbm"])

        if "isotonic_points" in data:
            self._isotonic_points = data["isotonic_points"]

        # Load MI feature selection indices
        if "mi_selection" in data:
            mi_data = data["mi_selection"]
            self._selected_feature_indices = np.array(mi_data["indices"])
            self._selected_feature_names = mi_data.get("feature_names", [])
            logger.info(
                f"Loaded MI feature selection: {len(self._selected_feature_indices)} features"
            )
        else:
            self._selected_feature_indices = None
            self._selected_feature_names = None

        if "metrics" in data:
            self._last_metrics = TrainingMetrics(**data["metrics"])

        self._is_trained = True

    @property
    def last_metrics(self) -> TrainingMetrics | None:
        """Get metrics from last training."""
        return self._last_metrics

    def optimize_hyperparameters(
        self,
        outcomes: list[dict[str, Any]],
        n_trials: int = 50,
        timeout_seconds: int | None = 600,
        n_cv_folds: int = 5,
        optimize_metric: str = "auc",
    ) -> dict[str, Any]:
        """Optimize hyperparameters using Optuna.

        Uses walk-forward cross-validation to prevent overfitting.

        Args:
            outcomes: Training data
            n_trials: Number of Optuna trials
            timeout_seconds: Maximum optimization time
            n_cv_folds: Number of cross-validation folds
            optimize_metric: Metric to optimize ("auc", "accuracy", "f1")

        Returns:
            Dictionary with best parameters and study statistics
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is required for hyperparameter optimization")
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for training")
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, optimizing linear model only")

        import pandas as pd

        logger.info(f"Starting Optuna optimization: {n_trials} trials, {optimize_metric} metric")

        # Prepare data
        df = pd.DataFrame(outcomes)
        if len(df) < self.config.min_samples:
            raise ValueError(f"Insufficient samples: {len(df)} < {self.config.min_samples}")

        # Fit feature engineer
        self.feature_engineer.fit(df)
        X, y = self.feature_engineer.extract_batch(df)

        # Compute weights
        weights = compute_sample_weights(
            outcomes,
            traded_weight=self.config.traded_weight,
            counterfactual_weight=self.config.counterfactual_weight,
        )

        # Balance dataset
        if self.config.balance_classes:
            X, y, weights = balance_dataset(X, y, weights, max_ratio=self.config.max_class_ratio)

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            if LIGHTGBM_AVAILABLE:
                params = {
                    "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }
                model = lgb.LGBMClassifier(
                    **params,
                    random_state=self.config.random_state,
                    verbose=-1,
                )
            else:
                params = {
                    "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
                    "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                }
                model = LogisticRegression(
                    **params,
                    solver="saga",
                    max_iter=1000,
                    random_state=self.config.random_state,
                )

            # Walk-forward cross-validation using TimeSeriesSplit
            # IMPORTANT: Do NOT use StratifiedKFold with shuffle=True for time series data
            # as it causes look-ahead bias (training on future data)
            from sklearn.model_selection import TimeSeriesSplit

            cv = TimeSeriesSplit(n_splits=n_cv_folds)

            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                w_train = weights[train_idx]

                model.fit(X_train, y_train, sample_weight=w_train)

                # Always use predict_proba for probability scores
                y_pred_proba = model.predict_proba(X_val)[:, 1]

                y_pred = (y_pred_proba > 0.5).astype(int)

                if optimize_metric == "auc":
                    try:
                        score = roc_auc_score(y_val, y_pred_proba)
                    except ValueError:
                        score = 0.5
                elif optimize_metric == "accuracy":
                    score = accuracy_score(y_val, y_pred)
                elif optimize_metric == "f1":
                    score = f1_score(y_val, y_pred, zero_division=0)
                else:
                    score = accuracy_score(y_val, y_pred)

                scores.append(score)

                # Report intermediate value for pruning
                trial.report(np.mean(scores), fold_idx)

                # Early stopping
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        # Create Optuna study
        sampler = TPESampler(seed=self.config.random_state)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=True,
        )

        # Extract results
        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"Optuna optimization complete: best {optimize_metric}={best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        # Update config with best parameters
        if LIGHTGBM_AVAILABLE:
            self.config.lgb_num_leaves = best_params.get("num_leaves", self.config.lgb_num_leaves)
            self.config.lgb_max_depth = best_params.get("max_depth", self.config.lgb_max_depth)
            self.config.lgb_learning_rate = best_params.get(
                "learning_rate", self.config.lgb_learning_rate
            )
            self.config.lgb_n_estimators = best_params.get(
                "n_estimators", self.config.lgb_n_estimators
            )
            self.config.lgb_min_child_samples = best_params.get(
                "min_child_samples", self.config.lgb_min_child_samples
            )
            self.config.lgb_reg_alpha = best_params.get("reg_alpha", self.config.lgb_reg_alpha)
            self.config.lgb_reg_lambda = best_params.get("reg_lambda", self.config.lgb_reg_lambda)

        return {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(study.trials),
            "n_complete": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "optimization_history": [
                {"trial": t.number, "value": t.value, "params": t.params}
                for t in study.trials
                if t.value is not None
            ],
        }

    def train_with_optimization(
        self,
        outcomes: list[dict[str, Any]],
        optimize: bool = True,
        n_trials: int = 30,
        optimize_metric: str = "auc",
    ) -> TrainingMetrics:
        """Train with optional hyperparameter optimization.

        Convenience method that runs optimization if enabled, then trains.

        Args:
            outcomes: Training data
            optimize: Whether to run hyperparameter optimization first
            n_trials: Number of optimization trials
            optimize_metric: Metric to optimize

        Returns:
            Training metrics
        """
        if optimize and OPTUNA_AVAILABLE and len(outcomes) >= self.config.min_samples:
            try:
                opt_results = self.optimize_hyperparameters(
                    outcomes,
                    n_trials=n_trials,
                    optimize_metric=optimize_metric,
                )
                logger.info(f"Optimization results: {opt_results['best_value']:.4f}")
            except Exception as e:
                logger.warning(f"Optimization failed, using default params: {e}")

        return self.train(outcomes)
