"""
Compact profitability learner used by distillation on the training server.

This implementation keeps the public contract expected by TeacherTrainer while
remaining lightweight and dependency-tolerant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .feature_engineering import FeatureEngineer

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def _detect_cuda() -> bool:
    """Detect CUDA GPU availability for tree-based model acceleration."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        import os
        return bool(os.environ.get("CUDA_VISIBLE_DEVICES"))


_CUDA_AVAILABLE = _detect_cuda()


@dataclass
class TrainingConfig:
    """Training parameters for profitability models."""

    use_lightgbm: bool = True
    use_xgboost: bool = True
    use_linear_fallback: bool = True
    use_ensemble: bool = True
    ensemble_weights: Dict[str, float] = field(
        default_factory=lambda: {"linear": 0.4, "lightgbm": 0.3, "xgboost": 0.3}
    )
    test_size: float = 0.2
    random_state: int = 42
    min_samples: int = 100
    min_positive_samples: int = 20
    use_mi_selection: bool = True
    mi_n_features: int = 35


@dataclass
class TrainingMetrics:
    """Summary metrics from training."""

    training_samples: int = 0
    test_samples: int = 0
    positive_samples: int = 0
    negative_samples: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    training_time_seconds: float = 0.0
    model_type: str = "unknown"
    feature_importance: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
            "training_time_seconds": self.training_time_seconds,
            "model_type": self.model_type,
            "feature_importance": self.feature_importance,
        }


class ProfitabilityLearner:
    """Learns profitability probabilities from outcome rows."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.feature_engineer = FeatureEngineer()
        self._models: Dict[str, Any] = {}
        self._selected_feature_indices: Optional[np.ndarray] = None
        self.is_trained = False

    def train(self, outcomes: list[dict[str, Any]]) -> TrainingMetrics:
        start = perf_counter()
        metrics = TrainingMetrics()
        metrics.model_type = "linear"

        if len(outcomes) < self.config.min_samples:
            metrics.training_time_seconds = perf_counter() - start
            metrics.model_type = "insufficient_data"
            return metrics

        df = pd.DataFrame(outcomes)
        self.feature_engineer.fit(df)
        X, y = self.feature_engineer.extract_batch(df)
        y = np.asarray(y).astype(int)

        if X.size == 0:
            metrics.training_time_seconds = perf_counter() - start
            metrics.model_type = "empty_features"
            return metrics

        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        metrics.positive_samples = pos
        metrics.negative_samples = neg
        if pos < self.config.min_positive_samples or neg == 0:
            metrics.training_time_seconds = perf_counter() - start
            metrics.model_type = "class_imbalance"
            return metrics

        X_model = X
        if self.config.use_mi_selection and X.shape[1] > self.config.mi_n_features:
            mi = mutual_info_classif(X, y, random_state=self.config.random_state)
            idx = np.argsort(mi)[::-1][: self.config.mi_n_features]
            idx = np.sort(idx)
            self._selected_feature_indices = idx
            X_model = X[:, idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X_model,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        metrics.training_samples = int(len(X_train))
        metrics.test_samples = int(len(X_test))

        linear = LogisticRegression(max_iter=1000, random_state=self.config.random_state)
        linear.fit(X_train, y_train)
        self._models["linear"] = linear

        if LIGHTGBM_AVAILABLE and self.config.use_lightgbm:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.03,
                random_state=self.config.random_state,
                device="gpu" if _CUDA_AVAILABLE else "cpu",
            )
            lgb_model.fit(X_train, y_train)
            self._models["lightgbm"] = lgb_model

        if XGBOOST_AVAILABLE and self.config.use_xgboost:
            xgb_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.03,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.config.random_state,
                tree_method="gpu_hist" if _CUDA_AVAILABLE else "hist",
                device="cuda" if _CUDA_AVAILABLE else "cpu",
            )
            xgb_model.fit(X_train, y_train)
            self._models["xgboost"] = xgb_model

        probs = self._blend_predictions(X_test)
        preds = (probs >= 0.5).astype(int)
        metrics.accuracy = float(accuracy_score(y_test, preds))
        metrics.precision = float(precision_score(y_test, preds, zero_division=0))
        metrics.recall = float(recall_score(y_test, preds, zero_division=0))
        metrics.f1_score = float(f1_score(y_test, preds, zero_division=0))
        metrics.auc_roc = float(roc_auc_score(y_test, probs))
        metrics.model_type = "ensemble" if len(self._models) > 1 else "linear"
        metrics.training_time_seconds = perf_counter() - start
        self.is_trained = True
        return metrics

    def _blend_predictions(self, X: np.ndarray) -> np.ndarray:
        active = [k for k in ("linear", "lightgbm", "xgboost") if k in self._models]
        if not active:
            return np.full(len(X), 0.5, dtype=float)

        weights_cfg = self.config.ensemble_weights
        raw_weights = np.array(
            [
                weights_cfg.get("linear", 0.4) if k == "linear" else
                weights_cfg.get("lightgbm", 0.3) if k == "lightgbm" else
                weights_cfg.get("xgboost", 0.3)
                for k in active
            ],
            dtype=float,
        )
        weights = raw_weights / max(raw_weights.sum(), 1e-8)

        preds = np.stack([self._models[k].predict_proba(X)[:, 1] for k in active], axis=1)
        return (preds * weights).sum(axis=1)

    def _get_ensemble_predictions(self, X: np.ndarray) -> np.ndarray:
        X_model = X
        if self._selected_feature_indices is not None and X.shape[1] != len(self._selected_feature_indices):
            max_idx = int(np.max(self._selected_feature_indices))
            if X.shape[1] > max_idx:
                X_model = X[:, self._selected_feature_indices]
        return self._blend_predictions(X_model)
