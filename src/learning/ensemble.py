"""
Lightweight ensemble implementation for training-server orchestration paths.

Provides the API expected by training.ensemble_trainer without depending on
ColdPath-only model classes.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

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


class _ConstantModel:
    def __init__(self, p: float):
        self.p = float(np.clip(p, 0.0, 1.0))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        p = np.full(n, self.p)
        return np.column_stack([1.0 - p, p])


@dataclass
class EnsembleDecision:
    confidence: float
    action: str

    @property
    def should_trade(self) -> bool:
        return self.action == "BUY"


class ModelEnsemble:
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float]):
        self.models = models
        self.weights = weights

    def evaluate(
        self,
        features_50: np.ndarray,
        price_sequence: Optional[np.ndarray] = None,
        current_price: float = 0.0,
    ) -> EnsembleDecision:
        x = np.asarray(features_50, dtype=float).reshape(1, -1)
        p = float(self.predict_proba(x)[0])
        action = "BUY" if p >= 0.5 else "HOLD"
        return EnsembleDecision(confidence=p, action=action)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        active = [k for k in ("linear", "lightgbm", "xgboost") if k in self.models]
        if not active:
            return np.full(len(X), 0.5, dtype=float)

        raw = np.array([self.weights.get(k, 1.0) for k in active], dtype=float)
        w = raw / max(raw.sum(), 1e-8)
        preds = np.stack([self.models[k].predict_proba(X)[:, 1] for k in active], axis=1)
        return (preds * w).sum(axis=1)

    def get_model_status(self) -> Dict[str, Any]:
        return {
            "models": sorted(self.models.keys()),
            "weights": self.weights,
            "lightgbm_available": LIGHTGBM_AVAILABLE,
            "xgboost_available": XGBOOST_AVAILABLE,
        }

    def save(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "ensemble.pkl", "wb") as f:
            pickle.dump({"models": self.models, "weights": self.weights}, f)
        with open(out / "ensemble_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.get_model_status(), f, indent=2)


def _to_binary(labels: np.ndarray) -> np.ndarray:
    y = np.asarray(labels)
    if y.dtype.kind in {"f"}:
        return (y > 0).astype(int)
    uniques = np.unique(y)
    if set(uniques.tolist()) <= {0, 1}:
        return y.astype(int)
    return (y.astype(float) > 0).astype(int)


def create_ensemble_from_training(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    price_sequences: Optional[np.ndarray] = None,
    price_targets: Optional[np.ndarray] = None,
    rug_labels: Optional[np.ndarray] = None,
) -> ModelEnsemble:
    x = np.asarray(features_train, dtype=float)
    y = _to_binary(labels_train)

    if len(np.unique(y)) < 2:
        models: Dict[str, Any] = {"linear": _ConstantModel(float(np.mean(y)))}
        return ModelEnsemble(models=models, weights={"linear": 1.0})

    models = {}
    weights = {"linear": 0.4, "lightgbm": 0.3, "xgboost": 0.3}

    linear = LogisticRegression(max_iter=1000, random_state=42)
    linear.fit(x, y)
    models["linear"] = linear

    if LIGHTGBM_AVAILABLE:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            random_state=42,
            device="gpu" if _CUDA_AVAILABLE else "cpu",
        )
        lgb_model.fit(x, y)
        models["lightgbm"] = lgb_model

    if XGBOOST_AVAILABLE:
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            tree_method="gpu_hist" if _CUDA_AVAILABLE else "hist",
            device="cuda" if _CUDA_AVAILABLE else "cpu",
        )
        xgb_model.fit(x, y)
        models["xgboost"] = xgb_model

    active = [k for k in ("linear", "lightgbm", "xgboost") if k in models]
    norm = 1.0 / len(active)
    normalized_weights = {k: norm for k in active}
    return ModelEnsemble(models=models, weights=normalized_weights)
