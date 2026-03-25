"""
Two-Level Stacking Meta-Learner Ensemble.

Implements a stacking architecture for improved trading signal classification:

Level 0 (Base Estimators):
    - XGBoost: Gradient boosted trees (primary)
    - LightGBM: Histogram-based gradient boosting (if available)
    - CatBoost: Ordered boosting (if available)
    - Fallback: ExtraTreesClassifier from sklearn

Level 1 (Meta-Learner):
    - LightGBM with shallow depth trained on base model out-of-fold predictions
    - Passthrough enabled: original features concatenated with base predictions

RegimeAwareEnsemble:
    - Dynamically adjusts model weights based on detected market regime
    - Veto logic for anomaly/rug-detected tokens
    - Returns structured EnsembleDecision with full model contribution breakdown
"""

import logging
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from coldpath.security import secure_load

try:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        StackingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --- Optional dependency probing ------------------------------------------------

XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    pass

LIGHTGBM_AVAILABLE = False
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    pass


def _detect_cuda() -> bool:
    """Detect CUDA GPU availability for tree-based model acceleration."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        import os

        return bool(os.environ.get("CUDA_VISIBLE_DEVICES"))


_CUDA_AVAILABLE = _detect_cuda()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class StackingSignal(Enum):
    """Signal emitted by the stacking meta-learner."""

    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    NO_TRADE = "no_trade"  # Vetoed by regime / anomaly guard


@dataclass
class StackingConfig:
    """Configuration for the stacking meta-learner."""

    # Level 0
    n_cv_folds: int = 5
    passthrough: bool = True  # Include raw features at level 1
    use_proba: bool = True  # Stack on predict_proba, not hard labels

    # XGBoost base
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05

    # LightGBM base (used when available)
    lgb_n_estimators: int = 300
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 31

    # CatBoost base (used when available)
    cat_n_estimators: int = 300
    cat_max_depth: int = 6
    cat_learning_rate: float = 0.05

    # ExtraTrees fallback
    et_n_estimators: int = 300
    et_max_depth: int = 12

    # Level 1 meta-learner
    meta_n_estimators: int = 100
    meta_max_depth: int = 3  # Shallow to reduce overfitting
    meta_learning_rate: float = 0.1
    meta_num_leaves: int = 15

    # Misc
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = 0


@dataclass
class ModelContribution:
    """Contribution from an individual base model."""

    name: str
    weight: float
    prediction: str  # BUY / HOLD / SELL
    confidence: float
    feature_importances: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": round(self.weight, 4),
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class EnsembleDecision:
    """Final decision from the regime-aware stacking ensemble."""

    signal: StackingSignal
    confidence: float  # 0-1 overall confidence
    position_size: float  # Suggested fraction of capital (0-1)
    regime: str  # Current detected regime
    model_contributions: list[ModelContribution] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    buy_prob: float = 0.0
    hold_prob: float = 0.0
    sell_prob: float = 0.0
    veto_applied: bool = False

    @property
    def should_trade(self) -> bool:
        return self.signal == StackingSignal.BUY and self.position_size > 0

    @property
    def should_exit(self) -> bool:
        return self.signal in (StackingSignal.SELL, StackingSignal.NO_TRADE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal.value,
            "confidence": round(self.confidence, 4),
            "position_size": round(self.position_size, 4),
            "regime": self.regime,
            "buy_prob": round(self.buy_prob, 4),
            "hold_prob": round(self.hold_prob, 4),
            "sell_prob": round(self.sell_prob, 4),
            "veto_applied": self.veto_applied,
            "model_contributions": [c.to_dict() for c in self.model_contributions],
            "reasons": self.reasons,
        }


# ---------------------------------------------------------------------------
# StackingMetaLearner
# ---------------------------------------------------------------------------


class StackingMetaLearner:
    """Two-level stacking classifier for trading signal generation.

    Level 0 base estimators are combined via 5-fold CV stacking using
    ``predict_proba`` outputs.  Level 1 is a shallow LightGBM (or
    LogisticRegression fallback) trained on the concatenated base model
    probabilities and (optionally) the original feature set.

    Usage::

        learner = StackingMetaLearner()
        learner.fit(X_train, y_train)
        probas = learner.predict_proba(X_test)
        importances = learner.feature_importances()
    """

    def __init__(self, config: StackingConfig | None = None) -> None:
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for StackingMetaLearner. "
                "Install it with: pip install scikit-learn"
            )

        self.config = config or StackingConfig()
        self._stacker: StackingClassifier | None = None
        self._scaler: StandardScaler | None = None
        self._base_names: list[str] = []
        self._is_fitted: bool = False
        self._n_classes: int = 3  # BUY / HOLD / SELL
        self._feature_dim: int = 0

        logger.info(
            "StackingMetaLearner initialised  "
            f"(xgb={XGBOOST_AVAILABLE}, lgb={LIGHTGBM_AVAILABLE}, "
            f"cat={CATBOOST_AVAILABLE})"
        )

    # ------------------------------------------------------------------
    # Base estimator construction
    # ------------------------------------------------------------------

    def _build_base_estimators(self) -> list[tuple[str, Any]]:
        """Construct the list of (name, estimator) pairs for level 0."""
        cfg = self.config
        estimators: list[tuple[str, Any]] = []

        # XGBoost ---------------------------------------------------------
        if XGBOOST_AVAILABLE:
            estimators.append(
                (
                    "xgb",
                    xgb.XGBClassifier(
                        n_estimators=cfg.xgb_n_estimators,
                        max_depth=cfg.xgb_max_depth,
                        learning_rate=cfg.xgb_learning_rate,
                        eval_metric="mlogloss",
                        random_state=cfg.random_state,
                        tree_method="gpu_hist" if _CUDA_AVAILABLE else "hist",
                        device="cuda" if _CUDA_AVAILABLE else "cpu",
                        n_jobs=cfg.n_jobs,
                        verbosity=cfg.verbosity,
                    ),
                )
            )
            logger.info("  Level-0: XGBoost added")
        else:
            # Fallback: ExtraTrees in place of XGBoost
            estimators.append(
                (
                    "et_xgb_fallback",
                    ExtraTreesClassifier(
                        n_estimators=cfg.et_n_estimators,
                        max_depth=cfg.et_max_depth,
                        random_state=cfg.random_state,
                        n_jobs=cfg.n_jobs,
                    ),
                )
            )
            logger.info("  Level-0: ExtraTrees (XGBoost fallback) added")

        # LightGBM --------------------------------------------------------
        if LIGHTGBM_AVAILABLE:
            estimators.append(
                (
                    "lgb",
                    lgb.LGBMClassifier(
                        n_estimators=cfg.lgb_n_estimators,
                        max_depth=cfg.lgb_max_depth,
                        learning_rate=cfg.lgb_learning_rate,
                        num_leaves=cfg.lgb_num_leaves,
                        random_state=cfg.random_state,
                        device="gpu" if _CUDA_AVAILABLE else "cpu",
                        n_jobs=cfg.n_jobs,
                        verbose=-1,
                    ),
                )
            )
            logger.info("  Level-0: LightGBM added")
        else:
            estimators.append(
                (
                    "et_lgb_fallback",
                    ExtraTreesClassifier(
                        n_estimators=cfg.et_n_estimators,
                        max_depth=cfg.et_max_depth,
                        random_state=cfg.random_state + 1,
                        n_jobs=cfg.n_jobs,
                    ),
                )
            )
            logger.info("  Level-0: ExtraTrees (LightGBM fallback) added")

        # CatBoost --------------------------------------------------------
        if CATBOOST_AVAILABLE:
            estimators.append(
                (
                    "cat",
                    CatBoostClassifier(
                        iterations=cfg.cat_n_estimators,
                        depth=cfg.cat_max_depth,
                        learning_rate=cfg.cat_learning_rate,
                        random_seed=cfg.random_state,
                        verbose=0,
                        thread_count=cfg.n_jobs if cfg.n_jobs > 0 else -1,
                    ),
                )
            )
            logger.info("  Level-0: CatBoost added")
        else:
            estimators.append(
                (
                    "et_cat_fallback",
                    ExtraTreesClassifier(
                        n_estimators=cfg.et_n_estimators,
                        max_depth=cfg.et_max_depth,
                        random_state=cfg.random_state + 2,
                        n_jobs=cfg.n_jobs,
                    ),
                )
            )
            logger.info("  Level-0: ExtraTrees (CatBoost fallback) added")

        self._base_names = [name for name, _ in estimators]
        return estimators

    def _build_meta_learner(self) -> Any:
        """Construct the level-1 meta-learner."""
        cfg = self.config

        if LIGHTGBM_AVAILABLE:
            meta = lgb.LGBMClassifier(
                n_estimators=cfg.meta_n_estimators,
                max_depth=cfg.meta_max_depth,
                learning_rate=cfg.meta_learning_rate,
                num_leaves=cfg.meta_num_leaves,
                random_state=cfg.random_state,
                device="gpu" if _CUDA_AVAILABLE else "cpu",
                n_jobs=cfg.n_jobs,
                verbose=-1,
            )
            logger.info("  Level-1: LightGBM meta-learner")
        else:
            meta = LogisticRegression(
                max_iter=1000,
                random_state=cfg.random_state,
                n_jobs=cfg.n_jobs,
            )
            logger.info("  Level-1: LogisticRegression meta-learner (LightGBM fallback)")

        return meta

    # ------------------------------------------------------------------
    # Fit / Predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "StackingMetaLearner":
        """Fit the two-level stacking ensemble.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels of shape (n_samples,). Values in {0, 1, 2}
               mapping to SELL, HOLD, BUY.
            sample_weight: Optional per-sample weights.

        Returns:
            self
        """
        cfg = self.config
        self._feature_dim = X.shape[1]
        self._n_classes = len(np.unique(y))

        # Standard-scale features for stability
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Build stacking classifier
        base_estimators = self._build_base_estimators()
        meta_learner = self._build_meta_learner()

        stack_method = "predict_proba" if cfg.use_proba else "predict"

        self._stacker = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=cfg.n_cv_folds,
            stack_method=stack_method,
            passthrough=cfg.passthrough,
            n_jobs=cfg.n_jobs,
        )

        logger.info(
            f"Fitting stacking ensemble: {len(base_estimators)} base models, "
            f"{cfg.n_cv_folds}-fold CV, passthrough={cfg.passthrough}"
        )
        start = time.time()

        fit_params: dict[str, Any] = {}
        if sample_weight is not None:
            # StackingClassifier does not natively forward sample_weight to
            # base estimators during CV, but the final estimator respects it.
            fit_params["sample_weight"] = sample_weight

        self._stacker.fit(X_scaled, y, **fit_params)

        elapsed = time.time() - start
        self._is_fitted = True
        logger.info(f"Stacking ensemble fit complete in {elapsed:.1f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels array of shape (n_samples,).
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        return self._stacker.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability matrix of shape (n_samples, n_classes).
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        return self._stacker.predict_proba(X_scaled)

    # ------------------------------------------------------------------
    # Feature importance aggregation
    # ------------------------------------------------------------------

    def feature_importances(self) -> dict[str, np.ndarray]:
        """Aggregate feature importances from all fitted base models.

        Returns:
            Dictionary mapping model name to importance array of length
            ``n_features``.  Also includes an ``"aggregated"`` key with
            the mean importance across models (normalised to sum to 1).
        """
        self._check_fitted()

        importances: dict[str, np.ndarray] = {}

        for name, estimator in self._stacker.named_estimators_.items():
            imp = self._extract_importance(name, estimator)
            if imp is not None:
                importances[name] = imp

        # Compute mean aggregation
        if importances:
            stacked = np.stack(list(importances.values()), axis=0)
            mean_imp = stacked.mean(axis=0)
            total = mean_imp.sum()
            if total > 0:
                mean_imp = mean_imp / total
            importances["aggregated"] = mean_imp

        return importances

    def _extract_importance(self, name: str, estimator: Any) -> np.ndarray | None:
        """Extract feature importance from a single fitted estimator."""
        try:
            if hasattr(estimator, "feature_importances_"):
                return np.asarray(estimator.feature_importances_)
            elif hasattr(estimator, "coef_"):
                return np.abs(estimator.coef_).mean(axis=0)
        except Exception as exc:
            logger.warning(f"Could not extract importances from {name}: {exc}")
        return None

    # ------------------------------------------------------------------
    # Cross-validation score
    # ------------------------------------------------------------------

    def cross_val_accuracy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> dict[str, float]:
        """Run cross-validation on the stacking ensemble.

        Args:
            X: Features.
            y: Labels.
            cv: Number of folds.
            scoring: Metric name (sklearn compatible).

        Returns:
            Dict with ``"mean"``, ``"std"``, and ``"scores"`` keys.
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(X)
        scores = cross_val_score(self._stacker, X_scaled, y, cv=cv, scoring=scoring)
        return {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "scores": scores.tolist(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def save(self, path: str) -> None:
        """Persist the entire stacking ensemble to disk."""
        state = {
            "config": self.config,
            "stacker": self._stacker,
            "scaler": self._scaler,
            "base_names": self._base_names,
            "is_fitted": self._is_fitted,
            "n_classes": self._n_classes,
            "feature_dim": self._feature_dim,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)
        logger.info(f"StackingMetaLearner saved to {path}")

    def load(self, path: str) -> "StackingMetaLearner":
        """Load a persisted stacking ensemble from disk."""
        with open(path, "rb") as fh:
            state = secure_load(fh)
        self.config = state["config"]
        self._stacker = state["stacker"]
        self._scaler = state["scaler"]
        self._base_names = state["base_names"]
        self._is_fitted = state["is_fitted"]
        self._n_classes = state["n_classes"]
        self._feature_dim = state["feature_dim"]
        logger.info(f"StackingMetaLearner loaded from {path}")
        return self

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._stacker is None:
            raise RuntimeError("StackingMetaLearner is not fitted. Call fit() first.")

    def get_status(self) -> dict[str, Any]:
        """Return diagnostic status dict."""
        return {
            "is_fitted": self._is_fitted,
            "n_base_models": len(self._base_names),
            "base_models": list(self._base_names),
            "feature_dim": self._feature_dim,
            "n_classes": self._n_classes,
            "passthrough": self.config.passthrough,
            "cv_folds": self.config.n_cv_folds,
            "xgboost_available": XGBOOST_AVAILABLE,
            "lightgbm_available": LIGHTGBM_AVAILABLE,
            "catboost_available": CATBOOST_AVAILABLE,
        }


# ---------------------------------------------------------------------------
# Regime-aware ensemble
# ---------------------------------------------------------------------------

# Default model weight profiles per regime.
# Keys are base-model names; values are multiplicative weights on the
# model's predicted probabilities before aggregation.
REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "BULL": {
        "xgb": 0.40,
        "lgb": 0.35,
        "cat": 0.25,
        # Fallback names use same weight as the model they replace
        "et_xgb_fallback": 0.40,
        "et_lgb_fallback": 0.35,
        "et_cat_fallback": 0.25,
    },
    "BEAR": {
        "xgb": 0.30,
        "lgb": 0.30,
        "cat": 0.40,
        "et_xgb_fallback": 0.30,
        "et_lgb_fallback": 0.30,
        "et_cat_fallback": 0.40,
    },
    "CHOP": {
        "xgb": 0.35,
        "lgb": 0.35,
        "cat": 0.30,
        "et_xgb_fallback": 0.35,
        "et_lgb_fallback": 0.35,
        "et_cat_fallback": 0.30,
    },
    "MEV_HEAVY": {
        "xgb": 0.45,
        "lgb": 0.30,
        "cat": 0.25,
        "et_xgb_fallback": 0.45,
        "et_lgb_fallback": 0.30,
        "et_cat_fallback": 0.25,
    },
}


@dataclass
class RegimeEnsembleConfig:
    """Configuration for the regime-aware stacking ensemble."""

    # Veto thresholds
    anomaly_veto_threshold: float = 0.80
    rug_veto_threshold: float = 0.75
    min_confidence_for_trade: float = 0.35

    # Position sizing
    max_position_fraction: float = 0.25
    kelly_fraction_cap: float = 0.20

    # Regime fallback
    default_regime: str = "CHOP"


class RegimeAwareEnsemble:
    """Stacking ensemble that adapts model weighting to the current market regime.

    Wraps a :class:`StackingMetaLearner` and provides:
    * Per-regime weight adjustment across base models.
    * Veto logic: high anomaly / rug scores suppress trading signals.
    * Structured :class:`EnsembleDecision` output with model contribution
      breakdown.

    Usage::

        stacker = StackingMetaLearner()
        stacker.fit(X_train, y_train)

        regime_ens = RegimeAwareEnsemble(stacker=stacker)
        decision = regime_ens.predict(X_new, regime="BULL",
                                       anomaly_score=0.12, rug_score=0.05)
    """

    def __init__(
        self,
        stacker: StackingMetaLearner | None = None,
        config: RegimeEnsembleConfig | None = None,
        regime_weights: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._stacker = stacker or StackingMetaLearner()
        self.config = config or RegimeEnsembleConfig()
        self._regime_weights = regime_weights or REGIME_WEIGHTS

        self._current_regime: str = self.config.default_regime
        self._regime_history: list[tuple[str, float]] = []
        self._decision_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def stacker(self) -> StackingMetaLearner:
        return self._stacker

    @property
    def is_fitted(self) -> bool:
        return self._stacker.is_fitted

    @property
    def current_regime(self) -> str:
        return self._current_regime

    def set_regime(self, regime: str) -> None:
        """Update the active market regime.

        Args:
            regime: One of BULL, BEAR, CHOP, MEV_HEAVY.
        """
        old = self._current_regime
        self._current_regime = regime
        self._regime_history.append((regime, time.time()))
        logger.info(f"RegimeAwareEnsemble regime: {old} -> {regime}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "RegimeAwareEnsemble":
        """Fit the underlying stacking meta-learner.

        Convenience pass-through so callers don't need to access the
        inner stacker directly.
        """
        self._stacker.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(
        self,
        X: np.ndarray,
        regime: str | None = None,
        anomaly_score: float = 0.0,
        rug_score: float = 0.0,
        capital: float = 1.0,
    ) -> EnsembleDecision:
        """Generate a regime-aware ensemble decision for a single sample.

        Args:
            X: Feature vector of shape ``(1, n_features)`` or ``(n_features,)``.
            regime: Market regime override. Uses current regime if *None*.
            anomaly_score: Anomaly detection score (0-1). High values
                indicate suspicious token behaviour.
            rug_score: Rug-pull risk score (0-1).
            capital: Available capital for position sizing (fraction mode).

        Returns:
            :class:`EnsembleDecision` with signal, confidence, and metadata.
        """
        if not self._stacker.is_fitted:
            raise RuntimeError("RegimeAwareEnsemble is not fitted. Call fit() first.")

        regime = regime or self._current_regime
        reasons: list[str] = [f"Regime: {regime}"]

        # Ensure 2-D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # --- Veto check --------------------------------------------------
        veto, veto_reason = self._check_veto(anomaly_score, rug_score)
        if veto:
            reasons.append(veto_reason)
            self._decision_count += 1
            return EnsembleDecision(
                signal=StackingSignal.NO_TRADE,
                confidence=0.0,
                position_size=0.0,
                regime=regime,
                model_contributions=[],
                reasons=reasons,
                veto_applied=True,
            )

        # --- Base model probabilities ------------------------------------
        probas = self._stacker.predict_proba(X)[0]  # shape (n_classes,)
        # probas order: SELL(0), HOLD(1), BUY(2)
        sell_prob, hold_prob, buy_prob = (
            float(probas[0]),
            float(probas[1]),
            float(probas[2]) if len(probas) > 2 else 0.0,
        )

        # --- Regime-weighted base contributions --------------------------
        contributions = self._compute_contributions(X, regime)

        # --- Aggregate to final signal -----------------------------------
        label = int(np.argmax(probas))
        signal_map = {0: StackingSignal.SELL, 1: StackingSignal.HOLD, 2: StackingSignal.BUY}
        raw_signal = signal_map.get(label, StackingSignal.HOLD)

        confidence = float(probas[label])

        # Apply minimum confidence gate
        if raw_signal == StackingSignal.BUY and confidence < self.config.min_confidence_for_trade:
            raw_signal = StackingSignal.HOLD
            reasons.append(
                f"Confidence {confidence:.2f} below threshold "
                f"{self.config.min_confidence_for_trade:.2f}"
            )

        # --- Position sizing (simple Kelly-inspired fraction) ------------
        position_size = 0.0
        if raw_signal == StackingSignal.BUY:
            # Scale position by confidence and cap
            safety_factor = (1.0 - rug_score) * (1.0 - anomaly_score)
            raw_fraction = confidence * safety_factor
            position_size = min(raw_fraction, self.config.max_position_fraction)
            position_size = min(position_size, self.config.kelly_fraction_cap)
            reasons.append(f"Position fraction: {position_size:.4f}")

        self._decision_count += 1

        return EnsembleDecision(
            signal=raw_signal,
            confidence=confidence,
            position_size=position_size,
            regime=regime,
            model_contributions=contributions,
            reasons=reasons,
            buy_prob=buy_prob,
            hold_prob=hold_prob,
            sell_prob=sell_prob,
            veto_applied=False,
        )

    def predict_batch(
        self,
        X: np.ndarray,
        regime: str | None = None,
        anomaly_scores: np.ndarray | None = None,
        rug_scores: np.ndarray | None = None,
    ) -> list[EnsembleDecision]:
        """Generate decisions for a batch of samples.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            regime: Regime applied uniformly to the batch.
            anomaly_scores: Per-sample anomaly scores. Defaults to 0.
            rug_scores: Per-sample rug scores. Defaults to 0.

        Returns:
            List of :class:`EnsembleDecision`, one per sample.
        """
        n = X.shape[0]
        if anomaly_scores is None:
            anomaly_scores = np.zeros(n)
        if rug_scores is None:
            rug_scores = np.zeros(n)

        return [
            self.predict(
                X[i],
                regime=regime,
                anomaly_score=float(anomaly_scores[i]),
                rug_score=float(rug_scores[i]),
            )
            for i in range(n)
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_veto(
        self,
        anomaly_score: float,
        rug_score: float,
    ) -> tuple[bool, str]:
        """Return (should_veto, reason) based on anomaly / rug thresholds."""
        if rug_score >= self.config.rug_veto_threshold:
            return True, f"VETO: rug_score {rug_score:.2f} >= {self.config.rug_veto_threshold:.2f}"
        if anomaly_score >= self.config.anomaly_veto_threshold:
            msg = (
                f"VETO: anomaly_score {anomaly_score:.2f} >= "
                f"{self.config.anomaly_veto_threshold:.2f}"
            )
            return True, msg
        return False, ""

    def _compute_contributions(
        self,
        X: np.ndarray,
        regime: str,
    ) -> list[ModelContribution]:
        """Compute per-base-model contributions weighted by regime."""
        weights = self._regime_weights.get(regime, self._regime_weights.get("CHOP", {}))
        contributions: list[ModelContribution] = []

        for name, estimator in self._stacker._stacker.named_estimators_.items():
            model_weight = weights.get(name, 1.0 / max(len(self._stacker._base_names), 1))

            try:
                X_scaled = self._stacker._scaler.transform(X)
                proba = estimator.predict_proba(X_scaled)[0]
                label = int(np.argmax(proba))
                signal_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
                pred_name = signal_names.get(label, "HOLD")
                conf = float(proba[label])
            except Exception as exc:
                logger.debug(f"Could not get proba from {name}: {exc}")
                pred_name = "HOLD"
                conf = 0.0

            fi = self._stacker._extract_importance(name, estimator)

            contributions.append(
                ModelContribution(
                    name=name,
                    weight=model_weight,
                    prediction=pred_name,
                    confidence=conf,
                    feature_importances=fi,
                )
            )

        return contributions

    # ------------------------------------------------------------------
    # Persistence & diagnostics
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save the ensemble to *directory*."""
        import os

        os.makedirs(directory, exist_ok=True)

        self._stacker.save(os.path.join(directory, "stacking_meta_learner.pkl"))

        config_state = {
            "config": self.config,
            "current_regime": self._current_regime,
            "regime_weights": self._regime_weights,
            "decision_count": self._decision_count,
        }
        with open(os.path.join(directory, "regime_ensemble_config.pkl"), "wb") as fh:
            pickle.dump(config_state, fh)

        logger.info(f"RegimeAwareEnsemble saved to {directory}")

    def load(self, directory: str) -> "RegimeAwareEnsemble":
        """Load a previously saved ensemble from *directory*."""
        import os

        self._stacker.load(os.path.join(directory, "stacking_meta_learner.pkl"))

        config_path = os.path.join(directory, "regime_ensemble_config.pkl")
        if os.path.exists(config_path):
            with open(config_path, "rb") as fh:
                state = secure_load(fh)
            self.config = state.get("config", self.config)
            self._current_regime = state.get("current_regime", self.config.default_regime)
            self._regime_weights = state.get("regime_weights", REGIME_WEIGHTS)
            self._decision_count = state.get("decision_count", 0)

        logger.info(f"RegimeAwareEnsemble loaded from {directory}")
        return self

    def get_status(self) -> dict[str, Any]:
        """Return diagnostic status dict."""
        return {
            "is_fitted": self.is_fitted,
            "current_regime": self._current_regime,
            "decision_count": self._decision_count,
            "regime_history_len": len(self._regime_history),
            "stacker_status": self._stacker.get_status(),
            "veto_thresholds": {
                "anomaly": self.config.anomaly_veto_threshold,
                "rug": self.config.rug_veto_threshold,
            },
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_stacking_ensemble(
    config: StackingConfig | None = None,
    regime_config: RegimeEnsembleConfig | None = None,
) -> RegimeAwareEnsemble:
    """Factory: build a full regime-aware stacking ensemble.

    Args:
        config: Stacking hyper-parameters.
        regime_config: Regime / veto configuration.

    Returns:
        An unfitted :class:`RegimeAwareEnsemble` ready for ``.fit()``.
    """
    stacker = StackingMetaLearner(config=config)
    return RegimeAwareEnsemble(stacker=stacker, config=regime_config)


def create_fitted_stacking_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: StackingConfig | None = None,
    regime_config: RegimeEnsembleConfig | None = None,
    sample_weight: np.ndarray | None = None,
) -> RegimeAwareEnsemble:
    """Factory: build **and train** a regime-aware stacking ensemble.

    Args:
        X_train: Training features.
        y_train: Training labels (0=SELL, 1=HOLD, 2=BUY).
        config: Stacking hyper-parameters.
        regime_config: Regime / veto configuration.
        sample_weight: Optional per-sample weights.

    Returns:
        A fitted :class:`RegimeAwareEnsemble`.
    """
    ensemble = create_stacking_ensemble(config=config, regime_config=regime_config)
    ensemble.fit(X_train, y_train, sample_weight=sample_weight)
    return ensemble
