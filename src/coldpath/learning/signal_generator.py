"""
XGBoost Signal Generator.

3-class signal classification for trading decisions:
- BUY: Price expected to rise >3.5% in 15 minutes
- SELL: Price expected to drop >4% in 15 minutes
- HOLD: Neither condition met

Uses XGBoost with optimized hyperparameters for imbalanced classification.
"""

import logging
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from coldpath.security import secure_load

logger = logging.getLogger(__name__)

# Check for XGBoost availability
XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb  # noqa: F401

    XGBOOST_AVAILABLE = True
    logger.info("XGBoost available for signal generator")
except ImportError:
    logger.warning("XGBoost not available, signal generator will use fallback")


class Signal(Enum):
    """Trading signal type."""

    SELL = 0
    HOLD = 1
    BUY = 2


@dataclass
class SignalPrediction:
    """Signal prediction result."""

    signal: Signal
    confidence: float  # 0-1 confidence in prediction
    buy_prob: float  # Probability of BUY signal
    hold_prob: float  # Probability of HOLD signal
    sell_prob: float  # Probability of SELL signal
    expected_return: float  # Expected return based on probabilities

    @property
    def should_trade(self) -> bool:
        """Check if signal suggests trading (not HOLD)."""
        return self.signal != Signal.HOLD

    @property
    def is_high_confidence(self) -> bool:
        """Check if prediction is high confidence (>0.7)."""
        return self.confidence > 0.7

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal": self.signal.name,
            "signal_value": self.signal.value,
            "confidence": self.confidence,
            "buy_prob": self.buy_prob,
            "hold_prob": self.hold_prob,
            "sell_prob": self.sell_prob,
            "expected_return": self.expected_return,
            "should_trade": self.should_trade,
            "is_high_confidence": self.is_high_confidence,
        }


# Signal thresholds for labeling - calibrated for realistic memecoin moves
BUY_THRESHOLD = 3.5  # +3.5% in 15 min (realistic for memecoins)
SELL_THRESHOLD = -4.0  # -4% in 15 min (avoid noise exits)


class XGBoostSignalGenerator:
    """XGBoost-based 3-class signal classifier.

    Predicts trading signals (BUY/HOLD/SELL) based on 50-feature vectors.
    Optimized for imbalanced data typical in trading scenarios.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        buy_threshold: float = BUY_THRESHOLD,
        sell_threshold: float = SELL_THRESHOLD,
    ):
        """Initialize the signal generator.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Row subsampling ratio
            colsample_bytree: Column subsampling ratio
            min_child_weight: Minimum sum of instance weight in child
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
            buy_threshold: Return threshold for BUY signal (default 6%)
            sell_threshold: Return threshold for SELL signal (default -2%)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        self.scaler = StandardScaler()
        self.is_fitted = False

        self._feature_importance: dict[str, float] | None = None
        self._class_weights: np.ndarray | None = None
        self._training_metrics: dict[str, Any] = {}

        if XGBOOST_AVAILABLE:
            import xgboost as xgb  # Re-import inside block for LSP

            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=-1,
            )
            logger.info("XGBoost signal generator initialized")
        else:
            self.model = None
            logger.warning("XGBoost not available, using fallback")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        early_stopping_rounds: int = 20,
    ) -> dict[str, Any]:
        """Train the signal generator.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=SELL, 1=HOLD, 2=BUY)
            feature_names: Optional feature names for importance tracking
            sample_weight: Optional sample weights for imbalanced data
            eval_set: Optional validation set for early stopping
            early_stopping_rounds: Rounds for early stopping

        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training XGBoost signal generator on {X.shape[0]} samples")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Calculate class weights if not provided
        if sample_weight is None:
            class_counts = np.bincount(y.astype(int), minlength=3)
            total = len(y)
            self._class_weights = total / (3.0 * class_counts + 1e-6)
            sample_weight = self._class_weights[y.astype(int)]

        if not XGBOOST_AVAILABLE or self.model is None:
            # Fallback: store class priors
            self._class_priors = np.bincount(y.astype(int), minlength=3) / len(y)
            self.is_fitted = True
            return {"accuracy": 0.0, "cv_score": 0.0}

        # Prepare validation set
        fit_params = {
            "sample_weight": sample_weight,
            "verbose": False,
        }

        if eval_set is not None:
            X_val_scaled = self.scaler.transform(eval_set[0])
            fit_params["eval_set"] = [(X_val_scaled, eval_set[1])]

        # Train model
        self.model.fit(X_scaled, y, **fit_params)

        # Calculate metrics
        train_accuracy = self.model.score(X_scaled, y)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring="accuracy")

        # Feature importance
        if feature_names is not None and len(feature_names) == X.shape[1]:
            importance = self.model.feature_importances_
            self._feature_importance = dict(zip(feature_names, importance, strict=False))
        else:
            self._feature_importance = {
                f"feature_{i}": v for i, v in enumerate(self.model.feature_importances_)
            }

        self._training_metrics = {
            "train_accuracy": train_accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "n_samples": len(y),
            "class_distribution": {
                "sell": int((y == 0).sum()),
                "hold": int((y == 1).sum()),
                "buy": int((y == 2).sum()),
            },
        }

        self.is_fitted = True
        logger.info(
            f"Training complete: accuracy={train_accuracy:.3f}, "
            f"cv={cv_scores.mean():.3f}±{cv_scores.std():.3f}"
        )

        return self._training_metrics

    def predict(self, features: np.ndarray) -> SignalPrediction:
        """Predict trading signal for a single sample.

        Args:
            features: Feature vector (n_features,) or (1, n_features)

        Returns:
            SignalPrediction with signal and probabilities
        """
        if not self.is_fitted:
            return SignalPrediction(
                signal=Signal.HOLD,
                confidence=0.0,
                buy_prob=0.33,
                hold_prob=0.34,
                sell_prob=0.33,
                expected_return=0.0,
            )

        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Scale and predict
        X_scaled = self.scaler.transform(features)

        if not XGBOOST_AVAILABLE or self.model is None:
            # Fallback: return based on class priors
            return SignalPrediction(
                signal=Signal.HOLD,
                confidence=0.0,
                buy_prob=self._class_priors[2] if hasattr(self, "_class_priors") else 0.33,
                hold_prob=self._class_priors[1] if hasattr(self, "_class_priors") else 0.34,
                sell_prob=self._class_priors[0] if hasattr(self, "_class_priors") else 0.33,
                expected_return=0.0,
            )

        # Get probabilities
        probs = self.model.predict_proba(X_scaled)[0]
        sell_prob, hold_prob, buy_prob = probs

        # Determine signal and confidence
        predicted_class = np.argmax(probs)
        confidence = float(probs[predicted_class])
        signal = Signal(predicted_class)

        # Calculate expected return from RAW probabilities only
        # Costs (slippage, fees, inclusion) are modeled downstream in HotPath calibrator
        # Using raw payoff estimates WITHOUT fee deductions to avoid double-counting
        # SELL=-3% (avg raw downside), HOLD=0.0 (neutral), BUY=+6% (avg raw upside)
        expected_return = sell_prob * -3.0 + hold_prob * 0.0 + buy_prob * 6.0

        return SignalPrediction(
            signal=signal,
            confidence=confidence,
            buy_prob=float(buy_prob),
            hold_prob=float(hold_prob),
            sell_prob=float(sell_prob),
            expected_return=float(expected_return),
        )

    def predict_batch(self, X: np.ndarray) -> list[SignalPrediction]:
        """Predict trading signals for multiple samples.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            List of SignalPrediction objects
        """
        if not self.is_fitted or not XGBOOST_AVAILABLE or self.model is None:
            return [
                SignalPrediction(
                    signal=Signal.HOLD,
                    confidence=0.0,
                    buy_prob=0.33,
                    hold_prob=0.34,
                    sell_prob=0.33,
                    expected_return=0.0,
                )
                for _ in range(len(X))
            ]

        X_scaled = self.scaler.transform(X)
        all_probs = self.model.predict_proba(X_scaled)

        predictions = []
        for probs in all_probs:
            sell_prob, hold_prob, buy_prob = probs
            predicted_class = np.argmax(probs)
            confidence = float(probs[predicted_class])
            signal = Signal(predicted_class)

            # Raw payoffs without fee deductions (costs modeled in HotPath calibrator)
            # SELL=-3% (raw downside), HOLD=0.0 (neutral), BUY=+6% (raw upside)
            expected_return = sell_prob * -3.0 + hold_prob * 0.0 + buy_prob * 6.0

            predictions.append(
                SignalPrediction(
                    signal=signal,
                    confidence=confidence,
                    buy_prob=float(buy_prob),
                    hold_prob=float(hold_prob),
                    sell_prob=float(sell_prob),
                    expected_return=float(expected_return),
                )
            )

        return predictions

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self._feature_importance is None:
            return {}
        return self._feature_importance.copy()

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N most important features.

        Args:
            n: Number of features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if self._feature_importance is None:
            return []

        sorted_features = sorted(self._feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def save(self, path: str) -> None:
        """Save model to file."""
        data = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
            "feature_importance": self._feature_importance,
            "class_weights": self._class_weights,
            "training_metrics": self._training_metrics,
        }

        if XGBOOST_AVAILABLE and self.model is not None:
            data["model"] = self.model

        if hasattr(self, "_class_priors"):
            data["class_priors"] = self._class_priors

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved XGBoost signal generator to {path}")

    def load(self, path: str) -> "XGBoostSignalGenerator":
        """Load model from file."""
        with open(path, "rb") as f:
            data = secure_load(f)

        self.n_estimators = data["n_estimators"]
        self.max_depth = data["max_depth"]
        self.learning_rate = data["learning_rate"]
        self.subsample = data["subsample"]
        self.colsample_bytree = data["colsample_bytree"]
        self.min_child_weight = data["min_child_weight"]
        self.gamma = data["gamma"]
        self.reg_alpha = data["reg_alpha"]
        self.reg_lambda = data["reg_lambda"]
        self.random_state = data["random_state"]
        self.buy_threshold = data["buy_threshold"]
        self.sell_threshold = data["sell_threshold"]
        self.scaler = data["scaler"]
        self.is_fitted = data["is_fitted"]
        self._feature_importance = data.get("feature_importance")
        self._class_weights = data.get("class_weights")
        self._training_metrics = data.get("training_metrics", {})

        if "model" in data and XGBOOST_AVAILABLE:
            self.model = data["model"]

        if "class_priors" in data:
            self._class_priors = data["class_priors"]

        logger.info(f"Loaded XGBoost signal generator from {path}")
        return self


def create_signal_labels(
    future_returns: np.ndarray,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
) -> np.ndarray:
    """Create signal labels from future returns.

    Args:
        future_returns: Array of future returns (%)
        buy_threshold: Return threshold for BUY (default 6%)
        sell_threshold: Return threshold for SELL (default -2%)

    Returns:
        Array of labels (0=SELL, 1=HOLD, 2=BUY)
    """
    labels = np.ones(len(future_returns), dtype=int)  # Default to HOLD
    labels[future_returns >= buy_threshold] = 2  # BUY
    labels[future_returns <= sell_threshold] = 0  # SELL

    return labels


def balance_signal_classes(
    X: np.ndarray,
    y: np.ndarray,
    max_ratio: float = 3.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance signal classes by undersampling majority class.

    Args:
        X: Feature matrix
        y: Label array
        max_ratio: Maximum ratio of majority to minority class
        random_state: Random seed

    Returns:
        Balanced (X, y) tuple
    """
    np.random.seed(random_state)

    class_indices = {
        0: np.where(y == 0)[0],  # SELL
        1: np.where(y == 1)[0],  # HOLD
        2: np.where(y == 2)[0],  # BUY
    }

    class_counts = [len(idx) for idx in class_indices.values()]
    min_count = min(class_counts)
    target_count = int(min_count * max_ratio)

    balanced_indices = []
    for _class_label, indices in class_indices.items():
        if len(indices) > target_count:
            selected = np.random.choice(indices, target_count, replace=False)
        else:
            selected = indices
        balanced_indices.extend(selected)

    np.random.shuffle(balanced_indices)

    return X[balanced_indices], y[balanced_indices]
