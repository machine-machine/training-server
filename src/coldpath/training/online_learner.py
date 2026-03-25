"""
Online learning module for continuous model adaptation.

Uses River library for incremental learning that updates models
with each new trade outcome, enabling rapid adaptation to changing
market conditions without full retraining.

Key Features:
- Single-sample updates (no batch retraining needed)
- Concept drift detection
- Multi-armed bandit for strategy selection
- Adaptive feature normalization

Usage:
    from coldpath.training.online_learner import OnlineProfitabilityLearner

    learner = OnlineProfitabilityLearner()

    # After each trade
    learner.update(features, actual_return)

    # Get prediction for new candidate
    prediction = learner.predict(features)
"""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coldpath.security import secure_load, secure_loads

if TYPE_CHECKING:
    pass

import numpy as np

logger = logging.getLogger(__name__)

RIVER_AVAILABLE = False
try:
    from river import (
        compose,
        linear_model,
        metrics,  # noqa: F401
        optim,
        preprocessing,
        stream,  # noqa: F401
    )
    from river.forest import ARFClassifier, ARFRegressor  # noqa: F401
    from river.linear_model import (
        LogisticRegression,  # noqa: F401
        Perceptron,  # noqa: F401
    )
    from river.multiclass import OneVsOneClassifier  # noqa: F401
    from river.tree import (
        HoeffdingTreeClassifier,  # noqa: F401
        HoeffdingTreeRegressor,  # noqa: F401
    )

    RIVER_AVAILABLE = True
    logger.info("River library available for online learning")
except ImportError:
    logger.warning(
        "River not available. Install with: pip install river>=0.21.0. "
        "Online learning will use numpy fallback."
    )


@dataclass
class OnlineLearningMetrics:
    """Metrics for online learning performance tracking."""

    total_updates: int = 0
    correct_predictions: int = 0
    total_predictions: int = 0
    sum_squared_error: float = 0.0
    sum_absolute_error: float = 0.0
    last_update_time: datetime | None = None
    drift_detections: int = 0
    model_rollbacks: int = 0

    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    @property
    def mse(self) -> float:
        if self.total_updates == 0:
            return 0.0
        return self.sum_squared_error / self.total_updates

    @property
    def mae(self) -> float:
        if self.total_updates == 0:
            return 0.0
        return self.sum_absolute_error / self.total_updates

    @property
    def rmse(self) -> float:
        return math.sqrt(self.mse)


@dataclass
class DriftAlert:
    """Alert when concept drift is detected."""

    timestamp: datetime
    feature_name: str | None
    drift_type: str  # "gradual", "sudden", "incremental"
    severity: float  # 0.0 - 1.0
    recommended_action: str


class OnlineProfitabilityLearner:
    """Online learner for trade profitability prediction.

    Continuously adapts to market conditions using incremental learning.
    Each trade outcome updates the model immediately.

    Features:
    - Streaming feature normalization
    - Adaptive learning rate
    - Concept drift detection
    - Automatic model rollback on performance degradation
    """

    def __init__(
        self,
        n_features: int = 7,
        learning_rate: float = 0.01,
        drift_detection_window: int = 100,
        performance_window: int = 50,
        rollback_threshold: float = 0.1,  # Rollback if accuracy drops > 10%
    ):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.drift_detection_window = drift_detection_window
        self.performance_window = performance_window
        self.rollback_threshold = rollback_threshold

        self.metrics = OnlineLearningMetrics()
        self.recent_performance: list[float] = []
        self.drift_alerts: list[DriftAlert] = []

        if RIVER_AVAILABLE:
            self._init_river_model()
        else:
            self._init_numpy_fallback()

        logger.info(f"Online learner initialized (River: {RIVER_AVAILABLE})")

    def _init_river_model(self):
        """Initialize River-based online model pipeline."""
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(
                optimizer=optim.SGD(lr=self.learning_rate),
                loss=optim.losses.Log(),
            ),
        )

        self.regression_model = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LinearRegression(optimizer=optim.SGD(lr=self.learning_rate)),
        )

        self.forest_model = ARFRegressor(
            n_models=5,
            max_features="sqrt",
            leaf_prediction="mean",
        )

        self._model_checkpoint = None
        self._checkpoint_metrics = None

    def _init_numpy_fallback(self):
        """Initialize numpy-based fallback for environments without River."""
        self._weights = np.zeros(self.n_features)
        self._bias = 0.0
        self._feature_means = np.zeros(self.n_features)
        self._feature_stds = np.ones(self.n_features)
        self._update_count = 0

    def predict(self, features: dict[str, float]) -> tuple[float, float]:
        """Predict profitability score and confidence.

        Args:
            features: Dictionary of feature name -> value

        Returns:
            Tuple of (score, confidence)
        """
        self.metrics.total_predictions += 1

        if RIVER_AVAILABLE:
            return self._predict_river(features)
        else:
            return self._predict_numpy(features)

    def _predict_river(self, features: dict[str, float]) -> tuple[float, float]:
        """Predict using River model."""
        try:
            prob = self.model.predict_proba_one(features)
            score = prob.get(True, 0.5)

            confidence = max(prob.values()) if prob else 0.5

            return score, confidence
        except Exception as e:
            logger.warning(f"River prediction failed: {e}")
            return 0.5, 0.0

    def _predict_numpy(self, features: dict[str, float]) -> tuple[float, float]:
        """Predict using numpy fallback."""
        feature_values = np.array(list(features.values()))

        if len(feature_values) != self.n_features:
            logger.warning(f"Feature count mismatch: {len(feature_values)} vs {self.n_features}")
            return 0.5, 0.0

        normalized = (feature_values - self._feature_means) / (self._feature_stds + 1e-8)
        logit = np.dot(self._weights, normalized) + self._bias
        score = 1.0 / (1.0 + math.exp(-logit))

        confidence = min(1.0, abs(logit) / 2.0)

        return score, confidence

    def update(
        self,
        features: dict[str, float],
        actual_outcome: bool,
        actual_return: float | None = None,
    ) -> dict[str, Any]:
        """Update model with new trade outcome.

        Args:
            features: Features used for prediction
            actual_outcome: Whether trade was profitable
            actual_return: Actual return percentage (optional)

        Returns:
            Update result with metrics
        """
        self.metrics.total_updates += 1
        self.metrics.last_update_time = datetime.now()

        if RIVER_AVAILABLE:
            result = self._update_river(features, actual_outcome, actual_return)
        else:
            result = self._update_numpy(features, actual_outcome)

        self._check_performance()
        self._check_drift(features, actual_outcome)

        return result

    def _update_river(
        self,
        features: dict[str, float],
        actual_outcome: bool,
        actual_return: float | None,
    ) -> dict[str, Any]:
        """Update River model."""
        prev_pred, _ = self.predict(features)

        self.model.learn_one(features, actual_outcome)

        if actual_return is not None:
            self.regression_model.learn_one(features, actual_return)
            self.forest_model.learn_one(features, actual_return)

        new_pred, confidence = self.predict(features)

        correct = (prev_pred > 0.5) == actual_outcome
        if correct:
            self.metrics.correct_predictions += 1

        return {
            "previous_prediction": prev_pred,
            "new_prediction": new_pred,
            "confidence": confidence,
            "correct": correct,
            "actual_outcome": actual_outcome,
        }

    def _update_numpy(
        self,
        features: dict[str, float],
        actual_outcome: bool,
    ) -> dict[str, Any]:
        """Update numpy fallback model."""
        feature_values = np.array(list(features.values()))

        self._update_count += 1
        n = self._update_count

        old_mean = self._feature_means.copy()
        self._feature_means = old_mean + (feature_values - old_mean) / n

        delta = feature_values - old_mean
        old_var = self._feature_stds**2
        new_var = old_var + delta * (feature_values - self._feature_means)
        self._feature_stds = np.sqrt(new_var / n + 1e-8)

        prev_pred, _ = self._predict_numpy(features)

        normalized = (feature_values - self._feature_means) / (self._feature_stds + 1e-8)
        target = 1.0 if actual_outcome else 0.0
        error = target - prev_pred

        self._weights += self.learning_rate * error * normalized
        self._bias += self.learning_rate * error

        new_pred, confidence = self._predict_numpy(features)

        correct = (prev_pred > 0.5) == actual_outcome
        if correct:
            self.metrics.correct_predictions += 1

        return {
            "previous_prediction": prev_pred,
            "new_prediction": new_pred,
            "confidence": confidence,
            "correct": correct,
            "actual_outcome": actual_outcome,
        }

    def _check_performance(self):
        """Check for performance degradation and potentially rollback."""
        if len(self.recent_performance) < self.performance_window:
            return

        recent_acc = (
            sum(self.recent_performance[-self.performance_window :]) / self.performance_window
        )

        if len(self.recent_performance) >= 2 * self.performance_window:
            older_acc = (
                sum(
                    self.recent_performance[-2 * self.performance_window : -self.performance_window]
                )
                / self.performance_window
            )

            if older_acc - recent_acc > self.rollback_threshold:
                self._rollback_model()
                self.metrics.model_rollbacks += 1
                logger.warning(
                    f"Performance degradation detected: {older_acc:.3f} -> {recent_acc:.3f}, "
                    f"rolling back model"
                )

    def _check_drift(self, features: dict[str, float], actual_outcome: bool):
        """Detect concept drift in incoming data."""
        pred, confidence = self.predict(features)
        is_correct = (pred > 0.5) == actual_outcome
        self.recent_performance.append(1.0 if is_correct else 0.0)

        if len(self.recent_performance) > 2 * self.performance_window:
            self.recent_performance = self.recent_performance[-2 * self.performance_window :]

        if len(self.recent_performance) >= self.drift_detection_window:
            recent = sum(self.recent_performance[-self.drift_detection_window // 2 :])
            older = sum(
                self.recent_performance[
                    -self.drift_detection_window : -self.drift_detection_window // 2
                ]
            )

            drift_score = abs(recent - older) / (self.drift_detection_window // 2)

            if drift_score > 0.15:  # 15% accuracy change
                drift_type = "sudden" if drift_score > 0.3 else "gradual"
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    feature_name=None,
                    drift_type=drift_type,
                    severity=drift_score,
                    recommended_action="Consider increasing learning rate or full retrain",
                )
                self.drift_alerts.append(alert)
                self.metrics.drift_detections += 1
                logger.info(f"Concept drift detected: {drift_type}, severity={drift_score:.3f}")

    def _rollback_model(self):
        """Rollback to previous model checkpoint."""
        if RIVER_AVAILABLE and self._model_checkpoint is not None:
            self.model = secure_loads(self._model_checkpoint)
            if self._checkpoint_metrics:
                self.metrics = self._checkpoint_metrics
            logger.info("Model rolled back to previous checkpoint")

    def save_checkpoint(self):
        """Save current model state as checkpoint for potential rollback."""
        if RIVER_AVAILABLE:
            self._model_checkpoint = pickle.dumps(self.model)
            self._checkpoint_metrics = OnlineLearningMetrics(
                total_updates=self.metrics.total_updates,
                correct_predictions=self.metrics.correct_predictions,
                total_predictions=self.metrics.total_predictions,
            )

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from linear model weights."""
        if RIVER_AVAILABLE:
            try:
                if hasattr(self.model, "steps") and len(self.model.steps) > 1:
                    regressor = self.model.steps[-1][1]
                    if hasattr(regressor, "weights"):
                        return dict(regressor.weights)
            except Exception:
                pass

        if hasattr(self, "_weights"):
            return {f"feature_{i}": float(w) for i, w in enumerate(self._weights)}

        return {}

    def get_metrics(self) -> OnlineLearningMetrics:
        """Get current learning metrics."""
        return self.metrics

    def get_drift_alerts(self, limit: int = 10) -> list[DriftAlert]:
        """Get recent drift alerts."""
        return self.drift_alerts[-limit:]

    def save(self, path: Path):
        """Save model to file."""
        data = {
            "metrics": self.metrics,
            "n_features": self.n_features,
            "learning_rate": self.learning_rate,
            "recent_performance": self.recent_performance,
        }

        if RIVER_AVAILABLE:
            data["model"] = pickle.dumps(self.model)
            data["regression_model"] = pickle.dumps(self.regression_model)
        else:
            data["weights"] = self._weights.tolist()
            data["bias"] = self._bias
            data["feature_means"] = self._feature_means.tolist()
            data["feature_stds"] = self._feature_stds.tolist()

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Online learner saved to {path}")

    def load(self, path: Path):
        """Load model from file."""
        with open(path, "rb") as f:
            data = secure_load(f)

        self.metrics = data["metrics"]
        self.n_features = data["n_features"]
        self.learning_rate = data["learning_rate"]
        self.recent_performance = data.get("recent_performance", [])

        if RIVER_AVAILABLE and "model" in data:
            self.model = secure_loads(data["model"])
            if "regression_model" in data:
                self.regression_model = secure_loads(data["regression_model"])
        elif "weights" in data:
            self._weights = np.array(data["weights"])
            self._bias = data["bias"]
            self._feature_means = np.array(data["feature_means"])
            self._feature_stds = np.array(data["feature_stds"])

        logger.info(f"Online learner loaded from {path}")


class OnlineBandit:
    """Multi-armed bandit for adaptive strategy selection.

    Uses Thompson Sampling for exploration/exploitation balance.
    """

    def __init__(self, n_arms: int = 5, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms) * alpha_prior
        self.beta = np.ones(n_arms) * beta_prior
        self.arm_pulls = np.zeros(n_arms)
        self.arm_rewards = np.zeros(n_arms)

    def select_arm(self) -> int:
        """Select arm using Thompson Sampling."""
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        """Update arm statistics with observed reward."""
        self.arm_pulls[arm] += 1
        self.arm_rewards[arm] += reward

        if reward > 0:
            self.alpha[arm] += reward
        else:
            self.beta[arm] += abs(reward)

    def get_best_arm(self) -> int:
        """Get arm with highest empirical mean."""
        empirical_means = np.where(self.arm_pulls > 0, self.arm_rewards / self.arm_pulls, 0)
        return int(np.argmax(empirical_means))

    def get_arm_stats(self) -> list[dict[str, float]]:
        """Get statistics for all arms."""
        return [
            {
                "arm": i,
                "pulls": self.arm_pulls[i],
                "total_reward": self.arm_rewards[i],
                "mean_reward": self.arm_rewards[i] / max(1, self.arm_pulls[i]),
                "alpha": self.alpha[i],
                "beta": self.beta[i],
            }
            for i in range(self.n_arms)
        ]
