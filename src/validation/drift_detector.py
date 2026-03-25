"""
Drift Detector - Parameter staleness and model drift detection.

Monitors for:
- Feature distribution drift (covariate shift)
- Prediction accuracy degradation
- Parameter staleness (time since last update)
- Regime-prediction mismatch

Triggers recalibration when drift exceeds thresholds.
"""

import asyncio
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Type of drift detected."""
    NONE = "none"
    FEATURE_DISTRIBUTION = "feature_distribution"
    ACCURACY_DEGRADATION = "accuracy_degradation"
    PARAMETER_STALENESS = "parameter_staleness"
    REGIME_MISMATCH = "regime_mismatch"
    PREDICTION_BIAS = "prediction_bias"


class DriftSeverity(Enum):
    """Severity level of detected drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Alert for detected drift."""
    drift_type: DriftType
    severity: DriftSeverity
    score: float  # 0.0 to 1.0
    timestamp_ms: int
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "score": self.score,
            "timestamp_ms": self.timestamp_ms,
            "description": self.description,
            "details": self.details,
            "recommended_action": self.recommended_action,
        }


@dataclass
class DriftDetectorConfig:
    """Configuration for drift detection."""
    # Feature drift detection (KS test)
    feature_drift_threshold: float = 0.1  # KS statistic threshold
    feature_drift_p_value: float = 0.05  # P-value for significance
    min_samples_for_drift_test: int = 100

    # Accuracy monitoring
    accuracy_window_size: int = 200  # Predictions to track
    accuracy_degradation_threshold: float = 0.15  # 15% drop from baseline

    # Parameter staleness
    max_parameter_age_hours: float = 24.0
    stale_warning_hours: float = 12.0

    # Regime mismatch
    regime_mismatch_window: int = 50  # Recent predictions
    regime_mismatch_threshold: float = 0.4  # 40% mismatch rate

    # Prediction bias
    bias_window_size: int = 100
    bias_threshold: float = 0.1  # Mean absolute bias

    # Alert thresholds
    low_severity_score: float = 0.3
    medium_severity_score: float = 0.5
    high_severity_score: float = 0.7
    critical_severity_score: float = 0.9

    # Check interval
    check_interval_seconds: float = 60.0


@dataclass
class PredictionRecord:
    """Record of a single prediction for drift tracking."""
    timestamp_ms: int
    predicted_probability: float
    predicted_class: bool
    actual_class: Optional[bool] = None
    regime_at_prediction: str = "chop"
    current_regime: Optional[str] = None
    features: Optional[np.ndarray] = None


@dataclass
class DriftState:
    """Current state of drift detection."""
    last_check_ms: int = 0
    current_severity: DriftSeverity = DriftSeverity.NONE
    active_alerts: List[DriftAlert] = field(default_factory=list)

    # Tracking metrics
    baseline_accuracy: float = 0.5
    current_accuracy: float = 0.5
    current_bias: float = 0.0
    parameter_age_hours: float = 0.0
    regime_mismatch_rate: float = 0.0

    # Feature distributions
    reference_feature_stats: Optional[Dict[str, Tuple[float, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "last_check_ms": self.last_check_ms,
            "current_severity": self.current_severity.value,
            "n_active_alerts": len(self.active_alerts),
            "baseline_accuracy": self.baseline_accuracy,
            "current_accuracy": self.current_accuracy,
            "current_bias": self.current_bias,
            "parameter_age_hours": self.parameter_age_hours,
            "regime_mismatch_rate": self.regime_mismatch_rate,
        }


class DriftDetector:
    """Detects model drift and parameter staleness.

    Monitors prediction performance and feature distributions
    to detect when models need recalibration.
    """

    def __init__(
        self,
        config: Optional[DriftDetectorConfig] = None,
        on_drift_alert: Optional[Callable[[DriftAlert], None]] = None,
    ):
        self.config = config or DriftDetectorConfig()
        self.on_drift_alert = on_drift_alert

        # State
        self.state = DriftState()

        # Prediction history
        self._predictions: Deque[PredictionRecord] = deque(maxlen=self.config.accuracy_window_size)

        # Feature history for distribution comparison
        self._reference_features: List[np.ndarray] = []
        self._recent_features: Deque[np.ndarray] = deque(maxlen=500)

        # Parameter tracking
        self._last_parameter_update_ms: int = int(datetime.now().timestamp() * 1000)

        # Running state
        self._running = False

    def record_prediction(
        self,
        predicted_probability: float,
        predicted_class: bool,
        regime: str,
        features: Optional[np.ndarray] = None,
    ) -> None:
        """Record a new prediction for drift tracking."""
        now_ms = int(datetime.now().timestamp() * 1000)

        record = PredictionRecord(
            timestamp_ms=now_ms,
            predicted_probability=predicted_probability,
            predicted_class=predicted_class,
            regime_at_prediction=regime,
            features=features,
        )

        self._predictions.append(record)

        if features is not None:
            self._recent_features.append(features)

    def record_outcome(
        self,
        prediction_index: int,
        actual_class: bool,
        current_regime: Optional[str] = None,
    ) -> None:
        """Record the actual outcome for a previous prediction."""
        if prediction_index < 0 or prediction_index >= len(self._predictions):
            return

        record = self._predictions[prediction_index]
        record.actual_class = actual_class
        record.current_regime = current_regime

    def update_outcome_by_time(
        self,
        timestamp_ms: int,
        actual_class: bool,
        tolerance_ms: int = 5000,
    ) -> bool:
        """Update outcome for prediction matching timestamp.

        Returns True if a matching prediction was found.
        """
        for record in self._predictions:
            if abs(record.timestamp_ms - timestamp_ms) <= tolerance_ms:
                record.actual_class = actual_class
                return True
        return False

    def record_parameter_update(self) -> None:
        """Record that parameters were updated."""
        self._last_parameter_update_ms = int(datetime.now().timestamp() * 1000)
        self.state.parameter_age_hours = 0.0

    def set_reference_features(self, features: List[np.ndarray]) -> None:
        """Set reference feature distribution for drift comparison."""
        self._reference_features = features.copy()

        # Compute reference statistics
        if len(features) > 0:
            features_array = np.array(features)
            self.state.reference_feature_stats = {}
            for i in range(features_array.shape[1]):
                self.state.reference_feature_stats[f"feature_{i}"] = (
                    float(np.mean(features_array[:, i])),
                    float(np.std(features_array[:, i])),
                )

    def set_baseline_accuracy(self, accuracy: float) -> None:
        """Set baseline accuracy for degradation comparison."""
        self.state.baseline_accuracy = accuracy

    async def start_monitoring(self) -> None:
        """Start continuous drift monitoring."""
        self._running = True
        logger.info("Drift detector started")

        while self._running:
            try:
                alerts = self.check_drift()
                for alert in alerts:
                    if self.on_drift_alert:
                        try:
                            self.on_drift_alert(alert)
                        except Exception as e:
                            logger.error(f"Drift alert callback error: {e}")
            except Exception as e:
                logger.error(f"Drift check error: {e}")

            await asyncio.sleep(self.config.check_interval_seconds)

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._running = False
        logger.info("Drift detector stopped")

    def check_drift(self) -> List[DriftAlert]:
        """Run all drift checks and return any alerts."""
        now_ms = int(datetime.now().timestamp() * 1000)
        self.state.last_check_ms = now_ms
        self.state.active_alerts = []

        # Update parameter age
        self.state.parameter_age_hours = (now_ms - self._last_parameter_update_ms) / (1000 * 3600)

        alerts = []

        # 1. Check parameter staleness
        staleness_alert = self._check_staleness()
        if staleness_alert:
            alerts.append(staleness_alert)

        # 2. Check accuracy degradation
        accuracy_alert = self._check_accuracy()
        if accuracy_alert:
            alerts.append(accuracy_alert)

        # 3. Check prediction bias
        bias_alert = self._check_bias()
        if bias_alert:
            alerts.append(bias_alert)

        # 4. Check regime mismatch
        regime_alert = self._check_regime_mismatch()
        if regime_alert:
            alerts.append(regime_alert)

        # 5. Check feature drift
        feature_alert = self._check_feature_drift()
        if feature_alert:
            alerts.append(feature_alert)

        # Update state
        self.state.active_alerts = alerts
        if alerts:
            max_severity = max(alerts, key=lambda a: a.score)
            self.state.current_severity = max_severity.severity
        else:
            self.state.current_severity = DriftSeverity.NONE

        return alerts

    def _check_staleness(self) -> Optional[DriftAlert]:
        """Check for parameter staleness."""
        age = self.state.parameter_age_hours
        now_ms = int(datetime.now().timestamp() * 1000)

        if age >= self.config.max_parameter_age_hours:
            score = min(1.0, age / (self.config.max_parameter_age_hours * 1.5))
            return DriftAlert(
                drift_type=DriftType.PARAMETER_STALENESS,
                severity=self._score_to_severity(score),
                score=score,
                timestamp_ms=now_ms,
                description=f"Parameters are {age:.1f} hours old (max: {self.config.max_parameter_age_hours}h)",
                details={"age_hours": age, "max_hours": self.config.max_parameter_age_hours},
                recommended_action="Trigger full model retraining",
            )
        elif age >= self.config.stale_warning_hours:
            score = (age - self.config.stale_warning_hours) / (
                self.config.max_parameter_age_hours - self.config.stale_warning_hours
            )
            return DriftAlert(
                drift_type=DriftType.PARAMETER_STALENESS,
                severity=DriftSeverity.LOW,
                score=score * 0.5,
                timestamp_ms=now_ms,
                description=f"Parameters approaching staleness ({age:.1f}h old)",
                details={"age_hours": age},
                recommended_action="Schedule retraining",
            )

        return None

    def _check_accuracy(self) -> Optional[DriftAlert]:
        """Check for accuracy degradation."""
        # Get predictions with outcomes
        with_outcomes = [p for p in self._predictions if p.actual_class is not None]

        if len(with_outcomes) < self.config.min_samples_for_drift_test:
            return None

        # Calculate current accuracy
        correct = sum(1 for p in with_outcomes if p.predicted_class == p.actual_class)
        current_accuracy = correct / len(with_outcomes)
        self.state.current_accuracy = current_accuracy

        # Calculate degradation
        degradation = self.state.baseline_accuracy - current_accuracy

        if degradation >= self.config.accuracy_degradation_threshold:
            score = min(1.0, degradation / (self.config.accuracy_degradation_threshold * 2))
            now_ms = int(datetime.now().timestamp() * 1000)

            return DriftAlert(
                drift_type=DriftType.ACCURACY_DEGRADATION,
                severity=self._score_to_severity(score),
                score=score,
                timestamp_ms=now_ms,
                description=f"Accuracy dropped from {self.state.baseline_accuracy:.1%} to {current_accuracy:.1%}",
                details={
                    "baseline_accuracy": self.state.baseline_accuracy,
                    "current_accuracy": current_accuracy,
                    "degradation": degradation,
                    "n_samples": len(with_outcomes),
                },
                recommended_action="Trigger model recalibration",
            )

        return None

    def _check_bias(self) -> Optional[DriftAlert]:
        """Check for prediction bias."""
        with_outcomes = [p for p in self._predictions if p.actual_class is not None]

        if len(with_outcomes) < self.config.bias_window_size:
            return None

        # Use recent predictions
        recent = list(with_outcomes)[-self.config.bias_window_size:]

        # Calculate bias (predicted - actual)
        biases = [
            p.predicted_probability - (1.0 if p.actual_class else 0.0)
            for p in recent
        ]
        mean_bias = np.mean(biases)
        self.state.current_bias = float(mean_bias)

        if abs(mean_bias) >= self.config.bias_threshold:
            score = min(1.0, abs(mean_bias) / (self.config.bias_threshold * 2))
            now_ms = int(datetime.now().timestamp() * 1000)

            direction = "over-predicting" if mean_bias > 0 else "under-predicting"

            return DriftAlert(
                drift_type=DriftType.PREDICTION_BIAS,
                severity=self._score_to_severity(score),
                score=score,
                timestamp_ms=now_ms,
                description=f"Model is {direction} by {abs(mean_bias):.1%}",
                details={
                    "mean_bias": mean_bias,
                    "n_samples": len(recent),
                },
                recommended_action="Recalibrate prediction thresholds",
            )

        return None

    def _check_regime_mismatch(self) -> Optional[DriftAlert]:
        """Check for regime prediction mismatch."""
        # Get predictions with current regime info
        with_regime = [
            p for p in self._predictions
            if p.current_regime is not None and p.current_regime != ""
        ]

        if len(with_regime) < self.config.regime_mismatch_window:
            return None

        recent = list(with_regime)[-self.config.regime_mismatch_window:]

        # Calculate mismatch rate
        mismatches = sum(
            1 for p in recent
            if p.regime_at_prediction != p.current_regime
        )
        mismatch_rate = mismatches / len(recent)
        self.state.regime_mismatch_rate = mismatch_rate

        if mismatch_rate >= self.config.regime_mismatch_threshold:
            score = min(1.0, mismatch_rate / 0.6)
            now_ms = int(datetime.now().timestamp() * 1000)

            return DriftAlert(
                drift_type=DriftType.REGIME_MISMATCH,
                severity=self._score_to_severity(score),
                score=score,
                timestamp_ms=now_ms,
                description=f"Regime predictions mismatched {mismatch_rate:.1%} of the time",
                details={
                    "mismatch_rate": mismatch_rate,
                    "n_samples": len(recent),
                },
                recommended_action="Update regime detection model",
            )

        return None

    def _check_feature_drift(self) -> Optional[DriftAlert]:
        """Check for feature distribution drift using KS test."""
        if len(self._reference_features) < self.config.min_samples_for_drift_test:
            return None
        if len(self._recent_features) < self.config.min_samples_for_drift_test:
            return None

        reference_array = np.array(self._reference_features)
        recent_array = np.array(list(self._recent_features))

        # Run KS test on each feature
        n_features = min(reference_array.shape[1], recent_array.shape[1])
        drift_scores = []
        drifted_features = []

        for i in range(n_features):
            ref_col = reference_array[:, i]
            recent_col = recent_array[:, i]

            # Handle constant columns
            if np.std(ref_col) < 1e-10 or np.std(recent_col) < 1e-10:
                continue

            ks_stat, p_value = stats.ks_2samp(ref_col, recent_col)

            if ks_stat >= self.config.feature_drift_threshold and p_value < self.config.feature_drift_p_value:
                drift_scores.append(ks_stat)
                drifted_features.append(i)

        if len(drifted_features) > 0:
            max_drift = max(drift_scores)
            pct_drifted = len(drifted_features) / n_features
            score = min(1.0, max_drift + pct_drifted * 0.5)
            now_ms = int(datetime.now().timestamp() * 1000)

            return DriftAlert(
                drift_type=DriftType.FEATURE_DISTRIBUTION,
                severity=self._score_to_severity(score),
                score=score,
                timestamp_ms=now_ms,
                description=f"Feature drift detected in {len(drifted_features)}/{n_features} features",
                details={
                    "n_drifted": len(drifted_features),
                    "n_features": n_features,
                    "max_ks_stat": max_drift,
                    "drifted_indices": drifted_features[:10],  # Top 10
                },
                recommended_action="Retrain model with recent data",
            )

        return None

    def _score_to_severity(self, score: float) -> DriftSeverity:
        """Convert drift score to severity level."""
        if score >= self.config.critical_severity_score:
            return DriftSeverity.CRITICAL
        elif score >= self.config.high_severity_score:
            return DriftSeverity.HIGH
        elif score >= self.config.medium_severity_score:
            return DriftSeverity.MEDIUM
        elif score >= self.config.low_severity_score:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE

    def get_drift_score(self) -> float:
        """Get overall drift score (max of all drift types)."""
        if not self.state.active_alerts:
            return 0.0
        return max(a.score for a in self.state.active_alerts)

    def should_recalibrate(self, threshold: float = 0.7) -> bool:
        """Check if recalibration is recommended."""
        return self.get_drift_score() >= threshold

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "state": self.state.to_dict(),
            "n_predictions_tracked": len(self._predictions),
            "n_reference_features": len(self._reference_features),
            "n_recent_features": len(self._recent_features),
            "drift_score": self.get_drift_score(),
            "should_recalibrate": self.should_recalibrate(),
        }

    def clear_history(self) -> None:
        """Clear prediction and feature history."""
        self._predictions.clear()
        self._recent_features.clear()
        self.state.active_alerts = []
        self.state.current_severity = DriftSeverity.NONE

    def record_regime_change(self, old_regime: str, new_regime: str) -> None:
        """Record a regime transition for mismatch tracking.

        Updates the current_regime field on recent predictions to enable
        regime mismatch detection.
        """
        now_ms = int(datetime.now().timestamp() * 1000)

        # Update recent predictions with the new regime
        for record in self._predictions:
            # Update predictions from the last few seconds
            if now_ms - record.timestamp_ms < 60000:  # 60 seconds
                if record.current_regime is None or record.current_regime == old_regime:
                    record.current_regime = new_regime

        logger.debug(f"Recorded regime change: {old_regime} -> {new_regime}")
