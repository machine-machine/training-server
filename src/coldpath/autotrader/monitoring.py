"""
Monitoring and Alerting System - Real-time performance tracking and alerts.

This module provides comprehensive monitoring for:
- Trading performance (win rate, P&L, drawdown)
- Model health (prediction accuracy, calibration drift)
- System health (latency, error rates)
- Alert management with configurable thresholds

The system integrates with the AutoTrader coordinator to provide
real-time visibility into trading operations.
"""

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ============================================================================
# ALERT SYSTEM
# ============================================================================


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Categories of alerts."""

    PERFORMANCE = "performance"
    MODEL = "model"
    SYSTEM = "system"
    RISK = "risk"
    TRADING = "trading"


@dataclass
class Alert:
    """An alert notification."""

    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    acknowledged: bool = False
    acknowledged_at: datetime | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "title": self.title,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": round(self.metric_value, 4),
            "threshold": round(self.threshold, 4),
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "details": self.details,
        }


@dataclass
class AlertThreshold:
    """Threshold configuration for an alert."""

    metric_name: str
    warning_threshold: float | None = None
    error_threshold: float | None = None
    critical_threshold: float | None = None
    comparison: str = "less_than"  # less_than, greater_than, equals
    min_sample_size: int = 10
    cooldown_seconds: int = 300  # 5 minutes between alerts of same type

    def check(self, value: float, sample_size: int) -> AlertSeverity | None:
        """Check if value breaches threshold.

        Returns the severity level if breached, None otherwise.
        """
        if sample_size < self.min_sample_size:
            return None

        if self.comparison == "less_than":
            if self.critical_threshold is not None and value < self.critical_threshold:
                return AlertSeverity.CRITICAL
            if self.error_threshold is not None and value < self.error_threshold:
                return AlertSeverity.ERROR
            if self.warning_threshold is not None and value < self.warning_threshold:
                return AlertSeverity.WARNING

        elif self.comparison == "greater_than":
            if self.critical_threshold is not None and value > self.critical_threshold:
                return AlertSeverity.CRITICAL
            if self.error_threshold is not None and value > self.error_threshold:
                return AlertSeverity.ERROR
            if self.warning_threshold is not None and value > self.warning_threshold:
                return AlertSeverity.WARNING

        return None


# Default alert thresholds
DEFAULT_ALERT_THRESHOLDS: dict[str, AlertThreshold] = {
    # Performance alerts
    "win_rate": AlertThreshold(
        metric_name="win_rate",
        warning_threshold=0.45,
        error_threshold=0.35,
        critical_threshold=0.25,
        comparison="less_than",
        min_sample_size=20,
    ),
    "daily_pnl_pct": AlertThreshold(
        metric_name="daily_pnl_pct",
        warning_threshold=0.3,
        error_threshold=0.1,
        critical_threshold=-0.5,
        comparison="less_than",
        min_sample_size=10,
    ),
    "max_drawdown_pct": AlertThreshold(
        metric_name="max_drawdown_pct",
        warning_threshold=15.0,
        error_threshold=20.0,
        critical_threshold=25.0,
        comparison="greater_than",
        min_sample_size=5,
    ),
    "consecutive_losses": AlertThreshold(
        metric_name="consecutive_losses",
        warning_threshold=5,
        error_threshold=8,
        critical_threshold=12,
        comparison="greater_than",
        min_sample_size=1,
    ),
    # Model health alerts
    "model_correlation": AlertThreshold(
        metric_name="model_correlation",
        warning_threshold=0.85,
        error_threshold=0.75,
        critical_threshold=0.60,
        comparison="less_than",
        min_sample_size=50,
    ),
    "prediction_calibration": AlertThreshold(
        metric_name="prediction_calibration",
        warning_threshold=0.08,
        error_threshold=0.12,
        critical_threshold=0.20,
        comparison="greater_than",
        min_sample_size=50,
    ),
    # System health alerts
    "execution_latency_ms": AlertThreshold(
        metric_name="execution_latency_ms",
        warning_threshold=200,
        error_threshold=500,
        critical_threshold=1000,
        comparison="greater_than",
        min_sample_size=5,
    ),
    "signal_rejection_rate": AlertThreshold(
        metric_name="signal_rejection_rate",
        warning_threshold=0.30,
        error_threshold=0.50,
        critical_threshold=0.70,
        comparison="greater_than",
        min_sample_size=20,
    ),
    # Training pipeline alerts
    "training_consecutive_failures": AlertThreshold(
        metric_name="training_consecutive_failures",
        warning_threshold=2,
        error_threshold=3,
        critical_threshold=5,
        comparison="greater_than",
        min_sample_size=1,
    ),
    "training_success_rate": AlertThreshold(
        metric_name="training_success_rate",
        warning_threshold=0.70,
        error_threshold=0.50,
        critical_threshold=0.30,
        comparison="less_than",
        min_sample_size=5,
    ),
    "training_hours_since_success": AlertThreshold(
        metric_name="training_hours_since_success",
        warning_threshold=6.0,
        error_threshold=12.0,
        critical_threshold=24.0,
        comparison="greater_than",
        min_sample_size=1,
    ),
    "student_model_correlation": AlertThreshold(
        metric_name="student_model_correlation",
        warning_threshold=0.85,
        error_threshold=0.80,
        critical_threshold=0.70,
        comparison="less_than",
        min_sample_size=1,
    ),
    "student_calibration_error": AlertThreshold(
        metric_name="student_calibration_error",
        warning_threshold=0.05,
        error_threshold=0.08,
        critical_threshold=0.12,
        comparison="greater_than",
        min_sample_size=1,
    ),
}


class AlertManager:
    """Manages alert generation, tracking, and notification.

    Features:
    - Configurable thresholds for multiple metrics
    - Alert cooldown to prevent spam
    - Alert history tracking
    - Acknowledgment workflow
    - Notification callbacks
    """

    def __init__(
        self,
        thresholds: dict[str, AlertThreshold] | None = None,
        max_history: int = 500,
    ):
        self.thresholds = thresholds or DEFAULT_ALERT_THRESHOLDS.copy()
        self.max_history = max_history

        # Alert tracking
        self._alert_history: list[Alert] = []
        self._last_alert_time: dict[str, datetime] = {}
        self._active_alerts: dict[str, Alert] = {}

        # Notification callbacks
        self._callbacks: list[Callable[[Alert], None]] = []

        # Alert counters
        self._alerts_by_severity: dict[AlertSeverity, int] = {s: 0 for s in AlertSeverity}
        self._alerts_by_category: dict[AlertCategory, int] = {c: 0 for c in AlertCategory}

    def check_metric(
        self,
        metric_name: str,
        value: float,
        sample_size: int = 100,
        category: AlertCategory = AlertCategory.PERFORMANCE,
        details: dict[str, Any] | None = None,
    ) -> Alert | None:
        """Check a metric against thresholds and generate alert if breached.

        Args:
            metric_name: Name of the metric being checked
            value: Current value of the metric
            sample_size: Number of samples used to calculate metric
            category: Alert category
            details: Additional context

        Returns:
            Alert if threshold breached, None otherwise
        """
        if metric_name not in self.thresholds:
            return None

        threshold = self.thresholds[metric_name]
        severity = threshold.check(value, sample_size)

        if severity is None:
            return None

        # Check cooldown
        if self._is_in_cooldown(metric_name, threshold.cooldown_seconds):
            return None

        # Generate alert
        alert = self._create_alert(
            metric_name=metric_name,
            value=value,
            threshold_config=threshold,
            severity=severity,
            category=category,
            details=details or {},
        )

        # Record alert
        self._record_alert(alert)

        return alert

    def _is_in_cooldown(self, metric_name: str, cooldown_seconds: int) -> bool:
        """Check if metric is in alert cooldown."""
        if metric_name not in self._last_alert_time:
            return False

        elapsed = (datetime.now() - self._last_alert_time[metric_name]).total_seconds()
        return elapsed < cooldown_seconds

    def _create_alert(
        self,
        metric_name: str,
        value: float,
        threshold_config: AlertThreshold,
        severity: AlertSeverity,
        category: AlertCategory,
        details: dict[str, Any],
    ) -> Alert:
        """Create an alert."""
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{metric_name}"

        # Determine threshold value that was breached
        if severity == AlertSeverity.CRITICAL:
            threshold_value = threshold_config.critical_threshold
        elif severity == AlertSeverity.ERROR:
            threshold_value = threshold_config.error_threshold
        else:
            threshold_value = threshold_config.warning_threshold

        # Generate title and message
        title = f"{metric_name.replace('_', ' ').title()} Alert"

        if threshold_config.comparison == "less_than":
            message = (
                f"{metric_name} is {value:.4f}, below {severity.value} threshold "
                f"of {threshold_value:.4f}"
            )
        else:
            message = (
                f"{metric_name} is {value:.4f}, above {severity.value} threshold "
                f"of {threshold_value:.4f}"
            )

        return Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            title=title,
            message=message,
            metric_name=metric_name,
            metric_value=value,
            threshold=threshold_value or 0,
            details=details,
        )

    def _record_alert(self, alert: Alert) -> None:
        """Record an alert and trigger notifications."""
        # Update tracking
        self._last_alert_time[alert.metric_name] = alert.timestamp
        self._alert_history.append(alert)
        self._active_alerts[alert.alert_id] = alert

        # Update counters
        self._alerts_by_severity[alert.severity] += 1
        self._alerts_by_category[alert.category] += 1

        # Trim history
        if len(self._alert_history) > self.max_history:
            removed = self._alert_history[: -self.max_history]
            for r in removed:
                self._active_alerts.pop(r.alert_id, None)
            self._alert_history = self._alert_history[-self.max_history :]

        # Log alert
        log_msg = f"🚨 {alert.severity.value.upper()}: {alert.message}"
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(log_msg)
        elif alert.severity == AlertSeverity.ERROR:
            logger.error(log_msg)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.warning(f"Alert callback error: {e}")

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self._active_alerts:
            return False

        alert = self._active_alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_at = datetime.now()

        logger.info(f"Alert acknowledged: {alert_id}")
        return True

    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback to be called when alerts are generated."""
        self._callbacks.append(callback)

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[Alert]:
        """Get active (unacknowledged) alerts."""
        alerts = [a for a in self._alert_history if not a.acknowledged]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    def get_alert_history(
        self,
        limit: int = 50,
        severity: AlertSeverity | None = None,
        category: AlertCategory | None = None,
    ) -> list[dict[str, Any]]:
        """Get alert history with optional filtering."""
        alerts = self._alert_history

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if category:
            alerts = [a for a in alerts if a.category == category]

        return [a.to_dict() for a in alerts[-limit:]]

    def get_statistics(self) -> dict[str, Any]:
        """Get alert statistics."""
        total = len(self._alert_history)
        active = len(self.get_active_alerts())

        return {
            "total_alerts": total,
            "active_alerts": active,
            "by_severity": {s.value: count for s, count in self._alerts_by_severity.items()},
            "by_category": {cat.value: count for cat, count in self._alerts_by_category.items()},
            "last_24h": len(
                [
                    a
                    for a in self._alert_history
                    if (datetime.now() - a.timestamp).total_seconds() < 86400
                ]
            ),
        }

    def clear_history(self) -> None:
        """Clear alert history."""
        self._alert_history.clear()
        self._active_alerts.clear()
        self._last_alert_time.clear()
        logger.info("Alert history cleared")


# ============================================================================
# MODEL PERFORMANCE MONITOR
# ============================================================================


@dataclass
class ModelMetrics:
    """Metrics for tracking ML model performance."""

    timestamp: datetime
    correlation: float
    calibration_error: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sample_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "correlation": round(self.correlation, 4),
            "calibration_error": round(self.calibration_error, 4),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "sample_size": self.sample_size,
        }


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""

    detected: bool
    drift_type: str  # "concept_drift", "data_drift", "prediction_drift"
    severity: str  # "low", "medium", "high"
    affected_features: list[str]
    recommendation: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "drift_type": self.drift_type,
            "severity": self.severity,
            "affected_features": self.affected_features,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat(),
        }


class ModelPerformanceMonitor:
    """Monitors ML model performance and detects drift.

    Features:
    - Tracks prediction accuracy over time
    - Detects model drift (concept, data, prediction)
    - Triggers retrain recommendations
    - Calibration tracking
    """

    def __init__(
        self,
        history_size: int = 1000,
        drift_window: int = 100,
        retrain_threshold: float = 0.15,
    ):
        self.history_size = history_size
        self.drift_window = drift_window
        self.retrain_threshold = retrain_threshold

        # Prediction tracking
        self._predictions: deque = deque(maxlen=history_size)
        self._outcomes: deque = deque(maxlen=history_size)

        # Metrics history
        self._metrics_history: deque = deque(maxlen=100)

        # Baseline metrics (from training)
        self._baseline_metrics: ModelMetrics | None = None

        # Drift tracking — FIX: Use deque with maxlen to prevent unbounded growth
        # Over 14 days at 1 check/min = 20,160 entries without this fix
        self._drift_history: deque = deque(maxlen=500)
        self._retrain_recommended: bool = False
        self._last_retrain: datetime | None = None

    def record_prediction(
        self,
        prediction: float,
        confidence: float,
        features: dict[str, float] | None = None,
    ) -> None:
        """Record a model prediction for tracking."""
        self._predictions.append(
            {
                "prediction": prediction,
                "confidence": confidence,
                "features": features or {},
                "timestamp": datetime.now(),
            }
        )

    def record_outcome(
        self,
        prediction_id: int,
        actual_outcome: float,
    ) -> None:
        """Record the actual outcome for a prediction."""
        self._outcomes.append(
            {
                "prediction_id": prediction_id,
                "actual": actual_outcome,
                "timestamp": datetime.now(),
            }
        )

    def set_baseline(self, metrics: ModelMetrics) -> None:
        """Set baseline metrics from training."""
        self._baseline_metrics = metrics
        logger.info(
            f"Model baseline set: correlation={metrics.correlation:.3f}, "
            f"calibration_error={metrics.calibration_error:.4f}"
        )

    def calculate_current_metrics(self) -> ModelMetrics | None:
        """Calculate current model performance metrics."""
        if len(self._predictions) < 50 or len(self._outcomes) < 50:
            return None

        # Match predictions with outcomes
        matched = []
        pred_list = list(self._predictions)
        outcome_list = list(self._outcomes)

        # Simple matching by order (would need proper ID matching in production)
        for _i, (pred, outcome) in enumerate(
            zip(pred_list[-100:], outcome_list[-100:], strict=False)
        ):
            matched.append(
                {
                    "prediction": pred["prediction"],
                    "confidence": pred["confidence"],
                    "actual": outcome["actual"],
                }
            )

        if not matched:
            return None

        # Calculate metrics
        predictions = [m["prediction"] for m in matched]
        actuals = [m["actual"] for m in matched]
        [m["confidence"] for m in matched]

        # Correlation (Pearson)
        if len(predictions) > 1:
            pred_mean = sum(predictions) / len(predictions)
            actual_mean = sum(actuals) / len(actuals)

            numerator = sum(
                (p - pred_mean) * (a - actual_mean)
                for p, a in zip(predictions, actuals, strict=False)
            )
            pred_var = sum((p - pred_mean) ** 2 for p in predictions)
            actual_var = sum((a - actual_mean) ** 2 for a in actuals)
            denominator = math.sqrt(pred_var * actual_var)

            correlation = numerator / denominator if denominator > 0 else 0
        else:
            correlation = 0

        # Calibration error (simplified ECE)
        # Bin predictions by confidence and compare with actual accuracy
        bins = defaultdict(list)
        for m in matched:
            bin_idx = int(m["confidence"] * 10)  # 10 bins
            bins[bin_idx].append(m)

        calibration_error = 0.0
        total_samples = 0
        for _bin_idx, bin_samples in bins.items():
            if not bin_samples:
                continue
            avg_confidence = sum(s["confidence"] for s in bin_samples) / len(bin_samples)
            accuracy = sum(1 for s in bin_samples if s["actual"] > 0) / len(bin_samples)
            calibration_error += abs(avg_confidence - accuracy) * len(bin_samples)
            total_samples += len(bin_samples)

        calibration_error /= max(1, total_samples)

        # Accuracy (for binary classification proxy)
        threshold = 0.5
        correct = sum(1 for m in matched if (m["prediction"] >= threshold) == (m["actual"] > 0))
        accuracy = correct / len(matched)

        # Precision/Recall
        true_positives = sum(1 for m in matched if m["prediction"] >= threshold and m["actual"] > 0)
        false_positives = sum(
            1 for m in matched if m["prediction"] >= threshold and m["actual"] <= 0
        )
        false_negatives = sum(1 for m in matched if m["prediction"] < threshold and m["actual"] > 0)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = ModelMetrics(
            timestamp=datetime.now(),
            correlation=correlation,
            calibration_error=calibration_error,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            sample_size=len(matched),
        )

        self._metrics_history.append(metrics)
        return metrics

    def detect_drift(self) -> DriftDetectionResult | None:
        """Detect model drift by comparing current vs baseline metrics."""
        if self._baseline_metrics is None:
            return None

        current = self.calculate_current_metrics()
        if current is None:
            return None

        baseline = self._baseline_metrics

        # Check for performance degradation
        correlation_drift = baseline.correlation - current.correlation
        calibration_drift = current.calibration_error - baseline.calibration_error

        drift_detected = False
        drift_type = ""
        severity = "low"
        affected = []
        recommendation = ""

        # Check correlation drift
        if correlation_drift > self.retrain_threshold:
            drift_detected = True
            drift_type = "concept_drift"
            severity = "high" if correlation_drift > 0.25 else "medium"
            affected.append("correlation")
            recommendation = "Model performance has degraded significantly. Retraining recommended."

        # Check calibration drift
        elif calibration_drift > 0.05:
            drift_detected = True
            drift_type = "prediction_drift"
            severity = "medium" if calibration_drift > 0.10 else "low"
            affected.append("calibration")
            recommendation = "Model calibration has drifted. Consider recalibration or retraining."

        if drift_detected:
            result = DriftDetectionResult(
                detected=True,
                drift_type=drift_type,
                severity=severity,
                affected_features=affected,
                recommendation=recommendation,
                timestamp=datetime.now(),
            )
            self._drift_history.append(result)
            self._retrain_recommended = severity in ["medium", "high"]

            logger.warning(
                f"🔄 Drift detected: type={drift_type}, severity={severity}, "
                f"correlation_drop={correlation_drift:.3f}"
            )

            return result

        # No drift detected
        self._retrain_recommended = False
        return None

    def should_retrain(self) -> bool:
        """Check if model retraining is recommended."""
        # Check drift recommendation
        if self._retrain_recommended:
            return True

        # Check time since last retrain
        min_retrain_interval = timedelta(hours=24)
        if self._last_retrain:
            if datetime.now() - self._last_retrain < min_retrain_interval:
                return False

        return False

    def mark_retrained(self) -> None:
        """Mark that the model has been retrained."""
        self._last_retrain = datetime.now()
        self._retrain_recommended = False
        logger.info("Model marked as retrained")

    def get_status(self) -> dict[str, Any]:
        """Get current monitor status."""
        current = self.calculate_current_metrics()

        return {
            "predictions_tracked": len(self._predictions),
            "outcomes_tracked": len(self._outcomes),
            "baseline_set": self._baseline_metrics is not None,
            "baseline_metrics": self._baseline_metrics.to_dict()
            if self._baseline_metrics
            else None,
            "current_metrics": current.to_dict() if current else None,
            "retrain_recommended": self._retrain_recommended,
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "drift_detections": len(self._drift_history),
            "metrics_history_count": len(self._metrics_history),
        }

    def get_metrics_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get historical metrics."""
        return [m.to_dict() for m in list(self._metrics_history)[-limit:]]

    def get_drift_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get drift detection history."""
        return [d.to_dict() for d in self._drift_history[-limit:]]


# ============================================================================
# SYSTEM MONITOR
# ============================================================================


@dataclass
class SystemHealth:
    """System health status."""

    timestamp: datetime
    autotrader_state: str
    hotpath_connected: bool
    db_healthy: bool
    model_loaded: bool
    pending_signals: int
    active_positions: int
    daily_trades: int
    daily_volume_sol: float
    daily_pnl_sol: float
    avg_latency_ms: float
    error_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "autotrader_state": self.autotrader_state,
            "hotpath_connected": self.hotpath_connected,
            "db_healthy": self.db_healthy,
            "model_loaded": self.model_loaded,
            "pending_signals": self.pending_signals,
            "active_positions": self.active_positions,
            "daily_trades": self.daily_trades,
            "daily_volume_sol": round(self.daily_volume_sol, 4),
            "daily_pnl_sol": round(self.daily_pnl_sol, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
        }


class SystemMonitor:
    """Monitors overall system health and performance.

    Features:
    - Component health checks
    - Latency tracking
    - Error rate monitoring
    - Resource utilization
    """

    def __init__(self, history_size: int = 100):
        self.history_size = history_size

        # Health history
        self._health_history: deque = deque(maxlen=history_size)

        # Latency tracking
        self._latencies: deque = deque(maxlen=100)

        # Error tracking
        self._errors: deque = deque(maxlen=100)
        self._total_operations: int = 0

    def record_latency(self, latency_ms: float) -> None:
        """Record an operation latency."""
        self._latencies.append(latency_ms)

    def record_error(self, error: str) -> None:
        """Record an error occurrence."""
        self._errors.append(
            {
                "error": error,
                "timestamp": datetime.now(),
            }
        )

    def record_operation(self) -> None:
        """Record a successful operation."""
        self._total_operations += 1

    def capture_health(
        self,
        autotrader_state: str = "unknown",
        hotpath_connected: bool = False,
        db_healthy: bool = True,
        model_loaded: bool = True,
        pending_signals: int = 0,
        active_positions: int = 0,
        daily_trades: int = 0,
        daily_volume_sol: float = 0.0,
        daily_pnl_sol: float = 0.0,
    ) -> SystemHealth:
        """Capture current system health snapshot."""
        # Calculate average latency
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0

        # Calculate error rate (last hour)
        recent_errors = len(
            [e for e in self._errors if (datetime.now() - e["timestamp"]).total_seconds() < 3600]
        )
        error_rate = recent_errors / max(1, self._total_operations)

        health = SystemHealth(
            timestamp=datetime.now(),
            autotrader_state=autotrader_state,
            hotpath_connected=hotpath_connected,
            db_healthy=db_healthy,
            model_loaded=model_loaded,
            pending_signals=pending_signals,
            active_positions=active_positions,
            daily_trades=daily_trades,
            daily_volume_sol=daily_volume_sol,
            daily_pnl_sol=daily_pnl_sol,
            avg_latency_ms=avg_latency,
            error_rate=error_rate,
        )

        self._health_history.append(health)
        return health

    def get_health_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get health history."""
        return [h.to_dict() for h in list(self._health_history)[-limit:]]

    def get_latest_health(self) -> SystemHealth | None:
        """Get latest health snapshot."""
        return self._health_history[-1] if self._health_history else None

    def get_summary(self) -> dict[str, Any]:
        """Get system health summary."""
        latest = self.get_latest_health()

        return {
            "current_health": latest.to_dict() if latest else None,
            "total_operations": self._total_operations,
            "recent_errors": len(
                [
                    e
                    for e in self._errors
                    if (datetime.now() - e["timestamp"]).total_seconds() < 3600
                ]
            ),
            "avg_latency_24h": (
                sum(self._latencies) / len(self._latencies) if self._latencies else 0
            ),
            "health_snapshots": len(self._health_history),
        }


# ============================================================================
# UNIFIED MONITORING SERVICE
# ============================================================================


class MonitoringService:
    """Unified monitoring service combining all monitoring components.

    Provides a single interface for:
    - Alert management
    - Model performance monitoring
    - System health monitoring
    - Dashboard metrics
    """

    def __init__(
        self,
        alert_thresholds: dict[str, AlertThreshold] | None = None,
    ):
        self.alert_manager = AlertManager(thresholds=alert_thresholds)
        self.model_monitor = ModelPerformanceMonitor()
        self.system_monitor = SystemMonitor()

        # Monitoring state
        self._started_at: datetime | None = None
        self._enabled: bool = True

    def start(self) -> None:
        """Start the monitoring service."""
        self._started_at = datetime.now()
        logger.info("Monitoring service started")

    def stop(self) -> None:
        """Stop the monitoring service."""
        self._enabled = False
        logger.info("Monitoring service stopped")

    def check_all_metrics(
        self,
        metrics: dict[str, float],
        sample_sizes: dict[str, int] | None = None,
    ) -> list[Alert]:
        """Check all metrics against thresholds.

        Args:
            metrics: Dict of metric_name -> value
            sample_sizes: Optional dict of metric_name -> sample_size

        Returns:
            List of alerts generated
        """
        if not self._enabled:
            return []

        alerts = []
        sample_sizes = sample_sizes or {}

        for metric_name, value in metrics.items():
            sample_size = sample_sizes.get(metric_name, 100)

            # Determine category based on metric name
            if metric_name in ["model_correlation", "prediction_calibration"]:
                category = AlertCategory.MODEL
            elif metric_name in ["execution_latency_ms", "signal_rejection_rate"]:
                category = AlertCategory.SYSTEM
            elif metric_name in ["consecutive_losses", "max_drawdown_pct"]:
                category = AlertCategory.RISK
            else:
                category = AlertCategory.PERFORMANCE

            alert = self.alert_manager.check_metric(
                metric_name=metric_name,
                value=value,
                sample_size=sample_size,
                category=category,
            )

            if alert:
                alerts.append(alert)

        return alerts

    def get_dashboard_metrics(self) -> dict[str, Any]:
        """Get all metrics for a monitoring dashboard."""
        return {
            "monitoring": {
                "enabled": self._enabled,
                "started_at": self._started_at.isoformat() if self._started_at else None,
                "uptime_seconds": (
                    (datetime.now() - self._started_at).total_seconds() if self._started_at else 0
                ),
            },
            "alerts": self.alert_manager.get_statistics(),
            "model": self.model_monitor.get_status(),
            "system": self.system_monitor.get_summary(),
        }

    def get_full_report(self) -> dict[str, Any]:
        """Get a comprehensive monitoring report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "dashboard": self.get_dashboard_metrics(),
            "active_alerts": [a.to_dict() for a in self.alert_manager.get_active_alerts()],
            "recent_drift": self.model_monitor.get_drift_history(limit=5),
            "health_history": self.system_monitor.get_health_history(limit=10),
        }


# ============================================================================
# TRAINING PIPELINE MONITOR
# ============================================================================


class TrainingStage(Enum):
    """Stages of the training pipeline."""

    IDLE = "idle"
    DATA_COLLECTION = "data_collection"
    TEACHER_TRAINING = "teacher_training"
    STUDENT_DISTILLATION = "student_distillation"
    QUALITY_VALIDATION = "quality_validation"
    MODEL_EXPORT = "model_export"
    HOTPATH_PUSH = "hotpath_push"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingStatus(Enum):
    """Overall training status."""

    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TrainingRun:
    """Record of a single training run."""

    run_id: str
    started_at: datetime
    stage: TrainingStage
    status: TrainingStatus
    outcomes_used: int = 0
    teacher_metrics: dict[str, float] | None = None
    student_metrics: dict[str, float] | None = None
    quality_gates_passed: bool = False
    correlation: float | None = None
    calibration_error: float | None = None
    model_version: int | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "stage": self.stage.value,
            "status": self.status.value,
            "outcomes_used": self.outcomes_used,
            "teacher_metrics": self.teacher_metrics,
            "student_metrics": self.student_metrics,
            "quality_gates_passed": self.quality_gates_passed,
            "correlation": round(self.correlation, 4) if self.correlation else None,
            "calibration_error": round(self.calibration_error, 4)
            if self.calibration_error
            else None,
            "model_version": self.model_version,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "error_message": self.error_message,
        }


@dataclass
class TrainingPipelineStatus:
    """Current status of the training pipeline."""

    current_stage: TrainingStage
    current_status: TrainingStatus
    current_run_id: str | None
    last_success_at: datetime | None
    last_failure_at: datetime | None
    consecutive_failures: int
    total_runs: int
    successful_runs: int
    failed_runs: int
    current_model_version: int
    outcomes_since_last_train: int
    next_scheduled_train: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_stage": self.current_stage.value,
            "current_status": self.current_status.value,
            "current_run_id": self.current_run_id,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_failure_at": self.last_failure_at.isoformat() if self.last_failure_at else None,
            "consecutive_failures": self.consecutive_failures,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": round(self.successful_runs / max(1, self.total_runs), 3),
            "current_model_version": self.current_model_version,
            "outcomes_since_last_train": self.outcomes_since_last_train,
            "next_scheduled_train": self.next_scheduled_train.isoformat()
            if self.next_scheduled_train
            else None,
        }


class TrainingPipelineMonitor:
    """Monitors the ML training pipeline status and health.

    Features:
    - Track training runs (teacher + student distillation)
    - Monitor training success/failure rates
    - Track model versions
    - Alert on training failures or quality degradation
    - Schedule monitoring for automated retraining
    """

    def __init__(
        self,
        history_size: int = 100,
        max_consecutive_failures: int = 3,
        min_outcomes_for_training: int = 200,
        training_interval_hours: float = 1.0,
    ):
        self.history_size = history_size
        self.max_consecutive_failures = max_consecutive_failures
        self.min_outcomes_for_training = min_outcomes_for_training
        self.training_interval_hours = training_interval_hours

        # Training run history
        self._run_history: deque = deque(maxlen=history_size)

        # Current state
        self._current_run: TrainingRun | None = None
        self._current_stage: TrainingStage = TrainingStage.IDLE
        self._current_status: TrainingStatus = TrainingStatus.IDLE
        self._current_model_version: int = 0

        # Counters
        self._total_runs: int = 0
        self._successful_runs: int = 0
        self._failed_runs: int = 0
        self._consecutive_failures: int = 0

        # Timestamps
        self._last_success_at: datetime | None = None
        self._last_failure_at: datetime | None = None
        self._last_train_at: datetime | None = None
        self._outcomes_since_last_train: int = 0

        # Error tracking
        self._last_error: str | None = None
        self._error_count: int = 0

        # Stage timing
        self._stage_start_times: dict[TrainingStage, datetime] = {}

    def start_training(self, outcomes_available: int) -> str:
        """Start a new training run.

        Args:
            outcomes_available: Number of outcomes available for training

        Returns:
            run_id for tracking this run
        """
        import uuid

        run_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        self._current_run = TrainingRun(
            run_id=run_id,
            started_at=datetime.now(),
            stage=TrainingStage.DATA_COLLECTION,
            status=TrainingStatus.RUNNING,
            outcomes_used=outcomes_available,
        )

        self._current_stage = TrainingStage.DATA_COLLECTION
        self._current_status = TrainingStatus.RUNNING
        self._total_runs += 1

        self._stage_start_times = {TrainingStage.DATA_COLLECTION: datetime.now()}

        logger.info(f"📊 Training pipeline started: {run_id} with {outcomes_available} outcomes")

        return run_id

    def update_stage(self, stage: TrainingStage, details: dict[str, Any] | None = None) -> None:
        """Update the current training stage.

        Args:
            stage: The new training stage
            details: Optional details about the stage progress
        """
        if self._current_run is None:
            logger.warning(f"Cannot update stage to {stage.value}: no active training run")
            return

        prev_stage = self._current_stage
        self._current_stage = stage
        self._current_run.stage = stage
        self._stage_start_times[stage] = datetime.now()

        # Log stage transition
        if prev_stage != stage:
            duration = 0.0
            if prev_stage in self._stage_start_times:
                duration = (datetime.now() - self._stage_start_times[prev_stage]).total_seconds()
            logger.info(f"📊 Training stage: {prev_stage.value} → {stage.value} ({duration:.1f}s)")

        # Handle specific stage updates
        if details:
            if stage == TrainingStage.TEACHER_TRAINING and "metrics" in details:
                self._current_run.teacher_metrics = details["metrics"]
            elif stage == TrainingStage.STUDENT_DISTILLATION and "metrics" in details:
                self._current_run.student_metrics = details["metrics"]
            elif stage == TrainingStage.QUALITY_VALIDATION:
                if "correlation" in details:
                    self._current_run.correlation = details["correlation"]
                if "calibration_error" in details:
                    self._current_run.calibration_error = details["calibration_error"]
                if "passed" in details:
                    self._current_run.quality_gates_passed = details["passed"]
            elif stage == TrainingStage.MODEL_EXPORT and "version" in details:
                self._current_run.model_version = details["version"]

    def complete_training(self, success: bool, error_message: str | None = None) -> None:
        """Complete the current training run.

        Args:
            success: Whether the training completed successfully
            error_message: Error message if failed
        """
        if self._current_run is None:
            return

        self._current_run.completed_at = datetime.now()
        self._current_run.duration_seconds = (
            self._current_run.completed_at - self._current_run.started_at
        ).total_seconds()

        if success:
            self._current_run.status = TrainingStatus.SUCCESS
            self._current_run.stage = TrainingStage.COMPLETED
            self._current_status = TrainingStatus.SUCCESS
            self._successful_runs += 1
            self._consecutive_failures = 0
            self._last_success_at = datetime.now()
            self._last_train_at = datetime.now()
            self._outcomes_since_last_train = 0

            if self._current_run.model_version:
                self._current_model_version = self._current_run.model_version

            logger.info(
                f"✅ Training completed: {self._current_run.run_id} "
                f"(v{self._current_model_version}, {self._current_run.duration_seconds:.1f}s)"
            )
        else:
            self._current_run.status = TrainingStatus.FAILED
            self._current_run.stage = TrainingStage.FAILED
            self._current_run.error_message = error_message
            self._current_status = TrainingStatus.FAILED
            self._failed_runs += 1
            self._consecutive_failures += 1
            self._last_failure_at = datetime.now()
            self._last_error = error_message
            self._error_count += 1

            logger.error(f"❌ Training failed: {self._current_run.run_id} - {error_message}")

        # Record in history
        self._run_history.append(self._current_run)

        # Reset current run
        self._current_run = None
        self._current_stage = TrainingStage.IDLE

    def record_outcome(self, count: int = 1) -> None:
        """Record new training outcomes (for tracking when to retrain)."""
        self._outcomes_since_last_train += count

    def should_train(self) -> bool:
        """Check if training should be triggered.

        Returns True if:
        - Enough outcomes collected (>= min_outcomes_for_training)
        - Enough time since last training (>= training_interval_hours)
        - Not currently training
        - Not in consecutive failure cooldown
        """
        if self._current_status == TrainingStatus.RUNNING:
            return False

        if self._consecutive_failures >= self.max_consecutive_failures:
            # Need cooldown after multiple failures
            if self._last_failure_at:
                cooldown_hours = 1.0 * self._consecutive_failures  # Exponential backoff
                elapsed = (datetime.now() - self._last_failure_at).total_seconds() / 3600
                if elapsed < cooldown_hours:
                    return False

        # Check outcomes threshold
        if self._outcomes_since_last_train < self.min_outcomes_for_training:
            return False

        # Check time threshold
        if self._last_train_at:
            elapsed = (datetime.now() - self._last_train_at).total_seconds() / 3600
            if elapsed < self.training_interval_hours:
                return False

        return True

    def get_next_scheduled_train(self) -> datetime | None:
        """Get the estimated next training time."""
        if self._last_train_at is None:
            return datetime.now()

        next_time = self._last_train_at + timedelta(hours=self.training_interval_hours)

        # Adjust for consecutive failures
        if self._consecutive_failures > 0 and self._last_failure_at:
            cooldown_hours = 1.0 * self._consecutive_failures
            cooldown_end = self._last_failure_at + timedelta(hours=cooldown_hours)
            next_time = max(next_time, cooldown_end)

        return next_time

    def get_status(self) -> TrainingPipelineStatus:
        """Get current training pipeline status."""
        return TrainingPipelineStatus(
            current_stage=self._current_stage,
            current_status=self._current_status,
            current_run_id=self._current_run.run_id if self._current_run else None,
            last_success_at=self._last_success_at,
            last_failure_at=self._last_failure_at,
            consecutive_failures=self._consecutive_failures,
            total_runs=self._total_runs,
            successful_runs=self._successful_runs,
            failed_runs=self._failed_runs,
            current_model_version=self._current_model_version,
            outcomes_since_last_train=self._outcomes_since_last_train,
            next_scheduled_train=self.get_next_scheduled_train(),
        )

    def get_current_run(self) -> dict[str, Any] | None:
        """Get details of the current training run."""
        if self._current_run is None:
            return None
        return self._current_run.to_dict()

    def get_run_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get history of training runs."""
        return [r.to_dict() for r in list(self._run_history)[-limit:]]

    def get_summary(self) -> dict[str, Any]:
        """Get training pipeline summary for dashboards."""
        return {
            "status": self.get_status().to_dict(),
            "current_run": self.get_current_run(),
            "recent_runs": self.get_run_history(limit=5),
            # Backward-compatible top-level field used by existing tests/dashboards.
            "outcomes_since_last_train": self._outcomes_since_last_train,
            "config": {
                "min_outcomes_for_training": self.min_outcomes_for_training,
                "training_interval_hours": self.training_interval_hours,
                "max_consecutive_failures": self.max_consecutive_failures,
            },
            "should_train_now": self.should_train(),
            "outcomes_until_training": max(
                0, self.min_outcomes_for_training - self._outcomes_since_last_train
            ),
        }

    def reset_failure_count(self) -> None:
        """Reset consecutive failure counter (after manual fix)."""
        self._consecutive_failures = 0
        logger.info("Training pipeline failure count reset")
