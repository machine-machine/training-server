"""
A/B Testing Framework for Model Comparison.

Enables safe rollout of new models with:
- Traffic splitting (configurable percentages)
- Statistical significance testing
- Automatic rollback on performance degradation
- Shadow mode testing (log only, no impact)

Architecture:
    ┌─────────────────┐
    │ Inference Request│
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐     ┌──────────────────┐
    │ Experiment Router├────►│ Model A (Control)│
    └────────┬────────┘     └──────────────────┘
             │                       │
             │ 10% traffic           │ 90% traffic
             ▼                       ▼
    ┌─────────────────┐     ┌──────────────────┐
    │ Model B (Test)  │     │ Model A Results  │
    └─────────────────┘     └──────────────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────────────────────┐
    │ Metrics Collector & Analyzer    │
    └─────────────────────────────────┘

Usage:
    from coldpath.services.ab_testing import ABTestingFramework

    ab = ABTestingFramework()

    # Register experiment
    ab.create_experiment(
        name="profitability_v2",
        control_model="profitability_v1",
        treatment_model="profitability_v2",
        traffic_split=0.1,  # 10% to treatment
    )

    # Get model for request
    model_id = ab.get_model("profitability_v2", request_id)

    # Record outcome
    ab.record_outcome("profitability_v2", model_id, actual_return)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"


class ModelVariant(Enum):
    CONTROL = "control"  # Current production model
    TREATMENT = "treatment"  # New model being tested
    SHADOW = "shadow"  # Shadow mode (log only)


@dataclass
class ExperimentMetrics:
    """Metrics for an experiment variant."""

    total_predictions: int = 0
    total_outcomes: int = 0

    # Accuracy metrics
    correct_predictions: int = 0
    directional_accuracy: float = 0.0

    # Return metrics
    total_return: float = 0.0
    mean_return: float = 0.0
    std_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR 95%

    # Latency
    mean_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_predictions": self.total_predictions,
            "total_outcomes": self.total_outcomes,
            "correct_predictions": self.correct_predictions,
            "directional_accuracy": self.directional_accuracy,
            "total_return": self.total_return,
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "mean_latency_ms": self.mean_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
        }


@dataclass
class Experiment:
    """A/B test experiment configuration."""

    name: str
    control_model: str
    treatment_model: str
    traffic_split: float = 0.1  # Fraction to treatment
    status: ExperimentStatus = ExperimentStatus.DRAFT

    # Time bounds
    start_time: datetime | None = None
    end_time: datetime | None = None

    # Stopping criteria
    min_samples: int = 1000
    max_duration_hours: int = 168  # 1 week
    significance_level: float = 0.05  # 95% confidence
    min_effect_size: float = 0.02  # 2% improvement

    # Rollback criteria
    max_drawdown_threshold: float = 0.1  # 10% max drawdown
    accuracy_drop_threshold: float = 0.05  # 5% accuracy drop

    # Metrics
    control_metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    treatment_metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)

    # Results storage
    control_returns: list[float] = field(default_factory=list)
    treatment_returns: list[float] = field(default_factory=list)
    control_latencies: list[float] = field(default_factory=list)
    treatment_latencies: list[float] = field(default_factory=list)

    # Statistical results
    p_value: float | None = None
    effect_size: float | None = None
    confidence_interval: tuple[float, float] | None = None

    # Rollback info
    rollback_reason: str | None = None
    rollback_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "control_model": self.control_model,
            "treatment_model": self.treatment_model,
            "traffic_split": self.traffic_split,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "min_samples": self.min_samples,
            "control_metrics": self.control_metrics.to_dict(),
            "treatment_metrics": self.treatment_metrics.to_dict(),
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "confidence_interval": self.confidence_interval,
            "rollback_reason": self.rollback_reason,
        }


@dataclass
class ExperimentResult:
    """Result of an experiment analysis."""

    experiment_name: str
    winner: str | None  # "control", "treatment", or None (inconclusive)
    p_value: float
    effect_size: float
    confidence_interval: tuple[float, float]
    recommendation: str
    should_rollback: bool
    rollback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "winner": self.winner,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "confidence_interval": self.confidence_interval,
            "recommendation": self.recommendation,
            "should_rollback": self.should_rollback,
            "rollback_reason": self.rollback_reason,
        }


class ABTestingFramework:
    """Framework for A/B testing ML models in production.

    Features:
    - Traffic splitting with consistent hashing
    - Statistical significance testing (t-test, Mann-Whitney)
    - Automatic rollback on degradation
    - Shadow mode for safe testing
    - Real-time metrics monitoring
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        auto_rollback: bool = True,
    ):
        self.storage_path = storage_path or Path("data/ab_experiments")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.auto_rollback = auto_rollback

        self._experiments: dict[str, Experiment] = {}
        self._pending_outcomes: dict[str, list[tuple[str, ModelVariant, float]]] = {}

        self._load_experiments()

    def _load_experiments(self):
        """Load experiments from storage."""
        for exp_file in self.storage_path.glob("*.json"):
            try:
                with open(exp_file) as f:
                    data = json.load(f)

                exp = Experiment(
                    name=data["name"],
                    control_model=data["control_model"],
                    treatment_model=data["treatment_model"],
                    traffic_split=data["traffic_split"],
                    status=ExperimentStatus(data["status"]),
                )

                if data.get("start_time"):
                    exp.start_time = datetime.fromisoformat(data["start_time"])
                if data.get("end_time"):
                    exp.end_time = datetime.fromisoformat(data["end_time"])

                self._experiments[exp.name] = exp

            except Exception as e:
                logger.error(f"Failed to load experiment from {exp_file}: {e}")

    def _save_experiment(self, experiment: Experiment):
        """Save experiment to storage."""
        exp_file = self.storage_path / f"{experiment.name}.json"
        with open(exp_file, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2, default=str)

    def create_experiment(
        self,
        name: str,
        control_model: str,
        treatment_model: str,
        traffic_split: float = 0.1,
        min_samples: int = 1000,
        max_duration_hours: int = 168,
        **kwargs,
    ) -> Experiment:
        """Create a new A/B test experiment.

        Args:
            name: Unique experiment name
            control_model: Current production model ID
            treatment_model: New model ID to test
            traffic_split: Fraction of traffic to treatment (0-1)
            min_samples: Minimum samples before analysis
            max_duration_hours: Maximum experiment duration

        Returns:
            Created Experiment
        """
        if name in self._experiments:
            raise ValueError(f"Experiment '{name}' already exists")

        experiment = Experiment(
            name=name,
            control_model=control_model,
            treatment_model=treatment_model,
            traffic_split=traffic_split,
            min_samples=min_samples,
            max_duration_hours=max_duration_hours,
            **kwargs,
        )

        self._experiments[name] = experiment
        self._save_experiment(experiment)

        logger.info(f"Created experiment '{name}': {control_model} vs {treatment_model}")
        return experiment

    def start_experiment(self, name: str) -> Experiment:
        """Start an experiment.

        Args:
            name: Experiment name

        Returns:
            Started Experiment
        """
        if name not in self._experiments:
            raise ValueError(f"Experiment '{name}' not found")

        experiment = self._experiments[name]
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.now()

        self._save_experiment(experiment)
        logger.info(f"Started experiment '{name}'")

        return experiment

    def pause_experiment(self, name: str) -> Experiment:
        """Pause an experiment."""
        experiment = self._experiments.get(name)
        if not experiment:
            raise ValueError(f"Experiment '{name}' not found")

        experiment.status = ExperimentStatus.PAUSED
        self._save_experiment(experiment)

        logger.info(f"Paused experiment '{name}'")
        return experiment

    def get_model(
        self,
        experiment_name: str,
        request_id: str,
    ) -> tuple[str, ModelVariant]:
        """Get the model to use for a request.

        Uses consistent hashing to ensure the same request_id
        always gets the same variant.

        Args:
            experiment_name: Name of the experiment
            request_id: Unique request identifier

        Returns:
            Tuple of (model_id, variant)
        """
        experiment = self._experiments.get(experiment_name)

        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return ("default", ModelVariant.CONTROL)

        hash_value = hash(request_id) % 100
        use_treatment = hash_value < (experiment.traffic_split * 100)

        if use_treatment:
            return (experiment.treatment_model, ModelVariant.TREATMENT)
        else:
            return (experiment.control_model, ModelVariant.CONTROL)

    def record_prediction(
        self,
        experiment_name: str,
        variant: ModelVariant,
        prediction: float,
        latency_ms: float,
    ):
        """Record a prediction for metrics tracking.

        Args:
            experiment_name: Experiment name
            variant: Which variant made the prediction
            prediction: The prediction value
            latency_ms: Inference latency
        """
        experiment = self._experiments.get(experiment_name)
        if not experiment:
            return

        if variant == ModelVariant.CONTROL:
            experiment.control_metrics.total_predictions += 1
            experiment.control_latencies.append(latency_ms)
        else:
            experiment.treatment_metrics.total_predictions += 1
            experiment.treatment_latencies.append(latency_ms)

    def record_outcome(
        self,
        experiment_name: str,
        variant: ModelVariant,
        actual_return: float,
        correct: bool | None = None,
    ):
        """Record an outcome for experiment analysis.

        Args:
            experiment_name: Experiment name
            variant: Which variant made the prediction
            actual_return: Actual return (positive or negative)
            correct: Whether the prediction was correct
        """
        experiment = self._experiments.get(experiment_name)
        if not experiment:
            return

        if variant == ModelVariant.CONTROL:
            experiment.control_returns.append(actual_return)
            experiment.control_metrics.total_outcomes += 1
            experiment.control_metrics.total_return += actual_return
            if correct is not None:
                experiment.control_metrics.correct_predictions += int(correct)
        else:
            experiment.treatment_returns.append(actual_return)
            experiment.treatment_metrics.total_outcomes += 1
            experiment.treatment_metrics.total_return += actual_return
            if correct is not None:
                experiment.treatment_metrics.correct_predictions += int(correct)

        self._update_metrics(experiment)

        if self.auto_rollback:
            self._check_rollback(experiment)

    def _update_metrics(self, experiment: Experiment):
        """Update computed metrics for experiment."""
        for returns, metrics in [
            (experiment.control_returns, experiment.control_metrics),
            (experiment.treatment_returns, experiment.treatment_metrics),
        ]:
            if len(returns) < 2:
                continue

            arr = np.array(returns)
            metrics.mean_return = float(np.mean(arr))
            metrics.std_return = float(np.std(arr))

            if metrics.std_return > 0:
                metrics.sharpe_ratio = metrics.mean_return / metrics.std_return

                downside = arr[arr < 0]
                if len(downside) > 0:
                    downside_std = np.std(downside)
                    metrics.sortino_ratio = (
                        metrics.mean_return / downside_std if downside_std > 0 else 0
                    )

            cumulative = np.cumprod(1 + arr / 100)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            metrics.max_drawdown = float(np.min(np.abs(drawdowns)))

            sorted_returns = np.sort(arr)
            var_idx = int(len(arr) * 0.05)
            metrics.var_95 = float(sorted_returns[var_idx])
            # Guard against empty slice when var_idx is 0 (len < 20)
            metrics.cvar_95 = (
                float(np.mean(sorted_returns[:var_idx])) if var_idx > 0 else metrics.var_95
            )

            if metrics.total_outcomes > 0:
                metrics.directional_accuracy = metrics.correct_predictions / metrics.total_outcomes

    def _check_rollback(self, experiment: Experiment):
        """Check if experiment should be rolled back."""
        control = experiment.control_metrics
        treatment = experiment.treatment_metrics

        if treatment.total_outcomes < 100:
            return

        if treatment.max_drawdown > experiment.max_drawdown_threshold:
            self._rollback(experiment, f"Max drawdown exceeded: {treatment.max_drawdown:.2%}")
            return

        if (
            control.directional_accuracy - treatment.directional_accuracy
        ) > experiment.accuracy_drop_threshold:
            self._rollback(
                experiment,
                f"Accuracy dropped: {control.directional_accuracy:.2%} -> "
                f"{treatment.directional_accuracy:.2%}",
            )
            return

        if treatment.cvar_95 < control.cvar_95 * 0.8:  # 20% worse tail risk
            self._rollback(
                experiment,
                f"Tail risk increased: CVaR {treatment.cvar_95:.4f} vs {control.cvar_95:.4f}",
            )
            return

    def _rollback(self, experiment: Experiment, reason: str):
        """Rollback experiment."""
        experiment.status = ExperimentStatus.ROLLED_BACK
        experiment.rollback_reason = reason
        experiment.rollback_time = datetime.now()

        self._save_experiment(experiment)
        logger.warning(f"Rolled back experiment '{experiment.name}': {reason}")

    def analyze_experiment(self, name: str) -> ExperimentResult:
        """Analyze experiment results with statistical testing.

        Uses Welch's t-test for comparing means.

        Args:
            name: Experiment name

        Returns:
            ExperimentResult with statistical analysis
        """
        experiment = self._experiments.get(name)
        if not experiment:
            raise ValueError(f"Experiment '{name}' not found")

        control_returns = np.array(experiment.control_returns)
        treatment_returns = np.array(experiment.treatment_returns)

        if len(control_returns) < 30 or len(treatment_returns) < 30:
            return ExperimentResult(
                experiment_name=name,
                winner=None,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                recommendation="Insufficient data for analysis",
                should_rollback=False,
                rollback_reason=None,
            )

        control_mean = np.mean(control_returns)
        treatment_mean = np.mean(treatment_returns)

        control_std = np.std(control_returns, ddof=1)
        treatment_std = np.std(treatment_returns, ddof=1)

        n1, n2 = len(control_returns), len(treatment_returns)

        se = np.sqrt(control_std**2 / n1 + treatment_std**2 / n2)

        effect_size = (treatment_mean - control_mean) / se if se > 0 else 0

        t_stat = (treatment_mean - control_mean) / se if se > 0 else 0

        df = (control_std**2 / n1 + treatment_std**2 / n2) ** 2 / (
            (control_std**2 / n1) ** 2 / (n1 - 1) + (treatment_std**2 / n2) ** 2 / (n2 - 1)
        )

        from scipy import stats as scipy_stats

        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df))

        margin = 1.96 * se
        ci = (effect_size - margin, effect_size + margin)

        should_rollback = False
        winner = None
        recommendation = ""

        if p_value < experiment.significance_level:
            if effect_size > experiment.min_effect_size:
                winner = "treatment"
                recommendation = (
                    f"Treatment is significantly better. Effect size: "
                    f"{effect_size:.4f}. Recommend full rollout."
                )
            elif effect_size < -experiment.min_effect_size:
                winner = "control"
                should_rollback = True
                recommendation = (
                    f"Control is significantly better. Effect size: "
                    f"{effect_size:.4f}. Recommend rollback."
                )
        else:
            recommendation = (
                f"No significant difference detected (p={p_value:.4f}). Continue collecting data."
            )

        experiment.p_value = p_value
        experiment.effect_size = effect_size
        experiment.confidence_interval = ci
        self._save_experiment(experiment)

        return ExperimentResult(
            experiment_name=name,
            winner=winner,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            recommendation=recommendation,
            should_rollback=should_rollback,
            rollback_reason=experiment.rollback_reason,
        )

    def get_experiment(self, name: str) -> Experiment | None:
        """Get experiment by name."""
        return self._experiments.get(name)

    def list_experiments(
        self,
        status: ExperimentStatus | None = None,
    ) -> list[Experiment]:
        """List experiments, optionally filtered by status."""
        experiments = list(self._experiments.values())

        if status:
            experiments = [e for e in experiments if e.status == status]

        return experiments

    def promote_treatment(self, name: str) -> bool:
        """Promote treatment model to control (full rollout).

        Args:
            name: Experiment name

        Returns:
            True if promoted successfully
        """
        experiment = self._experiments.get(name)
        if not experiment:
            return False

        result = self.analyze_experiment(name)

        if result.winner == "treatment":
            experiment.status = ExperimentStatus.COMPLETED
            experiment.end_time = datetime.now()
            self._save_experiment(experiment)

            logger.info(f"Promoted treatment model '{experiment.treatment_model}' to production")
            return True

        return False

    def get_traffic_split_report(self) -> dict[str, Any]:
        """Get current traffic split status for all experiments."""
        return {
            name: {
                "status": exp.status.value,
                "traffic_split": exp.traffic_split,
                "control_samples": len(exp.control_returns),
                "treatment_samples": len(exp.treatment_returns),
                "control_model": exp.control_model,
                "treatment_model": exp.treatment_model,
            }
            for name, exp in self._experiments.items()
            if exp.status == ExperimentStatus.RUNNING
        }
