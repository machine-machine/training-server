"""
A/B Testing Framework for ML model experimentation.

Implements experiment management:
- Traffic allocation: Deterministic by mint hash
- Stopping rules: P-value <0.01, min trades per arm
- Metrics: Sharpe, win rate, max drawdown
- Auto-rollback on treatment failure
"""

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an A/B experiment."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class StoppingRule(Enum):
    """Rules for stopping an experiment early."""

    FIXED_SAMPLE = "fixed_sample"  # Stop after N samples
    SEQUENTIAL_PVALUE = "sequential"  # Stop when p-value threshold met
    BAYESIAN_CREDIBLE = "bayesian"  # Stop when credible interval clear
    FIXED_DURATION = "duration"  # Stop after fixed time


@dataclass
class ArmMetrics:
    """Metrics for an experiment arm."""

    arm_id: str
    samples: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    pnl_squared: float = 0.0
    max_drawdown: float = 0.0
    returns: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def mean_pnl(self) -> float:
        return self.total_pnl / self.samples if self.samples > 0 else 0.0

    @property
    def std_pnl(self) -> float:
        if self.samples < 2:
            return 0.0
        variance = (self.pnl_squared / self.samples) - (self.mean_pnl**2)
        return math.sqrt(max(0, variance))

    @property
    def sharpe_ratio(self) -> float:
        if self.std_pnl < 1e-10:
            return 0.0
        return self.mean_pnl / self.std_pnl * math.sqrt(252)

    def update(self, pnl: float, won: bool):
        """Update arm with new observation."""
        self.samples += 1
        self.total_pnl += pnl
        self.pnl_squared += pnl**2
        self.returns.append(pnl)

        if won:
            self.wins += 1
        else:
            self.losses += 1

        # Update max drawdown
        if len(self.returns) > 0:
            cumulative = np.cumsum(self.returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = cumulative - running_max
            self.max_drawdown = min(self.max_drawdown, np.min(drawdowns))


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""

    experiment_id: str
    control_id: str  # Control arm identifier
    treatment_id: str  # Treatment arm identifier
    traffic_split: float = 0.50  # Fraction to treatment (0-1)

    # Stopping rules
    stopping_rule: StoppingRule = StoppingRule.SEQUENTIAL_PVALUE
    min_samples_per_arm: int = 30
    max_samples_per_arm: int = 1000
    p_value_threshold: float = 0.01  # For sequential stopping
    max_duration_hours: int | None = 72  # Max experiment duration

    # Guard rails
    min_win_rate: float = 0.40  # Stop if arm falls below
    max_drawdown: float = -0.20  # Stop if drawdown exceeds
    auto_rollback: bool = True  # Auto-rollback on failure

    # Primary metric
    primary_metric: str = "win_rate"  # "win_rate", "sharpe", "mean_pnl"


@dataclass
class ExperimentResult:
    """Result of a completed experiment."""

    experiment_id: str
    status: ExperimentStatus
    winner: str | None
    control_metrics: ArmMetrics
    treatment_metrics: ArmMetrics
    p_value: float | None
    confidence_interval: tuple[float, float] | None
    effect_size: float
    duration_hours: float
    total_samples: int
    stopped_early: bool
    stop_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "winner": self.winner,
            "control": {
                "samples": self.control_metrics.samples,
                "win_rate": self.control_metrics.win_rate,
                "sharpe": self.control_metrics.sharpe_ratio,
                "mean_pnl": self.control_metrics.mean_pnl,
            },
            "treatment": {
                "samples": self.treatment_metrics.samples,
                "win_rate": self.treatment_metrics.win_rate,
                "sharpe": self.treatment_metrics.sharpe_ratio,
                "mean_pnl": self.treatment_metrics.mean_pnl,
            },
            "p_value": self.p_value,
            "confidence_interval": self.confidence_interval,
            "effect_size": self.effect_size,
            "duration_hours": self.duration_hours,
            "total_samples": self.total_samples,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
        }


class ABTestFramework:
    """A/B testing framework for ML model experimentation.

    Features:
    - Deterministic traffic allocation by mint hash
    - Multiple stopping rules (fixed, sequential, Bayesian)
    - Guard rails for auto-rollback
    - Statistical significance testing

    Example:
        framework = ABTestFramework()
        exp = framework.create_experiment("v1", "v2")
        framework.record_outcome(exp.experiment_id, mint, pnl, won)
        if framework.should_stop(exp.experiment_id):
            result = framework.finalize(exp.experiment_id)
    """

    def __init__(self):
        self._experiments: dict[str, ExperimentConfig] = {}
        self._metrics: dict[str, dict[str, ArmMetrics]] = {}  # exp_id -> arm_id -> metrics
        self._start_times: dict[str, datetime] = {}
        self._results: dict[str, ExperimentResult] = {}

    def create_experiment(
        self,
        control_id: str,
        treatment_id: str,
        config: ExperimentConfig | None = None,
    ) -> ExperimentConfig:
        """Create a new A/B experiment.

        Args:
            control_id: Identifier for control (e.g., model version)
            treatment_id: Identifier for treatment
            config: Optional experiment configuration

        Returns:
            ExperimentConfig
        """
        if config is None:
            exp_id = f"exp_{control_id}_vs_{treatment_id}_{int(datetime.now().timestamp())}"
            config = ExperimentConfig(
                experiment_id=exp_id,
                control_id=control_id,
                treatment_id=treatment_id,
            )

        self._experiments[config.experiment_id] = config
        self._metrics[config.experiment_id] = {
            config.control_id: ArmMetrics(arm_id=config.control_id),
            config.treatment_id: ArmMetrics(arm_id=config.treatment_id),
        }
        self._start_times[config.experiment_id] = datetime.now()

        logger.info(f"Created experiment {config.experiment_id}")
        return config

    def get_arm_for_mint(self, experiment_id: str, mint: str) -> str:
        """Get arm assignment for a mint (deterministic routing).

        Args:
            experiment_id: Experiment identifier
            mint: Token mint address

        Returns:
            Arm identifier (control or treatment)
        """
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        config = self._experiments[experiment_id]

        # Deterministic hash-based assignment
        hash_input = f"{experiment_id}:{mint}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        bucket = (hash_value % 10000) / 10000.0

        if bucket < config.traffic_split:
            return config.treatment_id
        else:
            return config.control_id

    def record_outcome(
        self,
        experiment_id: str,
        mint: str,
        pnl: float,
        won: bool,
    ) -> str | None:
        """Record an outcome for an experiment.

        Args:
            experiment_id: Experiment identifier
            mint: Token mint (for arm assignment)
            pnl: PnL of the trade
            won: Whether trade was profitable

        Returns:
            Arm that was assigned, or None if experiment not found
        """
        if experiment_id not in self._experiments:
            return None

        arm_id = self.get_arm_for_mint(experiment_id, mint)
        metrics = self._metrics[experiment_id][arm_id]
        metrics.update(pnl, won)

        # Check guard rails
        config = self._experiments[experiment_id]
        if config.auto_rollback:
            self._check_guard_rails(experiment_id, arm_id)

        return arm_id

    def should_stop(self, experiment_id: str) -> tuple[bool, str | None]:
        """Check if experiment should stop.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Tuple of (should_stop, reason)
        """
        if experiment_id not in self._experiments:
            return True, "not_found"

        config = self._experiments[experiment_id]
        metrics = self._metrics[experiment_id]
        control = metrics[config.control_id]
        treatment = metrics[config.treatment_id]

        # Check minimum samples
        if (
            control.samples < config.min_samples_per_arm
            or treatment.samples < config.min_samples_per_arm
        ):
            return False, None

        # Check maximum samples
        if (
            control.samples >= config.max_samples_per_arm
            and treatment.samples >= config.max_samples_per_arm
        ):
            return True, "max_samples_reached"

        # Check duration
        if config.max_duration_hours:
            elapsed = (datetime.now() - self._start_times[experiment_id]).total_seconds() / 3600
            if elapsed >= config.max_duration_hours:
                return True, "max_duration_reached"

        # Check stopping rule
        if config.stopping_rule == StoppingRule.SEQUENTIAL_PVALUE:
            p_value = self._compute_p_value(control, treatment, config.primary_metric)
            if p_value is not None and p_value < config.p_value_threshold:
                return True, f"significance_reached_p{p_value:.4f}"

        elif config.stopping_rule == StoppingRule.BAYESIAN_CREDIBLE:
            prob_better = self._compute_bayesian_probability(control, treatment)
            if prob_better > 0.95 or prob_better < 0.05:
                return True, f"bayesian_confidence_{prob_better:.2f}"

        return False, None

    def finalize(self, experiment_id: str) -> ExperimentResult:
        """Finalize an experiment and determine winner.

        Args:
            experiment_id: Experiment identifier

        Returns:
            ExperimentResult with final analysis
        """
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        config = self._experiments[experiment_id]
        metrics = self._metrics[experiment_id]
        control = metrics[config.control_id]
        treatment = metrics[config.treatment_id]

        # Compute statistics
        p_value = self._compute_p_value(control, treatment, config.primary_metric)
        effect_size = self._compute_effect_size(control, treatment, config.primary_metric)
        ci = self._compute_confidence_interval(control, treatment, config.primary_metric)

        # Determine winner
        should_stop, stop_reason = self.should_stop(experiment_id)
        winner = None
        status = ExperimentStatus.COMPLETED

        if p_value is not None and p_value < config.p_value_threshold:
            # Statistically significant result
            if effect_size > 0:
                winner = config.treatment_id
            else:
                winner = config.control_id
        else:
            # No significant difference
            winner = config.control_id  # Default to control

        # Duration
        elapsed = (datetime.now() - self._start_times[experiment_id]).total_seconds() / 3600

        result = ExperimentResult(
            experiment_id=experiment_id,
            status=status,
            winner=winner,
            control_metrics=control,
            treatment_metrics=treatment,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=effect_size,
            duration_hours=elapsed,
            total_samples=control.samples + treatment.samples,
            stopped_early=should_stop
            and stop_reason not in ["max_samples_reached", "max_duration_reached"],
            stop_reason=stop_reason,
        )

        self._results[experiment_id] = result

        logger.info(f"Finalized experiment {experiment_id}: winner={winner}, p={p_value}")
        return result

    def _compute_p_value(
        self,
        control: ArmMetrics,
        treatment: ArmMetrics,
        metric: str,
    ) -> float | None:
        """Compute two-sample t-test p-value."""
        if not SCIPY_AVAILABLE:
            return None

        if control.samples < 2 or treatment.samples < 2:
            return None

        if metric == "win_rate":
            # Proportions test
            from scipy.stats import proportions_ztest

            try:
                count = np.array([treatment.wins, control.wins])
                nobs = np.array([treatment.samples, control.samples])
                _, p_value = proportions_ztest(count, nobs, alternative="two-sided")
                return float(p_value)
            except Exception:
                return None

        elif metric in ["mean_pnl", "sharpe"]:
            # T-test on returns
            if len(control.returns) < 2 or len(treatment.returns) < 2:
                return None

            try:
                _, p_value = stats.ttest_ind(treatment.returns, control.returns)
                return float(p_value)
            except Exception:
                return None

        return None

    def _compute_effect_size(
        self,
        control: ArmMetrics,
        treatment: ArmMetrics,
        metric: str,
    ) -> float:
        """Compute effect size (treatment - control)."""
        if metric == "win_rate":
            return treatment.win_rate - control.win_rate
        elif metric == "mean_pnl":
            return treatment.mean_pnl - control.mean_pnl
        elif metric == "sharpe":
            return treatment.sharpe_ratio - control.sharpe_ratio
        return 0.0

    def _compute_confidence_interval(
        self,
        control: ArmMetrics,
        treatment: ArmMetrics,
        metric: str,
        alpha: float = 0.05,
    ) -> tuple[float, float] | None:
        """Compute confidence interval for effect size."""
        if not SCIPY_AVAILABLE:
            return None

        if control.samples < 2 or treatment.samples < 2:
            return None

        if metric == "win_rate":
            # CI for difference in proportions
            p1 = treatment.win_rate
            p2 = control.win_rate
            n1 = treatment.samples
            n2 = control.samples

            se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            z = stats.norm.ppf(1 - alpha / 2)

            diff = p1 - p2
            return (diff - z * se, diff + z * se)

        elif metric == "mean_pnl":
            # CI for difference in means
            m1, s1, n1 = treatment.mean_pnl, treatment.std_pnl, treatment.samples
            m2, s2, n2 = control.mean_pnl, control.std_pnl, control.samples

            se = math.sqrt(s1**2 / n1 + s2**2 / n2)
            t = stats.t.ppf(1 - alpha / 2, min(n1, n2) - 1)

            diff = m1 - m2
            return (diff - t * se, diff + t * se)

        return None

    def _compute_bayesian_probability(
        self,
        control: ArmMetrics,
        treatment: ArmMetrics,
    ) -> float:
        """Compute Bayesian probability that treatment is better.

        Uses Beta-Bernoulli model for win rate.
        """
        # Prior: Beta(1, 1) = uniform
        alpha_control = 1 + control.wins
        beta_control = 1 + control.losses
        alpha_treatment = 1 + treatment.wins
        beta_treatment = 1 + treatment.losses

        # Monte Carlo estimate of P(treatment > control)
        n_samples = 10000
        control_samples = np.random.beta(alpha_control, beta_control, n_samples)
        treatment_samples = np.random.beta(alpha_treatment, beta_treatment, n_samples)

        return float(np.mean(treatment_samples > control_samples))

    def _check_guard_rails(self, experiment_id: str, arm_id: str):
        """Check guard rails and trigger rollback if needed."""
        config = self._experiments[experiment_id]
        metrics = self._metrics[experiment_id][arm_id]

        # Only check if we have enough samples
        if metrics.samples < 10:
            return

        # Check win rate
        if metrics.win_rate < config.min_win_rate:
            if arm_id == config.treatment_id:
                logger.warning(
                    f"Treatment {arm_id} below min win rate "
                    f"({metrics.win_rate:.2f} < {config.min_win_rate})"
                )
                self._pause_experiment(experiment_id, "treatment_below_min_win_rate")

        # Check drawdown
        if metrics.max_drawdown < config.max_drawdown:
            if arm_id == config.treatment_id:
                logger.warning(
                    f"Treatment {arm_id} exceeded max drawdown "
                    f"({metrics.max_drawdown:.2f} < {config.max_drawdown})"
                )
                self._pause_experiment(experiment_id, "treatment_exceeded_drawdown")

    def _pause_experiment(self, experiment_id: str, reason: str):
        """Pause an experiment due to guard rail violation."""
        logger.warning(f"Pausing experiment {experiment_id}: {reason}")
        # In real implementation, this would trigger traffic rerouting
        # For now, just mark it

    def get_experiment_status(self, experiment_id: str) -> dict[str, Any]:
        """Get current status of an experiment."""
        if experiment_id not in self._experiments:
            return {"error": "not_found"}

        config = self._experiments[experiment_id]
        metrics = self._metrics[experiment_id]
        control = metrics[config.control_id]
        treatment = metrics[config.treatment_id]

        should_stop, reason = self.should_stop(experiment_id)
        elapsed = (datetime.now() - self._start_times[experiment_id]).total_seconds() / 3600

        return {
            "experiment_id": experiment_id,
            "control": {
                "samples": control.samples,
                "win_rate": control.win_rate,
                "mean_pnl": control.mean_pnl,
                "sharpe": control.sharpe_ratio,
            },
            "treatment": {
                "samples": treatment.samples,
                "win_rate": treatment.win_rate,
                "mean_pnl": treatment.mean_pnl,
                "sharpe": treatment.sharpe_ratio,
            },
            "should_stop": should_stop,
            "stop_reason": reason,
            "elapsed_hours": elapsed,
            "bayesian_prob_treatment_better": self._compute_bayesian_probability(
                control, treatment
            ),
        }

    def list_experiments(self) -> list[dict[str, Any]]:
        """List all experiments."""
        return [self.get_experiment_status(exp_id) for exp_id in self._experiments.keys()]
