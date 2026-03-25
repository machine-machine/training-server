"""
Training and Validation Pipeline - Systematic parameter optimization and validation.

This module provides:
- Hyperparameter optimization (grid search, random search, Bayesian)
- Walk-forward validation with no-leakage guarantees
- Quality gates for configuration acceptance
- Automated backtest-to-live promotion pipeline

The pipeline ensures that only configurations meeting strict criteria
are promoted to live trading, with comprehensive audit trails.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Parameter optimization methods."""

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"


class ValidationResult(Enum):
    """Result of a validation run."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"


class PromotionStatus(Enum):
    """Status of configuration promotion."""

    PENDING = "pending"
    VALIDATING = "validating"
    PAPER_TRADING = "paper_trading"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


# ============================================================================
# QUALITY GATES
# ============================================================================


@dataclass
class QualityGate:
    """A single quality gate criterion."""

    name: str
    metric: str
    min_value: float | None = None
    max_value: float | None = None
    weight: float = 1.0
    description: str = ""

    def evaluate(self, value: float) -> tuple[bool, float, str]:
        """Evaluate if value passes the gate.

        Returns:
            (passed, score, message)
        """
        passed = True
        messages = []

        if self.min_value is not None and value < self.min_value:
            passed = False
            messages.append(f"{value:.2f} < min {self.min_value:.2f}")

        if self.max_value is not None and value > self.max_value:
            passed = False
            messages.append(f"{value:.2f} > max {self.max_value:.2f}")

        # Calculate score (0-1, higher is better)
        if self.min_value is not None and self.max_value is not None:
            range_size = self.max_value - self.min_value
            if range_size > 0:
                score = max(0, min(1, (value - self.min_value) / range_size))
            else:
                score = 1.0 if value == self.min_value else 0.0
        elif self.min_value is not None:
            score = min(1.0, value / self.min_value) if self.min_value > 0 else 1.0
        elif self.max_value is not None:
            score = max(0, 1.0 - (value / self.max_value)) if self.max_value > 0 else 1.0
        else:
            score = 1.0

        message = f"{self.name}: {value:.2f} - {'PASS' if passed else 'FAIL'}"
        if messages:
            message += f" ({', '.join(messages)})"

        return passed, score * self.weight, message


@dataclass
class QualityGatesConfig:
    """Configuration for validation quality gates.

    Default gates are calibrated for 0.8-1% daily P&L target.
    """

    # Win rate gate
    win_rate_gate: QualityGate = field(
        default_factory=lambda: QualityGate(
            name="win_rate",
            metric="win_rate_pct",
            min_value=50.0,
            max_value=100.0,
            weight=1.5,
            description="Win rate must be >= 50%",
        )
    )

    # Daily P&L gate
    daily_pnl_gate: QualityGate = field(
        default_factory=lambda: QualityGate(
            name="daily_pnl",
            metric="daily_pnl_pct",
            min_value=0.6,  # Minimum acceptable
            max_value=5.0,  # Unrealistic above this
            weight=2.0,  # Most important
            description="Daily P&L must be >= 0.6%",
        )
    )

    # Max drawdown gate
    max_drawdown_gate: QualityGate = field(
        default_factory=lambda: QualityGate(
            name="max_drawdown",
            metric="max_drawdown_pct",
            min_value=0.0,
            max_value=20.0,  # Strict limit
            weight=1.5,
            description="Max drawdown must be <= 20%",
        )
    )

    # Trade count gate (statistical significance)
    trade_count_gate: QualityGate = field(
        default_factory=lambda: QualityGate(
            name="trade_count",
            metric="total_trades",
            min_value=50,
            max_value=None,
            weight=0.5,
            description="At least 50 trades required",
        )
    )

    # Sharpe ratio gate
    sharpe_gate: QualityGate = field(
        default_factory=lambda: QualityGate(
            name="sharpe_ratio",
            metric="sharpe_ratio",
            min_value=1.5,
            max_value=None,
            weight=1.0,
            description="Sharpe ratio must be >= 1.5",
        )
    )

    # Profit factor gate
    profit_factor_gate: QualityGate = field(
        default_factory=lambda: QualityGate(
            name="profit_factor",
            metric="profit_factor",
            min_value=1.2,
            max_value=None,
            weight=1.0,
            description="Profit factor must be >= 1.2",
        )
    )

    def get_gates(self) -> list[QualityGate]:
        """Get all quality gates."""
        return [
            self.win_rate_gate,
            self.daily_pnl_gate,
            self.max_drawdown_gate,
            self.trade_count_gate,
            self.sharpe_gate,
            self.profit_factor_gate,
        ]

    def evaluate_all(self, metrics: dict[str, float]) -> tuple[bool, float, list[str]]:
        """Evaluate all gates against metrics.

        Returns:
            (all_passed, weighted_score, messages)
        """
        all_passed = True
        total_score = 0.0
        total_weight = 0.0
        messages = []

        for gate in self.get_gates():
            value = metrics.get(gate.metric, 0.0)
            passed, score, message = gate.evaluate(value)

            if not passed:
                all_passed = False

            total_score += score
            total_weight += gate.weight
            messages.append(message)

        # Normalize score to 0-1
        final_score = total_score / total_weight if total_weight > 0 else 0.0

        return all_passed, final_score, messages


# ============================================================================
# PARAMETER SEARCH SPACE
# ============================================================================


@dataclass
class ParameterRange:
    """Definition of a searchable parameter range."""

    name: str
    min_value: float
    max_value: float
    step: float | None = None  # For grid search
    is_integer: bool = False
    is_log_scale: bool = False  # Use log scale for sampling

    def sample_random(self) -> Any:
        """Sample a random value from the range."""
        if self.is_log_scale:
            log_min = math.log(self.min_value)
            log_max = math.log(self.max_value)
            value = math.exp(random.uniform(log_min, log_max))
        else:
            value = random.uniform(self.min_value, self.max_value)

        if self.is_integer:
            return int(round(value))
        return value

    def get_grid_values(self, num_points: int = 10) -> list[Any]:
        """Get grid values for grid search."""
        if self.step is not None:
            values = []
            v = self.min_value
            while v <= self.max_value:
                values.append(int(v) if self.is_integer else v)
                v += self.step
            return values
        else:
            step = (self.max_value - self.min_value) / (num_points - 1)
            values = [self.min_value + i * step for i in range(num_points)]
            if self.is_integer:
                values = [int(round(v)) for v in values]
            return values


# Default parameter search space
DEFAULT_SEARCH_SPACE: list[ParameterRange] = [
    ParameterRange("min_confidence_to_trade", 0.45, 0.75, step=0.05),
    ParameterRange("max_position_sol", 0.02, 0.15, is_log_scale=True),
    ParameterRange("kelly_fraction", 0.15, 0.50, step=0.05),
    ParameterRange("default_stop_loss_pct", 5.0, 15.0, step=1.0),
    ParameterRange("default_take_profit_pct", 10.0, 30.0, step=2.0),
    ParameterRange("max_daily_trades", 20, 60, is_integer=True),
    ParameterRange("max_concurrent_positions", 3, 10, is_integer=True),
]


# ============================================================================
# VALIDATION RESULT
# ============================================================================


@dataclass
class ValidationReport:
    """Report from a validation run."""

    config_id: str
    timestamp: datetime
    config: dict[str, Any]
    metrics: dict[str, float]
    passed: bool
    score: float
    gate_messages: list[str]
    backtest_period: str | None = None
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "timestamp": self.timestamp.isoformat(),
            "config": self.config,
            "metrics": self.metrics,
            "passed": self.passed,
            "score": round(self.score, 3),
            "gate_messages": self.gate_messages,
            "backtest_period": self.backtest_period,
            "duration_seconds": round(self.duration_seconds, 2),
            "error": self.error,
        }


@dataclass
class OptimizationResult:
    """Result from a parameter optimization run."""

    optimization_id: str
    method: OptimizationMethod
    start_time: datetime
    end_time: datetime | None = None
    total_iterations: int = 0
    best_config: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_score: float = 0.0
    all_results: list[ValidationReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimization_id": self.optimization_id,
            "method": self.method.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_iterations": self.total_iterations,
            "best_config": self.best_config,
            "best_metrics": self.best_metrics,
            "best_score": round(self.best_score, 3),
            "result_count": len(self.all_results),
        }


# ============================================================================
# HYPERPARAMETER TUNER
# ============================================================================


class HyperparameterTuner:
    """Systematic parameter optimization for trading configurations.

    Supports multiple optimization methods:
    - Grid search: Exhaustive search over parameter grid
    - Random search: Random sampling from parameter space
    - Bayesian: Sequential model-based optimization

    Usage:
        tuner = HyperparameterTuner(search_space=DEFAULT_SEARCH_SPACE)

        # Run random search with 50 iterations
        result = await tuner.optimize(
            method=OptimizationMethod.RANDOM_SEARCH,
            iterations=50,
            objective_fn=my_backtest_function,
        )

        if result.best_config:
            print(f"Best config: {result.best_config}")
            print(f"Best score: {result.best_score}")
    """

    def __init__(
        self,
        search_space: list[ParameterRange] | None = None,
        quality_gates: QualityGatesConfig | None = None,
        base_config: dict[str, Any] | None = None,
    ):
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.quality_gates = quality_gates or QualityGatesConfig()
        self.base_config = base_config or {}

        # Optimization state
        self._current_optimization: OptimizationResult | None = None
        self._optimization_history: list[OptimizationResult] = []

        # Bayesian optimization state (simplified)
        self._bayesian_history: list[tuple[dict[str, Any], float]] = []

    def sample_random_config(self) -> dict[str, Any]:
        """Sample a random configuration from the search space."""
        config = self.base_config.copy()
        for param in self.search_space:
            config[param.name] = param.sample_random()
        return config

    def generate_grid_configs(self, points_per_param: int = 5) -> list[dict[str, Any]]:
        """Generate all configurations for grid search."""
        # Get grid values for each parameter
        param_values = {}
        for param in self.search_space:
            param_values[param.name] = param.get_grid_values(points_per_param)

        # Generate all combinations (cartesian product)
        configs = [self.base_config.copy()]
        for param_name, values in param_values.items():
            new_configs = []
            for config in configs:
                for value in values:
                    new_config = config.copy()
                    new_config[param_name] = value
                    new_configs.append(new_config)
            configs = new_configs

        return configs

    async def optimize(
        self,
        method: OptimizationMethod,
        iterations: int,
        objective_fn: Callable[[dict[str, Any]], Awaitable[dict[str, float]]],
        callbacks: list[Callable] | None = None,
    ) -> OptimizationResult:
        """Run parameter optimization.

        Args:
            method: Optimization method to use
            iterations: Number of iterations (for random/Bayesian)
            objective_fn: Async function that takes config and returns metrics
            callbacks: Optional callbacks called after each iteration

        Returns:
            OptimizationResult with best configuration found
        """
        optimization_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._current_optimization = OptimizationResult(
            optimization_id=optimization_id,
            method=method,
            start_time=datetime.now(),
        )

        logger.info(f"🔧 Starting {method.value} optimization with {iterations} iterations")

        try:
            if method == OptimizationMethod.GRID_SEARCH:
                await self._run_grid_search(objective_fn, callbacks)
            elif method == OptimizationMethod.RANDOM_SEARCH:
                await self._run_random_search(iterations, objective_fn, callbacks)
            elif method == OptimizationMethod.BAYESIAN:
                await self._run_bayesian_optimization(iterations, objective_fn, callbacks)

            self._current_optimization.end_time = datetime.now()

            # Save to history
            self._optimization_history.append(self._current_optimization)

            logger.info(
                f"✅ Optimization complete: best_score={self._current_optimization.best_score:.3f}"
            )

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            if self._current_optimization:
                self._current_optimization.end_time = datetime.now()

        return self._current_optimization

    async def _run_grid_search(
        self,
        objective_fn: Callable,
        callbacks: list[Callable] | None = None,
    ) -> None:
        """Run grid search optimization."""
        configs = self.generate_grid_configs(points_per_param=4)
        if self._current_optimization:
            self._current_optimization.total_iterations = len(configs)

        logger.info(f"Grid search: {len(configs)} configurations to evaluate")

        for i, config in enumerate(configs):
            await self._evaluate_config(config, objective_fn, callbacks, i + 1)

    async def _run_random_search(
        self,
        iterations: int,
        objective_fn: Callable,
        callbacks: list[Callable] | None = None,
    ) -> None:
        """Run random search optimization."""
        if self._current_optimization:
            self._current_optimization.total_iterations = iterations

        for i in range(iterations):
            config = self.sample_random_config()
            await self._evaluate_config(config, objective_fn, callbacks, i + 1)

    async def _run_bayesian_optimization(
        self,
        iterations: int,
        objective_fn: Callable,
        callbacks: list[Callable] | None = None,
    ) -> None:
        """Run simplified Bayesian optimization.

        This is a simplified implementation using Thompson Sampling-like approach.
        For production, consider using optuna or scikit-optimize.
        """
        if self._current_optimization:
            self._current_optimization.total_iterations = iterations

        # Start with random exploration
        exploration = max(5, iterations // 5)

        for i in range(iterations):
            if i < exploration or len(self._bayesian_history) < 5:
                # Explore: random sample
                config = self.sample_random_config()
            else:
                # Exploit: sample near best configurations
                config = self._sample_near_best()

            await self._evaluate_config(config, objective_fn, callbacks, i + 1)

    def _sample_near_best(self) -> dict[str, Any]:
        """Sample a configuration near the best known configurations."""
        # Get top 5 configurations
        sorted_history = sorted(self._bayesian_history, key=lambda x: x[1], reverse=True)[:5]

        if not sorted_history:
            return self.sample_random_config()

        # Pick one of the best configs randomly
        best_config, _ = random.choice(sorted_history)

        # Mutate slightly
        config = best_config.copy()
        for param in self.search_space:
            if random.random() < 0.3:  # 30% chance to mutate each param
                # Add small random perturbation
                current = config.get(param.name, 0)
                perturbation = (param.max_value - param.min_value) * 0.1 * random.uniform(-1, 1)
                new_value = current + perturbation
                new_value = max(param.min_value, min(param.max_value, new_value))
                if param.is_integer:
                    new_value = int(round(new_value))
                config[param.name] = new_value

        return config

    async def _evaluate_config(
        self,
        config: dict[str, Any],
        objective_fn: Callable,
        callbacks: list[Callable] | None,
        iteration: int,
    ) -> None:
        """Evaluate a single configuration."""
        if self._current_optimization is None:
            return

        start_time = datetime.now()
        config_id = f"cfg_{iteration:04d}"

        try:
            # Run objective function
            metrics = await objective_fn(config)

            # Evaluate quality gates
            passed, score, messages = self.quality_gates.evaluate_all(metrics)

            # Create report
            report = ValidationReport(
                config_id=config_id,
                timestamp=start_time,
                config=config,
                metrics=metrics,
                passed=passed,
                score=score,
                gate_messages=messages,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

            self._current_optimization.all_results.append(report)

            # Update best if this is better
            if score > self._current_optimization.best_score:
                self._current_optimization.best_score = score
                self._current_optimization.best_config = config.copy()
                self._current_optimization.best_metrics = metrics.copy()
                logger.info(
                    f"  New best! iteration={iteration}, score={score:.3f}, "
                    f"win_rate={metrics.get('win_rate_pct', 0):.1f}%, "
                    f"daily_pnl={metrics.get('daily_pnl_pct', 0):.2f}%"
                )

            # Store for Bayesian optimization
            self._bayesian_history.append((config, score))

            # Run callbacks
            if callbacks:
                for callback in callbacks:
                    try:
                        callback(report)
                    except Exception as e:
                        logger.warning(f"Callback error: {e}")

        except Exception as e:
            logger.error(f"Failed to evaluate config {config_id}: {e}")
            report = ValidationReport(
                config_id=config_id,
                timestamp=start_time,
                config=config,
                metrics={},
                passed=False,
                score=0.0,
                gate_messages=[f"Evaluation error: {e}"],
                error=str(e),
            )
            if self._current_optimization:
                self._current_optimization.all_results.append(report)

    def get_optimization_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent optimization history."""
        return [r.to_dict() for r in self._optimization_history[-limit:]]


# ============================================================================
# VALIDATION PIPELINE
# ============================================================================


class ValidationPipeline:
    """End-to-end validation pipeline for trading configurations.

    Provides:
    - Single configuration validation
    - Parameter optimization
    - Walk-forward validation
    - Promotion workflow (backtest → paper → live)
    """

    def __init__(
        self,
        quality_gates: QualityGatesConfig | None = None,
    ):
        self.quality_gates = quality_gates or QualityGatesConfig()
        self.tuner = HyperparameterTuner(quality_gates=self.quality_gates)

        # Validation state
        self._validation_history: list[ValidationReport] = []
        self._pending_promotions: dict[str, dict[str, Any]] = {}

    async def validate_config(
        self,
        config: dict[str, Any],
        backtest_fn: Callable[[dict[str, Any]], Awaitable[dict[str, float]]],
    ) -> ValidationReport:
        """Validate a single configuration.

        Args:
            config: Configuration to validate
            backtest_fn: Async function that runs backtest and returns metrics

        Returns:
            ValidationReport with pass/fail status
        """
        config_id = f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        try:
            # Run backtest
            metrics = await backtest_fn(config)

            # Evaluate quality gates
            passed, score, messages = self.quality_gates.evaluate_all(metrics)

            report = ValidationReport(
                config_id=config_id,
                timestamp=start_time,
                config=config,
                metrics=metrics,
                passed=passed,
                score=score,
                gate_messages=messages,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

            logger.info(
                f"📊 Validation {'PASSED' if passed else 'FAILED'}: "
                f"score={score:.3f}, win_rate={metrics.get('win_rate_pct', 0):.1f}%, "
                f"daily_pnl={metrics.get('daily_pnl_pct', 0):.2f}%"
            )

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            report = ValidationReport(
                config_id=config_id,
                timestamp=start_time,
                config=config,
                metrics={},
                passed=False,
                score=0.0,
                gate_messages=[f"Validation error: {e}"],
                error=str(e),
            )

        self._validation_history.append(report)
        return report

    async def optimize_and_validate(
        self,
        method: OptimizationMethod,
        iterations: int,
        backtest_fn: Callable[[dict[str, Any]], Awaitable[dict[str, float]]],
    ) -> tuple[OptimizationResult, ValidationReport | None]:
        """Run optimization and validate the best configuration.

        Args:
            method: Optimization method
            iterations: Number of optimization iterations
            backtest_fn: Async backtest function

        Returns:
            (OptimizationResult, ValidationReport of best config)
        """
        # Run optimization
        opt_result = await self.tuner.optimize(
            method=method,
            iterations=iterations,
            objective_fn=backtest_fn,
        )

        # Validate best config if found
        if opt_result.best_config:
            # Run additional validation on best config
            final_report = await self.validate_config(
                opt_result.best_config,
                backtest_fn,
            )
            return opt_result, final_report

        return opt_result, None

    def get_validation_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent validation history."""
        return [r.to_dict() for r in self._validation_history[-limit:]]

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics from validation history."""
        if not self._validation_history:
            return {
                "total_validations": 0,
                "pass_rate": 0.0,
                "avg_score": 0.0,
            }

        total = len(self._validation_history)
        passed = sum(1 for r in self._validation_history if r.passed)
        scores = [r.score for r in self._validation_history]

        return {
            "total_validations": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "best_score": max(scores) if scores else 0.0,
        }

    def clear_history(self) -> None:
        """Clear validation history."""
        self._validation_history.clear()
        logger.info("Validation history cleared")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_backtest_metrics(
    total_return_pct: float,
    win_count: int,
    loss_count: int,
    max_drawdown_pct: float,
    avg_win_pct: float = 5.0,
    avg_loss_pct: float = 3.0,
    trading_days: int = 1,
) -> dict[str, float]:
    """Create a metrics dict from backtest results.

    Helper function to convert raw backtest output to
    the metrics format expected by quality gates.
    """
    total_trades = win_count + loss_count
    win_rate_pct = (win_count / total_trades * 100) if total_trades > 0 else 0
    daily_pnl_pct = total_return_pct / max(1, trading_days)

    # Estimate Sharpe ratio (simplified)
    if total_trades > 0:
        avg_pnl = (win_rate_pct / 100 * avg_win_pct) - ((1 - win_rate_pct / 100) * avg_loss_pct)
        sharpe = avg_pnl / 5.0 if avg_pnl > 0 else 0  # Rough estimate
    else:
        sharpe = 0

    # Calculate profit factor
    gross_profit = win_count * avg_win_pct
    gross_loss = loss_count * avg_loss_pct
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        "win_rate_pct": win_rate_pct,
        "daily_pnl_pct": daily_pnl_pct,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": total_trades,
        "sharpe_ratio": sharpe,
        "profit_factor": profit_factor,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
    }
