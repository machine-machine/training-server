"""
Ensemble Parameter Strategies - Combine multiple optimization methods.

Implements ensemble approaches for more robust parameter selection:
- Voting: Multiple optimizers vote on parameter values
- Stacking: Use meta-learner to combine optimizer outputs
- Bagging: Bootstrap aggregating of optimization results
- Weighted Average: Weight optimizers by historical performance

Usage:
    from coldpath.backtest.ensemble_optimizer import EnsembleOptimizer

    ensemble = EnsembleOptimizer(
        backtest_runner=my_backtest_fn,
        methods=["genetic", "pareto", "optuna"],
    )

    result = await ensemble.optimize(
        param_bounds={"stop_loss_pct": (3, 15), ...},
    )

    # Get ensemble-optimized parameters
    print(result.ensemble_params)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EnsembleMethod(Enum):
    """Ensemble combination method."""

    VOTING = "voting"  # Majority/plurality voting
    WEIGHTED_AVERAGE = "weighted"  # Weighted average of params
    MEDIAN = "median"  # Median of params
    BEST_PERFORMER = "best"  # Use best individual result
    STACKING = "stacking"  # Meta-learner combination
    BAGGING = "bagging"  # Bootstrap aggregating


class OptimizationMethod(Enum):
    """Available optimization methods."""

    OPTUNA = "optuna"  # Bayesian optimization
    GENETIC = "genetic"  # Genetic algorithm
    PARETO = "pareto"  # Multi-objective NSGA-II
    RANDOM = "random"  # Random search
    GRID = "grid"  # Grid search


@dataclass
class OptimizerResult:
    """Result from a single optimizer."""

    method: OptimizationMethod
    params: dict[str, float]
    metrics: dict[str, float]
    score: float
    time_seconds: float
    iterations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "params": self.params,
            "metrics": self.metrics,
            "score": self.score,
            "time_seconds": self.time_seconds,
            "iterations": self.iterations,
        }


@dataclass
class EnsembleResult:
    """Result from ensemble optimization."""

    # Ensemble output
    ensemble_params: dict[str, float]
    ensemble_metrics: dict[str, float]
    ensemble_score: float

    # Individual results
    individual_results: list[OptimizerResult]

    # Combination info
    method: EnsembleMethod
    weights: dict[str, float]  # Method -> weight

    # Performance
    best_individual: OptimizationMethod
    improvement_over_best: float  # % improvement over best individual

    # Timing
    total_time_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "ensemble_params": self.ensemble_params,
            "ensemble_score": self.ensemble_score,
            "method": self.method.value,
            "weights": self.weights,
            "best_individual": self.best_individual.value,
            "improvement_over_best": self.improvement_over_best,
            "individual_results": [r.to_dict() for r in self.individual_results],
            "total_time_seconds": self.total_time_seconds,
        }


@dataclass
class EnsembleConfig:
    """Configuration for ensemble optimizer."""

    # Methods to use
    methods: list[OptimizationMethod] = field(
        default_factory=lambda: [
            OptimizationMethod.OPTUNA,
            OptimizationMethod.GENETIC,
        ]
    )

    # Combination method
    ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE

    # Method weights (if using weighted average)
    method_weights: dict[str, float] = field(
        default_factory=lambda: {
            "optuna": 0.4,
            "genetic": 0.3,
            "pareto": 0.2,
            "random": 0.1,
        }
    )

    # Individual optimizer configs
    optuna_iterations: int = 50
    genetic_generations: int = 30
    genetic_population: int = 30
    pareto_generations: int = 50
    pareto_population: int = 50
    random_samples: int = 100

    # Parallelization
    run_parallel: bool = True

    # Historical performance tracking
    track_performance: bool = True
    performance_history_file: str | None = None


class EnsembleOptimizer:
    """Ensemble optimizer combining multiple optimization methods.

    Uses multiple optimization approaches and combines their results
    for more robust parameter selection that's less dependent on
    any single method's biases.

    Example:
        ensemble = EnsembleOptimizer(
            backtest_runner=my_backtest,
            config=EnsembleConfig(
                methods=[OptimizationMethod.OPTUNA, OptimizationMethod.GENETIC],
                ensemble_method=EnsembleMethod.WEIGHTED_AVERAGE,
            ),
        )

        result = await ensemble.optimize(
            param_bounds={
                "stop_loss_pct": (3.0, 15.0),
                "take_profit_pct": (8.0, 50.0),
            },
        )
    """

    def __init__(
        self,
        backtest_runner: Callable,
        config: EnsembleConfig | None = None,
        fitness_function: Callable | None = None,
    ):
        """Initialize the ensemble optimizer.

        Args:
            backtest_runner: Async function to run backtests
            config: Ensemble configuration
            fitness_function: Optional custom fitness function
        """
        self.backtest_runner = backtest_runner
        self.config = config or EnsembleConfig()
        self.fitness_function = fitness_function or self._default_fitness

        # Performance tracking
        self._method_performance: dict[str, list[float]] = {m.value: [] for m in OptimizationMethod}

    def _default_fitness(self, metrics: dict[str, float]) -> float:
        """Default fitness function."""
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate_pct", 0) / 100
        max_dd = metrics.get("max_drawdown_pct", 100)
        profit_factor = metrics.get("profit_factor", 0)

        sharpe_norm = min(1.0, max(0, sharpe / 3.0))
        win_norm = min(1.0, max(0, win_rate))
        dd_norm = max(0, 1 - max_dd / 50)
        pf_norm = min(1.0, max(0, profit_factor / 3.0))

        return sharpe_norm * 0.35 + win_norm * 0.20 + dd_norm * 0.25 + pf_norm * 0.20

    async def _run_optuna(
        self,
        param_bounds: dict[str, tuple[float, float]],
        fixed_params: dict[str, Any] | None,
    ) -> OptimizerResult:
        """Run Optuna (Bayesian) optimization."""
        start_time = datetime.utcnow()

        try:
            # Import and run Optuna
            from coldpath.backtest.auto_backtest import (
                AutoBacktestMode,
                AutoModeConfig,
            )

            # Create initial params at midpoint
            initial_params = {k: (v[0] + v[1]) / 2 for k, v in param_bounds.items()}
            if fixed_params:
                initial_params.update(fixed_params)

            config = AutoModeConfig(
                max_iterations=self.config.optuna_iterations,
                max_time_minutes=10,
            )

            auto_mode = AutoBacktestMode(
                backtest_runner=self.backtest_runner,
                config=config,
            )

            result = await auto_mode.run(initial_params=initial_params)

            elapsed = (datetime.utcnow() - start_time).total_seconds()

            return OptimizerResult(
                method=OptimizationMethod.OPTUNA,
                params=result.best_params,
                metrics=result.best_metrics,
                score=result.best_score,
                time_seconds=elapsed,
                iterations=result.total_iterations,
            )

        except Exception as e:
            logger.warning(f"Optuna optimization failed: {e}")
            # Return default result
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            return OptimizerResult(
                method=OptimizationMethod.OPTUNA,
                params={k: (v[0] + v[1]) / 2 for k, v in param_bounds.items()},
                metrics={},
                score=0.0,
                time_seconds=elapsed,
                iterations=0,
            )

    async def _run_genetic(
        self,
        param_bounds: dict[str, tuple[float, float]],
        fixed_params: dict[str, Any] | None,
    ) -> OptimizerResult:
        """Run genetic algorithm optimization."""
        start_time = datetime.utcnow()

        try:
            from coldpath.backtest.genetic_optimizer import (
                GAConfig,
                GeneticOptimizer,
            )

            config = GAConfig(
                population_size=self.config.genetic_population,
                max_generations=self.config.genetic_generations,
            )

            optimizer = GeneticOptimizer(
                backtest_runner=self.backtest_runner,
                config=config,
                fitness_function=self.fitness_function,
            )

            result = await optimizer.optimize(
                param_bounds=param_bounds,
                fixed_params=fixed_params,
            )

            elapsed = (datetime.utcnow() - start_time).total_seconds()

            metrics = result.best_individual.metrics

            return OptimizerResult(
                method=OptimizationMethod.GENETIC,
                params=result.best_individual.genes,
                metrics=metrics,
                score=result.best_individual.fitness,
                time_seconds=elapsed,
                iterations=result.generation,
            )

        except Exception as e:
            logger.warning(f"Genetic optimization failed: {e}")
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            return OptimizerResult(
                method=OptimizationMethod.GENETIC,
                params={k: (v[0] + v[1]) / 2 for k, v in param_bounds.items()},
                metrics={},
                score=0.0,
                time_seconds=elapsed,
                iterations=0,
            )

    async def _run_pareto(
        self,
        param_bounds: dict[str, tuple[float, float]],
        fixed_params: dict[str, Any] | None,
    ) -> OptimizerResult:
        """Run NSGA-II multi-objective optimization."""
        start_time = datetime.utcnow()

        try:
            from coldpath.backtest.pareto_optimizer import (
                NSGA2Config,
                Objective,
                ParetoOptimizer,
            )

            objectives = [
                Objective("sharpe_ratio", maximize=True, weight=0.4),
                Objective("max_drawdown_pct", maximize=False, weight=0.3),
                Objective("win_rate_pct", maximize=True, weight=0.3),
            ]

            config = NSGA2Config(
                population_size=self.config.pareto_population,
                max_generations=self.config.pareto_generations,
            )

            optimizer = ParetoOptimizer(
                backtest_runner=self.backtest_runner,
                objectives=objectives,
                config=config,
            )

            result = await optimizer.optimize(
                param_bounds=param_bounds,
                fixed_params=fixed_params,
            )

            # Get best balanced solution
            best = result.get_best_balanced()

            elapsed = (datetime.utcnow() - start_time).total_seconds()

            return OptimizerResult(
                method=OptimizationMethod.PARETO,
                params=best.params,
                metrics=best.metrics,
                score=sum(best.normalized_objectives.values()) / len(best.normalized_objectives),
                time_seconds=elapsed,
                iterations=result.generation,
            )

        except Exception as e:
            logger.warning(f"Pareto optimization failed: {e}")
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            return OptimizerResult(
                method=OptimizationMethod.PARETO,
                params={k: (v[0] + v[1]) / 2 for k, v in param_bounds.items()},
                metrics={},
                score=0.0,
                time_seconds=elapsed,
                iterations=0,
            )

    async def _run_random(
        self,
        param_bounds: dict[str, tuple[float, float]],
        fixed_params: dict[str, Any] | None,
    ) -> OptimizerResult:
        """Run random search optimization."""
        import random

        start_time = datetime.utcnow()
        rng = random.Random(42)

        best_params = {}
        best_metrics = {}
        best_score = float("-inf")

        for _ in range(self.config.random_samples):
            # Generate random params
            params = {k: rng.uniform(v[0], v[1]) for k, v in param_bounds.items()}
            if fixed_params:
                params.update(fixed_params)

            # Run backtest
            metrics = await self._run_backtest(params)
            score = self.fitness_function(metrics)

            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        return OptimizerResult(
            method=OptimizationMethod.RANDOM,
            params=best_params,
            metrics=best_metrics,
            score=best_score,
            time_seconds=elapsed,
            iterations=self.config.random_samples,
        )

    async def _run_backtest(self, params: dict[str, float]) -> dict[str, float]:
        """Run a single backtest."""
        try:
            result = await self.backtest_runner(params)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            return {"sharpe_ratio": 0, "win_rate_pct": 0, "max_drawdown_pct": 50}

    async def optimize(
        self,
        param_bounds: dict[str, tuple[float, float]],
        fixed_params: dict[str, Any] | None = None,
    ) -> EnsembleResult:
        """Run ensemble optimization.

        Args:
            param_bounds: Parameter bounds as {name: (min, max)}
            fixed_params: Fixed parameters to include

        Returns:
            EnsembleResult with combined optimization results
        """
        start_time = datetime.utcnow()

        # Run each optimizer
        results: list[OptimizerResult] = []

        if self.config.run_parallel:
            # Run all in parallel
            tasks = []
            for method in self.config.methods:
                if method == OptimizationMethod.OPTUNA:
                    tasks.append(self._run_optuna(param_bounds, fixed_params))
                elif method == OptimizationMethod.GENETIC:
                    tasks.append(self._run_genetic(param_bounds, fixed_params))
                elif method == OptimizationMethod.PARETO:
                    tasks.append(self._run_pareto(param_bounds, fixed_params))
                elif method == OptimizationMethod.RANDOM:
                    tasks.append(self._run_random(param_bounds, fixed_params))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            results = [r for r in results if isinstance(r, OptimizerResult)]
        else:
            # Run sequentially
            for method in self.config.methods:
                if method == OptimizationMethod.OPTUNA:
                    results.append(await self._run_optuna(param_bounds, fixed_params))
                elif method == OptimizationMethod.GENETIC:
                    results.append(await self._run_genetic(param_bounds, fixed_params))
                elif method == OptimizationMethod.PARETO:
                    results.append(await self._run_pareto(param_bounds, fixed_params))
                elif method == OptimizationMethod.RANDOM:
                    results.append(await self._run_random(param_bounds, fixed_params))

        # Combine results using ensemble method
        ensemble_params, weights = self._combine_results(results)

        # Run backtest with ensemble params
        ensemble_metrics = await self._run_backtest(ensemble_params)
        ensemble_score = self.fitness_function(ensemble_metrics)

        # Find best individual
        best_individual = max(results, key=lambda r: r.score)

        # Calculate improvement
        if best_individual.score > 0:
            improvement = (ensemble_score - best_individual.score) / best_individual.score * 100
        else:
            improvement = 0.0

        # Track performance
        for result in results:
            self._method_performance[result.method.value].append(result.score)

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        return EnsembleResult(
            ensemble_params=ensemble_params,
            ensemble_metrics=ensemble_metrics,
            ensemble_score=ensemble_score,
            individual_results=results,
            method=self.config.ensemble_method,
            weights=weights,
            best_individual=best_individual.method,
            improvement_over_best=improvement,
            total_time_seconds=elapsed,
        )

    def _combine_results(
        self,
        results: list[OptimizerResult],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Combine optimizer results using ensemble method."""
        if not results:
            return {}, {}

        # Get all parameter names
        param_names = set()
        for r in results:
            param_names.update(r.params.keys())

        ensemble_params = {}
        weights = {}

        if self.config.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
            # Calculate weights based on scores
            total_score = sum(r.score for r in results)
            if total_score > 0:
                for r in results:
                    weights[r.method.value] = r.score / total_score
            else:
                for r in results:
                    weights[r.method.value] = 1.0 / len(results)

            # Weighted average of params
            for param in param_names:
                weighted_sum = 0
                weight_sum = 0
                for r in results:
                    if param in r.params:
                        w = weights.get(r.method.value, 1.0)
                        weighted_sum += r.params[param] * w
                        weight_sum += w

                if weight_sum > 0:
                    ensemble_params[param] = weighted_sum / weight_sum

        elif self.config.ensemble_method == EnsembleMethod.MEDIAN:
            # Median of params
            for r in results:
                weights[r.method.value] = 1.0 / len(results)

            for param in param_names:
                values = [r.params[param] for r in results if param in r.params]
                if values:
                    values.sort()
                    mid = len(values) // 2
                    if len(values) % 2 == 0:
                        ensemble_params[param] = (values[mid - 1] + values[mid]) / 2
                    else:
                        ensemble_params[param] = values[mid]

        elif self.config.ensemble_method == EnsembleMethod.BEST_PERFORMER:
            # Use best individual's params
            best = max(results, key=lambda r: r.score)
            ensemble_params = best.params.copy()
            for r in results:
                weights[r.method.value] = 1.0 if r == best else 0.0

        elif self.config.ensemble_method == EnsembleMethod.VOTING:
            # Voting for each parameter
            for r in results:
                weights[r.method.value] = 1.0 / len(results)

            for param in param_names:
                # Discretize and vote
                values = [r.params[param] for r in results if param in r.params]
                if values:
                    # Use median as consensus
                    values.sort()
                    ensemble_params[param] = values[len(values) // 2]

        else:
            # Default to weighted average
            for r in results:
                weights[r.method.value] = 1.0 / len(results)
            for param in param_names:
                values = [r.params[param] for r in results if param in r.params]
                if values:
                    ensemble_params[param] = sum(values) / len(values)

        return ensemble_params, weights

    def get_method_performance(self) -> dict[str, dict[str, float]]:
        """Get historical performance of each method."""
        performance = {}

        for method, scores in self._method_performance.items():
            if scores:
                performance[method] = {
                    "avg_score": sum(scores) / len(scores),
                    "best_score": max(scores),
                    "runs": len(scores),
                }

        return performance


# Convenience function
async def ensemble_optimize(
    backtest_runner: Callable,
    param_bounds: dict[str, tuple[float, float]],
    methods: list[str] | None = None,
    **kwargs,
) -> EnsembleResult:
    """Run ensemble optimization with sensible defaults."""
    if methods is None:
        method_enums = [OptimizationMethod.OPTUNA, OptimizationMethod.GENETIC]
    else:
        method_map = {
            "optuna": OptimizationMethod.OPTUNA,
            "genetic": OptimizationMethod.GENETIC,
            "pareto": OptimizationMethod.PARETO,
            "random": OptimizationMethod.RANDOM,
        }
        method_enums = [method_map.get(m, OptimizationMethod.OPTUNA) for m in methods]

    config = EnsembleConfig(methods=method_enums, **kwargs)

    ensemble = EnsembleOptimizer(
        backtest_runner=backtest_runner,
        config=config,
    )

    return await ensemble.optimize(param_bounds=param_bounds)
