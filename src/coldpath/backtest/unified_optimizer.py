"""
Unified Optimizer Interface - Single API for all optimization methods.

Provides a consistent interface for running any optimization method:
- Automatic method selection based on problem size
- Fallback strategies when methods fail
- Consistent result format
- Easy switching between methods

Usage:
    from coldpath.backtest.unified_optimizer import UnifiedOptimizer, OptimizerType

    optimizer = UnifiedOptimizer(backtest_runner=my_backtest_fn)

    # Auto-select best method
    result = await optimizer.optimize(
        param_bounds={"stop_loss_pct": (3, 15), ...},
        method=OptimizerType.AUTO,
    )

    # Or specify method
    result = await optimizer.optimize(
        param_bounds={...},
        method=OptimizerType.GENETIC,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """Available optimizer types."""

    AUTO = "auto"  # Auto-select based on problem
    OPTUNA = "optuna"  # Bayesian (TPE)
    GENETIC = "genetic"  # Genetic algorithm
    PARETO = "pareto"  # NSGA-II multi-objective
    ENSEMBLE = "ensemble"  # Ensemble of methods
    RANDOM = "random"  # Random search
    RL = "rl"  # Reinforcement learning


@dataclass
class UnifiedOptimizationResult:
    """Unified result format for all optimizers."""

    # Best found
    best_params: dict[str, float]
    best_metrics: dict[str, float]
    best_score: float

    # Method used
    method: OptimizerType
    actual_method: OptimizerType | None = None  # If AUTO was used

    # Optimization info
    iterations: int = 0
    time_seconds: float = 0.0

    # Additional info
    convergence_achieved: bool = False
    pareto_front: list[dict[str, Any]] = field(default_factory=list)

    # Warnings/recommendations
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_metrics": self.best_metrics,
            "best_score": self.best_score,
            "method": self.method.value,
            "actual_method": self.actual_method.value if self.actual_method else None,
            "iterations": self.iterations,
            "time_seconds": self.time_seconds,
            "convergence_achieved": self.convergence_achieved,
            "pareto_front_size": len(self.pareto_front),
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


@dataclass
class UnifiedOptimizerConfig:
    """Configuration for unified optimizer."""

    # Default method
    default_method: OptimizerType = OptimizerType.AUTO

    # Auto-selection thresholds
    auto_use_genetic_if_params_above: int = 5
    auto_use_pareto_for_multi_objective: bool = True

    # Time limits
    max_time_minutes: float = 30.0

    # Iteration limits
    max_iterations: int = 100

    # Fallback
    enable_fallback: bool = True
    fallback_method: OptimizerType = OptimizerType.RANDOM

    # Parallel runs for ensemble
    ensemble_methods: list[OptimizerType] = field(
        default_factory=lambda: [
            OptimizerType.OPTUNA,
            OptimizerType.GENETIC,
        ]
    )


class UnifiedOptimizer:
    """Unified interface for all optimization methods.

    Features:
    - Single API for all optimizers
    - Auto method selection
    - Fallback on failure
    - Consistent result format

    Example:
        optimizer = UnifiedOptimizer(
            backtest_runner=my_backtest,
            config=UnifiedOptimizerConfig(default_method=OptimizerType.AUTO),
        )

        result = await optimizer.optimize(
            param_bounds={
                "stop_loss_pct": (3.0, 15.0),
                "take_profit_pct": (8.0, 50.0),
                "max_position_sol": (0.01, 0.3),
            },
            fixed_params={"initial_capital_sol": 1.0},
        )

        print(f"Best params: {result.best_params}")
        print(f"Method used: {result.actual_method}")
    """

    def __init__(
        self,
        backtest_runner: Callable,
        config: UnifiedOptimizerConfig | None = None,
        fitness_function: Callable | None = None,
    ):
        """Initialize the unified optimizer.

        Args:
            backtest_runner: Async function to run backtests
            config: Optimizer configuration
            fitness_function: Optional custom fitness function
        """
        self.backtest_runner = backtest_runner
        self.config = config or UnifiedOptimizerConfig()
        self.fitness_function = fitness_function or self._default_fitness

        # Statistics
        self._optimization_count = 0
        self._method_usage: dict[str, int] = {}

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

    def _select_method(
        self,
        param_bounds: dict[str, tuple[float, float]],
        objectives: list[str] | None = None,
    ) -> OptimizerType:
        """Auto-select best optimization method."""
        num_params = len(param_bounds)

        # Multi-objective -> Pareto
        if objectives and len(objectives) > 1 and self.config.auto_use_pareto_for_multi_objective:
            return OptimizerType.PARETO

        # Many parameters -> Genetic
        if num_params > self.config.auto_use_genetic_if_params_above:
            return OptimizerType.GENETIC

        # Default to Optuna (Bayesian)
        return OptimizerType.OPTUNA

    async def optimize(
        self,
        param_bounds: dict[str, tuple[float, float]],
        fixed_params: dict[str, Any] | None = None,
        method: OptimizerType | None = None,
        objectives: list[str] | None = None,
        initial_params: dict[str, float] | None = None,
    ) -> UnifiedOptimizationResult:
        """Run optimization with unified interface.

        Args:
            param_bounds: Parameter bounds {name: (min, max)}
            fixed_params: Parameters that don't change
            method: Optimization method (None = use default)
            objectives: List of objectives for multi-objective
            initial_params: Starting parameters

        Returns:
            UnifiedOptimizationResult
        """
        start_time = datetime.utcnow()

        # Determine method
        if method is None:
            method = self.config.default_method

        actual_method = method
        if method == OptimizerType.AUTO:
            actual_method = self._select_method(param_bounds, objectives)

        # Run optimization
        result = None
        warnings = []

        try:
            result = await self._run_method(
                method=actual_method,
                param_bounds=param_bounds,
                fixed_params=fixed_params,
                initial_params=initial_params,
                objectives=objectives,
            )
        except Exception as e:
            logger.warning(f"Optimization with {actual_method.value} failed: {e}")
            warnings.append(f"{actual_method.value} failed: {str(e)}")

            # Try fallback
            if self.config.enable_fallback:
                logger.info(f"Trying fallback method: {self.config.fallback_method.value}")
                actual_method = self.config.fallback_method

                try:
                    result = await self._run_method(
                        method=actual_method,
                        param_bounds=param_bounds,
                        fixed_params=fixed_params,
                        initial_params=initial_params,
                        objectives=objectives,
                    )
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
                    warnings.append(f"Fallback {actual_method.value} failed: {str(e2)}")

        # If all failed, return defaults
        if result is None:
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            return UnifiedOptimizationResult(
                best_params={k: (v[0] + v[1]) / 2 for k, v in param_bounds.items()},
                best_metrics={},
                best_score=0.0,
                method=method,
                actual_method=actual_method,
                iterations=0,
                time_seconds=elapsed,
                convergence_achieved=False,
                warnings=warnings,
                recommendations=["All optimization methods failed - check backtest runner"],
            )

        # Update statistics
        self._optimization_count += 1
        self._method_usage[actual_method.value] = self._method_usage.get(actual_method.value, 0) + 1

        # Add warnings
        result.warnings.extend(warnings)
        result.method = method
        result.actual_method = actual_method

        return result

    async def _run_method(
        self,
        method: OptimizerType,
        param_bounds: dict[str, tuple[float, float]],
        fixed_params: dict[str, Any] | None,
        initial_params: dict[str, float] | None,
        objectives: list[str] | None,
    ) -> UnifiedOptimizationResult:
        """Run a specific optimization method."""
        start_time = datetime.utcnow()

        if method == OptimizerType.OPTUNA:
            return await self._run_optuna(param_bounds, fixed_params, initial_params, start_time)

        elif method == OptimizerType.GENETIC:
            return await self._run_genetic(param_bounds, fixed_params, initial_params, start_time)

        elif method == OptimizerType.PARETO:
            return await self._run_pareto(
                param_bounds, fixed_params, initial_params, objectives, start_time
            )

        elif method == OptimizerType.ENSEMBLE:
            return await self._run_ensemble(param_bounds, fixed_params, initial_params, start_time)

        elif method == OptimizerType.RL:
            return await self._run_rl(param_bounds, fixed_params, initial_params, start_time)

        elif method == OptimizerType.RANDOM:
            return await self._run_random(param_bounds, fixed_params, initial_params, start_time)

        else:
            raise ValueError(f"Unknown method: {method}")

    async def _run_optuna(self, param_bounds, fixed_params, initial_params, start_time):
        """Run Optuna optimization."""
        from coldpath.backtest.auto_backtest import AutoBacktestMode, AutoModeConfig

        if initial_params is None:
            initial_params = {k: (v[0] + v[1]) / 2 for k, v in param_bounds.items()}
        if fixed_params:
            initial_params.update(fixed_params)

        config = AutoModeConfig(
            max_iterations=self.config.max_iterations,
            max_time_minutes=int(self.config.max_time_minutes),
        )

        auto = AutoBacktestMode(backtest_runner=self.backtest_runner, config=config)
        result = await auto.run(initial_params=initial_params)

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        return UnifiedOptimizationResult(
            best_params=result.best_params,
            best_metrics=result.best_metrics,
            best_score=result.best_score,
            method=OptimizerType.OPTUNA,
            iterations=result.total_iterations,
            time_seconds=elapsed,
            convergence_achieved=(result.state.value in ["completed", "converged"]),
        )

    async def _run_genetic(self, param_bounds, fixed_params, initial_params, start_time):
        """Run genetic algorithm optimization."""
        from coldpath.backtest.genetic_optimizer import GAConfig, GeneticOptimizer

        config = GAConfig(
            max_generations=self.config.max_iterations,
            max_time_minutes=self.config.max_time_minutes,
        )

        optimizer = GeneticOptimizer(
            backtest_runner=self.backtest_runner,
            config=config,
            fitness_function=self.fitness_function,
        )

        result = await optimizer.optimize(
            param_bounds=param_bounds,
            fixed_params=fixed_params,
            initial_population=[initial_params] if initial_params else None,
        )

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        return UnifiedOptimizationResult(
            best_params=result.best_individual.genes,
            best_metrics=result.best_individual.metrics,
            best_score=result.best_individual.fitness,
            method=OptimizerType.GENETIC,
            iterations=result.generation,
            time_seconds=elapsed,
            convergence_achieved=len(result.convergence_history) > 10,
            pareto_front=result.pareto_front,
        )

    async def _run_pareto(self, param_bounds, fixed_params, initial_params, objectives, start_time):
        """Run NSGA-II multi-objective optimization."""
        from coldpath.backtest.pareto_optimizer import NSGA2Config, Objective, ParetoOptimizer

        if objectives is None:
            objectives_list = [
                Objective("sharpe_ratio", maximize=True),
                Objective("max_drawdown_pct", maximize=False),
            ]
        else:
            objectives_list = [
                Objective(obj, maximize=(obj != "max_drawdown_pct")) for obj in objectives
            ]

        config = NSGA2Config(
            max_generations=self.config.max_iterations,
            max_time_minutes=self.config.max_time_minutes,
        )

        optimizer = ParetoOptimizer(
            backtest_runner=self.backtest_runner,
            objectives=objectives_list,
            config=config,
        )

        result = await optimizer.optimize(
            param_bounds=param_bounds,
            fixed_params=fixed_params,
        )

        best = result.get_best_balanced()
        elapsed = (datetime.utcnow() - start_time).total_seconds()

        return UnifiedOptimizationResult(
            best_params=best.params,
            best_metrics=best.metrics,
            best_score=sum(best.normalized_objectives.values()) / len(best.normalized_objectives),
            method=OptimizerType.PARETO,
            iterations=result.generation,
            time_seconds=elapsed,
            convergence_achieved=len(result.hypervolume_history) > 10,
            pareto_front=[s.to_dict() for s in result.pareto_front],
        )

    async def _run_ensemble(self, param_bounds, fixed_params, initial_params, start_time):
        """Run ensemble optimization."""
        from coldpath.backtest.ensemble_optimizer import (
            EnsembleConfig,
            EnsembleOptimizer,
            OptimizationMethod,
        )

        method_map = {
            OptimizerType.OPTUNA: OptimizationMethod.OPTUNA,
            OptimizerType.GENETIC: OptimizationMethod.GENETIC,
            OptimizerType.PARETO: OptimizationMethod.PARETO,
            OptimizerType.RANDOM: OptimizationMethod.RANDOM,
        }

        methods = [
            method_map.get(m, OptimizationMethod.OPTUNA) for m in self.config.ensemble_methods
        ]

        config = EnsembleConfig(methods=methods)

        optimizer = EnsembleOptimizer(
            backtest_runner=self.backtest_runner,
            config=config,
            fitness_function=self.fitness_function,
        )

        result = await optimizer.optimize(
            param_bounds=param_bounds,
            fixed_params=fixed_params,
        )

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        return UnifiedOptimizationResult(
            best_params=result.ensemble_params,
            best_metrics=result.ensemble_metrics,
            best_score=result.ensemble_score,
            method=OptimizerType.ENSEMBLE,
            iterations=sum(r.iterations for r in result.individual_results),
            time_seconds=elapsed,
            recommendations=[f"Ensemble used {len(result.individual_results)} methods"],
        )

    async def _run_rl(self, param_bounds, fixed_params, initial_params, start_time):
        """Run RL-based optimization."""
        from coldpath.backtest.rl_optimizer import (
            RLAgentConfig,
            RLParameterAgent,
        )

        # Convert param_bounds format
        bounds_dict = {k: (v[0], v[1], (v[1] - v[0]) / 10, False) for k, v in param_bounds.items()}

        config = RLAgentConfig(
            exploration_rate=0.3,
        )

        agent = RLParameterAgent(
            param_bounds=bounds_dict,
            config=config,
        )

        if initial_params is None:
            initial_params = {k: (v[0] + v[1]) / 2 for k, v in param_bounds.items()}
        if fixed_params:
            initial_params.update(fixed_params)

        # Train online
        train_result = await agent.train_online(
            initial_params=initial_params,
            backtest_runner=self.backtest_runner,
            num_episodes=30,
        )

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        # Get final tuned params
        final_params, confidence = agent.suggest_parameters(
            initial_params,
            deterministic=True,
        )
        final_metrics = await self._run_backtest(final_params)
        final_score = self.fitness_function(final_metrics)

        return UnifiedOptimizationResult(
            best_params=final_params,
            best_metrics=final_metrics,
            best_score=final_score,
            method=OptimizerType.RL,
            iterations=train_result.episodes_trained,
            time_seconds=elapsed,
            convergence_achieved=train_result.final_exploration_rate < 0.15,
        )

    async def _run_random(self, param_bounds, fixed_params, initial_params, start_time):
        """Run random search."""
        import random

        rng = random.Random(42)
        best_params = {}
        best_metrics = {}
        best_score = float("-inf")

        for _i in range(self.config.max_iterations):
            params = {k: rng.uniform(v[0], v[1]) for k, v in param_bounds.items()}
            if fixed_params:
                params.update(fixed_params)

            metrics = await self._run_backtest(params)
            score = self.fitness_function(metrics)

            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        return UnifiedOptimizationResult(
            best_params=best_params,
            best_metrics=best_metrics,
            best_score=best_score,
            method=OptimizerType.RANDOM,
            iterations=self.config.max_iterations,
            time_seconds=elapsed,
        )

    async def _run_backtest(self, params: dict[str, float]) -> dict[str, float]:
        """Run a single backtest."""
        try:
            result = await self.backtest_runner(params)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            return {"sharpe_ratio": 0, "win_rate_pct": 0, "max_drawdown_pct": 50}

    def get_statistics(self) -> dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "total_optimizations": self._optimization_count,
            "method_usage": self._method_usage,
        }


# Convenience function
async def optimize_params(
    backtest_runner: Callable,
    param_bounds: dict[str, tuple[float, float]],
    method: str = "auto",
    **kwargs,
) -> UnifiedOptimizationResult:
    """Optimize parameters with unified interface.

    Args:
        backtest_runner: Async backtest function
        param_bounds: Parameter bounds
        method: "auto", "optuna", "genetic", "pareto", "ensemble", "random"

    Returns:
        UnifiedOptimizationResult
    """
    method_map = {
        "auto": OptimizerType.AUTO,
        "optuna": OptimizerType.OPTUNA,
        "genetic": OptimizerType.GENETIC,
        "pareto": OptimizerType.PARETO,
        "ensemble": OptimizerType.ENSEMBLE,
        "rl": OptimizerType.RL,
        "random": OptimizerType.RANDOM,
    }

    optimizer = UnifiedOptimizer(backtest_runner=backtest_runner)

    return await optimizer.optimize(
        param_bounds=param_bounds, method=method_map.get(method, OptimizerType.AUTO), **kwargs
    )
