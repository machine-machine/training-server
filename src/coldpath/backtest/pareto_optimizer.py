"""
Multi-Objective Pareto Optimizer - NSGA-II for conflicting objectives.

Implements the Non-dominated Sorting Genetic Algorithm II (NSGA-II) for
optimizing multiple conflicting objectives simultaneously:

- Maximize returns (Sharpe, total return)
- Minimize risk (drawdown, volatility)
- Maximize robustness (stability across perturbations)
- Maximize win rate

The result is a Pareto front of non-dominated solutions where improving
one objective would worsen another.

Usage:
    from coldpath.backtest.pareto_optimizer import ParetoOptimizer, Objective

    optimizer = ParetoOptimizer(
        backtest_runner=my_backtest_fn,
        objectives=[
            Objective("sharpe_ratio", maximize=True, weight=0.4),
            Objective("max_drawdown_pct", maximize=False, weight=0.3),
            Objective("win_rate_pct", maximize=True, weight=0.3),
        ],
    )

    result = await optimizer.optimize(param_bounds={...})

    # Get Pareto front
    for solution in result.pareto_front:
        print(f"Sharpe: {solution.objectives['sharpe_ratio']:.2f}")
        print(f"MaxDD: {solution.objectives['max_drawdown_pct']:.1f}%")
"""

import asyncio
import copy
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Type of optimization objective."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class Objective:
    """Definition of an optimization objective."""

    name: str
    maximize: bool = True
    weight: float = 1.0
    target_value: float | None = None  # Aspiration level

    def better(self, val1: float, val2: float) -> bool:
        """Check if val1 is better than val2 for this objective."""
        if self.maximize:
            return val1 > val2
        return val1 < val2


@dataclass
class ParetoSolution:
    """A solution on the Pareto front."""

    params: dict[str, float]
    objectives: dict[str, float]  # Raw objective values
    normalized_objectives: dict[str, float] = field(default_factory=dict)
    rank: int = 0  # Pareto rank (0 = best front)
    crowding_distance: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)

    def dominates(self, other: "ParetoSolution", objectives: list[Objective]) -> bool:
        """Check if this solution dominates another (Pareto dominance)."""
        at_least_one_better = False

        for obj in objectives:
            self_val = self.objectives.get(obj.name, 0)
            other_val = other.objectives.get(obj.name, 0)

            if obj.maximize:
                if self_val < other_val:
                    return False  # Worse in at least one objective
                if self_val > other_val:
                    at_least_one_better = True
            else:
                if self_val > other_val:
                    return False
                if self_val < other_val:
                    at_least_one_better = True

        return at_least_one_better

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "objectives": self.objectives,
            "rank": self.rank,
            "crowding_distance": self.crowding_distance,
        }


@dataclass
class ParetoResult:
    """Result from multi-objective optimization."""

    pareto_front: list[ParetoSolution]  # Best non-dominated solutions
    all_fronts: list[list[ParetoSolution]]  # All Pareto fronts
    population: list[ParetoSolution]
    generation: int
    total_evaluations: int
    time_seconds: float
    hypervolume_history: list[float]
    best_by_objective: dict[str, ParetoSolution]

    def get_best_balanced(self, weights: dict[str, float] | None = None) -> ParetoSolution:
        """Get best solution based on weighted objectives."""
        if not self.pareto_front:
            raise ValueError("No solutions on Pareto front")

        best_score = float("-inf")
        best_solution = self.pareto_front[0]

        for solution in self.pareto_front:
            score = 0.0
            for obj_name, norm_val in solution.normalized_objectives.items():
                weight = (weights or {}).get(obj_name, 1.0)
                score += weight * norm_val

            if score > best_score:
                best_score = score
                best_solution = solution

        return best_solution

    def to_dict(self) -> dict[str, Any]:
        return {
            "pareto_front_size": len(self.pareto_front),
            "pareto_front": [s.to_dict() for s in self.pareto_front[:10]],
            "total_fronts": len(self.all_fronts),
            "generation": self.generation,
            "total_evaluations": self.total_evaluations,
            "time_seconds": self.time_seconds,
            "best_by_objective": {k: v.to_dict() for k, v in self.best_by_objective.items()},
        }


@dataclass
class NSGA2Config:
    """Configuration for NSGA-II optimizer."""

    # Population
    population_size: int = 100

    # Selection
    tournament_size: int = 2

    # Crossover
    crossover_rate: float = 0.9
    crossover_distribution_index: float = 20.0  # SBX distribution

    # Mutation
    mutation_rate: float = 0.1
    mutation_distribution_index: float = 20.0  # Polynomial mutation

    # Termination
    max_generations: int = 200
    max_time_minutes: float = 60.0

    # Reference point for hypervolume (for minimization objectives)
    reference_point: dict[str, float] | None = None

    # Parallelization
    parallel_evaluations: int = 10


class ParetoOptimizer:
    """NSGA-II Multi-objective optimizer for Pareto-optimal solutions.

    Finds trade-off solutions when objectives conflict:
    - High returns vs low drawdown
    - High win rate vs high profit per trade
    - Robustness vs raw performance

    Example:
        optimizer = ParetoOptimizer(
            backtest_runner=my_backtest,
            objectives=[
                Objective("sharpe_ratio", maximize=True),
                Objective("max_drawdown_pct", maximize=False),
            ],
            config=NSGA2Config(population_size=100),
        )

        result = await optimizer.optimize(
            param_bounds={"stop_loss_pct": (3, 15), "take_profit_pct": (8, 50)},
        )

        # Access Pareto front
        for solution in result.pareto_front:
            print(f"Sharpe={solution.objectives['sharpe_ratio']:.2f}, "
                  f"DD={solution.objectives['max_drawdown_pct']:.1f}%")
    """

    def __init__(
        self,
        backtest_runner: Callable,
        objectives: list[Objective],
        config: NSGA2Config | None = None,
    ):
        """Initialize the Pareto optimizer.

        Args:
            backtest_runner: Async function to run backtests
            objectives: List of optimization objectives
            config: NSGA-II configuration
        """
        if len(objectives) < 2:
            raise ValueError("Pareto optimization requires at least 2 objectives")

        self.backtest_runner = backtest_runner
        self.objectives = objectives
        self.config = config or NSGA2Config()

        # State
        self._population: list[ParetoSolution] = []
        self._generation = 0
        self._evaluations = 0
        self._objective_ranges: dict[str, tuple[float, float]] = {}

        # Param bounds (set during optimize)
        self._param_bounds: dict[str, tuple[float, float]] = {}
        self._fixed_params: dict[str, Any] = {}

        # Random state
        self._rng = random.Random(42)

    def _initialize_population(self) -> list[ParetoSolution]:
        """Initialize random population."""
        population = []

        for _ in range(self.config.population_size):
            params = {}
            for param, (low, high) in self._param_bounds.items():
                params[param] = self._rng.uniform(low, high)
            population.append(ParetoSolution(params=params, objectives={}))

        return population

    async def _evaluate_population(self, population: list[ParetoSolution]):
        """Evaluate all solutions in population."""
        tasks = []
        for solution in population:
            params = {**self._fixed_params, **solution.params}
            tasks.append(self.backtest_runner(params))

        batch_size = self.config.parallel_evaluations
        results = []

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)

        for solution, result in zip(population, results, strict=False):
            self._evaluations += 1

            if isinstance(result, Exception):
                solution.metrics = {}
                for obj in self.objectives:
                    solution.objectives[obj.name] = 0.0 if obj.maximize else 1000.0
            else:
                solution.metrics = result if isinstance(result, dict) else {}
                for obj in self.objectives:
                    solution.objectives[obj.name] = solution.metrics.get(obj.name, 0.0)

    def _update_objective_ranges(self, population: list[ParetoSolution]):
        """Update objective normalization ranges."""
        for obj in self.objectives:
            values = [s.objectives.get(obj.name, 0) for s in population]
            if values:
                self._objective_ranges[obj.name] = (min(values), max(values))

    def _normalize_objectives(self, population: list[ParetoSolution]):
        """Normalize objectives to [0, 1] range."""
        for solution in population:
            for obj in self.objectives:
                raw_val = solution.objectives.get(obj.name, 0)
                min_val, max_val = self._objective_ranges.get(obj.name, (0, 1))

                if max_val > min_val:
                    norm = (raw_val - min_val) / (max_val - min_val)
                else:
                    norm = 0.5

                # Flip for minimization (so higher is always better after normalization)
                if not obj.maximize:
                    norm = 1.0 - norm

                solution.normalized_objectives[obj.name] = norm

    def _non_dominated_sort(self, population: list[ParetoSolution]) -> list[list[ParetoSolution]]:
        """Sort population into Pareto fronts."""
        fronts: list[list[ParetoSolution]] = [[]]
        domination_count = {id(s): 0 for s in population}
        dominated_solutions = {id(s): [] for s in population}

        for p in population:
            for q in population:
                if p is q:
                    continue

                if p.dominates(q, self.objectives):
                    dominated_solutions[id(p)].append(q)
                elif q.dominates(p, self.objectives):
                    domination_count[id(p)] += 1

            if domination_count[id(p)] == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[id(p)]:
                    domination_count[id(q)] -= 1
                    if domination_count[id(q)] == 0:
                        q.rank = i + 1
                        next_front.append(q)

            i += 1
            fronts.append(next_front)

        return [f for f in fronts if f]

    def _calculate_crowding_distance(self, front: list[ParetoSolution]):
        """Calculate crowding distance for diversity preservation."""
        if len(front) <= 2:
            for solution in front:
                solution.crowding_distance = float("inf")
            return

        for solution in front:
            solution.crowding_distance = 0.0

        for obj in self.objectives:
            # Sort by this objective
            front.sort(key=lambda s: s.objectives.get(obj.name, 0))

            # Set boundary solutions to infinity
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            min_val = front[0].objectives.get(obj.name, 0)
            max_val = front[-1].objectives.get(obj.name, 0)

            if max_val > min_val:
                for i in range(1, len(front) - 1):
                    distance = (
                        front[i + 1].objectives.get(obj.name, 0)
                        - front[i - 1].objectives.get(obj.name, 0)
                    ) / (max_val - min_val)
                    front[i].crowding_distance += distance

    def _select_parent(self, population: list[ParetoSolution]) -> ParetoSolution:
        """Tournament selection based on rank and crowding distance."""
        tournament = self._rng.sample(population, min(self.config.tournament_size, len(population)))

        def compare(a: ParetoSolution, b: ParetoSolution) -> int:
            if a.rank < b.rank:
                return -1
            if a.rank > b.rank:
                return 1
            if a.crowding_distance > b.crowding_distance:
                return -1
            if a.crowding_distance < b.crowding_distance:
                return 1
            return 0

        tournament.sort(key=lambda x: (x.rank, -x.crowding_distance))
        return tournament[0]

    def _crossover(
        self, parent1: ParetoSolution, parent2: ParetoSolution
    ) -> tuple[ParetoSolution, ParetoSolution]:
        """Simulated Binary Crossover (SBX)."""
        if self._rng.random() > self.config.crossover_rate:
            return (
                ParetoSolution(params=copy.deepcopy(parent1.params), objectives={}),
                ParetoSolution(params=copy.deepcopy(parent2.params), objectives={}),
            )

        child1_params = {}
        child2_params = {}
        eta = self.config.crossover_distribution_index

        for param, (low, high) in self._param_bounds.items():
            p1_val = parent1.params.get(param, (low + high) / 2)
            p2_val = parent2.params.get(param, (low + high) / 2)

            if abs(p1_val - p2_val) < 1e-10:
                child1_params[param] = p1_val
                child2_params[param] = p2_val
                continue

            if p1_val > p2_val:
                p1_val, p2_val = p2_val, p1_val

            u = self._rng.random()
            beta = 1.0 + (2.0 * (p1_val - low) / (p2_val - p1_val))
            alpha = 2.0 - beta ** (-(eta + 1.0))

            if u <= 1.0 / alpha:
                beta_q = (u * alpha) ** (1.0 / (eta + 1.0))
            else:
                beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1.0))

            c1 = 0.5 * ((p1_val + p2_val) - beta_q * (p2_val - p1_val))
            c2 = 0.5 * ((p1_val + p2_val) + beta_q * (p2_val - p1_val))

            child1_params[param] = max(low, min(high, c1))
            child2_params[param] = max(low, min(high, c2))

        return (
            ParetoSolution(params=child1_params, objectives={}),
            ParetoSolution(params=child2_params, objectives={}),
        )

    def _mutate(self, solution: ParetoSolution) -> ParetoSolution:
        """Polynomial mutation."""
        mutated = ParetoSolution(
            params=copy.deepcopy(solution.params),
            objectives=copy.deepcopy(solution.objectives),
        )

        eta = self.config.mutation_distribution_index

        for param, (low, high) in self._param_bounds.items():
            if self._rng.random() < self.config.mutation_rate:
                val = mutated.params[param]
                delta1 = (val - low) / (high - low)
                delta2 = (high - val) / (high - low)

                u = self._rng.random()
                mut_pow = 1.0 / (eta + 1.0)

                if u < 0.5:
                    xy = 1.0 - delta1
                    val_mut = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1.0))
                    delta_q = val_mut**mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val_mut = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1.0))
                    delta_q = 1.0 - val_mut**mut_pow

                mutated.params[param] = max(low, min(high, val + delta_q * (high - low)))

        return mutated

    def _calculate_hypervolume(self, front: list[ParetoSolution]) -> float:
        """Calculate hypervolume indicator (simplified 2D version)."""
        if len(front) < 2 or len(self.objectives) < 2:
            return 0.0

        # Use first two objectives
        obj1, obj2 = self.objectives[0], self.objectives[1]
        ref1 = self.config.reference_point.get(obj1.name, 0) if self.config.reference_point else 0
        ref2 = self.config.reference_point.get(obj2.name, 0) if self.config.reference_point else 0

        # Sort by first objective
        sorted_front = sorted(front, key=lambda s: s.objectives.get(obj1.name, 0))

        hypervolume = 0.0
        prev_val = ref1 if obj1.maximize else ref1

        for solution in sorted_front:
            val1 = solution.objectives.get(obj1.name, 0)
            val2 = solution.objectives.get(obj2.name, 0)

            if obj1.maximize:
                width = val1 - prev_val
            else:
                width = prev_val - val1

            if obj2.maximize:
                height = val2 - ref2
            else:
                height = ref2 - val2

            hypervolume += max(0, width) * max(0, height)
            prev_val = val1

        return hypervolume

    async def optimize(
        self,
        param_bounds: dict[str, tuple[float, float]],
        fixed_params: dict[str, Any] | None = None,
    ) -> ParetoResult:
        """Run NSGA-II multi-objective optimization.

        Args:
            param_bounds: Dict mapping parameter name to (min, max) bounds
            fixed_params: Parameters that don't change

        Returns:
            ParetoResult with Pareto front and all fronts
        """
        start_time = datetime.utcnow()

        self._param_bounds = param_bounds
        self._fixed_params = fixed_params or {}
        self._generation = 0
        self._evaluations = 0

        logger.info(
            f"Starting NSGA-II optimization: "
            f"population={self.config.population_size}, "
            f"objectives={[o.name for o in self.objectives]}"
        )

        # Initialize population
        self._population = self._initialize_population()
        await self._evaluate_population(self._population)

        # Update ranges and normalize
        self._update_objective_ranges(self._population)
        self._normalize_objectives(self._population)

        # Non-dominated sort
        fronts = self._non_dominated_sort(self._population)
        for front in fronts:
            self._calculate_crowding_distance(front)

        hypervolume_history = [self._calculate_hypervolume(fronts[0])]

        logger.info(
            f"Generation 0: Pareto front size={len(fronts[0])}, "
            f"hypervolume={hypervolume_history[-1]:.4f}"
        )

        # Main loop
        max_time_seconds = self.config.max_time_minutes * 60

        while self._generation < self.config.max_generations:
            self._generation += 1

            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > max_time_seconds:
                logger.info(f"Time limit reached after {self._generation} generations")
                break

            # Create offspring
            offspring = []
            while len(offspring) < self.config.population_size:
                parent1 = self._select_parent(self._population)
                parent2 = self._select_parent(self._population)

                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                offspring.extend([child1, child2])

            offspring = offspring[: self.config.population_size]

            # Evaluate offspring
            await self._evaluate_population(offspring)

            # Combine and select
            combined = self._population + offspring
            self._update_objective_ranges(combined)
            self._normalize_objectives(combined)

            fronts = self._non_dominated_sort(combined)
            for front in fronts:
                self._calculate_crowding_distance(front)

            # Select new population
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.config.population_size:
                    new_population.extend(front)
                else:
                    # Sort by crowding distance and take remaining
                    front.sort(key=lambda x: -x.crowding_distance)
                    remaining = self.config.population_size - len(new_population)
                    new_population.extend(front[:remaining])
                    break

            self._population = new_population

            # Re-sort for next generation
            fronts = self._non_dominated_sort(self._population)
            for front in fronts:
                self._calculate_crowding_distance(front)

            hypervolume_history.append(self._calculate_hypervolume(fronts[0]))

            if self._generation % 20 == 0 or self._generation < 5:
                logger.info(
                    f"Generation {self._generation}: "
                    f"Pareto front={len(fronts[0])}, "
                    f"hypervolume={hypervolume_history[-1]:.4f}"
                )

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        # Find best by each objective
        best_by_objective = {}
        for obj in self.objectives:
            if obj.maximize:
                best = max(fronts[0], key=lambda s: s.objectives.get(obj.name, 0))
            else:
                best = min(fronts[0], key=lambda s: s.objectives.get(obj.name, float("inf")))
            best_by_objective[obj.name] = best

        logger.info(
            f"NSGA-II complete: generations={self._generation}, "
            f"Pareto front={len(fronts[0])}, time={elapsed:.1f}s"
        )

        return ParetoResult(
            pareto_front=fronts[0],
            all_fronts=fronts,
            population=self._population,
            generation=self._generation,
            total_evaluations=self._evaluations,
            time_seconds=elapsed,
            hypervolume_history=hypervolume_history,
            best_by_objective=best_by_objective,
        )


# Convenience function
async def pareto_optimize(
    backtest_runner: Callable,
    param_bounds: dict[str, tuple[float, float]],
    objectives: list[Objective] | None = None,
    **kwargs,
) -> ParetoResult:
    """Run NSGA-II optimization with sensible defaults.

    Default objectives: maximize Sharpe, minimize drawdown, maximize win rate.
    """
    if objectives is None:
        objectives = [
            Objective("sharpe_ratio", maximize=True, weight=0.4),
            Objective("max_drawdown_pct", maximize=False, weight=0.3),
            Objective("win_rate_pct", maximize=True, weight=0.3),
        ]

    config = NSGA2Config(**kwargs)
    optimizer = ParetoOptimizer(
        backtest_runner=backtest_runner,
        objectives=objectives,
        config=config,
    )

    return await optimizer.optimize(param_bounds=param_bounds)
