"""
Multi-Objective Optimizer - NSGA-II Pareto frontier optimization.

Optimizes multiple competing objectives simultaneously:
- Maximize Sharpe ratio
- Minimize max drawdown
- Maximize win rate
- Minimize fee drag

Returns Pareto frontier of non-dominated solutions.
User selects preferred trade-off point.

Algorithm: NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- Non-dominated sorting into Pareto fronts
- Crowding distance for diversity preservation
- Binary tournament selection
- Simulated binary crossover (SBX)
- Polynomial mutation
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Objective:
    """Optimization objective definition.

    Args:
        name: Objective name (e.g., "sharpe_ratio").
        maximize: True to maximize, False to minimize.
        weight: Relative importance weight (default 1.0).
        constraint: Optional (min, max) bounds for the objective.
    """

    name: str
    maximize: bool
    weight: float = 1.0
    constraint: tuple[float, float] | None = None


@dataclass
class ParetoSolution:
    """A solution on (or near) the Pareto frontier.

    Attributes:
        params: Strategy parameter dictionary.
        objectives: Evaluated objective values.
        rank: Pareto front rank (0 = non-dominated).
        crowding_distance: Diversity measure.
        is_dominated: Whether dominated by another solution.
    """

    params: dict[str, Any]
    objectives: dict[str, float]
    rank: int = 0
    crowding_distance: float = 0.0
    is_dominated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "params": self.params,
            "objectives": self.objectives,
            "rank": self.rank,
            "crowding_distance": self.crowding_distance,
            "is_dominated": self.is_dominated,
        }


@dataclass
class OptimizationProgress:
    """Progress update from the optimizer."""

    generation: int
    total_generations: int
    best_objectives: dict[str, float]
    pareto_front_size: int
    population_size: int


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization using NSGA-II algorithm.

    Optimizes multiple competing objectives simultaneously and returns
    the Pareto frontier of non-dominated solutions. The user then selects
    their preferred trade-off point.

    Usage:
        optimizer = MultiObjectiveOptimizer(
            objectives=[
                Objective("sharpe_ratio", maximize=True),
                Objective("max_drawdown", maximize=False),
                Objective("win_rate", maximize=True),
            ],
            param_bounds={
                "stop_loss_pct": (3.0, 20.0),
                "take_profit_pct": (10.0, 100.0),
                "min_confidence": (0.4, 0.9),
            },
        )
        pareto = await optimizer.optimize(
            initial_params={...},
            evaluate_fn=my_backtest_fn,
        )
    """

    def __init__(
        self,
        objectives: list[Objective],
        param_bounds: dict[str, tuple[float, float]],
        population_size: int = 100,
        generations: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        crossover_eta: float = 20.0,
        mutation_eta: float = 20.0,
        random_seed: int | None = None,
    ):
        """Initialize the multi-objective optimizer.

        Args:
            objectives: List of objectives to optimize.
            param_bounds: Parameter bounds {name: (min, max)}.
            population_size: Number of individuals in population.
            generations: Number of generations to evolve.
            crossover_prob: Probability of crossover.
            mutation_prob: Probability of mutation per gene.
            crossover_eta: Distribution index for SBX crossover.
            mutation_eta: Distribution index for polynomial mutation.
            random_seed: Random seed for reproducibility.
        """
        self.objectives = objectives
        self.param_bounds = param_bounds
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self._param_names = list(param_bounds.keys())
        self._n_params = len(self._param_names)
        self._n_objectives = len(objectives)

        # Progress callback
        self._progress_callback: Callable[[OptimizationProgress], None] | None = None

    def on_progress(
        self,
        callback: Callable[[OptimizationProgress], None],
    ) -> None:
        """Register a progress callback.

        Args:
            callback: Function called with OptimizationProgress each generation.
        """
        self._progress_callback = callback

    async def optimize(
        self,
        initial_params: dict[str, Any],
        evaluate_fn: Callable[[dict[str, Any]], Any],
        concurrent_evals: int = 10,
    ) -> list[ParetoSolution]:
        """
        Run NSGA-II multi-objective optimization.

        Args:
            initial_params: Starting parameter set (seed for population).
            evaluate_fn: Async function that takes params dict and returns
                         a dict of objective values.
            concurrent_evals: Max concurrent evaluations.

        Returns:
            Pareto frontier of non-dominated optimal solutions.
        """
        logger.info(
            "Starting NSGA-II optimization: %d objectives, %d params, "
            "%d population, %d generations",
            self._n_objectives,
            self._n_params,
            self.population_size,
            self.generations,
        )

        # Initialize population
        population = self._initialize_population(initial_params)

        for gen in range(self.generations):
            # Evaluate population
            evaluated = await self._evaluate_population(population, evaluate_fn, concurrent_evals)

            # Non-dominated sorting
            fronts = self._non_dominated_sort(evaluated)

            # Assign crowding distance within each front
            for front in fronts:
                self._assign_crowding_distance(front)

            # Report progress
            if self._progress_callback and fronts:
                best_objs = {}
                for obj in self.objectives:
                    vals = [s.objectives.get(obj.name, 0) for s in fronts[0]]
                    if vals:
                        best_objs[obj.name] = max(vals) if obj.maximize else min(vals)
                self._progress_callback(
                    OptimizationProgress(
                        generation=gen,
                        total_generations=self.generations,
                        best_objectives=best_objs,
                        pareto_front_size=len(fronts[0]) if fronts else 0,
                        population_size=len(evaluated),
                    )
                )

            if gen < self.generations - 1:
                # Selection
                parents = self._tournament_selection(evaluated)

                # Crossover and mutation
                offspring = self._create_offspring(parents)

                # Combine parent + offspring for next generation
                combined = evaluated + await self._evaluate_population(
                    offspring, evaluate_fn, concurrent_evals
                )

                # Non-dominated sort on combined population
                combined_fronts = self._non_dominated_sort(combined)
                for front in combined_fronts:
                    self._assign_crowding_distance(front)

                # Select next generation (elitist)
                population = self._select_next_generation(combined_fronts)

            if gen % 10 == 0:
                logger.info(
                    "NSGA-II generation %d/%d: Pareto front size = %d",
                    gen,
                    self.generations,
                    len(fronts[0]) if fronts else 0,
                )

        # Final evaluation
        final_evaluated = await self._evaluate_population(population, evaluate_fn, concurrent_evals)
        final_fronts = self._non_dominated_sort(final_evaluated)
        if final_fronts:
            for front in final_fronts:
                self._assign_crowding_distance(front)

        # Return Pareto frontier (rank 0)
        pareto_front = final_fronts[0] if final_fronts else []

        logger.info(
            "NSGA-II complete: %d solutions on Pareto frontier",
            len(pareto_front),
        )

        return pareto_front

    def _initialize_population(
        self,
        initial_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Initialize population with random individuals around seed params.

        Args:
            initial_params: Seed parameter set.

        Returns:
            List of parameter dictionaries.
        """
        population: list[dict[str, Any]] = []

        # First individual is the seed
        seed = {}
        for name in self._param_names:
            lb, ub = self.param_bounds[name]
            val = initial_params.get(name, (lb + ub) / 2)
            seed[name] = max(lb, min(ub, val))
        population.append(seed)

        # Generate random individuals
        for _ in range(self.population_size - 1):
            individual = {}
            for name in self._param_names:
                lb, ub = self.param_bounds[name]
                individual[name] = random.uniform(lb, ub)
            population.append(individual)

        return population

    async def _evaluate_population(
        self,
        population: list[dict[str, Any]],
        evaluate_fn: Callable,
        concurrent_evals: int = 10,
    ) -> list[ParetoSolution]:
        """Evaluate all individuals in the population.

        Args:
            population: List of parameter dictionaries.
            evaluate_fn: Evaluation function.
            concurrent_evals: Max concurrent evaluations.

        Returns:
            List of evaluated ParetoSolution instances.
        """
        solutions: list[ParetoSolution] = []

        # Process in batches for concurrency control
        for i in range(0, len(population), concurrent_evals):
            batch = population[i : i + concurrent_evals]

            tasks = []
            for params in batch:
                if asyncio.iscoroutinefunction(evaluate_fn):
                    tasks.append(evaluate_fn(params))
                else:
                    # Wrap sync function
                    tasks.append(
                        asyncio.get_event_loop().run_in_executor(None, evaluate_fn, params)
                    )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for params, result in zip(batch, results, strict=False):
                if isinstance(result, Exception):
                    logger.warning("Evaluation failed for params: %s", result)
                    # Use worst possible objectives
                    objectives = {}
                    for obj in self.objectives:
                        objectives[obj.name] = float("-inf") if obj.maximize else float("inf")
                else:
                    objectives = result if isinstance(result, dict) else {"value": result}

                solutions.append(
                    ParetoSolution(
                        params=dict(params),
                        objectives=objectives,
                        rank=0,
                        crowding_distance=0.0,
                    )
                )

        return solutions

    def _non_dominated_sort(
        self,
        solutions: list[ParetoSolution],
    ) -> list[list[ParetoSolution]]:
        """
        Sort solutions into Pareto fronts using non-dominated sorting.

        Front 0: Non-dominated solutions (Pareto optimal)
        Front 1: Dominated only by Front 0
        And so on.

        Args:
            solutions: List of evaluated solutions.

        Returns:
            List of fronts, each front being a list of ParetoSolution.
        """
        n = len(solutions)
        if n == 0:
            return []

        # domination_count[i] = number of solutions dominating solution i
        domination_count = [0] * n
        # dominated_by[i] = indices of solutions dominated by solution i
        dominated_by: list[list[int]] = [[] for _ in range(n)]

        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(solutions[i], solutions[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(solutions[j], solutions[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1

        # Build fronts
        fronts: list[list[ParetoSolution]] = []
        current_front_indices = [i for i in range(n) if domination_count[i] == 0]

        rank = 0
        while current_front_indices:
            front = []
            next_front_indices = []

            for i in current_front_indices:
                solutions[i].rank = rank
                solutions[i].is_dominated = rank > 0
                front.append(solutions[i])

                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front_indices.append(j)

            fronts.append(front)
            current_front_indices = next_front_indices
            rank += 1

        return fronts

    def _dominates(
        self,
        sol1: ParetoSolution,
        sol2: ParetoSolution,
    ) -> bool:
        """Check if sol1 dominates sol2.

        sol1 dominates sol2 if sol1 is at least as good as sol2 in all
        objectives, and strictly better in at least one.

        Args:
            sol1: First solution.
            sol2: Second solution.

        Returns:
            True if sol1 dominates sol2.
        """
        at_least_as_good = True
        strictly_better = False

        for obj in self.objectives:
            v1 = sol1.objectives.get(obj.name, 0)
            v2 = sol2.objectives.get(obj.name, 0)

            if obj.maximize:
                if v1 < v2:
                    at_least_as_good = False
                    break
                if v1 > v2:
                    strictly_better = True
            else:
                if v1 > v2:
                    at_least_as_good = False
                    break
                if v1 < v2:
                    strictly_better = True

        return at_least_as_good and strictly_better

    def _assign_crowding_distance(
        self,
        front: list[ParetoSolution],
    ) -> None:
        """Assign crowding distance to solutions in a front.

        Crowding distance measures how close a solution is to its neighbors
        in objective space. Higher distance = more diversity.

        Args:
            front: List of solutions in the same Pareto front.
        """
        n = len(front)
        if n <= 2:
            for sol in front:
                sol.crowding_distance = float("inf")
            return

        # Initialize distances
        for sol in front:
            sol.crowding_distance = 0.0

        # For each objective, sort and accumulate distance
        for obj in self.objectives:
            # Sort by this objective
            sorted_front = sorted(
                front,
                key=lambda s: s.objectives.get(obj.name, 0),
            )

            # Boundary solutions get infinite distance
            sorted_front[0].crowding_distance = float("inf")
            sorted_front[-1].crowding_distance = float("inf")

            # Objective range
            f_min = sorted_front[0].objectives.get(obj.name, 0)
            f_max = sorted_front[-1].objectives.get(obj.name, 0)
            f_range = f_max - f_min

            if f_range == 0:
                continue

            # Interior solutions
            for i in range(1, n - 1):
                prev_val = sorted_front[i - 1].objectives.get(obj.name, 0)
                next_val = sorted_front[i + 1].objectives.get(obj.name, 0)
                sorted_front[i].crowding_distance += (next_val - prev_val) / f_range

    def _tournament_selection(
        self,
        solutions: list[ParetoSolution],
    ) -> list[ParetoSolution]:
        """Binary tournament selection based on rank and crowding distance.

        Args:
            solutions: Evaluated solutions to select from.

        Returns:
            Selected parents.
        """
        parents: list[ParetoSolution] = []

        for _ in range(self.population_size):
            # Pick two random candidates
            i, j = random.sample(range(len(solutions)), 2)
            s1, s2 = solutions[i], solutions[j]

            # Prefer lower rank (closer to Pareto front)
            if s1.rank < s2.rank:
                winner = s1
            elif s2.rank < s1.rank:
                winner = s2
            else:
                # Same rank: prefer higher crowding distance
                winner = s1 if s1.crowding_distance >= s2.crowding_distance else s2

            parents.append(winner)

        return parents

    def _create_offspring(
        self,
        parents: list[ParetoSolution],
    ) -> list[dict[str, Any]]:
        """Create offspring population via crossover and mutation.

        Uses Simulated Binary Crossover (SBX) and Polynomial Mutation.

        Args:
            parents: Selected parent solutions.

        Returns:
            List of offspring parameter dictionaries.
        """
        offspring: list[dict[str, Any]] = []

        # Pair parents for crossover
        for i in range(0, len(parents) - 1, 2):
            p1 = parents[i]
            p2 = parents[i + 1]

            if random.random() < self.crossover_prob:
                child1, child2 = self._sbx_crossover(p1.params, p2.params)
            else:
                child1 = dict(p1.params)
                child2 = dict(p2.params)

            # Mutation
            child1 = self._polynomial_mutation(child1)
            child2 = self._polynomial_mutation(child2)

            offspring.append(child1)
            offspring.append(child2)

        return offspring[: self.population_size]

    def _sbx_crossover(
        self,
        p1: dict[str, Any],
        p2: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Simulated Binary Crossover (SBX).

        Args:
            p1: First parent parameters.
            p2: Second parent parameters.

        Returns:
            Tuple of two child parameter dictionaries.
        """
        child1 = {}
        child2 = {}
        eta = self.crossover_eta

        for name in self._param_names:
            lb, ub = self.param_bounds[name]
            v1 = p1.get(name, (lb + ub) / 2)
            v2 = p2.get(name, (lb + ub) / 2)

            if abs(v1 - v2) < 1e-14:
                child1[name] = v1
                child2[name] = v2
                continue

            u = random.random()
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

            c1 = 0.5 * ((1 + beta) * v1 + (1 - beta) * v2)
            c2 = 0.5 * ((1 - beta) * v1 + (1 + beta) * v2)

            child1[name] = max(lb, min(ub, c1))
            child2[name] = max(lb, min(ub, c2))

        return child1, child2

    def _polynomial_mutation(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Polynomial mutation.

        Args:
            params: Individual's parameters.

        Returns:
            Mutated parameters.
        """
        mutated = dict(params)
        eta = self.mutation_eta

        for name in self._param_names:
            if random.random() < self.mutation_prob:
                lb, ub = self.param_bounds[name]
                val = mutated[name]
                delta = ub - lb

                if delta < 1e-14:
                    continue

                # Normalized value
                r = random.random()
                if r < 0.5:
                    delta_q = (2.0 * r) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta_q = 1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta + 1.0))

                new_val = val + delta_q * delta
                mutated[name] = max(lb, min(ub, new_val))

        return mutated

    def _select_next_generation(
        self,
        fronts: list[list[ParetoSolution]],
    ) -> list[dict[str, Any]]:
        """Select next generation from ranked fronts (elitist strategy).

        Args:
            fronts: Pareto fronts from combined parent+offspring.

        Returns:
            Parameter dictionaries for next generation.
        """
        next_gen: list[dict[str, Any]] = []

        for front in fronts:
            if len(next_gen) + len(front) <= self.population_size:
                # Entire front fits
                next_gen.extend(s.params for s in front)
            else:
                # Need to partially fill from this front
                remaining = self.population_size - len(next_gen)
                # Sort by crowding distance (descending) for diversity
                sorted_front = sorted(
                    front,
                    key=lambda s: s.crowding_distance,
                    reverse=True,
                )
                next_gen.extend(s.params for s in sorted_front[:remaining])
                break

        return next_gen

    def select_best_compromise(
        self,
        pareto_front: list[ParetoSolution],
    ) -> ParetoSolution | None:
        """Select the best compromise solution from the Pareto front.

        Uses weighted objective normalization to find the solution
        closest to the ideal point.

        Args:
            pareto_front: Pareto frontier solutions.

        Returns:
            The best compromise solution, or None if front is empty.
        """
        if not pareto_front:
            return None

        if len(pareto_front) == 1:
            return pareto_front[0]

        # Compute ideal and nadir points
        ideal = {}
        nadir = {}
        for obj in self.objectives:
            vals = [s.objectives.get(obj.name, 0) for s in pareto_front]
            if obj.maximize:
                ideal[obj.name] = max(vals)
                nadir[obj.name] = min(vals)
            else:
                ideal[obj.name] = min(vals)
                nadir[obj.name] = max(vals)

        # Compute weighted distance to ideal for each solution
        best_sol = None
        best_dist = float("inf")

        for sol in pareto_front:
            dist = 0.0
            for obj in self.objectives:
                val = sol.objectives.get(obj.name, 0)
                range_val = abs(nadir[obj.name] - ideal[obj.name])
                if range_val < 1e-14:
                    continue

                if obj.maximize:
                    normalized = (ideal[obj.name] - val) / range_val
                else:
                    normalized = (val - ideal[obj.name]) / range_val

                dist += obj.weight * normalized**2

            if dist < best_dist:
                best_dist = dist
                best_sol = sol

        return best_sol
