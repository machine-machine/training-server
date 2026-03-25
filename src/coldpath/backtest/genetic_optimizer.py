"""
Genetic Algorithm Optimizer - Evolutionary parameter search for backtesting.

Uses genetic algorithms to efficiently explore large parameter spaces:
- Population-based search with crossover and mutation
- Elitism to preserve best solutions
- Adaptive mutation rates
- Tournament selection for parent selection

This can find optimal parameters that gradient-based methods might miss,
especially in non-convex, multi-modal optimization landscapes.

Usage:
    from coldpath.backtest.genetic_optimizer import GeneticOptimizer

    optimizer = GeneticOptimizer(
        backtest_runner=my_backtest_fn,
        population_size=50,
        generations=100,
    )

    result = await optimizer.optimize(
        param_bounds={
            "stop_loss_pct": (3.0, 15.0),
            "take_profit_pct": (8.0, 60.0),
            "max_position_sol": (0.01, 0.3),
        },
    )
"""

import asyncio
import copy
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """A single individual in the population."""

    genes: dict[str, float]
    fitness: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    age: int = 0

    def copy(self) -> "Individual":
        return Individual(
            genes=copy.deepcopy(self.genes),
            fitness=self.fitness,
            metrics=copy.deepcopy(self.metrics),
            age=self.age,
        )


@dataclass
class GAResult:
    """Result from genetic algorithm optimization."""

    best_individual: Individual
    population: list[Individual]
    generation: int
    total_evaluations: int
    time_seconds: float
    convergence_history: list[float]
    diversity_history: list[float]
    pareto_front: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_individual.genes,
            "best_fitness": self.best_individual.fitness,
            "best_metrics": self.best_individual.metrics,
            "generation": self.generation,
            "total_evaluations": self.total_evaluations,
            "time_seconds": self.time_seconds,
            "convergence_history": self.convergence_history,
            "diversity_history": self.diversity_history,
            "pareto_front_size": len(self.pareto_front),
        }


@dataclass
class GAConfig:
    """Configuration for genetic algorithm."""

    # Population
    population_size: int = 50
    elite_size: int = 5  # Number of best individuals to preserve

    # Selection
    tournament_size: int = 3
    selection_pressure: float = 0.7  # Probability of selecting fitter in tournament

    # Crossover
    crossover_rate: float = 0.8
    crossover_type: str = "uniform"  # uniform, single_point, two_point, blend

    # Mutation
    mutation_rate: float = 0.1
    mutation_type: str = "gaussian"  # gaussian, uniform, adaptive
    mutation_strength: float = 0.2  # Relative to parameter range

    # Adaptive
    adaptive_mutation: bool = True
    min_mutation_rate: float = 0.01
    max_mutation_rate: float = 0.5
    stagnation_generations: int = 10  # Generations without improvement before adapting

    # Termination
    max_generations: int = 100
    max_time_minutes: float = 30.0
    target_fitness: float | None = None
    convergence_threshold: float = 0.001  # Stop if improvement < this for N generations
    convergence_generations: int = 20

    # Parallelization
    parallel_evaluations: int = 10

    # Diversity preservation
    crowding_distance: bool = True
    fitness_sharing: bool = False
    sharing_radius: float = 0.1


class GeneticOptimizer:
    """Genetic algorithm for parameter optimization.

    Features:
    - Multiple crossover operators (uniform, blend, SBX)
    - Adaptive mutation rates
    - Tournament selection
    - Elitism
    - Diversity preservation
    - Parallel fitness evaluation

    Example:
        optimizer = GeneticOptimizer(
            backtest_runner=my_backtest_fn,
            config=GAConfig(population_size=100, max_generations=200),
        )

        result = await optimizer.optimize(
            param_bounds={
                "stop_loss_pct": (3.0, 15.0),
                "take_profit_pct": (8.0, 50.0),
            },
            fixed_params={"initial_capital_sol": 1.0},
        )
    """

    def __init__(
        self,
        backtest_runner: Callable,
        config: GAConfig | None = None,
        fitness_function: Callable | None = None,
    ):
        """Initialize the genetic optimizer.

        Args:
            backtest_runner: Async function to run backtests
            config: GA configuration
            fitness_function: Optional custom fitness function (default: Sharpe-based)
        """
        self.backtest_runner = backtest_runner
        self.config = config or GAConfig()
        self.fitness_function = fitness_function or self._default_fitness

        # State
        self._population: list[Individual] = []
        self._generation = 0
        self._evaluations = 0
        self._best_ever: Individual | None = None
        self._convergence_history: list[float] = []
        self._diversity_history: list[float] = []
        self._stagnation_count = 0
        self._current_mutation_rate = self.config.mutation_rate

        # Param bounds (set during optimize)
        self._param_bounds: dict[str, tuple[float, float]] = {}
        self._fixed_params: dict[str, Any] = {}

        # Random state
        self._rng = random.Random(42)

    def _default_fitness(self, metrics: dict[str, float]) -> float:
        """Default fitness function based on Sharpe and risk metrics."""
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate_pct", 0) / 100
        max_dd = metrics.get("max_drawdown_pct", 100)
        profit_factor = metrics.get("profit_factor", 0)

        # Normalize components
        sharpe_norm = min(1.0, max(0, sharpe / 3.0))
        win_norm = min(1.0, max(0, win_rate))
        dd_norm = max(0, 1 - max_dd / 50)
        pf_norm = min(1.0, max(0, profit_factor / 3.0))

        # Weighted combination
        fitness = sharpe_norm * 0.35 + win_norm * 0.20 + dd_norm * 0.25 + pf_norm * 0.20

        return fitness

    def _initialize_population(self) -> list[Individual]:
        """Initialize random population within bounds."""
        population = []

        for _ in range(self.config.population_size):
            genes = {}
            for param, (low, high) in self._param_bounds.items():
                genes[param] = self._rng.uniform(low, high)
            population.append(Individual(genes=genes))

        # Add some individuals near known good configurations
        if self._best_ever:
            for i in range(min(5, self.config.population_size // 10)):
                genes = {}
                for param, (low, high) in self._param_bounds.items():
                    base = self._best_ever.genes.get(param, (low + high) / 2)
                    noise = self._rng.gauss(0, (high - low) * 0.1)
                    genes[param] = max(low, min(high, base + noise))
                if i < len(population):
                    population[i] = Individual(genes=genes)

        return population

    async def _evaluate_population(self, population: list[Individual]):
        """Evaluate fitness for all individuals in population."""
        # Build tasks
        tasks = []
        for individual in population:
            params = {**self._fixed_params, **individual.genes}
            tasks.append(self.backtest_runner(params))

        # Run in batches for parallelization
        batch_size = self.config.parallel_evaluations
        results = []

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)

        # Update individuals
        for individual, result in zip(population, results, strict=False):
            self._evaluations += 1

            if isinstance(result, Exception):
                individual.fitness = 0.0
                individual.metrics = {}
            else:
                individual.metrics = result if isinstance(result, dict) else {}
                individual.fitness = self.fitness_function(individual.metrics)

    def _select_parent(self, population: list[Individual]) -> Individual:
        """Select parent using tournament selection."""
        tournament = self._rng.sample(population, min(self.config.tournament_size, len(population)))

        # Sort by fitness (descending)
        tournament.sort(key=lambda x: x.fitness, reverse=True)

        # Probabilistic selection based on pressure
        if self._rng.random() < self.config.selection_pressure:
            return tournament[0].copy()
        else:
            return self._rng.choice(tournament).copy()

    def _crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if self._rng.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1_genes = {}
        child2_genes = {}

        if self.config.crossover_type == "uniform":
            for param in self._param_bounds:
                if self._rng.random() < 0.5:
                    child1_genes[param] = parent1.genes[param]
                    child2_genes[param] = parent2.genes[param]
                else:
                    child1_genes[param] = parent2.genes[param]
                    child2_genes[param] = parent1.genes[param]

        elif self.config.crossover_type == "single_point":
            params = list(self._param_bounds.keys())
            point = self._rng.randint(1, len(params) - 1)
            for i, param in enumerate(params):
                if i < point:
                    child1_genes[param] = parent1.genes[param]
                    child2_genes[param] = parent2.genes[param]
                else:
                    child1_genes[param] = parent2.genes[param]
                    child2_genes[param] = parent1.genes[param]

        elif self.config.crossover_type == "blend":
            # Blend crossover (BLX-alpha)
            alpha = 0.5
            for param, (low, high) in self._param_bounds.items():
                p1_val = parent1.genes[param]
                p2_val = parent2.genes[param]
                min_val = min(p1_val, p2_val)
                max_val = max(p1_val, p2_val)
                range_val = max_val - min_val

                child1_genes[param] = self._rng.uniform(
                    max(low, min_val - alpha * range_val), min(high, max_val + alpha * range_val)
                )
                child2_genes[param] = self._rng.uniform(
                    max(low, min_val - alpha * range_val), min(high, max_val + alpha * range_val)
                )

        else:  # Default to uniform
            for param in self._param_bounds:
                if self._rng.random() < 0.5:
                    child1_genes[param] = parent1.genes[param]
                    child2_genes[param] = parent2.genes[param]
                else:
                    child1_genes[param] = parent2.genes[param]
                    child2_genes[param] = parent1.genes[param]

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def _mutate(self, individual: Individual) -> Individual:
        """Mutate an individual."""
        mutated = individual.copy()

        for param, (low, high) in self._param_bounds.items():
            if self._rng.random() < self._current_mutation_rate:
                if self.config.mutation_type == "gaussian":
                    # Gaussian mutation
                    current = mutated.genes[param]
                    sigma = (high - low) * self.config.mutation_strength
                    new_val = self._rng.gauss(current, sigma)
                    mutated.genes[param] = max(low, min(high, new_val))

                elif self.config.mutation_type == "uniform":
                    # Uniform random mutation
                    mutated.genes[param] = self._rng.uniform(low, high)

                elif self.config.mutation_type == "adaptive":
                    # Adaptive mutation based on fitness
                    strength = self.config.mutation_strength * (1 - mutated.fitness)
                    current = mutated.genes[param]
                    sigma = (high - low) * strength
                    new_val = self._rng.gauss(current, sigma)
                    mutated.genes[param] = max(low, min(high, new_val))

        return mutated

    def _calculate_diversity(self, population: list[Individual]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0

        total_distance = 0.0
        count = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = 0.0
                for param, (low, high) in self._param_bounds.items():
                    v1 = population[i].genes[param]
                    v2 = population[j].genes[param]
                    # Normalized distance
                    distance += abs(v1 - v2) / (high - low)
                total_distance += distance / len(self._param_bounds)
                count += 1

        return total_distance / count if count > 0 else 0.0

    def _adapt_mutation_rate(self, improvement: float):
        """Adapt mutation rate based on improvement."""
        if not self.config.adaptive_mutation:
            return

        if improvement < self.config.convergence_threshold:
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0

        # Increase mutation if stagnating
        if self._stagnation_count >= self.config.stagnation_generations:
            self._current_mutation_rate = min(
                self.config.max_mutation_rate, self._current_mutation_rate * 1.5
            )
            self._stagnation_count = 0
            logger.debug(f"Increased mutation rate to {self._current_mutation_rate:.3f}")
        else:
            # Gradually decrease mutation rate
            self._current_mutation_rate = max(
                self.config.min_mutation_rate, self._current_mutation_rate * 0.99
            )

    async def optimize(
        self,
        param_bounds: dict[str, tuple[float, float]],
        fixed_params: dict[str, Any] | None = None,
        initial_population: list[dict[str, float]] | None = None,
    ) -> GAResult:
        """Run genetic algorithm optimization.

        Args:
            param_bounds: Dict mapping parameter name to (min, max) bounds
            fixed_params: Parameters that don't change
            initial_population: Optional starting population

        Returns:
            GAResult with best individual and optimization history
        """
        start_time = datetime.utcnow()

        self._param_bounds = param_bounds
        self._fixed_params = fixed_params or {}
        self._generation = 0
        self._evaluations = 0
        self._convergence_history = []
        self._diversity_history = []
        self._current_mutation_rate = self.config.mutation_rate

        logger.info(f"Starting GA optimization with {self.config.population_size} individuals")

        # Initialize population
        self._population = self._initialize_population()

        # Override with initial population if provided
        if initial_population:
            for i, ind_genes in enumerate(initial_population[: len(self._population)]):
                for param, value in ind_genes.items():
                    if param in self._param_bounds:
                        low, high = self._param_bounds[param]
                        self._population[i].genes[param] = max(low, min(high, value))

        # Evaluate initial population
        await self._evaluate_population(self._population)
        self._population.sort(key=lambda x: x.fitness, reverse=True)
        self._best_ever = self._population[0].copy()

        self._convergence_history.append(self._best_ever.fitness)
        self._diversity_history.append(self._calculate_diversity(self._population))

        logger.info(f"Generation 0: best_fitness={self._best_ever.fitness:.4f}")

        # Main evolution loop
        max_time_seconds = self.config.max_time_minutes * 60
        generations_without_improvement = 0

        while self._generation < self.config.max_generations:
            self._generation += 1

            # Check time limit
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > max_time_seconds:
                logger.info(f"Time limit reached after {self._generation} generations")
                break

            # Create new generation
            new_population = []

            # Elitism: preserve best individuals
            elites = [ind.copy() for ind in self._population[: self.config.elite_size]]
            new_population.extend(elites)

            # Generate rest of population
            while len(new_population) < self.config.population_size:
                # Selection
                parent1 = self._select_parent(self._population)
                parent2 = self._select_parent(self._population)

                # Crossover
                child1, child2 = self._crossover(parent1, parent2)

                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.config.population_size:
                    new_population.append(child2)

            # Evaluate new population
            await self._evaluate_population(new_population)

            # Replace population
            self._population = new_population
            self._population.sort(key=lambda x: x.fitness, reverse=True)

            # Update best ever
            if self._population[0].fitness > self._best_ever.fitness:
                self._best_ever = self._population[0].copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Record history
            self._convergence_history.append(self._best_ever.fitness)
            self._diversity_history.append(self._calculate_diversity(self._population))

            # Calculate improvement
            if len(self._convergence_history) >= 2:
                improvement = self._convergence_history[-1] - self._convergence_history[-2]
            else:
                improvement = 0

            # Adapt mutation rate
            self._adapt_mutation_rate(improvement)

            # Logging
            if self._generation % 10 == 0 or self._generation < 10:
                logger.info(
                    f"Generation {self._generation}: "
                    f"best={self._best_ever.fitness:.4f}, "
                    f"avg={sum(i.fitness for i in self._population) / len(self._population):.4f}, "
                    f"diversity={self._diversity_history[-1]:.3f}"
                )

            # Check convergence
            if self.config.target_fitness and self._best_ever.fitness >= self.config.target_fitness:
                logger.info(f"Target fitness reached: {self._best_ever.fitness:.4f}")
                break

            if generations_without_improvement >= self.config.convergence_generations:
                logger.info(f"Converged after {self._generation} generations without improvement")
                break

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        # Build Pareto front (simplified - top individuals by different metrics)
        pareto_front = self._build_pareto_front()

        logger.info(
            f"GA optimization complete: "
            f"generations={self._generation}, "
            f"evaluations={self._evaluations}, "
            f"best_fitness={self._best_ever.fitness:.4f}, "
            f"time={elapsed:.1f}s"
        )

        return GAResult(
            best_individual=self._best_ever,
            population=self._population,
            generation=self._generation,
            total_evaluations=self._evaluations,
            time_seconds=elapsed,
            convergence_history=self._convergence_history,
            diversity_history=self._diversity_history,
            pareto_front=pareto_front,
        )

    def _build_pareto_front(self) -> list[dict[str, Any]]:
        """Build simplified Pareto front of diverse high-performers."""
        front = []

        # Get unique top performers by different metrics
        metrics_to_consider = ["sharpe_ratio", "win_rate_pct", "max_drawdown_pct", "profit_factor"]

        for metric in metrics_to_consider:
            sorted_pop = sorted(
                self._population,
                key=lambda x: x.metrics.get(metric, 0),
                reverse=(metric != "max_drawdown_pct"),  # Lower is better for drawdown
            )
            if sorted_pop:
                best = sorted_pop[0]
                front.append(
                    {
                        "params": best.genes,
                        "fitness": best.fitness,
                        "metric": metric,
                        "value": best.metrics.get(metric, 0),
                    }
                )

        # Remove duplicates
        seen = set()
        unique_front = []
        for item in front:
            key = tuple(sorted(item["params"].items()))
            if key not in seen:
                seen.add(key)
                unique_front.append(item)

        return unique_front


# Convenience function
async def genetic_optimize(
    backtest_runner: Callable,
    param_bounds: dict[str, tuple[float, float]],
    fixed_params: dict[str, Any] | None = None,
    population_size: int = 50,
    generations: int = 100,
    **kwargs,
) -> GAResult:
    """Run genetic algorithm optimization with sensible defaults.

    Args:
        backtest_runner: Async backtest function
        param_bounds: Parameter bounds as {name: (min, max)}
        fixed_params: Fixed parameters
        population_size: Population size
        generations: Max generations
        **kwargs: Additional GAConfig parameters

    Returns:
        GAResult with optimization results
    """
    config = GAConfig(population_size=population_size, max_generations=generations, **kwargs)

    optimizer = GeneticOptimizer(
        backtest_runner=backtest_runner,
        config=config,
    )

    return await optimizer.optimize(
        param_bounds=param_bounds,
        fixed_params=fixed_params,
    )
