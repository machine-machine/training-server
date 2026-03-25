"""
Monte Carlo Engine for strategy perturbation testing.

Implements Monte Carlo simulation with perturbations:
├── LatencyPerturbation (std=50ms)
├── FailureRatePerturbation (35-50%)
├── MEVPerturbation (10-25% sandwich prob)
└── LiquidityPerturbation (±10% std)

Outputs:
- Sharpe CI (95% confidence) with bootstrap
- Max drawdown distribution
- Risk of ruin probability
- VaR/CVaR at 95%
- Stress test scenarios
- Bootstrap confidence intervals for robustness
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    method: str = "percentile",
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: Input data array
        statistic: Function to compute statistic (e.g., np.mean, np.median)
        n_bootstrap: Number of bootstrap resamples
        alpha: Significance level (0.05 for 95% CI)
        method: CI method - "percentile", "basic", or "bca" (bias-corrected accelerated)

    Returns:
        Tuple of (lower_bound, point_estimate, upper_bound)
    """
    if len(data) == 0:
        return (0.0, 0.0, 0.0)

    point_estimate = statistic(data)

    # Bootstrap resampling
    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats[i] = statistic(sample)

    if method == "percentile":
        # Simple percentile method
        lower = np.percentile(boot_stats, alpha / 2 * 100)
        upper = np.percentile(boot_stats, (1 - alpha / 2) * 100)
    elif method == "basic":
        # Basic bootstrap (reverse percentile)
        lower = 2 * point_estimate - np.percentile(boot_stats, (1 - alpha / 2) * 100)
        upper = 2 * point_estimate - np.percentile(boot_stats, alpha / 2 * 100)
    elif method == "bca" and SCIPY_AVAILABLE:
        # Bias-corrected and accelerated
        # Compute bias correction factor
        z0 = stats.norm.ppf(np.mean(boot_stats < point_estimate))

        # Compute acceleration factor using jackknife
        n = len(data)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jackknife_stats[i] = statistic(np.delete(data, i))
        jack_mean = np.mean(jackknife_stats)

        # Guard against division by zero when all jackknife stats are equal
        jackknife_variance = np.sum((jack_mean - jackknife_stats) ** 2)
        if jackknife_variance > 1e-10:
            acceleration = np.sum((jack_mean - jackknife_stats) ** 3) / (
                6 * jackknife_variance**1.5
            )
        else:
            acceleration = 0.0

        # Adjusted percentiles
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

        p_lower = stats.norm.cdf(
            z0 + (z0 + z_alpha_lower) / (1 - acceleration * (z0 + z_alpha_lower))
        )
        p_upper = stats.norm.cdf(
            z0 + (z0 + z_alpha_upper) / (1 - acceleration * (z0 + z_alpha_upper))
        )

        lower = np.percentile(boot_stats, p_lower * 100)
        upper = np.percentile(boot_stats, p_upper * 100)
    else:
        # Fallback to percentile
        lower = np.percentile(boot_stats, alpha / 2 * 100)
        upper = np.percentile(boot_stats, (1 - alpha / 2) * 100)

    return (lower, point_estimate, upper)


def fat_tail_sample(size: int, df: float = 5.0, scale: float = 1.0) -> np.ndarray:
    """Generate samples from a fat-tailed distribution (Student-t).

    Args:
        size: Number of samples
        df: Degrees of freedom (lower = fatter tails)
        scale: Scale factor

    Returns:
        Array of samples
    """
    if SCIPY_AVAILABLE:
        return stats.t.rvs(df=df, scale=scale, size=size)
    else:
        # Fallback: use normal with occasional large shocks
        samples = np.random.normal(0, scale, size)
        shock_mask = np.random.random(size) < 0.05  # 5% chance of shock
        samples[shock_mask] *= 3.0  # Triple the magnitude for shocks
        return samples


class PerturbationType(Enum):
    """Types of perturbations for Monte Carlo."""

    LATENCY = "latency"
    FAILURE_RATE = "failure_rate"
    MEV = "mev"
    LIQUIDITY = "liquidity"
    SLIPPAGE = "slippage"
    VOLATILITY = "volatility"
    COMBINED = "combined"


@dataclass
class PerturbationConfig:
    """Configuration for a single perturbation type."""

    perturbation_type: PerturbationType
    enabled: bool = True
    mean: float = 0.0
    std: float = 1.0
    min_value: float | None = None
    max_value: float | None = None

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample perturbation values."""
        values = np.random.normal(self.mean, self.std, n)
        if self.min_value is not None:
            values = np.maximum(values, self.min_value)
        if self.max_value is not None:
            values = np.minimum(values, self.max_value)
        return values


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    n_paths: int = 1000
    random_seed: int | None = 42
    confidence_level: float = 0.95
    risk_of_ruin_threshold: float = -0.50  # -50% drawdown = ruin

    # Perturbation configs
    latency: PerturbationConfig = field(
        default_factory=lambda: PerturbationConfig(
            perturbation_type=PerturbationType.LATENCY,
            mean=0.0,
            std=50.0,  # 50ms std
            min_value=-100.0,
            max_value=200.0,
        )
    )

    failure_rate: PerturbationConfig = field(
        default_factory=lambda: PerturbationConfig(
            perturbation_type=PerturbationType.FAILURE_RATE,
            mean=0.40,  # Base 40%
            std=0.075,  # Ranges 35-50%
            min_value=0.20,
            max_value=0.70,
        )
    )

    mev: PerturbationConfig = field(
        default_factory=lambda: PerturbationConfig(
            perturbation_type=PerturbationType.MEV,
            mean=0.175,  # Sandwich prob
            std=0.075,  # 10-25% range
            min_value=0.05,
            max_value=0.35,
        )
    )

    liquidity: PerturbationConfig = field(
        default_factory=lambda: PerturbationConfig(
            perturbation_type=PerturbationType.LIQUIDITY,
            mean=0.0,
            std=0.10,  # ±10%
            min_value=-0.30,
            max_value=0.30,
        )
    )

    slippage: PerturbationConfig = field(
        default_factory=lambda: PerturbationConfig(
            perturbation_type=PerturbationType.SLIPPAGE,
            mean=0.0,
            std=0.005,  # 0.5% std
            min_value=-0.02,
            max_value=0.05,
        )
    )


@dataclass
class MonteCarloPath:
    """Results from a single Monte Carlo path."""

    path_id: int
    returns: np.ndarray
    cumulative_returns: np.ndarray
    sharpe_ratio: float
    max_drawdown: float
    final_return: float
    hit_ruin: bool
    perturbations: dict[str, float]


@dataclass
class MonteCarloResult:
    """Full Monte Carlo simulation results."""

    n_paths: int
    n_periods: int

    # Return statistics
    mean_return: float
    std_return: float
    median_return: float
    return_ci: tuple[float, float]
    return_ci_bootstrap: tuple[float, float] | None = None

    # Sharpe statistics
    mean_sharpe: float
    sharpe_ci: tuple[float, float]
    sharpe_ci_bootstrap: tuple[float, float] | None = None
    sharpe_percentiles: dict[int, float] = field(default_factory=dict)

    # Drawdown statistics
    mean_max_drawdown: float
    max_drawdown_ci: tuple[float, float]
    max_drawdown_ci_bootstrap: tuple[float, float] | None = None
    worst_drawdown: float = 0.0
    drawdown_percentiles: dict[int, float] = field(default_factory=dict)

    # Risk metrics
    var_95: float = 0.0  # Value at Risk at 95%
    cvar_95: float = 0.0  # Conditional VaR (Expected Shortfall)
    risk_of_ruin: float = 0.0  # Probability of hitting ruin threshold

    # Stress scenarios
    stress_scenarios: dict[str, float] = field(default_factory=dict)

    # Path data
    paths: list[MonteCarloPath] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_paths": self.n_paths,
            "n_periods": self.n_periods,
            "returns": {
                "mean": self.mean_return,
                "std": self.std_return,
                "median": self.median_return,
                "ci_95": self.return_ci,
                "ci_bootstrap": self.return_ci_bootstrap,
            },
            "sharpe": {
                "mean": self.mean_sharpe,
                "ci_95": self.sharpe_ci,
                "ci_bootstrap": self.sharpe_ci_bootstrap,
                "percentiles": self.sharpe_percentiles,
            },
            "drawdown": {
                "mean": self.mean_max_drawdown,
                "ci_95": self.max_drawdown_ci,
                "ci_bootstrap": self.max_drawdown_ci_bootstrap,
                "worst": self.worst_drawdown,
                "percentiles": self.drawdown_percentiles,
            },
            "risk": {
                "var_95": self.var_95,
                "cvar_95": self.cvar_95,
                "risk_of_ruin": self.risk_of_ruin,
            },
            "stress_scenarios": self.stress_scenarios,
        }


class MonteCarloEngine:
    """Monte Carlo simulation engine for strategy stress testing.

    Simulates strategy performance under various perturbations:
    - Latency: Random execution delay
    - Failure rate: Transaction failure probability
    - MEV: Sandwich attack probability
    - Liquidity: Market liquidity changes
    - Slippage: Price slippage variations

    Example:
        engine = MonteCarloEngine(config)
        result = engine.run_simulation(
            base_returns=returns,
            strategy_fn=simulate_strategy,
        )
        print(f"Sharpe 95% CI: {result.sharpe_ci}")
    """

    def __init__(self, config: MonteCarloConfig | None = None):
        self.config = config or MonteCarloConfig()

        if self.config.random_seed:
            np.random.seed(self.config.random_seed)

        self._paths: list[MonteCarloPath] = []

    def run_simulation(
        self,
        base_returns: np.ndarray,
        strategy_fn: Callable[[np.ndarray, dict[str, float]], np.ndarray] | None = None,
        n_paths: int | None = None,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation.

        Args:
            base_returns: Base return series to perturb
            strategy_fn: Optional function to compute returns under perturbations
                         Signature: (base_returns, perturbations) -> perturbed_returns
            n_paths: Number of simulation paths (default from config)

        Returns:
            MonteCarloResult with full statistics
        """
        n_paths = n_paths or self.config.n_paths
        n_periods = len(base_returns)

        logger.info(f"Running Monte Carlo with {n_paths} paths, {n_periods} periods")

        self._paths = []
        all_sharpes = []
        all_max_dds = []
        all_final_returns = []
        ruin_count = 0

        for path_id in range(n_paths):
            # Generate perturbations for this path
            perturbations = self._generate_perturbations()

            # Compute perturbed returns
            if strategy_fn:
                perturbed_returns = strategy_fn(base_returns, perturbations)
            else:
                perturbed_returns = self._apply_perturbations(base_returns, perturbations)

            # Compute path statistics
            cumulative = np.cumprod(1 + perturbed_returns)
            sharpe = self._compute_sharpe(perturbed_returns)
            max_dd = self._compute_max_drawdown(cumulative)
            final_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0

            hit_ruin = max_dd <= self.config.risk_of_ruin_threshold

            path = MonteCarloPath(
                path_id=path_id,
                returns=perturbed_returns,
                cumulative_returns=cumulative,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                final_return=final_return,
                hit_ruin=hit_ruin,
                perturbations=perturbations,
            )

            self._paths.append(path)
            all_sharpes.append(sharpe)
            all_max_dds.append(max_dd)
            all_final_returns.append(final_return)

            if hit_ruin:
                ruin_count += 1

        # Compute aggregate statistics
        all_sharpes = np.array(all_sharpes)
        all_max_dds = np.array(all_max_dds)
        all_final_returns = np.array(all_final_returns)

        # Confidence intervals
        alpha = 1 - self.config.confidence_level
        sharpe_ci = (
            np.percentile(all_sharpes, alpha / 2 * 100),
            np.percentile(all_sharpes, (1 - alpha / 2) * 100),
        )
        return_ci = (
            np.percentile(all_final_returns, alpha / 2 * 100),
            np.percentile(all_final_returns, (1 - alpha / 2) * 100),
        )
        dd_ci = (
            np.percentile(all_max_dds, alpha / 2 * 100),
            np.percentile(all_max_dds, (1 - alpha / 2) * 100),
        )

        # Risk metrics
        var_95 = np.percentile(all_final_returns, 5)  # 5th percentile for VaR
        cvar_95 = np.mean(all_final_returns[all_final_returns <= var_95])
        risk_of_ruin = ruin_count / n_paths

        # Stress scenarios
        stress_scenarios = self._compute_stress_scenarios(all_final_returns, all_max_dds)

        # Percentiles
        sharpe_pcts = {p: float(np.percentile(all_sharpes, p)) for p in [5, 25, 50, 75, 95]}
        dd_pcts = {p: float(np.percentile(all_max_dds, p)) for p in [5, 25, 50, 75, 95]}

        return MonteCarloResult(
            n_paths=n_paths,
            n_periods=n_periods,
            mean_return=float(np.mean(all_final_returns)),
            std_return=float(np.std(all_final_returns)),
            median_return=float(np.median(all_final_returns)),
            return_ci=return_ci,
            mean_sharpe=float(np.mean(all_sharpes)),
            sharpe_ci=sharpe_ci,
            sharpe_percentiles=sharpe_pcts,
            mean_max_drawdown=float(np.mean(all_max_dds)),
            max_drawdown_ci=dd_ci,
            worst_drawdown=float(np.min(all_max_dds)),
            drawdown_percentiles=dd_pcts,
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            risk_of_ruin=float(risk_of_ruin),
            stress_scenarios=stress_scenarios,
            paths=self._paths,
        )

    def _generate_perturbations(self) -> dict[str, float]:
        """Generate perturbation values for one simulation path."""
        perturbations = {}

        if self.config.latency.enabled:
            perturbations["latency_delta_ms"] = float(self.config.latency.sample()[0])

        if self.config.failure_rate.enabled:
            perturbations["failure_rate"] = float(self.config.failure_rate.sample()[0])

        if self.config.mev.enabled:
            perturbations["mev_prob"] = float(self.config.mev.sample()[0])

        if self.config.liquidity.enabled:
            perturbations["liquidity_delta"] = float(self.config.liquidity.sample()[0])

        if self.config.slippage.enabled:
            perturbations["slippage_delta"] = float(self.config.slippage.sample()[0])

        return perturbations

    def _apply_perturbations(
        self,
        base_returns: np.ndarray,
        perturbations: dict[str, float],
    ) -> np.ndarray:
        """Apply perturbations to base returns.

        Simple model:
        - Failure rate reduces returns (missed trades)
        - MEV and slippage reduce positive returns
        - Liquidity impacts both directions
        """
        perturbed = base_returns.copy()

        # Failure rate: randomly zero out some returns (missed trades)
        failure_rate = perturbations.get("failure_rate", 0.40)
        failure_mask = np.random.random(len(perturbed)) < failure_rate
        perturbed[failure_mask] = 0.0

        # MEV impact on positive returns
        mev_prob = perturbations.get("mev_prob", 0.15)
        mev_mask = (perturbed > 0) & (np.random.random(len(perturbed)) < mev_prob)
        perturbed[mev_mask] *= 1 - np.random.uniform(0.01, 0.03, np.sum(mev_mask))

        # Slippage: reduces all non-zero returns
        slippage = perturbations.get("slippage_delta", 0.0)
        perturbed[perturbed != 0] -= slippage

        # Liquidity: affects magnitude of returns
        liquidity_delta = perturbations.get("liquidity_delta", 0.0)
        perturbed *= 1 + liquidity_delta

        return perturbed

    def _compute_sharpe(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) < 1e-10:
            return 0.0

        excess = returns - risk_free / 252
        return float(np.mean(excess) / np.std(excess) * np.sqrt(252))

    def _compute_max_drawdown(self, cumulative: np.ndarray) -> float:
        """Compute maximum drawdown."""
        if len(cumulative) == 0:
            return 0.0

        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / running_max - 1
        return float(np.min(drawdowns))

    def _compute_stress_scenarios(
        self,
        final_returns: np.ndarray,
        max_drawdowns: np.ndarray,
    ) -> dict[str, float]:
        """Compute stress test scenarios."""
        scenarios = {}

        # Sort once, reuse for both 1% and 5% cutoffs
        sorted_idx = np.argsort(final_returns)

        # Worst 1% of paths
        worst_1_pct_idx = sorted_idx[: int(len(final_returns) * 0.01) + 1]
        if len(worst_1_pct_idx) > 0:
            scenarios["worst_1pct_return"] = float(np.mean(final_returns[worst_1_pct_idx]))
            scenarios["worst_1pct_drawdown"] = float(np.mean(max_drawdowns[worst_1_pct_idx]))

        # Worst 5% of paths
        worst_5_pct_idx = sorted_idx[: int(len(final_returns) * 0.05) + 1]
        if len(worst_5_pct_idx) > 0:
            scenarios["worst_5pct_return"] = float(np.mean(final_returns[worst_5_pct_idx]))
            scenarios["worst_5pct_drawdown"] = float(np.mean(max_drawdowns[worst_5_pct_idx]))

        # Absolute worst
        scenarios["worst_return"] = float(np.min(final_returns))
        scenarios["worst_drawdown"] = float(np.min(max_drawdowns))

        # Black swan (3+ std events)
        mean_ret = np.mean(final_returns)
        std_ret = np.std(final_returns)
        black_swan_mask = final_returns < (mean_ret - 3 * std_ret)
        scenarios["black_swan_probability"] = float(np.mean(black_swan_mask))

        return scenarios

    def sensitivity_analysis(
        self,
        base_returns: np.ndarray,
        perturbation_type: PerturbationType,
        values: list[float],
    ) -> dict[float, dict[str, float]]:
        """Run sensitivity analysis for a single perturbation type.

        Args:
            base_returns: Base return series
            perturbation_type: Which perturbation to vary
            values: Values to test

        Returns:
            Dict mapping value -> {sharpe, max_dd, var_95}
        """
        results = {}

        for value in values:
            # Create custom perturbation
            perturbations = self._generate_perturbations()

            if perturbation_type == PerturbationType.FAILURE_RATE:
                perturbations["failure_rate"] = value
            elif perturbation_type == PerturbationType.MEV:
                perturbations["mev_prob"] = value
            elif perturbation_type == PerturbationType.SLIPPAGE:
                perturbations["slippage_delta"] = value
            elif perturbation_type == PerturbationType.LIQUIDITY:
                perturbations["liquidity_delta"] = value

            # Run mini simulation
            sharpes = []
            max_dds = []
            final_rets = []

            for _ in range(100):  # 100 paths per value
                perturbed = self._apply_perturbations(base_returns, perturbations)
                cumulative = np.cumprod(1 + perturbed)

                sharpes.append(self._compute_sharpe(perturbed))
                max_dds.append(self._compute_max_drawdown(cumulative))
                final_rets.append(cumulative[-1] - 1 if len(cumulative) > 0 else 0)

            results[value] = {
                "mean_sharpe": float(np.mean(sharpes)),
                "mean_max_dd": float(np.mean(max_dds)),
                "var_95": float(np.percentile(final_rets, 5)),
            }

        return results

    def get_path_details(self, path_id: int) -> MonteCarloPath | None:
        """Get details for a specific simulation path."""
        if 0 <= path_id < len(self._paths):
            return self._paths[path_id]
        return None

    def get_worst_paths(self, n: int = 10) -> list[MonteCarloPath]:
        """Get the N worst performing paths."""
        sorted_paths = sorted(self._paths, key=lambda p: p.final_return)
        return sorted_paths[:n]

    def get_best_paths(self, n: int = 10) -> list[MonteCarloPath]:
        """Get the N best performing paths."""
        sorted_paths = sorted(self._paths, key=lambda p: -p.final_return)
        return sorted_paths[:n]


def run_quick_monte_carlo(
    returns: np.ndarray,
    n_paths: int = 500,
) -> dict[str, float]:
    """Run quick Monte Carlo for summary statistics.

    Args:
        returns: Return series
        n_paths: Number of paths

    Returns:
        Dict with key metrics
    """
    config = MonteCarloConfig(n_paths=n_paths)
    engine = MonteCarloEngine(config)
    result = engine.run_simulation(returns)

    return {
        "sharpe_mean": result.mean_sharpe,
        "sharpe_ci_low": result.sharpe_ci[0],
        "sharpe_ci_high": result.sharpe_ci[1],
        "max_dd_mean": result.mean_max_drawdown,
        "var_95": result.var_95,
        "cvar_95": result.cvar_95,
        "risk_of_ruin": result.risk_of_ruin,
    }
