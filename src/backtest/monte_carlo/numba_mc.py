"""
Numba-accelerated Monte Carlo simulation engine.

Provides high-performance simulation for:
- VaR/CVaR risk metrics
- Sharpe ratio confidence intervals
- Probability of ruin calculation
- Strategy robustness testing
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    n_simulations: int = 10000          # Number of simulations
    initial_capital: float = 10.0       # Starting capital (SOL)
    confidence_levels: List[float] = field(default_factory=lambda: [0.05, 0.50, 0.95])

    # Perturbation parameters
    slippage_std_bps: float = 50.0
    latency_mean_ms: float = 50.0
    failure_rate_range: Tuple[float, float] = (0.30, 0.50)
    mev_alpha: float = 2.0
    mev_beta: float = 5.0
    liquidity_std_pct: float = 15.0

    # Risk parameters
    var_confidence: float = 0.05         # VaR confidence (5% = 95% VaR)
    ruin_threshold_pct: float = 50.0     # Drawdown for "ruin"

    # Seed for reproducibility
    seed: Optional[int] = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "initial_capital": self.initial_capital,
            "confidence_levels": self.confidence_levels,
            "var_confidence": self.var_confidence,
            "ruin_threshold_pct": self.ruin_threshold_pct,
        }


@dataclass
class RiskMetrics:
    """Risk metrics from Monte Carlo simulation."""
    # Value at Risk
    var_5: float = 0.0                  # 5% VaR (95% confidence)
    var_1: float = 0.0                  # 1% VaR (99% confidence)

    # Conditional VaR (Expected Shortfall)
    cvar_5: float = 0.0
    cvar_1: float = 0.0

    # Maximum Drawdown stats
    max_drawdown_mean: float = 0.0
    max_drawdown_5th: float = 0.0
    max_drawdown_95th: float = 0.0

    # Probability of ruin
    prob_ruin: float = 0.0
    avg_ruin_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "var": {"5pct": self.var_5, "1pct": self.var_1},
            "cvar": {"5pct": self.cvar_5, "1pct": self.cvar_1},
            "max_drawdown": {
                "mean": self.max_drawdown_mean,
                "5th": self.max_drawdown_5th,
                "95th": self.max_drawdown_95th,
            },
            "ruin": {"probability": self.prob_ruin, "avg_time": self.avg_ruin_time},
        }


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo simulation result."""
    # Return distribution
    mean_return_pct: float = 0.0
    std_return_pct: float = 0.0
    median_return_pct: float = 0.0

    # Percentiles
    percentile_5: float = 0.0
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    percentile_95: float = 0.0

    # Sharpe ratio distribution
    sharpe_mean: float = 0.0
    sharpe_std: float = 0.0
    sharpe_5th: float = 0.0
    sharpe_median: float = 0.0
    sharpe_95th: float = 0.0

    # Trade statistics
    avg_trades: float = 0.0
    avg_win_rate: float = 0.0
    probability_of_profit: float = 0.0

    # Risk metrics
    risk: RiskMetrics = field(default_factory=RiskMetrics)

    # Simulation details
    n_simulations: int = 0
    final_equities: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "returns": {
                "mean": self.mean_return_pct,
                "std": self.std_return_pct,
                "median": self.median_return_pct,
                "percentile_5": self.percentile_5,
                "percentile_25": self.percentile_25,
                "percentile_75": self.percentile_75,
                "percentile_95": self.percentile_95,
            },
            "sharpe": {
                "mean": self.sharpe_mean,
                "std": self.sharpe_std,
                "5th": self.sharpe_5th,
                "median": self.sharpe_median,
                "95th": self.sharpe_95th,
            },
            "trades": {
                "avg_count": self.avg_trades,
                "avg_win_rate": self.avg_win_rate,
            },
            "probability_of_profit": self.probability_of_profit,
            "risk": self.risk.to_dict(),
            "n_simulations": self.n_simulations,
        }


# =============================================================================
# Numba-Accelerated Core Functions
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _simulate_single_path(
        prices: np.ndarray,
        entry_signals: np.ndarray,
        initial_capital: float,
        position_size_pct: float,
        take_profit_pct: float,
        stop_loss_pct: float,
        max_hold: int,
        slippage_std_bps: float,
        inclusion_prob: float,
        rng_seed: int,
    ) -> Tuple[float, int, int, float]:
        """Simulate a single Monte Carlo path with perturbations.

        Returns:
            (final_equity, n_trades, n_wins, max_drawdown_pct)
        """
        np.random.seed(rng_seed)

        n_tokens, n_timepoints = prices.shape
        capital = initial_capital
        peak_capital = capital
        max_drawdown = 0.0
        n_trades = 0
        n_wins = 0

        for i in range(n_tokens):
            t = 0
            while t < n_timepoints:
                if entry_signals[i, t]:
                    # Random inclusion check
                    if np.random.random() > inclusion_prob:
                        t += 1
                        continue

                    entry_price = prices[i, t]
                    if entry_price <= 0:
                        t += 1
                        continue

                    # Add random slippage
                    slippage_bps = abs(np.random.normal(0, slippage_std_bps))
                    entry_price *= (1 + slippage_bps / 10000)

                    # Position size
                    position_value = capital * position_size_pct
                    if position_value < 0.001:
                        t += 1
                        continue

                    tokens = position_value / entry_price
                    capital -= position_value

                    # Find exit
                    exit_t = t + 1
                    exit_price = entry_price
                    while exit_t < min(t + max_hold, n_timepoints):
                        if np.random.random() > inclusion_prob:
                            exit_t += 1
                            continue

                        current_price = prices[i, exit_t]
                        if current_price <= 0:
                            exit_t += 1
                            continue

                        pnl_pct = (current_price - entry_price) / entry_price * 100

                        if pnl_pct >= take_profit_pct or pnl_pct <= -stop_loss_pct:
                            exit_price = current_price
                            break
                        exit_t += 1

                    # Apply exit slippage
                    exit_slippage = abs(np.random.normal(0, slippage_std_bps))
                    exit_price *= (1 - exit_slippage / 10000)

                    # Close position
                    proceeds = tokens * exit_price
                    capital += proceeds

                    # Track results
                    n_trades += 1
                    if proceeds > position_value:
                        n_wins += 1

                    # Update peak and drawdown
                    if capital > peak_capital:
                        peak_capital = capital
                    drawdown = (peak_capital - capital) / peak_capital * 100
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

                    t = exit_t + 1
                else:
                    t += 1

        return capital, n_trades, n_wins, max_drawdown

    @njit(parallel=True, cache=True)
    def _run_simulations_parallel(
        prices: np.ndarray,
        entry_signals: np.ndarray,
        n_simulations: int,
        initial_capital: float,
        position_size_pct: float,
        take_profit_pct: float,
        stop_loss_pct: float,
        max_hold: int,
        slippage_std_bps: float,
        inclusion_prob_min: float,
        inclusion_prob_max: float,
        base_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run Monte Carlo simulations in parallel.

        Returns:
            (final_equities, trade_counts, win_counts, max_drawdowns)
        """
        final_equities = np.empty(n_simulations, dtype=np.float64)
        trade_counts = np.empty(n_simulations, dtype=np.int64)
        win_counts = np.empty(n_simulations, dtype=np.int64)
        max_drawdowns = np.empty(n_simulations, dtype=np.float64)

        for sim in prange(n_simulations):
            # Random inclusion probability for this simulation
            inclusion_prob = inclusion_prob_min + np.random.random() * (
                inclusion_prob_max - inclusion_prob_min
            )

            equity, trades, wins, max_dd = _simulate_single_path(
                prices,
                entry_signals,
                initial_capital,
                position_size_pct,
                take_profit_pct,
                stop_loss_pct,
                max_hold,
                slippage_std_bps,
                inclusion_prob,
                base_seed + sim,
            )

            final_equities[sim] = equity
            trade_counts[sim] = trades
            win_counts[sim] = wins
            max_drawdowns[sim] = max_dd

        return final_equities, trade_counts, win_counts, max_drawdowns

    @njit(cache=True)
    def _compute_var_cvar_core(
        returns: np.ndarray,
        confidence: float,
    ) -> Tuple[float, float]:
        """Compute VaR and CVaR (Expected Shortfall)."""
        sorted_returns = np.sort(returns)
        n = len(returns)

        # VaR: percentile of losses
        var_idx = int(n * confidence)
        var = sorted_returns[var_idx]

        # CVaR: mean of returns below VaR
        tail_returns = sorted_returns[:var_idx + 1]
        cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var

        return var, cvar

    @njit(cache=True)
    def _compute_sharpe_ratios(
        final_equities: np.ndarray,
        initial_capital: float,
        risk_free_rate: float = 0.0,
    ) -> np.ndarray:
        """Compute Sharpe ratios for each simulation."""
        n = len(final_equities)
        sharpes = np.empty(n, dtype=np.float64)

        for i in range(n):
            ret = (final_equities[i] - initial_capital) / initial_capital
            # Simplified Sharpe (would need return series for proper std)
            # Using return magnitude as proxy for volatility
            vol = abs(ret) * 0.5 + 0.1  # Prevent division by zero
            sharpes[i] = (ret - risk_free_rate) / vol

        return sharpes

else:
    # Pure Python fallbacks
    def _simulate_single_path(prices, entry_signals, initial_capital, position_size_pct,
                              take_profit_pct, stop_loss_pct, max_hold, slippage_std_bps,
                              inclusion_prob, rng_seed):
        np.random.seed(rng_seed)
        n_tokens, n_timepoints = prices.shape
        capital = initial_capital
        peak_capital = capital
        max_drawdown = 0.0
        n_trades = 0
        n_wins = 0

        for i in range(n_tokens):
            t = 0
            while t < n_timepoints:
                if entry_signals[i, t]:
                    if np.random.random() > inclusion_prob:
                        t += 1
                        continue
                    entry_price = prices[i, t]
                    if entry_price <= 0:
                        t += 1
                        continue
                    slippage_bps = abs(np.random.normal(0, slippage_std_bps))
                    entry_price *= (1 + slippage_bps / 10000)
                    position_value = capital * position_size_pct
                    if position_value < 0.001:
                        t += 1
                        continue
                    tokens = position_value / entry_price
                    capital -= position_value
                    exit_t = t + 1
                    exit_price = entry_price
                    while exit_t < min(t + max_hold, n_timepoints):
                        current_price = prices[i, exit_t]
                        if current_price > 0:
                            pnl_pct = (current_price - entry_price) / entry_price * 100
                            if pnl_pct >= take_profit_pct or pnl_pct <= -stop_loss_pct:
                                exit_price = current_price
                                break
                        exit_t += 1
                    exit_slippage = abs(np.random.normal(0, slippage_std_bps))
                    exit_price *= (1 - exit_slippage / 10000)
                    proceeds = tokens * exit_price
                    capital += proceeds
                    n_trades += 1
                    if proceeds > position_value:
                        n_wins += 1
                    if capital > peak_capital:
                        peak_capital = capital
                    drawdown = (peak_capital - capital) / peak_capital * 100
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    t = exit_t + 1
                else:
                    t += 1
        return capital, n_trades, n_wins, max_drawdown

    def _run_simulations_parallel(prices, entry_signals, n_simulations, initial_capital,
                                  position_size_pct, take_profit_pct, stop_loss_pct, max_hold,
                                  slippage_std_bps, inclusion_prob_min, inclusion_prob_max, base_seed):
        final_equities = np.empty(n_simulations)
        trade_counts = np.empty(n_simulations, dtype=np.int64)
        win_counts = np.empty(n_simulations, dtype=np.int64)
        max_drawdowns = np.empty(n_simulations)
        for sim in range(n_simulations):
            inclusion_prob = np.random.uniform(inclusion_prob_min, inclusion_prob_max)
            equity, trades, wins, max_dd = _simulate_single_path(
                prices, entry_signals, initial_capital, position_size_pct,
                take_profit_pct, stop_loss_pct, max_hold, slippage_std_bps,
                inclusion_prob, base_seed + sim
            )
            final_equities[sim] = equity
            trade_counts[sim] = trades
            win_counts[sim] = wins
            max_drawdowns[sim] = max_dd
        return final_equities, trade_counts, win_counts, max_drawdowns

    def _compute_var_cvar_core(returns, confidence):
        sorted_returns = np.sort(returns)
        n = len(returns)
        var_idx = int(n * confidence)
        var = sorted_returns[var_idx]
        cvar = np.mean(sorted_returns[:var_idx + 1])
        return var, cvar

    def _compute_sharpe_ratios(final_equities, initial_capital, risk_free_rate=0.0):
        returns = (final_equities - initial_capital) / initial_capital
        vols = np.abs(returns) * 0.5 + 0.1
        return (returns - risk_free_rate) / vols


# =============================================================================
# Public API Functions
# =============================================================================

def compute_var_cvar(
    returns: np.ndarray,
    confidence: float = 0.05,
) -> Tuple[float, float]:
    """Compute Value at Risk and Conditional VaR.

    Args:
        returns: Array of returns (as percentages or decimals)
        confidence: VaR confidence level (0.05 = 95% VaR)

    Returns:
        (VaR, CVaR) tuple
    """
    return _compute_var_cvar_core(returns, confidence)


def compute_probability_of_ruin(
    max_drawdowns: np.ndarray,
    ruin_threshold_pct: float = 50.0,
) -> Tuple[float, Optional[float]]:
    """Compute probability of ruin from drawdown distribution.

    Args:
        max_drawdowns: Array of max drawdown percentages
        ruin_threshold_pct: Drawdown percentage considered "ruin"

    Returns:
        (probability_of_ruin, avg_time_to_ruin)
    """
    ruin_count = np.sum(max_drawdowns >= ruin_threshold_pct)
    prob_ruin = ruin_count / len(max_drawdowns)

    return prob_ruin, None  # Time to ruin requires trade-level data


def compute_sharpe_confidence_interval(
    final_equities: np.ndarray,
    initial_capital: float,
    confidence_levels: List[float] = [0.05, 0.50, 0.95],
) -> Dict[str, float]:
    """Compute Sharpe ratio confidence intervals.

    Args:
        final_equities: Array of final portfolio values
        initial_capital: Starting capital
        confidence_levels: Percentiles to compute

    Returns:
        Dict with sharpe statistics
    """
    sharpes = _compute_sharpe_ratios(final_equities, initial_capital)

    return {
        "mean": float(np.mean(sharpes)),
        "std": float(np.std(sharpes)),
        "5th": float(np.percentile(sharpes, 5)),
        "50th": float(np.percentile(sharpes, 50)),
        "95th": float(np.percentile(sharpes, 95)),
    }


def run_monte_carlo_simulation(
    prices: np.ndarray,
    entry_signals: np.ndarray,
    config: Optional[MonteCarloConfig] = None,
    signal_config: Optional[Any] = None,
) -> MonteCarloResult:
    """Run full Monte Carlo simulation.

    Args:
        prices: Price matrix (n_tokens, n_timepoints)
        entry_signals: Entry signal matrix (same shape as prices)
        config: Monte Carlo configuration
        signal_config: Signal/trading configuration with TP/SL etc.

    Returns:
        MonteCarloResult with full statistics
    """
    cfg = config or MonteCarloConfig()

    # Extract trading params from signal_config or use defaults
    if signal_config:
        take_profit_pct = getattr(signal_config, "take_profit_pct", 50.0)
        stop_loss_pct = getattr(signal_config, "stop_loss_pct", 15.0)
        max_hold = getattr(signal_config, "max_hold_periods", 60)
        position_size_pct = getattr(signal_config, "max_position_pct", 0.1)
    else:
        take_profit_pct = 50.0
        stop_loss_pct = 15.0
        max_hold = 60
        position_size_pct = 0.1

    # Run parallel simulations
    final_equities, trade_counts, win_counts, max_drawdowns = _run_simulations_parallel(
        prices,
        entry_signals,
        cfg.n_simulations,
        cfg.initial_capital,
        position_size_pct,
        take_profit_pct,
        stop_loss_pct,
        max_hold,
        cfg.slippage_std_bps,
        1.0 - cfg.failure_rate_range[1],  # Convert failure to inclusion
        1.0 - cfg.failure_rate_range[0],
        cfg.seed or 42,
    )

    # Calculate returns
    returns_pct = (final_equities - cfg.initial_capital) / cfg.initial_capital * 100

    # Build result
    result = MonteCarloResult(
        n_simulations=cfg.n_simulations,
        final_equities=final_equities,
    )

    # Return statistics
    result.mean_return_pct = float(np.mean(returns_pct))
    result.std_return_pct = float(np.std(returns_pct))
    result.median_return_pct = float(np.median(returns_pct))
    result.percentile_5 = float(np.percentile(returns_pct, 5))
    result.percentile_25 = float(np.percentile(returns_pct, 25))
    result.percentile_75 = float(np.percentile(returns_pct, 75))
    result.percentile_95 = float(np.percentile(returns_pct, 95))

    # Sharpe statistics
    sharpe_stats = compute_sharpe_confidence_interval(
        final_equities, cfg.initial_capital
    )
    result.sharpe_mean = sharpe_stats["mean"]
    result.sharpe_std = sharpe_stats["std"]
    result.sharpe_5th = sharpe_stats["5th"]
    result.sharpe_median = sharpe_stats["50th"]
    result.sharpe_95th = sharpe_stats["95th"]

    # Trade statistics
    result.avg_trades = float(np.mean(trade_counts))
    valid_mask = trade_counts > 0
    if np.any(valid_mask):
        result.avg_win_rate = float(
            np.mean(win_counts[valid_mask] / trade_counts[valid_mask])
        )
    result.probability_of_profit = float(np.mean(returns_pct > 0))

    # Risk metrics
    result.risk = RiskMetrics()

    # VaR/CVaR
    var_5, cvar_5 = compute_var_cvar(returns_pct, 0.05)
    var_1, cvar_1 = compute_var_cvar(returns_pct, 0.01)
    result.risk.var_5 = float(var_5)
    result.risk.cvar_5 = float(cvar_5)
    result.risk.var_1 = float(var_1)
    result.risk.cvar_1 = float(cvar_1)

    # Drawdown stats
    result.risk.max_drawdown_mean = float(np.mean(max_drawdowns))
    result.risk.max_drawdown_5th = float(np.percentile(max_drawdowns, 5))
    result.risk.max_drawdown_95th = float(np.percentile(max_drawdowns, 95))

    # Probability of ruin
    prob_ruin, _ = compute_probability_of_ruin(
        max_drawdowns, cfg.ruin_threshold_pct
    )
    result.risk.prob_ruin = float(prob_ruin)

    return result


class MonteCarloEngine:
    """High-level Monte Carlo simulation engine.

    Wraps the low-level functions with a convenient API.

    Example:
        engine = MonteCarloEngine(config)
        result = engine.run(prices, entry_signals, signal_config)
        print(f"Sharpe 5th-95th: {result.sharpe_5th:.2f} - {result.sharpe_95th:.2f}")
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        self.config = config or MonteCarloConfig()

    def run(
        self,
        prices: np.ndarray,
        entry_signals: np.ndarray,
        signal_config: Optional[Any] = None,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation."""
        return run_monte_carlo_simulation(
            prices, entry_signals, self.config, signal_config
        )

    def run_with_backtester(
        self,
        backtester,
        data,
        signal_config,
    ) -> MonteCarloResult:
        """Run Monte Carlo using a backtester to generate signals.

        Args:
            backtester: VectorizedBacktester instance
            data: VectorizedData instance
            signal_config: SignalConfig instance

        Returns:
            MonteCarloResult
        """
        # Run backtester to get entry signals
        result = backtester.run(data, signal_config)

        return self.run(
            data.prices,
            result.entry_signals,
            signal_config,
        )

    def sensitivity_analysis(
        self,
        prices: np.ndarray,
        entry_signals: np.ndarray,
        signal_config: Optional[Any] = None,
        param_name: str = "slippage_std_bps",
        param_range: List[float] = None,
    ) -> Dict[float, MonteCarloResult]:
        """Run sensitivity analysis on a single parameter.

        Args:
            prices: Price matrix
            entry_signals: Entry signal matrix
            signal_config: Signal configuration
            param_name: Config parameter to vary
            param_range: Values to test

        Returns:
            Dict mapping parameter value to result
        """
        if param_range is None:
            param_range = [25, 50, 75, 100, 150]

        results = {}
        original_value = getattr(self.config, param_name)

        for value in param_range:
            setattr(self.config, param_name, value)
            result = self.run(prices, entry_signals, signal_config)
            results[value] = result

        # Restore original
        setattr(self.config, param_name, original_value)

        return results
