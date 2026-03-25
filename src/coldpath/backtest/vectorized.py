"""
Vectorized backtesting with Numba acceleration.

Provides 10-100x speedup over event-driven backtesting for parameter sweeps.
Uses Numba JIT compilation for numerical operations.

Three-tier architecture:
- Tier 1: Vectorized signal generation (this module) - FAST
- Tier 2: Event-driven execution simulation (engine.py) - REALISTIC
- Tier 3: Monte Carlo robustness testing - UNCERTAINTY
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from numba import boolean, float64, int64, jit, prange  # noqa: F401
    from numba.typed import List as NumbaList  # noqa: F401

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Fallback: define dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
    TORCH_MPS_AVAILABLE = (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
except Exception:
    torch = None
    TORCH_AVAILABLE = False
    TORCH_MPS_AVAILABLE = False


class MetalVectorEngine:
    """Apple Metal (MPS) acceleration for indicator calculations."""

    def __init__(self) -> None:
        if not TORCH_MPS_AVAILABLE or torch is None:
            raise RuntimeError("Metal MPS backend unavailable")
        self.device = torch.device("mps")

    def compute_returns(self, prices: np.ndarray, lookback: int = 1) -> np.ndarray:
        """Compute returns on MPS and return float64 numpy array."""
        if lookback < 1:
            raise ValueError("lookback must be >= 1")

        prices_t = torch.from_numpy(prices.astype(np.float32, copy=False)).to(self.device)
        n_tokens, n_timepoints = prices_t.shape
        out = torch.full(
            (n_tokens, n_timepoints),
            float("nan"),
            device=self.device,
            dtype=torch.float32,
        )

        if n_timepoints <= lookback:
            return out.cpu().numpy().astype(np.float64, copy=False)

        prev = prices_t[:, :-lookback]
        curr = prices_t[:, lookback:]
        valid = prev > 0

        returns = torch.zeros_like(curr)
        returns[valid] = (curr[valid] - prev[valid]) / prev[valid] * 100.0
        out[:, lookback:] = torch.where(valid, returns, torch.full_like(returns, float("nan")))
        return out.cpu().numpy().astype(np.float64, copy=False)

    def compute_volume_spikes(self, volumes: np.ndarray, window: int = 10) -> np.ndarray:
        """Compute volume spikes (current / rolling mean) on MPS."""
        if window < 1:
            raise ValueError("window must be >= 1")

        volumes_t = torch.from_numpy(volumes.astype(np.float32, copy=False)).to(self.device)
        n_tokens, n_timepoints = volumes_t.shape
        spikes = torch.ones((n_tokens, n_timepoints), device=self.device, dtype=torch.float32)

        if n_timepoints < window:
            return spikes.cpu().numpy().astype(np.float64, copy=False)

        rolling_means = volumes_t.unfold(dimension=1, size=window, step=1).mean(dim=2)
        current = volumes_t[:, window - 1 :]

        valid = rolling_means > 0
        ratios = torch.ones_like(current)
        ratios[valid] = current[valid] / rolling_means[valid]
        spikes[:, window - 1 :] = ratios

        return spikes.cpu().numpy().astype(np.float64, copy=False)


def _compute_profit_quality_metrics(
    n_trades: int,
    n_wins: int,
    n_losses: int,
    gross_wins_pct: float,
    gross_losses_pct: float,
    max_drawdown_pct: float,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> tuple[float, float, float]:
    """Compute expectancy, profit factor estimate, and profit quality score."""
    if n_trades <= 0:
        return 0.0, 0.0, 0.0

    win_rate = n_wins / max(1, n_trades)
    avg_win = gross_wins_pct / max(1, n_wins) if n_wins > 0 else take_profit_pct
    avg_loss = gross_losses_pct / max(1, n_losses) if n_losses > 0 else stop_loss_pct

    expectancy_pct = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
    estimated_profit_factor = (
        (gross_wins_pct / max(gross_losses_pct, 1e-9))
        if gross_losses_pct > 0
        else (999.0 if gross_wins_pct > 0 else 0.0)
    )
    drawdown_penalty = 1.0 + (max_drawdown_pct / 25.0)
    sample_penalty = min(1.0, n_trades / 20.0)
    quality_score = (
        (expectancy_pct + min(estimated_profit_factor, 5.0) * 2.0)
        * sample_penalty
        / drawdown_penalty
    )

    return float(expectancy_pct), float(estimated_profit_factor), float(quality_score)


@dataclass
class VectorizedData:
    """Vectorized market data for fast backtesting."""

    # All arrays have shape (n_tokens, n_timepoints)
    timestamps: np.ndarray  # int64: millisecond timestamps
    prices: np.ndarray  # float64: close prices
    volumes: np.ndarray  # float64: trading volumes
    liquidities: np.ndarray  # float64: pool liquidity USD

    # Token metadata (n_tokens,)
    token_ids: list[str]  # Token mint addresses

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def n_timepoints(self) -> int:
        return self.timestamps.shape[1] if len(self.timestamps.shape) > 1 else len(self.timestamps)


@dataclass
class SignalConfig:
    """Configuration for vectorized signal generation."""

    # Entry conditions
    min_price_change_pct: float = 5.0
    min_volume_spike: float = 2.0
    min_liquidity_usd: float = 10000.0

    # Exit conditions
    take_profit_pct: float = 15.0
    stop_loss_pct: float = 8.0
    max_hold_periods: int = 60  # 60 minutes for 1-min bars

    # Position sizing
    max_position_pct: float = 0.1  # 10% of capital per trade
    max_concurrent_positions: int = 5


@dataclass
class VectorizedResults:
    """Results from vectorized backtest."""

    # Signal arrays (n_tokens, n_timepoints)
    entry_signals: np.ndarray  # bool: entry signal at this point
    exit_signals: np.ndarray  # bool: exit signal at this point
    signal_strengths: np.ndarray  # float64: signal confidence 0-1

    # Trade tracking
    n_trades: int
    n_wins: int
    n_losses: int
    total_pnl_pct: float

    # Per-token performance
    token_returns: dict[str, float]

    # Optional diagnostics
    max_drawdown_pct: float = 0.0
    expectancy_pct: float = 0.0
    estimated_profit_factor: float = 0.0
    profit_quality_score: float = 0.0
    compute_backend: str = "cpu"

    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.n_trades == 0:
            return 0.0
        return self.n_wins / self.n_trades


# ============================================================
# Numba JIT-compiled functions for speed
# ============================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, cache=True, parallel=True)
    def _compute_returns(prices: np.ndarray, lookback: int = 1) -> np.ndarray:
        """Compute period returns for all tokens in parallel.

        Args:
            prices: (n_tokens, n_timepoints) price matrix
            lookback: Number of periods for return calculation

        Returns:
            (n_tokens, n_timepoints) return matrix (NaN for insufficient history)
        """
        n_tokens, n_timepoints = prices.shape
        returns = np.empty((n_tokens, n_timepoints), dtype=np.float64)
        returns[:, :lookback] = np.nan

        for i in prange(n_tokens):
            for t in range(lookback, n_timepoints):
                prev_price = prices[i, t - lookback]
                if prev_price > 0:
                    returns[i, t] = (prices[i, t] - prev_price) / prev_price * 100
                else:
                    returns[i, t] = np.nan

        return returns

    @jit(nopython=True, cache=True, parallel=True)
    def _compute_volume_spikes(volumes: np.ndarray, window: int = 10) -> np.ndarray:
        """Compute volume spike indicator (current / rolling average).

        Args:
            volumes: (n_tokens, n_timepoints) volume matrix
            window: Rolling window size

        Returns:
            (n_tokens, n_timepoints) volume spike ratios
        """
        n_tokens, n_timepoints = volumes.shape
        spikes = np.ones((n_tokens, n_timepoints), dtype=np.float64)

        for i in prange(n_tokens):
            for t in range(window, n_timepoints):
                # Compute rolling mean
                rolling_sum = 0.0
                for j in range(window):
                    rolling_sum += volumes[i, t - window + j]
                rolling_mean = rolling_sum / window

                if rolling_mean > 0:
                    spikes[i, t] = volumes[i, t] / rolling_mean
                else:
                    spikes[i, t] = 1.0

        return spikes

    @jit(nopython=True, cache=True, parallel=True)
    def _generate_entry_signals(
        returns: np.ndarray,
        volume_spikes: np.ndarray,
        liquidities: np.ndarray,
        min_return: float,
        min_volume_spike: float,
        min_liquidity: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate entry signals based on momentum and volume.

        Returns:
            Tuple of (entry_signals, signal_strengths)
        """
        n_tokens, n_timepoints = returns.shape
        entry_signals = np.zeros((n_tokens, n_timepoints), dtype=np.bool_)
        strengths = np.zeros((n_tokens, n_timepoints), dtype=np.float64)

        for i in prange(n_tokens):
            for t in range(n_timepoints):
                ret = returns[i, t]
                spike = volume_spikes[i, t]
                liq = liquidities[i, t]

                if np.isnan(ret):
                    continue

                # Entry conditions
                if ret >= min_return and spike >= min_volume_spike and liq >= min_liquidity:
                    entry_signals[i, t] = True
                    # Signal strength based on return magnitude and volume
                    strength = min(1.0, ret / (min_return * 2)) * min(
                        1.0, spike / (min_volume_spike * 2)
                    )
                    strengths[i, t] = strength

        return entry_signals, strengths

    @jit(nopython=True, cache=True)
    def _simulate_trades_fast(
        prices: np.ndarray,
        entry_signals: np.ndarray,
        take_profit_pct: float,
        stop_loss_pct: float,
        max_hold: int,
    ) -> tuple[int, int, int, float, float, float, float]:
        """Fast trade simulation using vectorized operations.

        Returns:
            Tuple of (n_trades, n_wins, n_losses, total_pnl_pct,
            max_drawdown_pct, gross_wins_pct, gross_losses_pct)
        """
        n_tokens, n_timepoints = prices.shape
        n_trades = 0
        n_wins = 0
        n_losses = 0
        total_pnl = 0.0
        gross_wins = 0.0
        gross_losses = 0.0
        equity = 1.0
        peak_equity = 1.0
        max_drawdown = 0.0

        for i in range(n_tokens):
            t = 0
            while t < n_timepoints:
                # Look for entry
                if entry_signals[i, t]:
                    entry_price = prices[i, t]
                    if entry_price <= 0:
                        t += 1
                        continue

                    # Track position until exit
                    exit_t = t + 1
                    exit_price = entry_price

                    while exit_t < min(t + max_hold, n_timepoints):
                        current_price = prices[i, exit_t]
                        if current_price <= 0:
                            exit_t += 1
                            continue

                        pnl_pct = (current_price - entry_price) / entry_price * 100

                        # Check take profit
                        if pnl_pct >= take_profit_pct:
                            exit_price = current_price
                            break

                        # Check stop loss
                        if pnl_pct <= -stop_loss_pct:
                            exit_price = current_price
                            break

                        exit_t += 1

                    # Record trade
                    if exit_t < n_timepoints:
                        exit_price = prices[i, exit_t]

                    if exit_price > 0:
                        trade_pnl = (exit_price - entry_price) / entry_price * 100
                        total_pnl += trade_pnl
                        n_trades += 1
                        if trade_pnl >= 0:
                            n_wins += 1
                            gross_wins += trade_pnl
                        else:
                            n_losses += 1
                            gross_losses += -trade_pnl

                        # Use compounded equity path to estimate drawdown risk.
                        equity = equity * (1.0 + trade_pnl / 100.0)
                        if equity > peak_equity:
                            peak_equity = equity
                        if peak_equity > 0:
                            drawdown = (peak_equity - equity) / peak_equity * 100.0
                            if drawdown > max_drawdown:
                                max_drawdown = drawdown

                    # Skip to after exit
                    t = exit_t + 1
                else:
                    t += 1

        return n_trades, n_wins, n_losses, total_pnl, max_drawdown, gross_wins, gross_losses

else:
    # Non-Numba fallback implementations
    def _compute_returns(prices: np.ndarray, lookback: int = 1) -> np.ndarray:
        """Compute returns without Numba."""
        n_tokens, n_timepoints = prices.shape
        returns = np.empty((n_tokens, n_timepoints))
        returns[:, :lookback] = np.nan

        for t in range(lookback, n_timepoints):
            prev_prices = prices[:, t - lookback]
            mask = prev_prices > 0
            returns[mask, t] = (prices[mask, t] - prev_prices[mask]) / prev_prices[mask] * 100
            returns[~mask, t] = np.nan

        return returns

    def _compute_volume_spikes(volumes: np.ndarray, window: int = 10) -> np.ndarray:
        """Compute volume spikes without Numba."""
        n_tokens, n_timepoints = volumes.shape
        spikes = np.ones((n_tokens, n_timepoints))

        for t in range(window, n_timepoints):
            rolling_mean = np.mean(volumes[:, t - window : t], axis=1)
            mask = rolling_mean > 0
            spikes[mask, t] = volumes[mask, t] / rolling_mean[mask]

        return spikes

    def _generate_entry_signals(
        returns: np.ndarray,
        volume_spikes: np.ndarray,
        liquidities: np.ndarray,
        min_return: float,
        min_volume_spike: float,
        min_liquidity: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate entry signals without Numba."""
        entry_signals = (
            (returns >= min_return)
            & (volume_spikes >= min_volume_spike)
            & (liquidities >= min_liquidity)
            & ~np.isnan(returns)
        )

        strengths = np.zeros_like(returns)
        strengths[entry_signals] = np.minimum(
            1.0, returns[entry_signals] / (min_return * 2)
        ) * np.minimum(1.0, volume_spikes[entry_signals] / (min_volume_spike * 2))

        return entry_signals, strengths

    def _simulate_trades_fast(
        prices: np.ndarray,
        entry_signals: np.ndarray,
        take_profit_pct: float,
        stop_loss_pct: float,
        max_hold: int,
    ) -> tuple[int, int, int, float, float, float, float]:
        """Simulate trades without Numba (slower)."""
        n_tokens, n_timepoints = prices.shape
        n_trades = 0
        n_wins = 0
        n_losses = 0
        total_pnl = 0.0
        gross_wins = 0.0
        gross_losses = 0.0
        equity = 1.0
        peak_equity = 1.0
        max_drawdown = 0.0

        for i in range(n_tokens):
            t = 0
            while t < n_timepoints:
                if entry_signals[i, t]:
                    entry_price = prices[i, t]
                    if entry_price <= 0:
                        t += 1
                        continue

                    exit_t = t + 1
                    while exit_t < min(t + max_hold, n_timepoints):
                        current_price = prices[i, exit_t]
                        if current_price <= 0:
                            exit_t += 1
                            continue

                        pnl_pct = (current_price - entry_price) / entry_price * 100

                        if pnl_pct >= take_profit_pct or pnl_pct <= -stop_loss_pct:
                            break
                        exit_t += 1

                    if exit_t < n_timepoints and prices[i, exit_t] > 0:
                        trade_pnl = (prices[i, exit_t] - entry_price) / entry_price * 100
                        total_pnl += trade_pnl
                        n_trades += 1
                        if trade_pnl >= 0:
                            n_wins += 1
                            gross_wins += trade_pnl
                        else:
                            n_losses += 1
                            gross_losses += -trade_pnl

                        equity = equity * (1.0 + trade_pnl / 100.0)
                        peak_equity = max(peak_equity, equity)
                        if peak_equity > 0:
                            max_drawdown = max(
                                max_drawdown, (peak_equity - equity) / peak_equity * 100.0
                            )

                    t = exit_t + 1
                else:
                    t += 1

        return n_trades, n_wins, n_losses, total_pnl, max_drawdown, gross_wins, gross_losses


class VectorizedBacktester:
    """High-speed vectorized backtester for parameter sweeps."""

    def __init__(self, config: SignalConfig | None = None, compute_backend: str | None = None):
        self.config = config or SignalConfig()
        self._metal_engine: MetalVectorEngine | None = None
        self.compute_backend = self._resolve_compute_backend(compute_backend)

        if not NUMBA_AVAILABLE:
            logger.warning(
                "Numba not available - vectorized backtesting will be slower. "
                "Install numba for 10-100x speedup."
            )
        if self.compute_backend == "metal":
            logger.info("Using Metal/MPS backend for vectorized indicators")

    def _resolve_compute_backend(self, requested: str | None) -> str:
        """Resolve compute backend preference."""
        backend = (requested or os.getenv("COLDPATH_BACKTEST_BACKEND", "auto")).lower()
        valid = {"auto", "metal", "numba", "cpu"}
        if backend not in valid:
            logger.warning("Unknown backend '%s', falling back to auto", backend)
            backend = "auto"

        if backend in {"auto", "metal"} and TORCH_MPS_AVAILABLE:
            try:
                self._metal_engine = MetalVectorEngine()
                return "metal"
            except Exception as exc:
                logger.warning("Metal backend initialization failed: %s", exc)

        if backend in {"auto", "numba"} and NUMBA_AVAILABLE:
            return "numba"

        return "cpu"

    def _compute_indicators(self, data: VectorizedData) -> tuple[np.ndarray, np.ndarray]:
        """Compute returns and volume spikes on selected backend."""
        if self.compute_backend == "metal" and self._metal_engine is not None:
            returns = self._metal_engine.compute_returns(data.prices, lookback=1)
            volume_spikes = self._metal_engine.compute_volume_spikes(data.volumes, window=10)
            return returns, volume_spikes
        returns = _compute_returns(data.prices, lookback=1)
        volume_spikes = _compute_volume_spikes(data.volumes, window=10)
        return returns, volume_spikes

    def run(self, data: VectorizedData, config: SignalConfig | None = None) -> VectorizedResults:
        """Run vectorized backtest on prepared data.

        Args:
            data: Prepared vectorized market data
            config: Optional signal configuration override

        Returns:
            VectorizedResults with signals and performance metrics
        """
        cfg = config or self.config

        # Compute returns and indicators
        returns, volume_spikes = self._compute_indicators(data)

        # Generate entry signals
        entry_signals, signal_strengths = _generate_entry_signals(
            returns,
            volume_spikes,
            data.liquidities,
            min_return=cfg.min_price_change_pct,
            min_volume_spike=cfg.min_volume_spike,
            min_liquidity=cfg.min_liquidity_usd,
        )

        # Simulate trades
        (
            n_trades,
            n_wins,
            n_losses,
            total_pnl,
            max_drawdown_pct,
            gross_wins_pct,
            gross_losses_pct,
        ) = _simulate_trades_fast(
            data.prices,
            entry_signals,
            take_profit_pct=cfg.take_profit_pct,
            stop_loss_pct=cfg.stop_loss_pct,
            max_hold=cfg.max_hold_periods,
        )
        expectancy_pct, estimated_profit_factor, profit_quality_score = (
            _compute_profit_quality_metrics(
                n_trades=n_trades,
                n_wins=n_wins,
                n_losses=n_losses,
                gross_wins_pct=gross_wins_pct,
                gross_losses_pct=gross_losses_pct,
                max_drawdown_pct=max_drawdown_pct,
                take_profit_pct=cfg.take_profit_pct,
                stop_loss_pct=cfg.stop_loss_pct,
            )
        )

        # Compute per-token returns (for analysis)
        token_returns = {}
        for i, token_id in enumerate(data.token_ids):
            token_entries = entry_signals[i].sum()
            if token_entries > 0:
                # Rough estimate of token contribution
                token_returns[token_id] = float(total_pnl / max(1, n_trades) * token_entries)

        return VectorizedResults(
            entry_signals=entry_signals,
            exit_signals=np.zeros_like(entry_signals),  # Filled by simulation
            signal_strengths=signal_strengths,
            n_trades=n_trades,
            n_wins=n_wins,
            n_losses=n_losses,
            total_pnl_pct=total_pnl,
            max_drawdown_pct=max_drawdown_pct,
            expectancy_pct=expectancy_pct,
            estimated_profit_factor=estimated_profit_factor,
            profit_quality_score=profit_quality_score,
            token_returns=token_returns,
            compute_backend=self.compute_backend,
        )

    def parameter_sweep(
        self,
        data: VectorizedData,
        param_grid: dict[str, list[float]],
        metric: str = "profit_quality",
        n_jobs: int = -1,
    ) -> list[dict[str, Any]]:
        """Run parameter sweep across configuration space.

        Uses Numba-accelerated backtesting for each parameter combination.
        Supports parallel execution with joblib for large parameter spaces.

        Args:
            data: Vectorized market data
            param_grid: Dictionary of parameter names to values to test
            metric: Metric to optimize ("sharpe", "total_pnl", "win_rate")
            n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)

        Returns:
            List of results sorted by target metric
        """
        import itertools

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info(f"Running parameter sweep: {len(combinations)} combinations")

        # Use parallel execution for large sweeps
        if len(combinations) > 10 and n_jobs != 1:
            results = self._parallel_sweep(data, param_names, combinations, metric, n_jobs)
        else:
            results = self._sequential_sweep(data, param_names, combinations, metric)

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        if results:
            logger.info(
                f"Best parameters: {results[0]['params']}, "
                f"score={results[0]['score']:.4f}, "
                f"win_rate={results[0]['win_rate']:.2%}"
            )

        return results

    def _sequential_sweep(
        self,
        data: VectorizedData,
        param_names: list[str],
        combinations: list[tuple],
        metric: str,
    ) -> list[dict[str, Any]]:
        """Sequential parameter sweep (for small param spaces)."""
        results = []
        for combo in combinations:
            params = dict(zip(param_names, combo, strict=False))
            result_dict = self._evaluate_params(data, params, metric)
            results.append(result_dict)
        return results

    def _parallel_sweep(
        self,
        data: VectorizedData,
        param_names: list[str],
        combinations: list[tuple],
        metric: str,
        n_jobs: int,
    ) -> list[dict[str, Any]]:
        """Parallel parameter sweep using multiprocessing."""
        try:
            import multiprocessing as mp
            from concurrent.futures import (
                ProcessPoolExecutor,  # noqa: F401
                as_completed,
            )

            if n_jobs == -1:
                n_jobs = mp.cpu_count()

            # Use ThreadPoolExecutor instead of ProcessPoolExecutor for Numba
            # (Numba releases GIL during computation)
            from concurrent.futures import ThreadPoolExecutor

            results = []
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all jobs
                futures = {
                    executor.submit(
                        self._evaluate_params,
                        data,
                        dict(zip(param_names, combo, strict=False)),
                        metric,
                    ): combo
                    for combo in combinations
                }

                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Parameter combination failed: {e}")

            return results

        except ImportError:
            # Fallback to sequential
            logger.warning("concurrent.futures not available, falling back to sequential")
            return self._sequential_sweep(data, param_names, combinations, metric)

    def _evaluate_params(
        self,
        data: VectorizedData,
        params: dict[str, float],
        metric: str,
    ) -> dict[str, Any]:
        """Evaluate a single parameter combination."""
        cfg = SignalConfig(
            min_price_change_pct=params.get(
                "min_price_change_pct", self.config.min_price_change_pct
            ),
            min_volume_spike=params.get("min_volume_spike", self.config.min_volume_spike),
            min_liquidity_usd=params.get("min_liquidity_usd", self.config.min_liquidity_usd),
            take_profit_pct=params.get("take_profit_pct", self.config.take_profit_pct),
            stop_loss_pct=params.get("stop_loss_pct", self.config.stop_loss_pct),
            max_hold_periods=int(params.get("max_hold_periods", self.config.max_hold_periods)),
        )

        # Run backtest
        result = self.run(data, cfg)

        # Calculate target metric from realized simulation outcomes.
        win_rate = result.win_rate()
        expectancy = result.expectancy_pct
        estimated_profit_factor = result.estimated_profit_factor
        robustness_penalty = 1.0 + (result.max_drawdown_pct / 25.0)
        sample_penalty = min(1.0, result.n_trades / 20.0)

        if metric == "sharpe":
            score = (
                expectancy * min(estimated_profit_factor, 5.0) * sample_penalty
            ) / robustness_penalty
        elif metric == "total_pnl":
            score = result.total_pnl_pct
        elif metric == "win_rate":
            score = win_rate
        elif metric == "profit_quality":
            score = result.profit_quality_score
        else:
            score = result.total_pnl_pct

        return {
            "params": params,
            "n_trades": result.n_trades,
            "n_wins": result.n_wins,
            "win_rate": win_rate,
            "total_pnl_pct": result.total_pnl_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "expectancy_pct": expectancy,
            "estimated_profit_factor": estimated_profit_factor,
            "profit_quality_score": result.profit_quality_score,
            "compute_backend": result.compute_backend,
            "score": score,
        }

    def optuna_optimize(
        self,
        data: VectorizedData,
        n_trials: int = 100,
        metric: str = "profit_quality",
        param_ranges: dict[str, tuple[float, float]] | None = None,
        timeout: int | None = None,
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Optimize parameters using Optuna Bayesian optimization.

        Much more efficient than grid search for high-dimensional spaces.
        Targets 10,000+ combinations/day capability.

        Args:
            data: Vectorized market data
            n_trials: Number of optimization trials
            metric: Metric to optimize
            param_ranges: Optional custom parameter ranges
            timeout: Optional timeout in seconds
            n_jobs: Number of parallel workers

        Returns:
            Best parameters and optimization results
        """
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.error("Optuna not installed. Run: pip install optuna")
            raise

        # Default parameter ranges
        ranges = param_ranges or {
            "min_price_change_pct": (1.0, 20.0),
            "min_volume_spike": (1.0, 5.0),
            "min_liquidity_usd": (5000.0, 50000.0),
            "take_profit_pct": (10.0, 100.0),
            "stop_loss_pct": (5.0, 30.0),
            "max_hold_periods": (10, 120),
        }

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for name, (low, high) in ranges.items():
                if name == "max_hold_periods":
                    params[name] = trial.suggest_int(name, int(low), int(high))
                else:
                    params[name] = trial.suggest_float(name, low, high)

            result = self._evaluate_params(data, params, metric)
            return result["score"]

        # Create study with pruning for efficiency
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )

        best_params = study.best_params
        best_score = study.best_value

        # Run final evaluation with best params
        final_result = self._evaluate_params(data, best_params, metric)

        logger.info(f"Optuna optimization complete: {n_trials} trials, best_score={best_score:.4f}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_result": final_result,
            "n_trials": n_trials,
            "study": study,
        }


# ============================================================
# Enhanced Signal Generators (Numba JIT)
# ============================================================

if NUMBA_AVAILABLE:

    @jit(nopython=True, cache=True)
    def _compute_rsi_signals(
        prices: np.ndarray,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> np.ndarray:
        """Compute RSI-based entry signals.

        Args:
            prices: (n_tokens, n_timepoints) price matrix
            period: RSI period
            oversold: RSI level for buy signal
            overbought: RSI level for sell signal

        Returns:
            (n_tokens, n_timepoints) signal matrix (1=buy, -1=sell, 0=hold)
        """
        n_tokens, n_timepoints = prices.shape
        signals = np.zeros((n_tokens, n_timepoints), dtype=np.int32)

        if n_timepoints < period + 1:
            return signals

        for i in range(n_tokens):
            for t in range(period, n_timepoints):
                gains = 0.0
                losses = 0.0

                for j in range(t - period + 1, t + 1):
                    change = prices[i, j] - prices[i, j - 1]
                    if change > 0:
                        gains += change
                    else:
                        losses -= change

                avg_gain = gains / period
                avg_loss = losses / period

                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                else:
                    rsi = 100.0 if avg_gain > 0 else 50.0

                # Generate signals on RSI crosses
                if t > period:
                    # Would need previous RSI to detect crosses properly
                    if rsi < oversold:
                        signals[i, t] = 1  # Buy signal
                    elif rsi > overbought:
                        signals[i, t] = -1  # Sell signal

        return signals

    @jit(nopython=True, cache=True)
    def _compute_breakout_signals(
        prices: np.ndarray,
        lookback: int = 20,
    ) -> np.ndarray:
        """Compute breakout trading signals.

        Args:
            prices: (n_tokens, n_timepoints) price matrix
            lookback: Lookback period for high/low

        Returns:
            (n_tokens, n_timepoints) signal matrix
        """
        n_tokens, n_timepoints = prices.shape
        signals = np.zeros((n_tokens, n_timepoints), dtype=np.int32)

        if n_timepoints < lookback + 1:
            return signals

        for i in range(n_tokens):
            for t in range(lookback, n_timepoints):
                # Calculate lookback high/low
                period_high = prices[i, t - lookback]
                period_low = prices[i, t - lookback]

                for j in range(t - lookback, t):
                    if prices[i, j] > period_high:
                        period_high = prices[i, j]
                    if prices[i, j] < period_low:
                        period_low = prices[i, j]

                # Breakout signals
                if prices[i, t] > period_high:
                    signals[i, t] = 1  # Breakout buy
                elif prices[i, t] < period_low:
                    signals[i, t] = -1  # Breakdown sell

        return signals

    @jit(nopython=True, cache=True)
    def _simulate_amm_fill(
        amount_in: float,
        liquidity_base: float,
        liquidity_quote: float,
        is_buy: boolean,
        mev_penalty_bps: float = 100.0,
    ) -> tuple[float, float, int]:
        """Simulate AMM fill using constant product formula.

        Args:
            amount_in: Amount being traded
            liquidity_base: Base token liquidity
            liquidity_quote: Quote token liquidity
            is_buy: True if buying tokens with base
            mev_penalty_bps: MEV penalty in bps

        Returns:
            (amount_out, fill_price, slippage_bps)
        """
        k = liquidity_base * liquidity_quote

        if is_buy:
            new_base = liquidity_base + amount_in
            new_quote = k / new_base
            amount_out = liquidity_quote - new_quote
            spot_price = liquidity_base / liquidity_quote
            actual_price = amount_in / amount_out if amount_out > 0 else 0.0
        else:
            new_quote = liquidity_quote + amount_in
            new_base = k / new_quote
            amount_out = liquidity_base - new_base
            spot_price = liquidity_quote / liquidity_base
            actual_price = amount_in / amount_out if amount_out > 0 else 0.0

        # Calculate slippage
        if spot_price > 0:
            slippage_bps = int(abs(actual_price / spot_price - 1.0) * 10000)
        else:
            slippage_bps = 0

        # Add MEV penalty (exponential-ish via uniform * 2)
        mev = int(np.random.random() * mev_penalty_bps * 2)
        mev = min(mev, 500)
        total_slippage = slippage_bps + mev

        # Adjust output
        slippage_factor = 1.0 - total_slippage / 10000.0
        amount_out_adjusted = amount_out * slippage_factor
        fill_price = amount_in / amount_out_adjusted if amount_out_adjusted > 0 else 0.0

        return amount_out_adjusted, fill_price, total_slippage

    @jit(nopython=True, cache=True, parallel=True)
    def _simulate_with_amm(
        prices: np.ndarray,
        liquidities: np.ndarray,
        entry_signals: np.ndarray,
        initial_capital: float,
        position_size_pct: float,
        take_profit_pct: float,
        stop_loss_pct: float,
        max_hold: int,
        mev_penalty_bps: float,
    ) -> tuple[np.ndarray, int, int, int, float, float, float, float, float]:
        """Simulate trades with realistic AMM fills.

        Returns:
            (equity_curve, n_trades, n_wins, n_losses, total_pnl_sol,
             max_drawdown_pct, expectancy_pct, estimated_profit_factor, profit_quality_score)
        """
        n_tokens, n_timepoints = prices.shape

        # Track global state
        equity = np.zeros(n_timepoints, dtype=np.float64)
        equity[0] = initial_capital
        capital = initial_capital

        n_trades = 0
        n_wins = 0
        n_losses = 0
        total_pnl = 0.0
        gross_wins = 0.0
        gross_losses = 0.0

        # Track positions per token
        in_position = np.zeros(n_tokens, dtype=np.bool_)
        entry_prices = np.zeros(n_tokens, dtype=np.float64)
        entry_times = np.zeros(n_tokens, dtype=np.int64)
        position_sizes = np.zeros(n_tokens, dtype=np.float64)

        for t in range(1, n_timepoints):
            # Start with previous equity
            equity[t] = equity[t - 1]

            for i in range(n_tokens):
                price = prices[i, t]
                liq = liquidities[i, t]

                if price <= 0:
                    continue

                # Check exits for open positions
                if in_position[i]:
                    hold_time = t - entry_times[i]
                    entry_px = entry_prices[i]

                    if entry_px > 0:
                        pnl_pct = (price - entry_px) / entry_px * 100

                        should_exit = (
                            pnl_pct >= take_profit_pct
                            or pnl_pct <= -stop_loss_pct
                            or hold_time >= max_hold
                        )

                        if should_exit:
                            # Simulate sell
                            tokens = position_sizes[i]
                            sol_out, _, _ = _simulate_amm_fill(
                                tokens, liq, liq * price, False, mev_penalty_bps
                            )

                            trade_pnl = sol_out - (entry_px * tokens)
                            total_pnl += trade_pnl
                            capital += sol_out
                            n_trades += 1

                            if trade_pnl >= 0:
                                n_wins += 1
                                gross_wins += trade_pnl
                            else:
                                n_losses += 1
                                gross_losses += -trade_pnl

                            in_position[i] = False
                            position_sizes[i] = 0.0
                            equity[t] = capital

                # Check entries
                if not in_position[i] and entry_signals[i, t]:
                    trade_sol = capital * position_size_pct

                    if trade_sol > 0.001:
                        tokens_out, fill_price, _ = _simulate_amm_fill(
                            trade_sol, liq, liq * price, True, mev_penalty_bps
                        )

                        if tokens_out > 0:
                            capital -= trade_sol
                            in_position[i] = True
                            entry_prices[i] = fill_price
                            entry_times[i] = t
                            position_sizes[i] = tokens_out
                            equity[t] = capital + tokens_out * price

        # Risk/quality metrics from realized equity/trade outcomes.
        peak = equity[0]
        max_drawdown = 0.0
        for idx in range(1, n_timepoints):
            if equity[idx] > peak:
                peak = equity[idx]
            if peak > 0:
                dd = (peak - equity[idx]) / peak * 100.0
                if dd > max_drawdown:
                    max_drawdown = dd

        if n_trades > 0:
            win_rate = n_wins / n_trades
            avg_win = gross_wins / n_wins if n_wins > 0 else 0.0
            avg_loss = gross_losses / n_losses if n_losses > 0 else 0.0
            expectancy_pct = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
        else:
            expectancy_pct = 0.0

        if gross_losses > 0:
            estimated_profit_factor = gross_wins / gross_losses
        elif gross_wins > 0:
            estimated_profit_factor = 999.0
        else:
            estimated_profit_factor = 0.0

        sample_penalty = min(1.0, n_trades / 20.0)
        drawdown_penalty = 1.0 + (max_drawdown / 25.0)
        quality = (
            (expectancy_pct + min(estimated_profit_factor, 5.0) * 2.0)
            * sample_penalty
            / drawdown_penalty
        )

        return (
            equity,
            n_trades,
            n_wins,
            n_losses,
            total_pnl,
            max_drawdown,
            expectancy_pct,
            estimated_profit_factor,
            quality,
        )

    @jit(nopython=True, cache=True)
    def _monte_carlo_simulation(
        prices: np.ndarray,
        entry_signals: np.ndarray,
        n_simulations: int,
        initial_capital: float,
        position_size_pct: float,
        take_profit_pct: float,
        stop_loss_pct: float,
        max_hold: int,
        inclusion_prob: float = 0.85,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Monte Carlo simulation for robustness testing.

        Randomizes transaction inclusion to model real-world uncertainty.

        Returns:
            (final_equities, trade_counts) for each simulation
        """
        n_tokens, n_timepoints = prices.shape

        final_equities = np.zeros(n_simulations, dtype=np.float64)
        trade_counts = np.zeros(n_simulations, dtype=np.int64)

        for sim in range(n_simulations):
            capital = initial_capital
            n_trades = 0

            for i in range(n_tokens):
                t = 0
                while t < n_timepoints:
                    if entry_signals[i, t]:
                        # Random inclusion
                        if np.random.random() > inclusion_prob:
                            t += 1
                            continue

                        entry_price = prices[i, t]
                        if entry_price <= 0:
                            t += 1
                            continue

                        trade_size = capital * position_size_pct
                        if trade_size < 0.001:
                            t += 1
                            continue

                        capital -= trade_size
                        tokens = trade_size / entry_price

                        # Find exit
                        exit_t = t + 1
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
                                break
                            exit_t += 1

                        # Close position
                        if exit_t < n_timepoints:
                            exit_price = prices[i, exit_t]
                        else:
                            exit_price = prices[i, -1]

                        if exit_price > 0:
                            capital += tokens * exit_price
                            n_trades += 1

                        t = exit_t + 1
                    else:
                        t += 1

            final_equities[sim] = capital
            trade_counts[sim] = n_trades

        return final_equities, trade_counts

else:
    # Fallback implementations for non-Numba
    def _compute_rsi_signals(prices, period=14, oversold=30.0, overbought=70.0):
        """RSI signals without Numba."""
        n_tokens, n_timepoints = prices.shape
        signals = np.zeros((n_tokens, n_timepoints), dtype=np.int32)
        # Simplified implementation
        for i in range(n_tokens):
            for t in range(period, n_timepoints):
                changes = np.diff(prices[i, t - period : t + 1])
                gains = np.sum(changes[changes > 0])
                losses = -np.sum(changes[changes < 0])
                rs = gains / (losses + 1e-10)
                rsi = 100.0 - (100.0 / (1.0 + rs))
                if rsi < oversold:
                    signals[i, t] = 1
                elif rsi > overbought:
                    signals[i, t] = -1
        return signals

    def _compute_breakout_signals(prices, lookback=20):
        """Breakout signals without Numba."""
        n_tokens, n_timepoints = prices.shape
        signals = np.zeros((n_tokens, n_timepoints), dtype=np.int32)
        for i in range(n_tokens):
            for t in range(lookback, n_timepoints):
                period_high = np.max(prices[i, t - lookback : t])
                period_low = np.min(prices[i, t - lookback : t])
                if prices[i, t] > period_high:
                    signals[i, t] = 1
                elif prices[i, t] < period_low:
                    signals[i, t] = -1
        return signals

    def _simulate_amm_fill(
        amount_in, liquidity_base, liquidity_quote, is_buy, mev_penalty_bps=100.0
    ):
        """AMM fill simulation without Numba."""
        k = liquidity_base * liquidity_quote
        if is_buy:
            new_base = liquidity_base + amount_in
            new_quote = k / new_base
            amount_out = liquidity_quote - new_quote
        else:
            new_quote = liquidity_quote + amount_in
            new_base = k / new_quote
            amount_out = liquidity_base - new_base

        slippage_factor = 1.0 - (np.random.random() * mev_penalty_bps * 2) / 10000
        amount_out_adjusted = amount_out * max(0, slippage_factor)
        fill_price = amount_in / amount_out_adjusted if amount_out_adjusted > 0 else 0
        return amount_out_adjusted, fill_price, int((1 - slippage_factor) * 10000)

    def _simulate_with_amm(
        prices,
        liquidities,
        entry_signals,
        initial_capital,
        position_size_pct,
        take_profit_pct,
        stop_loss_pct,
        max_hold,
        mev_penalty_bps,
    ):
        """AMM simulation without Numba."""
        n_tokens, n_timepoints = prices.shape
        equity = np.zeros(n_timepoints)
        equity[0] = initial_capital
        # Simplified - just use fast simulation
        (
            n_trades,
            n_wins,
            n_losses,
            total_pnl,
            max_drawdown_pct,
            gross_wins_pct,
            gross_losses_pct,
        ) = _simulate_trades_fast(prices, entry_signals, take_profit_pct, stop_loss_pct, max_hold)
        expectancy_pct, estimated_profit_factor, profit_quality_score = (
            _compute_profit_quality_metrics(
                n_trades=n_trades,
                n_wins=n_wins,
                n_losses=n_losses,
                gross_wins_pct=gross_wins_pct,
                gross_losses_pct=gross_losses_pct,
                max_drawdown_pct=max_drawdown_pct,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
            )
        )
        equity[-1] = initial_capital + total_pnl * initial_capital / 100
        return (
            equity,
            n_trades,
            n_wins,
            n_losses,
            total_pnl,
            max_drawdown_pct,
            expectancy_pct,
            estimated_profit_factor,
            profit_quality_score,
        )

    def _monte_carlo_simulation(
        prices,
        entry_signals,
        n_simulations,
        initial_capital,
        position_size_pct,
        take_profit_pct,
        stop_loss_pct,
        max_hold,
        inclusion_prob=0.85,
    ):
        """Monte Carlo without Numba."""
        final_equities = np.zeros(n_simulations)
        trade_counts = np.zeros(n_simulations, dtype=np.int64)
        for sim in range(n_simulations):
            n_trades, _, _, total_pnl, _, _, _ = _simulate_trades_fast(
                prices, entry_signals, take_profit_pct, stop_loss_pct, max_hold
            )
            final_equities[sim] = initial_capital * (
                1 + total_pnl / 100 * np.random.uniform(0.8, 1.2)
            )
            trade_counts[sim] = n_trades
        return final_equities, trade_counts


# ============================================================
# Enhanced VectorizedBacktester with Monte Carlo
# ============================================================


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation."""

    mean_return_pct: float
    std_return_pct: float
    percentile_5: float
    percentile_25: float
    median_return_pct: float
    percentile_75: float
    percentile_95: float
    probability_of_profit: float
    avg_trades: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_return_pct": self.mean_return_pct,
            "std_return_pct": self.std_return_pct,
            "percentile_5": self.percentile_5,
            "percentile_25": self.percentile_25,
            "median_return_pct": self.median_return_pct,
            "percentile_75": self.percentile_75,
            "percentile_95": self.percentile_95,
            "probability_of_profit": self.probability_of_profit,
            "avg_trades": self.avg_trades,
        }


class EnhancedVectorizedBacktester(VectorizedBacktester):
    """Enhanced backtester with AMM simulation and Monte Carlo support."""

    def run_with_amm(
        self,
        data: VectorizedData,
        config: SignalConfig | None = None,
        mev_penalty_bps: float = 100.0,
    ) -> tuple[VectorizedResults, np.ndarray]:
        """Run backtest with realistic AMM fill simulation.

        Returns:
            Tuple of (VectorizedResults, equity_curve)
        """
        cfg = config or self.config

        returns, volume_spikes = self._compute_indicators(data)

        entry_signals, signal_strengths = _generate_entry_signals(
            returns,
            volume_spikes,
            data.liquidities,
            min_return=cfg.min_price_change_pct,
            min_volume_spike=cfg.min_volume_spike,
            min_liquidity=cfg.min_liquidity_usd,
        )

        (
            equity,
            n_trades,
            n_wins,
            n_losses,
            total_pnl,
            max_drawdown_pct,
            expectancy_pct,
            estimated_profit_factor,
            profit_quality_score,
        ) = _simulate_with_amm(
            data.prices,
            data.liquidities,
            entry_signals,
            initial_capital=10.0,  # Default 10 SOL
            position_size_pct=cfg.max_position_pct,
            take_profit_pct=cfg.take_profit_pct,
            stop_loss_pct=cfg.stop_loss_pct,
            max_hold=cfg.max_hold_periods,
            mev_penalty_bps=mev_penalty_bps,
        )
        total_pnl_pct = (total_pnl / 10.0) * 100.0

        token_returns = {}
        for i, token_id in enumerate(data.token_ids):
            token_entries = entry_signals[i].sum()
            if token_entries > 0:
                token_returns[token_id] = float(total_pnl_pct / max(1, n_trades) * token_entries)

        results = VectorizedResults(
            entry_signals=entry_signals,
            exit_signals=np.zeros_like(entry_signals),
            signal_strengths=signal_strengths,
            n_trades=n_trades,
            n_wins=n_wins,
            n_losses=n_losses,
            total_pnl_pct=total_pnl_pct,
            max_drawdown_pct=max_drawdown_pct,
            expectancy_pct=expectancy_pct,
            estimated_profit_factor=estimated_profit_factor,
            profit_quality_score=profit_quality_score,
            token_returns=token_returns,
            compute_backend=self.compute_backend,
        )

        return results, equity

    def monte_carlo(
        self,
        data: VectorizedData,
        n_simulations: int = 1000,
        config: SignalConfig | None = None,
        inclusion_prob: float = 0.85,
    ) -> MonteCarloResults:
        """Run Monte Carlo simulation for robustness testing.

        Randomizes transaction inclusion to model real-world uncertainty.

        Args:
            data: Vectorized market data
            n_simulations: Number of Monte Carlo simulations
            config: Signal configuration
            inclusion_prob: Probability of transaction inclusion

        Returns:
            MonteCarloResults with distribution statistics
        """
        cfg = config or self.config

        returns, volume_spikes = self._compute_indicators(data)

        entry_signals, _ = _generate_entry_signals(
            returns,
            volume_spikes,
            data.liquidities,
            min_return=cfg.min_price_change_pct,
            min_volume_spike=cfg.min_volume_spike,
            min_liquidity=cfg.min_liquidity_usd,
        )

        initial_capital = 10.0  # 10 SOL

        final_equities, trade_counts = _monte_carlo_simulation(
            data.prices,
            entry_signals,
            n_simulations,
            initial_capital=initial_capital,
            position_size_pct=cfg.max_position_pct,
            take_profit_pct=cfg.take_profit_pct,
            stop_loss_pct=cfg.stop_loss_pct,
            max_hold=cfg.max_hold_periods,
            inclusion_prob=inclusion_prob,
        )

        returns_pct = (final_equities - initial_capital) / initial_capital * 100

        return MonteCarloResults(
            mean_return_pct=float(np.mean(returns_pct)),
            std_return_pct=float(np.std(returns_pct)),
            percentile_5=float(np.percentile(returns_pct, 5)),
            percentile_25=float(np.percentile(returns_pct, 25)),
            median_return_pct=float(np.percentile(returns_pct, 50)),
            percentile_75=float(np.percentile(returns_pct, 75)),
            percentile_95=float(np.percentile(returns_pct, 95)),
            probability_of_profit=float(np.mean(returns_pct > 0)),
            avg_trades=float(np.mean(trade_counts)),
        )

    def run_with_signal_type(
        self,
        data: VectorizedData,
        signal_type: str = "momentum",
        config: SignalConfig | None = None,
    ) -> VectorizedResults:
        """Run backtest with specified signal type.

        Args:
            data: Vectorized market data
            signal_type: One of "momentum", "rsi", "breakout"
            config: Signal configuration

        Returns:
            VectorizedResults
        """
        cfg = config or self.config

        if signal_type == "rsi":
            entry_signals = _compute_rsi_signals(data.prices)
            entry_bool = entry_signals == 1
            signal_strengths = np.abs(entry_signals).astype(np.float64)
        elif signal_type == "breakout":
            entry_signals = _compute_breakout_signals(data.prices)
            entry_bool = entry_signals == 1
            signal_strengths = np.abs(entry_signals).astype(np.float64)
        else:  # momentum (default)
            returns, volume_spikes = self._compute_indicators(data)
            entry_bool, signal_strengths = _generate_entry_signals(
                returns,
                volume_spikes,
                data.liquidities,
                min_return=cfg.min_price_change_pct,
                min_volume_spike=cfg.min_volume_spike,
                min_liquidity=cfg.min_liquidity_usd,
            )

        (
            n_trades,
            n_wins,
            n_losses,
            total_pnl,
            max_drawdown_pct,
            gross_wins_pct,
            gross_losses_pct,
        ) = _simulate_trades_fast(
            data.prices,
            entry_bool,
            take_profit_pct=cfg.take_profit_pct,
            stop_loss_pct=cfg.stop_loss_pct,
            max_hold=cfg.max_hold_periods,
        )
        expectancy_pct, estimated_profit_factor, profit_quality_score = (
            _compute_profit_quality_metrics(
                n_trades=n_trades,
                n_wins=n_wins,
                n_losses=n_losses,
                gross_wins_pct=gross_wins_pct,
                gross_losses_pct=gross_losses_pct,
                max_drawdown_pct=max_drawdown_pct,
                take_profit_pct=cfg.take_profit_pct,
                stop_loss_pct=cfg.stop_loss_pct,
            )
        )

        token_returns = {}
        for i, token_id in enumerate(data.token_ids):
            token_entries = entry_bool[i].sum()
            if token_entries > 0:
                token_returns[token_id] = float(total_pnl / max(1, n_trades) * token_entries)

        return VectorizedResults(
            entry_signals=entry_bool,
            exit_signals=np.zeros_like(entry_bool),
            signal_strengths=signal_strengths,
            n_trades=n_trades,
            n_wins=n_wins,
            n_losses=n_losses,
            total_pnl_pct=total_pnl,
            max_drawdown_pct=max_drawdown_pct,
            expectancy_pct=expectancy_pct,
            estimated_profit_factor=estimated_profit_factor,
            profit_quality_score=profit_quality_score,
            token_returns=token_returns,
            compute_backend=self.compute_backend,
        )


def prepare_data_from_ohlcv(
    ohlcv_data: dict[str, list[dict[str, Any]]],
) -> VectorizedData:
    """Convert OHLCV data to vectorized format.

    Args:
        ohlcv_data: Dictionary mapping token_id to list of OHLCV bars

    Returns:
        VectorizedData ready for backtesting
    """
    token_ids = list(ohlcv_data.keys())

    if not token_ids:
        raise ValueError("No token data provided")

    # Find common timestamp range
    all_timestamps = set()
    for bars in ohlcv_data.values():
        for bar in bars:
            all_timestamps.add(bar["timestamp_ms"])

    timestamps = sorted(all_timestamps)
    n_timepoints = len(timestamps)
    n_tokens = len(token_ids)

    # Create timestamp index mapping
    ts_to_idx = {ts: idx for idx, ts in enumerate(timestamps)}

    # Allocate arrays
    prices = np.zeros((n_tokens, n_timepoints), dtype=np.float64)
    volumes = np.zeros((n_tokens, n_timepoints), dtype=np.float64)
    liquidities = np.zeros((n_tokens, n_timepoints), dtype=np.float64)

    # Fill arrays
    for i, token_id in enumerate(token_ids):
        for bar in ohlcv_data[token_id]:
            t = ts_to_idx[bar["timestamp_ms"]]
            prices[i, t] = bar.get("close", 0)
            volumes[i, t] = bar.get("volume", 0)
            liquidities[i, t] = bar.get("liquidity_usd", 10000)

    # Forward fill missing values
    for i in range(n_tokens):
        last_price = 0.0
        last_liq = 10000.0
        for t in range(n_timepoints):
            if prices[i, t] == 0 and last_price > 0:
                prices[i, t] = last_price
            else:
                last_price = prices[i, t]

            if liquidities[i, t] == 0:
                liquidities[i, t] = last_liq
            else:
                last_liq = liquidities[i, t]

    return VectorizedData(
        timestamps=np.array([timestamps] * n_tokens, dtype=np.int64),
        prices=prices,
        volumes=volumes,
        liquidities=liquidities,
        token_ids=token_ids,
    )
