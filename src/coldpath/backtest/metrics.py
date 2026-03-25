"""
Backtest metrics calculation.

Computes Sharpe ratio, win rate, drawdown, etc.
Enhanced with anti-overfitting metrics for model validation.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AntiOverfitMetrics:
    """Anti-overfitting validation metrics."""

    # Generalization
    train_sharpe: float = 0.0
    val_sharpe: float = 0.0
    generalization_ratio: float = 0.0  # val_sharpe / train_sharpe

    # Out-of-sample
    oos_sharpe: float = 0.0
    oos_win_rate: float = 0.0

    # Walk-forward
    walk_forward_sharpe_mean: float = 0.0
    walk_forward_sharpe_std: float = 0.0
    walk_forward_windows: int = 0

    # Parameter sensitivity
    parameter_sensitivity: float = 0.0
    most_sensitive_param: str = ""

    # Monte Carlo
    monte_carlo_sharpe_median: float = 0.0
    monte_carlo_sharpe_std: float = 0.0
    monte_carlo_runs: int = 0

    # Overall validation
    all_tests_passed: bool = False
    deployment_ready: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "train_sharpe": self.train_sharpe,
            "val_sharpe": self.val_sharpe,
            "generalization_ratio": self.generalization_ratio,
            "oos_sharpe": self.oos_sharpe,
            "oos_win_rate": self.oos_win_rate,
            "walk_forward_sharpe_mean": self.walk_forward_sharpe_mean,
            "walk_forward_sharpe_std": self.walk_forward_sharpe_std,
            "walk_forward_windows": self.walk_forward_windows,
            "parameter_sensitivity": self.parameter_sensitivity,
            "most_sensitive_param": self.most_sensitive_param,
            "monte_carlo_sharpe_median": self.monte_carlo_sharpe_median,
            "monte_carlo_sharpe_std": self.monte_carlo_sharpe_std,
            "monte_carlo_runs": self.monte_carlo_runs,
            "all_tests_passed": self.all_tests_passed,
            "deployment_ready": self.deployment_ready,
        }

    def check_thresholds(
        self,
        min_generalization_ratio: float = 0.70,
        min_oos_sharpe: float = 1.5,
        min_walk_forward_sharpe: float = 1.5,
        max_walk_forward_std: float = 0.5,
        max_sensitivity: float = 0.20,
        min_monte_carlo_sharpe: float = 1.5,
    ) -> bool:
        """Check if all thresholds are met."""
        passed = (
            self.generalization_ratio >= min_generalization_ratio
            and self.oos_sharpe >= min_oos_sharpe
            and self.walk_forward_sharpe_mean >= min_walk_forward_sharpe
            and self.walk_forward_sharpe_std <= max_walk_forward_std
            and self.parameter_sensitivity <= max_sensitivity
            and self.monte_carlo_sharpe_median >= min_monte_carlo_sharpe
        )
        self.all_tests_passed = passed
        self.deployment_ready = passed and self.val_sharpe >= 1.5
        return passed


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics with anti-overfitting support."""

    # Returns
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float

    # Risk
    max_drawdown_pct: float
    avg_drawdown_pct: float
    volatility_pct: float

    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float

    # Fill stats
    fill_rate_pct: float
    avg_slippage_bps: float

    # Rug stats
    rugs_avoided: int
    rugs_hit: int
    rug_loss_sol: float

    # Anti-overfit metrics (optional)
    anti_overfit: AntiOverfitMetrics | None = None

    # Target metrics tracking
    meets_win_rate_target: bool = False
    meets_sharpe_target: bool = False
    meets_drawdown_target: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "total_return_pct": self.total_return_pct,
            "annualized_return_pct": self.annualized_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "avg_drawdown_pct": self.avg_drawdown_pct,
            "volatility_pct": self.volatility_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": self.win_rate_pct,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "profit_factor": self.profit_factor,
            "fill_rate_pct": self.fill_rate_pct,
            "avg_slippage_bps": self.avg_slippage_bps,
            "rugs_avoided": self.rugs_avoided,
            "rugs_hit": self.rugs_hit,
            "rug_loss_sol": self.rug_loss_sol,
            "meets_win_rate_target": self.meets_win_rate_target,
            "meets_sharpe_target": self.meets_sharpe_target,
            "meets_drawdown_target": self.meets_drawdown_target,
        }
        if self.anti_overfit:
            result["anti_overfit"] = self.anti_overfit.to_dict()
        return result

    def check_targets(
        self,
        target_win_rate: float = 55.0,
        target_sharpe: float = 2.0,
        target_max_drawdown: float = 25.0,
    ) -> bool:
        """Check if metrics meet target thresholds."""
        self.meets_win_rate_target = self.win_rate_pct >= target_win_rate
        self.meets_sharpe_target = self.sharpe_ratio >= target_sharpe
        self.meets_drawdown_target = self.max_drawdown_pct <= target_max_drawdown

        return (
            self.meets_win_rate_target and self.meets_sharpe_target and self.meets_drawdown_target
        )


class MetricsEngine:
    """Calculates backtest metrics from trade history."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def calculate(
        self,
        trades: list[dict],
        initial_capital: float,
    ) -> BacktestMetrics:
        """Calculate all metrics from trade list."""
        if not trades:
            return self._empty_metrics()

        # Extract PnL series
        pnls = [t.get("pnl_sol", 0) for t in trades]
        returns = [t.get("pnl_pct", 0) for t in trades]

        # Calculate equity curve
        equity = [initial_capital]
        for pnl in pnls:
            equity.append(equity[-1] + pnl)

        # Returns
        total_return = (equity[-1] - initial_capital) / initial_capital

        # Calculate actual trading period from timestamps
        first_trade_ts = trades[0].get("timestamp_ms", 0)
        last_trade_ts = trades[-1].get("timestamp_ms", 0)

        # Convert ms timestamps to days; fall back to trade count if timestamps unavailable
        if first_trade_ts > 0 and last_trade_ts > 0 and last_trade_ts > first_trade_ts:
            trading_days = max(1, (last_trade_ts - first_trade_ts) / (1000 * 86400))
        else:
            # Fallback: assume trades span 1 day per trade, capped at a
            # conservative estimate.  Using len(trades) directly would
            # make trades_per_day == 1 which is plausible-ish, but it
            # silently inflates Sharpe when many trades actually happen
            # in a short window.  Use a conservative 1 day minimum —
            # callers should provide timestamp_ms for accuracy.
            logger.warning(
                "Sharpe fallback: trades lack timestamp_ms — assuming 1 trading day for %d trades",
                len(trades),
            )
            trading_days = max(1, len(trades) / 10)  # assume ~10 trades/day

        trades_per_day = len(trades) / trading_days

        # Annualize correctly based on actual trading period
        annualized_return = total_return * (365 / max(trading_days, 1))

        # Annualization factor based on actual trade frequency
        annual_periods = trades_per_day * 365

        # Sharpe ratio with correct annualization
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) * annual_periods - self.risk_free_rate) / (
                np.std(returns) * np.sqrt(annual_periods)
            )
        else:
            sharpe = 0.0

        # Sortino (downside deviation)
        downside = [r for r in returns if r < 0]
        if len(downside) > 1:
            downside_std = np.std(downside)
            sortino = (np.mean(returns) * annual_periods - self.risk_free_rate) / (
                downside_std * np.sqrt(annual_periods) if downside_std > 0 else 1
            )
        else:
            sortino = sharpe

        # Drawdown
        peak = equity[0]
        drawdowns = []
        for e in equity:
            if e > peak:
                peak = e
            drawdown = (peak - e) / peak if peak > 0 else 0
            drawdowns.append(drawdown)

        max_drawdown = max(drawdowns) * 100
        avg_drawdown = np.mean(drawdowns) * 100

        # Volatility
        volatility = np.std(returns) * 100 if returns else 0

        # Trade stats
        winning = [r for r in returns if r > 0]
        losing = [r for r in returns if r < 0]

        win_rate = len(winning) / len(returns) * 100 if returns else 0
        avg_win = np.mean(winning) * 100 if winning else 0
        avg_loss = np.mean(losing) * 100 if losing else 0

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        # Cap at 999 to avoid inf when no losing trades (common in early paper trading)
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        )

        # Fill stats
        filled = sum(1 for t in trades if t.get("included", True))
        fill_rate = filled / len(trades) * 100 if trades else 0
        slippages = [t.get("realized_slippage_bps", 0) for t in trades if t.get("included")]
        avg_slippage = np.mean(slippages) if slippages else 0

        # Rug stats
        rugs_avoided = sum(1 for t in trades if t.get("rug_avoided", False))
        rugs_hit = sum(1 for t in trades if t.get("rug_hit", False))
        rug_loss = sum(t.get("pnl_sol", 0) for t in trades if t.get("rug_hit", False))

        return BacktestMetrics(
            total_return_pct=total_return * 100,
            annualized_return_pct=annualized_return * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_drawdown,
            avg_drawdown_pct=avg_drawdown,
            volatility_pct=volatility,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate_pct=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            fill_rate_pct=fill_rate,
            avg_slippage_bps=avg_slippage,
            rugs_avoided=rugs_avoided,
            rugs_hit=rugs_hit,
            rug_loss_sol=abs(rug_loss),
        )

    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics when no trades."""
        return BacktestMetrics(
            total_return_pct=0,
            annualized_return_pct=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown_pct=0,
            avg_drawdown_pct=0,
            volatility_pct=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0,
            avg_win_pct=0,
            avg_loss_pct=0,
            profit_factor=0,
            fill_rate_pct=0,
            avg_slippage_bps=0,
            rugs_avoided=0,
            rugs_hit=0,
            rug_loss_sol=0,
        )

    def calculate_with_validation(
        self,
        train_trades: list[dict],
        val_trades: list[dict],
        test_trades: list[dict],
        initial_capital: float,
    ) -> BacktestMetrics:
        """Calculate metrics with anti-overfitting validation.

        Args:
            train_trades: Trades from training period
            val_trades: Trades from validation period
            test_trades: Trades from test period (last 30 days)
            initial_capital: Initial capital

        Returns:
            BacktestMetrics with anti-overfit metrics included
        """
        # Calculate main metrics on combined data
        all_trades = train_trades + val_trades + test_trades
        metrics = self.calculate(all_trades, initial_capital)

        # Calculate separate metrics for each period
        train_metrics = self.calculate(train_trades, initial_capital)
        val_metrics = self.calculate(val_trades, initial_capital)
        test_metrics = self.calculate(test_trades, initial_capital)

        # Calculate generalization ratio
        gen_ratio = 0.0
        if train_metrics.sharpe_ratio > 0:
            gen_ratio = val_metrics.sharpe_ratio / train_metrics.sharpe_ratio

        # Populate anti-overfit metrics
        anti_overfit = AntiOverfitMetrics(
            train_sharpe=train_metrics.sharpe_ratio,
            val_sharpe=val_metrics.sharpe_ratio,
            generalization_ratio=gen_ratio,
            oos_sharpe=test_metrics.sharpe_ratio,
            oos_win_rate=test_metrics.win_rate_pct,
        )

        metrics.anti_overfit = anti_overfit

        return metrics


def calculate_sharpe_from_returns(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
    trading_days: int | None = None,
    trades_per_day: float | None = None,
) -> float:
    """Calculate Sharpe ratio from returns array.

    Uses actual trading frequency for correct annualization when
    trading_days and trades_per_day are provided.  Falls back to
    the legacy 365-day calendar assumption otherwise.

    Args:
        returns: Array of period returns
        risk_free_rate: Annualized risk-free rate
        annualize: Whether to annualize the ratio
        trading_days: Actual number of calendar days the trades span
        trades_per_day: Average number of trades per day

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    # Determine annualization factor from actual trade frequency
    if trades_per_day is not None and trades_per_day > 0:
        annual_periods = trades_per_day * 365
    else:
        # Legacy fallback: assume one return observation per day
        annual_periods = 365.0

    # Per-period risk-free rate
    per_period_rf = risk_free_rate / annual_periods if annual_periods > 0 else 0.0
    excess_return = mean_return - per_period_rf

    sharpe = excess_return / std_return

    if annualize:
        sharpe *= np.sqrt(annual_periods)

    return float(sharpe)


def calculate_sortino_from_returns(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """Calculate Sortino ratio from returns array.

    Args:
        returns: Array of period returns
        risk_free_rate: Annualized risk-free rate
        annualize: Whether to annualize the ratio

    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    mean_return = np.mean(returns)
    daily_rf = risk_free_rate / 365

    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return float("inf") if mean_return > daily_rf else 0.0

    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0.0

    sortino = (mean_return - daily_rf) / downside_std

    if annualize:
        sortino *= np.sqrt(365)

    return float(sortino)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Array of equity values

    Returns:
        Maximum drawdown as decimal (e.g., 0.25 for 25%)
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return float(max_dd)
