"""
Shadow Trading Optimizer

Optimizes trading parameters based on shadow trade history using:
- Statistical analysis of exit reasons
- Bayesian/evolutionary optimization of TP/SL
- Honeypot threshold calibration
- Kelly criterion position sizing
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from scipy.optimize import differential_evolution

if TYPE_CHECKING:
    from ..storage import DatabaseManager

logger = logging.getLogger(__name__)


class OptimizationTarget(Enum):
    """Target metric for optimization."""

    SHARPE = "sharpe"
    SORTINO = "sortino"
    EXPECTANCY = "expectancy"
    TOTAL_PNL = "total_pnl"
    WIN_RATE = "win_rate"
    CALMAR = "calmar"


@dataclass
class ShadowOptimizerConfig:
    """Configuration for shadow optimizer."""

    # Optimization target
    target: OptimizationTarget = OptimizationTarget.SHARPE

    # Parameter bounds (relative to current)
    param_change_limit_pct: float = 30.0

    # Minimum data requirements
    min_trades: int = 50
    min_honeypots: int = 5  # Need some honeypots to calibrate threshold

    # Optimization settings
    max_iterations: int = 100
    population_size: int = 15
    mutation_factor: float = 0.8
    crossover_prob: float = 0.7

    # Constraints
    min_take_profit_bps: int = 100
    max_take_profit_bps: int = 2000
    min_stop_loss_bps: int = -5000
    max_stop_loss_bps: int = -100
    min_honeypot_threshold: float = 0.2
    max_honeypot_threshold: float = 0.8

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShadowOptimizerConfig":
        """Create from dictionary."""
        target_str = data.get("target", "sharpe")
        try:
            target = OptimizationTarget(target_str)
        except ValueError:
            target = OptimizationTarget.SHARPE

        return cls(
            target=target,
            param_change_limit_pct=data.get("param_change_limit_pct", 30.0),
            min_trades=data.get("min_trades", 50),
            min_honeypots=data.get("min_honeypots", 5),
            max_iterations=data.get("max_iterations", 100),
        )


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""

    # Optimized parameters
    take_profit_bps: int
    stop_loss_bps: int
    trailing_stop_bps: int | None
    max_honeypot_score: float
    kelly_fraction: float

    # Expected performance
    expected_sharpe: float
    expected_win_rate: float
    expected_pnl_bps: float
    expected_max_drawdown: float

    # Optimization metadata
    confidence: float
    iterations_used: int
    convergence_achieved: bool
    improvement_vs_current: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "take_profit_bps": self.take_profit_bps,
            "stop_loss_bps": self.stop_loss_bps,
            "trailing_stop_bps": self.trailing_stop_bps,
            "max_honeypot_score": self.max_honeypot_score,
            "kelly_fraction": self.kelly_fraction,
            "expected_sharpe": self.expected_sharpe,
            "expected_win_rate": self.expected_win_rate,
            "expected_pnl_bps": self.expected_pnl_bps,
            "expected_max_drawdown": self.expected_max_drawdown,
            "confidence": self.confidence,
            "iterations_used": self.iterations_used,
            "convergence_achieved": self.convergence_achieved,
            "improvement_vs_current": self.improvement_vs_current,
        }


@dataclass
class CurrentParams:
    """Current trading parameters."""

    take_profit_bps: int = 500
    stop_loss_bps: int = -1500
    trailing_stop_bps: int | None = 300
    max_honeypot_score: float = 0.5
    kelly_fraction: float = 0.25


class ShadowOptimizer:
    """
    Optimizer for shadow trading parameters.

    Uses differential evolution to find optimal TP/SL/honeypot parameters
    based on historical shadow trade data.
    """

    def __init__(
        self,
        config: ShadowOptimizerConfig | None = None,
        db: Optional["DatabaseManager"] = None,
    ):
        self.config = config or ShadowOptimizerConfig()
        self.db = db

        # Cache for optimization
        self._trade_cache: list[dict[str, Any]] = []
        self._current_params: CurrentParams = CurrentParams()

    def set_current_params(self, params: dict[str, Any]) -> None:
        """Set current parameters for optimization bounds."""
        self._current_params = CurrentParams(
            take_profit_bps=params.get("take_profit_bps", 500),
            stop_loss_bps=params.get("stop_loss_bps", -1500),
            trailing_stop_bps=params.get("trailing_stop_bps"),
            max_honeypot_score=params.get("max_honeypot_score", 0.5),
            kelly_fraction=params.get("kelly_fraction", 0.25),
        )

    def optimize(
        self,
        trades: list[dict[str, Any]],
        current_metrics: dict[str, Any],
    ) -> OptimizationResult:
        """
        Run optimization to find better parameters.

        Args:
            trades: List of shadow trade records with fields:
                - net_pnl_bps: Realized P&L in basis points
                - peak_pnl_bps: Peak P&L during hold
                - trough_pnl_bps: Minimum P&L during hold
                - honeypot_score: Pre-trade honeypot score
                - was_honeypot: Whether trade was honeypot
                - exit_reason: Why trade was closed
                - total_fee_bps: Total fees in bps
            current_metrics: Current performance metrics

        Returns:
            OptimizationResult with optimized parameters
        """
        if len(trades) < self.config.min_trades:
            logger.warning(f"Insufficient trades ({len(trades)}) for optimization")
            return self._default_result()

        self._trade_cache = trades

        # Calculate bounds based on current params and limits
        bounds = self._get_bounds()

        # Run differential evolution
        result = differential_evolution(
            self._objective,
            bounds=bounds,
            maxiter=self.config.max_iterations,
            popsize=self.config.population_size,
            mutation=self.config.mutation_factor,
            recombination=self.config.crossover_prob,
            seed=42,
            workers=1,  # Thread-safe
            updating="deferred",
        )

        best_tp, best_sl, best_hp = result.x
        best_metrics = self._simulate_with_params(trades, int(best_tp), int(best_sl), best_hp)

        # Calculate Kelly fraction from win rate
        kelly = self._calculate_kelly(best_metrics)

        # Calculate improvement
        current_objective = self._calculate_objective_value(current_metrics)
        new_objective = -result.fun  # Negated because we minimize
        improvement = (
            (new_objective - current_objective) / abs(current_objective) * 100.0
            if current_objective != 0
            else 0.0
        )

        # Calculate confidence based on sample size and convergence
        confidence = self._calculate_confidence(len(trades), result.success, improvement)

        return OptimizationResult(
            take_profit_bps=int(best_tp),
            stop_loss_bps=int(best_sl),
            trailing_stop_bps=self._current_params.trailing_stop_bps,
            max_honeypot_score=best_hp,
            kelly_fraction=kelly,
            expected_sharpe=best_metrics.get("sharpe", 0.0),
            expected_win_rate=best_metrics.get("win_rate", 0.0),
            expected_pnl_bps=best_metrics.get("total_pnl_bps", 0.0),
            expected_max_drawdown=best_metrics.get("max_drawdown", 0.0),
            confidence=confidence,
            iterations_used=result.nit,
            convergence_achieved=result.success,
            improvement_vs_current=improvement,
        )

    def _get_bounds(self) -> list[tuple]:
        """Get optimization bounds based on config and current params."""
        limit = self.config.param_change_limit_pct / 100.0

        # Take profit bounds
        tp_low = max(
            self.config.min_take_profit_bps, int(self._current_params.take_profit_bps * (1 - limit))
        )
        tp_high = min(
            self.config.max_take_profit_bps, int(self._current_params.take_profit_bps * (1 + limit))
        )

        # Stop loss bounds (remember it's negative)
        sl_low = max(
            self.config.min_stop_loss_bps,
            int(self._current_params.stop_loss_bps * (1 + limit)),  # More negative
        )
        sl_high = min(
            self.config.max_stop_loss_bps,
            int(self._current_params.stop_loss_bps * (1 - limit)),  # Less negative
        )

        # Honeypot threshold bounds
        hp_low = max(
            self.config.min_honeypot_threshold,
            self._current_params.max_honeypot_score * (1 - limit),
        )
        hp_high = min(
            self.config.max_honeypot_threshold,
            self._current_params.max_honeypot_score * (1 + limit),
        )

        return [
            (tp_low, tp_high),
            (sl_low, sl_high),
            (hp_low, hp_high),
        ]

    def _objective(self, x: np.ndarray) -> float:
        """Objective function to minimize (returns negative of target)."""
        tp, sl, hp = x
        metrics = self._simulate_with_params(self._trade_cache, int(tp), int(sl), hp)
        return -self._calculate_objective_value(metrics)

    def _calculate_objective_value(self, metrics: dict[str, Any]) -> float:
        """Calculate objective value based on target."""
        target = self.config.target

        if target == OptimizationTarget.SHARPE:
            return metrics.get("sharpe", 0.0)
        elif target == OptimizationTarget.SORTINO:
            return metrics.get("sortino", 0.0)
        elif target == OptimizationTarget.EXPECTANCY:
            return metrics.get("expectancy_bps", 0.0)
        elif target == OptimizationTarget.TOTAL_PNL:
            return metrics.get("total_pnl_bps", 0.0)
        elif target == OptimizationTarget.WIN_RATE:
            return metrics.get("win_rate", 0.0)
        elif target == OptimizationTarget.CALMAR:
            sharpe = metrics.get("sharpe", 0.0)
            dd = metrics.get("max_drawdown", 1.0)
            return sharpe / max(dd, 0.01)
        else:
            return metrics.get("sharpe", 0.0)

    def _simulate_with_params(
        self,
        trades: list[dict[str, Any]],
        take_profit_bps: int,
        stop_loss_bps: int,
        max_honeypot_score: float,
    ) -> dict[str, Any]:
        """
        Simulate trade outcomes with different parameters.

        Uses peak/trough data to determine what would have happened
        with different TP/SL thresholds.
        """
        simulated_pnls = []
        filtered_count = 0

        for t in trades:
            # Filter by honeypot score
            if t.get("honeypot_score", 0) > max_honeypot_score:
                filtered_count += 1
                continue

            peak_bps = t.get("peak_pnl_bps", t.get("net_pnl_bps", 0))
            trough_bps = t.get("trough_pnl_bps", t.get("net_pnl_bps", 0))
            fee_bps = t.get("total_fee_bps", 0)

            # Simulate what would have happened with new TP/SL
            # Check if TP would have been hit first
            if peak_bps >= take_profit_bps:
                # Would have exited at TP
                sim_pnl = take_profit_bps - fee_bps
            elif trough_bps <= stop_loss_bps:
                # Would have hit SL
                sim_pnl = stop_loss_bps - fee_bps
            else:
                # Same exit as original (timeout or other)
                sim_pnl = t.get("net_pnl_bps", 0)

            simulated_pnls.append(sim_pnl)

        if not simulated_pnls:
            return {
                "sharpe": 0.0,
                "sortino": 0.0,
                "win_rate": 0.0,
                "total_pnl_bps": 0.0,
                "expectancy_bps": 0.0,
                "max_drawdown": 0.0,
                "trades_taken": 0,
                "trades_filtered": filtered_count,
            }

        pnls = np.array(simulated_pnls)
        returns = pnls / 10000.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns) if len(returns) > 1 else 1.0

        # Sharpe (annualized assuming 100 trades/day)
        annualization = np.sqrt(365 * 100)
        sharpe = (mean_ret / std_ret) * annualization if std_ret > 0 else 0.0

        # Sortino
        downside = returns[returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1.0
        sortino = (mean_ret / downside_std) * annualization if downside_std > 0 else float("inf")

        # Win rate
        win_rate = np.sum(pnls > 0) / len(pnls) * 100.0

        # Expectancy
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        expectancy = (win_rate / 100.0) * avg_win + (1 - win_rate / 100.0) * avg_loss

        # Max drawdown
        equity = np.cumsum(returns)
        running_max = np.maximum.accumulate(equity)
        drawdown = running_max - equity
        max_drawdown = np.max(drawdown) * 100.0 if len(drawdown) > 0 else 0.0

        return {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "win_rate": float(win_rate),
            "total_pnl_bps": float(np.sum(pnls)),
            "expectancy_bps": float(expectancy),
            "max_drawdown": float(max_drawdown),
            "trades_taken": len(simulated_pnls),
            "trades_filtered": filtered_count,
        }

    def _calculate_kelly(self, metrics: dict[str, Any]) -> float:
        """Calculate Kelly fraction for position sizing."""
        win_rate = metrics.get("win_rate", 0.0) / 100.0

        if win_rate <= 0 or win_rate >= 1:
            return 0.1  # Default conservative

        # Estimate win/loss ratio from total P&L and win rate
        total_pnl = metrics.get("total_pnl_bps", 0.0)
        trades = metrics.get("trades_taken", 1)

        if total_pnl <= 0:
            return 0.1

        # Approximate win/loss ratio
        avg_trade = total_pnl / trades
        # Assume wins are ~2x the average (rough approximation)
        est_win_loss_ratio = max(1.0, 2.0 * avg_trade / abs(avg_trade + 1))

        # Kelly = W - (1-W)/R
        kelly = win_rate - (1 - win_rate) / est_win_loss_ratio

        # Use fractional Kelly (25%)
        kelly *= 0.25

        return max(0.05, min(0.5, kelly))

    def _calculate_confidence(
        self,
        sample_size: int,
        converged: bool,
        improvement: float,
    ) -> float:
        """Calculate confidence in optimization result."""
        # Base confidence from sample size
        sample_conf = min(1.0, sample_size / 200.0)

        # Convergence bonus
        conv_bonus = 0.1 if converged else 0.0

        # Improvement sanity check (huge improvements are suspicious)
        if abs(improvement) > 100:
            improvement_penalty = 0.2
        else:
            improvement_penalty = 0.0

        confidence = sample_conf * 0.6 + conv_bonus + 0.3 - improvement_penalty
        return max(0.1, min(1.0, confidence))

    def _default_result(self) -> OptimizationResult:
        """Return default result when optimization can't run."""
        return OptimizationResult(
            take_profit_bps=self._current_params.take_profit_bps,
            stop_loss_bps=self._current_params.stop_loss_bps,
            trailing_stop_bps=self._current_params.trailing_stop_bps,
            max_honeypot_score=self._current_params.max_honeypot_score,
            kelly_fraction=self._current_params.kelly_fraction,
            expected_sharpe=0.0,
            expected_win_rate=0.0,
            expected_pnl_bps=0.0,
            expected_max_drawdown=0.0,
            confidence=0.0,
            iterations_used=0,
            convergence_achieved=False,
            improvement_vs_current=0.0,
        )


def analyze_exit_performance(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyze trade performance by exit reason.

    Returns statistics for each exit type.
    """
    by_exit: dict[str, dict[str, Any]] = {}

    for t in trades:
        reason = t.get("exit_reason", "unknown")
        if reason not in by_exit:
            by_exit[reason] = {
                "count": 0,
                "total_pnl": 0.0,
                "wins": 0,
                "pnls": [],
            }
        stats = by_exit[reason]
        stats["count"] += 1
        pnl = t.get("net_pnl_bps", 0)
        stats["total_pnl"] += pnl
        stats["pnls"].append(pnl)
        if pnl > 0:
            stats["wins"] += 1

    # Calculate averages
    for _reason, stats in by_exit.items():
        if stats["count"] > 0:
            stats["avg_pnl"] = stats["total_pnl"] / stats["count"]
            stats["win_rate"] = stats["wins"] / stats["count"] * 100.0
            stats["std_pnl"] = float(np.std(stats["pnls"])) if len(stats["pnls"]) > 1 else 0.0
        del stats["pnls"]  # Don't need in output

    return by_exit


def analyze_honeypot_distribution(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyze honeypot score distribution.

    Helps calibrate the honeypot threshold.
    """
    honeypot_trades = [t for t in trades if t.get("was_honeypot", False)]
    non_honeypot_trades = [t for t in trades if not t.get("was_honeypot", False)]

    hp_scores = [t.get("honeypot_score", 0) for t in honeypot_trades]
    non_hp_scores = [t.get("honeypot_score", 0) for t in non_honeypot_trades]

    return {
        "honeypot_count": len(honeypot_trades),
        "non_honeypot_count": len(non_honeypot_trades),
        "honeypot_rate": len(honeypot_trades) / len(trades) * 100.0 if trades else 0.0,
        "honeypot_avg_score": float(np.mean(hp_scores)) if hp_scores else 0.0,
        "honeypot_min_score": float(np.min(hp_scores)) if hp_scores else 0.0,
        "non_honeypot_avg_score": float(np.mean(non_hp_scores)) if non_hp_scores else 0.0,
        "non_honeypot_max_score": float(np.max(non_hp_scores)) if non_hp_scores else 0.0,
        "suggested_threshold": float(np.percentile(hp_scores, 25)) if len(hp_scores) >= 4 else 0.5,
    }
