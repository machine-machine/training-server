"""
Parameter Tuner for Shadow Trading

Implements various optimization algorithms:
- Grid search for quick exploration
- Bayesian optimization for efficient parameter search
- Local heuristics based on exit analysis
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

logger = logging.getLogger(__name__)


@dataclass
class TradingParams:
    """Trading parameters that can be tuned."""

    # Entry filters
    min_liquidity_usd: float = 5000.0
    max_fdv_usd: float = 5_000_000.0
    max_honeypot_score: float = 0.5

    # Exit conditions (aligned with strategy: 25% TP / 10% SL = 2.5:1 R/R)
    take_profit_bps: int = 2500  # 25% (aligned with user_strategy)
    stop_loss_bps: int = -1000  # -10% (aligned with user_strategy)
    trailing_stop_bps: int | None = 1200  # 12% (aligned)
    max_hold_time_ms: int = 300_000

    # Position sizing
    base_position_sol: float = 0.05
    kelly_fraction: float = 0.25

    # Slippage (realistic for memecoins)
    max_entry_slippage_bps: int = 300  # Restored: realistic entry slippage
    max_exit_slippage_bps: int = 300  # Restored: realistic exit slippage

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_liquidity_usd": self.min_liquidity_usd,
            "max_fdv_usd": self.max_fdv_usd,
            "max_honeypot_score": self.max_honeypot_score,
            "take_profit_bps": self.take_profit_bps,
            "stop_loss_bps": self.stop_loss_bps,
            "trailing_stop_bps": self.trailing_stop_bps,
            "max_hold_time_ms": self.max_hold_time_ms,
            "base_position_sol": self.base_position_sol,
            "kelly_fraction": self.kelly_fraction,
            "max_entry_slippage_bps": self.max_entry_slippage_bps,
            "max_exit_slippage_bps": self.max_exit_slippage_bps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradingParams":
        """Create from dictionary."""
        return cls(
            min_liquidity_usd=data.get("min_liquidity_usd", 5000.0),
            max_fdv_usd=data.get("max_fdv_usd", 5_000_000.0),
            max_honeypot_score=data.get("max_honeypot_score", 0.5),
            take_profit_bps=data.get("take_profit_bps", 500),
            stop_loss_bps=data.get("stop_loss_bps", -1500),
            trailing_stop_bps=data.get("trailing_stop_bps"),
            max_hold_time_ms=data.get("max_hold_time_ms", 300_000),
            base_position_sol=data.get("base_position_sol", 0.05),
            kelly_fraction=data.get("kelly_fraction", 0.25),
            max_entry_slippage_bps=data.get("max_entry_slippage_bps", 300),
            max_exit_slippage_bps=data.get("max_exit_slippage_bps", 300),
        )


@dataclass
class TuningResult:
    """Result of parameter tuning."""

    optimal_params: TradingParams
    expected_metrics: dict[str, float]
    confidence: float
    method_used: str
    iterations: int
    improvement_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "optimal_params": self.optimal_params.to_dict(),
            "expected_metrics": self.expected_metrics,
            "confidence": self.confidence,
            "method_used": self.method_used,
            "iterations": self.iterations,
            "improvement_pct": self.improvement_pct,
        }


class ParameterTuner:
    """
    Tune shadow trading parameters using various methods.

    Supports:
    - Local heuristics (fast, based on exit analysis)
    - Grid search (exhaustive)
    - Differential evolution (global optimization)
    """

    def __init__(
        self,
        current_params: TradingParams | None = None,
        change_limit_pct: float = 15.0,  # Tightened: prevent parameter drift
    ):
        self.current_params = current_params or TradingParams()
        self.change_limit_pct = change_limit_pct

    def tune_local(
        self,
        trades: list[dict[str, Any]],
        exit_analysis: dict[str, dict[str, Any]],
    ) -> TuningResult:
        """
        Fast local tuning based on exit reason analysis.

        This is a heuristic-based approach that doesn't require
        full simulation, making it suitable for frequent updates.
        """
        params = TradingParams.from_dict(self.current_params.to_dict())

        # Tune TP/SL based on exit analysis
        params = self._tune_exit_params(params, exit_analysis)

        # Tune honeypot threshold
        params = self._tune_honeypot_threshold(params, trades)

        # Tune position sizing with Kelly
        params = self._tune_position_sizing(params, trades)

        # Estimate improvement
        improvement = self._estimate_improvement(trades, params)

        return TuningResult(
            optimal_params=params,
            expected_metrics={"estimated_improvement_pct": improvement},
            confidence=0.5,  # Lower confidence for heuristic method
            method_used="local_heuristics",
            iterations=1,
            improvement_pct=improvement,
        )

    def tune_grid(
        self,
        trades: list[dict[str, Any]],
        tp_range: tuple[int, int, int] = (200, 1000, 100),
        sl_range: tuple[int, int, int] = (-2500, -500, 250),
        hp_range: tuple[float, float, float] = (0.3, 0.7, 0.1),
    ) -> TuningResult:
        """
        Grid search over parameter space.

        Slower but more thorough than local heuristics.
        """
        best_params = None
        best_sharpe = float("-inf")
        best_metrics = {}
        iterations = 0

        for tp in range(tp_range[0], tp_range[1] + 1, tp_range[2]):
            for sl in range(sl_range[0], sl_range[1] + 1, sl_range[2]):
                for hp in np.arange(hp_range[0], hp_range[1] + hp_range[2], hp_range[2]):
                    iterations += 1
                    metrics = self._simulate(trades, tp, sl, hp)

                    if metrics["sharpe"] > best_sharpe:
                        best_sharpe = metrics["sharpe"]
                        best_metrics = metrics
                        best_params = TradingParams(
                            take_profit_bps=tp,
                            stop_loss_bps=sl,
                            max_honeypot_score=hp,
                            # Copy other params from current
                            trailing_stop_bps=self.current_params.trailing_stop_bps,
                            max_hold_time_ms=self.current_params.max_hold_time_ms,
                            base_position_sol=self.current_params.base_position_sol,
                            kelly_fraction=self._calculate_kelly(metrics),
                        )

        if best_params is None:
            best_params = self.current_params

        current_metrics = self._simulate(
            trades,
            self.current_params.take_profit_bps,
            self.current_params.stop_loss_bps,
            self.current_params.max_honeypot_score,
        )

        improvement = (
            (best_sharpe - current_metrics["sharpe"]) / abs(current_metrics["sharpe"]) * 100.0
            if current_metrics["sharpe"] != 0
            else 0.0
        )

        return TuningResult(
            optimal_params=best_params,
            expected_metrics=best_metrics,
            confidence=0.7,
            method_used="grid_search",
            iterations=iterations,
            improvement_pct=improvement,
        )

    def tune_evolution(
        self,
        trades: list[dict[str, Any]],
        max_iterations: int = 50,
    ) -> TuningResult:
        """
        Differential evolution optimization.

        Most thorough but slowest method.
        """
        self._trades_cache = trades

        bounds = self._get_bounds()

        result = differential_evolution(
            self._objective,
            bounds=bounds,
            maxiter=max_iterations,
            seed=42,
            workers=1,
        )

        tp, sl, hp = result.x
        metrics = self._simulate(trades, int(tp), int(sl), hp)

        optimal_params = TradingParams(
            take_profit_bps=int(tp),
            stop_loss_bps=int(sl),
            max_honeypot_score=hp,
            trailing_stop_bps=self.current_params.trailing_stop_bps,
            max_hold_time_ms=self.current_params.max_hold_time_ms,
            base_position_sol=self.current_params.base_position_sol,
            kelly_fraction=self._calculate_kelly(metrics),
        )

        current_metrics = self._simulate(
            trades,
            self.current_params.take_profit_bps,
            self.current_params.stop_loss_bps,
            self.current_params.max_honeypot_score,
        )

        improvement = (
            (-result.fun - current_metrics["sharpe"]) / abs(current_metrics["sharpe"]) * 100.0
            if current_metrics["sharpe"] != 0
            else 0.0
        )

        return TuningResult(
            optimal_params=optimal_params,
            expected_metrics=metrics,
            confidence=0.8 if result.success else 0.6,
            method_used="differential_evolution",
            iterations=result.nit,
            improvement_pct=improvement,
        )

    def _get_bounds(self) -> list[tuple[float, float]]:
        """Get optimization bounds."""
        limit = self.change_limit_pct / 100.0

        tp = self.current_params.take_profit_bps
        sl = self.current_params.stop_loss_bps
        hp = self.current_params.max_honeypot_score

        return [
            (max(100, tp * (1 - limit)), min(2000, tp * (1 + limit))),
            (max(-5000, sl * (1 + limit)), min(-100, sl * (1 - limit))),
            (max(0.2, hp * (1 - limit)), min(0.8, hp * (1 + limit))),
        ]

    def _objective(self, x: np.ndarray) -> float:
        """Objective function for optimization."""
        tp, sl, hp = x
        metrics = self._simulate(self._trades_cache, int(tp), int(sl), hp)
        return -metrics["sharpe"]

    def _simulate(
        self,
        trades: list[dict[str, Any]],
        tp_bps: int,
        sl_bps: int,
        max_hp: float,
    ) -> dict[str, float]:
        """Simulate trades with given parameters.

        Uses probabilistic first-touch logic when both TP and SL could have been hit.
        Instead of always assuming SL hit first (which biases results negatively),
        we use a 60/40 SL-first bias based on memecoin price dynamics:
        - Memecoins tend to spike up fast and crash slower
        - TP is more likely hit first on strong momentum entries
        - 60% SL-first assumption balances conservatism with realism
        """
        pnls = []

        for t in trades:
            if t.get("honeypot_score", 0) > max_hp:
                continue

            peak = t.get("peak_pnl_bps", t.get("net_pnl_bps", 0))
            trough = t.get("trough_pnl_bps", t.get("net_pnl_bps", 0))
            fee = t.get("total_fee_bps", 0)

            tp_hit = peak >= tp_bps
            sl_hit = trough <= sl_bps

            if tp_hit and sl_hit:
                # BOTH were hit - probabilistic: 60% SL first, 40% TP first
                # This is more realistic than always assuming SL hit first
                # Use trade timestamp hash for deterministic pseudo-randomness
                trade_hash = hash(str(t.get("timestamp_ms", 0))) % 100
                if trade_hash < 60:
                    pnl = sl_bps - fee  # SL hit first (60% of cases)
                else:
                    pnl = tp_bps - fee  # TP hit first (40% of cases)
            elif sl_hit:
                pnl = sl_bps - fee
            elif tp_hit:
                pnl = tp_bps - fee
            else:
                pnl = t.get("net_pnl_bps", 0)

            pnls.append(pnl)

        if not pnls:
            return {"sharpe": -999, "win_rate": 0, "total_pnl": 0, "profit_factor": 0}

        returns = np.array(pnls) / 10000.0
        mean = np.mean(returns)
        std = np.std(returns) if len(returns) > 1 else 1.0

        sharpe = (mean / std) * np.sqrt(365 * 100) if std > 0 else 0.0
        win_rate = np.sum(np.array(pnls) > 0) / len(pnls) * 100.0

        # Calculate profit factor
        gross_wins = float(np.sum(np.array(pnls)[np.array(pnls) > 0]))
        gross_losses = float(abs(np.sum(np.array(pnls)[np.array(pnls) < 0])))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0.0

        return {
            "sharpe": float(sharpe),
            "win_rate": float(win_rate),
            "total_pnl": float(np.sum(pnls)),
            "trades": len(pnls),
            "profit_factor": float(profit_factor),
        }

    def _tune_exit_params(
        self,
        params: TradingParams,
        exit_analysis: dict[str, dict[str, Any]],
    ) -> TradingParams:
        """Tune TP/SL based on exit analysis."""

        tp_stats = exit_analysis.get("take_profit", {})
        sl_stats = exit_analysis.get("stop_loss", {})
        timeout_stats = exit_analysis.get("timeout", {})

        total_trades = sum(s.get("count", 0) for s in exit_analysis.values())
        if total_trades == 0:
            return params

        # If timeouts are profitable, hold longer
        if timeout_stats.get("count", 0) > total_trades * 0.25:
            if timeout_stats.get("avg_pnl", 0) > 0:
                params.max_hold_time_ms = int(params.max_hold_time_ms * 1.1)
                params.max_hold_time_ms = min(params.max_hold_time_ms, 600_000)

        # If SL rate is too high, adjust
        if sl_stats.get("count", 0) > total_trades * 0.4:
            avg_sl_loss = sl_stats.get("avg_pnl", params.stop_loss_bps)
            if avg_sl_loss < params.stop_loss_bps * 0.8:
                params.stop_loss_bps = int(params.stop_loss_bps * 0.9)
                params.stop_loss_bps = max(params.stop_loss_bps, -2500)

        # If TP rate is low but profitable, lower threshold
        if tp_stats.get("count", 0) < total_trades * 0.2:
            if tp_stats.get("avg_pnl", 0) > params.take_profit_bps * 1.5:
                params.take_profit_bps = int(params.take_profit_bps * 0.9)
                params.take_profit_bps = max(params.take_profit_bps, 200)

        return params

    def _tune_honeypot_threshold(
        self,
        params: TradingParams,
        trades: list[dict[str, Any]],
    ) -> TradingParams:
        """Tune honeypot threshold based on detection rate."""
        honeypots = [t for t in trades if t.get("was_honeypot", False)]
        if not trades:
            return params

        hp_rate = len(honeypots) / len(trades) * 100.0
        hp_loss = sum(
            abs(t.get("net_pnl_bps", 0)) for t in honeypots if t.get("net_pnl_bps", 0) < 0
        )

        if hp_rate > 10.0 and hp_loss > 0:
            params.max_honeypot_score *= 0.9
            params.max_honeypot_score = max(params.max_honeypot_score, 0.2)
        elif hp_rate < 2.0:
            params.max_honeypot_score *= 1.05
            params.max_honeypot_score = min(params.max_honeypot_score, 0.7)

        return params

    def _tune_position_sizing(
        self,
        params: TradingParams,
        trades: list[dict[str, Any]],
    ) -> TradingParams:
        """Tune position sizing using Kelly criterion."""
        if not trades:
            return params

        wins = [t for t in trades if t.get("net_pnl_bps", 0) > 0]
        losses = [t for t in trades if t.get("net_pnl_bps", 0) <= 0]

        if not wins or not losses:
            return params

        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t.get("net_pnl_bps", 0) for t in wins])
        avg_loss = abs(np.mean([t.get("net_pnl_bps", 0) for t in losses]))

        if avg_loss == 0:
            return params

        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / win_loss_ratio

        # Fractional Kelly (25%)
        kelly *= 0.25
        kelly = max(0.05, min(0.5, kelly))

        params.kelly_fraction = params.kelly_fraction * 0.7 + kelly * 0.3

        return params

    def _calculate_kelly(self, metrics: dict[str, float]) -> float:
        """Calculate Kelly fraction from metrics."""
        win_rate = metrics.get("win_rate", 50.0) / 100.0
        if win_rate <= 0 or win_rate >= 1:
            return 0.1

        total_pnl = metrics.get("total_pnl", 0)
        trades = metrics.get("trades", 1)

        if total_pnl <= 0:
            return 0.1

        avg = total_pnl / trades
        est_ratio = max(1.0, 2.0 * avg / abs(avg + 1))

        kelly = win_rate - (1 - win_rate) / est_ratio
        kelly *= 0.25

        return max(0.05, min(0.5, kelly))

    def _estimate_improvement(
        self,
        trades: list[dict[str, Any]],
        new_params: TradingParams,
    ) -> float:
        """Estimate improvement from new parameters."""
        current = self._simulate(
            trades,
            self.current_params.take_profit_bps,
            self.current_params.stop_loss_bps,
            self.current_params.max_honeypot_score,
        )

        proposed = self._simulate(
            trades,
            new_params.take_profit_bps,
            new_params.stop_loss_bps,
            new_params.max_honeypot_score,
        )

        if current["sharpe"] == 0:
            return 0.0

        return (proposed["sharpe"] - current["sharpe"]) / abs(current["sharpe"]) * 100.0
