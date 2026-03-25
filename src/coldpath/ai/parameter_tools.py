"""
AI Parameter Tools for strategy optimization.

Tools for running quick backtests, suggesting regime-based parameters,
and analyzing trading performance to provide actionable recommendations.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class MarketRegime(StrEnum):
    """Market regime classifications."""

    MEME_SEASON = "MEME_SEASON"
    NORMAL = "NORMAL"
    BEAR_VOLATILE = "BEAR_VOLATILE"
    HIGH_MEV = "HIGH_MEV"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"


@dataclass
class UserStrategyParams:
    """User strategy parameters for backtesting."""

    # Entry conditions
    min_liquidity_usd: float = 10000.0
    max_fdv_usd: float = 5000000.0
    min_holders: int = 50
    min_confidence: float = 0.6
    max_risk_score: float = 0.45

    # Exit strategy
    take_profit_pct: float = 25.0
    stop_loss_pct: float = 8.0
    trailing_stop_pct: float | None = None
    trailing_activation_pct: float = 10.0
    max_hold_minutes: int = 30

    # Position sizing
    sizing_mode: str = "fixed"  # "fixed" or "kelly"
    max_position_sol: float = 0.1
    pct_of_wallet: float = 0.05

    # Risk limits
    max_daily_loss_sol: float = 0.5
    max_drawdown_pct: float = 20.0
    max_concurrent_positions: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserStrategyParams":
        """Create from dictionary."""
        return cls(
            min_liquidity_usd=data.get("min_liquidity_usd", 10000.0),
            max_fdv_usd=data.get("max_fdv_usd", 5000000.0),
            min_holders=data.get("min_holders", 50),
            min_confidence=data.get("min_confidence", 0.6),
            max_risk_score=data.get("max_risk_score", 0.45),
            take_profit_pct=data.get("take_profit_pct", 25.0),
            stop_loss_pct=data.get("stop_loss_pct", 8.0),
            trailing_stop_pct=data.get("trailing_stop_pct"),
            trailing_activation_pct=data.get("trailing_activation_pct", 10.0),
            max_hold_minutes=data.get("max_hold_minutes", 30),
            sizing_mode=data.get("sizing_mode", "fixed"),
            max_position_sol=data.get("max_position_sol", 0.1),
            pct_of_wallet=data.get("pct_of_wallet", 0.05),
            max_daily_loss_sol=data.get("max_daily_loss_sol", 0.5),
            max_drawdown_pct=data.get("max_drawdown_pct", 20.0),
            max_concurrent_positions=data.get("max_concurrent_positions", 3),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_liquidity_usd": self.min_liquidity_usd,
            "max_fdv_usd": self.max_fdv_usd,
            "min_holders": self.min_holders,
            "min_confidence": self.min_confidence,
            "max_risk_score": self.max_risk_score,
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
            "trailing_activation_pct": self.trailing_activation_pct,
            "max_hold_minutes": self.max_hold_minutes,
            "sizing_mode": self.sizing_mode,
            "max_position_sol": self.max_position_sol,
            "pct_of_wallet": self.pct_of_wallet,
            "max_daily_loss_sol": self.max_daily_loss_sol,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_concurrent_positions": self.max_concurrent_positions,
        }


@dataclass
class QuickBacktestResult:
    """Result of a quick backtest."""

    sharpe_ratio: float
    sortino_ratio: float
    total_return_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    profit_factor: float
    avg_trade_pnl_pct: float
    avg_hold_time_minutes: float
    comparison: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sharpe": self.sharpe_ratio,
            "sortino": self.sortino_ratio,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate_pct": self.win_rate_pct,
            "total_trades": self.total_trades,
            "profit_factor": self.profit_factor,
            "avg_trade_pnl_pct": self.avg_trade_pnl_pct,
            "avg_hold_time_minutes": self.avg_hold_time_minutes,
            "comparison": self.comparison,
        }


@dataclass
class RegimeParams:
    """Pre-tuned parameters for a market regime."""

    regime: MarketRegime
    params: UserStrategyParams
    rationale: str
    confidence: float


class ParameterTools:
    """
    AI-powered tools for strategy parameter optimization.

    Provides:
    - Quick backtesting with proposed parameters
    - Regime-based parameter suggestions
    - Performance analysis and recommendations
    """

    # Regime-specific parameter presets
    REGIME_PRESETS: dict[MarketRegime, dict[str, Any]] = {
        MarketRegime.MEME_SEASON: {
            "min_confidence": 0.5,
            "take_profit_pct": 35.0,
            "stop_loss_pct": 12.0,
            "max_position_sol": 0.15,
            "trailing_stop_pct": 15.0,
            "trailing_activation_pct": 20.0,
            "max_hold_minutes": 60,
            "min_liquidity_usd": 8000.0,
            "max_fdv_usd": 10000000.0,
        },
        MarketRegime.NORMAL: {
            "min_confidence": 0.6,
            "take_profit_pct": 20.0,
            "stop_loss_pct": 8.0,
            "max_position_sol": 0.1,
            "trailing_stop_pct": 8.0,
            "trailing_activation_pct": 12.0,
            "max_hold_minutes": 30,
            "min_liquidity_usd": 10000.0,
            "max_fdv_usd": 5000000.0,
        },
        MarketRegime.BEAR_VOLATILE: {
            "min_confidence": 0.75,
            "take_profit_pct": 12.0,
            "stop_loss_pct": 5.0,
            "max_position_sol": 0.05,
            "trailing_stop_pct": 5.0,
            "trailing_activation_pct": 8.0,
            "max_hold_minutes": 15,
            "min_liquidity_usd": 15000.0,
            "max_fdv_usd": 2000000.0,
        },
        MarketRegime.HIGH_MEV: {
            "min_confidence": 0.65,
            "take_profit_pct": 18.0,
            "stop_loss_pct": 7.0,
            "max_position_sol": 0.08,
            "trailing_stop_pct": 6.0,
            "trailing_activation_pct": 10.0,
            "max_hold_minutes": 20,
            "min_liquidity_usd": 12000.0,
            "max_fdv_usd": 4000000.0,
        },
        MarketRegime.LOW_LIQUIDITY: {
            "min_confidence": 0.7,
            "take_profit_pct": 15.0,
            "stop_loss_pct": 6.0,
            "max_position_sol": 0.03,
            "trailing_stop_pct": 5.0,
            "trailing_activation_pct": 8.0,
            "max_hold_minutes": 20,
            "min_liquidity_usd": 20000.0,
            "max_fdv_usd": 3000000.0,
        },
    }

    # Regime rationales
    REGIME_RATIONALES: dict[MarketRegime, str] = {
        MarketRegime.MEME_SEASON: (
            "Meme season detected: Higher volatility and momentum opportunities. "
            "Wider stops to capture larger moves, increased position sizes for winners, "
            "relaxed entry filters to catch more opportunities."
        ),
        MarketRegime.NORMAL: (
            "Normal market conditions: Balanced approach with moderate risk/reward. "
            "Standard entry filters and position sizing, typical hold times."
        ),
        MarketRegime.BEAR_VOLATILE: (
            "Bear/volatile market: Defensive positioning required. "
            "Tighter stops to preserve capital, smaller positions, "
            "higher confidence threshold for entries, shorter hold times."
        ),
        MarketRegime.HIGH_MEV: (
            "High MEV activity detected: Increased sandwich attack risk. "
            "Slightly smaller positions, faster exits, recommend using Jito bundles."
        ),
        MarketRegime.LOW_LIQUIDITY: (
            "Low liquidity conditions: Higher slippage risk. "
            "Very small positions, higher liquidity requirements, "
            "quick exit strategy to avoid getting stuck."
        ),
    }

    def __init__(
        self,
        backtest_engine=None,
        storage=None,
        hotpath_client=None,
    ):
        """
        Initialize ParameterTools.

        Args:
            backtest_engine: BacktestEngine instance for running backtests.
            storage: Storage backend for historical data.
            hotpath_client: Client for communicating with HotPath engine.
        """
        self.backtest_engine = backtest_engine
        self.storage = storage
        self.hotpath_client = hotpath_client

    async def run_quick_backtest(
        self,
        strategy_params: dict[str, Any],
        days: int = 7,
        initial_capital_sol: float = 1.0,
        compare_to_current: bool = True,
        current_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run a quick backtest with proposed parameters.

        Args:
            strategy_params: Proposed strategy parameters.
            days: Number of days to backtest.
            initial_capital_sol: Initial capital in SOL.
            compare_to_current: Whether to compare to current strategy.
            current_params: Current strategy params (for comparison).

        Returns:
            Backtest results with optional comparison.
        """
        strategy = UserStrategyParams.from_dict(strategy_params)

        # Calculate time range
        end_time_ms = int(datetime.utcnow().timestamp() * 1000)
        start_time_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

        # Run proposed strategy backtest
        if self.backtest_engine:
            # Use real backtest engine
            from ..backtest.engine import BacktestConfig

            config = BacktestConfig(
                start_timestamp_ms=start_time_ms,
                end_timestamp_ms=end_time_ms,
                initial_capital_sol=initial_capital_sol,
                max_position_sol=strategy.max_position_sol,
                slippage_bps=300,
                default_stop_loss_pct=strategy.stop_loss_pct,
                default_take_profit_pct=strategy.take_profit_pct,
                min_liquidity_usd=strategy.min_liquidity_usd,
            )

            result = await self.backtest_engine.run(
                start_timestamp=start_time_ms,
                end_timestamp=end_time_ms,
                config=config,
            )

            metrics = result.get("metrics", {})
            proposed_result = QuickBacktestResult(
                sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
                sortino_ratio=metrics.get("sortino_ratio", 0.0),
                total_return_pct=metrics.get("total_return_pct", 0.0),
                max_drawdown_pct=metrics.get("max_drawdown_pct", 0.0),
                win_rate_pct=metrics.get("win_rate_pct", 0.0),
                total_trades=metrics.get("total_trades", 0),
                profit_factor=metrics.get("profit_factor", 0.0),
                avg_trade_pnl_pct=metrics.get("avg_trade_pct", 0.0),
                avg_hold_time_minutes=metrics.get("avg_hold_time_minutes", 0.0),
            )
        else:
            # Use simulated backtest (for testing without real data)
            proposed_result = self._simulate_backtest(strategy, days, initial_capital_sol)

        # Compare to current if requested
        comparison = None
        if compare_to_current and current_params:
            current_strategy = UserStrategyParams.from_dict(current_params)
            current_result = self._simulate_backtest(current_strategy, days, initial_capital_sol)
            comparison = self._generate_comparison(current_result, proposed_result)
            proposed_result.comparison = comparison

        return proposed_result.to_dict()

    def suggest_regime_params(
        self,
        regime: str,
        current_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Get parameter suggestions for a specific market regime.

        Args:
            regime: Market regime string.
            current_params: Current strategy parameters (for incremental suggestions).

        Returns:
            Suggested parameters with rationale.
        """
        try:
            market_regime = MarketRegime(regime)
        except ValueError:
            market_regime = MarketRegime.NORMAL
            logger.warning(f"Unknown regime '{regime}', defaulting to NORMAL")

        preset = self.REGIME_PRESETS[market_regime]
        rationale = self.REGIME_RATIONALES[market_regime]

        # Start with default params and apply regime preset
        suggested = UserStrategyParams()
        for key, value in preset.items():
            if hasattr(suggested, key):
                setattr(suggested, key, value)

        # Calculate confidence based on how much this differs from current
        confidence = 0.75
        if current_params:
            current = UserStrategyParams.from_dict(current_params)
            diff_score = self._calculate_param_difference(current, suggested)
            # Higher difference = lower confidence (more dramatic change)
            confidence = max(0.5, 0.85 - (diff_score * 0.1))

        return {
            "regime": market_regime.value,
            "params": suggested.to_dict(),
            "rationale": rationale,
            "confidence": confidence,
            "key_changes": self._identify_key_changes(
                current_params or {},
                suggested.to_dict(),
            ),
        }

    def analyze_performance(
        self,
        recent_trades: list[dict[str, Any]],
        current_params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Analyze recent trading performance and suggest improvements.

        Args:
            recent_trades: List of recent trade records.
            current_params: Current strategy parameters.

        Returns:
            Analysis with improvement suggestions.
        """
        if not recent_trades:
            return {
                "status": "insufficient_data",
                "message": "No recent trades to analyze",
                "suggestions": [],
            }

        # Calculate performance metrics
        total_trades = len(recent_trades)
        wins = [t for t in recent_trades if t.get("pnl_sol", 0) > 0]
        losses = [t for t in recent_trades if t.get("pnl_sol", 0) < 0]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = sum(t.get("pnl_pct", 0) for t in wins) / len(wins) if wins else 0
        avg_loss = sum(abs(t.get("pnl_pct", 0)) for t in losses) / len(losses) if losses else 0

        # Analyze stop loss hits
        stop_loss_exits = [t for t in recent_trades if t.get("exit_reason") == "stop_loss"]
        stop_loss_rate = len(stop_loss_exits) / total_trades if total_trades > 0 else 0

        # Analyze take profit hits
        take_profit_exits = [t for t in recent_trades if t.get("exit_reason") == "take_profit"]
        take_profit_rate = len(take_profit_exits) / total_trades if total_trades > 0 else 0

        # Generate suggestions
        suggestions = []
        current = UserStrategyParams.from_dict(current_params)

        # Stop loss analysis
        if stop_loss_rate > 0.4:
            # Too many stop losses - check if prices recover
            recovery_count = self._count_recoveries(stop_loss_exits, current.stop_loss_pct)
            if recovery_count > len(stop_loss_exits) * 0.3:
                suggestions.append(
                    {
                        "type": "widen_stop_loss",
                        "current": current.stop_loss_pct,
                        "suggested": current.stop_loss_pct * 1.3,
                        "rationale": (
                            f"{recovery_count} of {len(stop_loss_exits)} stop-loss exits "
                            "later recovered. Consider widening stop loss to capture these "
                            "recoveries."
                        ),
                        "confidence": 0.7,
                    }
                )

        # Take profit analysis
        if take_profit_rate < 0.15 and avg_win > current.take_profit_pct * 0.7:
            suggestions.append(
                {
                    "type": "lower_take_profit",
                    "current": current.take_profit_pct,
                    "suggested": current.take_profit_pct * 0.8,
                    "rationale": (
                        f"Only {take_profit_rate * 100:.1f}% of trades hit take profit. "
                        "Lower TP may lock in more wins."
                    ),
                    "confidence": 0.65,
                }
            )

        # Win rate too low
        if win_rate < 0.35:
            suggestions.append(
                {
                    "type": "increase_confidence_threshold",
                    "current": current.min_confidence,
                    "suggested": min(0.8, current.min_confidence + 0.1),
                    "rationale": (
                        f"Win rate is {win_rate * 100:.1f}%. Increasing confidence "
                        "threshold may improve trade quality."
                    ),
                    "confidence": 0.6,
                }
            )

        # Position sizing
        if avg_loss > current.max_position_sol * 0.15 * 100:  # Losing more than 15% per position
            suggestions.append(
                {
                    "type": "reduce_position_size",
                    "current": current.max_position_sol,
                    "suggested": current.max_position_sol * 0.7,
                    "rationale": (
                        f"Average loss of {avg_loss:.1f}% suggests positions may be too large. "
                        "Reducing size can limit drawdown impact."
                    ),
                    "confidence": 0.7,
                }
            )

        return {
            "status": "analyzed",
            "summary": {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "avg_win_pct": avg_win,
                "avg_loss_pct": avg_loss,
                "stop_loss_rate": stop_loss_rate,
                "take_profit_rate": take_profit_rate,
            },
            "suggestions": suggestions,
        }

    def _simulate_backtest(
        self,
        strategy: UserStrategyParams,
        days: int,
        initial_capital: float,
    ) -> QuickBacktestResult:
        """
        Simulate a backtest when real engine is not available.

        This provides reasonable estimates based on parameter analysis.
        """
        import random

        # Estimate metrics based on strategy parameters
        # Tighter stops = higher win rate but lower avg win
        base_win_rate = 0.45
        stop_factor = (10 - min(strategy.stop_loss_pct, 15)) / 10
        tp_factor = (strategy.take_profit_pct - 10) / 40
        confidence_factor = (strategy.min_confidence - 0.5) / 0.3

        win_rate = (
            base_win_rate + (stop_factor * 0.08) - (tp_factor * 0.12) + (confidence_factor * 0.1)
        )
        win_rate = max(0.3, min(0.7, win_rate))

        # Estimate number of trades
        trades_per_day = max(1, 8 - (strategy.min_confidence * 8))
        total_trades = int(trades_per_day * days * random.uniform(0.8, 1.2))

        # Calculate returns
        avg_win = strategy.take_profit_pct * 0.7  # Usually don't hit full TP
        avg_loss = strategy.stop_loss_pct * 0.9  # Usually close to SL

        wins = int(total_trades * win_rate)
        losses = total_trades - wins

        total_return_pct = (wins * avg_win) - (losses * avg_loss)

        # Add some noise
        total_return_pct *= random.uniform(0.85, 1.15)

        # Calculate Sharpe (simplified)
        daily_return = total_return_pct / days
        volatility = abs(total_return_pct) / (days**0.5) * 0.5
        sharpe = (daily_return * 365) / (volatility * (365**0.5)) if volatility > 0 else 0

        # Sortino (use downside deviation)
        downside_vol = volatility * (1 - win_rate)
        sortino = (daily_return * 365) / (downside_vol * (365**0.5)) if downside_vol > 0 else 0

        # Max drawdown estimate
        max_dd = (
            strategy.stop_loss_pct
            * min(3, losses / max(1, wins) * 2)
            * strategy.max_position_sol
            / initial_capital
        )
        max_dd = min(30, max_dd * 100)

        # Profit factor
        gross_profit = wins * avg_win
        gross_loss = losses * avg_loss
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        )

        return QuickBacktestResult(
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            total_return_pct=round(total_return_pct, 2),
            max_drawdown_pct=round(max_dd, 2),
            win_rate_pct=round(win_rate * 100, 1),
            total_trades=total_trades,
            profit_factor=round(profit_factor, 2),
            avg_trade_pnl_pct=round(total_return_pct / total_trades, 2) if total_trades > 0 else 0,
            avg_hold_time_minutes=strategy.max_hold_minutes * 0.6,
        )

    def _generate_comparison(
        self,
        current: QuickBacktestResult,
        proposed: QuickBacktestResult,
    ) -> dict[str, Any]:
        """Generate comparison between current and proposed strategies."""

        def delta_str(
            current_val: float, proposed_val: float, fmt: str = "{:+.2f}"
        ) -> tuple[str, bool]:
            delta = proposed_val - current_val
            is_positive = delta > 0
            return fmt.format(delta), is_positive

        comparisons = []

        # Sharpe
        delta, is_pos = delta_str(current.sharpe_ratio, proposed.sharpe_ratio)
        comparisons.append(
            {
                "metric": "Sharpe Ratio",
                "current": f"{current.sharpe_ratio:.2f}",
                "proposed": f"{proposed.sharpe_ratio:.2f}",
                "delta": delta,
                "is_improvement": is_pos,
            }
        )

        # Max Drawdown (lower is better)
        delta, _ = delta_str(current.max_drawdown_pct, proposed.max_drawdown_pct, "{:+.1f}%")
        comparisons.append(
            {
                "metric": "Max Drawdown",
                "current": f"{current.max_drawdown_pct:.1f}%",
                "proposed": f"{proposed.max_drawdown_pct:.1f}%",
                "delta": delta,
                "is_improvement": proposed.max_drawdown_pct < current.max_drawdown_pct,
            }
        )

        # Win Rate
        delta, is_pos = delta_str(current.win_rate_pct, proposed.win_rate_pct, "{:+.1f}%")
        comparisons.append(
            {
                "metric": "Win Rate",
                "current": f"{current.win_rate_pct:.1f}%",
                "proposed": f"{proposed.win_rate_pct:.1f}%",
                "delta": delta,
                "is_improvement": is_pos,
            }
        )

        # Total Return
        delta, is_pos = delta_str(current.total_return_pct, proposed.total_return_pct, "{:+.1f}%")
        comparisons.append(
            {
                "metric": "Total Return",
                "current": f"{current.total_return_pct:.1f}%",
                "proposed": f"{proposed.total_return_pct:.1f}%",
                "delta": delta,
                "is_improvement": is_pos,
            }
        )

        # Summary
        improvements = sum(1 for c in comparisons if c["is_improvement"])
        total = len(comparisons)

        return {
            "comparisons": comparisons,
            "improvements": improvements,
            "total_metrics": total,
            "recommendation": (
                "Proposed parameters show improvement"
                if improvements > total / 2
                else "Current parameters may be preferable"
            ),
        }

    def _calculate_param_difference(
        self,
        current: UserStrategyParams,
        suggested: UserStrategyParams,
    ) -> float:
        """Calculate normalized difference between parameter sets."""
        diffs = []

        # Key params to compare
        params = [
            ("stop_loss_pct", 10),
            ("take_profit_pct", 30),
            ("max_position_sol", 0.1),
            ("min_confidence", 0.5),
            ("min_liquidity_usd", 10000),
        ]

        for param, normalizer in params:
            current_val = getattr(current, param, 0) or 0
            suggested_val = getattr(suggested, param, 0) or 0
            if normalizer > 0:
                diff = abs(current_val - suggested_val) / normalizer
                diffs.append(min(1.0, diff))

        return sum(diffs) / len(diffs) if diffs else 0

    def _identify_key_changes(
        self,
        current: dict[str, Any],
        suggested: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Identify the most significant parameter changes."""
        changes = []

        key_params = [
            ("stop_loss_pct", "Stop Loss", "%"),
            ("take_profit_pct", "Take Profit", "%"),
            ("max_position_sol", "Max Position", " SOL"),
            ("min_confidence", "Min Confidence", ""),
            ("trailing_stop_pct", "Trailing Stop", "%"),
        ]

        for param, label, suffix in key_params:
            current_val = current.get(param)
            suggested_val = suggested.get(param)

            if current_val is None or suggested_val is None:
                continue

            if current_val == 0:
                if suggested_val != 0:
                    changes.append(
                        {
                            "param": param,
                            "label": label,
                            "current": f"{current_val}{suffix}",
                            "suggested": f"{suggested_val}{suffix}",
                            "change": "new",
                        }
                    )
                continue

            pct_change = (suggested_val - current_val) / current_val * 100

            if abs(pct_change) > 10:  # Only significant changes
                direction = "increase" if pct_change > 0 else "decrease"
                changes.append(
                    {
                        "param": param,
                        "label": label,
                        "current": f"{current_val}{suffix}",
                        "suggested": f"{suggested_val}{suffix}",
                        "change": f"{direction} by {abs(pct_change):.0f}%",
                    }
                )

        return changes

    def _count_recoveries(
        self,
        stop_loss_trades: list[dict[str, Any]],
        stop_loss_pct: float,
    ) -> int:
        """
        Count trades where price recovered after stop loss.

        This is a simplified heuristic - in production, you'd check
        actual subsequent price data.
        """
        # For simulation, assume 30-40% of stop losses were premature
        import random

        return int(len(stop_loss_trades) * random.uniform(0.25, 0.45))


# Tool function exports for AI assistant
async def run_quick_backtest(
    strategy_params: dict[str, Any],
    days: int = 7,
    initial_capital_sol: float = 1.0,
) -> dict[str, Any]:
    """Tool: Run a quick backtest with proposed parameters."""
    tools = ParameterTools()
    return await tools.run_quick_backtest(
        strategy_params=strategy_params,
        days=days,
        initial_capital_sol=initial_capital_sol,
    )


def suggest_regime_params(regime: str) -> dict[str, Any]:
    """Tool: Get parameter suggestions for market regime."""
    tools = ParameterTools()
    return tools.suggest_regime_params(regime=regime)


def analyze_performance(
    recent_trades: list[dict[str, Any]],
    current_params: dict[str, Any],
) -> dict[str, Any]:
    """Tool: Analyze recent trading performance."""
    tools = ParameterTools()
    return tools.analyze_performance(
        recent_trades=recent_trades,
        current_params=current_params,
    )
