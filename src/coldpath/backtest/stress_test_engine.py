"""
Stress Test Engine - Test strategies under extreme market conditions.

Scenarios:
1. Flash Crash: -50% price drop in 1 hour across all tokens
2. High Volatility: 5x normal volatility
3. Low Liquidity: TVL drops to 20% of normal
4. MEV Attack Swarm: 80% of trades sandwiched
5. Network Congestion: 60% inclusion rate (40% failures)
6. Rug Pull Cluster: 40% of tokens rug within 1 hour

Each scenario modifies historical data to simulate extreme conditions,
then runs the backtest engine to measure strategy resilience.
"""

import asyncio
import copy
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class StressScenario(Enum):
    """Predefined stress test scenarios."""

    FLASH_CRASH = "flash_crash"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    MEV_ATTACK_SWARM = "mev_attack_swarm"
    NETWORK_CONGESTION = "network_congestion"
    RUG_PULL_CLUSTER = "rug_pull_cluster"


@dataclass
class StressTestResult:
    """Results from a stress test scenario.

    Attributes:
        scenario: The stress scenario that was tested.
        sharpe: Sharpe ratio under stress.
        max_drawdown: Maximum drawdown percentage.
        win_rate: Win rate percentage.
        worst_trade_pnl: Worst individual trade PnL.
        recovery_time_hours: Hours to recover from worst drawdown.
        circuit_breaker_triggered: Whether the circuit breaker was hit.
        total_trades: Total number of trades.
        total_return_pct: Total return percentage.
        survived: Whether the strategy survived the scenario.
    """

    scenario: StressScenario
    sharpe: float
    max_drawdown: float
    win_rate: float
    worst_trade_pnl: float
    recovery_time_hours: float
    circuit_breaker_triggered: bool
    total_trades: int = 0
    total_return_pct: float = 0.0
    survived: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "scenario": self.scenario.value,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "worst_trade_pnl": self.worst_trade_pnl,
            "recovery_time_hours": self.recovery_time_hours,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "total_trades": self.total_trades,
            "total_return_pct": self.total_return_pct,
            "survived": self.survived,
        }


@dataclass
class StressTestReport:
    """Complete stress test report across all scenarios.

    Attributes:
        results: Results for each scenario.
        overall_resilience: Overall resilience score (0-1).
        scenarios_survived: Number of scenarios survived.
        total_scenarios: Total scenarios tested.
        weakest_scenario: The scenario with worst performance.
        recommendations: Actionable recommendations.
    """

    results: list[StressTestResult]
    overall_resilience: float
    scenarios_survived: int
    total_scenarios: int
    weakest_scenario: str | None
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "overall_resilience": self.overall_resilience,
            "scenarios_survived": self.scenarios_survived,
            "total_scenarios": self.total_scenarios,
            "weakest_scenario": self.weakest_scenario,
            "recommendations": self.recommendations,
        }


class StressTestEngine:
    """
    Test strategy under extreme market conditions.

    Modifies historical data to simulate extreme conditions, then
    evaluates strategy performance using a backtest function.

    Usage:
        engine = StressTestEngine()
        result = await engine.run_stress_test(
            params={"stop_loss_pct": 8.0, ...},
            scenario=StressScenario.FLASH_CRASH,
            historical_data=data,
            backtest_fn=my_backtest,
        )
    """

    # Scenario parameters
    SCENARIO_PARAMS = {
        StressScenario.FLASH_CRASH: {
            "price_drop_pct": 50.0,
            "duration_hours": 1.0,
            "recovery_hours": 4.0,
        },
        StressScenario.HIGH_VOLATILITY: {
            "volatility_multiplier": 5.0,
            "duration_hours": 24.0,
        },
        StressScenario.LOW_LIQUIDITY: {
            "liquidity_multiplier": 0.2,
            "duration_hours": 12.0,
        },
        StressScenario.MEV_ATTACK_SWARM: {
            "sandwich_probability": 0.8,
            "extraction_bps": 200,
            "duration_hours": 6.0,
        },
        StressScenario.NETWORK_CONGESTION: {
            "inclusion_rate": 0.6,
            "latency_multiplier": 5.0,
            "duration_hours": 4.0,
        },
        StressScenario.RUG_PULL_CLUSTER: {
            "rug_probability": 0.4,
            "price_crash_pct": 95.0,
            "duration_hours": 1.0,
        },
    }

    def __init__(
        self,
        circuit_breaker_drawdown_pct: float = 20.0,
        random_seed: int | None = 42,
    ):
        """Initialize the stress test engine.

        Args:
            circuit_breaker_drawdown_pct: Drawdown threshold for circuit breaker.
            random_seed: Random seed for reproducibility.
        """
        self.circuit_breaker_drawdown_pct = circuit_breaker_drawdown_pct

        if random_seed is not None:
            np.random.seed(random_seed)

    async def run_stress_test(
        self,
        params: dict[str, Any],
        scenario: StressScenario,
        historical_data: list[dict[str, Any]],
        backtest_fn: Callable | None = None,
    ) -> StressTestResult:
        """
        Run a single stress test scenario.

        Modifies historical data to simulate extreme conditions,
        then evaluates the strategy.

        Args:
            params: Strategy parameters to test.
            scenario: Stress scenario to simulate.
            historical_data: Historical market data.
            backtest_fn: Async function(params, data) -> results dict.
                        If None, uses internal simulation.

        Returns:
            StressTestResult with performance under stress.
        """
        logger.info(
            "Running stress test: scenario=%s, params=%d",
            scenario.value,
            len(params),
        )

        # Apply stress to historical data
        stressed_data = self._apply_stress(historical_data, scenario)

        if backtest_fn:
            # Use provided backtest function
            if asyncio.iscoroutinefunction(backtest_fn):
                result = await backtest_fn(params, stressed_data)
            else:
                result = backtest_fn(params, stressed_data)
        else:
            # Use internal simulation
            result = self._simulate_backtest(params, stressed_data)

        # Extract metrics from result
        sharpe = result.get("sharpe_ratio", result.get("sharpe", 0.0))
        max_dd = result.get("max_drawdown_pct", result.get("max_drawdown", 0.0))
        win_rate = result.get("win_rate_pct", result.get("win_rate", 0.0))
        worst_trade = result.get("worst_trade_pnl", 0.0)
        total_trades = result.get("total_trades", 0)
        total_return = result.get("total_return_pct", 0.0)

        # Compute recovery time
        recovery_hours = self._compute_recovery_time(result)

        # Check circuit breaker
        cb_threshold = params.get("max_drawdown_pct", self.circuit_breaker_drawdown_pct)
        circuit_breaker_triggered = abs(max_dd) > cb_threshold

        # Determine survival
        survived = (
            not circuit_breaker_triggered and total_return > -50.0  # Not wiped out
        )

        stress_result = StressTestResult(
            scenario=scenario,
            sharpe=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            worst_trade_pnl=worst_trade,
            recovery_time_hours=recovery_hours,
            circuit_breaker_triggered=circuit_breaker_triggered,
            total_trades=total_trades,
            total_return_pct=total_return,
            survived=survived,
        )

        logger.info(
            "Stress test result: scenario=%s, sharpe=%.2f, max_dd=%.1f%%, "
            "circuit_breaker=%s, survived=%s",
            scenario.value,
            sharpe,
            max_dd,
            circuit_breaker_triggered,
            survived,
        )

        return stress_result

    async def run_all_scenarios(
        self,
        params: dict[str, Any],
        historical_data: list[dict[str, Any]],
        backtest_fn: Callable | None = None,
    ) -> StressTestReport:
        """
        Run all stress test scenarios and produce a comprehensive report.

        Args:
            params: Strategy parameters.
            historical_data: Historical market data.
            backtest_fn: Backtest function.

        Returns:
            StressTestReport with all scenario results.
        """
        results: list[StressTestResult] = []

        for scenario in StressScenario:
            result = await self.run_stress_test(
                params=params,
                scenario=scenario,
                historical_data=historical_data,
                backtest_fn=backtest_fn,
            )
            results.append(result)

        # Calculate overall resilience
        survived_count = sum(1 for r in results if r.survived)
        total = len(results)

        if total > 0:
            resilience = survived_count / total
        else:
            resilience = 0.0

        # Find weakest scenario
        weakest = None
        if results:
            worst = min(results, key=lambda r: r.sharpe)
            weakest = worst.scenario.value

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        return StressTestReport(
            results=results,
            overall_resilience=resilience,
            scenarios_survived=survived_count,
            total_scenarios=total,
            weakest_scenario=weakest,
            recommendations=recommendations,
        )

    def _apply_stress(
        self,
        data: list[dict[str, Any]],
        scenario: StressScenario,
    ) -> list[dict[str, Any]]:
        """Apply stress scenario to historical data.

        Creates a deep copy and modifies prices, volumes, and liquidity
        to simulate the stress scenario.

        Args:
            data: Historical market data.
            scenario: Stress scenario to apply.

        Returns:
            Modified copy of the data.
        """
        if not data:
            return data

        # Deep copy to avoid modifying original
        stressed = copy.deepcopy(data)

        if scenario == StressScenario.FLASH_CRASH:
            stressed = self._simulate_flash_crash(stressed)
        elif scenario == StressScenario.HIGH_VOLATILITY:
            stressed = self._simulate_high_volatility(stressed)
        elif scenario == StressScenario.LOW_LIQUIDITY:
            stressed = self._simulate_low_liquidity(stressed)
        elif scenario == StressScenario.MEV_ATTACK_SWARM:
            stressed = self._simulate_mev_attack(stressed)
        elif scenario == StressScenario.NETWORK_CONGESTION:
            stressed = self._simulate_network_congestion(stressed)
        elif scenario == StressScenario.RUG_PULL_CLUSTER:
            stressed = self._simulate_rug_pull_cluster(stressed)

        return stressed

    def _simulate_flash_crash(
        self,
        data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Simulate a flash crash: -50% price drop in 1 hour.

        The crash happens at a random point in the data, followed by
        a partial recovery.
        """
        n = len(data)
        if n < 10:
            return data

        # Crash starts at random point in first 70% of data
        crash_start = np.random.randint(0, int(n * 0.7))
        crash_duration = max(1, int(n * 0.05))  # ~5% of data
        recovery_duration = crash_duration * 4

        for i in range(crash_start, min(n, crash_start + crash_duration)):
            # Linear price drop to -50%
            progress = (i - crash_start) / crash_duration
            drop_factor = 1.0 - 0.5 * progress

            for key in ("close", "price", "open", "high", "low"):
                if key in data[i]:
                    data[i][key] *= drop_factor

            # Volume spike during crash
            for key in ("volume", "volume_24h"):
                if key in data[i]:
                    data[i][key] *= 3.0

        # Partial recovery
        recovery_end = min(n, crash_start + crash_duration + recovery_duration)
        for i in range(crash_start + crash_duration, recovery_end):
            progress = (i - crash_start - crash_duration) / recovery_duration
            recovery_factor = 0.5 + 0.3 * progress  # Recover to 80%

            for key in ("close", "price", "open", "high", "low"):
                if key in data[i]:
                    data[i][key] *= recovery_factor

        return data

    def _simulate_high_volatility(
        self,
        data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Simulate high volatility: 5x normal price swings."""
        for entry in data:
            for key in ("close", "price", "open", "high", "low"):
                if key in entry:
                    # Add 5x volatility noise
                    base = entry[key]
                    noise = np.random.normal(0, base * 0.1)  # 10% std
                    entry[key] = max(0.0001, base + noise * 5)

            # Increase reported volatility
            if "volatility_1h" in entry:
                entry["volatility_1h"] *= 5.0
            if "volatility" in entry:
                entry["volatility"] *= 5.0

        return data

    def _simulate_low_liquidity(
        self,
        data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Simulate low liquidity: TVL drops to 20% of normal."""
        for entry in data:
            for key in (
                "liquidity_sol",
                "liquidity_usd",
                "liquidity_tokens",
                "tvl",
            ):
                if key in entry:
                    entry[key] *= 0.2

            # Increase slippage
            if "slippage_1pct" in entry:
                entry["slippage_1pct"] *= 3.0
            if "slippage_5pct" in entry:
                entry["slippage_5pct"] *= 3.0

        return data

    def _simulate_mev_attack(
        self,
        data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Simulate MEV attack swarm: 80% of trades sandwiched."""
        for entry in data:
            # 80% chance of being sandwiched
            if np.random.random() < 0.8:
                entry["sandwiched"] = True
                entry["mev_extraction_bps"] = np.random.randint(50, 300)

                # Sandwich worsens effective price by 1-3%
                extraction = entry["mev_extraction_bps"] / 10000
                for key in ("close", "price"):
                    if key in entry:
                        entry[key] *= 1 - extraction

            # Increase slippage
            if "slippage_1pct" in entry:
                entry["slippage_1pct"] *= 2.0

        return data

    def _simulate_network_congestion(
        self,
        data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Simulate network congestion: 60% inclusion rate."""
        for entry in data:
            # 40% chance of transaction failure
            if np.random.random() > 0.6:
                entry["tx_failed"] = True
                entry["included"] = False
            else:
                entry["tx_failed"] = False
                entry["included"] = True

            # Add latency
            entry["latency_ms"] = entry.get("latency_ms", 0) + np.random.randint(500, 5000)

        return data

    def _simulate_rug_pull_cluster(
        self,
        data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Simulate rug pull cluster: 40% of tokens rug within 1 hour."""
        n = len(data)
        if n < 5:
            return data

        # Mark 40% of entries as rugged
        rug_indices = np.random.choice(n, size=int(n * 0.4), replace=False)

        for i in rug_indices:
            # Token price crashes 95%
            for key in ("close", "price", "open", "high", "low"):
                if key in data[i]:
                    data[i][key] *= 0.05

            # Liquidity vanishes
            for key in ("liquidity_sol", "liquidity_usd", "tvl"):
                if key in data[i]:
                    data[i][key] *= 0.01

            data[i]["rug_pull"] = True
            data[i]["rug_hit"] = True

        return data

    def _simulate_backtest(
        self,
        params: dict[str, Any],
        data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Internal simplified backtest simulation.

        Used when no external backtest function is provided.

        Args:
            params: Strategy parameters.
            data: Historical data.

        Returns:
            Dictionary with simulated metrics.
        """
        if not data:
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate_pct": 0.0,
                "worst_trade_pnl": 0.0,
                "total_trades": 0,
                "total_return_pct": 0.0,
            }

        # Extract returns from price data
        prices = []
        for entry in data:
            price = entry.get("close", entry.get("price", 0))
            if price > 0:
                prices.append(price)

        if len(prices) < 2:
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate_pct": 0.0,
                "worst_trade_pnl": 0.0,
                "total_trades": 0,
                "total_return_pct": 0.0,
            }

        prices_arr = np.array(prices)
        returns = np.diff(prices_arr) / prices_arr[:-1]

        # Apply stop loss and take profit filters
        sl = params.get("stop_loss_pct", 10.0) / 100
        tp = params.get("take_profit_pct", 50.0) / 100

        filtered_returns = []
        for r in returns:
            if r < -sl:
                filtered_returns.append(-sl)  # Stop loss
            elif r > tp:
                filtered_returns.append(tp)  # Take profit
            else:
                filtered_returns.append(r)

        returns_arr = np.array(filtered_returns)

        # Calculate metrics
        sharpe = 0.0
        if len(returns_arr) > 1 and np.std(returns_arr) > 1e-10:
            sharpe = float(np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(365))

        # Cumulative returns for drawdown
        cumulative = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max * 100
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Win rate
        wins = np.sum(returns_arr > 0)
        total = len(returns_arr)
        win_rate = float(wins / total * 100) if total > 0 else 0.0

        worst_trade = float(np.min(returns_arr) * 100) if total > 0 else 0.0
        total_return = float((cumulative[-1] - 1) * 100) if len(cumulative) > 0 else 0.0

        return {
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd,
            "win_rate_pct": win_rate,
            "worst_trade_pnl": worst_trade,
            "total_trades": total,
            "total_return_pct": total_return,
        }

    def _compute_recovery_time(
        self,
        result: dict[str, Any],
    ) -> float:
        """Compute estimated recovery time from worst drawdown.

        Args:
            result: Backtest result dictionary.

        Returns:
            Estimated recovery time in hours.
        """
        max_dd = abs(result.get("max_drawdown_pct", result.get("max_drawdown", 0)))
        avg_return = result.get("avg_return_pct", 0.5)
        total_trades = result.get("total_trades", 0)

        if avg_return <= 0 or total_trades == 0:
            return float("inf") if max_dd > 0 else 0.0

        # Estimate: trades needed to recover
        trades_to_recover = max_dd / max(avg_return, 0.01)
        # Assume ~1 trade per hour on average
        return min(trades_to_recover, 720.0)  # Cap at 30 days

    def _generate_recommendations(
        self,
        results: list[StressTestResult],
    ) -> list[str]:
        """Generate recommendations based on stress test results.

        Args:
            results: All stress test results.

        Returns:
            List of recommendation strings.
        """
        recommendations: list[str] = []

        for r in results:
            if not r.survived:
                recommendations.append(
                    f"CRITICAL: Strategy did not survive {r.scenario.value}. "
                    f"Max drawdown: {r.max_drawdown:.1f}%. "
                    "Review risk limits."
                )

            if r.circuit_breaker_triggered:
                recommendations.append(
                    f"Circuit breaker triggered in {r.scenario.value}. "
                    f"Max drawdown {r.max_drawdown:.1f}% exceeded threshold."
                )

        # Flash crash specific
        flash = next(
            (r for r in results if r.scenario == StressScenario.FLASH_CRASH),
            None,
        )
        if flash and flash.max_drawdown > 30:
            recommendations.append(
                "High vulnerability to flash crashes. "
                "Consider tighter stop losses or trailing stops."
            )

        # MEV specific
        mev = next(
            (r for r in results if r.scenario == StressScenario.MEV_ATTACK_SWARM),
            None,
        )
        if mev and mev.win_rate < 40:
            recommendations.append(
                "Strategy vulnerable to MEV attacks. "
                "Consider using private transactions or adjusting slippage."
            )

        # Low liquidity specific
        liq = next(
            (r for r in results if r.scenario == StressScenario.LOW_LIQUIDITY),
            None,
        )
        if liq and liq.sharpe < 0:
            recommendations.append(
                "Negative Sharpe under low liquidity. "
                "Reduce position sizes or add liquidity filters."
            )

        if not recommendations:
            recommendations.append("Strategy shows good resilience across all stress scenarios.")

        return recommendations
