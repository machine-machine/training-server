"""
Daily Optimization Scheduler - Automated daily backtest optimization.

This module provides:
- Scheduled daily optimization runs
- Automatic parameter persistence
- Performance tracking over time
- Robustness validation on each run
- Integration with feedback loop for continuous learning

Usage:
    from coldpath.backtest.daily_optimizer import DailyOptimizer

    optimizer = DailyOptimizer(
        backtest_runner=my_backtest_fn,
        schedule_time="06:00",  # Run at 6 AM daily
        risk_tolerance="moderate",
    )

    # Start the scheduler
    await optimizer.start()

    # Get latest optimized params
    params = optimizer.get_best_params()
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class OptimizationFrequency(Enum):
    """Frequency of optimization runs."""

    HOURLY = "hourly"
    EVERY_6_HOURS = "every_6_hours"
    EVERY_12_HOURS = "every_12_hours"
    DAILY = "daily"
    WEEKLY = "weekly"


@dataclass
class OptimizationProfile:
    """Profile for optimization configuration."""

    name: str
    risk_tolerance: str = "moderate"
    primary_goal: str = "balanced"
    capital_sol: float = 1.0
    strategy: str = "balanced"
    max_iterations: int = 50
    max_time_minutes: int = 30
    enable_regime_detection: bool = True
    enable_robustness_validation: bool = True
    robustness_threshold: float = 0.6
    quick_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "risk_tolerance": self.risk_tolerance,
            "primary_goal": self.primary_goal,
            "capital_sol": self.capital_sol,
            "strategy": self.strategy,
            "max_iterations": self.max_iterations,
            "max_time_minutes": self.max_time_minutes,
            "enable_regime_detection": self.enable_regime_detection,
            "enable_robustness_validation": self.enable_robustness_validation,
            "robustness_threshold": self.robustness_threshold,
            "quick_mode": self.quick_mode,
        }


@dataclass
class OptimizationResult:
    """Result from a scheduled optimization run."""

    timestamp: str
    profile_name: str
    success: bool

    # Best parameters found
    best_params: dict[str, Any] = field(default_factory=dict)
    best_metrics: dict[str, float] = field(default_factory=dict)
    best_score: float = 0.0

    # Robustness
    robustness_score: float | None = None
    is_robust: bool | None = None

    # Performance comparison
    improvement_over_previous: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate_pct: float = 0.0
    max_drawdown_pct: float = 0.0

    # Details
    optimization_time_seconds: float = 0.0
    iterations: int = 0
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "profile_name": self.profile_name,
            "success": self.success,
            "best_params": self.best_params,
            "best_metrics": self.best_metrics,
            "best_score": self.best_score,
            "robustness_score": self.robustness_score,
            "is_robust": self.is_robust,
            "improvement_over_previous": self.improvement_over_previous,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate_pct": self.win_rate_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "optimization_time_seconds": self.optimization_time_seconds,
            "iterations": self.iterations,
            "error_message": self.error_message,
        }


# Pre-configured profiles for common use cases
DEFAULT_PROFILES = {
    "profit_maximizer": OptimizationProfile(
        name="profit_maximizer",
        risk_tolerance="moderate",
        primary_goal="max_sharpe",
        capital_sol=1.0,
        strategy="thorough",
        max_iterations=100,
        max_time_minutes=45,
        enable_regime_detection=True,
        enable_robustness_validation=True,
        robustness_threshold=0.65,
    ),
    "robust_conservative": OptimizationProfile(
        name="robust_conservative",
        risk_tolerance="conservative",
        primary_goal="min_drawdown",
        capital_sol=1.0,
        strategy="thorough",
        max_iterations=75,
        max_time_minutes=30,
        enable_regime_detection=True,
        enable_robustness_validation=True,
        robustness_threshold=0.75,  # Higher threshold
    ),
    "quick_daily": OptimizationProfile(
        name="quick_daily",
        risk_tolerance="moderate",
        primary_goal="balanced",
        capital_sol=1.0,
        strategy="fast",
        max_iterations=25,
        max_time_minutes=10,
        enable_regime_detection=True,
        enable_robustness_validation=False,  # Skip for speed
        quick_mode=True,
    ),
    "aggressive_growth": OptimizationProfile(
        name="aggressive_growth",
        risk_tolerance="aggressive",
        primary_goal="max_return",
        capital_sol=2.0,
        strategy="balanced",
        max_iterations=50,
        max_time_minutes=20,
        enable_regime_detection=True,
        enable_robustness_validation=True,
        robustness_threshold=0.55,
    ),
    "memecoin_hunter": OptimizationProfile(
        name="memecoin_hunter",
        risk_tolerance="degen",
        primary_goal="profit_quality",
        capital_sol=0.5,
        strategy="adaptive",
        max_iterations=75,
        max_time_minutes=25,
        enable_regime_detection=True,
        enable_robustness_validation=True,
        robustness_threshold=0.5,
    ),
}


class DailyOptimizer:
    """Automated daily optimization scheduler for backtesting.

    Features:
    - Scheduled optimization runs (hourly, daily, weekly)
    - Multiple optimization profiles
    - Automatic parameter persistence
    - Performance tracking and history
    - Robustness validation integration
    - Integration with feedback loop

    Example:
        optimizer = DailyOptimizer(
            backtest_runner=my_backtest_fn,
            profile=DEFAULT_PROFILES["profit_maximizer"],
            frequency=OptimizationFrequency.DAILY,
            storage_path="./optimization_history",
        )

        await optimizer.start()

        # Get best params
        best = optimizer.get_best_params()
        print(f"Best Sharpe: {best['sharpe_ratio']}")
    """

    def __init__(
        self,
        backtest_runner: Callable,
        profile: OptimizationProfile | None = None,
        frequency: OptimizationFrequency = OptimizationFrequency.DAILY,
        schedule_time: str = "06:00",
        storage_path: str | None = None,
        feedback_store: Any | None = None,
        on_optimization_complete: Callable | None = None,
    ):
        """Initialize the daily optimizer.

        Args:
            backtest_runner: Async function to run backtests
            profile: Optimization profile to use
            frequency: How often to run optimizations
            schedule_time: Time of day for daily runs (HH:MM format)
            storage_path: Path to store optimization history
            feedback_store: Optional feedback store for learning
            on_optimization_complete: Callback when optimization completes
        """
        self.backtest_runner = backtest_runner
        self.profile = profile or DEFAULT_PROFILES["profit_maximizer"]
        self.frequency = frequency
        self.schedule_time = schedule_time
        self.storage_path = Path(storage_path or "./data/optimization_history")
        self.feedback_store = feedback_store
        self.on_optimization_complete = on_optimization_complete

        # State
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_run: datetime | None = None
        self._last_result: OptimizationResult | None = None
        self._history: list[OptimizationResult] = []
        self._best_params: dict[str, Any] = {}
        self._best_score: float = 0.0

        # Components (lazy loaded)
        self._orchestrator = None
        self._robustness_validator = None

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load history
        self._load_history()

    def _get_orchestrator(self):
        """Lazy load smart orchestrator."""
        if self._orchestrator is None:
            from coldpath.backtest.smart_orchestrator import (
                OptimizationStrategy,
                SmartBacktestOrchestrator,
                SmartOrchestratorConfig,
            )

            strategy_map = {
                "fast": OptimizationStrategy.FAST,
                "balanced": OptimizationStrategy.BALANCED,
                "thorough": OptimizationStrategy.THOROUGH,
                "adaptive": OptimizationStrategy.ADAPTIVE,
            }

            config = SmartOrchestratorConfig(
                optimization_strategy=strategy_map.get(
                    self.profile.strategy, OptimizationStrategy.BALANCED
                ),
                max_optimization_iterations=self.profile.max_iterations,
                max_optimization_time_minutes=self.profile.max_time_minutes,
                enable_robustness_validation=self.profile.enable_robustness_validation,
                robustness_threshold=self.profile.robustness_threshold,
            )

            self._orchestrator = SmartBacktestOrchestrator(
                backtest_runner=self.backtest_runner,
                config=config,
                feedback_store=self.feedback_store,
                historical_results=[r.to_dict() for r in self._history[-100:]],
            )

        return self._orchestrator

    def _get_robustness_validator(self):
        """Lazy load robustness validator."""
        if self._robustness_validator is None and self.profile.enable_robustness_validation:
            from coldpath.backtest.robustness_validator import RobustnessValidator

            self._robustness_validator = RobustnessValidator(
                robustness_threshold=self.profile.robustness_threshold,
            )
        return self._robustness_validator

    def _load_history(self):
        """Load optimization history from disk."""
        history_file = self.storage_path / "optimization_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                    self._history = [OptimizationResult(**r) for r in data.get("results", [])]
                    self._best_params = data.get("best_params", {})
                    self._best_score = data.get("best_score", 0.0)
                logger.info(f"Loaded {len(self._history)} optimization results from history")
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")

    def _save_history(self):
        """Save optimization history to disk."""
        history_file = self.storage_path / "optimization_history.json"
        try:
            data = {
                "results": [r.to_dict() for r in self._history[-500:]],  # Keep last 500
                "best_params": self._best_params,
                "best_score": self._best_score,
                "last_updated": datetime.utcnow().isoformat(),
            }
            with open(history_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug("Saved optimization history")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    async def start(self):
        """Start the optimization scheduler."""
        if self._running:
            logger.warning("Optimizer already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            f"Started {self.frequency.value} optimization scheduler "
            f"for profile '{self.profile.name}'"
        )

    async def stop(self):
        """Stop the optimization scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._save_history()
        logger.info("Stopped optimization scheduler")

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                # Calculate next run time
                next_run = self._get_next_run_time()
                wait_seconds = (next_run - datetime.utcnow()).total_seconds()

                if wait_seconds > 0:
                    logger.info(f"Next optimization run at {next_run.isoformat()}")
                    await asyncio.sleep(min(wait_seconds, 60))  # Check every minute
                    continue

                # Run optimization
                await self.run_optimization()

                # Store result
                self._save_history()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(300)  # Wait 5 min on error

    def _get_next_run_time(self) -> datetime:
        """Calculate the next run time based on frequency."""
        now = datetime.utcnow()

        if self.frequency == OptimizationFrequency.HOURLY:
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        elif self.frequency == OptimizationFrequency.EVERY_6_HOURS:
            next_hour = (now.hour // 6 + 1) * 6
            if next_hour >= 24:
                return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            return now.replace(hour=next_hour, minute=0, second=0, microsecond=0)

        elif self.frequency == OptimizationFrequency.EVERY_12_HOURS:
            next_hour = (now.hour // 12 + 1) * 12
            if next_hour >= 24:
                return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            return now.replace(hour=next_hour, minute=0, second=0, microsecond=0)

        elif self.frequency == OptimizationFrequency.DAILY:
            # Parse schedule time
            hour, minute = map(int, self.schedule_time.split(":"))
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run

        elif self.frequency == OptimizationFrequency.WEEKLY:
            # Run on Monday at schedule time
            hour, minute = map(int, self.schedule_time.split(":"))
            days_until_monday = (7 - now.weekday()) % 7 or 7
            next_run = (now + timedelta(days=days_until_monday)).replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            return next_run

        return now + timedelta(hours=1)  # Default to hourly

    async def run_optimization(
        self,
        profile: OptimizationProfile | None = None,
        market_metrics: Any | None = None,
    ) -> OptimizationResult:
        """Run an optimization with the given profile.

        Args:
            profile: Override profile for this run
            market_metrics: Optional market data for regime detection

        Returns:
            OptimizationResult with optimized parameters
        """
        profile = profile or self.profile
        start_time = datetime.utcnow()

        logger.info(f"Starting optimization with profile '{profile.name}'")

        result = OptimizationResult(
            timestamp=start_time.isoformat(),
            profile_name=profile.name,
            success=False,
        )

        try:
            orchestrator = self._get_orchestrator()

            # Run smart optimization
            opt_result = await orchestrator.smart_optimize(
                risk_tolerance=profile.risk_tolerance,
                primary_goal=profile.primary_goal,
                capital_sol=profile.capital_sol,
                market_metrics=market_metrics,
            )

            result.success = True
            result.best_params = opt_result.best_params
            result.best_metrics = opt_result.best_metrics
            result.best_score = opt_result.best_score
            result.sharpe_ratio = opt_result.best_metrics.get("sharpe_ratio", 0)
            result.win_rate_pct = opt_result.best_metrics.get("win_rate_pct", 0)
            result.max_drawdown_pct = opt_result.best_metrics.get("max_drawdown_pct", 0)
            result.optimization_time_seconds = opt_result.optimization_time_seconds
            result.iterations = opt_result.optimization_iterations
            result.robustness_score = opt_result.robustness_score
            result.is_robust = opt_result.is_robust

            # Calculate improvement over previous
            if self._last_result and self._last_result.best_score > 0:
                result.improvement_over_previous = (
                    (result.best_score - self._last_result.best_score)
                    / self._last_result.best_score
                    * 100
                )

            # Update best if improved
            if result.best_score > self._best_score:
                self._best_score = result.best_score
                self._best_params = result.best_params
                logger.info(f"New best score: {result.best_score:.4f}")

            # Run robustness validation if enabled
            if profile.enable_robustness_validation and result.is_robust is None:
                validator = self._get_robustness_validator()
                if validator:
                    # Would need historical data for full validation
                    result.robustness_score = opt_result.robustness_score
                    result.is_robust = (
                        result.robustness_score
                        and result.robustness_score >= profile.robustness_threshold
                    )

            logger.info(
                f"Optimization complete: score={result.best_score:.4f}, "
                f"sharpe={result.sharpe_ratio:.2f}, "
                f"robust={result.is_robust}"
            )

        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Optimization failed: {e}")

        # Store result
        self._last_run = start_time
        self._last_result = result
        self._history.append(result)

        # Callback
        if self.on_optimization_complete:
            try:
                await self.on_optimization_complete(result)
            except Exception as e:
                logger.warning(f"Callback failed: {e}")

        return result

    def get_best_params(self) -> dict[str, Any]:
        """Get the best parameters found so far."""
        return self._best_params.copy()

    def get_last_result(self) -> OptimizationResult | None:
        """Get the last optimization result."""
        return self._last_result

    def get_history(self, limit: int = 50) -> list[OptimizationResult]:
        """Get optimization history."""
        return self._history[-limit:]

    def get_statistics(self) -> dict[str, Any]:
        """Get optimizer statistics."""
        if not self._history:
            return {"total_runs": 0}

        successful = [r for r in self._history if r.success]

        return {
            "total_runs": len(self._history),
            "successful_runs": len(successful),
            "best_score": self._best_score,
            "avg_score": sum(r.best_score for r in successful) / max(len(successful), 1),
            "avg_sharpe": sum(r.sharpe_ratio for r in successful) / max(len(successful), 1),
            "avg_win_rate": sum(r.win_rate_pct for r in successful) / max(len(successful), 1),
            "avg_drawdown": sum(r.max_drawdown_pct for r in successful) / max(len(successful), 1),
            "robust_runs": sum(1 for r in successful if r.is_robust),
            "profile": self.profile.name,
            "frequency": self.frequency.value,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "running": self._running,
        }

    def set_profile(self, profile: OptimizationProfile):
        """Change the optimization profile."""
        self.profile = profile
        self._orchestrator = None  # Reset to pick up new config
        logger.info(f"Changed to profile: {profile.name}")

    async def run_all_profiles(
        self,
        profiles: list[OptimizationProfile] | None = None,
    ) -> dict[str, OptimizationResult]:
        """Run optimization for multiple profiles and compare.

        Args:
            profiles: List of profiles to run (defaults to all DEFAULT_PROFILES)

        Returns:
            Dict mapping profile name to result
        """
        profiles = profiles or list(DEFAULT_PROFILES.values())
        results = {}

        for profile in profiles:
            logger.info(f"Running profile: {profile.name}")
            result = await self.run_optimization(profile=profile)
            results[profile.name] = result

        # Find best
        best_profile = max(results.items(), key=lambda x: x[1].best_score)
        logger.info(f"Best profile: {best_profile[0]} with score {best_profile[1].best_score:.4f}")

        return results


# Convenience function
def create_daily_optimizer(
    backtest_runner: Callable,
    profile_name: str = "profit_maximizer",
    frequency: str = "daily",
    **kwargs,
) -> DailyOptimizer:
    """Create a daily optimizer with pre-configured profile.

    Args:
        backtest_runner: Function to run backtests
        profile_name: Name of profile from DEFAULT_PROFILES
        frequency: "hourly", "daily", "weekly", etc.
        **kwargs: Additional arguments for DailyOptimizer

    Returns:
        Configured DailyOptimizer
    """
    profile = DEFAULT_PROFILES.get(profile_name, DEFAULT_PROFILES["profit_maximizer"])

    freq_map = {
        "hourly": OptimizationFrequency.HOURLY,
        "every_6_hours": OptimizationFrequency.EVERY_6_HOURS,
        "every_12_hours": OptimizationFrequency.EVERY_12_HOURS,
        "daily": OptimizationFrequency.DAILY,
        "weekly": OptimizationFrequency.WEEKLY,
    }

    return DailyOptimizer(
        backtest_runner=backtest_runner,
        profile=profile,
        frequency=freq_map.get(frequency, OptimizationFrequency.DAILY),
        **kwargs,
    )
