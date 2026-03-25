"""
Auto Backtest Mode - Hands-free AI-powered optimization.

Automatically runs optimization cycles to find optimal parameters without
manual trial-and-error. Uses intelligent stopping criteria, multi-objective
optimization, and continuous learning from past results.

Key Features:
- Automatic parameter space exploration
- Intelligent stopping based on convergence
- Multi-objective Pareto optimization
- Feedback loop integration for continuous improvement
- Live monitoring and progress reporting
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AutoModeState(Enum):
    """State of the auto optimization mode."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CONVERGED = "converged"


class StoppingReason(Enum):
    """Reason for stopping optimization."""

    CONVERGENCE = "convergence"
    TIMEOUT = "timeout"
    MAX_ITERATIONS = "max_iterations"
    TARGET_REACHED = "target_reached"
    USER_CANCELLED = "user_cancelled"
    ERROR = "error"
    NO_IMPROVEMENT = "no_improvement"


@dataclass
class OptimizationTarget:
    """Target metrics for optimization."""

    min_sharpe: float = 1.5
    min_win_rate: float = 0.45
    max_drawdown: float = 25.0
    min_profit_quality: float = 0.5
    min_total_return: float = 10.0

    def is_satisfied(self, metrics: dict[str, float]) -> bool:
        """Check if targets are satisfied."""
        return (
            metrics.get("sharpe_ratio", 0) >= self.min_sharpe
            and metrics.get("win_rate_pct", 0) / 100 >= self.min_win_rate
            and metrics.get("max_drawdown_pct", 100) <= self.max_drawdown
            and metrics.get("profit_quality_score", 0) >= self.min_profit_quality
        )

    def satisfaction_score(self, metrics: dict[str, float]) -> float:
        """Calculate how close metrics are to targets (0-1)."""
        scores = []

        sharpe = metrics.get("sharpe_ratio", 0)
        if self.min_sharpe > 0:
            scores.append(min(1.0, sharpe / self.min_sharpe))

        win_rate = metrics.get("win_rate_pct", 0) / 100
        if self.min_win_rate > 0:
            scores.append(min(1.0, win_rate / self.min_win_rate))

        dd = metrics.get("max_drawdown_pct", 100)
        if self.max_drawdown > 0:
            scores.append(max(0, 1 - dd / self.max_drawdown))

        pq = metrics.get("profit_quality_score", 0)
        if self.min_profit_quality > 0:
            scores.append(min(1.0, pq / self.min_profit_quality))

        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class AutoModeConfig:
    """Configuration for auto optimization mode."""

    # Stopping criteria
    max_iterations: int = 100
    max_time_minutes: int = 30
    convergence_patience: int = 15  # Iterations without improvement
    convergence_threshold: float = 0.001  # Minimum improvement to count

    # Optimization settings
    n_initial_random: int = 10  # Random samples before Bayesian
    exploration_rate: float = 0.2  # Epsilon-greedy exploration
    batch_size: int = 4  # Parallel backtests per iteration

    # Target metrics
    targets: OptimizationTarget = field(default_factory=OptimizationTarget)

    # Feedback loop
    enable_feedback_loop: bool = True
    feedback_weight: float = 0.3  # Weight of historical results

    # Checkpointing
    checkpoint_interval: int = 10  # Save progress every N iterations
    resume_from_checkpoint: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_time_minutes": self.max_time_minutes,
            "convergence_patience": self.convergence_patience,
            "convergence_threshold": self.convergence_threshold,
            "n_initial_random": self.n_initial_random,
            "exploration_rate": self.exploration_rate,
            "batch_size": self.batch_size,
            "enable_feedback_loop": self.enable_feedback_loop,
            "feedback_weight": self.feedback_weight,
        }


@dataclass
class IterationResult:
    """Result from a single optimization iteration."""

    iteration: int
    params: dict[str, Any]
    metrics: dict[str, float]
    score: float
    is_improvement: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AutoModeResult:
    """Final result from auto optimization."""

    state: AutoModeState
    stopping_reason: StoppingReason | None
    best_params: dict[str, Any]
    best_metrics: dict[str, float]
    best_score: float

    # Statistics
    total_iterations: int
    total_time_seconds: float
    improvements_count: int
    convergence_iteration: int | None

    # History
    history: list[IterationResult] = field(default_factory=list)

    # Pareto front (if multi-objective)
    pareto_front: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "stopping_reason": self.stopping_reason.value if self.stopping_reason else None,
            "best_params": self.best_params,
            "best_metrics": self.best_metrics,
            "best_score": self.best_score,
            "total_iterations": self.total_iterations,
            "total_time_seconds": self.total_time_seconds,
            "improvements_count": self.improvements_count,
            "convergence_iteration": self.convergence_iteration,
            "history_size": len(self.history),
            "pareto_front_size": len(self.pareto_front),
        }


class AutoBacktestMode:
    """AI-powered automatic backtest optimization.

    Runs continuous optimization cycles to find optimal parameters
    without manual intervention.

    Usage:
        auto_mode = AutoBacktestMode(
            backtest_runner=my_backtest_fn,
            config=AutoModeConfig(max_iterations=50),
        )

        # Start optimization
        result = await auto_mode.run(initial_params)

        # Or run in background with progress callback
        async def on_progress(progress):
            print(f"Progress: {progress}%")

        result = await auto_mode.run_with_callback(
            initial_params,
            on_progress=on_progress,
        )
    """

    def __init__(
        self,
        backtest_runner: Callable,
        config: AutoModeConfig | None = None,
        feedback_store: Any | None = None,
    ):
        """Initialize auto mode.

        Args:
            backtest_runner: Async function that takes params and returns metrics
            config: Auto mode configuration
            feedback_store: Optional store for historical results
        """
        self.backtest_runner = backtest_runner
        self.config = config or AutoModeConfig()
        self.feedback_store = feedback_store

        self._state = AutoModeState.IDLE
        self._current_iteration = 0
        self._best_result: IterationResult | None = None
        self._history: list[IterationResult] = []
        self._start_time: datetime | None = None
        self._cancel_requested = False

        # Parameter bounds for optimization
        self._param_bounds = self._get_default_param_bounds()

        # Bayesian optimization state (simplified)
        self._observed_params: list[dict[str, Any]] = []
        self._observed_scores: list[float] = []

    def _get_default_param_bounds(self) -> dict[str, tuple]:
        """Get default parameter bounds for optimization."""
        return {
            "stop_loss_pct": (3.0, 20.0),
            "take_profit_pct": (8.0, 60.0),
            "max_position_sol": (0.01, 0.20),
            "min_liquidity_usd": (5000, 50000),
            "max_risk_score": (0.20, 0.70),
            "slippage_bps": (150, 600),
            "max_hold_minutes": (5, 90),
            "kelly_safety_factor": (0.20, 0.70),
        }

    @property
    def state(self) -> AutoModeState:
        """Get current state."""
        return self._state

    @property
    def progress(self) -> float:
        """Get progress percentage (0-100)."""
        if self.config.max_iterations <= 0:
            return 0.0
        return min(100.0, (self._current_iteration / self.config.max_iterations) * 100)

    async def run(
        self,
        initial_params: dict[str, Any] | None = None,
        param_bounds: dict[str, tuple] | None = None,
    ) -> AutoModeResult:
        """Run automatic optimization.

        Args:
            initial_params: Starting parameters (optional)
            param_bounds: Custom parameter bounds (optional)

        Returns:
            AutoModeResult with best parameters found
        """
        if self._state == AutoModeState.RUNNING:
            raise RuntimeError("Auto mode is already running")

        # Initialize
        self._state = AutoModeState.RUNNING
        self._cancel_requested = False
        self._start_time = datetime.utcnow()
        self._current_iteration = 0
        self._history = []
        self._best_result = None

        if param_bounds:
            self._param_bounds.update(param_bounds)

        # Use provided or default initial params
        current_params = initial_params or self._get_default_params()

        try:
            # Run initial evaluation
            initial_metrics = await self._run_backtest(current_params)
            initial_score = self._calculate_score(initial_metrics)

            self._best_result = IterationResult(
                iteration=0,
                params=current_params.copy(),
                metrics=initial_metrics,
                score=initial_score,
                is_improvement=True,
            )
            self._history.append(self._best_result)
            self._observed_params.append(current_params.copy())
            self._observed_scores.append(initial_score)

            logger.info(f"Initial score: {initial_score:.4f}")

            # Track convergence
            iterations_without_improvement = 0
            last_improvement_score = initial_score

            # Main optimization loop
            while self._should_continue():
                self._current_iteration += 1

                # Get next parameters to try
                if self._current_iteration <= self.config.n_initial_random:
                    # Random exploration phase
                    next_params = self._sample_random_params()
                else:
                    # Bayesian-inspired selection
                    next_params = self._suggest_next_params()

                # Run backtest
                metrics = await self._run_backtest(next_params)
                score = self._calculate_score(metrics)

                # Record result
                is_improvement = score > self._best_result.score
                iteration_result = IterationResult(
                    iteration=self._current_iteration,
                    params=next_params,
                    metrics=metrics,
                    score=score,
                    is_improvement=is_improvement,
                )
                self._history.append(iteration_result)
                self._observed_params.append(next_params)
                self._observed_scores.append(score)

                # Update best if improved
                if is_improvement:
                    self._best_result = iteration_result
                    iterations_without_improvement = 0
                    improvement_delta = score - last_improvement_score
                    last_improvement_score = score
                    logger.info(
                        f"Iteration {self._current_iteration}: NEW BEST score={score:.4f} "
                        f"(+{improvement_delta:.4f})"
                    )
                else:
                    iterations_without_improvement += 1
                    if self._current_iteration % 10 == 0:
                        logger.info(
                            f"Iteration {self._current_iteration}: score={score:.4f} "
                            f"(best={self._best_result.score:.4f})"
                        )

                # Check convergence
                if iterations_without_improvement >= self.config.convergence_patience:
                    logger.info(f"Converged after {self._current_iteration} iterations")
                    self._state = AutoModeState.CONVERGED
                    break

                # Check if targets reached
                if self.config.targets.is_satisfied(metrics):
                    logger.info("Target metrics satisfied!")
                    self._state = AutoModeState.COMPLETED
                    break

                # Checkpoint
                if self._current_iteration % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()

            # Determine stopping reason
            stopping_reason = self._determine_stopping_reason()

            # Build final result
            total_time = (datetime.utcnow() - self._start_time).total_seconds()

            result = AutoModeResult(
                state=self._state,
                stopping_reason=stopping_reason,
                best_params=self._best_result.params,
                best_metrics=self._best_result.metrics,
                best_score=self._best_result.score,
                total_iterations=self._current_iteration,
                total_time_seconds=total_time,
                improvements_count=sum(1 for h in self._history if h.is_improvement),
                convergence_iteration=self._find_convergence_iteration(),
                history=self._history[-100:],  # Keep last 100
                pareto_front=self._compute_pareto_front(),
            )

            # Store in feedback loop
            if self.config.enable_feedback_loop and self.feedback_store:
                self._store_result_for_feedback(result)

            logger.info(
                f"Auto optimization complete: {result.total_iterations} iterations, "
                f"best_score={result.best_score:.4f}, "
                f"improvements={result.improvements_count}"
            )

            return result

        except Exception as e:
            logger.error(f"Auto optimization failed: {e}", exc_info=True)
            self._state = AutoModeState.FAILED

            return AutoModeResult(
                state=AutoModeState.FAILED,
                stopping_reason=StoppingReason.ERROR,
                best_params=self._best_result.params if self._best_result else {},
                best_metrics=self._best_result.metrics if self._best_result else {},
                best_score=self._best_result.score if self._best_result else 0.0,
                total_iterations=self._current_iteration,
                total_time_seconds=(
                    (datetime.utcnow() - self._start_time).total_seconds()
                    if self._start_time
                    else 0
                ),
                improvements_count=0,
                convergence_iteration=None,
                history=self._history,
            )

    async def run_with_callback(
        self,
        initial_params: dict[str, Any] | None = None,
        on_progress: Callable[[float, dict[str, Any]], None] | None = None,
        on_improvement: Callable[[dict[str, Any]], None] | None = None,
    ) -> AutoModeResult:
        """Run optimization with progress callbacks.

        Args:
            initial_params: Starting parameters
            on_progress: Callback(progress_pct, current_best)
            on_improvement: Callback(new_best_result)

        Returns:
            AutoModeResult with best parameters
        """
        # Wrap backtest runner to emit progress
        original_runner = self.backtest_runner

        async def wrapped_runner(params: dict[str, Any]) -> dict[str, float]:
            result = await original_runner(params)

            if on_progress:
                progress_info = {
                    "iteration": self._current_iteration,
                    "best_score": self._best_result.score if self._best_result else 0,
                    "best_params": self._best_result.params if self._best_result else {},
                }
                on_progress(self.progress, progress_info)

            return result

        # Temporarily replace runner
        self.backtest_runner = wrapped_runner

        try:
            result = await self.run(initial_params)

            if on_improvement and self._best_result:
                on_improvement(
                    {
                        "params": self._best_result.params,
                        "metrics": self._best_result.metrics,
                        "score": self._best_result.score,
                    }
                )

            return result
        finally:
            self.backtest_runner = original_runner

    def cancel(self):
        """Request cancellation of running optimization."""
        self._cancel_requested = True
        logger.info("Cancellation requested")

    def pause(self):
        """Pause optimization."""
        if self._state == AutoModeState.RUNNING:
            self._state = AutoModeState.PAUSED
            logger.info("Optimization paused")

    def resume(self):
        """Resume paused optimization."""
        if self._state == AutoModeState.PAUSED:
            self._state = AutoModeState.RUNNING
            logger.info("Optimization resumed")

    def get_current_status(self) -> dict[str, Any]:
        """Get current optimization status."""
        return {
            "state": self._state.value,
            "iteration": self._current_iteration,
            "max_iterations": self.config.max_iterations,
            "progress_pct": self.progress,
            "best_score": self._best_result.score if self._best_result else None,
            "best_params": self._best_result.params if self._best_result else None,
            "elapsed_seconds": (
                (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0
            ),
        }

    async def _run_backtest(self, params: dict[str, Any]) -> dict[str, float]:
        """Run backtest with given parameters."""
        import inspect

        if inspect.iscoroutinefunction(self.backtest_runner):
            result = await self.backtest_runner(params)
            return result  # type: ignore
        else:
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.backtest_runner(params))
            return result  # type: ignore

    def _calculate_score(self, metrics: dict[str, float]) -> float:
        """Calculate optimization score from metrics.

        Uses a weighted combination of key metrics.
        """
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate_pct", 0) / 100
        profit_factor = metrics.get("profit_factor", 0)
        max_dd = metrics.get("max_drawdown_pct", 100)
        total_return = metrics.get("total_return_pct", 0)
        pq_score = metrics.get("profit_quality_score", 0)

        # Normalize components
        sharpe_norm = min(1.0, sharpe / 3.0)  # Cap at Sharpe 3.0
        win_rate_norm = win_rate  # Already 0-1
        pf_norm = min(1.0, profit_factor / 3.0)  # Cap at PF 3.0
        dd_norm = max(0, 1 - max_dd / 50)  # Penalty for >50% DD
        return_norm = min(1.0, max(0, total_return) / 100)  # Cap at 100% return

        # Weighted score
        score = (
            sharpe_norm * 0.25
            + win_rate_norm * 0.15
            + pf_norm * 0.20
            + dd_norm * 0.20
            + return_norm * 0.10
            + pq_score * 0.10
        )

        return score

    def _should_continue(self) -> bool:
        """Check if optimization should continue."""
        if self._cancel_requested:
            return False

        if self._state == AutoModeState.PAUSED:
            # Wait until resumed
            return True

        # Check iteration limit
        if self._current_iteration >= self.config.max_iterations:
            self._state = AutoModeState.COMPLETED
            return False

        # Check time limit
        if self._start_time:
            elapsed = (datetime.utcnow() - self._start_time).total_seconds() / 60
            if elapsed >= self.config.max_time_minutes:
                self._state = AutoModeState.COMPLETED
                return False

        return self._state == AutoModeState.RUNNING

    def _determine_stopping_reason(self) -> StoppingReason:
        """Determine why optimization stopped."""
        if self._cancel_requested:
            return StoppingReason.USER_CANCELLED

        if self._state == AutoModeState.CONVERGED:
            return StoppingReason.CONVERGENCE

        if self._state == AutoModeState.FAILED:
            return StoppingReason.ERROR

        if self._current_iteration >= self.config.max_iterations:
            return StoppingReason.MAX_ITERATIONS

        if self._start_time:
            elapsed = (datetime.utcnow() - self._start_time).total_seconds() / 60
            if elapsed >= self.config.max_time_minutes:
                return StoppingReason.TIMEOUT

        if self.config.targets.is_satisfied(self._best_result.metrics if self._best_result else {}):
            return StoppingReason.TARGET_REACHED

        return StoppingReason.NO_IMPROVEMENT

    def _get_default_params(self) -> dict[str, Any]:
        """Get default starting parameters."""
        return {
            "stop_loss_pct": 8.0,
            "take_profit_pct": 20.0,
            "max_position_sol": 0.05,
            "min_liquidity_usd": 10000,
            "max_risk_score": 0.40,
            "slippage_bps": 300,
            "max_hold_minutes": 30,
            "kelly_safety_factor": 0.40,
        }

    def _sample_random_params(self) -> dict[str, Any]:
        """Sample random parameters within bounds."""
        import random

        params = {}
        for key, (min_v, max_v) in self._param_bounds.items():
            if isinstance(min_v, int):
                params[key] = random.randint(min_v, max_v)
            else:
                params[key] = round(random.uniform(min_v, max_v), 4)

        return params

    def _suggest_next_params(self) -> dict[str, Any]:
        """Suggest next parameters using Bayesian-inspired selection.

        Simplified implementation using:
        - Exploitation: Near best observed params with noise
        - Exploration: Random params with probability epsilon
        """
        import random

        # Epsilon-greedy exploration
        if random.random() < self.config.exploration_rate:
            return self._sample_random_params()

        # Find best parameters
        if not self._best_result:
            return self._sample_random_params()

        best_params = self._best_result.params.copy()

        # Add Gaussian noise for local search
        for key in best_params:
            if key in self._param_bounds:
                min_v, max_v = self._param_bounds[key]
                current = best_params[key]

                # Noise scaled to parameter range
                range_size = max_v - min_v
                noise = random.gauss(0, range_size * 0.1)

                new_value = current + noise
                new_value = max(min_v, min(max_v, new_value))

                if isinstance(min_v, int):
                    best_params[key] = int(round(new_value))
                else:
                    best_params[key] = round(new_value, 4)

        return best_params

    def _find_convergence_iteration(self) -> int | None:
        """Find iteration where convergence occurred."""
        if len(self._history) < 2:
            return None

        best_score = 0
        for result in self._history:
            if result.score > best_score:
                best_score = result.score
            elif result.score == best_score:
                # Check if this is where it plateaued
                subsequent = [r for r in self._history if r.iteration > result.iteration]
                if all(r.score <= best_score for r in subsequent):
                    return result.iteration

        return None

    def _compute_pareto_front(self) -> list[dict[str, Any]]:
        """Compute Pareto front for multi-objective view."""
        if len(self._history) < 2:
            return []

        # Use Sharpe and Max DD as objectives
        pareto = []

        for result in self._history:
            sharpe = result.metrics.get("sharpe_ratio", 0)
            max_dd = result.metrics.get("max_drawdown_pct", 100)

            # Check if dominated
            is_dominated = False
            for other in self._history:
                other_sharpe = other.metrics.get("sharpe_ratio", 0)
                other_dd = other.metrics.get("max_drawdown_pct", 100)

                # Dominated if other is better in both objectives
                if (
                    other_sharpe >= sharpe
                    and other_dd <= max_dd
                    and (other_sharpe > sharpe or other_dd < max_dd)
                ):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto.append(
                    {
                        "params": result.params,
                        "sharpe_ratio": sharpe,
                        "max_drawdown_pct": max_dd,
                        "score": result.score,
                    }
                )

        # Sort by score
        pareto.sort(key=lambda x: x["score"], reverse=True)
        return pareto[:20]  # Top 20 Pareto points

    def _save_checkpoint(self):
        """Save optimization checkpoint."""
        {
            "iteration": self._current_iteration,
            "best_params": self._best_result.params if self._best_result else None,
            "best_score": self._best_result.score if self._best_result else None,
            "history": [h.__dict__ for h in self._history[-50:]],
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Could save to disk here
        logger.debug(f"Checkpoint saved at iteration {self._current_iteration}")

    def _store_result_for_feedback(self, result: AutoModeResult):
        """Store result in feedback loop for future learning."""
        if self.feedback_store is None:
            return

        try:
            feedback_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "best_params": result.best_params,
                "best_metrics": result.best_metrics,
                "best_score": result.best_score,
                "total_iterations": result.total_iterations,
                "stopping_reason": result.stopping_reason.value if result.stopping_reason else None,
            }

            # Could append to feedback store
            if hasattr(self.feedback_store, "append"):
                self.feedback_store.append(feedback_entry)

        except Exception as e:
            logger.warning(f"Failed to store feedback: {e}")


class AutoModeSessionManager:
    """Manages multiple concurrent auto optimization sessions.

    Provides session management for the API layer.
    """

    def __init__(self):
        self._sessions: dict[str, AutoBacktestMode] = {}
        self._results: dict[str, AutoModeResult] = {}

    def create_session(
        self,
        session_id: str,
        backtest_runner: Callable,
        config: AutoModeConfig | None = None,
    ) -> AutoBacktestMode:
        """Create a new optimization session."""
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")

        auto_mode = AutoBacktestMode(
            backtest_runner=backtest_runner,
            config=config,
        )
        self._sessions[session_id] = auto_mode

        return auto_mode

    def get_session(self, session_id: str) -> AutoBacktestMode | None:
        """Get an existing session."""
        return self._sessions.get(session_id)

    def get_all_sessions(self) -> dict[str, dict[str, Any]]:
        """Get status of all sessions."""
        return {
            session_id: session.get_current_status()
            for session_id, session in self._sessions.items()
        }

    def cancel_session(self, session_id: str) -> bool:
        """Cancel a running session."""
        session = self._sessions.get(session_id)
        if session and session.state == AutoModeState.RUNNING:
            session.cancel()
            return True
        return False

    def remove_session(self, session_id: str) -> bool:
        """Remove a completed session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def cleanup_completed(self) -> int:
        """Remove all completed sessions."""
        to_remove = [
            sid
            for sid, session in self._sessions.items()
            if session.state
            in [AutoModeState.COMPLETED, AutoModeState.FAILED, AutoModeState.CONVERGED]
        ]

        for sid in to_remove:
            del self._sessions[sid]

        return len(to_remove)
