"""
Rollback manager for deployed parameter monitoring and auto-revert.

Monitors trading performance after parameter deployment and automatically
rolls back to previous parameters if performance degrades beyond thresholds.

Monitoring window: 24 hours (configurable)
Check frequency: Every 5 minutes (configurable)

Rollback triggers (any condition):
- Sharpe ratio < 50% of baseline
- Win rate < 40%
- Drawdown > baseline * 1.3
- 5+ consecutive losses

Usage:
    manager = RollbackManager(parameter_store, performance_engine)
    asyncio.create_task(
        manager.monitor_deployment(deployment_id)
    )
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class RollbackThresholds:
    """Configurable thresholds for rollback triggers."""

    # Sharpe ratio threshold: rollback if recent Sharpe < baseline * this
    min_sharpe_ratio_factor: float = 0.5  # 50% of baseline

    # Win rate absolute threshold
    min_win_rate: float = 0.40  # 40%

    # Drawdown threshold: rollback if drawdown > baseline * this
    max_drawdown_factor: float = 1.3  # 130% of baseline drawdown

    # Consecutive losses threshold
    max_consecutive_losses: int = 5

    # Minimum trades before evaluation (avoids premature rollback)
    min_trades_before_evaluation: int = 5

    # Maximum negative P&L before forced rollback (SOL)
    max_negative_pnl_sol: float = -2.0


@dataclass
class DeploymentMonitor:
    """State for a monitored deployment."""

    deployment_id: str
    optimization_id: str | None
    params_version: int
    started_at: datetime
    monitoring_window_hours: int
    baseline_sharpe: float
    baseline_win_rate: float
    baseline_drawdown: float

    # Monitoring state
    trades_since_deployment: int = 0
    current_pnl_sol: float = 0.0
    current_sharpe: float = 0.0
    current_win_rate: float = 0.0
    current_drawdown: float = 0.0
    consecutive_losses: int = 0
    check_count: int = 0
    last_check_at: datetime | None = None
    rolled_back: bool = False
    rollback_reason: str | None = None
    completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "optimization_id": self.optimization_id,
            "params_version": self.params_version,
            "started_at": self.started_at.isoformat(),
            "monitoring_window_hours": self.monitoring_window_hours,
            "baseline_sharpe": round(self.baseline_sharpe, 4),
            "baseline_win_rate": round(self.baseline_win_rate, 4),
            "baseline_drawdown": round(self.baseline_drawdown, 4),
            "trades_since_deployment": self.trades_since_deployment,
            "current_pnl_sol": round(self.current_pnl_sol, 6),
            "current_sharpe": round(self.current_sharpe, 4),
            "current_win_rate": round(self.current_win_rate, 4),
            "current_drawdown": round(self.current_drawdown, 4),
            "consecutive_losses": self.consecutive_losses,
            "check_count": self.check_count,
            "last_check_at": (self.last_check_at.isoformat() if self.last_check_at else None),
            "rolled_back": self.rolled_back,
            "rollback_reason": self.rollback_reason,
            "completed": self.completed,
        }


class RollbackManager:
    """
    Monitor deployed parameters and auto-rollback if performance degrades.

    After parameter deployment, this manager:
    1. Starts a background monitoring task
    2. Checks performance every check_interval_seconds
    3. Evaluates rollback conditions against thresholds
    4. Automatically rolls back if any condition triggers
    5. Notifies via callbacks when rollback occurs

    Multiple deployments can be monitored simultaneously (though typically
    only one is active at a time via the AIStrategyOrchestrator).
    """

    def __init__(
        self,
        parameter_store: Any = None,
        performance_engine: Any = None,
        thresholds: RollbackThresholds | None = None,
        check_interval_seconds: float = 300.0,  # 5 minutes
        task_router: Any = None,
        cost_tracker: Any = None,
    ) -> None:
        """
        Initialize the rollback manager.

        Args:
            parameter_store: ParameterStore for rollback operations.
            performance_engine: PerformanceFeedbackEngine for metrics.
            thresholds: Configurable rollback thresholds.
            check_interval_seconds: How often to check (default 300s = 5 min).
            task_router: TaskComplexityRouter for hybrid Opus/Sonnet routing.
            cost_tracker: CostTracker for tracking AI call costs.
        """
        self.parameter_store = parameter_store
        self.performance_engine = performance_engine
        self.thresholds = thresholds or RollbackThresholds()
        self.check_interval_seconds = check_interval_seconds
        self.task_router = task_router
        self.cost_tracker = cost_tracker

        # Active monitors
        self._monitors: dict[str, DeploymentMonitor] = {}
        self._monitor_tasks: dict[str, asyncio.Task] = {}

        # Callbacks
        self._on_rollback: list[Callable[[str, str, dict[str, Any]], None]] = []
        self._on_monitoring_complete: list[Callable[[str, dict[str, Any]], None]] = []

        self._lock = asyncio.Lock()

    async def monitor_deployment(
        self,
        deployment_id: str,
        optimization_id: str | None = None,
        params_version: int = 0,
        monitoring_window_hours: int = 24,
        baseline_sharpe: float = 0.0,
        baseline_win_rate: float = 0.5,
        baseline_drawdown: float = 0.0,
    ) -> None:
        """
        Start monitoring a deployment for performance degradation.

        Runs in the background, checking every check_interval_seconds.
        Automatically stops after monitoring_window_hours.

        Args:
            deployment_id: Unique deployment identifier.
            optimization_id: Associated optimization run ID.
            params_version: Parameter version deployed.
            monitoring_window_hours: How long to monitor (default 24h).
            baseline_sharpe: Sharpe ratio before deployment.
            baseline_win_rate: Win rate before deployment.
            baseline_drawdown: Max drawdown before deployment.
        """
        monitor = DeploymentMonitor(
            deployment_id=deployment_id,
            optimization_id=optimization_id,
            params_version=params_version,
            started_at=datetime.now(),
            monitoring_window_hours=monitoring_window_hours,
            baseline_sharpe=baseline_sharpe,
            baseline_win_rate=baseline_win_rate,
            baseline_drawdown=baseline_drawdown,
        )

        async with self._lock:
            self._monitors[deployment_id] = monitor

        # Start background monitoring task
        task = asyncio.create_task(
            self._monitoring_loop(deployment_id),
            name=f"rollback-monitor-{deployment_id}",
        )
        self._monitor_tasks[deployment_id] = task

        logger.info(
            "Started monitoring deployment %s (window: %dh, "
            "baseline Sharpe: %.2f, baseline WR: %.1f%%)",
            deployment_id,
            monitoring_window_hours,
            baseline_sharpe,
            baseline_win_rate * 100,
        )

    async def _monitoring_loop(self, deployment_id: str) -> None:
        """Background loop that periodically checks deployment health."""
        monitor = self._monitors.get(deployment_id)
        if monitor is None:
            return

        end_time = monitor.started_at + timedelta(hours=monitor.monitoring_window_hours)

        try:
            while datetime.now() < end_time and not monitor.completed:
                await asyncio.sleep(self.check_interval_seconds)

                if monitor.completed:
                    break

                # Gather current performance metrics
                await self._update_monitor_metrics(monitor)

                # Check rollback conditions
                should_rollback, reason = self._check_rollback_conditions(monitor)

                monitor.check_count += 1
                monitor.last_check_at = datetime.now()

                if should_rollback:
                    await self.rollback(deployment_id, reason)
                    return

            # Monitoring window completed without rollback
            if not monitor.rolled_back:
                monitor.completed = True
                logger.info(
                    "Deployment %s monitoring completed successfully after %d checks",
                    deployment_id,
                    monitor.check_count,
                )
                for cb in self._on_monitoring_complete:
                    try:
                        cb(deployment_id, monitor.to_dict())
                    except Exception:
                        pass

                # Cleanup completed monitors to prevent unbounded dict growth
                self._cleanup_completed_monitors()

        except asyncio.CancelledError:
            logger.info("Monitoring task for %s was cancelled", deployment_id)
        except Exception as exc:
            logger.error(
                "Monitoring loop error for %s: %s",
                deployment_id,
                exc,
            )

    async def _update_monitor_metrics(
        self,
        monitor: DeploymentMonitor,
    ) -> None:
        """Update monitor metrics from the performance engine."""
        if self.performance_engine is None:
            return

        try:
            # Get recent outcomes since deployment
            outcomes = self.performance_engine.get_outcomes()
            recent = [o for o in outcomes if o.timestamp >= monitor.started_at]

            monitor.trades_since_deployment = len(recent)

            if not recent:
                return

            # Compute current metrics
            pnls = [o.pnl_pct for o in recent]
            sol_pnls = [o.pnl_sol for o in recent]
            wins = sum(1 for p in sol_pnls if p > 0)

            monitor.current_pnl_sol = sum(sol_pnls)
            monitor.current_win_rate = wins / len(recent) if recent else 0.0

            # Compute Sharpe
            import math

            if len(pnls) >= 2:
                mean = sum(pnls) / len(pnls)
                variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
                std = math.sqrt(variance) if variance > 0 else 0.001
                monitor.current_sharpe = mean / std
            else:
                monitor.current_sharpe = 0.0

            # Compute drawdown
            cumulative = 0.0
            peak = 0.0
            max_dd = 0.0
            for pnl in sol_pnls:
                cumulative += pnl
                if cumulative > peak:
                    peak = cumulative
                dd = peak - cumulative
                if dd > max_dd:
                    max_dd = dd
            monitor.current_drawdown = max_dd

            # Track consecutive losses
            monitor.consecutive_losses = 0
            for o in reversed(recent):
                if o.pnl_sol < 0:
                    monitor.consecutive_losses += 1
                else:
                    break

        except Exception as exc:
            logger.warning(
                "Failed to update monitor metrics for %s: %s",
                monitor.deployment_id,
                exc,
            )

    def _check_rollback_conditions(
        self,
        monitor: DeploymentMonitor,
    ) -> tuple[bool, str]:
        """
        Check if any rollback condition is triggered.

        Returns:
            (should_rollback, reason)
        """
        # Don't evaluate until we have enough trades
        if monitor.trades_since_deployment < self.thresholds.min_trades_before_evaluation:
            return False, ""

        # Condition 1: Sharpe degradation
        if monitor.baseline_sharpe > 0:
            sharpe_threshold = monitor.baseline_sharpe * self.thresholds.min_sharpe_ratio_factor
            if monitor.current_sharpe < sharpe_threshold:
                return True, (
                    f"Sharpe ratio {monitor.current_sharpe:.2f} is below "
                    f"threshold {sharpe_threshold:.2f} "
                    f"(baseline: {monitor.baseline_sharpe:.2f})"
                )

        # Condition 2: Win rate too low
        if monitor.current_win_rate < self.thresholds.min_win_rate:
            return True, (
                f"Win rate {monitor.current_win_rate:.1%} is below "
                f"threshold {self.thresholds.min_win_rate:.1%}"
            )

        # Condition 3: Drawdown exceeds threshold
        if monitor.baseline_drawdown > 0:
            dd_threshold = monitor.baseline_drawdown * self.thresholds.max_drawdown_factor
            if monitor.current_drawdown > dd_threshold:
                return True, (
                    f"Drawdown {monitor.current_drawdown:.4f} exceeds "
                    f"threshold {dd_threshold:.4f} "
                    f"(baseline: {monitor.baseline_drawdown:.4f})"
                )

        # Condition 4: Consecutive losses
        if monitor.consecutive_losses >= self.thresholds.max_consecutive_losses:
            return True, (
                f"{monitor.consecutive_losses} consecutive losses "
                f"(threshold: {self.thresholds.max_consecutive_losses})"
            )

        # Condition 5: Maximum negative P&L
        if monitor.current_pnl_sol < self.thresholds.max_negative_pnl_sol:
            return True, (
                f"Cumulative P&L {monitor.current_pnl_sol:.4f} SOL "
                f"exceeds max loss threshold "
                f"{self.thresholds.max_negative_pnl_sol} SOL"
            )

        return False, ""

    async def rollback(
        self,
        deployment_id: str,
        reason: str,
    ) -> bool:
        """
        Revert to previous parameters and notify.

        Args:
            deployment_id: The deployment to rollback.
            reason: Why the rollback is happening.

        Returns:
            True if rollback succeeded, False otherwise.
        """
        async with self._lock:
            monitor = self._monitors.get(deployment_id)
            if monitor is None:
                logger.warning(
                    "Cannot rollback unknown deployment: %s",
                    deployment_id,
                )
                return False

            if monitor.rolled_back:
                logger.info("Deployment %s already rolled back", deployment_id)
                return True

            logger.warning("ROLLING BACK deployment %s: %s", deployment_id, reason)

            # Perform the rollback
            if self.parameter_store:
                try:
                    await self.parameter_store.rollback(reason=f"Auto-rollback: {reason}")
                except Exception as exc:
                    logger.error(
                        "Parameter rollback failed for %s: %s",
                        deployment_id,
                        exc,
                    )
                    return False

            monitor.rolled_back = True
            monitor.rollback_reason = reason
            monitor.completed = True

            # Cancel the monitoring task
            task = self._monitor_tasks.get(deployment_id)
            if task and not task.done():
                task.cancel()

            # Notify callbacks
            for cb in self._on_rollback:
                try:
                    cb(deployment_id, reason, monitor.to_dict())
                except Exception:
                    pass

            logger.info(
                "Deployment %s rolled back successfully. "
                "Trades during deployment: %d, P&L: %.4f SOL",
                deployment_id,
                monitor.trades_since_deployment,
                monitor.current_pnl_sol,
            )

            return True

    async def cancel_monitoring(self, deployment_id: str) -> bool:
        """
        Cancel monitoring for a deployment without rolling back.

        Args:
            deployment_id: The deployment to stop monitoring.

        Returns:
            True if cancelled, False if not found.
        """
        monitor = self._monitors.get(deployment_id)
        if monitor is None:
            return False

        monitor.completed = True
        task = self._monitor_tasks.get(deployment_id)
        if task and not task.done():
            task.cancel()

        logger.info("Monitoring cancelled for deployment %s", deployment_id)
        return True

    def get_monitor_status(
        self,
        deployment_id: str,
    ) -> dict[str, Any] | None:
        """Get the current status of a monitored deployment."""
        monitor = self._monitors.get(deployment_id)
        if monitor is None:
            return None
        return monitor.to_dict()

    def get_all_monitors(self) -> dict[str, dict[str, Any]]:
        """Get status of all monitored deployments."""
        return {did: monitor.to_dict() for did, monitor in self._monitors.items()}

    def on_rollback(
        self,
        callback: Callable[[str, str, dict[str, Any]], None],
    ) -> None:
        """Register a callback for rollback events."""
        self._on_rollback.append(callback)

    def on_monitoring_complete(
        self,
        callback: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """Register a callback for successful monitoring completion."""
        self._on_monitoring_complete.append(callback)

    def _cleanup_completed_monitors(self) -> None:
        """Remove completed monitors and their tasks to prevent unbounded growth."""
        completed_ids = [did for did, m in self._monitors.items() if m.completed]
        for did in completed_ids:
            self._monitors.pop(did, None)
            task = self._monitor_tasks.pop(did, None)
            if task and not task.done():
                task.cancel()
        if completed_ids:
            logger.debug("Cleaned up %d completed monitors", len(completed_ids))

    @property
    def active_monitors(self) -> int:
        """Number of currently active monitors."""
        return sum(1 for m in self._monitors.values() if not m.completed)

    def get_routing_decision(self) -> Any | None:
        """
        Get a model routing decision for rollback-related AI calls.

        Rollback decisions are always high-stakes (financial risk),
        so they route to Opus 4.6 by default.

        Returns:
            ModelRoutingDecision if task_router is available, else None.
        """
        if self.task_router is None:
            return None

        decision = self.task_router.route_task(
            task_type="rollback_decision",
            num_parameters=0,
            has_constraints=True,
            requires_explanation=True,
            is_high_stakes=True,
        )

        return decision

    async def record_rollback_cost(
        self,
        deployment_id: str,
        action: str = "monitor_check",
        tokens_input: int = 200,
        tokens_output: int = 400,
        latency_ms: int = 100,
    ) -> None:
        """
        Record the cost of a rollback monitoring AI call.

        Args:
            deployment_id: The deployment being monitored.
            action: Type of rollback action ("monitor_check", "rollback_decision").
            tokens_input: Input tokens consumed.
            tokens_output: Output tokens consumed.
            latency_ms: Request latency.
        """
        if self.cost_tracker is None:
            return

        routing = self.get_routing_decision()
        model = routing.model_id if routing else "claude-sonnet-4-5-20250929"

        await self.cost_tracker.record_api_call(
            optimization_id=deployment_id,
            model=model,
            task_type=f"rollback_{action}",
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            cache_hit=False,
        )
