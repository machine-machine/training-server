"""
Periodic health monitoring with automatic recovery.

Provides continuous health checking of all registered resources
with automatic recovery attempts and alerting callbacks.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

from .resource_registry import ResourceHealth, ResourceRegistry

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Overall health status of the system."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthReport:
    """Comprehensive health report for all resources."""

    timestamp: datetime
    overall_status: HealthStatus
    resource_count: int
    healthy_count: int
    degraded_count: int
    unhealthy_count: int
    resources: dict[str, ResourceHealth]
    recovery_attempts: int = 0
    last_check_duration_ms: float = 0

    @property
    def health_percentage(self) -> float:
        if self.resource_count == 0:
            return 100.0
        return (self.healthy_count / self.resource_count) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "resource_count": self.resource_count,
            "healthy_count": self.healthy_count,
            "degraded_count": self.degraded_count,
            "unhealthy_count": self.unhealthy_count,
            "health_percentage": self.health_percentage,
            "recovery_attempts": self.recovery_attempts,
            "last_check_duration_ms": self.last_check_duration_ms,
            "resources": {
                rid: {
                    "healthy": health.healthy,
                    "message": health.message,
                    "metrics": health.metrics,
                }
                for rid, health in self.resources.items()
            },
        }


@dataclass
class RecoveryConfig:
    """Configuration for automatic recovery."""

    enabled: bool = True
    max_attempts: int = 3
    initial_backoff_seconds: float = 5.0
    max_backoff_seconds: float = 300.0
    backoff_multiplier: float = 2.0


@dataclass
class ResourceRecoveryState:
    """Recovery state for a single resource."""

    resource_id: str
    attempt_count: int = 0
    last_attempt: datetime | None = None
    next_backoff_seconds: float = 5.0
    is_recovering: bool = False
    last_error: str | None = None


class HealthMonitor:
    """
    Continuous health monitoring with automatic recovery.

    Features:
    - Periodic health checks for all resources
    - Automatic recovery attempts with exponential backoff
    - Callback system for alerts and notifications
    - Comprehensive health reporting
    """

    def __init__(
        self,
        registry: ResourceRegistry,
        check_interval_seconds: float = 30.0,
        recovery_config: RecoveryConfig | None = None,
    ):
        self._registry = registry
        self._check_interval = check_interval_seconds
        self._recovery_config = recovery_config or RecoveryConfig()

        self._monitor_task: asyncio.Task[None] | None = None
        self._running = False
        self._recovery_states: dict[str, ResourceRecoveryState] = {}
        self._reports: list[HealthReport] = []
        self._max_reports = 100

        # Callbacks
        self._on_degraded: list[Callable[[str, ResourceHealth], None]] = []
        self._on_recovered: list[Callable[[str], None]] = []
        self._on_failure: list[Callable[[str, str], None]] = []

        self._total_checks = 0
        self._total_recovery_attempts = 0

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    @property
    def last_report(self) -> HealthReport | None:
        """Get the most recent health report."""
        return self._reports[-1] if self._reports else None

    def on_degraded(self, callback: Callable[[str, ResourceHealth], None]) -> None:
        """Register callback for when a resource becomes degraded."""
        self._on_degraded.append(callback)

    def on_recovered(self, callback: Callable[[str], None]) -> None:
        """Register callback for when a resource recovers."""
        self._on_recovered.append(callback)

    def on_failure(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for when recovery fails."""
        self._on_failure.append(callback)

    async def start(self) -> None:
        """Start the health monitor."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(),
            name="health_monitor",
        )
        logger.info(f"Health monitor started (interval: {self._check_interval}s)")

    async def stop(self) -> None:
        """Stop the health monitor."""
        if not self._running:
            return

        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Health monitor stopped")

    async def check_health(self) -> HealthReport:
        """
        Perform a health check on all resources.

        Returns:
            Comprehensive health report
        """
        start_time = datetime.now()
        self._total_checks += 1

        # Get health from all resources
        health_results = await self._registry.check_health()

        # Calculate counts
        healthy_count = sum(1 for h in health_results.values() if h.healthy)
        degraded_count = len(health_results) - healthy_count

        # Determine overall status
        if degraded_count == 0:
            overall_status = HealthStatus.HEALTHY
        elif healthy_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNHEALTHY

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        report = HealthReport(
            timestamp=start_time,
            overall_status=overall_status,
            resource_count=len(health_results),
            healthy_count=healthy_count,
            degraded_count=degraded_count,
            unhealthy_count=degraded_count,  # In this model, degraded = unhealthy
            resources=health_results,
            recovery_attempts=self._total_recovery_attempts,
            last_check_duration_ms=duration_ms,
        )

        # Store report
        self._reports.append(report)
        if len(self._reports) > self._max_reports:
            self._reports = self._reports[-self._max_reports :]

        return report

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                report = await self.check_health()

                # Log periodic status
                if self._total_checks % 2 == 0:  # Every other check
                    logger.debug(
                        f"Health check: {report.overall_status.value} "
                        f"({report.healthy_count}/{report.resource_count} healthy)"
                    )

                # Handle degraded resources
                for resource_id, health in report.resources.items():
                    if not health.healthy:
                        await self._handle_degraded_resource(resource_id, health)
                    else:
                        await self._handle_healthy_resource(resource_id)

            except Exception as e:
                logger.error(f"Health monitor error: {e}", exc_info=True)

            await asyncio.sleep(self._check_interval)

    async def _handle_degraded_resource(
        self,
        resource_id: str,
        health: ResourceHealth,
    ) -> None:
        """Handle a degraded resource."""
        # Get or create recovery state
        if resource_id not in self._recovery_states:
            self._recovery_states[resource_id] = ResourceRecoveryState(
                resource_id=resource_id,
                next_backoff_seconds=self._recovery_config.initial_backoff_seconds,
            )

            # Notify callbacks
            for callback in self._on_degraded:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(resource_id, health)
                    else:
                        callback(resource_id, health)
                except Exception as e:
                    logger.error(f"Error in degraded callback: {e}")

        state = self._recovery_states[resource_id]

        # Check if we should attempt recovery
        if not self._recovery_config.enabled:
            return

        if state.is_recovering:
            return

        if state.attempt_count >= self._recovery_config.max_attempts:
            # Max attempts exceeded
            if state.attempt_count == self._recovery_config.max_attempts:
                state.attempt_count += 1  # Prevent repeated notifications
                logger.error(f"Max recovery attempts exceeded for: {resource_id}")
                for callback in self._on_failure:
                    try:
                        callback(resource_id, "Max recovery attempts exceeded")
                    except Exception as e:
                        logger.error(f"Error in failure callback: {e}")
            return

        # Check backoff
        if state.last_attempt:
            elapsed = (datetime.now() - state.last_attempt).total_seconds()
            if elapsed < state.next_backoff_seconds:
                return

        # Attempt recovery
        await self._attempt_recovery(state)

    async def _handle_healthy_resource(self, resource_id: str) -> None:
        """Handle a resource that is now healthy."""
        if resource_id in self._recovery_states:
            state = self._recovery_states[resource_id]

            if state.attempt_count > 0:
                logger.info(f"Resource recovered: {resource_id}")
                for callback in self._on_recovered:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(resource_id)
                        else:
                            callback(resource_id)
                    except Exception as e:
                        logger.error(f"Error in recovered callback: {e}")

            del self._recovery_states[resource_id]

    async def _attempt_recovery(self, state: ResourceRecoveryState) -> None:
        """Attempt to recover a resource."""
        resource = self._registry.get(state.resource_id)
        if not resource:
            return

        state.is_recovering = True
        state.attempt_count += 1
        state.last_attempt = datetime.now()
        self._total_recovery_attempts += 1

        logger.info(
            f"Attempting recovery for {state.resource_id} "
            f"(attempt {state.attempt_count}/{self._recovery_config.max_attempts})"
        )

        try:
            # Attempt recovery by stopping and restarting
            await resource.stop()
            await resource.initialize()
            await resource.start()

            # Verify health
            health = await resource.health_check()

            if health.healthy:
                logger.info(f"Recovery successful: {state.resource_id}")
                state.is_recovering = False
                # State will be cleared on next healthy check
            else:
                raise Exception(f"Health check failed after recovery: {health.message}")

        except Exception as e:
            state.last_error = str(e)
            state.is_recovering = False

            # Increase backoff
            state.next_backoff_seconds = min(
                state.next_backoff_seconds * self._recovery_config.backoff_multiplier,
                self._recovery_config.max_backoff_seconds,
            )

            logger.error(
                f"Recovery failed for {state.resource_id}: {e}. "
                f"Next attempt in {state.next_backoff_seconds:.0f}s"
            )

    async def force_recovery(self, resource_id: str) -> bool:
        """
        Force a recovery attempt for a specific resource.

        Returns True if recovery was successful.
        """
        resource = self._registry.get(resource_id)
        if not resource:
            logger.warning(f"Resource not found: {resource_id}")
            return False

        # Reset recovery state
        self._recovery_states[resource_id] = ResourceRecoveryState(
            resource_id=resource_id,
            next_backoff_seconds=self._recovery_config.initial_backoff_seconds,
        )

        state = self._recovery_states[resource_id]
        await self._attempt_recovery(state)

        # Check if healthy
        health = await resource.health_check()
        return health.healthy

    def get_recovery_status(self) -> dict[str, Any]:
        """Get status of all recovery operations."""
        return {
            "total_recovery_attempts": self._total_recovery_attempts,
            "active_recoveries": sum(1 for s in self._recovery_states.values() if s.is_recovering),
            "resources_in_recovery": {
                rid: {
                    "attempt_count": state.attempt_count,
                    "last_attempt": (
                        state.last_attempt.isoformat() if state.last_attempt else None
                    ),
                    "next_backoff_seconds": state.next_backoff_seconds,
                    "is_recovering": state.is_recovering,
                    "last_error": state.last_error,
                }
                for rid, state in self._recovery_states.items()
            },
        }

    def get_history(self, limit: int = 10) -> list[HealthReport]:
        """Get recent health reports."""
        return self._reports[-limit:]

    def get_uptime_percentage(self, window_seconds: int = 3600) -> float:
        """
        Calculate uptime percentage over the specified window.

        Returns percentage of time the system was healthy.
        """
        if not self._reports:
            return 100.0

        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        recent_reports = [r for r in self._reports if r.timestamp >= cutoff]

        if not recent_reports:
            return 100.0

        healthy_count = sum(1 for r in recent_reports if r.overall_status == HealthStatus.HEALTHY)
        return (healthy_count / len(recent_reports)) * 100

    def get_diagnostics(self) -> dict[str, Any]:
        """Get comprehensive diagnostics."""
        return {
            "is_running": self._running,
            "check_interval_seconds": self._check_interval,
            "total_checks": self._total_checks,
            "total_recovery_attempts": self._total_recovery_attempts,
            "report_count": len(self._reports),
            "last_report": self.last_report.to_dict() if self.last_report else None,
            "recovery_status": self.get_recovery_status(),
            "uptime_1h": self.get_uptime_percentage(3600),
            "uptime_24h": self.get_uptime_percentage(86400),
        }

    async def __aenter__(self) -> "HealthMonitor":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()
