"""
Resource Guard

Prevents optimization from starving the trading pipeline.
Monitors memory, CPU, and concurrent optimization limits.

Problem solved:
    - Optimization runs Monte Carlo with 1000 simulations
    - Memory spikes to 2GB
    - Trading pipeline starved, trades fail

Solution:
    - Check resources before starting optimization
    - Block if memory < 500MB free
    - Block if CPU > 70%
    - Block if concurrent optimizations at limit

Usage:
    guard = ResourceGuard()

    # Check before starting optimization
    status = await guard.can_start_optimization()

    if status.can_proceed:
        await guard.reserve_resources("optimization_123")
        try:
            await run_optimization()
        finally:
            await guard.release_resources("optimization_123")
    else:
        logger.warning(f"Cannot optimize: {status.reason}")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# psutil is optional - graceful degradation if not available
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - resource monitoring will use mock values")


@dataclass
class ResourceStatus:
    """Current resource availability status."""

    can_proceed: bool
    """Whether optimization can proceed."""

    reason: str
    """Reason for go/no-go decision."""

    free_memory_mb: float
    """Available memory in MB."""

    cpu_percent: float
    """Current CPU utilization percentage."""

    active_optimizations: int
    """Number of currently active optimizations."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When this status was captured."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "can_proceed": self.can_proceed,
            "reason": self.reason,
            "free_memory_mb": round(self.free_memory_mb, 1),
            "cpu_percent": round(self.cpu_percent, 1),
            "active_optimizations": self.active_optimizations,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ResourceGuardStats:
    """Statistics for the resource guard."""

    total_checks: int = 0
    approved_checks: int = 0
    rejected_checks: int = 0
    memory_rejections: int = 0
    cpu_rejections: int = 0
    concurrency_rejections: int = 0
    current_active: int = 0
    peak_active: int = 0


class ResourceGuard:
    """
    Prevent optimization from impacting trading performance.

    Guards:
    - Memory: Must have MIN_FREE_MEMORY_MB free
    - CPU: Must be below MAX_CPU_PERCENT
    - Concurrency: Max MAX_CONCURRENT_OPTIMIZATIONS at a time

    The guard provides:
    1. Pre-flight checks before optimization starts
    2. Resource reservation (soft limit tracking)
    3. Automatic cleanup on release
    4. Statistics for monitoring

    Example:
        guard = ResourceGuard()

        # Pre-flight check
        status = await guard.can_start_optimization()
        print(f"Memory free: {status.free_memory_mb}MB")
        print(f"CPU: {status.cpu_percent}%")

        if status.can_proceed:
            # Reserve and run
            await guard.reserve_resources("opt_123")
            try:
                await run_optimization()
            finally:
                await guard.release_resources("opt_123")
    """

    # Default thresholds
    MIN_FREE_MEMORY_MB = 500
    MAX_CPU_PERCENT = 70
    MAX_CONCURRENT_OPTIMIZATIONS = 1

    def __init__(
        self,
        min_free_memory_mb: float = MIN_FREE_MEMORY_MB,
        max_cpu_percent: float = MAX_CPU_PERCENT,
        max_concurrent: int = MAX_CONCURRENT_OPTIMIZATIONS,
    ):
        """
        Initialize the resource guard.

        Args:
            min_free_memory_mb: Minimum free memory required (MB).
            max_cpu_percent: Maximum CPU percentage allowed.
            max_concurrent: Maximum concurrent optimizations.
        """
        self.min_free_memory_mb = min_free_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.max_concurrent = max_concurrent

        self._active_optimizations: set[str] = set()
        self._lock = asyncio.Lock()
        self._stats = ResourceGuardStats()

    def get_stats(self) -> ResourceGuardStats:
        """Get current statistics."""
        return ResourceGuardStats(
            total_checks=self._stats.total_checks,
            approved_checks=self._stats.approved_checks,
            rejected_checks=self._stats.rejected_checks,
            memory_rejections=self._stats.memory_rejections,
            cpu_rejections=self._stats.cpu_rejections,
            concurrency_rejections=self._stats.concurrency_rejections,
            current_active=len(self._active_optimizations),
            peak_active=self._stats.peak_active,
        )

    async def can_start_optimization(self) -> ResourceStatus:
        """
        Check if system has resources for optimization.

        This performs a non-blocking check of all guard conditions.
        Use reserve_resources() to actually reserve if check passes.

        Returns:
            ResourceStatus with go/no-go decision and reason.

        Example:
            status = await guard.can_start_optimization()
            if not status.can_proceed:
                logger.warning(f"Blocked: {status.reason}")
        """
        self._stats.total_checks += 1

        # Get current resource usage
        free_mb, cpu_percent = self._get_resource_usage()
        active_count = len(self._active_optimizations)

        # Check memory
        if free_mb < self.min_free_memory_mb:
            self._stats.rejected_checks += 1
            self._stats.memory_rejections += 1
            return ResourceStatus(
                can_proceed=False,
                reason=(
                    f"Insufficient memory: {free_mb:.0f}MB free (need {self.min_free_memory_mb}MB)"
                ),
                free_memory_mb=free_mb,
                cpu_percent=cpu_percent,
                active_optimizations=active_count,
            )

        # Check CPU
        if cpu_percent > self.max_cpu_percent:
            self._stats.rejected_checks += 1
            self._stats.cpu_rejections += 1
            return ResourceStatus(
                can_proceed=False,
                reason=f"High CPU usage: {cpu_percent:.0f}% (threshold {self.max_cpu_percent}%)",
                free_memory_mb=free_mb,
                cpu_percent=cpu_percent,
                active_optimizations=active_count,
            )

        # Check concurrency
        if active_count >= self.max_concurrent:
            self._stats.rejected_checks += 1
            self._stats.concurrency_rejections += 1
            return ResourceStatus(
                can_proceed=False,
                reason=f"Max concurrent optimizations: {active_count}/{self.max_concurrent}",
                free_memory_mb=free_mb,
                cpu_percent=cpu_percent,
                active_optimizations=active_count,
            )

        # All checks passed
        self._stats.approved_checks += 1
        return ResourceStatus(
            can_proceed=True,
            reason="Resources available",
            free_memory_mb=free_mb,
            cpu_percent=cpu_percent,
            active_optimizations=active_count,
        )

    async def reserve_resources(self, optimization_id: str) -> bool:
        """
        Reserve resources for an optimization.

        This is a soft reservation - it tracks the optimization as active
        but doesn't actually allocate memory. Use this to enforce concurrency
        limits.

        Args:
            optimization_id: Unique ID for this optimization run.

        Returns:
            True if reservation successful, False if blocked.

        Example:
            if await guard.reserve_resources("opt_123"):
                try:
                    await run_optimization()
                finally:
                    await guard.release_resources("opt_123")
        """
        status = await self.can_start_optimization()

        if not status.can_proceed:
            logger.warning(
                "Cannot reserve resources for %s: %s",
                optimization_id,
                status.reason,
            )
            return False

        async with self._lock:
            self._active_optimizations.add(optimization_id)
            current_active = len(self._active_optimizations)
            self._stats.peak_active = max(self._stats.peak_active, current_active)

        logger.info(
            "Reserved resources for optimization %s (%d active)",
            optimization_id,
            len(self._active_optimizations),
        )

        return True

    async def release_resources(self, optimization_id: str) -> None:
        """
        Release reserved resources.

        Always call this after optimization completes, even if it failed.

        Args:
            optimization_id: The ID that was used to reserve.
        """
        async with self._lock:
            self._active_optimizations.discard(optimization_id)

        logger.info(
            "Released resources for optimization %s (%d remaining)",
            optimization_id,
            len(self._active_optimizations),
        )

    async def get_status(self) -> ResourceStatus:
        """
        Get current resource status without reservation check.

        Use this for monitoring dashboards.

        Returns:
            ResourceStatus with current metrics.
        """
        free_mb, cpu_percent = self._get_resource_usage()
        active_count = len(self._active_optimizations)

        return ResourceStatus(
            can_proceed=active_count < self.max_concurrent,
            reason="Status check (no reservation)",
            free_memory_mb=free_mb,
            cpu_percent=cpu_percent,
            active_optimizations=active_count,
        )

    def _get_resource_usage(self) -> tuple:
        """
        Get current memory and CPU usage.

        Returns:
            Tuple of (free_memory_mb, cpu_percent).
        """
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                free_mb = mem.available / (1024 * 1024)
                cpu_percent = psutil.cpu_percent(interval=0.1)
                return (free_mb, cpu_percent)
            except Exception as e:
                logger.warning("Failed to get resource usage: %s", e)

        # Fallback: assume resources available
        return (1024.0, 30.0)

    @property
    def active_count(self) -> int:
        """Number of currently active optimizations."""
        return len(self._active_optimizations)

    @property
    def active_optimizations(self) -> set[str]:
        """Set of currently active optimization IDs."""
        return self._active_optimizations.copy()

    def to_dict(self) -> dict[str, Any]:
        """Serialize guard state to dictionary."""
        return {
            "thresholds": {
                "min_free_memory_mb": self.min_free_memory_mb,
                "max_cpu_percent": self.max_cpu_percent,
                "max_concurrent": self.max_concurrent,
            },
            "current": {
                "active_optimizations": list(self._active_optimizations),
                "active_count": len(self._active_optimizations),
            },
            "stats": {
                "total_checks": self._stats.total_checks,
                "approved_checks": self._stats.approved_checks,
                "rejected_checks": self._stats.rejected_checks,
                "memory_rejections": self._stats.memory_rejections,
                "cpu_rejections": self._stats.cpu_rejections,
                "concurrency_rejections": self._stats.concurrency_rejections,
                "peak_active": self._stats.peak_active,
            },
        }


# Singleton instance
_resource_guard: ResourceGuard | None = None


def get_resource_guard() -> ResourceGuard:
    """
    Get or create the global resource guard.

    Returns:
        The singleton ResourceGuard instance.
    """
    global _resource_guard
    if _resource_guard is None:
        _resource_guard = ResourceGuard()
    return _resource_guard


def reset_resource_guard() -> None:
    """Reset the global resource guard (for testing)."""
    global _resource_guard
    _resource_guard = None
