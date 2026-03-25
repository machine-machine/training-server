"""
Resource guard for optimization tasks.

Prevents optimization and AI workloads from exhausting system resources
by checking memory, CPU, and concurrent-task limits before starting.

Usage:
    guard = ResourceGuard()
    can_start, reason = await guard.can_start_optimization()
    if can_start:
        async with guard.reserve_resources("opt_123"):
            # Run optimization
            pass
"""

import asyncio
import logging
from dataclasses import dataclass

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limits for optimization tasks."""

    max_concurrent_optimizations: int = 1
    min_free_memory_mb: int = 500
    max_cpu_percent: float = 70.0
    max_optimization_memory_mb: int = 800


class ResourceGuard:
    """Prevent optimization from exhausting system resources.

    Usage::

        guard = ResourceGuard()
        can_start, reason = await guard.can_start_optimization()
        if can_start:
            async with guard.reserve_resources("opt_123"):
                # Run optimization
                pass
    """

    def __init__(self, limits: ResourceLimits | None = None) -> None:
        self.limits = limits or ResourceLimits()
        self.active_optimizations: set[str] = set()
        self._lock = asyncio.Lock()

    async def can_start_optimization(self) -> tuple[bool, str]:
        """Check if system has resources for a new optimization."""
        if psutil is None:
            logger.warning("psutil not installed; skipping resource checks")
            return True, "psutil unavailable, checks skipped"

        # Check concurrent limit
        if len(self.active_optimizations) >= self.limits.max_concurrent_optimizations:
            return False, (
                f"Max concurrent optimizations reached: "
                f"{len(self.active_optimizations)}/"
                f"{self.limits.max_concurrent_optimizations}"
            )

        # Check memory
        mem = psutil.virtual_memory()
        free_mb = mem.available / 1024 / 1024

        if free_mb < self.limits.min_free_memory_mb:
            return False, (
                f"Insufficient memory: {free_mb:.0f}MB free "
                f"(need {self.limits.min_free_memory_mb}MB)"
            )

        # Check CPU (1-second sample)
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.limits.max_cpu_percent:
            return False, (
                f"High CPU usage: {cpu_percent:.0f}% (threshold {self.limits.max_cpu_percent}%)"
            )

        return True, "Resources available"

    def reserve_resources(self, optimization_id: str) -> "_ResourceReservation":
        """Return an async context manager that reserves resources."""
        return _ResourceReservation(self, optimization_id)

    async def _add_optimization(self, optimization_id: str) -> None:
        """Mark optimization as active."""
        async with self._lock:
            self.active_optimizations.add(optimization_id)
            logger.info(
                "Reserved resources for optimization %s (%d active)",
                optimization_id,
                len(self.active_optimizations),
            )

    async def _remove_optimization(self, optimization_id: str) -> None:
        """Release resources for optimization."""
        async with self._lock:
            self.active_optimizations.discard(optimization_id)
            logger.info(
                "Released resources for optimization %s (%d active)",
                optimization_id,
                len(self.active_optimizations),
            )

    @property
    def active_count(self) -> int:
        """Number of currently active optimizations."""
        return len(self.active_optimizations)


class _ResourceReservation:
    """Async context manager for resource reservation."""

    def __init__(self, guard: ResourceGuard, optimization_id: str) -> None:
        self.guard = guard
        self.optimization_id = optimization_id

    async def __aenter__(self) -> "_ResourceReservation":
        await self.guard._add_optimization(self.optimization_id)
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.guard._remove_optimization(self.optimization_id)
