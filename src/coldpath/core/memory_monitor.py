"""
System memory monitoring with pressure-based load shedding.

Monitors system memory usage and triggers automatic cache eviction
and load shedding when memory pressure is detected.
"""

import asyncio
import gc
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Any, Callable

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)


class MemoryPressureLevel(IntEnum):
    """Memory pressure levels based on system memory usage."""

    NORMAL = 0  # 0-70% - Normal operation
    ELEVATED = 1  # 70-85% - Start gentle cache eviction
    HIGH = 2  # 85-95% - Aggressive cache eviction
    CRITICAL = 3  # 95%+ - Emergency load shedding


@dataclass
class MemoryStats:
    """Current memory statistics."""

    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    process_mb: float
    process_percent: float
    pressure_level: MemoryPressureLevel
    timestamp: datetime

    @property
    def available_percent(self) -> float:
        return 100.0 - self.percent_used


@dataclass
class MemoryThresholds:
    """Configurable thresholds for memory pressure levels."""

    elevated_percent: float = 70.0
    high_percent: float = 85.0
    critical_percent: float = 95.0


class MemoryPressureCallback:
    """Callback registration for memory pressure events."""

    def __init__(
        self,
        callback: Callable[[MemoryPressureLevel, MemoryStats], None],
        trigger_levels: list[MemoryPressureLevel],
    ):
        self.callback = callback
        self.trigger_levels = trigger_levels
        self.last_triggered_level: MemoryPressureLevel | None = None


class MemoryMonitor:
    """
    System memory monitor with pressure-based callbacks.

    Features:
    - Periodic memory usage monitoring
    - Pressure level detection with hysteresis
    - Callback system for pressure events
    - Automatic garbage collection at high pressure
    - Integration with cache managers for eviction
    """

    def __init__(
        self,
        check_interval_seconds: float = 5.0,
        thresholds: MemoryThresholds | None = None,
    ):
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, memory monitoring will be limited")

        self._check_interval = check_interval_seconds
        self._thresholds = thresholds or MemoryThresholds()
        self._callbacks: list[MemoryPressureCallback] = []
        self._current_level = MemoryPressureLevel.NORMAL
        self._monitor_task: asyncio.Task[None] | None = None
        self._running = False
        self._history: list[MemoryStats] = []
        self._max_history = 120  # ~10 minutes at 5s intervals
        self._cache_managers: list[Any] = []  # Objects with evict() method

    @property
    def current_level(self) -> MemoryPressureLevel:
        """Get current memory pressure level."""
        return self._current_level

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if not PSUTIL_AVAILABLE:
            return MemoryStats(
                total_mb=0,
                available_mb=0,
                used_mb=0,
                percent_used=0,
                process_mb=0,
                process_percent=0,
                pressure_level=MemoryPressureLevel.NORMAL,
                timestamp=datetime.now(),
            )

        mem = psutil.virtual_memory()
        process = psutil.Process()
        process_mem = process.memory_info()

        total_mb = mem.total / (1024 * 1024)
        available_mb = mem.available / (1024 * 1024)
        used_mb = mem.used / (1024 * 1024)
        percent_used = mem.percent
        process_mb = process_mem.rss / (1024 * 1024)
        process_percent = (process_mem.rss / mem.total) * 100

        pressure_level = self._calculate_pressure_level(percent_used)

        return MemoryStats(
            total_mb=total_mb,
            available_mb=available_mb,
            used_mb=used_mb,
            percent_used=percent_used,
            process_mb=process_mb,
            process_percent=process_percent,
            pressure_level=pressure_level,
            timestamp=datetime.now(),
        )

    def _calculate_pressure_level(self, percent_used: float) -> MemoryPressureLevel:
        """Calculate pressure level from memory percentage."""
        if percent_used >= self._thresholds.critical_percent:
            return MemoryPressureLevel.CRITICAL
        elif percent_used >= self._thresholds.high_percent:
            return MemoryPressureLevel.HIGH
        elif percent_used >= self._thresholds.elevated_percent:
            return MemoryPressureLevel.ELEVATED
        return MemoryPressureLevel.NORMAL

    def register_callback(
        self,
        callback: Callable[[MemoryPressureLevel, MemoryStats], None],
        trigger_levels: list[MemoryPressureLevel] | None = None,
    ) -> None:
        """
        Register a callback for memory pressure events.

        Args:
            callback: Function to call when pressure changes
            trigger_levels: List of levels that trigger the callback
                          (None = all levels)
        """
        if trigger_levels is None:
            trigger_levels = list(MemoryPressureLevel)

        self._callbacks.append(MemoryPressureCallback(callback, trigger_levels))
        logger.debug(f"Registered memory pressure callback for levels: {trigger_levels}")

    def register_cache_manager(self, cache_manager: Any) -> None:
        """
        Register a cache manager for automatic eviction.

        The cache manager should have an evict() or cleanup() method.
        """
        self._cache_managers.append(cache_manager)
        logger.debug(f"Registered cache manager: {type(cache_manager).__name__}")

    async def start(self) -> None:
        """Start the memory monitor."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(),
            name="memory_monitor",
        )
        logger.info(
            f"Memory monitor started (interval: {self._check_interval}s, "
            f"thresholds: {self._thresholds.elevated_percent}%/"
            f"{self._thresholds.high_percent}%/{self._thresholds.critical_percent}%)"
        )

    async def stop(self) -> None:
        """Stop the memory monitor."""
        if not self._running:
            return

        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Memory monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                stats = self.get_memory_stats()
                self._history.append(stats)

                # Trim history
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history :]

                # Check for pressure level change
                new_level = stats.pressure_level
                if new_level != self._current_level:
                    old_level = self._current_level
                    self._current_level = new_level

                    logger.info(
                        f"Memory pressure changed: {old_level.name} -> {new_level.name} "
                        f"({stats.percent_used:.1f}% used)"
                    )

                    # Handle pressure increase
                    if new_level > old_level:
                        await self._handle_pressure_increase(new_level, stats)

                    # Trigger callbacks
                    await self._trigger_callbacks(new_level, stats)

                # Periodic logging
                if len(self._history) % 12 == 0:  # Every minute at 5s interval
                    logger.debug(
                        f"Memory: {stats.percent_used:.1f}% system, "
                        f"{stats.process_mb:.0f}MB process, "
                        f"pressure: {stats.pressure_level.name}"
                    )

            except Exception as e:
                logger.error(f"Memory monitor error: {e}", exc_info=True)

            await asyncio.sleep(self._check_interval)

    async def _handle_pressure_increase(
        self,
        level: MemoryPressureLevel,
        stats: MemoryStats,
    ) -> None:
        """Handle memory pressure increase."""
        if level == MemoryPressureLevel.ELEVATED:
            # Gentle cache eviction
            await self._evict_caches(percentage=0.25)
            gc.collect(1)  # Generation 1 collection

        elif level == MemoryPressureLevel.HIGH:
            # Aggressive cache eviction
            await self._evict_caches(percentage=0.50)
            gc.collect(2)  # Full collection

        elif level == MemoryPressureLevel.CRITICAL:
            # Emergency load shedding
            logger.warning("CRITICAL memory pressure - emergency load shedding")
            await self._evict_caches(percentage=0.75)
            gc.collect()  # Full collection

    async def _evict_caches(self, percentage: float) -> None:
        """Evict entries from registered cache managers."""
        for cache_manager in self._cache_managers:
            try:
                if hasattr(cache_manager, "evict"):
                    await cache_manager.evict(percentage)
                elif hasattr(cache_manager, "cleanup"):
                    await cache_manager.cleanup()
                elif hasattr(cache_manager, "cleanup_expired"):
                    await cache_manager.cleanup_expired()
            except Exception as e:
                logger.error(f"Cache eviction error: {e}")

    async def _trigger_callbacks(
        self,
        level: MemoryPressureLevel,
        stats: MemoryStats,
    ) -> None:
        """Trigger registered callbacks for the pressure level."""
        for cb in self._callbacks:
            if level in cb.trigger_levels and level != cb.last_triggered_level:
                try:
                    if asyncio.iscoroutinefunction(cb.callback):
                        await cb.callback(level, stats)
                    else:
                        cb.callback(level, stats)
                    cb.last_triggered_level = level
                except Exception as e:
                    logger.error(f"Memory callback error: {e}")

    def get_memory_trend(self, window_seconds: int = 300) -> float | None:
        """
        Calculate memory usage trend over the specified window.

        Returns MB per second growth rate (positive = increasing).
        """
        if len(self._history) < 2:
            return None

        # Find entries within window
        cutoff = datetime.now()
        window_entries = [
            s for s in self._history if (cutoff - s.timestamp).total_seconds() <= window_seconds
        ]

        if len(window_entries) < 2:
            return None

        first = window_entries[0]
        last = window_entries[-1]

        time_delta = (last.timestamp - first.timestamp).total_seconds()
        if time_delta == 0:
            return None

        mem_delta = last.process_mb - first.process_mb
        return mem_delta / time_delta

    def get_diagnostics(self) -> dict[str, Any]:
        """Get comprehensive memory diagnostics."""
        stats = self.get_memory_stats()
        trend = self.get_memory_trend()

        return {
            "current_stats": {
                "total_mb": stats.total_mb,
                "available_mb": stats.available_mb,
                "used_mb": stats.used_mb,
                "percent_used": stats.percent_used,
                "process_mb": stats.process_mb,
                "process_percent": stats.process_percent,
            },
            "pressure_level": stats.pressure_level.name,
            "memory_trend_mb_per_sec": trend,
            "is_running": self._running,
            "thresholds": {
                "elevated": self._thresholds.elevated_percent,
                "high": self._thresholds.high_percent,
                "critical": self._thresholds.critical_percent,
            },
            "registered_callbacks": len(self._callbacks),
            "registered_cache_managers": len(self._cache_managers),
            "history_entries": len(self._history),
        }

    async def force_gc(self) -> dict[str, int]:
        """Force garbage collection and return stats."""
        before = self.get_memory_stats()

        # Run full garbage collection
        collected = {
            "gen0": gc.collect(0),
            "gen1": gc.collect(1),
            "gen2": gc.collect(2),
        }

        after = self.get_memory_stats()
        collected["freed_mb"] = int(before.process_mb - after.process_mb)

        logger.info(
            f"Forced GC: collected {sum(collected.values()) - collected['freed_mb']} objects, "
            f"freed {collected['freed_mb']}MB"
        )

        return collected

    async def __aenter__(self) -> "MemoryMonitor":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()
