"""
Model version lock for coordinating online learning and distillation.

Prevents conflicts when online learning tries to apply buffered updates to
a model version that has been superseded by distillation.

Problem solved:
  - Online learner buffers updates for model V5
  - Distillation starts, creates model V6
  - Online learner tries to apply buffered updates to V5 (stale!)

Solution:
  - Lock during distillation
  - Buffer online updates while locked
  - Replay buffered updates on the new model after distillation completes

Usage:
    lock = ModelVersionLock()
    await lock.initialize(model_version=5)

    # Online learning (non-blocking)
    async with lock.online_learning_context() as ctx:
        if ctx.can_update:
            ctx.apply_update(gradient_update)
        else:
            ctx.buffer_update(gradient_update)

    # Distillation (exclusive)
    async with lock.distillation_context() as ctx:
        new_model = await train_distilled_model()
        ctx.set_new_version(6, new_model)
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BufferedUpdate:
    """A buffered model update that could not be applied immediately."""

    update_id: str
    timestamp: datetime
    update_data: dict[str, Any]
    source: str  # "online_learning", "feedback", etc.
    model_version_target: int  # Which model version this was intended for


@dataclass
class LockStats:
    """Statistics for the model version lock."""

    total_online_updates: int = 0
    total_buffered_updates: int = 0
    total_replayed_updates: int = 0
    total_dropped_updates: int = 0
    total_distillations: int = 0
    distillation_time_seconds: float = 0.0
    current_buffer_size: int = 0
    last_distillation_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_online_updates": self.total_online_updates,
            "total_buffered_updates": self.total_buffered_updates,
            "total_replayed_updates": self.total_replayed_updates,
            "total_dropped_updates": self.total_dropped_updates,
            "total_distillations": self.total_distillations,
            "distillation_time_seconds": round(self.distillation_time_seconds, 2),
            "current_buffer_size": self.current_buffer_size,
            "last_distillation_at": (
                self.last_distillation_at.isoformat() if self.last_distillation_at else None
            ),
        }


class OnlineLearningContext:
    """Context manager result for online learning operations."""

    def __init__(
        self,
        can_update: bool,
        model_version: int,
        lock: "ModelVersionLock",
    ) -> None:
        self.can_update = can_update
        self.model_version = model_version
        self._lock = lock
        self._updates_applied: list[str] = []

    def buffer_update(self, update: BufferedUpdate) -> None:
        """
        Buffer an update for later replay.

        Called when distillation is in progress and direct updates
        cannot be applied.
        """
        self._lock._buffer_update(update)

    def record_applied(self, update_id: str) -> None:
        """Record that an update was applied successfully."""
        self._updates_applied.append(update_id)
        self._lock._stats.total_online_updates += 1


class DistillationContext:
    """Context manager result for distillation operations."""

    def __init__(self, lock: "ModelVersionLock") -> None:
        self._lock = lock
        self._new_version: int | None = None

    def set_new_version(self, version: int) -> None:
        """
        Set the new model version after distillation.

        Args:
            version: The new model version number.
        """
        self._new_version = version

    @property
    def buffered_updates(self) -> list[BufferedUpdate]:
        """
        Get buffered updates to replay on the new model.

        Returns:
            List of BufferedUpdate that were buffered during distillation.
        """
        return list(self._lock._update_buffer)


class ModelVersionLock:
    """
    Coordinate online learning and distillation to prevent conflicts.

    The lock has two modes:
    1. Normal mode: Online learning can freely apply updates
    2. Distillation mode: Online learning buffers updates

    State transitions:
    - NORMAL -> DISTILLING: When distillation_context() is entered
    - DISTILLING -> NORMAL: When distillation_context() exits
    - DISTILLING -> REPLAYING: Briefly, when replaying buffered updates

    Buffer management:
    - Max buffer size configurable (default 1000 updates)
    - Oldest updates dropped when buffer is full
    - Updates older than max_buffer_age_seconds are dropped on replay
    """

    def __init__(
        self,
        max_buffer_size: int = 1000,
        max_buffer_age_seconds: float = 3600.0,
        task_router: Any = None,
        cost_tracker: Any = None,
    ) -> None:
        """
        Initialize the model version lock.

        Args:
            max_buffer_size: Maximum number of updates to buffer.
            max_buffer_age_seconds: Maximum age of buffered updates to
                replay (older ones are dropped).
            task_router: TaskComplexityRouter for hybrid Opus/Sonnet routing.
            cost_tracker: CostTracker for tracking AI call costs.
        """
        self.max_buffer_size = max_buffer_size
        self.max_buffer_age_seconds = max_buffer_age_seconds
        self.task_router = task_router
        self.cost_tracker = cost_tracker

        self._model_version: int = 0
        self._is_distilling: bool = False
        self._distillation_lock = asyncio.Lock()
        self._online_semaphore = asyncio.Semaphore(10)  # Allow up to 10 concurrent online learners
        self._update_buffer: deque[BufferedUpdate] = deque(maxlen=max_buffer_size)
        self._stats = LockStats()
        self._initialized = False

    async def initialize(self, model_version: int = 0) -> None:
        """
        Initialize the lock with the current model version.

        Args:
            model_version: The current model version number.
        """
        self._model_version = model_version
        self._initialized = True
        logger.info(
            "ModelVersionLock initialized for model version %d",
            model_version,
        )

    @property
    def model_version(self) -> int:
        """Current model version."""
        return self._model_version

    @property
    def is_distilling(self) -> bool:
        """Whether distillation is currently in progress."""
        return self._is_distilling

    @property
    def buffer_size(self) -> int:
        """Current number of buffered updates."""
        return len(self._update_buffer)

    def get_stats(self) -> LockStats:
        """Get a copy of lock statistics."""
        stats = LockStats(
            total_online_updates=self._stats.total_online_updates,
            total_buffered_updates=self._stats.total_buffered_updates,
            total_replayed_updates=self._stats.total_replayed_updates,
            total_dropped_updates=self._stats.total_dropped_updates,
            total_distillations=self._stats.total_distillations,
            distillation_time_seconds=self._stats.distillation_time_seconds,
            current_buffer_size=len(self._update_buffer),
            last_distillation_at=self._stats.last_distillation_at,
        )
        return stats

    def online_learning_context(self) -> "_OnlineLearningContextManager":
        """
        Get a context manager for online learning operations.

        Usage:
            async with lock.online_learning_context() as ctx:
                if ctx.can_update:
                    # Apply update directly to model
                    ctx.record_applied(update_id)
                else:
                    # Distillation in progress, buffer the update
                    ctx.buffer_update(buffered_update)
        """
        return _OnlineLearningContextManager(self)

    def distillation_context(self) -> "_DistillationContextManager":
        """
        Get a context manager for distillation operations (exclusive).

        Usage:
            async with lock.distillation_context() as ctx:
                new_model = await train_distilled_model()
                ctx.set_new_version(6)
                # On exit, buffered updates are available for replay
        """
        return _DistillationContextManager(self)

    def _buffer_update(self, update: BufferedUpdate) -> None:
        """Buffer an update (called from OnlineLearningContext)."""
        if len(self._update_buffer) >= self.max_buffer_size:
            dropped = self._update_buffer.popleft()
            self._stats.total_dropped_updates += 1
            logger.debug(
                "Dropped oldest buffered update %s (buffer full)",
                dropped.update_id,
            )

        self._update_buffer.append(update)
        self._stats.total_buffered_updates += 1
        self._stats.current_buffer_size = len(self._update_buffer)

    async def replay_buffered_updates(
        self,
    ) -> list[BufferedUpdate]:
        """
        Get buffered updates that are still valid for replay.

        Filters out updates that are too old or targeted at incompatible
        model versions.

        Returns:
            List of BufferedUpdate that should be replayed.
        """
        now = datetime.now()
        valid_updates: list[BufferedUpdate] = []
        dropped = 0

        while self._update_buffer:
            update = self._update_buffer.popleft()
            age = (now - update.timestamp).total_seconds()

            if age > self.max_buffer_age_seconds:
                dropped += 1
                continue

            valid_updates.append(update)

        self._stats.total_dropped_updates += dropped
        self._stats.total_replayed_updates += len(valid_updates)
        self._stats.current_buffer_size = 0

        if dropped > 0:
            logger.info("Dropped %d stale buffered updates during replay", dropped)

        logger.info(
            "Replaying %d buffered updates on model v%d",
            len(valid_updates),
            self._model_version,
        )

        return valid_updates

    async def clear_buffer(self) -> int:
        """
        Clear all buffered updates.

        Returns:
            Number of updates cleared.
        """
        count = len(self._update_buffer)
        self._update_buffer.clear()
        self._stats.current_buffer_size = 0
        self._stats.total_dropped_updates += count
        return count

    def get_routing_decision(
        self,
        is_distillation: bool = False,
    ) -> Any | None:
        """
        Get a model routing decision for version lock coordination.

        Distillation decisions use Opus (complex), online learning
        uses Sonnet (routine).

        Args:
            is_distillation: Whether this is for a distillation operation.

        Returns:
            ModelRoutingDecision if task_router is available, else None.
        """
        if self.task_router is None:
            return None

        if is_distillation:
            decision = self.task_router.route_task(
                task_type="strategy_analysis",
                num_parameters=0,
                has_constraints=False,
                requires_explanation=False,
                is_high_stakes=True,
            )
        else:
            decision = self.task_router.route_task(
                task_type="quick_summary",
                num_parameters=0,
                has_constraints=False,
                requires_explanation=False,
                is_high_stakes=False,
            )

        return decision

    async def record_lock_cost(
        self,
        optimization_id: str,
        action: str = "version_check",
        tokens_input: int = 100,
        tokens_output: int = 200,
        latency_ms: int = 50,
    ) -> None:
        """
        Record the cost of a model version lock operation.

        Args:
            optimization_id: The optimization/distillation run ID.
            action: Type of lock action.
            tokens_input: Input tokens consumed.
            tokens_output: Output tokens consumed.
            latency_ms: Request latency.
        """
        if self.cost_tracker is None:
            return

        is_distill = action == "distillation"
        routing = self.get_routing_decision(is_distillation=is_distill)
        model = routing.model_id if routing else "claude-sonnet-4-5-20250929"

        await self.cost_tracker.record_api_call(
            optimization_id=optimization_id,
            model=model,
            task_type=f"model_lock_{action}",
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            cache_hit=False,
        )


class _OnlineLearningContextManager:
    """Async context manager for online learning."""

    def __init__(self, lock: ModelVersionLock) -> None:
        self._lock = lock

    async def __aenter__(self) -> OnlineLearningContext:
        await self._lock._online_semaphore.acquire()
        ctx = OnlineLearningContext(
            can_update=not self._lock._is_distilling,
            model_version=self._lock._model_version,
            lock=self._lock,
        )
        return ctx

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self._lock._online_semaphore.release()


class _DistillationContextManager:
    """Async context manager for distillation (exclusive)."""

    def __init__(self, lock: ModelVersionLock) -> None:
        self._lock = lock
        self._ctx: DistillationContext | None = None
        self._start_time: float = 0.0

    async def __aenter__(self) -> DistillationContext:
        # Acquire the distillation lock (only one at a time)
        await self._lock._distillation_lock.acquire()
        self._lock._is_distilling = True
        self._start_time = time.monotonic()

        self._ctx = DistillationContext(self._lock)
        logger.info(
            "Distillation started. Model version: %d. Online learning updates will be buffered.",
            self._lock._model_version,
        )
        return self._ctx

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        elapsed = time.monotonic() - self._start_time

        # Update model version if distillation set one
        if self._ctx and self._ctx._new_version is not None:
            old_version = self._lock._model_version
            self._lock._model_version = self._ctx._new_version
            logger.info(
                "Model version updated: %d -> %d",
                old_version,
                self._ctx._new_version,
            )

        # Update stats
        self._lock._stats.total_distillations += 1
        self._lock._stats.distillation_time_seconds += elapsed
        self._lock._stats.last_distillation_at = datetime.now()

        # Release distillation lock
        self._lock._is_distilling = False
        self._lock._distillation_lock.release()

        if exc_type:
            logger.error(
                "Distillation failed after %.1fs: %s",
                elapsed,
                exc_val,
            )
        else:
            logger.info(
                "Distillation completed in %.1fs. Buffer has %d updates.",
                elapsed,
                len(self._lock._update_buffer),
            )
