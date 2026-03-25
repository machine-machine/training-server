"""
Async task queue manager with priority support and backpressure handling.

Provides bounded task execution with configurable priorities,
timeouts, and automatic retry support.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TaskPriority(IntEnum):
    """Task priority levels (higher value = higher priority)."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskState(IntEnum):
    """Task execution states."""

    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4
    TIMEOUT = 5


@dataclass
class QueuedTask:
    """A task queued for execution."""

    id: str
    name: str
    priority: TaskPriority
    func: Callable[[], Awaitable[Any]]
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    state: TaskState = TaskState.PENDING
    result: Any = None
    error: Exception | None = None
    timeout_seconds: float | None = None
    max_retries: int = 0
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        """Get task execution duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None

    @property
    def wait_time_ms(self) -> float | None:
        """Get time spent waiting in queue in milliseconds."""
        if self.started_at:
            return (self.started_at - self.created_at).total_seconds() * 1000
        return None

    def __lt__(self, other: "QueuedTask") -> bool:
        """Compare tasks by priority (higher priority first) then by creation time."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


@dataclass
class QueueStats:
    """Statistics for the task queue."""

    total_submitted: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_cancelled: int = 0
    total_timeout: int = 0
    total_retried: int = 0
    total_rejected: int = 0
    current_queue_size: int = 0
    current_running: int = 0
    total_wait_time_ms: float = 0
    total_execution_time_ms: float = 0

    @property
    def average_wait_time_ms(self) -> float:
        completed = self.total_completed + self.total_failed
        if completed == 0:
            return 0
        return self.total_wait_time_ms / completed

    @property
    def average_execution_time_ms(self) -> float:
        if self.total_completed == 0:
            return 0
        return self.total_execution_time_ms / self.total_completed

    @property
    def success_rate(self) -> float:
        total = self.total_completed + self.total_failed
        if total == 0:
            return 1.0
        return self.total_completed / total


class TaskQueueFullError(Exception):
    """Raised when the task queue is full and rejecting new tasks."""

    pass


class TaskQueueManager:
    """
    Bounded async task queue with priority execution.

    Features:
    - Priority-based execution (CRITICAL > HIGH > NORMAL > LOW)
    - Configurable worker count and queue size
    - Timeout and retry support per task
    - Backpressure handling (reject when full)
    - Statistics tracking
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        worker_count: int = 4,
        default_timeout: float | None = 60.0,
        max_tracked_tasks: int = 5000,
        auto_cleanup_age_seconds: float = 1800.0,
    ):
        self._max_queue_size = max_queue_size
        self._worker_count = worker_count
        self._default_timeout = default_timeout
        self._max_tracked_tasks = max_tracked_tasks
        self._auto_cleanup_age_seconds = auto_cleanup_age_seconds

        self._queue: asyncio.PriorityQueue[QueuedTask] = asyncio.PriorityQueue()
        self._tasks: dict[str, QueuedTask] = {}
        self._pruned_terminal_states: dict[str, TaskState] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._retry_timers: set[asyncio.Task[None]] = set()
        self._stats = QueueStats()
        self._running = False
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

    @property
    def stats(self) -> QueueStats:
        """Get current queue statistics."""
        self._stats.current_queue_size = self._queue.qsize()
        self._stats.current_running = sum(
            1 for t in self._tasks.values() if t.state == TaskState.RUNNING
        )
        return self._stats

    @property
    def is_running(self) -> bool:
        """Check if the queue is running."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        """Check if the queue is at capacity."""
        return self._queue.qsize() >= self._max_queue_size

    async def start(self) -> None:
        """Start the task queue workers."""
        if self._running:
            return

        self._running = True
        self._shutdown_event.clear()

        for i in range(self._worker_count):
            worker = asyncio.create_task(self._worker_loop(i), name=f"queue_worker_{i}")
            self._workers.append(worker)

        logger.info(
            f"Task queue started with {self._worker_count} workers, "
            f"max queue size: {self._max_queue_size}"
        )

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the task queue gracefully.

        Args:
            timeout: Maximum time to wait for workers to finish
        """
        if not self._running:
            return

        logger.info("Stopping task queue...")
        self._running = False
        self._shutdown_event.set()

        # Wait for workers with timeout
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=timeout,
                )
            except TimeoutError:
                logger.warning("Worker shutdown timed out, cancelling...")
                for worker in self._workers:
                    worker.cancel()
                await asyncio.gather(*self._workers, return_exceptions=True)

        if self._retry_timers:
            for timer in list(self._retry_timers):
                timer.cancel()
            await asyncio.gather(*self._retry_timers, return_exceptions=True)
            self._retry_timers.clear()

        self._workers.clear()
        logger.info("Task queue stopped")

    async def submit(
        self,
        func: Callable[[], Awaitable[T]],
        name: str | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float | None = None,
        max_retries: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Submit a task for execution.

        Args:
            func: Async function to execute
            name: Optional task name for logging
            priority: Task priority level
            timeout: Timeout in seconds (None = use default)
            max_retries: Maximum retry attempts on failure
            metadata: Optional metadata to attach to task

        Returns:
            Task ID

        Raises:
            TaskQueueFullError: If queue is at capacity
        """
        if self.is_full:
            self._stats.total_rejected += 1
            raise TaskQueueFullError(
                f"Task queue full ({self._max_queue_size} tasks), rejecting new task"
            )

        task_id = str(uuid.uuid4())
        task = QueuedTask(
            id=task_id,
            name=name or f"task_{task_id[:8]}",
            priority=priority,
            func=func,
            timeout_seconds=timeout if timeout is not None else self._default_timeout,
            max_retries=max_retries,
            metadata=metadata or {},
        )

        async with self._lock:
            self._tasks[task_id] = task
            await self._queue.put(task)
            self._stats.total_submitted += 1

        logger.debug(f"Task submitted: {task.name} (priority: {priority.name})")
        return task_id

    async def get_task(self, task_id: str) -> QueuedTask | None:
        """Get a task by its ID."""
        return self._tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task.

        Returns True if the task was cancelled, False if it was already running/completed.
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            if task.state == TaskState.PENDING:
                task.state = TaskState.CANCELLED
                task.completed_at = datetime.now()
                self._stats.total_cancelled += 1
                logger.debug(f"Task cancelled: {task.name}")
                self._enforce_task_retention()
                return True

        return False

    async def wait_for_task(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> QueuedTask:
        """
        Wait for a task to complete.

        Args:
            task_id: Task ID to wait for
            timeout: Maximum time to wait

        Returns:
            The completed task
        """
        start = datetime.now()
        while True:
            task = self._tasks.get(task_id)
            if not task:
                if task_id in self._pruned_terminal_states:
                    # Task record was compacted to control memory growth.
                    return QueuedTask(
                        id=task_id,
                        name=f"task_{task_id[:8]}",
                        priority=TaskPriority.NORMAL,
                        func=_completed_task_placeholder,
                        state=self._pruned_terminal_states[task_id],
                    )
                raise ValueError(f"Task {task_id} not found")

            if task.state in (
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.CANCELLED,
                TaskState.TIMEOUT,
            ):
                return task

            if timeout:
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed >= timeout:
                    raise TimeoutError(f"Timeout waiting for task {task_id}")

            await asyncio.sleep(0.1)

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop that processes tasks from the queue."""
        logger.debug(f"Worker {worker_id} started")

        while self._running or not self._queue.empty():
            try:
                # Use timeout to check for shutdown periodically
                try:
                    task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                # Skip cancelled tasks
                if task.state == TaskState.CANCELLED:
                    self._queue.task_done()
                    continue

                await self._execute_task(task, worker_id)
                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)

        logger.debug(f"Worker {worker_id} stopped")

    async def _execute_task(self, task: QueuedTask, worker_id: int) -> None:
        """Execute a single task with timeout and retry support."""
        task.started_at = datetime.now()
        task.state = TaskState.RUNNING

        logger.debug(f"Worker {worker_id} executing: {task.name}")

        try:
            if task.timeout_seconds:
                result = await asyncio.wait_for(
                    task.func(),
                    timeout=task.timeout_seconds,
                )
            else:
                result = await task.func()

            task.result = result
            task.state = TaskState.COMPLETED
            task.completed_at = datetime.now()

            self._stats.total_completed += 1
            if task.duration_ms:
                self._stats.total_execution_time_ms += task.duration_ms
            if task.wait_time_ms:
                self._stats.total_wait_time_ms += task.wait_time_ms

            logger.debug(f"Task completed: {task.name} ({task.duration_ms:.1f}ms)")

        except TimeoutError:
            task.state = TaskState.TIMEOUT
            task.completed_at = datetime.now()
            self._stats.total_timeout += 1
            logger.warning(f"Task timeout: {task.name}")

            # Retry on timeout if allowed
            if task.retry_count < task.max_retries:
                await self._retry_task(task)
            else:
                self._enforce_task_retention()

        except Exception as e:
            task.error = e
            task.completed_at = datetime.now()
            logger.error(f"Task failed: {task.name} - {e}")

            # Retry on failure if allowed
            if task.retry_count < task.max_retries:
                await self._retry_task(task)
            else:
                task.state = TaskState.FAILED
                self._stats.total_failed += 1
                self._enforce_task_retention()
        else:
            self._enforce_task_retention()

    async def _retry_task(self, task: QueuedTask) -> None:
        """Requeue a task for retry."""
        task.retry_count += 1
        task.state = TaskState.PENDING
        task.started_at = None
        task.completed_at = None
        task.result = None
        task.error = None

        self._stats.total_retried += 1

        # Exponential backoff
        delay = 2**task.retry_count
        logger.info(
            f"Retrying task {task.name} (attempt {task.retry_count + 1}/{task.max_retries + 1}) "
            f"after {delay}s"
        )
        retry_timer = asyncio.create_task(
            self._delayed_requeue(task, delay),
            name=f"task_retry_{task.id[:8]}",
        )
        self._retry_timers.add(retry_timer)
        retry_timer.add_done_callback(self._retry_timers.discard)

    async def _delayed_requeue(self, task: QueuedTask, delay: float) -> None:
        """Delay task requeue without blocking a worker."""
        try:
            await asyncio.sleep(delay)
            if self._running and task.state == TaskState.PENDING:
                await self._queue.put(task)
        except asyncio.CancelledError:
            raise

    def get_pending_tasks(self) -> list[QueuedTask]:
        """Get all pending tasks."""
        return [t for t in self._tasks.values() if t.state == TaskState.PENDING]

    def get_running_tasks(self) -> list[QueuedTask]:
        """Get all currently running tasks."""
        return [t for t in self._tasks.values() if t.state == TaskState.RUNNING]

    def cleanup_completed(self, max_age_seconds: float = 3600) -> int:
        """
        Remove completed tasks older than max_age_seconds.

        Returns the number of tasks removed.
        """
        cutoff = datetime.now()
        removed = 0

        task_ids_to_remove = []
        for task_id, task in self._tasks.items():
            if task.state in (
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.CANCELLED,
                TaskState.TIMEOUT,
            ):
                if task.completed_at:
                    age = (cutoff - task.completed_at).total_seconds()
                    if age > max_age_seconds:
                        task_ids_to_remove.append(task_id)

        for task_id in task_ids_to_remove:
            self._pruned_terminal_states[task_id] = self._tasks[task_id].state
            del self._tasks[task_id]
            removed += 1

        if removed:
            logger.debug(f"Cleaned up {removed} completed tasks")

        return removed

    def _enforce_task_retention(self) -> None:
        """Bound memory usage from task history growth."""
        self.cleanup_completed(max_age_seconds=self._auto_cleanup_age_seconds)

        overshoot = len(self._tasks) - self._max_tracked_tasks
        if overshoot <= 0:
            return

        terminal_tasks = sorted(
            (
                (task_id, task)
                for task_id, task in self._tasks.items()
                if task.state
                in (
                    TaskState.COMPLETED,
                    TaskState.FAILED,
                    TaskState.CANCELLED,
                    TaskState.TIMEOUT,
                )
            ),
            key=lambda item: item[1].completed_at or item[1].created_at,
        )

        removed = 0
        for task_id, _ in terminal_tasks:
            if removed >= overshoot:
                break
            self._pruned_terminal_states[task_id] = self._tasks[task_id].state
            del self._tasks[task_id]
            removed += 1

        self._trim_pruned_terminal_states()

        if removed:
            logger.debug(
                "Pruned %s terminal task records to keep max tracked tasks at %s",
                removed,
                self._max_tracked_tasks,
            )

    def _trim_pruned_terminal_states(self) -> None:
        """Bound compacted task metadata used for wait_for_task lookups."""
        max_entries = self._max_tracked_tasks * 2
        if len(self._pruned_terminal_states) <= max_entries:
            return

        # Dict preserves insertion order in Python 3.7+.
        overshoot = len(self._pruned_terminal_states) - max_entries
        for task_id in list(self._pruned_terminal_states.keys())[:overshoot]:
            del self._pruned_terminal_states[task_id]


async def _completed_task_placeholder() -> None:
    """Placeholder coroutine for compacted task records."""
    return None
