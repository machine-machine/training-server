"""Task supervision with automatic restart and exponential backoff.

Provides a TaskSupervisor that wraps asyncio tasks with:
- Exponential backoff on repeated failures
- Configurable max restart count per task
- Critical task flagging (logged at ERROR/CRITICAL level)
- Restart counter reset after sustained uptime (30 minutes)
- Graceful shutdown integration via asyncio.Event
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass
class TaskState:
    """Tracks the state of a supervised task."""

    name: str
    restart_count: int = 0
    max_restarts: int = 10
    backoff_base: float = 5.0
    backoff_max: float = 300.0  # 5 minutes max
    last_failure: datetime | None = None
    last_restart: datetime | None = None
    task: asyncio.Task | None = None
    is_critical: bool = False  # Critical tasks log at CRITICAL on max restarts

    @property
    def current_backoff(self) -> float:
        """Calculate current backoff with exponential increase."""
        backoff = self.backoff_base * (2 ** min(self.restart_count, 8))
        return min(backoff, self.backoff_max)


class TaskSupervisor:
    """
    Supervises async tasks with automatic restart on failure.

    Features:
    - Exponential backoff on repeated failures
    - Configurable max restart count per task
    - Critical task flagging (logged at CRITICAL level)
    - Restart counter reset after sustained uptime (30 minutes)
    - Graceful shutdown integration via asyncio.Event
    """

    def __init__(self, shutdown_event: asyncio.Event):
        self._shutdown_event = shutdown_event
        self._task_states: dict[str, TaskState] = {}
        self._tasks: list[asyncio.Task] = []
        self._healthy_uptime_reset = 1800  # Reset restart counter after 30 min

    def register(
        self,
        name: str,
        coro_fn: Callable[[], Coroutine[Any, Any, None]],
        max_restarts: int = 10,
        is_critical: bool = False,
        backoff_base: float = 5.0,
    ) -> asyncio.Task:
        """Register and start a supervised task."""
        state = TaskState(
            name=name,
            max_restarts=max_restarts,
            is_critical=is_critical,
            backoff_base=backoff_base,
        )
        self._task_states[name] = state

        task = asyncio.create_task(
            self._supervise(state, coro_fn),
            name=f"supervised-{name}",
        )
        state.task = task
        self._tasks.append(task)
        return task

    async def _supervise(
        self,
        state: TaskState,
        coro_fn: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """Supervision loop with restart logic."""
        last_healthy_start = datetime.now()

        while not self._shutdown_event.is_set():
            try:
                start_time = datetime.now()
                await coro_fn()
                # Normal exit (e.g., shutdown event set inside the loop)
                logger.info(f"Task '{state.name}' exited normally")
                break

            except asyncio.CancelledError:
                logger.debug(f"Task '{state.name}' cancelled")
                raise  # Propagate cancellation

            except Exception as e:
                state.restart_count += 1
                state.last_failure = datetime.now()
                run_duration = (datetime.now() - start_time).total_seconds()

                log_fn = logger.critical if state.is_critical else logger.error
                log_fn(
                    f"Task '{state.name}' failed after {run_duration:.1f}s "
                    f"(restart {state.restart_count}/{state.max_restarts}): {e}",
                    exc_info=True,
                )

                if state.restart_count >= state.max_restarts:
                    logger.critical(
                        f"Task '{state.name}' exceeded {state.max_restarts} restarts. "
                        f"Giving up. Manual intervention required."
                    )
                    break

                # Reset restart counter if task ran successfully for a while
                uptime = (datetime.now() - last_healthy_start).total_seconds()
                if uptime > self._healthy_uptime_reset:
                    old_count = state.restart_count
                    state.restart_count = 1  # Reset to 1 (current failure)
                    logger.info(
                        f"Task '{state.name}' ran for {uptime:.0f}s, "
                        f"reset restart counter {old_count} -> 1"
                    )

                backoff = state.current_backoff
                logger.info(
                    f"Restarting '{state.name}' in {backoff:.1f}s (attempt {state.restart_count})"
                )

                # Wait for backoff or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=backoff,
                    )
                    break  # Shutdown event was set during backoff
                except TimeoutError:
                    pass  # Backoff completed, restart

                state.last_restart = datetime.now()
                last_healthy_start = datetime.now()

    def get_status(self) -> dict[str, Any]:
        """Get status of all supervised tasks."""
        return {
            name: {
                "running": state.task is not None and not state.task.done(),
                "restart_count": state.restart_count,
                "max_restarts": state.max_restarts,
                "last_failure": (state.last_failure.isoformat() if state.last_failure else None),
                "is_critical": state.is_critical,
            }
            for name, state in self._task_states.items()
        }

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Shut down all supervised tasks."""
        self._shutdown_event.set()

        active = [t for t in self._tasks if not t.done()]
        if not active:
            return

        logger.info(f"Supervisor shutting down {len(active)} active tasks")

        for task in active:
            task.cancel()

        try:
            await asyncio.wait_for(
                asyncio.gather(*active, return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            still_running = [t for t in active if not t.done()]
            logger.warning(f"Supervisor shutdown timeout, {len(still_running)} tasks forced")
