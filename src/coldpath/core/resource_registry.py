"""
Central resource registry for lifecycle management.

Provides unified registration, initialization, and shutdown
of all managed resources with dependency ordering.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ResourceState(Enum):
    """Lifecycle states for managed resources."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ResourceHealth:
    """Health status of a resource."""

    healthy: bool
    message: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.now)

    @classmethod
    def ok(cls, message: str | None = None, **metrics: float) -> "ResourceHealth":
        return cls(healthy=True, message=message, metrics=metrics)

    @classmethod
    def degraded(cls, message: str, **metrics: float) -> "ResourceHealth":
        return cls(healthy=False, message=message, metrics=metrics)


class ManagedResource(ABC):
    """
    Abstract base class for all managed resources.

    Implementations must provide lifecycle methods and
    can optionally specify dependencies on other resources.
    """

    @property
    @abstractmethod
    def resource_id(self) -> str:
        """Unique identifier for this resource."""
        pass

    @property
    @abstractmethod
    def resource_type(self) -> str:
        """Type name for this resource."""
        pass

    @property
    def dependencies(self) -> list[str]:
        """List of resource IDs this resource depends on."""
        return []

    @property
    def state(self) -> ResourceState:
        """Current lifecycle state."""
        return getattr(self, "_state", ResourceState.UNINITIALIZED)

    @state.setter
    def state(self, value: ResourceState) -> None:
        self._state = value

    async def initialize(self) -> None:  # noqa: B027
        """Optional hook to initialize the resource. Override in subclass if needed."""

    async def start(self) -> None:  # noqa: B027
        """Optional hook to start the resource. Override in subclass if needed."""

    async def stop(self) -> None:  # noqa: B027
        """Optional hook to stop the resource gracefully. Override in subclass if needed."""

    async def health_check(self) -> ResourceHealth:
        """Check resource health. Override in subclass if needed."""
        return ResourceHealth.ok()

    async def cleanup(self) -> None:  # noqa: B027
        """Optional hook to cleanup resources. Override in subclass if needed."""


@dataclass
class ResourceInfo:
    """Information about a registered resource."""

    resource: ManagedResource
    registered_at: datetime = field(default_factory=datetime.now)
    initialized_at: datetime | None = None
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    last_health_check: ResourceHealth | None = None
    initialization_order: int = 0
    error_count: int = 0
    last_error: str | None = None


class ResourceRegistry:
    """
    Central registry for managing resource lifecycles.

    Features:
    - Dependency-ordered initialization and shutdown
    - Health check aggregation
    - Automatic recovery callbacks
    - Resource state tracking
    """

    def __init__(self):
        self._resources: dict[str, ResourceInfo] = {}
        self._initialization_order: list[str] = []
        self._lock = asyncio.Lock()
        self._on_failure_callbacks: list[Callable[[str, Exception], None]] = []

    @property
    def resource_count(self) -> int:
        """Get number of registered resources."""
        return len(self._resources)

    def register(self, resource: ManagedResource) -> None:
        """
        Register a resource with the registry.

        Args:
            resource: The resource to register
        """
        resource_id = resource.resource_id

        if resource_id in self._resources:
            logger.warning(f"Resource '{resource_id}' already registered, updating")

        self._resources[resource_id] = ResourceInfo(resource=resource)
        logger.info(f"Registered resource: {resource_id} (type: {resource.resource_type})")

    def unregister(self, resource_id: str) -> bool:
        """
        Unregister a resource.

        Returns True if the resource was found and removed.
        """
        if resource_id in self._resources:
            del self._resources[resource_id]
            if resource_id in self._initialization_order:
                self._initialization_order.remove(resource_id)
            logger.info(f"Unregistered resource: {resource_id}")
            return True
        return False

    def get(self, resource_id: str) -> ManagedResource | None:
        """Get a resource by ID."""
        info = self._resources.get(resource_id)
        return info.resource if info else None

    def get_by_type(self, resource_type: str) -> list[ManagedResource]:
        """Get all resources of a specific type."""
        return [
            info.resource
            for info in self._resources.values()
            if info.resource.resource_type == resource_type
        ]

    def on_failure(
        self,
        callback: Callable[[str, Exception], None],
    ) -> None:
        """Register a callback for resource failures."""
        self._on_failure_callbacks.append(callback)

    async def initialize_all(self, timeout: float = 60.0) -> dict[str, bool]:
        """
        Initialize all resources in dependency order.

        Args:
            timeout: Maximum time for initialization

        Returns:
            Dict mapping resource ID to success status
        """
        async with self._lock:
            # Compute initialization order
            order = self._compute_dependency_order()
            self._initialization_order = order

            results: dict[str, bool] = {}

            for idx, resource_id in enumerate(order):
                info = self._resources[resource_id]
                resource = info.resource

                try:
                    resource.state = ResourceState.INITIALIZING
                    logger.debug(f"Initializing: {resource_id}")

                    await asyncio.wait_for(
                        resource.initialize(),
                        timeout=timeout / len(order),
                    )

                    info.initialized_at = datetime.now()
                    info.initialization_order = idx
                    resource.state = ResourceState.RUNNING
                    results[resource_id] = True

                    logger.info(f"Initialized: {resource_id}")

                except TimeoutError:
                    resource.state = ResourceState.FAILED
                    info.error_count += 1
                    info.last_error = "Initialization timeout"
                    results[resource_id] = False
                    logger.error(f"Timeout initializing: {resource_id}")

                except Exception as e:
                    resource.state = ResourceState.FAILED
                    info.error_count += 1
                    info.last_error = str(e)
                    results[resource_id] = False
                    logger.error(f"Error initializing {resource_id}: {e}")
                    self._notify_failure(resource_id, e)

            return results

    async def start_all(self) -> dict[str, bool]:
        """
        Start all initialized resources in dependency order.

        Returns:
            Dict mapping resource ID to success status
        """
        results: dict[str, bool] = {}

        for resource_id in self._initialization_order:
            info = self._resources.get(resource_id)
            if not info or info.resource.state != ResourceState.RUNNING:
                continue

            try:
                await info.resource.start()
                info.started_at = datetime.now()
                results[resource_id] = True
                logger.debug(f"Started: {resource_id}")

            except Exception as e:
                info.resource.state = ResourceState.DEGRADED
                info.error_count += 1
                info.last_error = str(e)
                results[resource_id] = False
                logger.error(f"Error starting {resource_id}: {e}")
                self._notify_failure(resource_id, e)

        return results

    async def shutdown_all(self, timeout: float = 30.0) -> dict[str, bool]:
        """
        Shutdown all resources in reverse dependency order.

        Args:
            timeout: Maximum time for shutdown

        Returns:
            Dict mapping resource ID to success status
        """
        async with self._lock:
            # Shutdown in reverse initialization order
            shutdown_order = list(reversed(self._initialization_order))
            results: dict[str, bool] = {}

            for resource_id in shutdown_order:
                info = self._resources.get(resource_id)
                if not info:
                    continue

                resource = info.resource
                if resource.state not in (
                    ResourceState.RUNNING,
                    ResourceState.DEGRADED,
                ):
                    results[resource_id] = True
                    continue

                try:
                    resource.state = ResourceState.STOPPING
                    logger.debug(f"Stopping: {resource_id}")

                    await asyncio.wait_for(
                        resource.stop(),
                        timeout=timeout / max(len(shutdown_order), 1),
                    )

                    await resource.cleanup()

                    resource.state = ResourceState.STOPPED
                    info.stopped_at = datetime.now()
                    results[resource_id] = True

                    logger.info(f"Stopped: {resource_id}")

                except TimeoutError:
                    resource.state = ResourceState.FAILED
                    results[resource_id] = False
                    logger.warning(f"Timeout stopping: {resource_id}")

                except Exception as e:
                    resource.state = ResourceState.FAILED
                    results[resource_id] = False
                    logger.error(f"Error stopping {resource_id}: {e}")

            return results

    async def check_health(self) -> dict[str, ResourceHealth]:
        """
        Check health of all running resources.

        Returns:
            Dict mapping resource ID to health status
        """
        results: dict[str, ResourceHealth] = {}

        for resource_id, info in self._resources.items():
            if info.resource.state not in (
                ResourceState.RUNNING,
                ResourceState.DEGRADED,
            ):
                continue

            try:
                health = await info.resource.health_check()
                info.last_health_check = health
                results[resource_id] = health

                # Update state based on health
                if health.healthy and info.resource.state == ResourceState.DEGRADED:
                    info.resource.state = ResourceState.RUNNING
                elif not health.healthy and info.resource.state == ResourceState.RUNNING:
                    info.resource.state = ResourceState.DEGRADED

            except Exception as e:
                health = ResourceHealth.degraded(f"Health check failed: {e}")
                info.last_health_check = health
                results[resource_id] = health
                logger.error(f"Health check error for {resource_id}: {e}")

        return results

    def _compute_dependency_order(self) -> list[str]:
        """Compute initialization order using topological sort."""
        order: list[str] = []
        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(resource_id: str) -> None:
            if resource_id in visited:
                return
            if resource_id in visiting:
                logger.warning(f"Circular dependency detected involving: {resource_id}")
                return

            visiting.add(resource_id)

            info = self._resources.get(resource_id)
            if info:
                for dep_id in info.resource.dependencies:
                    if dep_id in self._resources:
                        visit(dep_id)

            visiting.remove(resource_id)
            visited.add(resource_id)
            order.append(resource_id)

        for resource_id in self._resources:
            visit(resource_id)

        return order

    def _notify_failure(self, resource_id: str, error: Exception) -> None:
        """Notify registered callbacks of a resource failure."""
        for callback in self._on_failure_callbacks:
            try:
                callback(resource_id, error)
            except Exception as e:
                logger.error(f"Error in failure callback: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get status summary of all resources."""
        status: dict[str, Any] = {
            "total": len(self._resources),
            "by_state": {},
            "resources": {},
        }

        for resource_id, info in self._resources.items():
            state = info.resource.state
            status["by_state"][state.value] = status["by_state"].get(state.value, 0) + 1
            status["resources"][resource_id] = {
                "type": info.resource.resource_type,
                "state": state.value,
                "initialized_at": (
                    info.initialized_at.isoformat() if info.initialized_at else None
                ),
                "error_count": info.error_count,
                "last_error": info.last_error,
                "last_health": (
                    {
                        "healthy": info.last_health_check.healthy,
                        "message": info.last_health_check.message,
                    }
                    if info.last_health_check
                    else None
                ),
            }

        return status

    async def __aenter__(self) -> "ResourceRegistry":
        await self.initialize_all()
        await self.start_all()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.shutdown_all()


# Convenience decorator for creating managed resources
def managed_resource(
    resource_id: str,
    resource_type: str,
    dependencies: list[str] | None = None,
) -> Callable[[type], type]:
    """
    Decorator to convert a class into a managed resource.

    Example:
        @managed_resource("my-service", "service", dependencies=["database"])
        class MyService:
            async def initialize(self):
                ...
    """

    def decorator(cls: type) -> type:
        original_init = cls.__init__

        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            self._resource_id = resource_id
            self._resource_type = resource_type
            self._dependencies = dependencies or []
            self._state = ResourceState.UNINITIALIZED
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        cls.resource_id = property(lambda self: self._resource_id)
        cls.resource_type = property(lambda self: self._resource_type)
        cls.dependencies = property(lambda self: self._dependencies)
        cls.state = property(
            lambda self: self._state,
            lambda self, v: setattr(self, "_state", v),
        )

        # Add default implementations if not present
        if not hasattr(cls, "health_check"):

            async def health_check(self: Any) -> ResourceHealth:
                return ResourceHealth.ok()

            cls.health_check = health_check

        return cls

    return decorator
