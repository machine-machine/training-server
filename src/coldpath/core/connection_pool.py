"""
Centralized HTTP connection pooling with circuit breaker pattern.

Provides managed connection pools with automatic retry, circuit breaking,
and health tracking for all HTTP-based services.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    half_open_max_requests: int = 3


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    state_changes: int = 0


class CircuitBreaker:
    """
    Circuit breaker implementation for service protection.

    Automatically opens when failure threshold is exceeded,
    and transitions through half-open state to test recovery.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._opened_at: datetime | None = None
        self._half_open_count = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for timeout-based transitions."""
        if self._state == CircuitState.OPEN and self._opened_at:
            elapsed = (datetime.now() - self._opened_at).total_seconds()
            if elapsed >= self.config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        return self._stats

    @property
    def is_allowing_requests(self) -> bool:
        """Check if requests are allowed."""
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._stats.state_changes += 1

        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.now()
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_count = 0

        logger.info(f"Circuit breaker '{self.name}': {old_state.value} -> {new_state.value}")

    async def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        async with self._lock:
            state = self.state

            if state == CircuitState.CLOSED:
                return True

            if state == CircuitState.OPEN:
                self._stats.rejected_requests += 1
                return False

            # HALF_OPEN
            if self._half_open_count >= self.config.half_open_max_requests:
                self._stats.rejected_requests += 1
                return False

            self._half_open_count += 1
            return True

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            self._stats.total_requests += 1
            self._stats.successful_requests += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    async def record_failure(self) -> None:
        """Record a failed request."""
        async with self._lock:
            self._stats.total_requests += 1
            self._stats.failed_requests += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = datetime.now()

            if self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)

    async def execute(self, func: Callable[[], T]) -> T:
        """Execute a function with circuit breaker protection."""
        if not await self.allow_request():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open",
                remaining_seconds=self._time_until_half_open(),
            )

        try:
            result = await func() if asyncio.iscoroutinefunction(func) else func()
            await self.record_success()
            return result
        except Exception:
            await self.record_failure()
            raise

    def _time_until_half_open(self) -> float:
        """Calculate time remaining until half-open transition."""
        if self._opened_at and self._state == CircuitState.OPEN:
            elapsed = (datetime.now() - self._opened_at).total_seconds()
            return max(0, self.config.timeout_seconds - elapsed)
        return 0

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._opened_at = None
        self._half_open_count = 0
        logger.info(f"Circuit breaker '{self.name}' reset")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, remaining_seconds: float = 0):
        super().__init__(message)
        self.remaining_seconds = remaining_seconds


@dataclass
class PoolStats:
    """Statistics for a connection pool."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    request_count_for_avg: int = 0

    @property
    def average_latency_ms(self) -> float:
        if self.request_count_for_avg == 0:
            return 0
        return self.total_latency_ms / self.request_count_for_avg

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


@dataclass
class ServiceConfig:
    """Configuration for a managed service."""

    base_url: str
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    circuit_breaker_config: CircuitBreakerConfig | None = None
    headers: dict[str, str] = field(default_factory=dict)


class ConnectionPoolManager:
    """
    Centralized connection pool manager with circuit breakers.

    Provides managed HTTP clients for all services with automatic
    retry, circuit breaking, and statistics tracking.
    """

    def __init__(self, default_timeout: float = 30.0):
        self._default_timeout = default_timeout
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._stats: dict[str, PoolStats] = {}
        self._configs: dict[str, ServiceConfig] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def register_service(
        self,
        name: str,
        config: ServiceConfig,
    ) -> None:
        """Register a service with its configuration."""
        # Get old client if exists (outside lock to prevent deadlock)
        old_client = None
        async with self._lock:
            if name in self._clients:
                logger.warning(f"Service '{name}' already registered, updating config")
                old_client = self._clients[name]

        # Close outside lock to prevent deadlock if close times out
        if old_client:
            await old_client.aclose()

        # Now create new client
        async with self._lock:
            self._configs[name] = config

            # Create HTTP client with connection pooling
            self._clients[name] = httpx.AsyncClient(
                base_url=config.base_url,
                timeout=httpx.Timeout(config.timeout_seconds),
                headers=config.headers,
                limits=httpx.Limits(
                    max_keepalive_connections=20,
                    max_connections=100,
                    keepalive_expiry=30,
                ),
            )

            # Create circuit breaker
            cb_config = config.circuit_breaker_config or CircuitBreakerConfig()
            self._circuit_breakers[name] = CircuitBreaker(name, cb_config)

            # Initialize stats
            self._stats[name] = PoolStats()

            logger.info(f"Registered service '{name}' with base URL: {config.base_url}")

    async def get_client(self, service: str) -> httpx.AsyncClient:
        """Get the HTTP client for a service."""
        if service not in self._clients:
            raise ValueError(f"Service '{service}' not registered")
        return self._clients[service]

    def get_circuit_breaker(self, service: str) -> CircuitBreaker:
        """Get the circuit breaker for a service."""
        if service not in self._circuit_breakers:
            raise ValueError(f"Service '{service}' not registered")
        return self._circuit_breakers[service]

    async def request(
        self,
        service: str,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Make a request to a service with circuit breaker protection.

        Args:
            service: Name of the registered service
            method: HTTP method (GET, POST, etc.)
            path: URL path (appended to service base URL)
            **kwargs: Additional arguments for httpx request
        """
        if service not in self._clients:
            raise ValueError(f"Service '{service}' not registered")

        client = self._clients[service]
        circuit_breaker = self._circuit_breakers[service]
        config = self._configs[service]
        stats = self._stats[service]

        async def make_request() -> httpx.Response:
            start_time = time.monotonic()
            try:
                response = await client.request(method, path, **kwargs)
                response.raise_for_status()

                latency = (time.monotonic() - start_time) * 1000
                stats.total_requests += 1
                stats.successful_requests += 1
                stats.total_latency_ms += latency
                stats.request_count_for_avg += 1

                return response
            except Exception:
                stats.total_requests += 1
                stats.failed_requests += 1
                raise

        # Execute with circuit breaker
        last_error: Exception | None = None
        for attempt in range(config.max_retries):
            try:
                return await circuit_breaker.execute(make_request)
            except CircuitBreakerOpenError:
                raise
            except Exception as e:
                last_error = e
                if attempt < config.max_retries - 1:
                    await asyncio.sleep(config.retry_delay_seconds * (2**attempt))
                    logger.warning(
                        f"Request to {service}{path} failed (attempt {attempt + 1}), retrying..."
                    )

        raise last_error or Exception("Request failed after all retries")

    async def get(self, service: str, path: str, **kwargs: Any) -> httpx.Response:
        """Convenience method for GET requests."""
        return await self.request(service, "GET", path, **kwargs)

    async def post(self, service: str, path: str, **kwargs: Any) -> httpx.Response:
        """Convenience method for POST requests."""
        return await self.request(service, "POST", path, **kwargs)

    def get_stats(self, service: str) -> PoolStats | None:
        """Get statistics for a service."""
        return self._stats.get(service)

    def get_all_stats(self) -> dict[str, PoolStats]:
        """Get statistics for all services."""
        return dict(self._stats)

    def get_circuit_states(self) -> dict[str, CircuitState]:
        """Get circuit breaker states for all services."""
        return {name: cb.state for name, cb in self._circuit_breakers.items()}

    async def health_check(self) -> dict[str, bool]:
        """Check health of all registered services."""
        results = {}
        for name in self._clients:
            cb = self._circuit_breakers[name]
            results[name] = cb.is_allowing_requests
        return results

    async def close(self) -> None:
        """Close all connections and cleanup."""
        if self._closed:
            return

        self._closed = True
        async with self._lock:
            for name, client in self._clients.items():
                try:
                    await client.aclose()
                    logger.debug(f"Closed client for service '{name}'")
                except Exception as e:
                    logger.error(f"Error closing client for '{name}': {e}")

            self._clients.clear()
            self._circuit_breakers.clear()

        logger.info("Connection pool manager closed")

    async def __aenter__(self) -> "ConnectionPoolManager":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


# Convenience function for creating a managed Anthropic client wrapper
async def create_anthropic_pool(
    pool_manager: ConnectionPoolManager,
    api_key: str,
    base_url: str = "https://api.anthropic.com",
) -> None:
    """Register an Anthropic API service with the pool manager."""
    config = ServiceConfig(
        base_url=base_url,
        timeout_seconds=120.0,  # LLM requests can be slow
        max_retries=2,
        retry_delay_seconds=2.0,
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout_seconds=120.0,
        ),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    await pool_manager.register_service("anthropic", config)
    logger.info("Anthropic API service registered with connection pool")
