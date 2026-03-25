"""
Monitoring and observability for ColdPath services.

Provides Prometheus-style metrics, health checks, and dashboards for:
- Feature Store
- A/B Testing Framework
- Social Features Service
- Telegram Signal Collector

Usage:
    from coldpath.services.monitoring import ServicesMonitor

    monitor = ServicesMonitor()
    monitor.register_feature_store(feature_store)
    monitor.register_ab_framework(ab_framework)
    monitor.register_social_service(social_service)

    # Get Prometheus metrics
    metrics = monitor.get_prometheus_metrics()

    # Get health report
    health = await monitor.get_health_report()

    # FastAPI endpoint
    @app.get("/metrics")
    async def metrics():
        return Response(content=monitor.get_prometheus_metrics(), media_type="text/plain")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from coldpath.services.ab_testing import ABTestingFramework
    from coldpath.services.feature_store import FeatureStore
    from coldpath.services.social_features_service import SocialFeaturesService
    from coldpath.services.telegram_signals import TelegramSignalCollector


@dataclass
class ServiceMetrics:
    """Metrics for a single service."""

    service_name: str
    healthy: bool = True
    last_check_ms: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    last_error: str | None = None
    last_error_time: datetime | None = None
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "healthy": self.healthy,
            "last_check_ms": self.last_check_ms,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                self.successful_requests / self.total_requests if self.total_requests > 0 else 1.0
            ),
            "avg_latency_ms": self.avg_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "last_error": self.last_error,
            "last_error_time": (self.last_error_time.isoformat() if self.last_error_time else None),
            "custom_metrics": self.custom_metrics,
        }


@dataclass
class ServicesHealthReport:
    """Aggregated health report for all services."""

    timestamp: datetime
    overall_healthy: bool
    services: dict[str, ServiceMetrics]
    uptime_seconds: float

    @property
    def healthy_count(self) -> int:
        return sum(1 for m in self.services.values() if m.healthy)

    @property
    def unhealthy_count(self) -> int:
        return sum(1 for m in self.services.values() if not m.healthy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_healthy": self.overall_healthy,
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
            "uptime_seconds": self.uptime_seconds,
            "services": {k: v.to_dict() for k, v in self.services.items()},
        }


class ServicesMonitor:
    """Centralized monitoring for all ColdPath services.

    Features:
    - Prometheus-compatible metrics export
    - Health checks with configurable intervals
    - Latency tracking with percentiles
    - Error rate monitoring
    - Service-specific custom metrics
    """

    def __init__(self, service_name: str = "coldpath"):
        self.service_name = service_name
        self._start_time = time.time()

        self._feature_store: FeatureStore | None = None
        self._ab_framework: ABTestingFramework | None = None
        self._social_service: SocialFeaturesService | None = None
        self._telegram_collector: TelegramSignalCollector | None = None

        self._metrics: dict[str, ServiceMetrics] = {}
        self._latency_samples: dict[str, list[float]] = {}

        self._init_metrics()

    def _init_metrics(self):
        """Initialize metrics for all services."""
        self._metrics["feature_store"] = ServiceMetrics(
            service_name="feature_store",
            custom_metrics={
                "cache_size": 0,
                "backend": "unknown",
            },
        )
        self._metrics["ab_testing"] = ServiceMetrics(
            service_name="ab_testing",
            custom_metrics={
                "active_experiments": 0,
                "total_outcomes": 0,
            },
        )
        self._metrics["social_features"] = ServiceMetrics(
            service_name="social_features",
            custom_metrics={
                "cache_size": 0,
                "registered_tokens": 0,
                "fresh_entries": 0,
            },
        )
        self._metrics["telegram"] = ServiceMetrics(
            service_name="telegram",
            custom_metrics={
                "available": False,
                "groups_tracked": 0,
            },
        )

        for service in self._metrics:
            self._latency_samples[service] = []

    def register_feature_store(self, store: FeatureStore):
        """Register feature store for monitoring."""
        self._feature_store = store

    def register_ab_framework(self, framework: ABTestingFramework):
        """Register A/B testing framework for monitoring."""
        self._ab_framework = framework

    def register_social_service(self, service: SocialFeaturesService):
        """Register social features service for monitoring."""
        self._social_service = service

    def register_telegram_collector(self, collector: TelegramSignalCollector):
        """Register Telegram collector for monitoring."""
        self._telegram_collector = collector

    def record_request(
        self,
        service: str,
        success: bool,
        latency_ms: float,
        error: str | None = None,
    ):
        """Record a request to a service."""
        if service not in self._metrics:
            return

        metrics = self._metrics[service]
        metrics.total_requests += 1
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
            metrics.last_error = error
            metrics.last_error_time = datetime.now()

        self._latency_samples[service].append(latency_ms)
        if len(self._latency_samples[service]) > 1000:
            self._latency_samples[service] = self._latency_samples[service][-1000:]

        self._update_latency_stats(service)

    def _update_latency_stats(self, service: str):
        """Update latency statistics for a service."""
        samples = self._latency_samples[service]
        if not samples:
            return

        metrics = self._metrics[service]
        metrics.avg_latency_ms = sum(samples) / len(samples)
        sorted_samples = sorted(samples)
        p99_idx = int(len(sorted_samples) * 0.99)
        metrics.p99_latency_ms = sorted_samples[min(p99_idx, len(sorted_samples) - 1)]

    async def check_health(self) -> ServicesHealthReport:
        """Perform health check on all registered services."""
        now = datetime.now()
        check_start = time.time()

        for service_name, metrics in self._metrics.items():
            check_result = await self._check_service_health(service_name)
            metrics.healthy = check_result["healthy"]
            metrics.custom_metrics.update(check_result.get("metrics", {}))
            metrics.last_check_ms = (time.time() - check_start) * 1000

        overall_healthy = all(m.healthy for m in self._metrics.values())
        uptime = time.time() - self._start_time

        return ServicesHealthReport(
            timestamp=now,
            overall_healthy=overall_healthy,
            services=dict(self._metrics),
            uptime_seconds=uptime,
        )

    async def _check_service_health(self, service_name: str) -> dict[str, Any]:
        """Check health of a specific service."""
        result = {"healthy": True, "metrics": {}}

        try:
            if service_name == "feature_store" and self._feature_store:
                stats = self._feature_store.get_stats()
                result["metrics"] = {
                    "cache_size": stats.get("total_tokens", 0),
                    "backend": stats.get("backend", "unknown"),
                }

            elif service_name == "ab_testing" and self._ab_framework:
                experiments = self._ab_framework.list_experiments()
                active = sum(1 for e in experiments if e.get("status") == "running")
                result["metrics"] = {
                    "active_experiments": active,
                    "total_experiments": len(experiments),
                }

            elif service_name == "social_features" and self._social_service:
                stats = self._social_service.get_stats()
                result["metrics"] = {
                    "cache_size": stats.get("cache_size", 0),
                    "registered_tokens": stats.get("registered_tokens", 0),
                    "fresh_entries": stats.get("fresh_cache_entries", 0),
                    "running": stats.get("running", False),
                }

            elif service_name == "telegram":
                result["metrics"] = {
                    "available": self._telegram_collector is not None,
                    "groups_tracked": 0,
                }

        except Exception as e:
            logger.warning(f"Health check failed for {service_name}: {e}")
            result["healthy"] = False
            result["metrics"]["error"] = str(e)

        return result

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics output."""
        lines = []
        now = time.time()
        uptime = now - self._start_time

        lines.append(f"# HELP {self.service_name}_uptime_seconds Service uptime in seconds")
        lines.append(f"# TYPE {self.service_name}_uptime_seconds gauge")
        lines.append(f"{self.service_name}_uptime_seconds {uptime:.2f}")

        lines.append("")
        lines.append(f"# HELP {self.service_name}_services_total Total number of services")
        lines.append(f"# TYPE {self.service_name}_services_total gauge")
        lines.append(f"{self.service_name}_services_total {len(self._metrics)}")

        lines.append("")
        lines.append(f"# HELP {self.service_name}_services_healthy Number of healthy services")
        lines.append(f"# TYPE {self.service_name}_services_healthy gauge")
        healthy_count = sum(1 for m in self._metrics.values() if m.healthy)
        lines.append(f"{self.service_name}_services_healthy {healthy_count}")

        for service_name, metrics in self._metrics.items():
            prefix = f"{self.service_name}_{service_name}"

            lines.append("")
            lines.append(f"# HELP {prefix}_healthy Whether the service is healthy")
            lines.append(f"# TYPE {prefix}_healthy gauge")
            lines.append(f"{prefix}_healthy {1 if metrics.healthy else 0}")

            lines.append(f"# HELP {prefix}_requests_total Total requests")
            lines.append(f"# TYPE {prefix}_requests_total counter")
            lines.append(f"{prefix}_requests_total {metrics.total_requests}")

            lines.append(f"# HELP {prefix}_requests_successful Successful requests")
            lines.append(f"# TYPE {prefix}_requests_successful counter")
            lines.append(f"{prefix}_requests_successful {metrics.successful_requests}")

            lines.append(f"# HELP {prefix}_requests_failed Failed requests")
            lines.append(f"# TYPE {prefix}_requests_failed counter")
            lines.append(f"{prefix}_requests_failed {metrics.failed_requests}")

            lines.append(f"# HELP {prefix}_latency_avg_ms Average latency in ms")
            lines.append(f"# TYPE {prefix}_latency_avg_ms gauge")
            lines.append(f"{prefix}_latency_avg_ms {metrics.avg_latency_ms:.2f}")

            lines.append(f"# HELP {prefix}_latency_p99_ms P99 latency in ms")
            lines.append(f"# TYPE {prefix}_latency_p99_ms gauge")
            lines.append(f"{prefix}_latency_p99_ms {metrics.p99_latency_ms:.2f}")

            for metric_name, value in metrics.custom_metrics.items():
                if isinstance(value, (int, float)):
                    lines.append(f"# HELP {prefix}_{metric_name} Custom metric")
                    lines.append(f"# TYPE {prefix}_{metric_name} gauge")
                    lines.append(f"{prefix}_{metric_name} {value}")

        return "\n".join(lines)

    def get_json_metrics(self) -> dict[str, Any]:
        """Get metrics as JSON-compatible dictionary."""
        uptime = time.time() - self._start_time
        return {
            "service_name": self.service_name,
            "uptime_seconds": uptime,
            "services_count": len(self._metrics),
            "healthy_count": sum(1 for m in self._metrics.values() if m.healthy),
            "services": {k: v.to_dict() for k, v in self._metrics.items()},
        }

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get data formatted for dashboard display."""
        uptime = time.time() - self._start_time

        service_status = []
        for name, metrics in self._metrics.items():
            service_status.append(
                {
                    "name": name,
                    "status": "healthy" if metrics.healthy else "unhealthy",
                    "success_rate": (
                        metrics.successful_requests / metrics.total_requests * 100
                        if metrics.total_requests > 0
                        else 100.0
                    ),
                    "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                    "total_requests": metrics.total_requests,
                    "last_error": metrics.last_error,
                }
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_human": self._format_uptime(uptime),
            "uptime_seconds": uptime,
            "overall_status": (
                "healthy" if all(m.healthy for m in self._metrics.values()) else "degraded"
            ),
            "services": service_status,
        }

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format uptime seconds to human-readable string."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")

        return " ".join(parts)


def create_metrics_endpoint(monitor: ServicesMonitor):
    """Create a FastAPI endpoint factory for metrics.

    Usage:
        from fastapi import FastAPI, Response

        app = FastAPI()
        monitor = ServicesMonitor()

        @app.get("/metrics")
        async def metrics():
            return Response(
                content=monitor.get_prometheus_metrics(),
                media_type="text/plain"
            )

        @app.get("/health")
        async def health():
            report = await monitor.check_health()
            return report.to_dict()

        @app.get("/dashboard")
        async def dashboard():
            return monitor.get_dashboard_data()
    """
    from fastapi import Response

    async def metrics_endpoint() -> Response:
        return Response(
            content=monitor.get_prometheus_metrics(),
            media_type="text/plain",
        )

    async def health_endpoint() -> dict[str, Any]:
        report = await monitor.check_health()
        return report.to_dict()

    async def json_metrics_endpoint() -> dict[str, Any]:
        return monitor.get_json_metrics()

    async def dashboard_endpoint() -> dict[str, Any]:
        return monitor.get_dashboard_data()

    return {
        "metrics": metrics_endpoint,
        "health": health_endpoint,
        "json": json_metrics_endpoint,
        "dashboard": dashboard_endpoint,
    }
