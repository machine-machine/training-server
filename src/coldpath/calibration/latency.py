"""
Latency distribution fitting.

Models the detection -> plan -> sign -> submit latency chain.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class LatencyBreakdown:
    """Breakdown of latency components."""

    detection_ms: int
    enrichment_ms: int
    validation_ms: int
    routing_ms: int
    signing_ms: int
    submission_ms: int

    @property
    def total_ms(self) -> int:
        return (
            self.detection_ms
            + self.enrichment_ms
            + self.validation_ms
            + self.routing_ms
            + self.signing_ms
            + self.submission_ms
        )


class LatencyModel:
    """Models latency distribution for paper trading simulation."""

    def __init__(self):
        # Default parameters (log-normal distribution)
        self.mu = 5.0  # ln(mean) ≈ 150ms
        self.sigma = 0.3

        # Component breakdowns (fractions of total)
        self.component_fractions = {
            "detection": 0.10,
            "enrichment": 0.25,
            "validation": 0.20,
            "routing": 0.15,
            "signing": 0.15,
            "submission": 0.15,
        }

    def fit(self, latencies: list[int]):
        """Fit log-normal distribution to observed latencies."""
        if len(latencies) < 10:
            return

        log_latencies = np.log(np.array(latencies))
        self.mu = np.mean(log_latencies)
        self.sigma = np.std(log_latencies)

    def sample(self) -> int:
        """Sample a latency from the fitted distribution."""
        latency = np.random.lognormal(self.mu, self.sigma)
        return max(10, int(latency))  # Minimum 10ms

    def sample_breakdown(self) -> LatencyBreakdown:
        """Sample a full latency breakdown."""
        total = self.sample()

        return LatencyBreakdown(
            detection_ms=int(total * self.component_fractions["detection"]),
            enrichment_ms=int(total * self.component_fractions["enrichment"]),
            validation_ms=int(total * self.component_fractions["validation"]),
            routing_ms=int(total * self.component_fractions["routing"]),
            signing_ms=int(total * self.component_fractions["signing"]),
            submission_ms=int(total * self.component_fractions["submission"]),
        )

    def percentile(self, p: float) -> int:
        """Get the p-th percentile latency."""
        return int(np.exp(self.mu + self.sigma * stats.norm.ppf(p)))

    @property
    def mean(self) -> float:
        """Mean latency."""
        return np.exp(self.mu + self.sigma**2 / 2)

    @property
    def p50(self) -> int:
        """Median latency."""
        return self.percentile(0.5)

    @property
    def p95(self) -> int:
        """95th percentile latency."""
        return self.percentile(0.95)

    @property
    def p99(self) -> int:
        """99th percentile latency."""
        return self.percentile(0.99)

    def to_params(self) -> dict:
        """Export model parameters."""
        return {
            "latency_mu": self.mu,
            "latency_sigma": self.sigma,
            "latency_mean_ms": self.mean,
            "latency_p50_ms": self.p50,
            "latency_p95_ms": self.p95,
            "latency_p99_ms": self.p99,
        }
