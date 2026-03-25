"""
Server Self-Tuning Engine - Dynamic configuration adjustment.

Implements adaptive server configuration that adjusts based on:
- Performance metrics (latency, error rates, throughput)
- Resource utilization (cache efficiency, token usage)
- Model behavior (agreement rates, routing accuracy)
- Cost optimization (Opus vs Sonnet balance)

Features:
- Real-time metrics collection
- Automatic threshold adjustment
- Gradual parameter tuning with safety bounds
- Configuration hot-reload without restart
- Rollback capability for failed adjustments
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TuningMode(Enum):
    """Server tuning modes."""

    DISABLED = "disabled"  # No auto-tuning
    CONSERVATIVE = "conservative"  # Small, infrequent adjustments
    MODERATE = "moderate"  # Balanced tuning
    AGGRESSIVE = "aggressive"  # Fast adaptation, larger changes


class MetricType(Enum):
    """Types of metrics tracked."""

    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CACHE_HIT_RATE = "cache_hit_rate"
    TOKEN_SAVINGS = "token_savings"
    OPUS_USAGE = "opus_usage"
    SONNET_USAGE = "sonnet_usage"
    AGREEMENT_RATE = "agreement_rate"
    FALLBACK_RATE = "fallback_rate"
    COST_PER_REQUEST = "cost_per_request"


@dataclass
class MetricWindow:
    """Sliding window of metric values."""

    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add(self, value: float, timestamp: float | None = None):
        self.values.append(value)
        self.timestamps.append(timestamp or time.time())

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return mean(self.values) if self.values else 0.0

    @property
    def std(self) -> float:
        return stdev(self.values) if len(self.values) > 1 else 0.0

    def percentile(self, p: float) -> float:
        if not self.values:
            return 0.0
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def recent(self, window_seconds: float = 60) -> list[float]:
        """Get values from recent time window."""
        now = time.time()
        return [
            v
            for v, t in zip(self.values, self.timestamps, strict=False)
            if now - t <= window_seconds
        ]


@dataclass
class TunableParameter:
    """A parameter that can be auto-tuned."""

    name: str
    current_value: float
    min_value: float
    max_value: float
    default_value: float
    step_size: float
    cooldown_seconds: float = 60.0
    last_adjusted: float = 0.0
    adjustment_history: list[tuple[float, float, str]] = field(default_factory=list)

    def can_adjust(self) -> bool:
        return time.time() - self.last_adjusted >= self.cooldown_seconds

    def adjust(self, delta: float, reason: str) -> bool:
        """Adjust parameter by delta. Returns True if adjustment was made."""
        if not self.can_adjust():
            return False

        new_value = max(self.min_value, min(self.max_value, self.current_value + delta))
        if new_value == self.current_value:
            return False

        old_value = self.current_value
        self.current_value = new_value
        self.last_adjusted = time.time()
        self.adjustment_history.append((old_value, new_value, reason))

        logger.info(f"Tuned {self.name}: {old_value:.4f} -> {new_value:.4f} ({reason})")
        return True

    def rollback(self) -> bool:
        """Rollback to previous value."""
        if not self.adjustment_history:
            return False

        old_value, _, _ = self.adjustment_history.pop()
        self.current_value = old_value
        self.last_adjusted = time.time()
        return True

    def reset(self):
        """Reset to default value."""
        self.current_value = self.default_value
        self.last_adjusted = time.time()


@dataclass
class ServerConfig:
    """Complete server configuration with all tunable parameters."""

    # Model routing thresholds
    opus_complexity_threshold: TunableParameter = field(
        default_factory=lambda: TunableParameter(
            name="opus_complexity_threshold",
            current_value=0.7,
            min_value=0.3,
            max_value=0.95,
            default_value=0.7,
            step_size=0.05,
        )
    )

    # Latency budgets (ms)
    scoring_budget_ms: TunableParameter = field(
        default_factory=lambda: TunableParameter(
            name="scoring_budget_ms",
            current_value=100.0,
            min_value=50.0,
            max_value=500.0,
            default_value=100.0,
            step_size=25.0,
        )
    )

    chat_budget_ms: TunableParameter = field(
        default_factory=lambda: TunableParameter(
            name="chat_budget_ms",
            current_value=500.0,
            min_value=200.0,
            max_value=2000.0,
            default_value=500.0,
            step_size=100.0,
        )
    )

    analysis_budget_ms: TunableParameter = field(
        default_factory=lambda: TunableParameter(
            name="analysis_budget_ms",
            current_value=2000.0,
            min_value=500.0,
            max_value=10000.0,
            default_value=2000.0,
            step_size=500.0,
        )
    )

    # Aggregation weights
    ml_weight: TunableParameter = field(
        default_factory=lambda: TunableParameter(
            name="ml_weight",
            current_value=0.6,
            min_value=0.3,
            max_value=0.9,
            default_value=0.6,
            step_size=0.05,
        )
    )

    llm_weight: TunableParameter = field(
        default_factory=lambda: TunableParameter(
            name="llm_weight",
            current_value=0.4,
            min_value=0.1,
            max_value=0.7,
            default_value=0.4,
            step_size=0.05,
        )
    )

    # Consensus threshold
    consensus_threshold: TunableParameter = field(
        default_factory=lambda: TunableParameter(
            name="consensus_threshold",
            current_value=0.7,
            min_value=0.5,
            max_value=0.95,
            default_value=0.7,
            step_size=0.05,
        )
    )

    # Confidence thresholds
    min_confidence_for_trade: TunableParameter = field(
        default_factory=lambda: TunableParameter(
            name="min_confidence_for_trade",
            current_value=0.6,
            min_value=0.4,
            max_value=0.85,
            default_value=0.6,
            step_size=0.05,
        )
    )

    # Cache settings
    cache_ttl_multiplier: TunableParameter = field(
        default_factory=lambda: TunableParameter(
            name="cache_ttl_multiplier",
            current_value=1.0,
            min_value=0.5,
            max_value=3.0,
            default_value=1.0,
            step_size=0.25,
        )
    )

    # Temperature settings
    opus_temperature: TunableParameter = field(
        default_factory=lambda: TunableParameter(
            name="opus_temperature",
            current_value=0.5,
            min_value=0.1,
            max_value=0.9,
            default_value=0.5,
            step_size=0.1,
        )
    )

    sonnet_temperature: TunableParameter = field(
        default_factory=lambda: TunableParameter(
            name="sonnet_temperature",
            current_value=0.7,
            min_value=0.3,
            max_value=1.0,
            default_value=0.7,
            step_size=0.1,
        )
    )

    # Retry settings
    max_retries: TunableParameter = field(
        default_factory=lambda: TunableParameter(
            name="max_retries",
            current_value=3.0,
            min_value=1.0,
            max_value=5.0,
            default_value=3.0,
            step_size=1.0,
        )
    )

    # Request timeout (seconds)
    request_timeout: TunableParameter = field(
        default_factory=lambda: TunableParameter(
            name="request_timeout",
            current_value=60.0,
            min_value=30.0,
            max_value=120.0,
            default_value=60.0,
            step_size=10.0,
        )
    )

    def get_all_parameters(self) -> dict[str, TunableParameter]:
        """Get all tunable parameters."""
        return {
            name: getattr(self, name)
            for name in dir(self)
            if isinstance(getattr(self, name), TunableParameter)
        }

    def to_dict(self) -> dict[str, float]:
        """Export current values as dict."""
        return {name: param.current_value for name, param in self.get_all_parameters().items()}

    def from_dict(self, values: dict[str, float]):
        """Import values from dict."""
        for name, value in values.items():
            param = getattr(self, name, None)
            if isinstance(param, TunableParameter):
                param.current_value = max(param.min_value, min(param.max_value, value))


@dataclass
class TuningRule:
    """A rule for automatic parameter adjustment."""

    name: str
    metric_type: MetricType
    parameter_name: str
    threshold_low: float
    threshold_high: float
    adjustment_low: float  # Adjustment when metric < threshold_low
    adjustment_high: float  # Adjustment when metric > threshold_high
    priority: int = 0
    enabled: bool = True

    def evaluate(
        self,
        metric_value: float,
        param: TunableParameter,
    ) -> tuple[float, str] | None:
        """Evaluate rule and return (adjustment, reason) or None."""
        if not self.enabled or not param.can_adjust():
            return None

        if metric_value < self.threshold_low:
            return (
                self.adjustment_low,
                f"{self.metric_type.value} below threshold "
                f"({metric_value:.3f} < {self.threshold_low})",
            )
        elif metric_value > self.threshold_high:
            return (
                self.adjustment_high,
                f"{self.metric_type.value} above threshold "
                f"({metric_value:.3f} > {self.threshold_high})",
            )

        return None


class ServerTuner:
    """Self-tuning engine for server configuration.

    Monitors performance metrics and automatically adjusts server
    parameters to optimize for:
    - Low latency
    - High cache efficiency
    - Cost optimization
    - Error reduction

    Usage:
        tuner = ServerTuner()
        await tuner.start()

        # Record metrics as requests come in
        tuner.record_metric(MetricType.LATENCY_P50, 150.0)
        tuner.record_metric(MetricType.ERROR_RATE, 0.02)

        # Get current optimized config
        config = tuner.get_config()
    """

    def __init__(
        self,
        mode: TuningMode = TuningMode.MODERATE,
        config: ServerConfig | None = None,
        config_path: Path | None = None,
    ):
        """Initialize the server tuner.

        Args:
            mode: Tuning aggressiveness
            config: Initial configuration (creates default if None)
            config_path: Path to persist configuration
        """
        self.mode = mode
        self.config = config or ServerConfig()
        self.config_path = config_path

        # Metrics tracking
        self._metrics: dict[MetricType, MetricWindow] = {mt: MetricWindow() for mt in MetricType}

        # Tuning rules
        self._rules: list[TuningRule] = self._create_default_rules()

        # State
        self._running = False
        self._task: asyncio.Task | None = None
        self._tune_interval = self._get_tune_interval()
        self._last_tune = 0.0

        # Callbacks for config changes
        self._on_config_change: list[Callable[[ServerConfig], None]] = []

        # Statistics
        self._stats = {
            "total_adjustments": 0,
            "rollbacks": 0,
            "rules_triggered": 0,
            "tune_cycles": 0,
        }

        logger.info(f"ServerTuner initialized in {mode.value} mode")

    def _get_tune_interval(self) -> float:
        """Get tuning interval based on mode."""
        intervals = {
            TuningMode.DISABLED: float("inf"),
            TuningMode.CONSERVATIVE: 300.0,  # 5 minutes
            TuningMode.MODERATE: 60.0,  # 1 minute
            TuningMode.AGGRESSIVE: 15.0,  # 15 seconds
        }
        return intervals.get(self.mode, 60.0)

    def _create_default_rules(self) -> list[TuningRule]:
        """Create default tuning rules."""
        return [
            # Latency-based rules
            TuningRule(
                name="high_latency_increase_budget",
                metric_type=MetricType.LATENCY_P95,
                parameter_name="chat_budget_ms",
                threshold_low=0,
                threshold_high=800,  # P95 > 800ms is too slow
                adjustment_low=0,
                adjustment_high=100,  # Increase budget
                priority=1,
            ),
            TuningRule(
                name="low_latency_decrease_budget",
                metric_type=MetricType.LATENCY_P95,
                parameter_name="chat_budget_ms",
                threshold_low=200,  # P95 < 200ms means we have headroom
                threshold_high=float("inf"),
                adjustment_low=-50,  # Decrease budget to save resources
                adjustment_high=0,
                priority=2,
            ),
            # Error rate rules
            TuningRule(
                name="high_error_rate_increase_timeout",
                metric_type=MetricType.ERROR_RATE,
                parameter_name="request_timeout",
                threshold_low=0,
                threshold_high=0.05,  # > 5% error rate
                adjustment_low=0,
                adjustment_high=10,  # Increase timeout
                priority=0,
            ),
            TuningRule(
                name="high_error_rate_increase_retries",
                metric_type=MetricType.ERROR_RATE,
                parameter_name="max_retries",
                threshold_low=0,
                threshold_high=0.08,  # > 8% error rate
                adjustment_low=0,
                adjustment_high=1,
                priority=0,
            ),
            # Cache efficiency rules
            TuningRule(
                name="low_cache_hit_extend_ttl",
                metric_type=MetricType.CACHE_HIT_RATE,
                parameter_name="cache_ttl_multiplier",
                threshold_low=0.3,  # < 30% hit rate
                threshold_high=float("inf"),
                adjustment_low=0.25,  # Extend TTLs
                adjustment_high=0,
                priority=3,
            ),
            TuningRule(
                name="high_cache_hit_reduce_ttl",
                metric_type=MetricType.CACHE_HIT_RATE,
                parameter_name="cache_ttl_multiplier",
                threshold_low=0,
                threshold_high=0.7,  # > 70% hit rate means TTLs may be too long
                adjustment_low=0,
                adjustment_high=-0.1,
                priority=4,
            ),
            # Model balance rules
            TuningRule(
                name="high_opus_usage_raise_threshold",
                metric_type=MetricType.OPUS_USAGE,
                parameter_name="opus_complexity_threshold",
                threshold_low=0,
                threshold_high=0.4,  # > 40% Opus usage is expensive
                adjustment_low=0,
                adjustment_high=0.05,  # Raise threshold to reduce Opus usage
                priority=5,
            ),
            TuningRule(
                name="low_agreement_reduce_consensus",
                metric_type=MetricType.AGREEMENT_RATE,
                parameter_name="consensus_threshold",
                threshold_low=0.5,  # < 50% agreement
                threshold_high=float("inf"),
                adjustment_low=-0.05,  # Lower consensus requirement
                adjustment_high=0,
                priority=6,
            ),
            # Fallback rate rules
            TuningRule(
                name="high_fallback_increase_budget",
                metric_type=MetricType.FALLBACK_RATE,
                parameter_name="analysis_budget_ms",
                threshold_low=0,
                threshold_high=0.15,  # > 15% fallback rate
                adjustment_low=0,
                adjustment_high=250,  # Increase analysis budget
                priority=7,
            ),
        ]

    async def start(self):
        """Start the tuning loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._tune_loop())
        logger.info("Server tuner started")

    async def stop(self):
        """Stop the tuning loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Save config on shutdown
        await self.save_config()
        logger.info("Server tuner stopped")

    async def _tune_loop(self):
        """Main tuning loop."""
        while self._running:
            try:
                await asyncio.sleep(self._tune_interval)

                if self.mode == TuningMode.DISABLED:
                    continue

                await self._run_tuning_cycle()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Tuning error: {e}")

    async def _run_tuning_cycle(self):
        """Run one tuning cycle."""
        self._stats["tune_cycles"] += 1

        # Collect current metric values
        metrics = self._collect_metrics()

        # Evaluate rules
        adjustments: list[tuple[TuningRule, float, str]] = []

        for rule in sorted(self._rules, key=lambda r: r.priority):
            if not rule.enabled:
                continue

            metric_value = metrics.get(rule.metric_type, 0.0)
            param = getattr(self.config, rule.parameter_name, None)

            if not isinstance(param, TunableParameter):
                continue

            result = rule.evaluate(metric_value, param)
            if result:
                adjustment, reason = result
                adjustments.append((rule, adjustment, reason))

        # Apply adjustments (respect priority order)
        config_changed = False
        for rule, adjustment, reason in adjustments:
            param = getattr(self.config, rule.parameter_name)
            if param.adjust(adjustment * param.step_size, reason):
                self._stats["total_adjustments"] += 1
                self._stats["rules_triggered"] += 1
                config_changed = True

        # Notify listeners
        if config_changed:
            for callback in self._on_config_change:
                try:
                    callback(self.config)
                except Exception as e:
                    logger.error(f"Config change callback error: {e}")

        self._last_tune = time.time()

    def _collect_metrics(self) -> dict[MetricType, float]:
        """Collect current metric values."""
        metrics = {}

        for metric_type, window in self._metrics.items():
            recent = window.recent(window_seconds=self._tune_interval * 2)
            if recent:
                if metric_type in (
                    MetricType.LATENCY_P50,
                    MetricType.LATENCY_P95,
                    MetricType.LATENCY_P99,
                ):
                    # Use percentiles for latency
                    percentile = {
                        MetricType.LATENCY_P50: 50,
                        MetricType.LATENCY_P95: 95,
                        MetricType.LATENCY_P99: 99,
                    }[metric_type]
                    sorted_recent = sorted(recent)
                    idx = int(len(sorted_recent) * percentile / 100)
                    metrics[metric_type] = sorted_recent[min(idx, len(sorted_recent) - 1)]
                else:
                    # Use mean for other metrics
                    metrics[metric_type] = mean(recent)

        return metrics

    def record_metric(self, metric_type: MetricType, value: float):
        """Record a metric value.

        Args:
            metric_type: Type of metric
            value: Metric value
        """
        self._metrics[metric_type].add(value)

    def record_request(
        self,
        latency_ms: float,
        model_used: str,
        cache_hit: bool,
        error: bool = False,
        fallback: bool = False,
    ):
        """Record a complete request with all metrics.

        Args:
            latency_ms: Request latency in milliseconds
            model_used: Which model was used (opus/sonnet)
            cache_hit: Whether cache was hit
            error: Whether request errored
            fallback: Whether fallback was used
        """
        self.record_metric(MetricType.LATENCY_P50, latency_ms)
        self.record_metric(MetricType.LATENCY_P95, latency_ms)
        self.record_metric(MetricType.LATENCY_P99, latency_ms)

        self.record_metric(MetricType.ERROR_RATE, 1.0 if error else 0.0)
        self.record_metric(MetricType.CACHE_HIT_RATE, 1.0 if cache_hit else 0.0)
        self.record_metric(MetricType.FALLBACK_RATE, 1.0 if fallback else 0.0)

        if "opus" in model_used.lower():
            self.record_metric(MetricType.OPUS_USAGE, 1.0)
            self.record_metric(MetricType.SONNET_USAGE, 0.0)
        else:
            self.record_metric(MetricType.OPUS_USAGE, 0.0)
            self.record_metric(MetricType.SONNET_USAGE, 1.0)

    def get_config(self) -> ServerConfig:
        """Get current configuration."""
        return self.config

    def get_config_dict(self) -> dict[str, float]:
        """Get current configuration as dictionary."""
        return self.config.to_dict()

    def set_parameter(self, name: str, value: float, reason: str = "manual"):
        """Manually set a parameter value.

        Args:
            name: Parameter name
            value: New value
            reason: Reason for change
        """
        param = getattr(self.config, name, None)
        if not isinstance(param, TunableParameter):
            raise ValueError(f"Unknown parameter: {name}")

        old_value = param.current_value
        param.current_value = max(param.min_value, min(param.max_value, value))
        param.last_adjusted = time.time()
        param.adjustment_history.append((old_value, param.current_value, reason))

        logger.info(f"Manually set {name}: {old_value:.4f} -> {param.current_value:.4f}")

    def reset_parameter(self, name: str):
        """Reset a parameter to default value."""
        param = getattr(self.config, name, None)
        if isinstance(param, TunableParameter):
            param.reset()

    def reset_all(self):
        """Reset all parameters to defaults."""
        for param in self.config.get_all_parameters().values():
            param.reset()

    def rollback_parameter(self, name: str) -> bool:
        """Rollback a parameter to previous value."""
        param = getattr(self.config, name, None)
        if isinstance(param, TunableParameter):
            return param.rollback()
        return False

    def add_rule(self, rule: TuningRule):
        """Add a tuning rule."""
        self._rules.append(rule)

    def remove_rule(self, name: str):
        """Remove a tuning rule by name."""
        self._rules = [r for r in self._rules if r.name != name]

    def enable_rule(self, name: str, enabled: bool = True):
        """Enable or disable a rule."""
        for rule in self._rules:
            if rule.name == name:
                rule.enabled = enabled
                return

    def set_mode(self, mode: TuningMode):
        """Set tuning mode."""
        self.mode = mode
        self._tune_interval = self._get_tune_interval()
        logger.info(f"Tuning mode set to {mode.value}")

    def on_config_change(self, callback: Callable[[ServerConfig], None]):
        """Register callback for configuration changes."""
        self._on_config_change.append(callback)

    async def save_config(self, path: Path | None = None):
        """Save configuration to file."""
        path = path or self.config_path
        if not path:
            return

        config_dict = {
            "values": self.config.to_dict(),
            "mode": self.mode.value,
            "rules_enabled": {r.name: r.enabled for r in self._rules},
            "saved_at": time.time(),
        }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config_dict, indent=2))
            logger.info(f"Config saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    async def load_config(self, path: Path | None = None):
        """Load configuration from file."""
        path = path or self.config_path
        if not path or not path.exists():
            return

        try:
            config_dict = json.loads(path.read_text())

            # Load values
            self.config.from_dict(config_dict.get("values", {}))

            # Load mode
            mode_str = config_dict.get("mode", "moderate")
            self.mode = TuningMode(mode_str)
            self._tune_interval = self._get_tune_interval()

            # Load rule states
            rules_enabled = config_dict.get("rules_enabled", {})
            for rule in self._rules:
                if rule.name in rules_enabled:
                    rule.enabled = rules_enabled[rule.name]

            logger.info(f"Config loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get tuner statistics."""
        return {
            **self._stats,
            "mode": self.mode.value,
            "tune_interval_seconds": self._tune_interval,
            "last_tune": self._last_tune,
            "rules_count": len(self._rules),
            "rules_enabled": sum(1 for r in self._rules if r.enabled),
            "parameters_count": len(self.config.get_all_parameters()),
        }

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of tracked metrics."""
        summary = {}
        for metric_type, window in self._metrics.items():
            if window.count > 0:
                summary[metric_type.value] = {
                    "count": window.count,
                    "mean": window.mean,
                    "std": window.std,
                    "p50": window.percentile(50),
                    "p95": window.percentile(95),
                    "p99": window.percentile(99),
                }
        return summary

    def get_parameter_info(self, name: str) -> dict[str, Any]:
        """Get detailed information about a parameter."""
        param = getattr(self.config, name, None)
        if not isinstance(param, TunableParameter):
            return {}

        return {
            "name": param.name,
            "current_value": param.current_value,
            "default_value": param.default_value,
            "min_value": param.min_value,
            "max_value": param.max_value,
            "step_size": param.step_size,
            "cooldown_seconds": param.cooldown_seconds,
            "last_adjusted": param.last_adjusted,
            "can_adjust": param.can_adjust(),
            "adjustment_count": len(param.adjustment_history),
            "recent_adjustments": param.adjustment_history[-5:],
        }


# Singleton instance for global access
_server_tuner: ServerTuner | None = None


def get_server_tuner() -> ServerTuner:
    """Get or create global server tuner instance."""
    global _server_tuner
    if _server_tuner is None:
        _server_tuner = ServerTuner()
    return _server_tuner


def set_server_tuner(tuner: ServerTuner):
    """Set global server tuner instance."""
    global _server_tuner
    _server_tuner = tuner
