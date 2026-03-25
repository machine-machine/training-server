"""
Cost Tracker - Track and analyze Claude API costs.

Records all Claude API calls with per-optimization cost tracking,
monthly aggregation, model usage statistics, cache hit rate monitoring,
and cost optimization recommendations.

Target: <$40/month Claude API costs.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# Pricing per 1M tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6-20250514": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
}


@dataclass
class APICallCost:
    """Individual API call cost record."""

    timestamp: datetime
    optimization_id: str
    model: str
    task_type: str
    tokens_input: int
    tokens_output: int
    cost_usd: float
    latency_ms: int
    cache_hit: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "optimization_id": self.optimization_id,
            "model": self.model,
            "task_type": self.task_type,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
        }


class CostTracker:
    """
    Track and analyze Claude API costs.

    Features:
    - Per-optimization cost tracking
    - Monthly cost aggregation
    - Model usage statistics
    - Cache hit rate monitoring
    - Cost optimization recommendations

    Usage:
        tracker = CostTracker(monthly_budget_usd=40.0)
        cost = await tracker.record_api_call(
            optimization_id="opt_123",
            model="claude-opus-4-6-20250514",
            task_type="strategy_analysis",
            tokens_input=1500,
            tokens_output=3000,
            latency_ms=2500,
            cache_hit=False,
        )
    """

    def __init__(
        self,
        monthly_budget_usd: float = 40.0,
        max_history: int = 10000,
    ):
        """Initialize cost tracker.

        Args:
            monthly_budget_usd: Monthly budget limit in USD.
            max_history: Maximum number of call records to keep.
        """
        self.monthly_budget_usd = monthly_budget_usd
        self.max_history = max_history

        self._calls: list[APICallCost] = []
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    @staticmethod
    def compute_cost(
        model: str,
        tokens_input: int,
        tokens_output: int,
    ) -> float:
        """Compute cost for an API call.

        Args:
            model: Model ID string.
            tokens_input: Number of input tokens.
            tokens_output: Number of output tokens.

        Returns:
            Cost in USD.
        """
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            # Default to Sonnet pricing for unknown models
            pricing = {"input": 3.0, "output": 15.0}

        cost = (tokens_input / 1_000_000) * pricing["input"] + (
            tokens_output / 1_000_000
        ) * pricing["output"]
        return cost

    async def record_api_call(
        self,
        optimization_id: str,
        model: str,
        task_type: str,
        tokens_input: int,
        tokens_output: int,
        latency_ms: int,
        cache_hit: bool = False,
    ) -> APICallCost:
        """Record an API call and compute cost.

        Args:
            optimization_id: Optimization run identifier.
            model: Model ID used.
            task_type: Type of task performed.
            tokens_input: Number of input tokens consumed.
            tokens_output: Number of output tokens consumed.
            latency_ms: Request latency in milliseconds.
            cache_hit: Whether the request hit the cache.

        Returns:
            APICallCost record with computed cost.
        """
        cost_usd = self.compute_cost(model, tokens_input, tokens_output)

        record = APICallCost(
            timestamp=datetime.now(),
            optimization_id=optimization_id,
            model=model,
            task_type=task_type,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        )

        self._calls.append(record)
        self._total_cost += cost_usd
        self._total_input_tokens += tokens_input
        self._total_output_tokens += tokens_output

        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        # Trim history if needed
        if len(self._calls) > self.max_history:
            self._calls[: len(self._calls) - self.max_history]
            self._calls = self._calls[-self.max_history :]

        # Warn if approaching budget
        monthly_cost = self._get_current_month_cost()
        if monthly_cost > self.monthly_budget_usd * 0.8:
            logger.warning(
                "API costs approaching monthly budget: $%.2f / $%.2f (%.0f%%)",
                monthly_cost,
                self.monthly_budget_usd,
                monthly_cost / self.monthly_budget_usd * 100,
            )

        logger.debug(
            "API call recorded: model=%s, task=%s, cost=$%.4f, "
            "tokens=%d/%d, latency=%dms, cache=%s",
            model,
            task_type,
            cost_usd,
            tokens_input,
            tokens_output,
            latency_ms,
            cache_hit,
        )

        return record

    async def get_monthly_cost(self) -> dict[str, Any]:
        """Get monthly cost breakdown.

        Returns:
            Dictionary with monthly cost breakdown by model, task type,
            and day.
        """
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        monthly_calls = [c for c in self._calls if c.timestamp >= month_start]

        # Aggregate by model
        by_model: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"cost": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}
        )
        for c in monthly_calls:
            by_model[c.model]["cost"] += c.cost_usd
            by_model[c.model]["calls"] += 1
            by_model[c.model]["input_tokens"] += c.tokens_input
            by_model[c.model]["output_tokens"] += c.tokens_output

        # Aggregate by task type
        by_task: dict[str, dict[str, Any]] = defaultdict(lambda: {"cost": 0.0, "calls": 0})
        for c in monthly_calls:
            by_task[c.task_type]["cost"] += c.cost_usd
            by_task[c.task_type]["calls"] += 1

        # Aggregate by day
        by_day: dict[str, float] = defaultdict(float)
        for c in monthly_calls:
            day_key = c.timestamp.strftime("%Y-%m-%d")
            by_day[day_key] += c.cost_usd

        total_monthly_cost = sum(c.cost_usd for c in monthly_calls)

        return {
            "month": now.strftime("%Y-%m"),
            "total_cost_usd": total_monthly_cost,
            "budget_usd": self.monthly_budget_usd,
            "budget_remaining_usd": max(0, self.monthly_budget_usd - total_monthly_cost),
            "budget_utilization_pct": (
                total_monthly_cost / self.monthly_budget_usd * 100
                if self.monthly_budget_usd > 0
                else 0
            ),
            "total_calls": len(monthly_calls),
            "by_model": dict(by_model),
            "by_task_type": dict(by_task),
            "by_day": dict(by_day),
        }

    async def get_optimization_cost(
        self,
        optimization_id: str,
    ) -> float:
        """Get total cost for a specific optimization.

        Args:
            optimization_id: The optimization run ID to query.

        Returns:
            Total cost in USD for the optimization.
        """
        return sum(c.cost_usd for c in self._calls if c.optimization_id == optimization_id)

    async def get_optimization_breakdown(
        self,
        optimization_id: str,
    ) -> dict[str, Any]:
        """Get detailed cost breakdown for an optimization.

        Args:
            optimization_id: The optimization run ID.

        Returns:
            Detailed breakdown by task type and model.
        """
        opt_calls = [c for c in self._calls if c.optimization_id == optimization_id]

        if not opt_calls:
            return {
                "optimization_id": optimization_id,
                "total_cost_usd": 0.0,
                "total_calls": 0,
            }

        return {
            "optimization_id": optimization_id,
            "total_cost_usd": sum(c.cost_usd for c in opt_calls),
            "total_calls": len(opt_calls),
            "total_input_tokens": sum(c.tokens_input for c in opt_calls),
            "total_output_tokens": sum(c.tokens_output for c in opt_calls),
            "avg_latency_ms": sum(c.latency_ms for c in opt_calls) / len(opt_calls),
            "cache_hit_rate": sum(1 for c in opt_calls if c.cache_hit) / len(opt_calls),
            "calls": [c.to_dict() for c in opt_calls],
        }

    async def get_cost_recommendations(self) -> list[str]:
        """
        Analyze usage patterns and recommend cost optimizations.

        Returns:
            List of actionable recommendation strings.
        """
        recommendations: list[str] = []

        if not self._calls:
            return ["No API calls recorded yet. No recommendations available."]

        total_calls = len(self._calls)

        # 1. Cache hit rate analysis
        total_cache_calls = self._cache_hits + self._cache_misses
        if total_cache_calls > 0:
            hit_rate = self._cache_hits / total_cache_calls * 100
            if hit_rate < 20:
                recommendations.append(
                    f"Cache hit rate is {hit_rate:.0f}%. "
                    "Consider increasing cache TTL or caching more prompt prefixes."
                )
            elif hit_rate < 40:
                recommendations.append(
                    f"Cache hit rate is {hit_rate:.0f}%. "
                    "Moderate cache effectiveness. Review cache key strategy."
                )

        # 2. Model mix analysis
        opus_calls = sum(1 for c in self._calls if "opus" in c.model.lower())
        opus_pct = opus_calls / total_calls * 100
        if opus_pct > 80:
            recommendations.append(
                f"{opus_pct:.0f}% of calls use Opus. "
                "Consider routing more tasks to Sonnet for cost savings."
            )
        elif opus_pct > 60:
            recommendations.append(
                f"{opus_pct:.0f}% of calls use Opus. "
                "Review task complexity thresholds - some tasks may be over-classified."
            )

        # 3. Token usage analysis
        if total_calls > 10:
            avg_output = self._total_output_tokens / total_calls
            if avg_output > 6000:
                recommendations.append(
                    f"Average output tokens is {avg_output:.0f}. "
                    "Consider reducing max_tokens to limit output length."
                )

            avg_input = self._total_input_tokens / total_calls
            if avg_input > 4000:
                recommendations.append(
                    f"Average input tokens is {avg_input:.0f}. "
                    "Consider using more concise prompts or summarizing context."
                )

        # 4. Monthly budget projection
        monthly_cost = self._get_current_month_cost()
        days_elapsed = datetime.now().day
        if days_elapsed > 0:
            projected_monthly = monthly_cost / days_elapsed * 30
            if projected_monthly > self.monthly_budget_usd:
                recommendations.append(
                    f"Projected monthly cost: ${projected_monthly:.2f} "
                    f"(budget: ${self.monthly_budget_usd:.2f}). "
                    "Reduce API calls or switch more tasks to Sonnet."
                )

        # 5. High-cost task types
        task_costs: dict[str, float] = defaultdict(float)
        for c in self._calls:
            task_costs[c.task_type] += c.cost_usd

        if task_costs:
            most_expensive = max(task_costs, key=task_costs.get)  # type: ignore
            if task_costs[most_expensive] > self._total_cost * 0.5:
                recommendations.append(
                    f"Task type '{most_expensive}' accounts for "
                    f"${task_costs[most_expensive]:.2f} "
                    f"({task_costs[most_expensive] / self._total_cost * 100:.0f}% of total). "
                    "Consider optimizing prompts for this task type."
                )

        if not recommendations:
            recommendations.append("Cost usage looks healthy. No optimizations needed.")

        return recommendations

    def get_stats(self) -> dict[str, Any]:
        """Get overall cost statistics.

        Returns:
            Dictionary with aggregate cost statistics.
        """
        total_calls = len(self._calls)
        total_cache = self._cache_hits + self._cache_misses

        return {
            "total_calls": total_calls,
            "total_cost_usd": self._total_cost,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "avg_cost_per_call": self._total_cost / max(1, total_calls),
            "avg_input_tokens": self._total_input_tokens / max(1, total_calls),
            "avg_output_tokens": self._total_output_tokens / max(1, total_calls),
            "cache_hit_rate": (self._cache_hits / total_cache if total_cache > 0 else 0.0),
            "monthly_budget_usd": self.monthly_budget_usd,
            "current_month_cost_usd": self._get_current_month_cost(),
        }

    def _get_current_month_cost(self) -> float:
        """Get cost for the current month."""
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return sum(c.cost_usd for c in self._calls if c.timestamp >= month_start)

    def is_within_budget(self) -> bool:
        """Check if current spending is within monthly budget.

        Returns:
            True if within budget.
        """
        return self._get_current_month_cost() < self.monthly_budget_usd

    def reset(self) -> None:
        """Reset all tracked data."""
        self._calls.clear()
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._cache_hits = 0
        self._cache_misses = 0
