"""
Task Complexity Router - Dynamic Opus/Sonnet model selection.

Routes optimization tasks to the appropriate Claude model based on
complexity scoring. Simple tasks go to Sonnet 4.5 for speed and cost,
complex tasks go to Opus 4.6 for deep reasoning.

Complexity Tiers:
  0-30:  Trivial    -> Sonnet 4.5 (fast, cheap)
 30-60:  Moderate   -> Sonnet 4.5 (standard)
 60-80:  Complex    -> Opus 4.6 (deep reasoning)
 80-100: Critical   -> Opus 4.6 (extended thinking)

Factors:
- Task type (10-40 points base complexity)
- Parameter count (0-20 points)
- Has constraints (0-10 points)
- Requires explanation (0-10 points)
- Is high-stakes/safety-critical (0-20 points)
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TaskComplexity:
    """Task complexity scoring result."""

    score: int  # 0-100
    reasoning: str
    factors: dict[str, int]
    tier: str = ""  # "trivial", "moderate", "complex", "critical"

    def __post_init__(self):
        if not self.tier:
            if self.score < 30:
                self.tier = "trivial"
            elif self.score < 60:
                self.tier = "moderate"
            elif self.score < 80:
                self.tier = "complex"
            else:
                self.tier = "critical"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "score": self.score,
            "tier": self.tier,
            "reasoning": self.reasoning,
            "factors": self.factors,
        }


@dataclass
class ModelRoutingDecision:
    """Model routing result."""

    model_id: str
    max_tokens: int
    temperature: float
    timeout_seconds: int
    estimated_cost: float
    reasoning: str
    complexity: TaskComplexity | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "estimated_cost": self.estimated_cost,
            "reasoning": self.reasoning,
            "complexity": self.complexity.to_dict() if self.complexity else None,
        }


class TaskComplexityRouter:
    """
    Route optimization tasks to appropriate Claude model.

    Uses complexity scoring to determine whether a task should be
    handled by the faster, cheaper Sonnet 4.5 or the more capable
    Opus 4.6 model.

    Usage:
        router = TaskComplexityRouter()
        decision = router.route_task(
            task_type="strategy_analysis",
            num_parameters=42,
            has_constraints=True,
            is_high_stakes=True,
        )
        # decision.model_id -> "claude-opus-4-6-20250514"
    """

    # Model configurations
    OPUS_4_6 = "claude-opus-4-6-20250514"
    SONNET_4_5 = "claude-sonnet-4-5-20250929"

    # Pricing (per 1M tokens, USD)
    PRICING = {
        OPUS_4_6: {"input": 15.0, "output": 75.0},
        SONNET_4_5: {"input": 3.0, "output": 15.0},
    }

    # Task type base complexity scores
    TASK_COMPLEXITY: dict[str, int] = {
        "quick_summary": 10,
        "parameter_explanation": 25,
        "regime_notification": 30,
        "performance_summary": 35,
        "backtest_interpretation": 55,
        "strategy_analysis": 70,
        "risk_assessment": 85,
        "rollback_decision": 90,
        "multi_objective_optimization": 95,
    }

    # Complexity thresholds
    OPUS_THRESHOLD = 60
    CRITICAL_THRESHOLD = 80

    def __init__(
        self,
        opus_model: str | None = None,
        sonnet_model: str | None = None,
        opus_threshold: int = 60,
        critical_threshold: int = 80,
    ):
        """Initialize the router.

        Args:
            opus_model: Override Opus model ID.
            sonnet_model: Override Sonnet model ID.
            opus_threshold: Minimum score to route to Opus (default 60).
            critical_threshold: Score above which extended thinking is used.
        """
        if opus_model:
            self.OPUS_4_6 = opus_model
        if sonnet_model:
            self.SONNET_4_5 = sonnet_model
        self.OPUS_THRESHOLD = opus_threshold
        self.CRITICAL_THRESHOLD = critical_threshold

        # Track routing decisions for analysis
        self._routing_history: list[dict[str, Any]] = []
        self._max_history = 200

    def score_complexity(
        self,
        task_type: str,
        num_parameters: int = 0,
        has_constraints: bool = False,
        requires_explanation: bool = False,
        is_high_stakes: bool = False,
    ) -> TaskComplexity:
        """Calculate task complexity score.

        Args:
            task_type: Type of task (see TASK_COMPLEXITY).
            num_parameters: Number of parameters being optimized.
            has_constraints: Whether hard constraints are present.
            requires_explanation: Whether explanation is needed.
            is_high_stakes: Whether this is a safety-critical decision.

        Returns:
            TaskComplexity with score and breakdown.
        """
        factors: dict[str, int] = {}

        # Base complexity from task type
        base = self.TASK_COMPLEXITY.get(task_type, 50)
        factors["task_type"] = base

        # Parameter complexity (more params = more complex)
        param_score = min(num_parameters * 2, 20)
        factors["parameters"] = param_score

        # Constraints add complexity
        constraint_score = 10 if has_constraints else 0
        factors["constraints"] = constraint_score

        # Explanation requirement
        explanation_score = 10 if requires_explanation else 0
        factors["explanation"] = explanation_score

        # High-stakes decisions need careful reasoning
        stakes_score = 20 if is_high_stakes else 0
        factors["high_stakes"] = stakes_score

        # Total score (clamped to 0-100)
        total = sum(factors.values())
        total = max(0, min(total, 100))

        # Build reasoning string
        reasoning_parts = [f"Task type: {task_type} (base {base})"]
        if num_parameters > 0:
            reasoning_parts.append(f"{num_parameters} parameters (+{param_score})")
        if has_constraints:
            reasoning_parts.append("has constraints (+10)")
        if requires_explanation:
            reasoning_parts.append("requires explanation (+10)")
        if is_high_stakes:
            reasoning_parts.append("high-stakes (+20)")

        reasoning = ", ".join(reasoning_parts)

        return TaskComplexity(
            score=total,
            reasoning=reasoning,
            factors=factors,
        )

    def route_task(
        self,
        task_type: str,
        num_parameters: int = 0,
        has_constraints: bool = False,
        requires_explanation: bool = False,
        is_high_stakes: bool = False,
    ) -> ModelRoutingDecision:
        """
        Route task to appropriate model based on complexity.

        Args:
            task_type: Type of task to route.
            num_parameters: Number of parameters being optimized.
            has_constraints: Whether hard constraints are present.
            requires_explanation: Whether explanation is needed.
            is_high_stakes: Whether this is a safety-critical decision.

        Returns:
            ModelRoutingDecision with model selection and configuration.
        """
        complexity = self.score_complexity(
            task_type=task_type,
            num_parameters=num_parameters,
            has_constraints=has_constraints,
            requires_explanation=requires_explanation,
            is_high_stakes=is_high_stakes,
        )

        # Decision thresholds
        if complexity.score >= self.OPUS_THRESHOLD:
            # Complex/Critical -> Opus 4.6
            model = self.OPUS_4_6

            if complexity.score >= self.CRITICAL_THRESHOLD:
                # Critical tasks get extended thinking
                max_tokens = 8000
                temperature = 0.2  # Very precise
                timeout = 30
            else:
                # Complex tasks
                max_tokens = 4000
                temperature = 0.3  # Precise
                timeout = 20
        else:
            # Trivial/Moderate -> Sonnet 4.5
            model = self.SONNET_4_5

            if complexity.score >= 30:
                # Moderate complexity
                max_tokens = 2000
                temperature = 0.5  # Balanced
                timeout = 8
            else:
                # Trivial tasks
                max_tokens = 512
                temperature = 0.7  # Creative OK
                timeout = 5

        # Estimate cost (input + output tokens)
        avg_input = max_tokens * 0.5  # Assume 50% of max for input
        avg_output = max_tokens

        pricing = self.PRICING[model]
        estimated_cost = (avg_input / 1_000_000) * pricing["input"] + (
            avg_output / 1_000_000
        ) * pricing["output"]

        # Determine model name for logging
        model_name = "Opus" if model == self.OPUS_4_6 else "Sonnet"

        routing_reasoning = (
            f"Complexity: {complexity.score}/100 ({complexity.tier}). "
            f"Using {model_name} "
            f"({max_tokens} tokens, {temperature} temp, ${estimated_cost:.4f} est.)"
        )

        decision = ModelRoutingDecision(
            model_id=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout,
            estimated_cost=estimated_cost,
            reasoning=routing_reasoning,
            complexity=complexity,
        )

        # Record in history
        self._record_decision(task_type, decision)

        logger.info(
            "Routed task '%s' -> %s (complexity: %d/100, tier: %s)",
            task_type,
            model_name,
            complexity.score,
            complexity.tier,
        )

        return decision

    def _record_decision(
        self,
        task_type: str,
        decision: ModelRoutingDecision,
    ) -> None:
        """Record routing decision for analysis."""
        self._routing_history.append(
            {
                "task_type": task_type,
                "model": decision.model_id,
                "complexity_score": decision.complexity.score if decision.complexity else 0,
                "estimated_cost": decision.estimated_cost,
            }
        )
        if len(self._routing_history) > self._max_history:
            self._routing_history = self._routing_history[-self._max_history :]

    def get_routing_stats(self) -> dict[str, Any]:
        """Get routing statistics for monitoring.

        Returns:
            Dictionary with routing statistics including model usage
            breakdown, average complexity, and estimated costs.
        """
        if not self._routing_history:
            return {
                "total_decisions": 0,
                "opus_pct": 0.0,
                "sonnet_pct": 0.0,
                "avg_complexity": 0.0,
                "total_estimated_cost": 0.0,
            }

        total = len(self._routing_history)
        opus_count = sum(1 for d in self._routing_history if "opus" in d["model"].lower())
        sonnet_count = total - opus_count

        avg_complexity = sum(d["complexity_score"] for d in self._routing_history) / total
        total_cost = sum(d["estimated_cost"] for d in self._routing_history)

        # Task type breakdown
        task_counts: dict[str, int] = {}
        for d in self._routing_history:
            tt = d["task_type"]
            task_counts[tt] = task_counts.get(tt, 0) + 1

        return {
            "total_decisions": total,
            "opus_count": opus_count,
            "sonnet_count": sonnet_count,
            "opus_pct": opus_count / total * 100,
            "sonnet_pct": sonnet_count / total * 100,
            "avg_complexity": avg_complexity,
            "total_estimated_cost": total_cost,
            "task_type_breakdown": task_counts,
        }

    def get_model_for_tier(self, tier: str) -> str:
        """Get model ID for a given complexity tier.

        Args:
            tier: One of "trivial", "moderate", "complex", "critical".

        Returns:
            Model ID string.
        """
        if tier in ("complex", "critical"):
            return self.OPUS_4_6
        return self.SONNET_4_5
