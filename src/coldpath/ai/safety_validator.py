"""
Multi-layer safety validation for strategy parameter deployment.

Validates proposed parameter changes against hard constraints, maximum
change limits, parameter sanity rules, risk/reward ratios, and
interaction conflicts before allowing deployment.

Safety layers:
1. Hard constraints    - Absolute limits (max position, max loss)
2. Change limits       - Maximum 50% change per parameter
3. Parameter sanity    - Logical consistency (stop loss < take profit)
4. Risk/reward ratio   - Minimum 2:1 reward-to-risk
5. Interaction checks  - No contradictory parameter combinations

Usage:
    validator = SafetyValidator()
    is_safe, violations = await validator.validate(proposed, current)
    if not is_safe:
        reject_deployment(violations)
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SafetyReport:
    """Comprehensive safety validation results."""

    is_safe: bool
    violations: list[str]
    warnings: list[str]
    risk_score: float  # 0-100, higher = riskier
    checks_passed: int
    checks_total: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "is_safe": self.is_safe,
            "violations": self.violations,
            "warnings": self.warnings,
            "risk_score": round(self.risk_score, 1),
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
        }


# Hard constraint definitions
HARD_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "max_position_sol": {"min": 0.001, "max": 10.0},
    "max_daily_loss_sol": {"min": 0.01, "max": 50.0},
    "max_drawdown_pct": {"min": 1.0, "max": 50.0},
    "stop_loss_pct": {"min": 0.5, "max": 50.0},
    "take_profit_pct": {"min": 1.0, "max": 200.0},
    "min_confidence": {"min": 0.1, "max": 1.0},
    "max_risk_score": {"min": 0.05, "max": 0.95},
    "max_concurrent_positions": {"min": 1, "max": 20},
    "pct_of_wallet": {"min": 0.001, "max": 0.5},
    "min_liquidity_usd": {"min": 100.0, "max": 10000000.0},
    "max_hold_minutes": {"min": 1, "max": 1440},
    "trailing_activation_pct": {"min": 1.0, "max": 100.0},
}

# Maximum percentage change per parameter per deployment
MAX_CHANGE_PCT: dict[str, float] = {
    "max_position_sol": 50.0,
    "max_daily_loss_sol": 50.0,
    "stop_loss_pct": 50.0,
    "take_profit_pct": 50.0,
    "min_confidence": 30.0,
    "max_risk_score": 30.0,
    "max_drawdown_pct": 30.0,
    "pct_of_wallet": 50.0,
    "min_liquidity_usd": 100.0,
    "max_hold_minutes": 100.0,
    # Default for unspecified parameters
    "_default": 50.0,
}


class SafetyValidator:
    """
    Multi-layer safety validation before parameter deployment.

    Each validation layer adds independent checks. All layers must pass
    for the parameters to be considered safe.

    The risk_score (0-100) provides a continuous measure even when
    all checks pass, allowing the ApprovalWorkflow to make nuanced
    decisions about whether to auto-apply.
    """

    def __init__(
        self,
        hard_constraints: dict[str, dict[str, Any]] | None = None,
        max_change_pct: dict[str, float] | None = None,
        min_reward_risk_ratio: float = 2.0,
        task_router: Any = None,
        cost_tracker: Any = None,
    ) -> None:
        """
        Initialize the safety validator.

        Args:
            hard_constraints: Override hard constraint definitions.
            max_change_pct: Override maximum change percentages.
            min_reward_risk_ratio: Minimum reward-to-risk ratio.
            task_router: TaskComplexityRouter for hybrid Opus/Sonnet routing.
            cost_tracker: CostTracker for tracking AI call costs.
        """
        self.hard_constraints = hard_constraints or HARD_CONSTRAINTS
        self.max_change_pct = max_change_pct or MAX_CHANGE_PCT
        self.min_reward_risk_ratio = min_reward_risk_ratio
        self.task_router = task_router
        self.cost_tracker = cost_tracker

    async def validate(
        self,
        proposed_params: dict[str, Any],
        current_params: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """
        Validate proposed parameters against all safety rules.

        Args:
            proposed_params: The proposed parameter values.
            current_params: The currently deployed parameter values.

        Returns:
            Tuple of (is_safe, violations).
            is_safe is True only if ALL checks pass.
            violations is a list of human-readable violation descriptions.
        """
        report = await self.validate_full(proposed_params, current_params)
        return report.is_safe, report.violations

    async def validate_full(
        self,
        proposed_params: dict[str, Any],
        current_params: dict[str, Any],
    ) -> SafetyReport:
        """
        Run full validation and return a detailed SafetyReport.

        Args:
            proposed_params: The proposed parameter values.
            current_params: The currently deployed parameter values.

        Returns:
            SafetyReport with all violations, warnings, and risk score.
        """
        violations: list[str] = []
        warnings: list[str] = []
        risk_score = 0.0
        checks_passed = 0
        checks_total = 0

        # Layer 1: Hard constraints
        layer_violations, layer_warnings, layer_risk = self._check_hard_constraints(proposed_params)
        violations.extend(layer_violations)
        warnings.extend(layer_warnings)
        risk_score += layer_risk
        checks_total += 1
        if not layer_violations:
            checks_passed += 1

        # Layer 2: Change limits
        layer_violations, layer_warnings, layer_risk = self._check_change_limits(
            proposed_params, current_params
        )
        violations.extend(layer_violations)
        warnings.extend(layer_warnings)
        risk_score += layer_risk
        checks_total += 1
        if not layer_violations:
            checks_passed += 1

        # Layer 3: Parameter sanity
        layer_violations, layer_warnings, layer_risk = self._check_parameter_sanity(proposed_params)
        violations.extend(layer_violations)
        warnings.extend(layer_warnings)
        risk_score += layer_risk
        checks_total += 1
        if not layer_violations:
            checks_passed += 1

        # Layer 4: Risk/reward validation
        layer_violations, layer_warnings, layer_risk = self._check_risk_reward(proposed_params)
        violations.extend(layer_violations)
        warnings.extend(layer_warnings)
        risk_score += layer_risk
        checks_total += 1
        if not layer_violations:
            checks_passed += 1

        # Layer 5: Interaction conflicts
        layer_violations, layer_warnings, layer_risk = self._check_interactions(proposed_params)
        violations.extend(layer_violations)
        warnings.extend(layer_warnings)
        risk_score += layer_risk
        checks_total += 1
        if not layer_violations:
            checks_passed += 1

        # Normalize risk score to 0-100
        risk_score = min(100.0, risk_score)

        is_safe = len(violations) == 0

        if violations:
            logger.warning(
                "Safety validation FAILED with %d violations: %s",
                len(violations),
                "; ".join(violations[:5]),
            )
        elif warnings:
            logger.info(
                "Safety validation passed with %d warnings (risk: %.1f)",
                len(warnings),
                risk_score,
            )

        return SafetyReport(
            is_safe=is_safe,
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
            checks_passed=checks_passed,
            checks_total=checks_total,
        )

    def _check_hard_constraints(
        self,
        proposed: dict[str, Any],
    ) -> tuple[list[str], list[str], float]:
        """
        Layer 1: Check absolute hard constraints.

        Ensures all parameters are within their allowed ranges.
        """
        violations: list[str] = []
        warnings: list[str] = []
        risk = 0.0

        for param_name, limits in self.hard_constraints.items():
            if param_name not in proposed:
                continue

            value = proposed[param_name]
            if value is None:
                continue

            if not isinstance(value, (int, float)):
                continue

            param_min = limits.get("min")
            param_max = limits.get("max")

            if param_min is not None and value < param_min:
                violations.append(
                    f"HARD CONSTRAINT: {param_name}={value} is below minimum {param_min}"
                )
                risk += 20.0

            if param_max is not None and value > param_max:
                violations.append(
                    f"HARD CONSTRAINT: {param_name}={value} exceeds maximum {param_max}"
                )
                risk += 20.0

            # Warning if close to limits (within 10%)
            if param_min is not None and param_max is not None:
                range_size = param_max - param_min
                if range_size > 0:
                    margin_low = (value - param_min) / range_size
                    margin_high = (param_max - value) / range_size
                    if margin_low < 0.1 or margin_high < 0.1:
                        warnings.append(
                            f"NEAR LIMIT: {param_name}={value} is close "
                            f"to bounds [{param_min}, {param_max}]"
                        )
                        risk += 5.0

        return violations, warnings, risk

    def _check_change_limits(
        self,
        proposed: dict[str, Any],
        current: dict[str, Any],
    ) -> tuple[list[str], list[str], float]:
        """
        Layer 2: Check maximum change limits per parameter.

        No single parameter should change by more than its max_change_pct
        in a single deployment.
        """
        violations: list[str] = []
        warnings: list[str] = []
        risk = 0.0

        for param_name in proposed:
            if param_name not in current:
                continue

            old_val = current[param_name]
            new_val = proposed[param_name]

            if old_val is None or new_val is None:
                continue
            if not isinstance(old_val, (int, float)):
                continue
            if not isinstance(new_val, (int, float)):
                continue
            if old_val == 0:
                continue

            change_pct = abs((new_val - old_val) / old_val) * 100
            max_allowed = self.max_change_pct.get(
                param_name,
                self.max_change_pct.get("_default", 50.0),
            )

            if change_pct > max_allowed:
                violations.append(
                    f"CHANGE LIMIT: {param_name} changed {change_pct:.1f}% "
                    f"({old_val} -> {new_val}), max allowed {max_allowed}%"
                )
                risk += 15.0
            elif change_pct > max_allowed * 0.7:
                warnings.append(
                    f"LARGE CHANGE: {param_name} changed {change_pct:.1f}% ({old_val} -> {new_val})"
                )
                risk += 5.0

        return violations, warnings, risk

    def _check_parameter_sanity(
        self,
        proposed: dict[str, Any],
    ) -> tuple[list[str], list[str], float]:
        """
        Layer 3: Check logical parameter consistency.

        Rules:
        - stop_loss_pct must be < take_profit_pct
        - trailing_activation_pct must be < take_profit_pct
        - min_confidence must be > max_risk_score (for standard mode)
        """
        violations: list[str] = []
        warnings: list[str] = []
        risk = 0.0

        stop_loss = proposed.get("stop_loss_pct")
        take_profit = proposed.get("take_profit_pct")
        trailing_activation = proposed.get("trailing_activation_pct")
        min_confidence = proposed.get("min_confidence")
        max_risk_score = proposed.get("max_risk_score")

        # Stop loss must be less than take profit
        if stop_loss is not None and take_profit is not None and stop_loss >= take_profit:
            violations.append(
                f"SANITY: stop_loss_pct ({stop_loss}) must be less than "
                f"take_profit_pct ({take_profit})"
            )
            risk += 25.0

        # Trailing activation must be less than take profit
        if (
            trailing_activation is not None
            and take_profit is not None
            and trailing_activation >= take_profit
        ):
            violations.append(
                f"SANITY: trailing_activation_pct ({trailing_activation}) "
                f"must be less than take_profit_pct ({take_profit})"
            )
            risk += 15.0

        # min_confidence should be meaningfully above max_risk_score
        if min_confidence is not None and max_risk_score is not None:
            if min_confidence < max_risk_score:
                warnings.append(
                    f"QUESTIONABLE: min_confidence ({min_confidence}) is "
                    f"less than max_risk_score ({max_risk_score})"
                )
                risk += 10.0

        return violations, warnings, risk

    def _check_risk_reward(
        self,
        proposed: dict[str, Any],
    ) -> tuple[list[str], list[str], float]:
        """
        Layer 4: Validate risk/reward ratio.

        The take_profit / stop_loss ratio should meet the minimum
        reward-to-risk requirement.
        """
        violations: list[str] = []
        warnings: list[str] = []
        risk = 0.0

        stop_loss = proposed.get("stop_loss_pct")
        take_profit = proposed.get("take_profit_pct")

        if stop_loss and take_profit and stop_loss > 0:
            ratio = take_profit / stop_loss
            if ratio < self.min_reward_risk_ratio:
                violations.append(
                    f"RISK/REWARD: ratio {ratio:.2f} is below minimum "
                    f"{self.min_reward_risk_ratio:.1f}:1 "
                    f"(TP={take_profit}%, SL={stop_loss}%)"
                )
                risk += 20.0
            elif ratio < self.min_reward_risk_ratio * 1.2:
                warnings.append(
                    f"LOW RATIO: reward/risk {ratio:.2f} is close to "
                    f"minimum {self.min_reward_risk_ratio:.1f}:1"
                )
                risk += 5.0

        return violations, warnings, risk

    def _check_interactions(
        self,
        proposed: dict[str, Any],
    ) -> tuple[list[str], list[str], float]:
        """
        Layer 5: Check for contradictory parameter combinations.

        Detects parameter combinations that would create ineffective
        or dangerous trading behavior.
        """
        violations: list[str] = []
        warnings: list[str] = []
        risk = 0.0

        # Very tight stop loss with very long hold time is risky
        stop_loss = proposed.get("stop_loss_pct", 0)
        max_hold = proposed.get("max_hold_minutes", 0)
        if isinstance(stop_loss, (int, float)) and isinstance(max_hold, (int, float)):
            if stop_loss < 3.0 and max_hold > 120:
                warnings.append(
                    f"INTERACTION: Tight stop loss ({stop_loss}%) with "
                    f"long hold time ({max_hold}min) may cause excessive "
                    f"stop-outs in volatile markets"
                )
                risk += 10.0

        # Very high position size with many concurrent positions
        max_pos = proposed.get("max_position_sol", 0)
        max_concurrent = proposed.get("max_concurrent_positions", 0)
        max_daily_loss = proposed.get("max_daily_loss_sol", 0)
        if (
            isinstance(max_pos, (int, float))
            and isinstance(max_concurrent, (int, float))
            and isinstance(max_daily_loss, (int, float))
        ):
            total_exposure = max_pos * max_concurrent
            if max_daily_loss > 0 and total_exposure > max_daily_loss * 2:
                warnings.append(
                    f"INTERACTION: Max exposure ({total_exposure:.2f} SOL) "
                    f"is more than 2x daily loss limit "
                    f"({max_daily_loss} SOL)"
                )
                risk += 10.0

        # High confidence threshold with low liquidity requirement
        min_conf = proposed.get("min_confidence", 0)
        min_liq = proposed.get("min_liquidity_usd", 0)
        if isinstance(min_conf, (int, float)) and isinstance(min_liq, (int, float)):
            if min_conf > 0.9 and min_liq < 1000:
                warnings.append(
                    f"INTERACTION: High confidence threshold ({min_conf}) "
                    f"with low liquidity requirement (${min_liq}) may "
                    f"produce low-quality signals"
                )
                risk += 5.0

        return violations, warnings, risk

    def get_routing_decision(
        self,
        proposed_params: dict[str, Any],
    ) -> Any | None:
        """
        Get a model routing decision for AI-enhanced safety assessment.

        Uses TaskComplexityRouter to select the appropriate model
        (Opus 4.6 for high-stakes safety reviews, Sonnet 4.5 for
        routine parameter checks).

        Safety validation is always classified as high-stakes when
        parameters have many changes or involve risk-critical values.

        Args:
            proposed_params: The proposed parameter values.

        Returns:
            ModelRoutingDecision if task_router is available, else None.
        """
        if self.task_router is None:
            return None

        num_params = len(proposed_params)
        has_risk_params = any(
            k in proposed_params
            for k in (
                "max_daily_loss_sol",
                "max_drawdown_pct",
                "max_position_sol",
            )
        )

        decision = self.task_router.route_task(
            task_type="risk_assessment",
            num_parameters=num_params,
            has_constraints=True,
            requires_explanation=True,
            is_high_stakes=has_risk_params,
        )

        return decision

    async def record_validation_cost(
        self,
        optimization_id: str,
        task_type: str = "safety_validation",
        tokens_input: int = 500,
        tokens_output: int = 1000,
        latency_ms: int = 100,
        cache_hit: bool = False,
    ) -> None:
        """
        Record the cost of a safety validation AI call.

        Integrates with CostTracker to track Phase 3 governance costs
        alongside Phase 2 optimization costs.

        Args:
            optimization_id: The optimization run ID.
            task_type: Type of safety task performed.
            tokens_input: Input tokens consumed.
            tokens_output: Output tokens consumed.
            latency_ms: Request latency.
            cache_hit: Whether result was cached.
        """
        if self.cost_tracker is None:
            return

        routing = self.get_routing_decision({})
        model = routing.model_id if routing else "claude-sonnet-4-5-20250929"

        await self.cost_tracker.record_api_call(
            optimization_id=optimization_id,
            model=model,
            task_type=task_type,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        )
