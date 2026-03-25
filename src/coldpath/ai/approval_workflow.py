"""
Approval workflow for strategy optimization deployment.

Determines whether an optimization result can be auto-applied or
requires explicit user approval based on configurable thresholds.

Auto-apply conditions (ALL must be met):
- Paper mode (not live trading)
- Small parameter changes (<10% per parameter)
- High backtest confidence (>0.9)
- Significant Sharpe improvement (+0.2)

Require approval if ANY:
- Live trading mode
- Large parameter changes (>10%)
- Lower backtest confidence
- Safety concerns flagged

Usage:
    workflow = ApprovalWorkflow(trading_mode="paper")
    decision = await workflow.determine_approval_required(context)
    if decision.auto_apply:
        deploy_immediately()
    else:
        queue_for_user_review(decision.approval_request)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ApprovalDecision:
    """Result of the approval determination."""

    auto_apply: bool
    reason: str
    requires_user_approval: bool
    risk_level: str  # "low", "medium", "high"
    approval_request: dict[str, Any] | None = None

    # Detailed reasoning
    conditions_met: list[str] = field(default_factory=list)
    conditions_failed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "auto_apply": self.auto_apply,
            "reason": self.reason,
            "requires_user_approval": self.requires_user_approval,
            "risk_level": self.risk_level,
            "conditions_met": self.conditions_met,
            "conditions_failed": self.conditions_failed,
            "approval_request": self.approval_request,
        }


@dataclass
class ApprovalThresholds:
    """Configurable thresholds for auto-apply decisions."""

    # Parameter change limits for auto-apply
    max_param_change_pct: float = 10.0

    # Backtest confidence requirements
    min_backtest_confidence: float = 0.9

    # Performance improvement requirements
    min_sharpe_improvement: float = 0.2
    min_win_rate_improvement: float = 0.02  # 2 percentage points

    # Safety score threshold
    max_risk_score: float = 30.0  # SafetyValidator risk_score

    # Trading mode restrictions
    auto_apply_modes: list[str] = field(default_factory=lambda: ["paper", "backtest"])


class ApprovalWorkflow:
    """
    Determine whether optimization results require user approval.

    Evaluates multiple conditions to decide between auto-apply and
    manual approval. Errs on the side of caution: if any condition
    fails, approval is required.

    The workflow also generates structured approval requests for the
    UI to present to the user when manual approval is needed.
    """

    def __init__(
        self,
        trading_mode: str = "paper",
        thresholds: ApprovalThresholds | None = None,
        task_router: Any = None,
        cost_tracker: Any = None,
    ) -> None:
        """
        Initialize the approval workflow.

        Args:
            trading_mode: Current trading mode ("paper", "live", "backtest").
            thresholds: Override approval thresholds.
            task_router: TaskComplexityRouter for hybrid Opus/Sonnet routing.
            cost_tracker: CostTracker for tracking AI call costs.
        """
        self.trading_mode = trading_mode
        self.thresholds = thresholds or ApprovalThresholds()
        self.task_router = task_router
        self.cost_tracker = cost_tracker

    async def determine_approval_required(
        self,
        context: Any,
    ) -> ApprovalDecision:
        """
        Determine if an optimization requires user approval.

        Evaluates the optimization context against all auto-apply
        conditions. If all conditions are met, auto-apply is allowed.
        If any condition fails, manual approval is required.

        Args:
            context: OptimizationContext with proposed_params,
                     current_params, backtest_results, etc.

        Returns:
            ApprovalDecision with auto_apply flag and reasoning.
        """
        conditions_met: list[str] = []
        conditions_failed: list[str] = []

        # Condition 1: Trading mode allows auto-apply
        if self.trading_mode in self.thresholds.auto_apply_modes:
            conditions_met.append(f"Trading mode '{self.trading_mode}' allows auto-apply")
        else:
            conditions_failed.append(
                f"Trading mode '{self.trading_mode}' requires approval "
                f"(auto-apply only in: {self.thresholds.auto_apply_modes})"
            )

        # Condition 2: Small parameter changes
        if (
            hasattr(context, "proposed_params")
            and context.proposed_params is not None
            and hasattr(context, "current_params")
            and context.current_params is not None
        ):
            max_change = self._compute_max_change(context.current_params, context.proposed_params)
            if max_change <= self.thresholds.max_param_change_pct:
                conditions_met.append(
                    f"Max parameter change {max_change:.1f}% is within "
                    f"limit ({self.thresholds.max_param_change_pct}%)"
                )
            else:
                conditions_failed.append(
                    f"Max parameter change {max_change:.1f}% exceeds "
                    f"limit ({self.thresholds.max_param_change_pct}%)"
                )
        else:
            conditions_failed.append("Missing proposed or current parameters")

        # Condition 3: High backtest confidence
        backtest = getattr(context, "backtest_results", None)
        if backtest and isinstance(backtest, dict):
            confidence = backtest.get("confidence", 0.0)
            if confidence >= self.thresholds.min_backtest_confidence:
                conditions_met.append(
                    f"Backtest confidence {confidence:.2f} meets minimum "
                    f"({self.thresholds.min_backtest_confidence})"
                )
            else:
                conditions_failed.append(
                    f"Backtest confidence {confidence:.2f} below minimum "
                    f"({self.thresholds.min_backtest_confidence})"
                )

            # Condition 4: Significant improvement
            sharpe_improvement = backtest.get("sharpe_improvement", 0.0)
            if sharpe_improvement >= self.thresholds.min_sharpe_improvement:
                conditions_met.append(
                    f"Sharpe improvement +{sharpe_improvement:.2f} meets "
                    f"minimum (+{self.thresholds.min_sharpe_improvement})"
                )
            else:
                conditions_failed.append(
                    f"Sharpe improvement +{sharpe_improvement:.2f} below "
                    f"minimum (+{self.thresholds.min_sharpe_improvement})"
                )
        else:
            conditions_failed.append("No backtest results available")

        # Determine decision
        auto_apply = len(conditions_failed) == 0

        if auto_apply:
            risk_level = "low"
            reason = (
                f"All {len(conditions_met)} auto-apply conditions met. "
                f"Safe for automatic deployment."
            )
        else:
            risk_level = "high" if len(conditions_failed) >= 3 else "medium"
            reason = (
                f"{len(conditions_failed)} condition(s) require manual "
                f"approval: {'; '.join(conditions_failed[:3])}"
            )

        # Build approval request for UI if approval needed
        approval_request = None
        if not auto_apply:
            approval_request = self._build_approval_request(
                context, conditions_met, conditions_failed
            )

        decision = ApprovalDecision(
            auto_apply=auto_apply,
            reason=reason,
            requires_user_approval=not auto_apply,
            risk_level=risk_level,
            conditions_met=conditions_met,
            conditions_failed=conditions_failed,
            approval_request=approval_request,
        )

        logger.info(
            "Approval decision: auto_apply=%s, risk=%s, reason=%s",
            auto_apply,
            risk_level,
            reason,
        )

        return decision

    def _compute_max_change(
        self,
        current: dict[str, Any],
        proposed: dict[str, Any],
    ) -> float:
        """Compute the maximum percentage change across all parameters."""
        max_change = 0.0

        for key in proposed:
            if key not in current:
                continue

            old_val = current[key]
            new_val = proposed[key]

            if old_val is None or new_val is None:
                continue
            if not isinstance(old_val, (int, float)):
                continue
            if not isinstance(new_val, (int, float)):
                continue
            if old_val == 0:
                continue

            change_pct = abs((new_val - old_val) / old_val) * 100
            if change_pct > max_change:
                max_change = change_pct

        return max_change

    def _build_approval_request(
        self,
        context: Any,
        conditions_met: list[str],
        conditions_failed: list[str],
    ) -> dict[str, Any]:
        """Build a structured approval request for the UI."""
        request: dict[str, Any] = {
            "optimization_id": getattr(context, "optimization_id", "unknown"),
            "requested_at": datetime.now().isoformat(),
            "trading_mode": self.trading_mode,
            "conditions_met": conditions_met,
            "conditions_failed": conditions_failed,
        }

        # Add parameter comparison if available
        if (
            hasattr(context, "proposed_params")
            and context.proposed_params is not None
            and hasattr(context, "current_params")
            and context.current_params is not None
        ):
            changes: dict[str, Any] = {}
            for key in context.proposed_params:
                old_val = context.current_params.get(key)
                new_val = context.proposed_params[key]
                if old_val != new_val:
                    changes[key] = {
                        "current": old_val,
                        "proposed": new_val,
                    }
                    if (
                        isinstance(old_val, (int, float))
                        and isinstance(new_val, (int, float))
                        and old_val != 0
                    ):
                        changes[key]["change_pct"] = round(
                            ((new_val - old_val) / abs(old_val)) * 100, 1
                        )
            request["parameter_changes"] = changes

        # Add backtest summary if available
        backtest = getattr(context, "backtest_results", None)
        if backtest and isinstance(backtest, dict):
            request["backtest_summary"] = {
                "confidence": backtest.get("confidence"),
                "sharpe_improvement": backtest.get("sharpe_improvement"),
                "win_rate_improvement": backtest.get("win_rate_improvement"),
                "total_trades_tested": backtest.get("total_trades_tested"),
            }

        return request

    def update_trading_mode(self, mode: str) -> None:
        """
        Update the trading mode (affects auto-apply eligibility).

        Args:
            mode: New trading mode ("paper", "live", "backtest").
        """
        self.trading_mode = mode
        logger.info("Trading mode updated to '%s'", mode)

    def get_routing_decision(self) -> dict[str, Any] | None:
        """
        Get a model routing decision for approval-related AI calls.

        Routes to Opus 4.6 for live trading approvals (high-stakes)
        and Sonnet 4.5 for paper/backtest approvals (routine).

        Returns:
            ModelRoutingDecision dict if task_router is available, else None.
        """
        if self.task_router is None:
            return None

        is_live = self.trading_mode == "live"

        decision = self.task_router.route_task(
            task_type="risk_assessment" if is_live else "performance_summary",
            num_parameters=0,
            has_constraints=True,
            requires_explanation=True,
            is_high_stakes=is_live,
        )

        return decision.to_dict()

    async def record_approval_cost(
        self,
        optimization_id: str,
        decision: "ApprovalDecision",
        tokens_input: int = 300,
        tokens_output: int = 500,
        latency_ms: int = 80,
    ) -> None:
        """
        Record the cost of an approval workflow AI call.

        Args:
            optimization_id: The optimization run ID.
            decision: The approval decision made.
            tokens_input: Input tokens consumed.
            tokens_output: Output tokens consumed.
            latency_ms: Request latency.
        """
        if self.cost_tracker is None:
            return

        routing = self.get_routing_decision()
        model = routing["model_id"] if routing else "claude-sonnet-4-5-20250929"

        await self.cost_tracker.record_api_call(
            optimization_id=optimization_id,
            model=model,
            task_type="approval_workflow",
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            cache_hit=False,
        )
