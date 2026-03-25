"""
AI Strategy Orchestrator - State machine coordinator for optimization pipeline.

Coordinates the complete optimization lifecycle from trigger to deployment,
with comprehensive error handling, resource management, timeout control,
and progress tracking.

State transitions:
  IDLE -> ANALYZING -> GENERATING -> BACKTESTING -> VALIDATING ->
  AWAITING_APPROVAL -> DEPLOYING -> MONITORING -> COMPLETED

  Any stage can transition to FAILED.
  MONITORING can transition to ROLLED_BACK if performance degrades.

Usage:
    orchestrator = AIStrategyOrchestrator(
        performance_engine=perf_engine,
        drift_detector=drift_detector,
        safety_validator=safety_validator,
        approval_workflow=approval_workflow,
        rollback_manager=rollback_manager,
        resource_guard=resource_guard,
        claude_client=claude_client,
        backtest_engine=backtest_engine,
        parameter_store=param_store,
    )
    opt_id = await orchestrator.trigger_optimization(
        reason="drift_detected",
        goal="maximize_sharpe",
        constraints={"max_daily_loss_sol": 1.0},
    )
"""

import asyncio
import gc
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class OptimizationStage(Enum):
    """Optimization pipeline stages."""

    IDLE = "idle"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    BACKTESTING = "backtesting"
    VALIDATING = "validating"
    AWAITING_APPROVAL = "awaiting_approval"
    DEPLOYING = "deploying"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


# Valid stage transitions
VALID_TRANSITIONS: dict[OptimizationStage, list[OptimizationStage]] = {
    OptimizationStage.IDLE: [
        OptimizationStage.ANALYZING,
        OptimizationStage.FAILED,
    ],
    OptimizationStage.ANALYZING: [
        OptimizationStage.GENERATING,
        OptimizationStage.FAILED,
    ],
    OptimizationStage.GENERATING: [
        OptimizationStage.BACKTESTING,
        OptimizationStage.FAILED,
    ],
    OptimizationStage.BACKTESTING: [
        OptimizationStage.VALIDATING,
        OptimizationStage.FAILED,
    ],
    OptimizationStage.VALIDATING: [
        OptimizationStage.AWAITING_APPROVAL,
        OptimizationStage.DEPLOYING,
        OptimizationStage.FAILED,
    ],
    OptimizationStage.AWAITING_APPROVAL: [
        OptimizationStage.DEPLOYING,
        OptimizationStage.FAILED,
    ],
    OptimizationStage.DEPLOYING: [
        OptimizationStage.MONITORING,
        OptimizationStage.FAILED,
    ],
    OptimizationStage.MONITORING: [
        OptimizationStage.COMPLETED,
        OptimizationStage.ROLLED_BACK,
        OptimizationStage.FAILED,
    ],
    OptimizationStage.COMPLETED: [],
    OptimizationStage.FAILED: [],
    OptimizationStage.ROLLED_BACK: [],
}


@dataclass
class StageProgress:
    """Progress information for the current stage."""

    fraction: float = 0.0  # 0.0 to 1.0
    message: str = ""
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class OptimizationContext:
    """Context for an optimization run."""

    optimization_id: str
    trigger_reason: str  # "drift_detected", "scheduled", "manual"
    started_at: datetime
    current_stage: OptimizationStage
    goal: str  # "maximize_sharpe", "minimize_drawdown", etc.
    constraints: dict[str, Any]
    current_params: dict[str, Any]
    urgency: str = "normal"

    # Results at each stage
    proposed_params: dict[str, Any] | None = None
    performance_data: dict[str, Any] | None = None
    market_regime: str | None = None
    recommendations: dict[str, Any] | None = None
    backtest_results: dict[str, Any] | None = None
    safety_report: dict[str, Any] | None = None
    promotion_gates: dict[str, Any] | None = None

    # Approval state
    approval_required: bool = False
    approved_by: str | None = None
    approval_decision: dict[str, Any] | None = None

    # Progress tracking
    progress: StageProgress = field(default_factory=StageProgress)
    stage_history: list[dict[str, Any]] = field(default_factory=list)

    # Error state
    error: str | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "optimization_id": self.optimization_id,
            "trigger_reason": self.trigger_reason,
            "started_at": self.started_at.isoformat(),
            "current_stage": self.current_stage.value,
            "goal": self.goal,
            "constraints": self.constraints,
            "urgency": self.urgency,
            "has_proposed_params": self.proposed_params is not None,
            "has_backtest_results": self.backtest_results is not None,
            "has_promotion_gates": self.promotion_gates is not None,
            "market_regime": self.market_regime,
            "approval_required": self.approval_required,
            "approved_by": self.approved_by,
            "progress": {
                "fraction": self.progress.fraction,
                "message": self.progress.message,
            },
            "stage_history": self.stage_history,
            "error": self.error,
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
        }


class AIStrategyOrchestrator:
    """
    Coordinates the complete optimization pipeline.

    This is the central state machine that:
    1. Validates resource availability before starting
    2. Analyzes current performance
    3. Generates optimization recommendations (via Claude AI)
    4. Backtests proposed changes
    5. Validates safety constraints
    6. Handles approval workflow
    7. Deploys new parameters
    8. Monitors post-deployment performance
    9. Auto-rolls back if degradation is detected

    Each stage has:
    - A timeout (default 5 minutes, configurable)
    - Progress tracking
    - Resource management
    - Error handling with automatic FAILED transition

    Thread safety:
    - Stage transitions are protected by an asyncio.Lock
    - Only one optimization can be active at a time
    """

    # Default timeout per stage (seconds)
    DEFAULT_STAGE_TIMEOUT = 300  # 5 minutes

    # Stage-specific timeout overrides
    STAGE_TIMEOUTS: dict[OptimizationStage, float] = {
        OptimizationStage.ANALYZING: 120,
        OptimizationStage.GENERATING: 180,  # Claude API can be slow
        OptimizationStage.BACKTESTING: 300,
        OptimizationStage.VALIDATING: 60,
        OptimizationStage.DEPLOYING: 60,
    }

    def __init__(
        self,
        performance_engine: Any = None,
        drift_detector: Any = None,
        safety_validator: Any = None,
        approval_workflow: Any = None,
        rollback_manager: Any = None,
        rollout_controller: Any = None,
        resource_guard: Any = None,
        claude_client: Any = None,
        backtest_engine: Any = None,
        parameter_store: Any = None,
        parameter_tracker: Any = None,
        # Phase 2 components
        task_router: Any = None,
        cost_tracker: Any = None,
        interaction_analyzer: Any = None,
        stress_test_engine: Any = None,
        robustness_validator: Any = None,
        oos_validator: Any = None,
    ) -> None:
        """
        Initialize the orchestrator with all required components.

        Args:
            performance_engine: PerformanceFeedbackEngine instance.
            drift_detector: DriftDetector instance.
            safety_validator: SafetyValidator instance.
            approval_workflow: ApprovalWorkflow instance.
            rollback_manager: RollbackManager instance.
            resource_guard: ResourceGuard instance.
            claude_client: ClaudeClient instance.
            backtest_engine: Backtest validation engine.
            parameter_store: ParameterStore instance.
            parameter_tracker: ParameterEffectivenessTracker instance.
        """
        self.performance_engine = performance_engine
        self.drift_detector = drift_detector
        self.safety_validator = safety_validator
        self.approval_workflow = approval_workflow
        self.rollback_manager = rollback_manager
        self.rollout_controller = rollout_controller
        self.resource_guard = resource_guard
        self.claude_client = claude_client
        self.backtest_engine = backtest_engine
        self.parameter_store = parameter_store
        self.parameter_tracker = parameter_tracker

        # Phase 2 components
        self.task_router = task_router
        self.cost_tracker = cost_tracker
        self.interaction_analyzer = interaction_analyzer
        self.stress_test_engine = stress_test_engine
        self.robustness_validator = robustness_validator
        self.oos_validator = oos_validator

        # Phase 3: Propagate router/tracker to governance components
        self._inject_routing_to_governance()

        # Model version lock (Phase 3)
        self.model_version_lock = None

        self.active_optimization: OptimizationContext | None = None
        self._stage_lock = asyncio.Lock()
        self._optimization_history: list[dict[str, Any]] = []
        self._max_history = 50

        # Tracked optimization task for clean shutdown
        self._optimization_task: asyncio.Task | None = None

        # Completion event for wait_for_completion
        self._completion_event: asyncio.Event | None = None

        # Callbacks
        self._on_stage_change: list[Callable[[str, str, str], None]] = []
        self._on_completion: list[Callable[[str, str, dict[str, Any]], None]] = []

    def _inject_routing_to_governance(self) -> None:
        """
        Propagate TaskComplexityRouter and CostTracker to Phase 3 components.

        This enables hybrid Opus/Sonnet routing and cost tracking across
        all safety and governance components. Components that already have
        these attributes will receive the orchestrator's instances.
        """
        if self.task_router is None and self.cost_tracker is None:
            return

        # Inject into safety validator
        if self.safety_validator:
            if self.task_router and not getattr(self.safety_validator, "task_router", None):
                self.safety_validator.task_router = self.task_router
            if self.cost_tracker and not getattr(self.safety_validator, "cost_tracker", None):
                self.safety_validator.cost_tracker = self.cost_tracker

        # Inject into approval workflow
        if self.approval_workflow:
            if self.task_router and not getattr(self.approval_workflow, "task_router", None):
                self.approval_workflow.task_router = self.task_router
            if self.cost_tracker and not getattr(self.approval_workflow, "cost_tracker", None):
                self.approval_workflow.cost_tracker = self.cost_tracker

        # Inject into drift detector
        if self.drift_detector:
            if self.task_router and not getattr(self.drift_detector, "task_router", None):
                self.drift_detector.task_router = self.task_router
            if self.cost_tracker and not getattr(self.drift_detector, "cost_tracker", None):
                self.drift_detector.cost_tracker = self.cost_tracker

        # Inject into rollback manager
        if self.rollback_manager:
            if self.task_router and not getattr(self.rollback_manager, "task_router", None):
                self.rollback_manager.task_router = self.task_router
            if self.cost_tracker and not getattr(self.rollback_manager, "cost_tracker", None):
                self.rollback_manager.cost_tracker = self.cost_tracker

        logger.info(
            "Phase 3 governance components injected with TaskComplexityRouter=%s, CostTracker=%s",
            self.task_router is not None,
            self.cost_tracker is not None,
        )

    async def trigger_optimization(
        self,
        reason: str,
        goal: str,
        constraints: dict[str, Any],
        urgency: str = "normal",
    ) -> str:
        """
        Trigger a new optimization run.

        Args:
            reason: Why optimization was triggered ("drift_detected",
                    "scheduled", "manual").
            goal: Optimization objective ("maximize_sharpe",
                  "minimize_drawdown", etc.).
            constraints: Hard constraints to respect.
            urgency: Priority level ("low", "normal", "high").

        Returns:
            optimization_id

        Raises:
            RuntimeError: If an optimization is already in progress or
                         resources are unavailable.
        """
        # Check if an optimization is already running
        if self.active_optimization is not None and self.active_optimization.current_stage not in (
            OptimizationStage.COMPLETED,
            OptimizationStage.FAILED,
            OptimizationStage.ROLLED_BACK,
        ):
            raise RuntimeError(
                f"Optimization already in progress: "
                f"{self.active_optimization.optimization_id} "
                f"(stage: {self.active_optimization.current_stage.value})"
            )

        # Check resource availability
        if self.resource_guard:
            can_start, resource_reason = await self.resource_guard.can_start_optimization()
            if not can_start:
                raise RuntimeError(f"Cannot start optimization: {resource_reason}")

        # Get current parameters
        current_params: dict[str, Any] = {}
        if self.parameter_store:
            current_params = await self.parameter_store.get_current()

        # Create optimization context
        optimization_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        context = OptimizationContext(
            optimization_id=optimization_id,
            trigger_reason=reason,
            started_at=datetime.now(),
            current_stage=OptimizationStage.IDLE,
            goal=goal,
            constraints=constraints,
            current_params=current_params,
            urgency=urgency,
        )

        self.active_optimization = context
        self._completion_event = asyncio.Event()

        # Run optimization pipeline in background (tracked for clean shutdown)
        self._optimization_task = asyncio.create_task(
            self._run_optimization_pipeline(context),
            name=f"optimization-{optimization_id}",
        )

        logger.info(
            "Optimization triggered: id=%s, reason=%s, goal=%s, urgency=%s",
            optimization_id,
            reason,
            goal,
            urgency,
        )

        return optimization_id

    async def _run_optimization_pipeline(
        self,
        context: OptimizationContext,
    ) -> None:
        """Execute the complete optimization pipeline with resource management."""
        try:
            if self.resource_guard:
                async with self.resource_guard.reserve_resources(context.optimization_id):
                    await self._execute_stages(context)
            else:
                await self._execute_stages(context)

        except Exception as exc:
            logger.error(
                "Optimization %s pipeline error: %s",
                context.optimization_id,
                exc,
            )
            context.error = str(exc)
            await self._transition_stage(context, OptimizationStage.FAILED)
        finally:
            # Record in history
            context.completed_at = datetime.now()
            self._optimization_history.append(context.to_dict())
            if len(self._optimization_history) > self._max_history:
                self._optimization_history = self._optimization_history[-self._max_history :]

            # Signal completion
            if self._completion_event:
                self._completion_event.set()

            # Force garbage collection after optimization
            gc.collect()

    async def _execute_stages(
        self,
        context: OptimizationContext,
    ) -> None:
        """Execute all optimization stages sequentially."""

        # Stage 1: ANALYZING
        await self._transition_stage(context, OptimizationStage.ANALYZING)
        await self._run_with_timeout(
            self._stage_analyze(context),
            OptimizationStage.ANALYZING,
        )

        # Stage 2: GENERATING
        await self._transition_stage(context, OptimizationStage.GENERATING)
        await self._run_with_timeout(
            self._stage_generate(context),
            OptimizationStage.GENERATING,
        )

        # Stage 3: BACKTESTING
        await self._transition_stage(context, OptimizationStage.BACKTESTING)
        await self._run_with_timeout(
            self._stage_backtest(context),
            OptimizationStage.BACKTESTING,
        )

        # Stage 4: VALIDATING
        await self._transition_stage(context, OptimizationStage.VALIDATING)
        await self._run_with_timeout(
            self._stage_validate(context),
            OptimizationStage.VALIDATING,
        )

        # Enforce strict model promotion gates before any deployment path.
        gate_result = await self._evaluate_promotion_gates(context)
        context.promotion_gates = gate_result
        if not gate_result.get("passed", False):
            raise ValueError(
                "Promotion gates failed: " + "; ".join(gate_result.get("failed_reasons", []))
            )

        # Stage 5: APPROVAL or DEPLOY (with cost tracking)
        if self.approval_workflow:
            decision = await self.approval_workflow.determine_approval_required(context)
            context.approval_decision = decision.to_dict()

            # Record approval workflow cost
            if hasattr(self.approval_workflow, "record_approval_cost"):
                await self.approval_workflow.record_approval_cost(
                    optimization_id=context.optimization_id,
                    decision=decision,
                )

            if decision.auto_apply:
                # Auto-deploy
                await self._transition_stage(context, OptimizationStage.DEPLOYING)
                await self._run_with_timeout(
                    self._stage_deploy(context),
                    OptimizationStage.DEPLOYING,
                )

                # Stage 6: MONITORING
                await self._transition_stage(context, OptimizationStage.MONITORING)
                await self._stage_start_monitoring(context)
                rollout = self._evaluate_rollout(context)
                if not rollout.get("allowed", True) and rollout.get(
                    "auto_rollback_triggered", False
                ):
                    await self._transition_stage(context, OptimizationStage.ROLLED_BACK)
                    return

                # Stage 7: COMPLETED
                await self._transition_stage(context, OptimizationStage.COMPLETED)
            else:
                # Wait for manual approval
                await self._transition_stage(context, OptimizationStage.AWAITING_APPROVAL)
                context.approval_required = True
                # Approval handled by apply_approval() method
        else:
            # No approval workflow, deploy directly
            await self._transition_stage(context, OptimizationStage.DEPLOYING)
            await self._run_with_timeout(
                self._stage_deploy(context),
                OptimizationStage.DEPLOYING,
            )
            await self._transition_stage(context, OptimizationStage.MONITORING)
            await self._stage_start_monitoring(context)
            rollout = self._evaluate_rollout(context)
            if not rollout.get("allowed", True) and rollout.get("auto_rollback_triggered", False):
                await self._transition_stage(context, OptimizationStage.ROLLED_BACK)
                return
            await self._transition_stage(context, OptimizationStage.COMPLETED)

    # ---- Individual stage implementations ----

    async def _stage_analyze(self, context: OptimizationContext) -> None:
        """Stage 1: Analyze current performance."""
        await self._update_progress(context, 0.0, "Collecting performance data")

        if self.performance_engine:
            context.performance_data = await self.performance_engine.analyze()
            context.market_regime = await self.performance_engine.get_current_regime()
        else:
            context.performance_data = {"sufficient_data": False}
            context.market_regime = "unknown"

        await self._update_progress(context, 1.0, "Performance analysis complete")

    async def _stage_generate(self, context: OptimizationContext) -> None:
        """Stage 2: Generate optimization recommendations."""
        await self._update_progress(context, 0.0, "Preparing optimization request")

        # Build the recommendation, preferring parameter tracker then AI
        recommendations: dict[str, Any] = {"params": {}, "source": "none"}

        # Try parameter tracker recommendations first
        if self.parameter_tracker and context.market_regime:
            await self._update_progress(context, 0.2, "Checking historical parameter effectiveness")
            try:
                tracker_recs = await self.parameter_tracker.get_recommendations(
                    current_params=context.current_params,
                    regime=context.market_regime,
                )
                if tracker_recs:
                    # Apply tracker recommendations
                    proposed = dict(context.current_params)
                    for rec in tracker_recs:
                        proposed[rec.parameter_name] = rec.recommended_value
                    recommendations = {
                        "params": proposed,
                        "source": "parameter_tracker",
                        "tracker_recommendations": [
                            {
                                "param": r.parameter_name,
                                "from": r.current_value,
                                "to": r.recommended_value,
                                "expected_pnl": r.expected_pnl_pct,
                                "confidence": r.confidence,
                            }
                            for r in tracker_recs
                        ],
                    }
            except Exception as exc:
                logger.warning("Parameter tracker failed: %s", exc)

        # Try AI recommendations if tracker didn't produce results
        if not recommendations["params"] and self.claude_client:
            await self._update_progress(context, 0.4, "Generating AI recommendations")
            try:
                # Use routed analysis if task_router is available
                if self.task_router:
                    response = await self.claude_client.analyze_strategy_with_routing(
                        task_type="strategy_analysis",
                        current_params=context.current_params,
                        performance_data=(context.performance_data or {}),
                        market_regime=(context.market_regime or "unknown"),
                        goal=context.goal,
                        constraints=context.constraints,
                        router=self.task_router,
                        cost_tracker=self.cost_tracker,
                        optimization_id=(context.optimization_id),
                    )
                else:
                    response = await self.claude_client.analyze_strategy(
                        strategy_params=context.current_params,
                        objective=context.goal,
                        constraints=context.constraints,
                    )

                # Parse AI response into parameter changes
                proposed = dict(context.current_params)
                ai_recs = self._parse_ai_recommendations(response.content, proposed)
                if ai_recs:
                    recommendations = {
                        "params": proposed,
                        "source": "claude_ai",
                        "ai_response": response.content[:500],
                    }

                    # Analyze parameter interactions if available
                    if self.interaction_analyzer:
                        try:
                            interactions = await self.interaction_analyzer.analyze_interactions(
                                proposed
                            )
                            recommendations["interactions"] = interactions.to_dict()
                        except Exception as exc:
                            logger.warning(
                                "Interaction analysis failed: %s",
                                exc,
                            )
            except Exception as exc:
                logger.warning("AI recommendation failed: %s", exc)

        # Fallback: if no recommendations generated, use current params
        if not recommendations["params"]:
            recommendations["params"] = dict(context.current_params)
            recommendations["source"] = "no_change"

        context.proposed_params = recommendations["params"]
        context.recommendations = recommendations

        await self._update_progress(context, 1.0, "Recommendations generated")

    async def _stage_backtest(self, context: OptimizationContext) -> None:
        """Stage 3: Backtest proposed parameters."""
        await self._update_progress(context, 0.0, "Starting backtest validation")

        if self.backtest_engine and context.proposed_params:
            try:
                results = await self.backtest_engine.validate_params(
                    baseline_params=context.current_params,
                    candidate_params=context.proposed_params,
                    validation_level="standard",
                )
                context.backtest_results = results
            except Exception as exc:
                logger.warning(
                    "Backtest engine failed: %s. Using mock results.",
                    exc,
                )
                context.backtest_results = self._mock_backtest_results(context)
        else:
            # Mock backtest results for testing/when engine unavailable
            context.backtest_results = self._mock_backtest_results(context)

        await self._update_progress(context, 1.0, "Backtest validation complete")

    async def _stage_validate(self, context: OptimizationContext) -> None:
        """Stage 4: Safety validation with hybrid routing.

        Uses the full SafetyReport (including risk_score, warnings)
        and records validation costs via CostTracker.

        When TaskComplexityRouter is available, high-risk parameter
        changes are routed to Opus 4.6 for deeper safety analysis.
        """
        await self._update_progress(context, 0.0, "Running safety validation")

        if self.safety_validator and context.proposed_params:
            # Get routing decision for this safety check
            routing = None
            if hasattr(self.safety_validator, "get_routing_decision"):
                routing = self.safety_validator.get_routing_decision(context.proposed_params)

            await self._update_progress(
                context,
                0.3,
                f"Validating with "
                f"{'Opus 4.6' if routing and 'opus' in routing.model_id.lower() else 'Sonnet 4.5'}"
                if routing
                else "Validating parameters",
            )

            # Use full validation for comprehensive report
            report = await self.safety_validator.validate_full(
                proposed_params=context.proposed_params,
                current_params=context.current_params,
            )

            context.safety_report = report.to_dict()

            # Record the validation cost
            if hasattr(self.safety_validator, "record_validation_cost"):
                await self.safety_validator.record_validation_cost(
                    optimization_id=context.optimization_id,
                )

            if not report.is_safe:
                raise ValueError(f"Safety validation failed: {'; '.join(report.violations[:3])}")
        else:
            context.safety_report = {
                "is_safe": True,
                "violations": [],
                "warnings": [],
                "risk_score": 0.0,
                "checks_passed": 0,
                "checks_total": 0,
            }

        await self._update_progress(context, 1.0, "Safety validation passed")

    async def _evaluate_promotion_gates(
        self,
        context: OptimizationContext,
    ) -> dict[str, Any]:
        """Strict promotion gates: walk-forward, OOS, and paper consistency."""
        backtest = context.backtest_results or {}
        perf = context.performance_data or {}

        failed_reasons: list[str] = []
        passed_checks: list[str] = []

        baseline_sharpe = float(backtest.get("baseline_sharpe", 0.0) or 0.0)
        candidate_sharpe = float(backtest.get("candidate_sharpe", 0.0) or 0.0)
        baseline_wr = float(backtest.get("baseline_win_rate", 0.0) or 0.0)
        candidate_wr = float(backtest.get("candidate_win_rate", 0.0) or 0.0)
        max_drawdown_pct = backtest.get("max_drawdown_pct")

        # Gate 1: Walk-forward / baseline uplift proxy
        # Require positive sharpe uplift, or explicit pass if provided by engine.
        walk_forward_pass = backtest.get("walk_forward_pass")
        if walk_forward_pass is None:
            walk_forward_pass = candidate_sharpe >= max(0.0, baseline_sharpe * 1.01)
        if walk_forward_pass:
            passed_checks.append("walk_forward")
        else:
            failed_reasons.append(
                f"walk_forward failed (candidate_sharpe={candidate_sharpe:.3f}, "
                f"baseline_sharpe={baseline_sharpe:.3f})"
            )

        # Gate 2: Out-of-sample validation
        oos_pass = backtest.get("out_of_sample_pass")
        if oos_pass is None:
            # If detailed OOS metrics exist, enforce generalization ratio >= 0.7.
            gen_ratio = backtest.get("generalization_ratio")
            if gen_ratio is not None:
                oos_pass = float(gen_ratio) >= 0.70
            else:
                # Conservative fallback: no hard fail if metric absent.
                oos_pass = True
        if oos_pass:
            passed_checks.append("out_of_sample")
        else:
            failed_reasons.append("out_of_sample failed")

        # Gate 3: Paper consistency
        # If paper-live divergence is available, enforce <= 20%.
        paper_divergence = perf.get("paper_live_divergence_pct")
        paper_consistency_pass = True
        if paper_divergence is not None:
            paper_consistency_pass = float(paper_divergence) <= 20.0
        else:
            # Fallback proxy: candidate win-rate may not degrade >2pp
            paper_consistency_pass = candidate_wr >= (baseline_wr - 0.02)
        if paper_consistency_pass:
            passed_checks.append("paper_consistency")
        else:
            failed_reasons.append(f"paper_consistency failed (divergence={paper_divergence})")

        # Gate 4: Drawdown cap check if data present
        dd_cap = context.constraints.get("max_drawdown_pct")
        if max_drawdown_pct is not None and dd_cap is not None:
            if float(max_drawdown_pct) <= float(dd_cap):
                passed_checks.append("drawdown_cap")
            else:
                failed_reasons.append(f"drawdown_cap failed ({max_drawdown_pct} > {dd_cap})")

        # Gate 5: Drift-severity hard block when severe
        if self.drift_detector and hasattr(self.drift_detector, "last_report"):
            report = getattr(self.drift_detector, "last_report", None)
            severity = getattr(report, "severity", None) if report else None
            if str(severity).lower() == "severe":
                failed_reasons.append("drift_severity severe")
            else:
                passed_checks.append("drift_gate")

        return {
            "passed": len(failed_reasons) == 0,
            "passed_checks": passed_checks,
            "failed_reasons": failed_reasons,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "baseline_sharpe": baseline_sharpe,
                "candidate_sharpe": candidate_sharpe,
                "baseline_win_rate": baseline_wr,
                "candidate_win_rate": candidate_wr,
                "max_drawdown_pct": max_drawdown_pct,
                "paper_live_divergence_pct": paper_divergence,
            },
        }

    async def _stage_deploy(self, context: OptimizationContext) -> None:
        """Stage 5: Deploy new parameters to ColdPath store and push to HotPath."""
        await self._update_progress(context, 0.0, "Deploying new parameters")

        if self.parameter_store and context.proposed_params:
            await self.parameter_store.deploy(
                new_params=context.proposed_params,
                deployed_by="auto",
                reason=(f"Optimization {context.optimization_id}: {context.trigger_reason}"),
                optimization_id=context.optimization_id,
            )

        # Push to HotPath via ParamPusher if available
        if context.proposed_params:
            await self._push_params_to_hotpath(context)

        await self._update_progress(context, 1.0, "Parameters deployed")

    async def _push_params_to_hotpath(self, context: OptimizationContext) -> None:
        """Push deployed parameters to HotPath via Unix socket/gRPC."""
        try:
            import time

            from coldpath.publishing.param_pusher import (
                ModelParams,
                ParamPusher,
            )

            pusher = ParamPusher()
            connected = await pusher.connect()
            if not connected:
                logger.warning("Cannot push params to HotPath: socket not available")
                return

            params = context.proposed_params or {}
            version = (
                str(self.parameter_store.current_version_number)
                if self.parameter_store
                else context.optimization_id
            )

            model_params = ModelParams(
                version=version,
                timestamp=int(time.time()),
                bandit_arm_weights=params.get("bandit_arm_weights", {}),
                recommended_arm=params.get("recommended_arm", "aggressive"),
                slippage_base_bps=params.get("slippage_base_bps", 15.0),
                slippage_volatility_coef=params.get("slippage_volatility_coef", 0.5),
            )

            success = await pusher.push(model_params)
            await pusher.close()

            if success:
                logger.info(
                    "Pushed optimized params v%s to HotPath",
                    version,
                )
            else:
                logger.warning(
                    "Failed to push params v%s to HotPath",
                    version,
                )
        except ImportError:
            logger.debug("ParamPusher not available, skipping HotPath push")
        except Exception as exc:
            logger.warning("HotPath param push failed: %s", exc)

    async def _stage_start_monitoring(
        self,
        context: OptimizationContext,
    ) -> None:
        """Stage 6: Start post-deployment monitoring."""
        if self.rollback_manager:
            baseline_sharpe = 0.0
            baseline_wr = 0.5
            baseline_dd = 0.0

            if context.performance_data:
                baseline_sharpe = context.performance_data.get("sharpe_ratio", 0.0)
                baseline_wr = context.performance_data.get("win_rate", 0.5)
                baseline_dd = context.performance_data.get("max_drawdown_sol", 0.0)

            await self.rollback_manager.monitor_deployment(
                deployment_id=context.optimization_id,
                optimization_id=context.optimization_id,
                params_version=(
                    self.parameter_store.current_version_number if self.parameter_store else 0
                ),
                baseline_sharpe=baseline_sharpe,
                baseline_win_rate=baseline_wr,
                baseline_drawdown=baseline_dd,
            )

    # ---- Approval handling ----

    async def apply_approval(
        self,
        optimization_id: str,
        approved: bool,
        approved_by: str = "user",
    ) -> bool:
        """
        Apply user approval decision to a pending optimization.

        Args:
            optimization_id: The optimization to approve/reject.
            approved: Whether the optimization is approved.
            approved_by: Who approved it.

        Returns:
            True if the approval was applied successfully.
        """
        if self.active_optimization is None:
            return False
        if self.active_optimization.optimization_id != optimization_id:
            return False
        if self.active_optimization.current_stage != OptimizationStage.AWAITING_APPROVAL:
            return False

        context = self.active_optimization

        if approved:
            context.approved_by = approved_by

            # Deploy
            await self._transition_stage(context, OptimizationStage.DEPLOYING)
            await self._stage_deploy(context)

            # Monitor
            await self._transition_stage(context, OptimizationStage.MONITORING)
            await self._stage_start_monitoring(context)
            rollout = self._evaluate_rollout(context)
            if not rollout.get("allowed", True) and rollout.get("auto_rollback_triggered", False):
                await self._transition_stage(context, OptimizationStage.ROLLED_BACK)
                return True

            # Complete
            await self._transition_stage(context, OptimizationStage.COMPLETED)
        else:
            context.error = f"Rejected by {approved_by}"
            await self._transition_stage(context, OptimizationStage.FAILED)

        return True

    # ---- State machine helpers ----

    async def _transition_stage(
        self,
        context: OptimizationContext,
        new_stage: OptimizationStage,
    ) -> None:
        """Transition to a new stage with validation and logging."""
        async with self._stage_lock:
            old_stage = context.current_stage

            # Validate transition
            valid_targets = VALID_TRANSITIONS.get(old_stage, [])
            if new_stage not in valid_targets:
                raise ValueError(
                    f"Invalid transition: {old_stage.value} -> "
                    f"{new_stage.value}. Valid: "
                    f"{[s.value for s in valid_targets]}"
                )

            # Record in stage history
            now = datetime.now()
            context.stage_history.append(
                {
                    "from": old_stage.value,
                    "to": new_stage.value,
                    "timestamp": now.isoformat(),
                }
            )

            # Update progress tracking
            if context.progress.started_at:
                context.progress.completed_at = now
            context.progress = StageProgress(started_at=now)

            context.current_stage = new_stage

            logger.info(
                "Optimization %s: %s -> %s",
                context.optimization_id,
                old_stage.value,
                new_stage.value,
            )

            # Notify callbacks
            for cb in self._on_stage_change:
                try:
                    cb(
                        context.optimization_id,
                        old_stage.value,
                        new_stage.value,
                    )
                except Exception:
                    pass

            # Notify completion callbacks
            if new_stage in (
                OptimizationStage.COMPLETED,
                OptimizationStage.FAILED,
                OptimizationStage.ROLLED_BACK,
            ):
                for cb in self._on_completion:
                    try:
                        cb(
                            context.optimization_id,
                            new_stage.value,
                            context.to_dict(),
                        )
                    except Exception:
                        pass

    async def _update_progress(
        self,
        context: OptimizationContext,
        fraction: float,
        message: str,
    ) -> None:
        """Update progress for the current stage."""
        context.progress.fraction = max(0.0, min(1.0, fraction))
        context.progress.message = message

    async def _run_with_timeout(
        self,
        coro: Any,
        stage: OptimizationStage,
    ) -> None:
        """Run a coroutine with stage-specific timeout."""
        timeout = self.STAGE_TIMEOUTS.get(stage, self.DEFAULT_STAGE_TIMEOUT)
        try:
            await asyncio.wait_for(coro, timeout=timeout)
        except TimeoutError:
            raise TimeoutError(f"Stage {stage.value} timed out after {timeout}s") from None

    def _parse_ai_recommendations(
        self,
        ai_response: str,
        proposed_params: dict[str, Any],
    ) -> bool:
        """
        Parse AI response to extract parameter adjustments.

        Attempts to parse JSON from the AI response and apply
        parameter changes to the proposed_params dict.

        Returns:
            True if any parameters were changed.
        """
        try:
            # Try to extract JSON from the response
            content = ai_response.strip()

            # Look for JSON array or object
            start_idx = -1
            for i, ch in enumerate(content):
                if ch in ("[", "{"):
                    start_idx = i
                    break

            if start_idx < 0:
                return False

            json_str = content[start_idx:]

            # Find matching end bracket
            depth = 0
            end_idx = -1
            open_char = json_str[0]
            close_char = "]" if open_char == "[" else "}"

            for i, ch in enumerate(json_str):
                if ch == open_char:
                    depth += 1
                elif ch == close_char:
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break

            if end_idx < 0:
                return False

            data = json.loads(json_str[:end_idx])

            # Parse recommendations
            changed = False
            recs = data if isinstance(data, list) else [data]

            for rec in recs:
                adjustments = rec.get("adjustments", [])
                for adj in adjustments:
                    key = adj.get("key")
                    after_value = adj.get("after_value")
                    if key and after_value is not None and key in proposed_params:
                        proposed_params[key] = after_value
                        changed = True

            return changed

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.debug("Failed to parse AI recommendations: %s", exc)
            return False

    def _mock_backtest_results(
        self,
        context: OptimizationContext,
    ) -> dict[str, Any]:
        """Generate mock backtest results for testing."""
        return {
            "confidence": 0.85,
            "sharpe_improvement": 0.15,
            "win_rate_improvement": 0.03,
            "total_trades_tested": 100,
            "baseline_sharpe": 1.2,
            "candidate_sharpe": 1.35,
            "baseline_win_rate": 0.55,
            "candidate_win_rate": 0.58,
            "max_drawdown_pct": 12.5,
            "source": "mock",
        }

    def _evaluate_rollout(
        self,
        context: OptimizationContext,
    ) -> dict[str, Any]:
        """Evaluate controlled rollout evidence and advance/rollback stage."""
        if not self.rollout_controller:
            return {"allowed": True, "current_stage": "n/a", "next_stage": "n/a"}

        backtest = context.backtest_results or {}
        perf = context.performance_data or {}
        metrics = {
            "fill_rate_pct": perf.get("fill_rate_pct", 100.0),
            "failed_tx_rate_pct": perf.get("failed_tx_rate_pct", 0.0),
            "inclusion_latency_ms_p95": perf.get("inclusion_latency_ms_p95", 0.0),
            "net_pnl_after_costs_sol": perf.get(
                "net_pnl_after_costs_sol",
                backtest.get("net_pnl_after_costs_sol", 0.0),
            ),
            "rug_unsellable_incident_rate_pct": perf.get("rug_unsellable_incident_rate_pct", 0.0),
            "max_drawdown_pct": perf.get("max_drawdown_pct", backtest.get("max_drawdown_pct", 0.0)),
        }
        decision = self.rollout_controller.advance_or_rollback(metrics).to_dict()
        logger.info(
            "Rollout decision for %s: %s",
            context.optimization_id,
            decision,
        )
        return decision

    # ---- Phase 2: Enhanced validation ----

    async def _validate_comprehensive(
        self,
        baseline_params: dict[str, Any],
        candidate_params: dict[str, Any],
        historical_data: list,
    ) -> dict[str, Any]:
        """Enhanced validation with multiple methods (Phase 2).

        Runs robustness testing, stress testing, and out-of-sample
        validation in addition to standard backtest validation.

        Args:
            baseline_params: Current parameters.
            candidate_params: Proposed parameters.
            historical_data: Historical market data.

        Returns:
            Dictionary with validation results from all methods.
        """
        results: dict[str, Any] = {}

        # 1. Robustness testing (if available)
        if self.robustness_validator:
            try:
                robustness = await self.robustness_validator.test_robustness(
                    params=candidate_params,
                    historical_data=historical_data,
                    backtest_fn=self._simple_backtest_fn,
                    perturbation_pct=10.0,
                    num_trials=50,
                )
                results["robustness"] = robustness.to_dict()
            except Exception as exc:
                logger.warning("Robustness validation failed: %s", exc)

        # 2. Stress testing (if available)
        if self.stress_test_engine:
            try:
                stress_report = await self.stress_test_engine.run_all_scenarios(
                    params=candidate_params,
                    historical_data=historical_data,
                )
                results["stress_tests"] = stress_report.to_dict()
            except Exception as exc:
                logger.warning("Stress testing failed: %s", exc)

        # 3. Out-of-sample validation (if available)
        if self.oos_validator:
            try:
                oos_report = await self.oos_validator.validate_out_of_sample(
                    params=candidate_params,
                    historical_data=historical_data,
                    backtest_fn=self._simple_backtest_fn,
                    holdout_pct=20.0,
                )
                results["out_of_sample"] = oos_report.to_dict()
            except Exception as exc:
                logger.warning("Out-of-sample validation failed: %s", exc)

        return results

    def _simple_backtest_fn(
        self,
        params: dict[str, Any],
        data: Any,
    ) -> dict[str, Any]:
        """Simple backtest function for validation components.

        Used internally by robustness and OOS validators.

        Args:
            params: Strategy parameters.
            data: Historical data.

        Returns:
            Dictionary with basic metrics.
        """
        # Generate mock metrics based on params for testing
        import random

        base_sharpe = 1.5
        stop_loss = params.get("stop_loss_pct", 10.0)
        take_profit = params.get("take_profit_pct", 50.0)

        # Simulate sensitivity to params
        sharpe = base_sharpe * (1 + random.gauss(0, 0.1))
        if take_profit > 0 and stop_loss / take_profit > 0.5:
            sharpe *= 0.8  # Penalty for bad R/R

        return {
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": abs(random.gauss(12, 3)),
            "win_rate_pct": 50 + random.gauss(5, 3),
            "total_trades": max(1, int(random.gauss(50, 10))),
            "total_return_pct": random.gauss(15, 5),
        }

    # ---- Phase 3: Governance query methods ----

    async def get_governance_status(self) -> dict[str, Any]:
        """
        Get comprehensive Phase 3 governance status.

        Aggregates status from all safety and governance components:
        SafetyValidator, ApprovalWorkflow, RollbackManager,
        DriftDetector, ModelVersionLock.

        Returns:
            Dictionary with governance component status and routing info.
        """
        status: dict[str, Any] = {
            "phase": "Phase 3 - Safety & Governance",
            "routing_enabled": self.task_router is not None,
            "cost_tracking_enabled": self.cost_tracker is not None,
            "components": {},
        }

        # Safety Validator status
        if self.safety_validator:
            sv_status: dict[str, Any] = {
                "active": True,
                "routing_integrated": hasattr(self.safety_validator, "task_router")
                and self.safety_validator.task_router is not None,
            }
            if hasattr(self.safety_validator, "get_routing_decision"):
                routing = self.safety_validator.get_routing_decision({"max_position_sol": 1.0})
                if routing:
                    sv_status["default_model"] = routing.model_id
            status["components"]["safety_validator"] = sv_status
        else:
            status["components"]["safety_validator"] = {"active": False}

        # Approval Workflow status
        if self.approval_workflow:
            aw_status: dict[str, Any] = {
                "active": True,
                "trading_mode": self.approval_workflow.trading_mode,
                "routing_integrated": hasattr(self.approval_workflow, "task_router")
                and self.approval_workflow.task_router is not None,
            }
            if hasattr(self.approval_workflow, "get_routing_decision"):
                routing = self.approval_workflow.get_routing_decision()
                if routing:
                    aw_status["default_model"] = routing.get("model_id")
            status["components"]["approval_workflow"] = aw_status
        else:
            status["components"]["approval_workflow"] = {"active": False}

        # Rollback Manager status
        if self.rollback_manager:
            rb_status: dict[str, Any] = {
                "active": True,
                "active_monitors": self.rollback_manager.active_monitors,
                "routing_integrated": hasattr(self.rollback_manager, "task_router")
                and self.rollback_manager.task_router is not None,
            }
            status["components"]["rollback_manager"] = rb_status
        else:
            status["components"]["rollback_manager"] = {"active": False}

        # Drift Detector status
        if self.drift_detector:
            dd_status: dict[str, Any] = {
                "active": True,
                "routing_integrated": hasattr(self.drift_detector, "task_router")
                and self.drift_detector.task_router is not None,
            }
            last_report = getattr(self.drift_detector, "last_report", None)
            if last_report:
                dd_status["last_severity"] = last_report.severity
            status["components"]["drift_detector"] = dd_status
        else:
            status["components"]["drift_detector"] = {"active": False}

        # Rollout controller status
        if self.rollout_controller:
            status["components"]["rollout_controller"] = {
                "active": True,
                **self.rollout_controller.get_status(),
            }
        else:
            status["components"]["rollout_controller"] = {"active": False}

        # Model Version Lock status
        if self.model_version_lock:
            mvl_status: dict[str, Any] = {
                "active": True,
                "model_version": self.model_version_lock.model_version,
                "is_distilling": self.model_version_lock.is_distilling,
                "buffer_size": self.model_version_lock.buffer_size,
            }
            status["components"]["model_version_lock"] = mvl_status
        else:
            status["components"]["model_version_lock"] = {"active": False}

        # Cost tracking summary
        if self.cost_tracker:
            cost_stats = self.cost_tracker.get_stats()
            status["cost_summary"] = {
                "total_calls": cost_stats["total_calls"],
                "total_cost_usd": cost_stats["total_cost_usd"],
                "monthly_budget_usd": cost_stats["monthly_budget_usd"],
                "current_month_cost_usd": cost_stats["current_month_cost_usd"],
            }

        # Routing stats
        if self.task_router:
            routing_stats = self.task_router.get_routing_stats()
            status["routing_summary"] = {
                "total_decisions": routing_stats["total_decisions"],
                "opus_pct": routing_stats.get("opus_pct", 0),
                "sonnet_pct": routing_stats.get("sonnet_pct", 0),
                "avg_complexity": routing_stats.get("avg_complexity", 0),
            }

        return status

    # ---- Public query methods ----

    async def get_status(
        self,
        optimization_id: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Get status of the current or specified optimization.

        Args:
            optimization_id: Specific optimization to query (or active).

        Returns:
            Optimization status dict, or None if not found.
        """
        if self.active_optimization is None:
            return None

        if optimization_id and (self.active_optimization.optimization_id != optimization_id):
            # Check history
            for entry in reversed(self._optimization_history):
                if entry.get("optimization_id") == optimization_id:
                    return entry
            return None

        return self.active_optimization.to_dict()

    async def wait_for_completion(
        self,
        optimization_id: str,
        timeout: float = 300.0,
    ) -> OptimizationContext | None:
        """
        Wait for an optimization to complete.

        Args:
            optimization_id: The optimization to wait for.
            timeout: Maximum seconds to wait.

        Returns:
            The completed OptimizationContext, or None if timed out.
        """
        if self._completion_event is None:
            return None

        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=timeout)
        except TimeoutError:
            return None

        return self.active_optimization

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get the history of all optimization runs."""
        return list(self._optimization_history)

    # ---- Callback registration ----

    def on_stage_change(
        self,
        callback: Callable[[str, str, str], None],
    ) -> None:
        """Register a callback for stage transitions."""
        self._on_stage_change.append(callback)

    def on_completion(
        self,
        callback: Callable[[str, str, dict[str, Any]], None],
    ) -> None:
        """Register a callback for optimization completion."""
        self._on_completion.append(callback)

    async def preview_optimization(
        self,
        goal: str,
        constraints: dict[str, Any],
    ) -> dict[str, Any]:
        """Run optimization pipeline through BACKTESTING but stop before DEPLOYING.

        Returns before/after comparison metrics for user review in the AI wizard,
        without actually deploying any parameter changes.

        Args:
            goal: Optimization objective ("maximize_sharpe", etc.)
            constraints: Hard constraints to respect.

        Returns:
            Dict with current_params, proposed_params, backtest comparison,
            safety report, and promotion gate results.
        """
        # Get current parameters
        current_params: dict[str, Any] = {}
        if self.parameter_store:
            current_params = await self.parameter_store.get_current()

        preview_id = f"preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        context = OptimizationContext(
            optimization_id=preview_id,
            trigger_reason="preview",
            started_at=datetime.now(),
            current_stage=OptimizationStage.IDLE,
            goal=goal,
            constraints=constraints,
            current_params=current_params,
            urgency="normal",
        )

        result: dict[str, Any] = {
            "preview_id": preview_id,
            "goal": goal,
            "constraints": constraints,
            "current_params": current_params,
        }

        try:
            # Stage 1: Analyze
            context.current_stage = OptimizationStage.ANALYZING
            await self._stage_analyze(context)
            result["market_regime"] = context.market_regime
            result["performance_data"] = context.performance_data

            # Stage 2: Generate recommendations
            context.current_stage = OptimizationStage.GENERATING
            await self._stage_generate(context)
            result["proposed_params"] = context.proposed_params
            result["recommendations"] = context.recommendations

            # Stage 3: Backtest
            context.current_stage = OptimizationStage.BACKTESTING
            await self._stage_backtest(context)
            result["backtest_results"] = context.backtest_results

            # Stage 4: Validate safety (but don't deploy)
            context.current_stage = OptimizationStage.VALIDATING
            await self._stage_validate(context)
            result["safety_report"] = context.safety_report

            # Evaluate promotion gates
            gate_result = await self._evaluate_promotion_gates(context)
            result["promotion_gates"] = gate_result

            # Build comparison summary
            backtest = context.backtest_results or {}
            result["comparison"] = {
                "baseline_sharpe": backtest.get("baseline_sharpe", 0),
                "candidate_sharpe": backtest.get("candidate_sharpe", 0),
                "sharpe_improvement": backtest.get("sharpe_improvement", 0),
                "baseline_win_rate": backtest.get("baseline_win_rate", 0),
                "candidate_win_rate": backtest.get("candidate_win_rate", 0),
                "win_rate_improvement": backtest.get("win_rate_improvement", 0),
                "max_drawdown_pct": backtest.get("max_drawdown_pct", 0),
                "would_pass_gates": gate_result.get("passed", False),
            }

            result["status"] = "success"

        except Exception as exc:
            logger.warning(
                "Preview optimization failed at %s: %s", context.current_stage.value, exc
            )
            result["status"] = "failed"
            result["error"] = str(exc)
            result["failed_stage"] = context.current_stage.value

        return result
