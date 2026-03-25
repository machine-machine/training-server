"""
AI module for Claude integration, LLM orchestration, and parameter optimization.

Components:
- claude_client: Base Claude API client
- claude_router: Opus/Sonnet task routing
- glm_model_selector: GLM model selection for trading tasks
- llm_orchestrator: Hybrid LLM + ML orchestration
- kv_cache_manager: Token recaching for efficiency
- parameter_tools: Strategy parameter optimization
- resource_guard: System resource management
- strategy_orchestrator: AI optimization pipeline state machine
- parameter_store: Versioned parameter management
- model_version_lock: Online learning / distillation coordination
- performance_engine: Trade outcome tracking and learning
- drift_detector: Statistical performance drift detection
- parameter_tracker: Regime-specific parameter effectiveness
- safety_validator: Multi-layer deployment safety checks
- approval_workflow: Auto-apply / manual approval logic
- rollback_manager: Post-deployment monitoring and auto-rollback
"""

from .approval_workflow import (
    ApprovalDecision,
    ApprovalThresholds,
    ApprovalWorkflow,
)
from .claude_client import ClaudeClient
from .claude_router import ClaudeRouter, TaskType
from .cost_tracker import APICallCost, CostTracker
from .drift_detector import DriftDetector, DriftReport
from .glm_model_selector import (
    GLMModel,
    GLMModelSelector,
    GLMModelSelectorConfig,
    GLMRoutingDecision,
    ModelCapabilities,
    TaskCategory,
    select_model_for_analysis,
    select_model_for_trading,
    select_model_for_user,
)
from .interaction_analyzer import (
    InteractionReport,
    ParameterInteraction,
    ParameterInteractionAnalyzer,
)
from .kv_cache_manager import (
    CacheEntry,
    CacheLayer,
    CacheStats,
    KVCacheManager,
    SemanticDeduplicator,
)
from .llm_orchestrator import (
    HybridDecision,
    LLMOrchestrator,
    MLContext,
    OrchestratorConfig,
    PromptDetailLevel,
)
from .model_version_lock import BufferedUpdate, ModelVersionLock
from .multi_objective_optimizer import (
    MultiObjectiveOptimizer,
    Objective,
    OptimizationProgress,
    ParetoSolution,
)
from .parameter_store import (
    ParameterStore,
    PerformanceBaseline,
    VersionedStrategyParams,
)
from .parameter_tools import (
    MarketRegime,
    ParameterTools,
    QuickBacktestResult,
    UserStrategyParams,
    analyze_performance,
    run_quick_backtest,
    suggest_regime_params,
)
from .parameter_tracker import (
    ParameterEffectivenessTracker,
    ParameterRecommendation,
    ParameterScore,
)
from .performance_engine import (
    FeedbackResult,
    PerformanceFeedbackEngine,
    TradeOutcome,
)
from .prompt_optimizer import PromptBudget, PromptOptimizer
from .resource_guard import ResourceGuard, ResourceLimits
from .rollback_manager import RollbackManager, RollbackThresholds
from .rollout_controller import (
    EvidenceGates,
    RolloutController,
    RolloutDecision,
    RolloutStage,
)
from .safety_validator import SafetyReport, SafetyValidator
from .strategy_orchestrator import (
    AIStrategyOrchestrator,
    OptimizationContext,
    OptimizationStage,
)

# Phase 2: Advanced AI Integration
from .task_complexity_router import (
    ModelRoutingDecision,
    TaskComplexity,
    TaskComplexityRouter,
)

__all__ = [
    # Claude client
    "ClaudeClient",
    "ClaudeRouter",
    "TaskType",
    # LLM Orchestrator
    "LLMOrchestrator",
    "OrchestratorConfig",
    "MLContext",
    "HybridDecision",
    "PromptDetailLevel",
    # KV Cache
    "KVCacheManager",
    "CacheEntry",
    "CacheStats",
    "CacheLayer",
    "SemanticDeduplicator",
    # Parameter tools
    "ParameterTools",
    "UserStrategyParams",
    "QuickBacktestResult",
    "MarketRegime",
    "run_quick_backtest",
    "suggest_regime_params",
    "analyze_performance",
    # Resource management
    "ResourceGuard",
    "ResourceLimits",
    # Strategy orchestrator
    "AIStrategyOrchestrator",
    "OptimizationStage",
    "OptimizationContext",
    # Parameter store
    "ParameterStore",
    "VersionedStrategyParams",
    "PerformanceBaseline",
    # Model version lock
    "ModelVersionLock",
    "BufferedUpdate",
    # Performance engine
    "PerformanceFeedbackEngine",
    "TradeOutcome",
    "FeedbackResult",
    # Drift detection
    "DriftDetector",
    "DriftReport",
    # Parameter tracker
    "ParameterEffectivenessTracker",
    "ParameterScore",
    "ParameterRecommendation",
    # Safety validation
    "SafetyValidator",
    "SafetyReport",
    # Approval workflow
    "ApprovalWorkflow",
    "ApprovalDecision",
    "ApprovalThresholds",
    # Rollback management
    "RollbackManager",
    "RollbackThresholds",
    # Controlled rollout
    "RolloutController",
    "RolloutStage",
    "EvidenceGates",
    "RolloutDecision",
    # Phase 2: Advanced AI Integration
    "TaskComplexityRouter",
    "TaskComplexity",
    "ModelRoutingDecision",
    "CostTracker",
    "APICallCost",
    "MultiObjectiveOptimizer",
    "Objective",
    "ParetoSolution",
    "OptimizationProgress",
    "PromptOptimizer",
    "PromptBudget",
    "ParameterInteractionAnalyzer",
    "ParameterInteraction",
    "InteractionReport",
    # GLM Model Selection
    "GLMModel",
    "TaskCategory",
    "GLMModelSelector",
    "GLMModelSelectorConfig",
    "GLMRoutingDecision",
    "ModelCapabilities",
    "select_model_for_trading",
    "select_model_for_analysis",
    "select_model_for_user",
]
