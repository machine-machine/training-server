"""
GLM Model Selector - Dynamic GLM model selection for 2DEXY trading platform.

Routes AI tasks to appropriate GLM models based on latency requirements,
task complexity, and cost considerations.

Model Tiers:
  - GLM-4-Flash: <50ms, high-volume, simple tasks (hot path)
  - GLM-4-Air:   50-100ms, balanced performance (default)
  - GLM-4-Plus:  200-500ms, complex reasoning (cold path analysis)
  - GLM-4-Long:  200-400ms, 1M context (documentation, reports)

Integration Points:
  - HotPath (Rust): Real-time trading decisions -> GLM-4-Flash
  - ColdPath (Python): ML training, analysis -> GLM-4-Plus
  - 2DEXY App (SwiftUI): User interactions -> GLM-4-Air
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class GLMModel(Enum):
    """Available GLM model variants."""

    FLASH = "glm-4-flash"  # <50ms, high volume, hot path
    AIR = "glm-4-air"  # 50-100ms, balanced, default
    PLUS = "glm-4-plus"  # 200-500ms, complex reasoning
    LONG = "glm-4-long"  # 1M context, long documents


class TaskCategory(Enum):
    """Categories of tasks for model routing."""

    # Hot path - real-time trading
    TRADING_SIGNAL = "trading_signal"
    RISK_ASSESSMENT = "risk_assessment"
    ALERT_FILTERING = "alert_filtering"

    # Cold path - batch processing
    FEATURE_ENGINEERING = "feature_engineering"
    BACKTEST_ANALYSIS = "backtest_analysis"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"

    # User interaction
    USER_QUERY = "user_query"
    DASHBOARD_EXPLANATION = "dashboard_explanation"
    ERROR_EXPLANATION = "error_explanation"

    # Documentation
    REPORT_GENERATION = "report_generation"
    REGULATORY_DOCS = "regulatory_docs"


@dataclass
class ModelCapabilities:
    """Capabilities and characteristics of a GLM model."""

    model_id: str
    max_context_tokens: int
    typical_latency_ms: int
    relative_cost: float  # 1.0 = base cost

    # Capabilities
    supports_vision: bool = False
    supports_tools: bool = True
    supports_json: bool = True


# Model capability registry
GLM_CAPABILITIES: dict[GLMModel, ModelCapabilities] = {
    GLMModel.FLASH: ModelCapabilities(
        model_id="glm-4-flash",
        max_context_tokens=128_000,
        typical_latency_ms=30,
        relative_cost=1.0,
    ),
    GLMModel.AIR: ModelCapabilities(
        model_id="glm-4-air",
        max_context_tokens=128_000,
        typical_latency_ms=75,
        relative_cost=4.0,
    ),
    GLMModel.PLUS: ModelCapabilities(
        model_id="glm-4-plus",
        max_context_tokens=128_000,
        typical_latency_ms=350,
        relative_cost=20.0,
    ),
    GLMModel.LONG: ModelCapabilities(
        model_id="glm-4-long",
        max_context_tokens=1_000_000,
        typical_latency_ms=300,
        relative_cost=15.0,
    ),
}


@dataclass
class GLMRoutingDecision:
    """Result of model routing decision."""

    model: GLMModel
    model_id: str
    max_tokens: int
    temperature: float
    timeout_seconds: int
    estimated_latency_ms: int
    estimated_cost: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model": self.model.value,
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_cost": self.estimated_cost,
            "reasoning": self.reasoning,
        }


@dataclass
class GLMModelSelectorConfig:
    """Configuration for GLM model selector."""

    # Default model for unspecified tasks
    default_model: GLMModel = GLMModel.AIR

    # Latency thresholds (ms) for auto-routing
    low_latency_threshold: int = 100  # Use Flash if needed faster
    high_latency_threshold: int = 300  # Use Plus if slower is OK

    # Cost optimization
    cost_optimization_enabled: bool = True
    max_cost_per_call: float = 0.10  # Maximum cost in USD per call

    # Fallback behavior
    fallback_model: GLMModel = GLMModel.AIR

    # Task-specific overrides (from environment)
    task_overrides: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "GLMModelSelectorConfig":
        """Create configuration from environment variables."""
        overrides = {}

        # Check for task-specific model overrides
        for task in TaskCategory:
            env_key = f"GLM_MODEL_{task.value.upper()}"
            if os.environ.get(env_key):
                overrides[task.value] = os.environ[env_key]

        return cls(
            default_model=GLMModel(os.environ.get("GLM_DEFAULT_MODEL", "glm-4-air")),
            cost_optimization_enabled=os.environ.get("GLM_COST_OPTIMIZATION", "true").lower()
            == "true",
            max_cost_per_call=float(os.environ.get("GLM_MAX_COST_PER_CALL", "0.10")),
            task_overrides=overrides,
        )


class GLMModelSelector:
    """
    Selects appropriate GLM model based on task requirements.

    Usage:
        selector = GLMModelSelector()
        decision = selector.select(TaskCategory.TRADING_SIGNAL)
        print(decision.model_id)  # "glm-4-flash"
    """

    # Task category to recommended model mapping
    TASK_MODEL_MAP: dict[TaskCategory, GLMModel] = {
        # Hot path - minimize latency
        TaskCategory.TRADING_SIGNAL: GLMModel.FLASH,
        TaskCategory.RISK_ASSESSMENT: GLMModel.AIR,  # Needs accuracy
        TaskCategory.ALERT_FILTERING: GLMModel.FLASH,
        # Cold path - prioritize quality
        TaskCategory.FEATURE_ENGINEERING: GLMModel.PLUS,
        TaskCategory.BACKTEST_ANALYSIS: GLMModel.AIR,
        TaskCategory.HYPERPARAMETER_TUNING: GLMModel.AIR,
        # User interaction - balance speed/quality
        TaskCategory.USER_QUERY: GLMModel.AIR,
        TaskCategory.DASHBOARD_EXPLANATION: GLMModel.AIR,
        TaskCategory.ERROR_EXPLANATION: GLMModel.FLASH,
        # Documentation - long context
        TaskCategory.REPORT_GENERATION: GLMModel.LONG,
        TaskCategory.REGULATORY_DOCS: GLMModel.LONG,
    }

    # Temperature settings by task type
    TASK_TEMPERATURE_MAP: dict[TaskCategory, float] = {
        # Hot path - deterministic
        TaskCategory.TRADING_SIGNAL: 0.0,
        TaskCategory.RISK_ASSESSMENT: 0.1,
        TaskCategory.ALERT_FILTERING: 0.0,
        # Cold path - slight variation OK
        TaskCategory.FEATURE_ENGINEERING: 0.3,
        TaskCategory.BACKTEST_ANALYSIS: 0.2,
        TaskCategory.HYPERPARAMETER_TUNING: 0.3,
        # User interaction - natural
        TaskCategory.USER_QUERY: 0.5,
        TaskCategory.DASHBOARD_EXPLANATION: 0.4,
        TaskCategory.ERROR_EXPLANATION: 0.2,
        # Documentation - consistent
        TaskCategory.REPORT_GENERATION: 0.3,
        TaskCategory.REGULATORY_DOCS: 0.1,
    }

    def __init__(self, config: GLMModelSelectorConfig | None = None):
        self.config = config or GLMModelSelectorConfig.from_env()

    def select(
        self,
        task: TaskCategory,
        max_latency_ms: int | None = None,
        require_long_context: bool = False,
        estimated_tokens: int = 1000,
    ) -> GLMRoutingDecision:
        """
        Select the best GLM model for a given task.

        Args:
            task: Category of task to perform
            max_latency_ms: Maximum acceptable latency (None = no constraint)
            require_long_context: Whether >128K context is needed
            estimated_tokens: Estimated tokens for cost calculation

        Returns:
            GLMRoutingDecision with selected model and parameters
        """
        # Check for task-specific override
        if task.value in self.config.task_overrides:
            override_model = self.config.task_overrides[task.value]
            try:
                model = GLMModel(override_model)
            except ValueError:
                logger.warning(f"Invalid model override: {override_model}")
                model = self.TASK_MODEL_MAP.get(task, self.config.default_model)
        else:
            model = self.TASK_MODEL_MAP.get(task, self.config.default_model)

        # Handle long context requirement
        if require_long_context:
            model = GLMModel.LONG

        # Handle latency constraint
        if max_latency_ms is not None:
            capabilities = GLM_CAPABILITIES[model]
            if capabilities.typical_latency_ms > max_latency_ms:
                # Find fastest model that meets requirement
                for candidate in [GLMModel.FLASH, GLMModel.AIR, GLMModel.PLUS]:
                    if GLM_CAPABILITIES[candidate].typical_latency_ms <= max_latency_ms:
                        model = candidate
                        break

        capabilities = GLM_CAPABILITIES[model]
        temperature = self.TASK_TEMPERATURE_MAP.get(task, 0.5)

        # Calculate estimated cost (simplified)
        # Assuming ~$0.01 per 1K tokens for Flash, scaled by relative cost
        base_cost_per_1k = 0.01
        estimated_cost = (estimated_tokens / 1000) * base_cost_per_1k * capabilities.relative_cost

        # Check cost constraint
        if self.config.cost_optimization_enabled and estimated_cost > self.config.max_cost_per_call:
            # Downgrade to cheaper model if possible
            for candidate in [GLMModel.FLASH, GLMModel.AIR]:
                candidate_cost = (
                    (estimated_tokens / 1000)
                    * base_cost_per_1k
                    * GLM_CAPABILITIES[candidate].relative_cost
                )
                if candidate_cost <= self.config.max_cost_per_call:
                    if candidate.value != model.value:
                        logger.info(
                            f"Downgraded from {model.value} to "
                            f"{candidate.value} due to cost constraint"
                        )
                    model = candidate
                    capabilities = GLM_CAPABILITIES[model]
                    estimated_cost = candidate_cost
                    break

        return GLMRoutingDecision(
            model=model,
            model_id=capabilities.model_id,
            max_tokens=min(estimated_tokens * 2, capabilities.max_context_tokens),
            temperature=temperature,
            timeout_seconds=max(30, capabilities.typical_latency_ms * 3 // 1000),
            estimated_latency_ms=capabilities.typical_latency_ms,
            estimated_cost=estimated_cost,
            reasoning=(
                f"Selected {model.value} for {task.value} "
                f"(latency: {max_latency_ms}ms, long_context: {require_long_context})"
            ),
        )

    def get_model_for_hot_path(self) -> str:
        """Get recommended model for HotPath real-time operations."""
        return GLMModel.FLASH.value

    def get_model_for_cold_path(self) -> str:
        """Get recommended model for ColdPath batch processing."""
        return GLMModel.PLUS.value

    def get_model_for_user_interaction(self) -> str:
        """Get recommended model for user-facing operations."""
        return GLMModel.AIR.value

    def get_capabilities(self, model: GLMModel) -> ModelCapabilities:
        """Get capabilities for a specific model."""
        return GLM_CAPABILITIES[model]


# Convenience functions for common use cases
def select_model_for_trading() -> str:
    """Quick access to trading signal model."""
    return GLMModel.FLASH.value


def select_model_for_analysis() -> str:
    """Quick access to analysis model."""
    return GLMModel.PLUS.value


def select_model_for_user() -> str:
    """Quick access to user interaction model."""
    return GLMModel.AIR.value
