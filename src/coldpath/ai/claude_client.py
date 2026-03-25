"""
Anthropic Claude API client wrapper.

Provides a unified interface for communicating with Claude models.
Integrates with ServerTuner for dynamic configuration adjustments.
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Claude model tiers for routing."""

    OPUS = "opus"
    SONNET = "sonnet"


@dataclass
class ClaudeResponse:
    """Response from Claude API."""

    content: str
    model_used: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    latency_ms: float
    cache_hit: bool = False
    retry_count: int = 0


@dataclass
class ClientStats:
    """Statistics for the Claude client."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    opus_requests: int = 0
    sonnet_requests: int = 0
    fallback_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    retry_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.successful_requests)

    @property
    def success_rate(self) -> float:
        return self.successful_requests / max(1, self.total_requests)

    @property
    def opus_ratio(self) -> float:
        return self.opus_requests / max(1, self.total_requests)


class ClaudeClient:
    """Wrapper for Anthropic Claude API.

    Supports both Opus 4.6 and Sonnet 4.5 models with automatic
    fallback and error handling.

    Integrates with ServerTuner for dynamic configuration:
    - Temperature adjustments
    - Timeout adjustments
    - Retry configuration

    Prompt Caching Strategy (optimized for cost & latency):
    - System prompts use cache_control: {"type": "ephemeral"} (5-min TTL)
    - Multi-turn conversations reuse cached system prompts (90% cheaper)
    - Strategy analysis shares a SINGLE cached system prompt across calls
    - Backtest analysis batches use the same cached context
    """

    DEFAULT_OPUS_MODEL = "claude-opus-4-6-20250514"
    DEFAULT_SONNET_MODEL = "claude-sonnet-4-5-20250929"

    # Prompt caching: system prompts >1024 tokens qualify for caching
    # Cache lives for 5 minutes, 90% cheaper on cache reads
    ENABLE_PROMPT_CACHING = True

    # Shared system prompt templates (cached across calls for massive savings)
    _STRATEGY_SYSTEM_PROMPT = """You are an expert quantitative trading strategist specializing in
Solana memecoin markets. You analyze trading strategy parameters using these frameworks:

ANALYSIS FRAMEWORK:
1. Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
2. Drawdown management and recovery patterns
3. Position sizing optimization (Kelly Criterion variants)
4. Entry/exit timing with regime-aware thresholds
5. Memecoin-specific risks: rug pulls, MEV, liquidity dynamics

MARKET CONTEXT:
- Solana DEX trading (Raydium, Jupiter, Orca)
- Sub-second execution via Jito bundles
- 4-model ensemble: Isolation Forest (rug), LSTM (price), XGBoost (signal), Kelly (sizing)
- Regime-adaptive thresholds across 6 market states

OUTPUT FORMAT: Provide specific, actionable parameter recommendations with expected
impact ranges. Format as structured JSON when requested. Focus on improving:
win rate, profit factor, Sharpe ratio, and reducing max drawdown."""

    _BACKTEST_SYSTEM_PROMPT = """You are an expert at analyzing trading backtest results for
Solana memecoin strategies. Compare baseline vs candidate strategies focusing on:

1. Key metric improvements and degradations (win rate, PF, Sharpe, drawdown)
2. Risk/return trade-offs and regime-specific performance
3. Statistical significance of differences (min 30 trades for reliable comparison)
4. Actionable recommendations for further optimization

Be concise but thorough. Focus on whether changes improve risk-adjusted returns."""

    def __init__(
        self,
        api_key: str | None = None,
        opus_model: str | None = None,
        sonnet_model: str | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        tuner_integration: bool = True,
    ):
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
            opus_model: Override Opus model ID.
            sonnet_model: Override Sonnet model ID.
            max_retries: Number of retries on transient failures.
            timeout: Request timeout in seconds.
            tuner_integration: Enable integration with ServerTuner.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.opus_model = opus_model or os.environ.get("CLAUDE_OPUS_MODEL", self.DEFAULT_OPUS_MODEL)
        self.sonnet_model = sonnet_model or os.environ.get(
            "CLAUDE_SONNET_MODEL", self.DEFAULT_SONNET_MODEL
        )
        self.max_retries = max_retries
        self.timeout = timeout
        self.tuner_integration = tuner_integration

        self._client = None
        self._stats = ClientStats()
        self._on_request_complete: list[Callable[[ClaudeResponse], None]] = []
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Anthropic async client."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        try:
            import anthropic

            self._client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            logger.info("Claude async client initialized successfully")
        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
            raise

    def _get_model_id(self, tier: ModelTier) -> str:
        """Get model ID for the specified tier."""
        if tier == ModelTier.OPUS:
            return self.opus_model
        return self.sonnet_model

    def _get_tuner_config(self):
        """Get configuration from ServerTuner if available."""
        if not self.tuner_integration:
            return None
        try:
            from coldpath.ai.server_tuner import get_server_tuner

            tuner = get_server_tuner()
            return tuner.get_config() if tuner else None
        except Exception:
            return None

    async def chat(
        self,
        message: str,
        tier: ModelTier = ModelTier.SONNET,
        system_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> ClaudeResponse:
        """Send a chat message to Claude.

        Args:
            message: The user message to send.
            tier: Model tier to use (Opus for complex, Sonnet for simple).
            system_prompt: Optional system prompt.
            conversation_history: Previous messages for context.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature (uses tuner config if None).

        Returns:
            ClaudeResponse with the model's reply.
        """
        import time

        model_id = self._get_model_id(tier)
        self._stats.total_requests += 1

        # Get temperature from tuner if not specified
        if temperature is None:
            tuner_config = self._get_tuner_config()
            if tuner_config:
                if tier == ModelTier.OPUS:
                    temperature = tuner_config.opus_temperature.current_value
                else:
                    temperature = tuner_config.sonnet_temperature.current_value
            else:
                temperature = 0.7 if tier == ModelTier.SONNET else 0.5

        # Build messages
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": message})

        start_time = time.time()
        retry_count = 0

        try:
            # Build system prompt with prompt caching for token savings
            sys_text = system_prompt or self._default_system_prompt()
            system_with_cache = [
                {
                    "type": "text",
                    "text": sys_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

            response = await self._client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_with_cache,
                messages=messages,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Update stats
            self._stats.successful_requests += 1
            self._stats.total_input_tokens += response.usage.input_tokens
            self._stats.total_output_tokens += response.usage.output_tokens
            self._stats.total_latency_ms += latency_ms
            if tier == ModelTier.OPUS:
                self._stats.opus_requests += 1
            else:
                self._stats.sonnet_requests += 1

            # Detect prompt cache hit from Anthropic response
            getattr(response.usage, "cache_creation_input_tokens", 0)
            cache_read = getattr(response.usage, "cache_read_input_tokens", 0)
            is_cache_hit = cache_read > 0

            result = ClaudeResponse(
                content=response.content[0].text,
                model_used=model_id,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                stop_reason=response.stop_reason,
                latency_ms=latency_ms,
                cache_hit=is_cache_hit,
                retry_count=retry_count,
            )

            # Notify listeners
            for callback in self._on_request_complete:
                try:
                    callback(result)
                except Exception as e:
                    logger.warning(f"Request callback error: {e}")

            return result

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            self._stats.failed_requests += 1

            # Try fallback to Sonnet if Opus fails
            if tier == ModelTier.OPUS:
                logger.info("Falling back to Sonnet model")
                self._stats.fallback_count += 1
                return await self.chat(
                    message=message,
                    tier=ModelTier.SONNET,
                    system_prompt=system_prompt,
                    conversation_history=conversation_history,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            raise

    async def analyze_strategy(
        self,
        strategy_params: dict[str, Any],
        objective: str,
        constraints: dict[str, Any],
    ) -> ClaudeResponse:
        """Analyze trading strategy and provide recommendations.

        Uses Opus for deep reasoning about strategy trade-offs.
        System prompt is cached across calls for 90% cost savings.

        Args:
            strategy_params: Current strategy parameters.
            objective: Optimization objective (e.g., "reduce drawdown").
            constraints: Hard constraints to respect.

        Returns:
            ClaudeResponse with analysis and recommendations.
        """
        message = f"""Analyze this trading strategy:

**Current Parameters:**
{self._format_params(strategy_params)}

**Optimization Objective:** {objective}

**Constraints:**
{self._format_params(constraints)}

Provide recommendations for parameter adjustments that achieve the objective
while respecting constraints. Include confidence levels and expected impact
for each recommendation."""

        return await self.chat(
            message=message,
            tier=ModelTier.OPUS,  # Use Opus for complex strategy analysis
            system_prompt=self._STRATEGY_SYSTEM_PROMPT,  # Cached system prompt
            max_tokens=4096,
            temperature=0.5,  # Lower temperature for more focused analysis
        )

    async def analyze_strategy_structured(
        self,
        strategy_params: dict[str, Any],
        objective: str,
        constraints: dict[str, Any],
    ) -> ClaudeResponse:
        """Analyze trading strategy and return structured JSON recommendations.

        Uses Opus for deep reasoning with structured JSON output.

        Args:
            strategy_params: Current strategy parameters.
            objective: Optimization objective (e.g., "reduce drawdown").
            constraints: Hard constraints to respect.

        Returns:
            ClaudeResponse with JSON-formatted recommendations.
        """
        import json

        system_prompt = (
            self._STRATEGY_SYSTEM_PROMPT
            + """

OUTPUT FORMAT: Return recommendations as a JSON array with exactly this structure:
[
  {
    "title": "Short recommendation title",
    "rationale": "Brief explanation of why this change helps",
    "confidence": 0.75,
    "adjustments": [
      {
        "key": "parameter_name",
        "label": "Human-readable label",
        "before_value": 8.0,
        "after_value": 6.0,
        "format": "percent"
      }
    ],
    "impacts": [
      {
        "label": "Metric Name",
        "value": "+5%",
        "is_positive": true
      }
    ]
  }
]

Valid formats: "percent", "sol", "usd", "number", "bps". Confidence: 0-1.
IMPORTANT: Return ONLY valid JSON. No markdown outside JSON."""
        )

        message = f"""Analyze and optimize this trading strategy:

Current Parameters:
{json.dumps(strategy_params, indent=2)}

Optimization Objective: {objective}

Constraints:
{json.dumps(constraints, indent=2)}

Return 2-4 recommendations as a JSON array."""

        return await self.chat(
            message=message,
            tier=ModelTier.OPUS,
            system_prompt=system_prompt,
            max_tokens=4096,
            temperature=0.3,  # Lower temperature for structured output
        )

    async def analyze_backtest(
        self,
        baseline_metrics: dict[str, Any],
        candidate_metrics: dict[str, Any],
        parameter_changes: dict[str, Any],
    ) -> ClaudeResponse:
        """Analyze backtest results and explain trade-offs.

        Uses Opus for deep understanding of metric relationships.

        Args:
            baseline_metrics: Metrics from baseline backtest.
            candidate_metrics: Metrics from candidate (modified) backtest.
            parameter_changes: What parameters were changed.

        Returns:
            ClaudeResponse with analysis.
        """
        system_prompt = self._BACKTEST_SYSTEM_PROMPT  # Cached across calls

        message = f"""Compare these backtest results:

**Parameter Changes:**
{self._format_params(parameter_changes)}

**Baseline Metrics:**
{self._format_params(baseline_metrics)}

**Candidate Metrics:**
{self._format_params(candidate_metrics)}

Analyze the trade-offs and provide a recommendation on whether to accept
the candidate strategy."""

        return await self.chat(
            message=message,
            tier=ModelTier.OPUS,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.5,
        )

    async def explain_risk(
        self,
        risk_metrics: dict[str, Any],
        context: str | None = None,
    ) -> ClaudeResponse:
        """Explain current risk status to user.

        Uses Sonnet for fast, clear explanations.

        Args:
            risk_metrics: Current risk metrics.
            context: Additional context about user's question.

        Returns:
            ClaudeResponse with explanation.
        """
        system_prompt = """You are a helpful trading assistant explaining risk to users.
Keep explanations clear, concise, and actionable. Use simple language but maintain
accuracy. Format with markdown for readability."""

        message = f"""Explain the current risk status:

**Risk Metrics:**
{self._format_params(risk_metrics)}

{f"**User Context:** {context}" if context else ""}

Provide a clear summary of the risk level and any recommended actions."""

        return await self.chat(
            message=message,
            tier=ModelTier.SONNET,  # Use Sonnet for quick explanations
            system_prompt=system_prompt,
            max_tokens=1024,
            temperature=0.7,
        )

    def _default_system_prompt(self) -> str:
        """Default system prompt for general queries."""
        return """You are an AI trading assistant for a cryptocurrency trading platform.
You help users understand their trading settings, risk levels, and performance.
Be helpful, accurate, and concise. Use markdown formatting for clarity.
When discussing trades or strategies, always emphasize risk management."""

    def _format_params(self, params: dict[str, Any]) -> str:
        """Format parameters for display in prompts."""
        lines = []
        for key, value in params.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    @property
    def is_available(self) -> bool:
        """Check if the client is properly configured."""
        return self._client is not None and self.api_key is not None

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "total_requests": self._stats.total_requests,
            "successful_requests": self._stats.successful_requests,
            "failed_requests": self._stats.failed_requests,
            "success_rate": self._stats.success_rate,
            "opus_requests": self._stats.opus_requests,
            "sonnet_requests": self._stats.sonnet_requests,
            "opus_ratio": self._stats.opus_ratio,
            "fallback_count": self._stats.fallback_count,
            "total_input_tokens": self._stats.total_input_tokens,
            "total_output_tokens": self._stats.total_output_tokens,
            "avg_latency_ms": self._stats.avg_latency_ms,
            "retry_count": self._stats.retry_count,
        }

    def reset_stats(self):
        """Reset client statistics."""
        self._stats = ClientStats()

    def on_request_complete(self, callback: Callable[[ClaudeResponse], None]):
        """Register callback for completed requests."""
        self._on_request_complete.append(callback)

    async def close(self):
        """Close the Anthropic client and release resources."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as e:
                logger.warning(f"Error closing Claude client: {e}")
            self._client = None

    async def analyze_strategy_with_routing(
        self,
        task_type: str,
        current_params: dict,
        performance_data: dict,
        market_regime: str,
        goal: str,
        constraints: dict,
        router: Any = None,
        cost_tracker: Any = None,
        optimization_id: str = "",
    ) -> ClaudeResponse:
        """
        Analyze strategy with automatic model routing and cost tracking.

        Uses TaskComplexityRouter to select the optimal model and
        tracks costs via CostTracker.

        Args:
            task_type: Type of analysis task.
            current_params: Current strategy parameters.
            performance_data: Recent performance data.
            market_regime: Current market regime string.
            goal: Optimization goal.
            constraints: Hard constraints.
            router: TaskComplexityRouter instance (optional).
            cost_tracker: CostTracker instance (optional).
            optimization_id: Optimization run ID for cost tracking.

        Returns:
            ClaudeResponse with analysis results.
        """
        import time as _time

        # Determine routing
        if router:
            routing = router.route_task(
                task_type=task_type,
                num_parameters=len(current_params),
                has_constraints=bool(constraints),
                requires_explanation=True,
                is_high_stakes=("risk" in goal.lower() or "safety" in goal.lower()),
            )
            model_id = routing.model_id
            max_tokens = routing.max_tokens
            temperature = routing.temperature
            tier = ModelTier.OPUS if "opus" in model_id.lower() else ModelTier.SONNET
        else:
            tier = ModelTier.OPUS
            model_id = self.opus_model
            max_tokens = 4096
            temperature = 0.5

        # Build prompt (use PromptOptimizer if available)
        try:
            from coldpath.ai.prompt_optimizer import PromptOptimizer

            optimizer = PromptOptimizer()
            message = optimizer.build_strategy_analysis_prompt(
                current_params=current_params,
                performance_data=performance_data,
                regime=market_regime,
                goal=goal,
                constraints=constraints,
                max_tokens=max_tokens,
            )
            system_prompt = optimizer.build_system_prompt(task_type)
        except ImportError:
            message = f"""Analyze this trading strategy for {goal}:
Parameters: {self._format_params(current_params)}
Performance: {self._format_params(performance_data)}
Regime: {market_regime}
Constraints: {self._format_params(constraints)}"""
            system_prompt = None

        # Make API call
        start = _time.time()
        response = await self.chat(
            message=message,
            tier=tier,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = int((_time.time() - start) * 1000)

        # Track cost
        if cost_tracker and optimization_id:
            await cost_tracker.record_api_call(
                optimization_id=optimization_id,
                model=response.model_used,
                task_type=task_type,
                tokens_input=response.input_tokens,
                tokens_output=response.output_tokens,
                latency_ms=latency_ms,
                cache_hit=response.cache_hit,
            )

        return response

    async def stream_message(
        self,
        message: str,
        tier: ModelTier = ModelTier.SONNET,
        system_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ):
        """Stream a chat message from Claude, yielding text chunks.

        Async generator that yields text deltas as they arrive.
        Ideal for real-time display in the UI.

        Yields:
            str: Text chunks as they arrive from the API
        """
        import time

        model_id = self._get_model_id(tier)
        self._stats.total_requests += 1

        if temperature is None:
            tuner_config = self._get_tuner_config()
            if tuner_config:
                if tier == ModelTier.OPUS:
                    temperature = tuner_config.opus_temperature.current_value
                else:
                    temperature = tuner_config.sonnet_temperature.current_value
            else:
                temperature = 0.7 if tier == ModelTier.SONNET else 0.5

        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": message})

        sys_text = system_prompt or self._default_system_prompt()
        system_with_cache = [
            {
                "type": "text",
                "text": sys_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        start_time = time.time()

        try:
            async with self._client.messages.stream(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_with_cache,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

                # Get final message for stats
                response = await stream.get_final_message()
                latency_ms = (time.time() - start_time) * 1000

                self._stats.successful_requests += 1
                self._stats.total_input_tokens += response.usage.input_tokens
                self._stats.total_output_tokens += response.usage.output_tokens
                self._stats.total_latency_ms += latency_ms
                if tier == ModelTier.OPUS:
                    self._stats.opus_requests += 1
                else:
                    self._stats.sonnet_requests += 1

        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            self._stats.failed_requests += 1

            if tier == ModelTier.OPUS:
                logger.info("Streaming fallback to Sonnet")
                self._stats.fallback_count += 1
                async for chunk in self.stream_message(
                    message=message,
                    tier=ModelTier.SONNET,
                    system_prompt=system_prompt,
                    conversation_history=conversation_history,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    yield chunk
            else:
                raise

    async def message_with_thinking(
        self,
        message: str,
        system_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        max_tokens: int = 16000,
        thinking_budget: int = 10000,
    ) -> dict[str, Any]:
        """Send a message to Opus with extended thinking enabled.

        Uses Claude's extended thinking API for deep analysis tasks.
        Returns both the thinking process and the final response.

        Args:
            message: The user message to send.
            system_prompt: Optional system prompt.
            conversation_history: Previous messages.
            max_tokens: Maximum output tokens (thinking + response).
            thinking_budget: Budget for thinking tokens.

        Returns:
            Dict with 'thinking', 'response', and usage info.
        """
        import time

        model_id = self._get_model_id(ModelTier.OPUS)
        self._stats.total_requests += 1
        self._stats.opus_requests += 1

        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": message})

        start_time = time.time()

        try:
            response = await self._client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                thinking={
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                },
                system=system_prompt or self._STRATEGY_SYSTEM_PROMPT,
                messages=messages,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract thinking and response text from content blocks
            thinking_text = ""
            response_text = ""
            for block in response.content:
                if block.type == "thinking":
                    thinking_text = block.thinking
                elif block.type == "text":
                    response_text = block.text

            self._stats.successful_requests += 1
            self._stats.total_input_tokens += response.usage.input_tokens
            self._stats.total_output_tokens += response.usage.output_tokens
            self._stats.total_latency_ms += latency_ms

            return {
                "thinking": thinking_text,
                "response": response_text,
                "model_used": model_id,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "latency_ms": latency_ms,
                "stop_reason": response.stop_reason,
            }

        except Exception as e:
            logger.error(f"Extended thinking error: {e}")
            self._stats.failed_requests += 1

            # Fallback to regular chat without thinking
            logger.info("Falling back to regular Opus chat (no thinking)")
            response = await self.chat(
                message=message,
                tier=ModelTier.OPUS,
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                max_tokens=max_tokens,
                temperature=0.5,
            )
            return {
                "thinking": "",
                "response": response.content,
                "model_used": response.model_used,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
                "stop_reason": response.stop_reason,
                "fallback": True,
            }

    def update_from_tuner(self):
        """Update client configuration from ServerTuner."""
        tuner_config = self._get_tuner_config()
        if tuner_config:
            new_timeout = tuner_config.request_timeout.current_value
            new_retries = int(tuner_config.max_retries.current_value)

            if new_timeout != self.timeout or new_retries != self.max_retries:
                self.timeout = new_timeout
                self.max_retries = new_retries
                self._initialize_client()
                logger.info(f"Client updated: timeout={new_timeout}s, retries={new_retries}")
