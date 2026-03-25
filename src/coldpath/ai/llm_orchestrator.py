"""
LLM Orchestrator - Coordinate Opus 4.5, Sonnet 4.5, and Ollama with ML ensemble.

Implements hybrid LLM+ML decision making:
- Parallel execution: LLM analysis + ML ensemble scoring
- Response aggregation: Merge LLM insights with ML predictions
- Latency budgets: <100ms scoring, <500ms chat, <30s optimization
- Graceful fallbacks: Opus → Sonnet → Ollama → Cached → ML-only
- Local inference: Ollama for fast/simple tasks and offline capability
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .claude_client import ClaudeClient, ClaudeResponse, ModelTier
from .claude_router import ClaudeRouter, RoutingDecision, TaskType
from .kv_cache_manager import KVCacheManager
from .ollama_client import OllamaClient, OllamaModel, OllamaResponse, get_ollama_client
from .prompt_optimizer import PromptOptimizer

logger = logging.getLogger(__name__)


class LatencyBudget(Enum):
    """Latency budgets for different operation types."""

    SCORING = 100  # <100ms for real-time scoring
    CHAT = 500  # <500ms for chat responses
    ANALYSIS = 2000  # <2s for backtest analysis
    OPTIMIZATION = 30000  # <30s for strategy optimization


class AggregationStrategy(Enum):
    """Strategies for combining LLM and ML outputs."""

    ML_PRIMARY = "ml_primary"  # ML drives decisions, LLM explains
    LLM_PRIMARY = "llm_primary"  # LLM drives decisions with ML context
    ENSEMBLE_WEIGHTED = "weighted"  # Weighted combination
    CONSENSUS_REQUIRED = "consensus"  # Both must agree
    FASTEST_WINS = "fastest"  # Use whichever responds first


class PromptDetailLevel(Enum):
    """Controls how much ML context is exposed to the LLM."""

    ULTRA_COMPACT = "ultra_compact"
    COMPACT = "compact"
    BALANCED = "balanced"
    DETAILED = "detailed"


@dataclass
class MLContext:
    """ML context for LLM analysis."""

    ensemble_score: float
    confidence: float
    rug_risk: float
    signal: str  # "BUY", "HOLD", "SELL"
    price_predictions: dict[str, float]  # T+5, T+10, T+15
    regime: str
    feature_importances: dict[str, float]
    reasons: list[str]

    def _top_feature_items(self, limit: int) -> list[tuple[str, float]]:
        """Return the highest-signal feature importances first."""
        items = [
            (name, float(value))
            for name, value in self.feature_importances.items()
            if isinstance(value, (int, float)) and np.isfinite(value)
        ]
        items.sort(key=lambda item: (-abs(item[1]), item[0]))
        return items[:limit]

    def _format_percentage(self, value: float) -> str:
        return f"{value:+.2f}%"

    def _format_prediction_line(self) -> str:
        return (
            f"pred[t5={self._format_percentage(self.price_predictions.get('t5', 0.0))}, "
            f"t10={self._format_percentage(self.price_predictions.get('t10', 0.0))}, "
            f"t15={self._format_percentage(self.price_predictions.get('t15', 0.0))}]"
        )

    def to_prompt_context(
        self,
        detail_level: PromptDetailLevel = PromptDetailLevel.BALANCED,
        max_features: int = 5,
        max_reasons: int = 3,
    ) -> str:
        """Convert to prompt text with a selectable token budget."""
        top_features = self._top_feature_items(max_features)
        reasons = [r.strip() for r in self.reasons if r and r.strip()][:max_reasons]

        if detail_level == PromptDetailLevel.ULTRA_COMPACT:
            parts = [
                f"signal={self.signal}",
                f"conf={self.confidence:.0%}",
                f"rug={self.rug_risk:.0%}",
                f"regime={self.regime}",
                self._format_prediction_line(),
            ]
            if top_features:
                parts.append(
                    "top=" + ", ".join(f"{name}:{value:+.3f}" for name, value in top_features[:3])
                )
            if reasons:
                parts.append("why=" + " | ".join(reasons[:2]))
            return "; ".join(parts)

        if detail_level == PromptDetailLevel.COMPACT:
            parts = [
                f"signal={self.signal}",
                f"conf={self.confidence:.0%}",
                f"rug={self.rug_risk:.0%}",
                f"regime={self.regime}",
                self._format_prediction_line(),
            ]
            if top_features:
                parts.append(
                    "factors="
                    + ", ".join(f"{name}:{value:+.3f}" for name, value in top_features[:4])
                )
            if reasons:
                parts.append("why=" + " | ".join(reasons[:2]))
            return " | ".join(parts)

        lines = [
            "ML Analysis Results:",
            f"- Signal: {self.signal} (confidence: {self.confidence:.2%})",
            f"- Rug Risk: {self.rug_risk:.2%}",
            f"- Current Regime: {self.regime}",
            f"- Price Predictions: {self._format_prediction_line()}",
        ]

        if top_features:
            lines.append("- Key Factors:")
            for name, value in top_features:
                lines.append(f"  - {name}: {value:+.3f}")

        if reasons:
            if detail_level == PromptDetailLevel.DETAILED:
                lines.append(f"- Reasoning: {'; '.join(reasons)}")
            else:
                lines.append(f"- Reasoning: {'; '.join(reasons[:2])}")

        return "\n".join(lines)

    def estimated_prompt_tokens(
        self,
        detail_level: PromptDetailLevel = PromptDetailLevel.BALANCED,
    ) -> int:
        """Rough token estimate for the chosen prompt detail level."""
        prompt = self.to_prompt_context(detail_level=detail_level)
        return max(1, int(len(prompt) / 4))


@dataclass
class HybridDecision:
    """Combined LLM + ML decision."""

    # Final recommendation
    action: str  # "BUY", "HOLD", "SELL", "EXIT"
    confidence: float
    position_size: float

    # LLM contribution
    llm_response: ClaudeResponse | None
    llm_recommendation: str | None
    llm_reasoning: str | None
    llm_tier_used: ModelTier | None

    # ML contribution
    ml_signal: str
    ml_confidence: float
    ml_score: float

    # Aggregation metadata
    strategy_used: AggregationStrategy
    agreement_score: float  # How much LLM and ML agree (0-1)
    latency_ms: float
    cache_hit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "position_size": self.position_size,
            "llm": {
                "recommendation": self.llm_recommendation,
                "reasoning": self.llm_reasoning,
                "tier": self.llm_tier_used.value if self.llm_tier_used else None,
            },
            "ml": {
                "signal": self.ml_signal,
                "confidence": self.ml_confidence,
                "score": self.ml_score,
            },
            "meta": {
                "strategy": self.strategy_used.value,
                "agreement": self.agreement_score,
                "latency_ms": self.latency_ms,
                "cache_hit": self.cache_hit,
            },
        }


@dataclass
class OrchestratorConfig:
    """Configuration for LLM orchestrator."""

    # Latency budgets (ms)
    scoring_budget_ms: int = 100
    chat_budget_ms: int = 500
    analysis_budget_ms: int = 2000
    optimization_budget_ms: int = 30000

    # Aggregation settings
    default_strategy: AggregationStrategy = AggregationStrategy.ML_PRIMARY
    ml_weight: float = 0.6  # Weight for ML in ensemble strategy
    llm_weight: float = 0.4  # Weight for LLM in ensemble strategy
    consensus_threshold: float = 0.7  # Agreement needed for consensus

    # Fallback settings
    enable_llm_fallback: bool = True
    fallback_to_ml_only: bool = True
    max_llm_retries: int = 2

    # Ollama settings
    enable_ollama: bool = True  # Enable Ollama as fallback
    ollama_for_simple_tasks: bool = True  # Use Ollama for simple/fast tasks
    ollama_latency_threshold_ms: int = 100  # Use Ollama for tasks under this budget
    prefer_ollama_offline: bool = True  # Prefer Ollama when Claude unavailable

    # Caching
    enable_cache: bool = True
    cache_llm_responses: bool = True

    # Confidence thresholds
    min_confidence_for_trade: float = 0.6
    high_confidence_threshold: float = 0.85

    def get_budget(self, task_type: TaskType) -> int:
        """Get latency budget for task type."""
        budgets = {
            TaskType.STRATEGY_OPTIMIZATION: self.optimization_budget_ms,
            TaskType.BACKTEST_ANALYSIS: self.analysis_budget_ms,
            TaskType.RISK_ASSESSMENT: self.analysis_budget_ms,
            TaskType.GENERAL_CHAT: self.chat_budget_ms,
            TaskType.STATUS_UPDATE: self.scoring_budget_ms,
            TaskType.SETTINGS_EXPLANATION: self.chat_budget_ms,
            TaskType.PERFORMANCE_QUERY: self.chat_budget_ms,
        }
        return budgets.get(task_type, self.chat_budget_ms)

    def should_use_ollama(self, task_type: TaskType, budget_ms: int) -> bool:
        """Determine if Ollama should be used for this task."""
        if not self.enable_ollama:
            return False

        # Simple tasks with tight latency budgets → Ollama
        simple_tasks = {
            TaskType.STATUS_UPDATE,
            TaskType.SETTINGS_EXPLANATION,
        }

        if self.ollama_for_simple_tasks and task_type in simple_tasks:
            return True

        if budget_ms <= self.ollama_latency_threshold_ms:
            return True

        return False


class LLMOrchestrator:
    """Orchestrates hybrid LLM + ML decision making.

    Coordinates between:
    - Opus 4.5: Complex reasoning, strategy optimization, deep analysis
    - Sonnet 4.5: Fast responses, explanations, status updates
    - Ollama (local): Fast fallback, simple tasks, offline capability
    - ML Ensemble: Real-time scoring, rug detection, signal generation

    Features:
    - Parallel execution for latency optimization
    - KV cache for token savings (target: 40%)
    - Graceful degradation: Opus → Sonnet → Ollama → ML-only
    - Agreement scoring between LLM and ML
    - Local inference for cost savings and latency
    """

    def __init__(
        self,
        client: ClaudeClient,
        router: ClaudeRouter | None = None,
        cache_manager: KVCacheManager | None = None,
        ollama_client: OllamaClient | None = None,
        config: OrchestratorConfig | None = None,
    ):
        self.client = client
        self.router = router or ClaudeRouter(client)
        self.cache = cache_manager or KVCacheManager()
        self.ollama = ollama_client or get_ollama_client()
        self.config = config or OrchestratorConfig()
        self.prompt_optimizer = PromptOptimizer(default_max_tokens=2048)

        # Ollama availability (checked lazily)
        self._ollama_available: bool | None = None

        # Statistics
        self._stats = {
            "total_requests": 0,
            "llm_calls": 0,
            "ollama_calls": 0,
            "cache_hits": 0,
            "ml_only_fallbacks": 0,
            "consensus_achieved": 0,
            "avg_latency_ms": 0.0,
            "token_savings_pct": 0.0,
            "ollama_fallbacks": 0,
        }

    async def _check_ollama(self) -> bool:
        """Check if Ollama is available (cached)."""
        if self._ollama_available is None:
            self._ollama_available = await self.ollama.is_available()
            if self._ollama_available:
                logger.info("Ollama available for local inference")
        return self._ollama_available

    async def make_hybrid_decision(
        self,
        query: str,
        ml_context: MLContext,
        strategy: AggregationStrategy | None = None,
        max_latency_ms: int | None = None,
    ) -> HybridDecision:
        """Make a hybrid LLM + ML decision.

        Args:
            query: User query or token analysis request
            ml_context: ML ensemble results
            strategy: Aggregation strategy (default from config)
            max_latency_ms: Override latency budget

        Returns:
            HybridDecision combining LLM and ML insights
        """
        start_time = time.time()
        self._stats["total_requests"] += 1

        strategy = strategy or self.config.default_strategy

        # Classify the query for routing
        routing = self.router.classify(query)
        budget_ms = max_latency_ms or self.config.get_budget(routing.task_type)
        detail_level = self._select_prompt_detail_level(routing, budget_ms, ml_context)

        # Check cache first
        cache_key = self._compute_cache_key(query, ml_context, detail_level)
        cached = await self.cache.get(cache_key)
        if cached and self.config.enable_cache:
            self._stats["cache_hits"] += 1
            return self._build_cached_decision(cached, ml_context, strategy, start_time)

        # Determine if we have time for LLM
        use_llm = budget_ms >= self.config.scoring_budget_ms

        # Determine if we should use Ollama for this task
        use_ollama = (
            self.config.should_use_ollama(routing.task_type, budget_ms)
            and await self._check_ollama()
        )

        if use_ollama:
            # Use Ollama for fast local inference
            try:
                ollama_response = await self._get_ollama_analysis(
                    query, ml_context, budget_ms, detail_level
                )
                self._stats["ollama_calls"] += 1

                decision = self._aggregate_ollama_decision(ollama_response, ml_context, strategy)

                # Cache the response
                if self.config.cache_llm_responses:
                    await self.cache.set(
                        cache_key,
                        {
                            "llm_recommendation": decision.llm_recommendation,
                            "llm_reasoning": decision.llm_reasoning,
                        },
                        ttl=300,
                    )

            except Exception as e:
                logger.warning(f"Ollama error: {e}, falling back to ML")
                decision = self._ml_only_decision(ml_context, strategy)
                self._stats["ml_only_fallbacks"] += 1

        elif use_llm:
            try:
                # Run Claude LLM analysis
                llm_response = await self._get_llm_analysis(
                    query, ml_context, routing, budget_ms, detail_level
                )
                self._stats["llm_calls"] += 1

                decision = self._aggregate_decisions(llm_response, ml_context, strategy, routing)

                # Cache the response
                if self.config.cache_llm_responses:
                    await self.cache.set(
                        cache_key,
                        {
                            "llm_recommendation": decision.llm_recommendation,
                            "llm_reasoning": decision.llm_reasoning,
                        },
                        ttl=300,
                    )  # 5 min cache

            except TimeoutError:
                logger.warning(f"Claude timeout after {budget_ms}ms, trying Ollama fallback")
                decision = await self._try_ollama_fallback(query, ml_context, strategy)
            except Exception as e:
                logger.error(f"Claude error: {e}, trying Ollama fallback")
                decision = await self._try_ollama_fallback(query, ml_context, strategy)
        else:
            # No time for LLM, use ML only
            decision = self._ml_only_decision(ml_context, strategy)
            self._stats["ml_only_fallbacks"] += 1

        # Update latency
        decision.latency_ms = (time.time() - start_time) * 1000
        self._update_stats(decision)

        return decision

    async def analyze_proposal(
        self,
        proposal: dict[str, Any],
        ml_scores: dict[str, float],
        regime: str,
    ) -> dict[str, Any]:
        """Analyze a strategy proposal using Opus.

        Args:
            proposal: Strategy proposal parameters
            ml_scores: Backtest scores from ML
            regime: Current market regime

        Returns:
            Analysis with trade-offs and recommendations
        """
        system_prompt = self.prompt_optimizer.get_optimized_system_prompt("strategy_analysis")

        prompt = f"""Analyze this trading strategy proposal:

**Proposal Parameters:**
{json.dumps(proposal, indent=2)}

**ML Backtest Scores:**
{json.dumps(ml_scores, indent=2)}

**Current Market Regime:** {regime}

Provide your analysis as JSON."""

        # Use Opus for complex analysis
        response = await self.client.chat(
            message=prompt,
            tier=ModelTier.OPUS,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.3,
        )

        try:
            analysis = json.loads(response.content)
        except json.JSONDecodeError:
            analysis = {
                "summary": response.content,
                "confidence": 0.5,
            }

        analysis["model_used"] = response.model_used
        analysis["latency_ms"] = response.latency_ms

        return analysis

    async def explain_score(
        self,
        mint: str,
        features: dict[str, float],
        ml_score: float,
        top_factors: list[tuple[str, float]],
        use_ollama: bool = True,
    ) -> str:
        """Generate quick score explanation using Ollama (fast) or Sonnet (quality).

        Args:
            mint: Token mint address
            features: Feature values
            ml_score: ML profitability score
            top_factors: Top contributing factors (name, contribution)
            use_ollama: Prefer Ollama for fast local inference

        Returns:
            Human-readable explanation
        """
        # Check cache
        cache_key = f"explain:{mint}:{int(ml_score * 100)}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached.get("explanation", "")

        factors_text = "\n".join(f"- {name}: {value:+.3f}" for name, value in top_factors[:5])

        prompt = f"""Briefly explain this token's ML score of {ml_score:.2f} based on these factors:

{factors_text}

Keep it to 2-3 sentences. Focus on the most important factor."""

        # Try Ollama first for fast response
        if use_ollama and self.config.enable_ollama and await self._check_ollama():
            try:
                response = await self.ollama.chat(
                    message=prompt,
                    model=OllamaModel.LLAMA_8B,
                    max_tokens=128,
                    temperature=0.5,
                )
                explanation = response.content
                self._stats["ollama_calls"] += 1
            except Exception as e:
                logger.debug(f"Ollama explain failed, using Claude: {e}")
                response = await self.client.chat(
                    message=prompt,
                    tier=ModelTier.SONNET,
                    max_tokens=256,
                    temperature=0.5,
                )
                explanation = response.content
        else:
            response = await self.client.chat(
                message=prompt,
                tier=ModelTier.SONNET,
                max_tokens=256,
                temperature=0.5,
            )
            explanation = response.content

        # Cache for 1 minute
        await self.cache.set(cache_key, {"explanation": explanation}, ttl=60)

        return explanation

    async def get_regime_analysis(
        self,
        regime: str,
        regime_features: dict[str, float],
        historical_performance: dict[str, float],
    ) -> dict[str, Any]:
        """Analyze current regime and recommend adjustments using Opus.

        Args:
            regime: Current detected regime
            regime_features: Features that led to detection
            historical_performance: Past performance in this regime

        Returns:
            Regime analysis with recommendations
        """
        system_prompt = self.prompt_optimizer.get_optimized_system_prompt("risk_assessment")

        prompt = f"""Analyze the current market regime:

**Detected Regime:** {regime}

**Detection Features:**
{json.dumps(regime_features, indent=2)}

**Historical Performance in This Regime:**
{json.dumps(historical_performance, indent=2)}

What adjustments should be made to trading strategy?"""

        response = await self.client.chat(
            message=prompt,
            tier=ModelTier.OPUS,
            system_prompt=system_prompt,
            max_tokens=1024,
            temperature=0.3,
        )

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "regime_assessment": response.content,
                "confidence_adjustments": {},
            }

    async def _get_llm_analysis(
        self,
        query: str,
        ml_context: MLContext,
        routing: RoutingDecision,
        budget_ms: int,
        detail_level: PromptDetailLevel,
    ) -> ClaudeResponse:
        """Get Claude LLM analysis with timeout."""
        # Build context-enhanced prompt with a cacheable system prefix and compressed context.
        system_prompt = self._get_system_prompt_for_routing(routing)
        enhanced_prompt = f"""{query}

{ml_context.to_prompt_context(detail_level=detail_level)}

Based on the ML analysis, provide your assessment."""

        timeout = budget_ms / 1000.0

        response = await asyncio.wait_for(
            self.client.chat(
                message=enhanced_prompt,
                tier=routing.model_tier,
                system_prompt=system_prompt,
                max_tokens=1024,
                temperature=0.5,
            ),
            timeout=timeout,
        )

        return response

    async def _get_ollama_analysis(
        self,
        query: str,
        ml_context: MLContext,
        budget_ms: int,
        detail_level: PromptDetailLevel = PromptDetailLevel.ULTRA_COMPACT,
    ) -> OllamaResponse:
        """Get Ollama local LLM analysis with timeout."""
        # Build context-enhanced prompt (more concise for local models)
        enhanced_prompt = f"""{query}

{ml_context.to_prompt_context(detail_level=detail_level, max_features=3, max_reasons=2)}

Briefly assess this trading opportunity."""

        timeout = budget_ms / 1000.0

        response = await asyncio.wait_for(
            self.ollama.chat(
                message=enhanced_prompt,
                model=OllamaModel.LLAMA_8B,  # Fast model for quick responses
                max_tokens=256,
                temperature=0.5,
            ),
            timeout=timeout,
        )

        return response

    def _aggregate_ollama_decision(
        self,
        ollama_response: OllamaResponse,
        ml_context: MLContext,
        strategy: AggregationStrategy,
    ) -> HybridDecision:
        """Aggregate Ollama and ML decisions."""
        # Parse Ollama recommendation
        llm_action, llm_confidence = self._parse_llm_recommendation(ollama_response.content)

        # Calculate agreement
        ml_action = ml_context.signal
        agreement = self._calculate_agreement(
            llm_action, llm_confidence, ml_action, ml_context.confidence
        )

        if agreement >= self.config.consensus_threshold:
            self._stats["consensus_achieved"] += 1

        # For Ollama, always use ML_PRIMARY strategy (ML drives, Ollama explains)
        final_action = ml_action
        final_confidence = ml_context.confidence
        position_size = self._calculate_position_size(ml_context)

        return HybridDecision(
            action=final_action,
            confidence=final_confidence,
            position_size=position_size,
            llm_response=None,  # Not a Claude response
            llm_recommendation=llm_action,
            llm_reasoning=ollama_response.content[:500],
            llm_tier_used=None,  # Ollama, not Claude
            ml_signal=ml_context.signal,
            ml_confidence=ml_context.confidence,
            ml_score=ml_context.ensemble_score,
            strategy_used=strategy,
            agreement_score=agreement,
            latency_ms=ollama_response.latency_ms,
            cache_hit=False,
        )

    async def _try_ollama_fallback(
        self,
        query: str,
        ml_context: MLContext,
        strategy: AggregationStrategy,
    ) -> HybridDecision:
        """Try Ollama as fallback when Claude fails."""
        if self.config.prefer_ollama_offline and await self._check_ollama():
            try:
                ollama_response = await self._get_ollama_analysis(
                    query,
                    ml_context,
                    budget_ms=5000,  # 5s budget for fallback
                )
                self._stats["ollama_fallbacks"] += 1

                return self._aggregate_ollama_decision(ollama_response, ml_context, strategy)
            except Exception as e:
                logger.warning(f"Ollama fallback also failed: {e}")

        # Final fallback: ML only
        self._stats["ml_only_fallbacks"] += 1
        return self._ml_only_decision(ml_context, strategy)

    def _aggregate_decisions(
        self,
        llm_response: ClaudeResponse,
        ml_context: MLContext,
        strategy: AggregationStrategy,
        routing: RoutingDecision,
    ) -> HybridDecision:
        """Aggregate LLM and ML decisions based on strategy."""
        # Parse LLM recommendation
        llm_action, llm_confidence = self._parse_llm_recommendation(llm_response.content)

        # Calculate agreement
        ml_action = ml_context.signal
        agreement = self._calculate_agreement(
            llm_action, llm_confidence, ml_action, ml_context.confidence
        )

        if agreement >= self.config.consensus_threshold:
            self._stats["consensus_achieved"] += 1

        # Determine final action based on strategy
        if strategy == AggregationStrategy.ML_PRIMARY:
            final_action = ml_action
            final_confidence = ml_context.confidence
            position_size = self._calculate_position_size(ml_context)

        elif strategy == AggregationStrategy.LLM_PRIMARY:
            final_action = llm_action or ml_action
            final_confidence = llm_confidence if llm_confidence > 0 else ml_context.confidence
            position_size = self._calculate_position_size(ml_context) * (llm_confidence or 1.0)

        elif strategy == AggregationStrategy.ENSEMBLE_WEIGHTED:
            # Weighted combination
            ml_score = self._action_to_score(ml_action) * self.config.ml_weight
            llm_score = self._action_to_score(llm_action) * self.config.llm_weight
            combined_score = ml_score + llm_score

            final_action = self._score_to_action(combined_score)
            final_confidence = (
                ml_context.confidence * self.config.ml_weight
                + (llm_confidence or 0.5) * self.config.llm_weight
            )
            position_size = self._calculate_position_size(ml_context) * final_confidence

        elif strategy == AggregationStrategy.CONSENSUS_REQUIRED:
            if agreement >= self.config.consensus_threshold:
                final_action = ml_action
                final_confidence = max(ml_context.confidence, llm_confidence or 0)
            else:
                final_action = "HOLD"
                final_confidence = 0.5
            position_size = (
                self._calculate_position_size(ml_context) if final_action != "HOLD" else 0.0
            )

        else:  # FASTEST_WINS - in this case ML is always faster
            final_action = ml_action
            final_confidence = ml_context.confidence
            position_size = self._calculate_position_size(ml_context)

        return HybridDecision(
            action=final_action,
            confidence=final_confidence,
            position_size=position_size,
            llm_response=llm_response,
            llm_recommendation=llm_action,
            llm_reasoning=llm_response.content[:500],
            llm_tier_used=routing.model_tier,
            ml_signal=ml_context.signal,
            ml_confidence=ml_context.confidence,
            ml_score=ml_context.ensemble_score,
            strategy_used=strategy,
            agreement_score=agreement,
            latency_ms=llm_response.latency_ms,
            cache_hit=False,
        )

    def _ml_only_decision(
        self,
        ml_context: MLContext,
        strategy: AggregationStrategy,
    ) -> HybridDecision:
        """Create decision using ML only."""
        return HybridDecision(
            action=ml_context.signal,
            confidence=ml_context.confidence,
            position_size=self._calculate_position_size(ml_context),
            llm_response=None,
            llm_recommendation=None,
            llm_reasoning=None,
            llm_tier_used=None,
            ml_signal=ml_context.signal,
            ml_confidence=ml_context.confidence,
            ml_score=ml_context.ensemble_score,
            strategy_used=strategy,
            agreement_score=1.0,  # Perfect agreement with itself
            latency_ms=0.0,
            cache_hit=False,
        )

    def _build_cached_decision(
        self,
        cached: dict[str, Any],
        ml_context: MLContext,
        strategy: AggregationStrategy,
        start_time: float,
    ) -> HybridDecision:
        """Build decision from cached LLM response."""
        llm_recommendation = cached.get("llm_recommendation")

        # Use ML for final decision but include cached LLM reasoning
        return HybridDecision(
            action=ml_context.signal,
            confidence=ml_context.confidence,
            position_size=self._calculate_position_size(ml_context),
            llm_response=None,
            llm_recommendation=llm_recommendation,
            llm_reasoning=cached.get("llm_reasoning"),
            llm_tier_used=None,
            ml_signal=ml_context.signal,
            ml_confidence=ml_context.confidence,
            ml_score=ml_context.ensemble_score,
            strategy_used=strategy,
            agreement_score=1.0,
            latency_ms=(time.time() - start_time) * 1000,
            cache_hit=True,
        )

    def _parse_llm_recommendation(self, content: str) -> tuple[str | None, float]:
        """Parse LLM response to extract action and confidence."""
        content_upper = content.upper()

        # Look for explicit action words
        if "STRONG BUY" in content_upper or "STRONGLY RECOMMEND BUY" in content_upper:
            return "BUY", 0.9
        elif "BUY" in content_upper and "DON'T BUY" not in content_upper:
            return "BUY", 0.7
        elif "EXIT" in content_upper or "SELL IMMEDIATELY" in content_upper:
            return "EXIT", 0.9
        elif "SELL" in content_upper:
            return "SELL", 0.7
        elif "HOLD" in content_upper or "WAIT" in content_upper:
            return "HOLD", 0.6
        else:
            return None, 0.0

    def _calculate_agreement(
        self,
        llm_action: str | None,
        llm_confidence: float,
        ml_action: str,
        ml_confidence: float,
    ) -> float:
        """Calculate agreement score between LLM and ML."""
        if llm_action is None:
            return 0.5  # Neutral

        if llm_action == ml_action:
            # Perfect agreement, weighted by confidences
            return 0.5 + 0.5 * min(llm_confidence, ml_confidence)

        # Map actions to directional values
        action_values = {"BUY": 1, "HOLD": 0, "SELL": -1, "EXIT": -1}
        llm_val = action_values.get(llm_action, 0)
        ml_val = action_values.get(ml_action, 0)

        # Calculate disagreement penalty
        diff = abs(llm_val - ml_val)
        return max(0.0, 1.0 - diff * 0.4)

    def _action_to_score(self, action: str | None) -> float:
        """Convert action to numerical score."""
        scores = {"BUY": 1.0, "HOLD": 0.0, "SELL": -0.5, "EXIT": -1.0}
        return scores.get(action, 0.0) if action else 0.0

    def _score_to_action(self, score: float) -> str:
        """Convert numerical score to action."""
        if score >= 0.5:
            return "BUY"
        elif score <= -0.5:
            return "SELL"
        else:
            return "HOLD"

    def _calculate_position_size(self, ml_context: MLContext) -> float:
        """Calculate position size based on ML context."""
        # Base position from ML
        base_size = 0.1  # 10% of capital

        # Adjust for confidence
        confidence_factor = ml_context.confidence

        # Adjust for rug risk
        rug_factor = 1.0 - ml_context.rug_risk

        # Only size for BUY signals
        if ml_context.signal != "BUY":
            return 0.0

        return base_size * confidence_factor * rug_factor

    def _compute_cache_key(
        self,
        query: str,
        ml_context: MLContext,
        detail_level: PromptDetailLevel,
    ) -> str:
        """Compute cache key for a request."""
        key_parts = [
            query[:100],
            ml_context.signal,
            ml_context.regime,
            f"{ml_context.confidence:.2f}",
            f"rug={ml_context.rug_risk:.2f}",
            f"t5={ml_context.price_predictions.get('t5', 0.0):.2f}",
            f"t10={ml_context.price_predictions.get('t10', 0.0):.2f}",
            f"t15={ml_context.price_predictions.get('t15', 0.0):.2f}",
            f"profile={detail_level.value}",
            self._signature_from_features(ml_context.feature_importances),
        ]
        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _signature_from_features(
        self,
        feature_importances: dict[str, float],
        limit: int = 3,
    ) -> str:
        """Create a short signature from the strongest features."""
        items = [
            (name, float(value))
            for name, value in feature_importances.items()
            if isinstance(value, (int, float)) and np.isfinite(value)
        ]
        items.sort(key=lambda item: (-abs(item[1]), item[0]))
        if not items:
            return "no-features"
        return ",".join(f"{name}:{value:+.2f}" for name, value in items[:limit])

    def _select_prompt_detail_level(
        self,
        routing: RoutingDecision,
        budget_ms: int,
        ml_context: MLContext,
    ) -> PromptDetailLevel:
        """Select how much context to send based on the task and latency budget."""
        if budget_ms <= self.config.scoring_budget_ms:
            return PromptDetailLevel.ULTRA_COMPACT

        if routing.task_type in {
            TaskType.STATUS_UPDATE,
            TaskType.SETTINGS_EXPLANATION,
            TaskType.QUICK_SCORE_EXPLANATION,
            TaskType.FEATURE_IMPORTANCE,
            TaskType.SIGNAL_SUMMARY,
            TaskType.REGIME_UPDATE,
        }:
            return PromptDetailLevel.ULTRA_COMPACT

        if routing.task_type in {
            TaskType.BACKTEST_ANALYSIS,
            TaskType.STRATEGY_OPTIMIZATION,
            TaskType.RISK_ASSESSMENT,
            TaskType.ML_PROPOSAL_RANKING,
            TaskType.REGIME_ANALYSIS,
            TaskType.ENSEMBLE_EXPLANATION,
            TaskType.MONTE_CARLO_ANALYSIS,
        }:
            return (
                PromptDetailLevel.BALANCED
                if ml_context.confidence < 0.9
                else PromptDetailLevel.COMPACT
            )

        return PromptDetailLevel.COMPACT

    def _get_system_prompt_for_routing(self, routing: RoutingDecision) -> str:
        """Choose a short cacheable system prompt for the task class."""
        task_type = routing.task_type

        if task_type in {
            TaskType.STRATEGY_OPTIMIZATION,
            TaskType.BACKTEST_ANALYSIS,
            TaskType.ML_PROPOSAL_RANKING,
        }:
            return self.prompt_optimizer.get_optimized_system_prompt("strategy_analysis")

        if task_type in {
            TaskType.RISK_ASSESSMENT,
            TaskType.REGIME_ANALYSIS,
            TaskType.MONTE_CARLO_ANALYSIS,
            TaskType.ENSEMBLE_EXPLANATION,
        }:
            return self.prompt_optimizer.get_optimized_system_prompt("risk_assessment")

        if task_type in {
            TaskType.STATUS_UPDATE,
            TaskType.SETTINGS_EXPLANATION,
            TaskType.QUICK_SCORE_EXPLANATION,
            TaskType.FEATURE_IMPORTANCE,
            TaskType.SIGNAL_SUMMARY,
            TaskType.REGIME_UPDATE,
        }:
            return self.prompt_optimizer.get_optimized_system_prompt("quick_summary")

        return self.prompt_optimizer.get_optimized_system_prompt("performance_summary")

    def _update_stats(self, decision: HybridDecision):
        """Update running statistics."""
        n = self._stats["total_requests"]
        old_avg = self._stats["avg_latency_ms"]
        self._stats["avg_latency_ms"] = old_avg + (decision.latency_ms - old_avg) / n

        # Calculate token savings from cache
        if self._stats["total_requests"] > 0:
            self._stats["token_savings_pct"] = (
                self._stats["cache_hits"] / self._stats["total_requests"] * 40.0
            )

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        total_llm = self._stats["llm_calls"] + self._stats["ollama_calls"]
        return {
            **self._stats,
            "llm_call_rate": (self._stats["llm_calls"] / max(1, self._stats["total_requests"])),
            "ollama_call_rate": (
                self._stats["ollama_calls"] / max(1, self._stats["total_requests"])
            ),
            "cache_hit_rate": (self._stats["cache_hits"] / max(1, self._stats["total_requests"])),
            "consensus_rate": (self._stats["consensus_achieved"] / max(1, total_llm)),
            "ollama_fallback_rate": (
                self._stats["ollama_fallbacks"] / max(1, self._stats["llm_calls"])
            ),
            "ollama_available": self._ollama_available,
        }

    def reset_stats(self):
        """Reset statistics."""
        self._stats = {
            "total_requests": 0,
            "llm_calls": 0,
            "ollama_calls": 0,
            "cache_hits": 0,
            "ml_only_fallbacks": 0,
            "consensus_achieved": 0,
            "avg_latency_ms": 0.0,
            "token_savings_pct": 0.0,
            "ollama_fallbacks": 0,
        }


class ConversationManager:
    """Multi-turn conversation manager with market context injection.

    Automatically injects current market state into each turn as system context
    (invisible to user). Trims conversation history to stay within token limits.
    """

    # Approximate tokens per message (conservative estimate)
    _TOKENS_PER_CHAR = 0.25

    def __init__(
        self,
        claude_client: ClaudeClient,
        max_history_turns: int = 10,
        max_context_tokens: int = 80000,
    ):
        self.client = claude_client
        self.max_history_turns = max_history_turns
        self.max_context_tokens = max_context_tokens

        # Conversation history: list of {"role": "user"|"assistant", "content": str}
        self._history: list[dict[str, str]] = []

        # Market snapshot injected as system context each turn
        self._market_snapshot: dict[str, Any] | None = None

        # Statistics
        self._total_turns: int = 0
        self._trims: int = 0

    def update_market_snapshot(
        self,
        regime: str,
        top_positions: list[dict[str, Any]],
        recent_pnl: float,
        win_rate: float,
        drawdown_pct: float,
        active_alerts: list[str],
        extra: dict[str, Any] | None = None,
    ):
        """Update the market snapshot injected into each conversation turn."""
        self._market_snapshot = {
            "regime": regime,
            "top_positions": top_positions[:5],
            "recent_pnl_pct": recent_pnl,
            "win_rate": win_rate,
            "drawdown_pct": drawdown_pct,
            "active_alerts": active_alerts[:5],
        }
        if extra:
            self._market_snapshot.update(extra)

    async def send_message(
        self,
        user_message: str,
        tier: ModelTier | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """Send a user message with market context injected.

        Args:
            user_message: The user's message text
            tier: Model tier (defaults to Sonnet for chat)
            max_tokens: Max response tokens

        Returns:
            Assistant response text
        """
        self._total_turns += 1
        tier = tier or ModelTier.SONNET

        # Add user message to history
        self._history.append({"role": "user", "content": user_message})

        # Trim if over limit
        self._trim_history()

        # Build system prompt with market context
        system_prompt = self._build_system_prompt()

        # Build messages for API
        messages = [{"role": m["role"], "content": m["content"]} for m in self._history]

        try:
            response = await self.client.chat(
                message=messages[-1]["content"],
                tier=tier,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=0.5,
                conversation_history=[
                    {"role": m["role"], "content": m["content"]} for m in messages[:-1]
                ]
                if len(messages) > 1
                else None,
            )

            assistant_text = response.content
            self._history.append({"role": "assistant", "content": assistant_text})
            return assistant_text

        except Exception as e:
            logger.error(f"ConversationManager send_message failed: {e}")
            error_msg = f"I'm having trouble processing that right now. Error: {str(e)[:100]}"
            self._history.append({"role": "assistant", "content": error_msg})
            return error_msg

    def _build_system_prompt(self) -> str:
        """Build system prompt with injected market context."""
        base = (
            "You are an AI trading assistant for 2DEXY, a cryptocurrency trading platform. "
            "Help users understand their trading settings, risk levels, and performance. "
            "Be helpful, accurate, and concise. Use markdown for clarity. "
            "Always emphasize risk management."
        )

        if self._market_snapshot:
            snap = self._market_snapshot
            context = (
                f"\n\n[Current Market Context — not visible to user]\n"
                f"- Regime: {snap.get('regime', 'Unknown')}\n"
                f"- Recent PnL: {snap.get('recent_pnl_pct', 0):.2f}%\n"
                f"- Win Rate: {snap.get('win_rate', 0):.1%}\n"
                f"- Drawdown: {snap.get('drawdown_pct', 0):.1f}%\n"
            )
            positions = snap.get("top_positions", [])
            if positions:
                context += "- Top Positions:\n"
                for p in positions[:3]:
                    mint = str(p.get("mint", "?"))[:8]
                    pnl = p.get("pnl_pct", 0)
                    context += f"  - {mint}... ({pnl:+.1f}%)\n"

            alerts = snap.get("active_alerts", [])
            if alerts:
                context += f"- Active Alerts: {', '.join(alerts[:3])}\n"

            base += context

        return base

    def _trim_history(self):
        """Trim conversation history to stay within limits."""
        # Keep at most max_history_turns pairs (user + assistant)
        max_messages = self.max_history_turns * 2
        if len(self._history) > max_messages:
            excess = len(self._history) - max_messages
            self._history = self._history[excess:]
            self._trims += 1

        # Estimate total tokens and trim further if needed
        total_chars = sum(len(m["content"]) for m in self._history)
        estimated_tokens = int(total_chars * self._TOKENS_PER_CHAR)

        while estimated_tokens > self.max_context_tokens and len(self._history) > 2:
            # Remove oldest pair
            self._history = self._history[2:]
            total_chars = sum(len(m["content"]) for m in self._history)
            estimated_tokens = int(total_chars * self._TOKENS_PER_CHAR)
            self._trims += 1

    def clear_history(self):
        """Clear conversation history."""
        self._history.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get conversation manager statistics."""
        return {
            "total_turns": self._total_turns,
            "history_length": len(self._history),
            "trims": self._trims,
            "has_market_snapshot": self._market_snapshot is not None,
        }


async def create_orchestrator(
    api_key: str | None = None,
    config: OrchestratorConfig | None = None,
    enable_ollama: bool = True,
) -> LLMOrchestrator:
    """Factory function to create a configured orchestrator.

    Args:
        api_key: Anthropic API key (uses env var if not provided)
        config: Orchestrator configuration
        enable_ollama: Enable Ollama for local inference fallback

    Returns:
        Configured LLMOrchestrator with Claude and optionally Ollama
    """
    client = ClaudeClient(api_key=api_key)
    cache = KVCacheManager()

    config = config or OrchestratorConfig()
    config.enable_ollama = enable_ollama

    ollama = get_ollama_client() if enable_ollama else None

    orchestrator = LLMOrchestrator(
        client=client,
        cache_manager=cache,
        ollama_client=ollama,
        config=config,
    )

    # Pre-check Ollama availability
    if enable_ollama:
        await orchestrator._check_ollama()

    return orchestrator
