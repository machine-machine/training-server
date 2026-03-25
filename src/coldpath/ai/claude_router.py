"""
Claude Router - Routes queries to appropriate model tier.

Uses Opus 4.5 for complex reasoning tasks and Sonnet 4.5 for
fast, straightforward responses.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .claude_client import ClaudeClient, ClaudeResponse, ModelTier

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks for routing decisions."""

    # Original task types
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    BACKTEST_ANALYSIS = "backtest_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    GENERAL_CHAT = "general_chat"
    STATUS_UPDATE = "status_update"
    SETTINGS_EXPLANATION = "settings_explanation"
    PERFORMANCE_QUERY = "performance_query"

    # ML-specific task types (Opus - complex reasoning)
    ML_PROPOSAL_RANKING = "ml_proposal_ranking"  # Multi-armed bandit analysis
    REGIME_ANALYSIS = "regime_analysis"  # HMM interpretation
    ENSEMBLE_EXPLANATION = "ensemble_explanation"  # Explain ensemble decisions
    MONTE_CARLO_ANALYSIS = "monte_carlo_analysis"  # Risk simulation analysis

    # ML-specific task types (Sonnet - fast inference)
    QUICK_SCORE_EXPLANATION = "quick_score_explanation"  # Fast score explanation
    FEATURE_IMPORTANCE = "feature_importance"  # Structured feature output
    SIGNAL_SUMMARY = "signal_summary"  # Quick signal summary
    REGIME_UPDATE = "regime_update"  # Quick regime status


@dataclass
class RoutingDecision:
    """Result of routing classification."""

    task_type: TaskType
    model_tier: ModelTier
    confidence: float
    reasoning: str


class ClaudeRouter:
    """Routes queries to the appropriate Claude model tier.

    Opus 4.5 is used for:
    - Strategy optimization (complex multi-variable reasoning)
    - Backtest analysis (deep KPI trade-off understanding)
    - Risk assessment (critical decisions need best reasoning)

    Sonnet 4.5 is used for:
    - Chat queries (fast, cost-effective)
    - Status updates (low-latency responses)
    - Settings explanations (straightforward answers)
    """

    # Patterns that indicate Opus should be used
    OPUS_PATTERNS = [
        (r"\b(optimize|optimization)\b", TaskType.STRATEGY_OPTIMIZATION, 0.9),
        (r"\brecommend.*(?:adjustment|change|setting)", TaskType.STRATEGY_OPTIMIZATION, 0.85),
        (r"\bbacktest.*(?:analysis|result|compare)", TaskType.BACKTEST_ANALYSIS, 0.9),
        (r"\b(?:analyze|compare).*(?:strategy|performance)", TaskType.BACKTEST_ANALYSIS, 0.85),
        (r"\brisk.*(?:assessment|evaluation|analysis)", TaskType.RISK_ASSESSMENT, 0.9),
        (r"\b(?:critical|important).*(?:decision|trade)", TaskType.RISK_ASSESSMENT, 0.8),
        (r"\bstrategy.*(?:change|update|modify)", TaskType.STRATEGY_OPTIMIZATION, 0.85),
        (
            r"\b(?:improve|reduce|increase).*(?:drawdown|sharpe|return)",
            TaskType.STRATEGY_OPTIMIZATION,
            0.9,
        ),
        (
            r"\b(?:trade-off|tradeoff|balance).*(?:risk|return)",
            TaskType.STRATEGY_OPTIMIZATION,
            0.85,
        ),
        (r"\b(?:why|explain).*(?:recommend|suggestion)", TaskType.BACKTEST_ANALYSIS, 0.8),
        # ML-specific Opus patterns
        (r"\b(?:rank|compare|evaluate).*proposal", TaskType.ML_PROPOSAL_RANKING, 0.9),
        (r"\b(?:multi.?armed|bandit|thompson|ucb)", TaskType.ML_PROPOSAL_RANKING, 0.9),
        (r"\bproposal.*(?:analysis|ranking|selection)", TaskType.ML_PROPOSAL_RANKING, 0.85),
        (
            r"\b(?:regime|market.?state).*(?:analysis|interpret|explain)",
            TaskType.REGIME_ANALYSIS,
            0.9,
        ),
        (r"\b(?:hmm|hidden.?markov).*(?:state|interpret)", TaskType.REGIME_ANALYSIS, 0.9),
        (r"\b(?:why|explain).*(?:regime|market.?condition)", TaskType.REGIME_ANALYSIS, 0.85),
        (
            r"\b(?:ensemble|model).*(?:decision|explain|interpret)",
            TaskType.ENSEMBLE_EXPLANATION,
            0.85,
        ),
        (r"\bmonte.?carlo.*(?:analysis|result|interpret)", TaskType.MONTE_CARLO_ANALYSIS, 0.9),
        (
            r"\b(?:var|cvar|risk.?of.?ruin).*(?:analysis|explain)",
            TaskType.MONTE_CARLO_ANALYSIS,
            0.85,
        ),
        (
            r"\b(?:stress.?test|perturbation).*(?:result|analysis)",
            TaskType.MONTE_CARLO_ANALYSIS,
            0.85,
        ),
    ]

    # Patterns that indicate Sonnet should be used
    SONNET_PATTERNS = [
        (r"^what\s+(?:is|are)\b", TaskType.SETTINGS_EXPLANATION, 0.85),
        (r"^how\s+(?:do|does|can)\s+i\b", TaskType.SETTINGS_EXPLANATION, 0.85),
        (r"^show\s+(?:me|my)\b", TaskType.STATUS_UPDATE, 0.9),
        (r"\bcurrent.*(?:status|state|level)\b", TaskType.STATUS_UPDATE, 0.85),
        (r"\b(?:hi|hello|hey)\b", TaskType.GENERAL_CHAT, 0.95),
        (r"^(?:thanks|thank you)\b", TaskType.GENERAL_CHAT, 0.95),
        (r"\bwhat\s+does.*(?:do|mean)\b", TaskType.SETTINGS_EXPLANATION, 0.9),
        (r"\bexplain.*(?:setting|feature|option)\b", TaskType.SETTINGS_EXPLANATION, 0.85),
        (r"\b(?:my|current).*(?:balance|position|trade)\b", TaskType.STATUS_UPDATE, 0.85),
        (r"\bhow.*(?:performing|doing)\b", TaskType.PERFORMANCE_QUERY, 0.8),
        # ML-specific Sonnet patterns (fast inference)
        (r"\b(?:quick|brief|short).*(?:score|explain)", TaskType.QUICK_SCORE_EXPLANATION, 0.9),
        (r"\b(?:what|why).*(?:score|rating)", TaskType.QUICK_SCORE_EXPLANATION, 0.85),
        (r"\bscore.*(?:mean|indicate)", TaskType.QUICK_SCORE_EXPLANATION, 0.85),
        (
            r"\b(?:feature|factor).*(?:importance|weight|contribution)",
            TaskType.FEATURE_IMPORTANCE,
            0.9,
        ),
        (r"\b(?:top|main|key).*(?:feature|factor)", TaskType.FEATURE_IMPORTANCE, 0.85),
        (r"\bwhat.*(?:driving|affect)", TaskType.FEATURE_IMPORTANCE, 0.8),
        (r"\b(?:signal|summary|tldr)", TaskType.SIGNAL_SUMMARY, 0.85),
        (r"\b(?:quick|current).*(?:signal|recommendation)", TaskType.SIGNAL_SUMMARY, 0.85),
        (r"\b(?:current|what).*regime", TaskType.REGIME_UPDATE, 0.85),
        (r"\b(?:market|regime).*(?:now|current)", TaskType.REGIME_UPDATE, 0.85),
    ]

    def __init__(
        self,
        client: ClaudeClient,
        default_tier: ModelTier = ModelTier.SONNET,
        opus_threshold: float = 0.75,
    ):
        """Initialize router.

        Args:
            client: Claude client instance.
            default_tier: Default model tier when uncertain.
            opus_threshold: Minimum confidence to use Opus.
        """
        self.client = client
        self.default_tier = default_tier
        self.opus_threshold = opus_threshold

        # Compile patterns for efficiency
        self._opus_compiled = [
            (re.compile(pattern, re.IGNORECASE), task_type, confidence)
            for pattern, task_type, confidence in self.OPUS_PATTERNS
        ]
        self._sonnet_compiled = [
            (re.compile(pattern, re.IGNORECASE), task_type, confidence)
            for pattern, task_type, confidence in self.SONNET_PATTERNS
        ]

    def classify(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> RoutingDecision:
        """Classify a query to determine the appropriate model tier.

        Args:
            query: User's query text.
            context: Optional context (e.g., conversation history, user state).

        Returns:
            RoutingDecision with task type and model tier.
        """
        query_lower = query.lower().strip()

        # Check Opus patterns first (higher priority)
        for pattern, task_type, confidence in self._opus_compiled:
            if pattern.search(query_lower):
                return RoutingDecision(
                    task_type=task_type,
                    model_tier=ModelTier.OPUS,
                    confidence=confidence,
                    reasoning=f"Matched Opus pattern for {task_type.value}",
                )

        # Check Sonnet patterns
        for pattern, task_type, confidence in self._sonnet_compiled:
            if pattern.search(query_lower):
                return RoutingDecision(
                    task_type=task_type,
                    model_tier=ModelTier.SONNET,
                    confidence=confidence,
                    reasoning=f"Matched Sonnet pattern for {task_type.value}",
                )

        # Context-based routing
        if context:
            decision = self._route_by_context(query, context)
            if decision:
                return decision

        # Default fallback
        return RoutingDecision(
            task_type=TaskType.GENERAL_CHAT,
            model_tier=self.default_tier,
            confidence=0.5,
            reasoning="No specific pattern matched, using default tier",
        )

    def _route_by_context(
        self,
        query: str,
        context: dict[str, Any],
    ) -> RoutingDecision | None:
        """Route based on conversation context.

        Args:
            query: User's query.
            context: Conversation context.

        Returns:
            RoutingDecision if context provides routing hints, None otherwise.
        """
        # Check if we're in a strategy optimization flow
        if context.get("in_optimization_flow"):
            return RoutingDecision(
                task_type=TaskType.STRATEGY_OPTIMIZATION,
                model_tier=ModelTier.OPUS,
                confidence=0.8,
                reasoning="In optimization flow context",
            )

        # Check if we're analyzing backtest results
        if context.get("pending_backtest"):
            return RoutingDecision(
                task_type=TaskType.BACKTEST_ANALYSIS,
                model_tier=ModelTier.OPUS,
                confidence=0.85,
                reasoning="Pending backtest analysis",
            )

        # Check message complexity (length, question marks, etc.)
        if len(query) > 200 or query.count("?") > 2:
            return RoutingDecision(
                task_type=TaskType.STRATEGY_OPTIMIZATION,
                model_tier=ModelTier.OPUS,
                confidence=0.7,
                reasoning="Complex query detected (length/questions)",
            )

        return None

    async def route_and_respond(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[ClaudeResponse, RoutingDecision]:
        """Classify query and get response from appropriate model.

        Args:
            query: User's query.
            context: Optional context for routing.
            system_prompt: Optional system prompt override.
            conversation_history: Previous messages.

        Returns:
            Tuple of (ClaudeResponse, RoutingDecision).
        """
        # Classify the query
        decision = self.classify(query, context)

        logger.info(
            f"Routing query to {decision.model_tier.value} "
            f"(task: {decision.task_type.value}, confidence: {decision.confidence:.2f})"
        )

        # Get response from appropriate model
        response = await self.client.chat(
            message=query,
            tier=decision.model_tier,
            system_prompt=system_prompt or self._get_system_prompt(decision.task_type),
            conversation_history=conversation_history,
        )

        return response, decision

    def _get_system_prompt(self, task_type: TaskType) -> str:
        """Get appropriate system prompt for task type."""
        prompts = {
            TaskType.STRATEGY_OPTIMIZATION: """You are an expert quantitative trading strategist.
Help the user optimize their trading strategy by analyzing parameters, identifying trade-offs,
and providing specific recommendations with expected impacts. Be thorough and analytical.""",
            TaskType.BACKTEST_ANALYSIS: """You are an expert at analyzing trading backtest results.
Compare strategies, explain metric changes, identify statistical significance, and provide
actionable recommendations. Focus on risk-adjusted returns and practical implications.""",
            TaskType.RISK_ASSESSMENT: """You are a risk management expert for crypto trading.
Assess the user's current risk exposure, identify potential issues, and recommend risk mitigation
strategies. Be conservative and prioritize capital preservation.""",
            TaskType.GENERAL_CHAT: """You are a helpful AI trading assistant.
Answer questions clearly and concisely. Use markdown formatting for readability.
Always emphasize risk management when discussing trading.""",
            TaskType.STATUS_UPDATE: """You are a helpful AI trading assistant providing status
updates. Be concise and clear. Format information for easy reading. Highlight important changes.""",
            TaskType.SETTINGS_EXPLANATION: """You are a helpful AI trading assistant explaining
settings. Provide clear, simple explanations of trading settings and features. Use examples when
helpful. Avoid jargon unless necessary, and always explain technical terms.""",
            TaskType.PERFORMANCE_QUERY: """You are a helpful AI trading assistant analyzing
performance. Summarize trading performance clearly. Highlight wins and areas for improvement.
Provide context for metrics and actionable suggestions.""",
            # ML-specific Opus prompts
            TaskType.ML_PROPOSAL_RANKING: """You are an expert at multi-armed bandit optimization
and strategy selection. Analyze the proposed strategies considering exploration/exploitation
trade-offs, historical performance, and current market regime. Provide clear rankings with
confidence levels and reasoning. Return structured JSON with rankings and explanations.""",
            TaskType.REGIME_ANALYSIS: """You are an expert at interpreting Hidden Markov Model
market regime detection. Analyze the current regime state, transition probabilities, and
historical patterns. Explain what the regime means for trading strategy and recommend specific
adjustments. Consider volatility, momentum, MEV activity, and liquidity conditions.""",
            TaskType.ENSEMBLE_EXPLANATION: """You are an expert at explaining ensemble ML model
decisions. Break down how the Isolation Forest, LSTM, XGBoost, and Kelly Criterion components
contributed to the final decision. Explain any veto logic applied and why.
Make technical concepts accessible while maintaining accuracy.""",
            TaskType.MONTE_CARLO_ANALYSIS: """You are an expert at interpreting Monte Carlo risk
simulations. Analyze the distribution of outcomes, confidence intervals, and tail risks.
Explain VaR, CVaR, and risk of ruin in practical terms.
Identify key sensitivities and stress test failures.""",
            # ML-specific Sonnet prompts (fast)
            TaskType.QUICK_SCORE_EXPLANATION: """You are a trading assistant explaining ML scores
briefly. Give a 1-2 sentence explanation of what the score means.
Mention only the most important factor. Be concise.""",
            TaskType.FEATURE_IMPORTANCE: """You are a trading assistant explaining feature
importance. List the top factors affecting the decision in order of importance.
Use bullet points. Keep it brief and actionable.""",
            TaskType.SIGNAL_SUMMARY: """You are a trading assistant summarizing signals.
Provide a one-line signal summary with confidence level.
Be direct: BUY, HOLD, or SELL with brief reasoning.""",
            TaskType.REGIME_UPDATE: """You are a trading assistant providing regime updates.
State the current regime in one line with key characteristics.
Note any recommended adjustments briefly.""",
        }
        return prompts.get(task_type, prompts[TaskType.GENERAL_CHAT])

    def get_routing_stats(self) -> dict[str, Any]:
        """Get statistics about routing decisions (for monitoring)."""
        return {
            "opus_pattern_count": len(self.OPUS_PATTERNS),
            "sonnet_pattern_count": len(self.SONNET_PATTERNS),
            "default_tier": self.default_tier.value,
            "opus_threshold": self.opus_threshold,
        }
