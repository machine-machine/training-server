"""
Trade Explanation Engine.

Generates human-readable explanations for trading decisions:
- Top-5 features that drove the decision
- Market regime context
- Confidence level explanation
- Risk factors

Uses Claude Sonnet 4.5 for <500ms latency.
Cached per (token, decision) for 60s to avoid redundant API calls.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TradeExplanation:
    """Explanation for a trade decision."""

    mint: str
    decision: str  # "GO", "NOGO", "REVIEW"
    explanation: str  # 2-3 sentence explanation
    top_features: list[tuple[str, float]]  # (name, contribution)
    regime: str
    confidence: float
    latency_ms: float
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "decision": self.decision,
            "explanation": self.explanation,
            "top_features": [{"name": n, "contribution": c} for n, c in self.top_features],
            "regime": self.regime,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
        }


class TradeExplainer:
    """Generates natural-language explanations for trade decisions.

    Uses Claude Sonnet for fast, concise explanations.
    Implements caching to avoid redundant API calls.
    """

    def __init__(
        self,
        claude_client=None,
        cache_ttl_s: float = 60.0,
        max_cache_size: int = 100,
    ):
        self.client = claude_client
        self.cache_ttl_s = cache_ttl_s
        self.max_cache_size = max_cache_size

        # Cache: key -> (explanation, timestamp)
        self._cache: dict[str, tuple[str, float]] = {}

        # Statistics
        self._total_requests: int = 0
        self._cache_hits: int = 0
        self._total_latency_ms: float = 0.0

    async def explain_trade(
        self,
        mint: str,
        decision: str,
        confidence: float,
        top_features: list[tuple[str, float]],
        regime: str,
        additional_context: dict[str, Any] | None = None,
    ) -> TradeExplanation:
        """Generate explanation for a trade decision.

        Args:
            mint: Token mint address (truncated for display)
            decision: "GO", "NOGO", or "REVIEW"
            confidence: Confidence score [0, 1]
            top_features: Top 5 features and their contributions
            regime: Current market regime
            additional_context: Optional extra context (PnL, risk, etc.)

        Returns:
            TradeExplanation with human-readable explanation
        """
        self._total_requests += 1
        start_time = time.time()

        # Check cache
        cache_key = self._cache_key(mint, decision)
        cached_explanation = self._get_cached(cache_key)
        if cached_explanation:
            self._cache_hits += 1
            latency = (time.time() - start_time) * 1000
            return TradeExplanation(
                mint=mint,
                decision=decision,
                explanation=cached_explanation,
                top_features=top_features,
                regime=regime,
                confidence=confidence,
                latency_ms=latency,
                cached=True,
            )

        # Generate explanation
        explanation = await self._generate(
            mint, decision, confidence, top_features, regime, additional_context
        )

        latency = (time.time() - start_time) * 1000
        self._total_latency_ms += latency

        # Cache the result
        self._set_cached(cache_key, explanation)

        return TradeExplanation(
            mint=mint,
            decision=decision,
            explanation=explanation,
            top_features=top_features,
            regime=regime,
            confidence=confidence,
            latency_ms=latency,
        )

    async def _generate(
        self,
        mint: str,
        decision: str,
        confidence: float,
        top_features: list[tuple[str, float]],
        regime: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate explanation using Claude Sonnet."""
        # Build feature description
        features_text = "\n".join(f"- {name}: {value:+.3f}" for name, value in top_features[:5])

        prompt = f"""In 2-3 concise sentences, explain this trading decision to a trader:

Token: {mint[:8]}...{mint[-4:] if len(mint) > 12 else mint}
Decision: {decision} (confidence: {confidence:.0%})
Market Regime: {regime}

Top Contributing Features:
{features_text}

{f"Additional Context: {context}" if context else ""}

Focus on the most important factor driving this decision. Be specific."""

        if self.client is not None:
            try:
                from .claude_client import ModelTier

                response = await asyncio.wait_for(
                    self.client.chat(
                        message=prompt,
                        tier=ModelTier.SONNET,
                        max_tokens=200,
                        temperature=0.3,
                    ),
                    timeout=5.0,
                )
                return response.content
            except Exception as e:
                logger.debug(f"LLM explanation failed: {e}")

        # Fallback: generate rule-based explanation
        return self._rule_based_explanation(decision, confidence, top_features, regime)

    def _rule_based_explanation(
        self,
        decision: str,
        confidence: float,
        top_features: list[tuple[str, float]],
        regime: str,
    ) -> str:
        """Generate a rule-based explanation when LLM is unavailable."""
        if not top_features:
            return f"Decision: {decision} with {confidence:.0%} confidence in {regime} market."

        top_name, top_value = top_features[0]

        if decision == "GO":
            direction = "positive" if top_value > 0 else "negative"
            return (
                f"Trade approved ({confidence:.0%} confidence). "
                f"Primary driver: {top_name} ({direction} signal at {top_value:+.3f}). "
                f"Current {regime} regime supports this setup."
            )
        elif decision == "NOGO":
            return (
                f"Trade rejected ({confidence:.0%} confidence). "
                f"Insufficient edge — {top_name} at {top_value:+.3f} in {regime} regime. "
                f"Risk/reward ratio does not justify entry."
            )
        else:
            return (
                f"Trade needs review ({confidence:.0%} confidence). "
                f"Borderline signal: {top_name} at {top_value:+.3f}. "
                f"Consider market conditions before proceeding."
            )

    def _cache_key(self, mint: str, decision: str) -> str:
        """Compute cache key from mint and decision."""
        raw = f"{mint}:{decision}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> str | None:
        """Get cached explanation if still valid."""
        if key not in self._cache:
            return None
        explanation, timestamp = self._cache[key]
        if time.time() - timestamp > self.cache_ttl_s:
            del self._cache[key]
            return None
        return explanation

    def _set_cached(self, key: str, explanation: str):
        """Cache an explanation."""
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_cache_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[key] = (explanation, time.time())

    def get_stats(self) -> dict[str, Any]:
        """Get explainer statistics."""
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._total_requests),
            "avg_latency_ms": (
                self._total_latency_ms / max(1, self._total_requests - self._cache_hits)
            ),
            "cache_size": len(self._cache),
        }
