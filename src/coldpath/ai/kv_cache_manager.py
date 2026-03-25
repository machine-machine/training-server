"""
KV Cache Manager - Token recaching for LLM efficiency.

Implements multi-tier caching architecture:
├── Static Context Cache (24h TTL)
│   ├── System prompts by task type
│   ├── Strategy schema documentation
│   └── Risk limits documentation
├── Dynamic Context Cache (5-60s TTL)
│   ├── Conversation history (last 10 turns)
│   ├── Backtest results (15min TTL)
│   └── ML ensemble state (5min TTL)
└── Cache Strategy
    ├── Prefix caching (Anthropic-managed)
    ├── Semantic deduplication
    └── LRU eviction (100MB budget)

Target: 40% token savings on repeated context
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CacheLayer(Enum):
    """Cache layer types."""

    STATIC = "static"  # Long-lived content (prompts, docs)
    DYNAMIC = "dynamic"  # Short-lived content (conversation, results)
    SEMANTIC = "semantic"  # Semantically deduplicated content


@dataclass
class CacheEntry:
    """A cache entry with metadata."""

    key: str
    value: Any
    layer: CacheLayer
    created_at: float
    ttl: int  # Seconds
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    token_count: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Cache statistics."""

    total_entries: int = 0
    total_size_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    token_savings: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class SemanticDeduplicator:
    """Semantic deduplication for similar content."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self._fingerprints: dict[str, str] = {}  # fingerprint -> canonical key

    def compute_fingerprint(self, content: str) -> str:
        """Compute semantic fingerprint for content."""
        # Normalize content
        normalized = content.lower().strip()
        # Remove common variations
        normalized = " ".join(normalized.split())

        # Use n-gram shingling for fingerprint
        shingles = set()
        words = normalized.split()
        for i in range(len(words) - 2):
            shingle = " ".join(words[i : i + 3])
            shingles.add(shingle)

        # Hash the shingle set
        shingle_str = ":".join(sorted(shingles))
        return hashlib.md5(shingle_str.encode()).hexdigest()[:16]

    def find_duplicate(self, content: str) -> str | None:
        """Find a semantically similar cached entry."""
        fingerprint = self.compute_fingerprint(content)

        # Check exact fingerprint match
        if fingerprint in self._fingerprints:
            return self._fingerprints[fingerprint]

        # For more sophisticated deduplication, we could use:
        # - MinHash for Jaccard similarity
        # - SimHash for Hamming distance
        # - Embedding similarity (requires model)

        return None

    def register(self, content: str, key: str):
        """Register content with its cache key."""
        fingerprint = self.compute_fingerprint(content)
        self._fingerprints[fingerprint] = key

    def remove(self, content: str):
        """Remove content registration."""
        fingerprint = self.compute_fingerprint(content)
        self._fingerprints.pop(fingerprint, None)


class KVCacheManager:
    """Multi-tier KV cache for LLM token optimization.

    Implements:
    - Static context caching (system prompts, documentation)
    - Dynamic context caching (conversation history, results)
    - Semantic deduplication (similar content reuse)
    - LRU eviction with configurable memory budget

    Usage:
        cache = KVCacheManager(max_size_mb=100)
        await cache.set("key", value, ttl=300)
        value = await cache.get("key")
    """

    # Default TTLs in seconds
    TTL_STATIC = 86400  # 24 hours
    TTL_CONVERSATION = 300  # 5 minutes
    TTL_BACKTEST = 900  # 15 minutes
    TTL_ML_STATE = 300  # 5 minutes
    TTL_SCORE = 3  # 3 seconds
    TTL_FEATURE = 60  # 1 minute (adaptive 5-60s)

    # Cleanup interval
    CLEANUP_INTERVAL_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        max_size_mb: int = 100,
        enable_semantic_dedup: bool = True,
        max_conversation_turns: int = 10,
    ):
        """Initialize cache manager.

        Args:
            max_size_mb: Maximum cache size in megabytes
            enable_semantic_dedup: Enable semantic deduplication
            max_conversation_turns: Max conversation turns to cache
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_conversation_turns = max_conversation_turns

        # Layer-specific caches using OrderedDict for LRU
        self._static_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._dynamic_cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Semantic deduplication
        self._deduplicator = SemanticDeduplicator() if enable_semantic_dedup else None

        # Statistics
        self._stats = CacheStats()

        # Locks for thread safety
        self._lock = asyncio.Lock()

        # Pre-loaded static prompts
        self._system_prompts: dict[str, str] = {}

        logger.info(f"KVCacheManager initialized with {max_size_mb}MB budget")

    async def get(self, key: str) -> Any | None:
        """Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            # Check dynamic cache first (more recent)
            entry = self._dynamic_cache.get(key)
            if entry and not entry.is_expired:
                entry.access_count += 1
                entry.last_accessed = time.time()
                # Move to end (most recently used)
                self._dynamic_cache.move_to_end(key)
                self._stats.hits += 1
                self._stats.token_savings += entry.token_count
                return entry.value

            # Check static cache
            entry = self._static_cache.get(key)
            if entry and not entry.is_expired:
                entry.access_count += 1
                entry.last_accessed = time.time()
                self._static_cache.move_to_end(key)
                self._stats.hits += 1
                self._stats.token_savings += entry.token_count
                return entry.value

            # Clean up expired entry
            if entry and entry.is_expired:
                self._evict_entry(key, entry.layer)

            self._stats.misses += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        layer: CacheLayer = CacheLayer.DYNAMIC,
        token_count: int = 0,
    ) -> bool:
        """Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses layer default if not specified)
            layer: Cache layer
            token_count: Estimated token count for savings tracking

        Returns:
            True if successfully cached
        """
        # Determine TTL
        if ttl is None:
            ttl = self.TTL_STATIC if layer == CacheLayer.STATIC else self.TTL_CONVERSATION

        # Calculate size
        value_str = json.dumps(value) if not isinstance(value, str) else value
        size_bytes = len(value_str.encode())

        # Check if we need to evict
        async with self._lock:
            while self._get_total_size() + size_bytes > self.max_size_bytes:
                if not self._evict_lru():
                    logger.warning("Could not evict entry, cache may be corrupted")
                    return False

            entry = CacheEntry(
                key=key,
                value=value,
                layer=layer,
                created_at=time.time(),
                ttl=ttl,
                size_bytes=size_bytes,
                token_count=token_count or self._estimate_tokens(value_str),
            )

            cache = self._static_cache if layer == CacheLayer.STATIC else self._dynamic_cache
            cache[key] = entry

            # Register for semantic deduplication
            if self._deduplicator and isinstance(value, str):
                self._deduplicator.register(value, key)

            self._stats.total_entries = len(self._static_cache) + len(self._dynamic_cache)
            self._stats.total_size_bytes = self._get_total_size()

            return True

    async def set_system_prompt(self, task_type: str, prompt: str):
        """Cache a system prompt for a task type.

        Args:
            task_type: Task type identifier
            prompt: System prompt content
        """
        key = f"system_prompt:{task_type}"
        await self.set(key, prompt, ttl=self.TTL_STATIC, layer=CacheLayer.STATIC)
        self._system_prompts[task_type] = prompt

    async def get_system_prompt(self, task_type: str) -> str | None:
        """Get cached system prompt for a task type."""
        key = f"system_prompt:{task_type}"
        return await self.get(key)

    async def set_conversation(self, session_id: str, messages: list[dict[str, str]]):
        """Cache conversation history.

        Args:
            session_id: Session identifier
            messages: List of conversation messages
        """
        # Keep only last N turns
        messages = messages[-self.max_conversation_turns * 2 :]

        key = f"conversation:{session_id}"
        await self.set(key, messages, ttl=self.TTL_CONVERSATION, layer=CacheLayer.DYNAMIC)

    async def get_conversation(self, session_id: str) -> list[dict[str, str]]:
        """Get cached conversation history."""
        key = f"conversation:{session_id}"
        return await self.get(key) or []

    async def set_backtest_result(self, backtest_id: str, result: dict[str, Any]):
        """Cache backtest result.

        Args:
            backtest_id: Backtest identifier
            result: Backtest result data
        """
        key = f"backtest:{backtest_id}"
        await self.set(key, result, ttl=self.TTL_BACKTEST, layer=CacheLayer.DYNAMIC)

    async def get_backtest_result(self, backtest_id: str) -> dict[str, Any] | None:
        """Get cached backtest result."""
        key = f"backtest:{backtest_id}"
        return await self.get(key)

    async def set_ml_state(self, model_name: str, state: dict[str, Any]):
        """Cache ML model state.

        Args:
            model_name: Model identifier
            state: Model state data
        """
        key = f"ml_state:{model_name}"
        await self.set(key, state, ttl=self.TTL_ML_STATE, layer=CacheLayer.DYNAMIC)

    async def get_ml_state(self, model_name: str) -> dict[str, Any] | None:
        """Get cached ML model state."""
        key = f"ml_state:{model_name}"
        return await self.get(key)

    async def find_semantic_match(self, content: str) -> Any | None:
        """Find semantically similar cached content.

        Args:
            content: Content to match

        Returns:
            Similar cached content or None
        """
        if not self._deduplicator:
            return None

        matching_key = self._deduplicator.find_duplicate(content)
        if matching_key:
            return await self.get(matching_key)

        return None

    async def invalidate(self, pattern: str | None = None):
        """Invalidate cache entries.

        Args:
            pattern: Key pattern to match (None = invalidate all)
        """
        async with self._lock:
            if pattern is None:
                self._static_cache.clear()
                self._dynamic_cache.clear()
                self._stats = CacheStats()
                return

            # Pattern matching (simple prefix match)
            for cache in [self._static_cache, self._dynamic_cache]:
                keys_to_remove = [k for k in cache.keys() if k.startswith(pattern)]
                for key in keys_to_remove:
                    del cache[key]
                    self._stats.evictions += 1

    async def cleanup_expired(self):
        """Remove expired entries from all caches."""
        async with self._lock:
            total_removed = 0
            for cache in [self._static_cache, self._dynamic_cache]:
                expired_keys = [k for k, v in cache.items() if v.is_expired]
                for key in expired_keys:
                    del cache[key]
                    self._stats.evictions += 1
                    total_removed += 1

            if total_removed > 0:
                logger.debug(f"Cleaned up {total_removed} expired cache entries")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": self._stats.total_entries,
            "total_size_mb": self._stats.total_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization_pct": (
                self._stats.total_size_bytes / self.max_size_bytes * 100
                if self.max_size_bytes > 0
                else 0
            ),
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "hit_rate": self._stats.hit_rate,
            "evictions": self._stats.evictions,
            "token_savings": self._stats.token_savings,
            "estimated_savings_pct": self._estimate_savings_pct(),
            "static_entries": len(self._static_cache),
            "dynamic_entries": len(self._dynamic_cache),
        }

    def _get_total_size(self) -> int:
        """Get total size of all cache entries."""
        static_size = sum(e.size_bytes for e in self._static_cache.values())
        dynamic_size = sum(e.size_bytes for e in self._dynamic_cache.values())
        return static_size + dynamic_size

    def _evict_lru(self) -> bool:
        """Evict least recently used entry. Returns True if evicted."""
        # Prefer evicting from dynamic cache first
        if self._dynamic_cache:
            key, _ = self._dynamic_cache.popitem(last=False)
            self._stats.evictions += 1
            return True

        if self._static_cache:
            key, _ = self._static_cache.popitem(last=False)
            self._stats.evictions += 1
            return True

        return False

    def _evict_entry(self, key: str, layer: CacheLayer):
        """Evict a specific entry."""
        cache = self._static_cache if layer == CacheLayer.STATIC else self._dynamic_cache
        if key in cache:
            del cache[key]
            self._stats.evictions += 1

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough approximation: ~4 characters per token
        return len(text) // 4

    def _estimate_savings_pct(self) -> float:
        """Estimate token savings percentage."""
        total_tokens = self._stats.hits + self._stats.misses
        if total_tokens == 0:
            return 0.0

        # Assume each cache hit saves average tokens
        avg_tokens_per_entry = (
            sum(e.token_count for e in self._static_cache.values())
            + sum(e.token_count for e in self._dynamic_cache.values())
        ) / max(1, self._stats.total_entries)

        saved = self._stats.hits * avg_tokens_per_entry
        total_would_be = total_tokens * avg_tokens_per_entry

        return (saved / total_would_be * 100) if total_would_be > 0 else 0.0

    async def get_cache_statistics(self) -> dict[str, Any]:
        """
        Compute comprehensive cache effectiveness metrics.

        Returns:
            Dictionary with:
            - hit_rate_pct: Cache hit rate percentage
            - token_savings: Total tokens saved
            - cost_savings_usd: Estimated cost savings
            - avg_ttl_seconds: Average TTL of entries
            - entries_by_layer: Entry count per cache layer
            - access_distribution: Top accessed entries
        """
        async with self._lock:
            total_requests = self._stats.hits + self._stats.misses
            hit_rate = self._stats.hits / total_requests * 100 if total_requests > 0 else 0.0

            # Estimate cost savings (assuming Sonnet pricing: $3/M input)
            cost_per_token = 3.0 / 1_000_000
            cost_savings = self._stats.token_savings * cost_per_token

            # Average TTL
            all_entries = list(self._static_cache.values()) + list(self._dynamic_cache.values())
            avg_ttl = sum(e.ttl for e in all_entries) / len(all_entries) if all_entries else 0

            # Top accessed entries
            sorted_entries = sorted(
                all_entries,
                key=lambda e: e.access_count,
                reverse=True,
            )
            top_accessed = [
                {
                    "key": e.key,
                    "access_count": e.access_count,
                    "layer": e.layer.value,
                    "age_seconds": e.age_seconds,
                }
                for e in sorted_entries[:10]
            ]

            return {
                "hit_rate_pct": hit_rate,
                "total_hits": self._stats.hits,
                "total_misses": self._stats.misses,
                "total_requests": total_requests,
                "token_savings": self._stats.token_savings,
                "cost_savings_usd": cost_savings,
                "avg_ttl_seconds": avg_ttl,
                "entries_by_layer": {
                    "static": len(self._static_cache),
                    "dynamic": len(self._dynamic_cache),
                },
                "total_size_mb": self._get_total_size() / (1024 * 1024),
                "utilization_pct": (
                    self._get_total_size() / self.max_size_bytes * 100
                    if self.max_size_bytes > 0
                    else 0
                ),
                "top_accessed": top_accessed,
                "evictions": self._stats.evictions,
            }

    async def optimize_ttl(self) -> int:
        """
        Automatically optimize cache TTL based on usage patterns.

        Logic:
        - If hit rate > 60%: Increase TTL (+50%)
        - If hit rate < 20%: Decrease TTL (-30%)
        - If data changing frequently: Shorter TTL
        - Clamp between 30 seconds and 1 hour

        Returns:
            New recommended TTL in seconds.
        """
        total_requests = self._stats.hits + self._stats.misses
        if total_requests < 10:
            return self.TTL_CONVERSATION  # Not enough data

        hit_rate = self._stats.hits / total_requests

        current_ttl = self.TTL_CONVERSATION

        if hit_rate > 0.60:
            # High hit rate: increase TTL for more savings
            new_ttl = int(current_ttl * 1.5)
            logger.info(
                "Cache hit rate %.0f%% > 60%%: increasing TTL %ds -> %ds",
                hit_rate * 100,
                current_ttl,
                new_ttl,
            )
        elif hit_rate < 0.20:
            # Low hit rate: decrease TTL (data may be stale)
            new_ttl = int(current_ttl * 0.7)
            logger.info(
                "Cache hit rate %.0f%% < 20%%: decreasing TTL %ds -> %ds",
                hit_rate * 100,
                current_ttl,
                new_ttl,
            )
        else:
            new_ttl = current_ttl

        # Clamp to reasonable range (30s to 3600s)
        new_ttl = max(30, min(3600, new_ttl))

        # Apply the new TTL
        self.TTL_CONVERSATION = new_ttl

        return new_ttl

    async def get_optimization_recommendations(self) -> list[str]:
        """
        Analyze cache usage and recommend optimizations.

        Returns:
            List of actionable recommendation strings.
        """
        recommendations: list[str] = []

        stats = await self.get_cache_statistics()
        hit_rate = stats["hit_rate_pct"]
        total_requests = stats["total_requests"]

        if total_requests < 5:
            return ["Insufficient data for cache optimization recommendations."]

        # Hit rate recommendations
        if hit_rate < 20:
            recommendations.append(
                f"Cache hit rate is {hit_rate:.0f}%. "
                "Consider increasing cache TTL or caching more prompt prefixes."
            )
        elif hit_rate < 40:
            recommendations.append(
                f"Cache hit rate is {hit_rate:.0f}%. "
                "Moderate effectiveness. Review which data changes frequently."
            )

        # Size utilization
        util_pct = stats["utilization_pct"]
        if util_pct > 90:
            recommendations.append(
                f"Cache utilization at {util_pct:.0f}%. "
                "Consider increasing max_size_mb to reduce evictions."
            )
        elif util_pct < 10 and total_requests > 50:
            recommendations.append(
                f"Cache utilization only {util_pct:.0f}%. "
                "Consider caching more data or reducing max_size_mb."
            )

        # Eviction rate
        if stats["evictions"] > total_requests * 0.5:
            recommendations.append(
                f"High eviction rate ({stats['evictions']} evictions). "
                "Increase cache size or reduce TTL to prevent thrashing."
            )

        if not recommendations:
            recommendations.append(
                f"Cache performing well: {hit_rate:.0f}% hit rate, "
                f"saving ~{stats['token_savings']} tokens."
            )

        return recommendations


class PrefixCacheOptimizer:
    """Optimize prompts for Anthropic's prefix caching.

    Anthropic automatically caches common prefixes. This optimizer
    structures prompts to maximize prefix reuse:
    - Static content first (system prompts, documentation)
    - Shared context second (ML results, regime info)
    - Dynamic content last (specific query)
    """

    def __init__(self):
        self._static_prefixes: dict[str, str] = {}

    def register_static_prefix(self, name: str, content: str):
        """Register a static prefix for reuse."""
        self._static_prefixes[name] = content

    def build_optimized_prompt(
        self,
        static_parts: list[str],
        shared_context: str,
        dynamic_query: str,
    ) -> str:
        """Build a prompt optimized for prefix caching.

        Args:
            static_parts: Names of registered static prefixes
            shared_context: Shared context (ML results, etc.)
            dynamic_query: The specific query

        Returns:
            Optimized prompt string
        """
        parts = []

        # Add static prefixes first (most cacheable)
        for name in static_parts:
            if name in self._static_prefixes:
                parts.append(self._static_prefixes[name])

        # Add shared context
        if shared_context:
            parts.append(shared_context)

        # Add dynamic query last
        parts.append(dynamic_query)

        return "\n\n".join(parts)

    def estimate_cache_benefit(self, prompt: str) -> dict[str, Any]:
        """Estimate caching benefit for a prompt."""
        total_chars = len(prompt)

        # Find how much matches static prefixes
        cached_chars = 0
        for prefix in self._static_prefixes.values():
            if prompt.startswith(prefix):
                cached_chars = max(cached_chars, len(prefix))
                break

        return {
            "total_chars": total_chars,
            "cacheable_chars": cached_chars,
            "cache_ratio": cached_chars / total_chars if total_chars > 0 else 0,
            "estimated_token_savings": cached_chars // 4,
        }
