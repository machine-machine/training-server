"""
Inference Pipeline with multi-level caching.

Implements multi-level caching for fast ML scoring:
L1: Score cache (DashMap-like, 3s TTL) - per mint+model_version
L2: Feature cache (DashMap-like, 5-60s adaptive) - volatility-based TTL
L3: Enrichment cache (Redis-like, 1-5min) - per mint
L4: Dune query cache (Parquet-like, 1h-1d) - query_hash

Latency target: <50ms cache hit, <100ms miss
"""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels in order of speed."""

    L1_SCORE = "l1_score"  # Fastest: final scores
    L2_FEATURE = "l2_feature"  # Fast: computed features
    L3_ENRICHMENT = "l3_enrichment"  # Medium: external enrichment
    L4_QUERY = "l4_query"  # Slow: query results


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    level: CacheLevel
    created_at: float
    ttl: float
    hit_count: int = 0
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl

    @property
    def age_ms(self) -> float:
        return (time.time() - self.created_at) * 1000


@dataclass
class CacheStats:
    """Statistics for a cache level."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    avg_latency_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class ScoringResult:
    """Result of scoring a token."""

    mint: str
    score: float
    confidence: float
    signal: str
    features_used: int
    cache_hits: dict[str, bool]
    latency_ms: float
    model_version: int
    timestamp_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "score": self.score,
            "confidence": self.confidence,
            "signal": self.signal,
            "features_used": self.features_used,
            "cache_hits": self.cache_hits,
            "latency_ms": self.latency_ms,
            "model_version": self.model_version,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""

    # L1 Score cache
    l1_ttl_seconds: float = 3.0
    l1_max_entries: int = 10000

    # L2 Feature cache (adaptive TTL based on volatility)
    l2_ttl_min_seconds: float = 5.0
    l2_ttl_max_seconds: float = 60.0
    l2_max_entries: int = 50000

    # L3 Enrichment cache
    l3_ttl_seconds: float = 60.0  # 1 minute default
    l3_max_entries: int = 20000

    # L4 Query cache
    l4_ttl_seconds: float = 3600.0  # 1 hour default
    l4_max_entries: int = 1000

    # Model settings
    current_model_version: int = 1
    feature_count: int = 50

    # Latency targets
    target_latency_cache_hit_ms: float = 50.0
    target_latency_cache_miss_ms: float = 100.0

    # Volatility thresholds for adaptive TTL
    high_volatility_threshold: float = 0.3
    low_volatility_threshold: float = 0.1


class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_entries: int, default_ttl: float):
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = CacheStats()

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hit_count += 1
            self._stats.hits += 1

            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
        level: CacheLevel = CacheLevel.L1_SCORE,
    ):
        """Set value in cache."""
        ttl = ttl or self.default_ttl

        async with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_entries:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1

            entry = CacheEntry(
                key=key,
                value=value,
                level=level,
                created_at=time.time(),
                ttl=ttl,
            )

            self._cache[key] = entry

    async def invalidate(self, key: str) -> bool:
        """Invalidate a specific key."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self):
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class InferencePipeline:
    """Multi-level caching inference pipeline for ML scoring.

    Implements 4-level cache hierarchy:
    - L1: Score cache (3s TTL) - final scores by mint+model_version
    - L2: Feature cache (5-60s adaptive) - computed features
    - L3: Enrichment cache (1-5min) - external data enrichment
    - L4: Query cache (1h-1d) - expensive query results

    Example:
        pipeline = InferencePipeline(config)
        result = await pipeline.score_token(mint, raw_data)
        print(f"Score: {result.score}, Latency: {result.latency_ms}ms")
    """

    def __init__(
        self,
        config: InferenceConfig | None = None,
        feature_extractor: Callable[[dict], np.ndarray] | None = None,
        model_scorer: Callable[[np.ndarray], tuple[float, float]] | None = None,
    ):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration
            feature_extractor: Function to extract features from raw data
            model_scorer: Function to score features (returns score, confidence)
        """
        self.config = config or InferenceConfig()

        # Initialize cache levels
        self._l1_cache = LRUCache(
            max_entries=self.config.l1_max_entries,
            default_ttl=self.config.l1_ttl_seconds,
        )
        self._l2_cache = LRUCache(
            max_entries=self.config.l2_max_entries,
            default_ttl=self.config.l2_max_seconds,
        )
        self._l3_cache = LRUCache(
            max_entries=self.config.l3_max_entries,
            default_ttl=self.config.l3_ttl_seconds,
        )
        self._l4_cache = LRUCache(
            max_entries=self.config.l4_max_entries,
            default_ttl=self.config.l4_ttl_seconds,
        )

        # Model components
        self.feature_extractor = feature_extractor
        self.model_scorer = model_scorer

        # Statistics
        self._total_requests = 0
        self._total_latency_ms = 0.0

    async def score_token(
        self,
        mint: str,
        raw_data: dict[str, Any] | None = None,
        features: np.ndarray | None = None,
        enrichment_fn: Callable[[str], dict] | None = None,
        volatility: float | None = None,
    ) -> ScoringResult:
        """Score a token with multi-level caching.

        Args:
            mint: Token mint address
            raw_data: Raw token data for feature extraction
            features: Pre-computed features (skip L2)
            enrichment_fn: Function to fetch enrichment data
            volatility: Current volatility for adaptive TTL

        Returns:
            ScoringResult with score and metadata
        """
        start_time = time.time()
        self._total_requests += 1

        cache_hits = {
            "l1_score": False,
            "l2_feature": False,
            "l3_enrichment": False,
            "l4_query": False,
        }

        # L1: Check score cache
        score_key = self._score_key(mint)
        cached_score = await self._l1_cache.get(score_key)

        if cached_score is not None:
            cache_hits["l1_score"] = True
            latency_ms = (time.time() - start_time) * 1000
            return ScoringResult(
                mint=mint,
                score=cached_score["score"],
                confidence=cached_score["confidence"],
                signal=cached_score["signal"],
                features_used=cached_score.get("features_used", 0),
                cache_hits=cache_hits,
                latency_ms=latency_ms,
                model_version=self.config.current_model_version,
                timestamp_ms=int(time.time() * 1000),
            )

        # L2: Check/compute features
        if features is None:
            feature_key = self._feature_key(mint)
            cached_features = await self._l2_cache.get(feature_key)

            if cached_features is not None:
                cache_hits["l2_feature"] = True
                features = np.array(cached_features)
            elif raw_data and self.feature_extractor:
                # Check L3: enrichment
                if enrichment_fn:
                    enrichment_key = f"enrichment:{mint}"
                    enrichment = await self._l3_cache.get(enrichment_key)

                    if enrichment is not None:
                        cache_hits["l3_enrichment"] = True
                        raw_data.update(enrichment)
                    else:
                        try:
                            enrichment = enrichment_fn(mint)
                            await self._l3_cache.set(
                                enrichment_key,
                                enrichment,
                                level=CacheLevel.L3_ENRICHMENT,
                            )
                            raw_data.update(enrichment)
                        except Exception as e:
                            logger.warning(f"Enrichment failed for {mint}: {e}")

                features = self.feature_extractor(raw_data)

                # Cache features with adaptive TTL
                feature_ttl = self._compute_adaptive_ttl(volatility)
                await self._l2_cache.set(
                    feature_key,
                    features.tolist(),
                    ttl=feature_ttl,
                    level=CacheLevel.L2_FEATURE,
                )

        # Score with model
        if features is not None and self.model_scorer:
            score, confidence = self.model_scorer(features)
            signal = self._score_to_signal(score, confidence)
            features_used = len(features)
        else:
            # Fallback: no model available
            score = 0.5
            confidence = 0.0
            signal = "HOLD"
            features_used = 0

        # Cache score in L1
        score_data = {
            "score": score,
            "confidence": confidence,
            "signal": signal,
            "features_used": features_used,
        }
        await self._l1_cache.set(
            score_key,
            score_data,
            level=CacheLevel.L1_SCORE,
        )

        latency_ms = (time.time() - start_time) * 1000
        self._total_latency_ms += latency_ms

        return ScoringResult(
            mint=mint,
            score=score,
            confidence=confidence,
            signal=signal,
            features_used=features_used,
            cache_hits=cache_hits,
            latency_ms=latency_ms,
            model_version=self.config.current_model_version,
            timestamp_ms=int(time.time() * 1000),
        )

    async def batch_score(
        self,
        mints: list[str],
        raw_data_map: dict[str, dict[str, Any]],
        max_concurrent: int = 10,
    ) -> list[ScoringResult]:
        """Score multiple tokens concurrently.

        Args:
            mints: List of mint addresses
            raw_data_map: Map of mint -> raw data
            max_concurrent: Maximum concurrent scoring tasks

        Returns:
            List of ScoringResult
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def score_with_limit(mint: str) -> ScoringResult:
            async with semaphore:
                return await self.score_token(
                    mint=mint,
                    raw_data=raw_data_map.get(mint),
                )

        tasks = [score_with_limit(mint) for mint in mints]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, ScoringResult):
                valid_results.append(result)
            else:
                logger.error(f"Batch scoring error: {result}")

        return valid_results

    async def cache_query_result(
        self,
        query_hash: str,
        result: Any,
        ttl_hours: float = 1.0,
    ):
        """Cache an expensive query result in L4.

        Args:
            query_hash: Hash of the query
            result: Query result to cache
            ttl_hours: TTL in hours
        """
        key = f"query:{query_hash}"
        await self._l4_cache.set(
            key,
            result,
            ttl=ttl_hours * 3600,
            level=CacheLevel.L4_QUERY,
        )

    async def get_cached_query(self, query_hash: str) -> Any | None:
        """Get cached query result from L4."""
        key = f"query:{query_hash}"
        return await self._l4_cache.get(key)

    async def invalidate_token(self, mint: str):
        """Invalidate all cache entries for a token."""
        await self._l1_cache.invalidate(self._score_key(mint))
        await self._l2_cache.invalidate(self._feature_key(mint))
        await self._l3_cache.invalidate(f"enrichment:{mint}")

    async def update_model_version(self, new_version: int):
        """Update model version and invalidate score cache."""
        self.config.current_model_version = new_version
        await self._l1_cache.clear()  # Clear all scores
        logger.info(f"Model version updated to {new_version}, L1 cache cleared")

    def _score_key(self, mint: str) -> str:
        """Generate L1 cache key for score."""
        return f"score:{mint}:v{self.config.current_model_version}"

    def _feature_key(self, mint: str) -> str:
        """Generate L2 cache key for features."""
        return f"features:{mint}"

    def _compute_adaptive_ttl(self, volatility: float | None) -> float:
        """Compute adaptive TTL based on volatility.

        Higher volatility = shorter TTL (faster refresh)
        Lower volatility = longer TTL (save compute)
        """
        if volatility is None:
            return self.config.l2_ttl_max_seconds / 2

        if volatility >= self.config.high_volatility_threshold:
            return self.config.l2_ttl_min_seconds
        elif volatility <= self.config.low_volatility_threshold:
            return self.config.l2_ttl_max_seconds
        else:
            # Linear interpolation
            ratio = (volatility - self.config.low_volatility_threshold) / (
                self.config.high_volatility_threshold - self.config.low_volatility_threshold
            )
            return self.config.l2_ttl_max_seconds - ratio * (
                self.config.l2_ttl_max_seconds - self.config.l2_ttl_min_seconds
            )

    def _score_to_signal(self, score: float, confidence: float) -> str:
        """Convert score and confidence to trading signal."""
        if confidence < 0.5:
            return "HOLD"
        elif score >= 0.7:
            return "BUY"
        elif score <= 0.3:
            return "SELL"
        else:
            return "HOLD"

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        avg_latency = (
            self._total_latency_ms / self._total_requests if self._total_requests > 0 else 0
        )

        return {
            "total_requests": self._total_requests,
            "avg_latency_ms": avg_latency,
            "model_version": self.config.current_model_version,
            "l1_cache": {
                "hit_rate": self._l1_cache.get_stats().hit_rate,
                "hits": self._l1_cache.get_stats().hits,
                "misses": self._l1_cache.get_stats().misses,
            },
            "l2_cache": {
                "hit_rate": self._l2_cache.get_stats().hit_rate,
                "hits": self._l2_cache.get_stats().hits,
                "misses": self._l2_cache.get_stats().misses,
            },
            "l3_cache": {
                "hit_rate": self._l3_cache.get_stats().hit_rate,
                "hits": self._l3_cache.get_stats().hits,
                "misses": self._l3_cache.get_stats().misses,
            },
            "l4_cache": {
                "hit_rate": self._l4_cache.get_stats().hit_rate,
                "hits": self._l4_cache.get_stats().hits,
                "misses": self._l4_cache.get_stats().misses,
            },
        }

    async def warmup(self, mints: list[str], raw_data_map: dict[str, dict]):
        """Warm up caches with initial data.

        Args:
            mints: List of mints to warm up
            raw_data_map: Raw data for each mint
        """
        logger.info(f"Warming up cache with {len(mints)} tokens")

        # Score all tokens to populate caches
        await self.batch_score(mints, raw_data_map, max_concurrent=20)

        logger.info("Cache warmup complete")


def compute_query_hash(query: str, params: dict[str, Any]) -> str:
    """Compute hash for a query and its parameters."""
    key_str = f"{query}:{sorted(params.items())}"
    return hashlib.md5(key_str.encode()).hexdigest()
