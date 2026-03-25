"""
Social Features Service for HotPath integration.

Provides real social metrics for feature indices 43-49 by:
1. Collecting signals from Telegram (and extensible to other sources)
2. Exposing via gRPC for HotPath consumption
3. Caching for low-latency feature retrieval

Feature Index Mapping (indices 43-49):
    43: twitter_mention_velocity (proxy from engagement)
    44: twitter_sentiment_score (proxy from sentiment)
    45: telegram_user_growth (real from Telegram API)
    46: telegram_message_velocity (real from Telegram API)
    47: discord_invite_activity (always 0, no Discord API)
    48: influencer_mention_flag (proxy from engagement spikes)
    49: social_authenticity_score (calculated from multiple signals)

Usage:
    from coldpath.services.social_features_service import SocialFeaturesService

    service = SocialFeaturesService()
    await service.start()

    # Get features for a token
    features = await service.get_social_features("TokenMint123...")

    # For HotPath integration
    feature_vector = service.get_feature_array("TokenMint123...")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from coldpath.services.telegram_signals import TelegramMetrics, TelegramSignalCollector

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


@dataclass
class SocialFeatures:
    """Social features for a token (indices 43-49)."""

    token_mint: str
    timestamp: datetime

    # Index 43: Twitter mention velocity (normalized -1 to 1)
    twitter_mention_velocity: float = 0.0

    # Index 44: Twitter sentiment score (normalized -1 to 1)
    twitter_sentiment_score: float = 0.0

    # Index 45: Telegram user growth rate (normalized -1 to 1)
    telegram_user_growth: float = 0.0

    # Index 46: Telegram message velocity (normalized -1 to 1)
    telegram_message_velocity: float = 0.0

    # Index 47: Discord invite activity (always 0 - no API connected)
    discord_invite_activity: float = 0.0

    # Index 48: Influencer mention flag (1.0 or -1.0)
    influencer_mention_flag: float = -1.0

    # Index 49: Social authenticity score (0 to 1, then mapped to -1 to 1)
    social_authenticity_score: float = 0.5

    # Additional metadata
    telegram_group: str | None = None
    data_freshness_ms: int = 0
    source: str = "unknown"

    def to_feature_array(self) -> list[float]:
        """Convert to feature array for indices 43-49."""
        return [
            self.twitter_mention_velocity,
            self.twitter_sentiment_score,
            self.telegram_user_growth,
            self.telegram_message_velocity,
            self.discord_invite_activity,
            self.influencer_mention_flag,
            self.social_authenticity_score * 2 - 1,  # Map 0-1 to -1 to 1
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_mint": self.token_mint,
            "timestamp": self.timestamp.isoformat(),
            "twitter_mention_velocity": self.twitter_mention_velocity,
            "twitter_sentiment_score": self.twitter_sentiment_score,
            "telegram_user_growth": self.telegram_user_growth,
            "telegram_message_velocity": self.telegram_message_velocity,
            "discord_invite_activity": self.discord_invite_activity,
            "influencer_mention_flag": self.influencer_mention_flag,
            "social_authenticity_score": self.social_authenticity_score,
            "telegram_group": self.telegram_group,
            "data_freshness_ms": self.data_freshness_ms,
            "source": self.source,
        }


@dataclass
class TokenSocialMapping:
    """Maps token mint to Telegram group and other social handles."""

    token_mint: str
    token_symbol: str
    telegram_group: str | None = None
    twitter_handle: str | None = None
    discord_invite: str | None = None
    discovered_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class SocialFeaturesService:
    """Service for providing social features to HotPath.

    Architecture:
        ┌─────────────────┐
        │ Token Discovery │
        │ (HotPath)       │
        └────────┬────────┘
                 │ new token
                 ▼
        ┌─────────────────┐     ┌──────────────────┐
        │ Social Features │────►│ Telegram API     │
        │ Service         │     │ (Telethon)       │
        └────────┬────────┘     └──────────────────┘
                 │ gRPC
                 ▼
        ┌─────────────────┐
        │ HotPath         │
        │ Feature Builder │
        └─────────────────┘
    """

    FEATURE_INDICES = {
        "twitter_mention_velocity": 43,
        "twitter_sentiment_score": 44,
        "telegram_user_growth": 45,
        "telegram_message_velocity": 46,
        "discord_invite_activity": 47,
        "influencer_mention_flag": 48,
        "social_authenticity_score": 49,
    }

    def __init__(
        self,
        cache_ttl_seconds: int = 300,
        max_cache_size: int = 10000,
        telegram_api_id: str | None = None,
        telegram_api_hash: str | None = None,
    ):
        self.cache_ttl = cache_ttl_seconds
        self.max_cache_size = max_cache_size

        self._cache: dict[str, SocialFeatures] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._token_mappings: dict[str, TokenSocialMapping] = {}
        self._telegram_collector: TelegramSignalCollector | None = None
        self._running = False
        self._refresh_task: asyncio.Task | None = None

        if telegram_api_id and telegram_api_hash:
            self._telegram_collector = TelegramSignalCollector(
                api_id=telegram_api_id,
                api_hash=telegram_api_hash,
            )

    async def start(self):
        """Start the social features service."""
        if self._running:
            return

        self._running = True
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info("Social features service started")

    async def stop(self):
        """Stop the social features service."""
        self._running = False

        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

        logger.info("Social features service stopped")

    async def _refresh_loop(self):
        """Periodically refresh stale cache entries."""
        while self._running:
            try:
                await asyncio.sleep(60)
                await self._refresh_stale_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in refresh loop: {e}")

    async def _refresh_stale_entries(self):
        """Refresh cache entries that are older than TTL."""
        now = time.time()
        stale_tokens = [
            token for token, ts in self._cache_timestamps.items() if now - ts > self.cache_ttl
        ]

        for token in stale_tokens[:100]:
            try:
                await self._refresh_token(token)
            except Exception as e:
                logger.warning(f"Failed to refresh {token}: {e}")

    async def _refresh_token(self, token_mint: str) -> SocialFeatures | None:
        """Refresh social features for a single token."""
        mapping = self._token_mappings.get(token_mint)
        if not mapping or not mapping.telegram_group:
            return None

        if not self._telegram_collector:
            return None

        try:
            metrics = await self._telegram_collector.get_group_metrics(mapping.telegram_group)
            features = self._convert_telegram_metrics(token_mint, metrics)
            self._cache[token_mint] = features
            self._cache_timestamps[token_mint] = time.time()
            return features
        except Exception as e:
            logger.warning(f"Failed to get Telegram metrics for {token_mint}: {e}")
            return None

    def _convert_telegram_metrics(
        self, token_mint: str, metrics: TelegramMetrics
    ) -> SocialFeatures:
        """Convert Telegram metrics to SocialFeatures."""
        feature_dict = metrics.to_feature_dict()

        return SocialFeatures(
            token_mint=token_mint,
            timestamp=datetime.now(),
            twitter_mention_velocity=feature_dict.get("twitter_mention_velocity", 0.0),
            twitter_sentiment_score=feature_dict.get("twitter_sentiment_score", 0.0),
            telegram_user_growth=feature_dict.get("telegram_user_growth", 0.0),
            telegram_message_velocity=feature_dict.get("telegram_message_velocity", 0.0),
            discord_invite_activity=feature_dict.get("discord_invite_activity", 0.0),
            influencer_mention_flag=feature_dict.get("influencer_mention_flag", -1.0),
            social_authenticity_score=feature_dict.get("social_authenticity_score", 0.5),
            telegram_group=metrics.group_username,
            data_freshness_ms=0,
            source="telegram",
        )

    async def register_token(
        self,
        token_mint: str,
        token_symbol: str,
        telegram_group: str | None = None,
        twitter_handle: str | None = None,
    ):
        """Register a token with its social handles."""
        mapping = TokenSocialMapping(
            token_mint=token_mint,
            token_symbol=token_symbol,
            telegram_group=telegram_group,
            twitter_handle=twitter_handle,
        )
        self._token_mappings[token_mint] = mapping

        if telegram_group:
            await self._refresh_token(token_mint)

        logger.debug(f"Registered token {token_symbol} with social handles")

    async def get_social_features(self, token_mint: str) -> SocialFeatures:
        """Get social features for a token.

        Returns cached features if available and fresh, otherwise fetches new.
        """
        now = time.time()

        if token_mint in self._cache:
            cached = self._cache[token_mint]
            cache_age = now - self._cache_timestamps.get(token_mint, 0)

            if cache_age < self.cache_ttl:
                cached.data_freshness_ms = int(cache_age * 1000)
                return cached

        if token_mint in self._token_mappings:
            features = await self._refresh_token(token_mint)
            if features:
                return features

        return SocialFeatures(
            token_mint=token_mint,
            timestamp=datetime.now(),
            source="fallback",
        )

    def get_feature_array(self, token_mint: str) -> list[float]:
        """Get feature array for indices 43-49 (synchronous, for HotPath).

        Returns cached value or default zeros. This is a fast path for
        HotPath inference that can't await.
        """
        if token_mint in self._cache:
            return self._cache[token_mint].to_feature_array()

        return [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]

    def get_feature_array_numpy(self, token_mint: str) -> np.ndarray:
        """Get feature array as numpy array."""
        return np.array(self.get_feature_array(token_mint), dtype=np.float64)

    async def batch_get_features(self, token_mints: list[str]) -> dict[str, SocialFeatures]:
        """Get social features for multiple tokens."""
        results = {}
        for mint in token_mints:
            results[mint] = await self.get_social_features(mint)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        now = time.time()
        fresh_count = sum(1 for ts in self._cache_timestamps.values() if now - ts < self.cache_ttl)

        return {
            "running": self._running,
            "cache_size": len(self._cache),
            "fresh_cache_entries": fresh_count,
            "registered_tokens": len(self._token_mappings),
            "telegram_available": self._telegram_collector is not None,
            "cache_ttl_seconds": self.cache_ttl,
        }

    def clear_cache(self):
        """Clear the feature cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Social features cache cleared")


# gRPC servicer for HotPath integration
class SocialFeaturesServicer:
    """gRPC servicer for serving social features to HotPath."""

    def __init__(self, service: SocialFeaturesService):
        self.service = service

    async def GetSocialFeatures(
        self,
        request: Any,
        context: Any,
    ) -> Any:
        """gRPC handler for GetSocialFeatures."""
        from coldpath.ipc import hotpath_pb2

        features = await self.service.get_social_features(request.token_mint)

        return hotpath_pb2.SocialFeaturesResponse(
            token_mint=features.token_mint,
            twitter_mention_velocity=features.twitter_mention_velocity,
            twitter_sentiment_score=features.twitter_sentiment_score,
            telegram_user_growth=features.telegram_user_growth,
            telegram_message_velocity=features.telegram_message_velocity,
            discord_invite_activity=features.discord_invite_activity,
            influencer_mention_flag=features.influencer_mention_flag,
            social_authenticity_score=features.social_authenticity_score,
            data_freshness_ms=features.data_freshness_ms,
            source=features.source,
        )

    async def RegisterTokenSocial(
        self,
        request: Any,
        context: Any,
    ) -> Any:
        """gRPC handler for RegisterTokenSocial."""
        from coldpath.ipc import hotpath_pb2

        await self.service.register_token(
            token_mint=request.token_mint,
            token_symbol=request.token_symbol,
            telegram_group=request.telegram_group if request.telegram_group else None,
            twitter_handle=request.twitter_handle if request.twitter_handle else None,
        )

        return hotpath_pb2.RegisterTokenSocialResponse(success=True)


def add_social_features_servicer(server: Any, service: SocialFeaturesService):
    """Add social features servicer to a gRPC server."""
    try:
        from coldpath.ipc import (
            hotpath_pb2,  # noqa: F401
            hotpath_pb2_grpc,
        )

        servicer = SocialFeaturesServicer(service)
        hotpath_pb2_grpc.add_SocialFeaturesServiceServicer_to_server(servicer, server)
        logger.info("Social features gRPC servicer added")
    except ImportError:
        logger.warning("gRPC protobuf stubs not available, skipping servicer registration")
