"""
Telegram API integration for social signal extraction.

Provides real social metrics for trading signals:
- Telegram group member counts
- Message velocity
- Sentiment analysis of messages
- User growth tracking

These metrics populate feature indices 43-49 in the HotPath feature vector.

Usage:
    from coldpath.services.telegram_signals import TelegramSignalCollector

    collector = TelegramSignalCollector()

    # Get metrics for a token's Telegram group
    metrics = await collector.get_group_metrics("https://t.me/SomeTokenGroup")

    # Extract sentiment from recent messages
    sentiment = await collector.analyze_sentiment("SomeTokenGroup", limit=100)
"""

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger(__name__)

TELEGRAM_AVAILABLE = False
try:
    from telethon import TelegramClient  # noqa: F401

    TELEGRAM_AVAILABLE = True
except ImportError:
    logger.debug("Telethon not available. Install with: pip install telethon")


@dataclass
class TelegramMetrics:
    """Social metrics from Telegram."""

    group_username: str
    timestamp: datetime

    member_count: int = 0
    member_change_24h: int = 0
    member_growth_rate: float = 0.0

    messages_24h: int = 0
    messages_1h: int = 0
    message_velocity: float = 0.0
    active_users_24h: int = 0

    avg_reactions_per_message: float = 0.0
    avg_forwards_per_message: float = 0.0
    engagement_score: float = 0.0

    sentiment_score: float = 0.5
    bullish_ratio: float = 0.5
    fear_greed_index: float = 0.5

    bot_detected: bool = False
    spam_score: float = 0.0
    authenticity_score: float = 1.0

    raw_messages: list[dict[str, Any]] = field(default_factory=list)

    def to_feature_dict(self) -> dict[str, float]:
        """Convert to features for ML model (indices 43-49)."""
        return {
            "twitter_mention_velocity": 0.0,
            "twitter_sentiment_score": 0.0,
            "telegram_user_growth": np.tanh(self.member_growth_rate / 100.0),
            "telegram_message_velocity": min(1.0, self.message_velocity / 100.0),
            "discord_invite_activity": 0.0,
            "influencer_mention_flag": 1.0 if self.engagement_score > 0.7 else -1.0,
            "social_authenticity_score": self.authenticity_score,
        }


class TelegramSignalCollector:
    """Collects social signals from Telegram groups."""

    BULLISH_KEYWORDS = {
        "moon",
        "pump",
        "bull",
        "bullish",
        "buy",
        "long",
        "rocket",
        "gem",
        "undervalued",
        "breakout",
        "rally",
        "surge",
        "accumulate",
        "hold",
        "hodl",
        "support",
        "bounce",
        "reversal",
        "uptrend",
        "ath",
        "gains",
    }

    BEARISH_KEYWORDS = {
        "dump",
        "crash",
        "bear",
        "bearish",
        "sell",
        "short",
        "scam",
        "rug",
        "overvalued",
        "breakdown",
        "decline",
        "drop",
        "exit",
        "rekt",
        "resistance",
        "downtrend",
        "loss",
        "warning",
        "avoid",
    }

    def __init__(
        self,
        api_id: str | None = None,
        api_hash: str | None = None,
    ):
        self.api_id = api_id or os.getenv("TELEGRAM_API_ID")
        self.api_hash = api_hash or os.getenv("TELEGRAM_API_HASH")
        self._client: Any | None = None
        self._session: Any | None = None
        self._metrics_cache: dict[str, tuple[datetime, TelegramMetrics]] = {}
        self._cache_ttl_seconds = 300

        logger.info(f"TelegramSignalCollector initialized (Telethon: {TELEGRAM_AVAILABLE})")

    def _extract_username(self, url: str) -> str:
        """Extract username from various URL formats."""
        if url.startswith("@"):
            return url[1:]

        if "t.me/" in url:
            parsed = urlparse(url)
            path = parsed.path.lstrip("/")
            return path.split("/")[0].split("?")[0]

        url = url.strip()
        if url.startswith("https://") or url.startswith("http://"):
            url = url.split("/")[-1]

        return url.replace("@", "")

    async def get_group_metrics(
        self,
        group_url: str,
        include_messages: bool = True,
        message_limit: int = 100,
    ) -> TelegramMetrics:
        """Get comprehensive metrics for a Telegram group."""
        username = self._extract_username(group_url)

        if username in self._metrics_cache:
            cached_time, cached_metrics = self._metrics_cache[username]
            if datetime.now() - cached_time < timedelta(seconds=self._cache_ttl_seconds):
                return cached_metrics

        if TELEGRAM_AVAILABLE and self._client:
            metrics = await self._get_metrics_telethon(username, include_messages, message_limit)
        else:
            metrics = await self._get_metrics_web(username)

        self._metrics_cache[username] = (datetime.now(), metrics)
        return metrics

    async def _get_metrics_telethon(
        self,
        username: str,
        include_messages: bool,
        message_limit: int,
    ) -> TelegramMetrics:
        """Get metrics using Telethon API."""
        metrics = TelegramMetrics(
            group_username=username,
            timestamp=datetime.now(),
        )

        try:
            entity = await self._client.get_entity(username)

            if hasattr(entity, "participants_count"):
                metrics.member_count = entity.participants_count or 0

            if include_messages:
                messages = await self._client.get_messages(entity, limit=message_limit)
                metrics.raw_messages = [
                    {
                        "id": m.id,
                        "text": m.text or "",
                        "date": m.date.isoformat() if m.date else None,
                    }
                    for m in messages
                    if m.text
                ]

                now = datetime.now()
                messages_24h = [
                    m
                    for m in messages
                    if m.date and (now - m.date.replace(tzinfo=None)).total_seconds() < 86400
                ]
                messages_1h = [
                    m
                    for m in messages
                    if m.date and (now - m.date.replace(tzinfo=None)).total_seconds() < 3600
                ]

                metrics.messages_24h = len(messages_24h)
                metrics.messages_1h = len(messages_1h)
                metrics.message_velocity = (
                    metrics.messages_24h / 24.0 if metrics.messages_24h > 0 else 0
                )

                if messages_24h:
                    unique_users = set()
                    for m in messages_24h:
                        if m.sender_id:
                            unique_users.add(m.sender_id)
                    metrics.active_users_24h = len(unique_users)

                await self._analyze_sentiment(metrics, messages)

        except Exception as e:
            logger.debug(f"Telethon fetch failed for {username}: {e}")

        return metrics

    async def _get_metrics_web(self, username: str) -> TelegramMetrics:
        """Get metrics using web scraping (fallback)."""
        metrics = TelegramMetrics(
            group_username=username,
            timestamp=datetime.now(),
        )

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                url = f"https://t.me/{username}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        html = await resp.text()

                        match = re.search(r'"member_count":(\d+)', html)
                        if match:
                            metrics.member_count = int(match.group(1))
                        else:
                            match = re.search(r"(\d[\d,]*)\s*members?", html, re.I)
                            if match:
                                metrics.member_count = int(match.group(1).replace(",", ""))

        except Exception as e:
            logger.debug(f"Web fetch failed for {username}: {e}")

        return metrics

    async def _analyze_sentiment(self, metrics: TelegramMetrics, messages: list[Any]):
        """Analyze sentiment of messages."""
        if not messages:
            return

        bullish_count = 0
        bearish_count = 0
        total_with_sentiment = 0

        for msg in messages:
            text = (msg.text or "").lower()
            if not text:
                continue

            has_bullish = any(kw in text for kw in self.BULLISH_KEYWORDS)
            has_bearish = any(kw in text for kw in self.BEARISH_KEYWORDS)

            if has_bullish or has_bearish:
                total_with_sentiment += 1
                if has_bullish and not has_bearish:
                    bullish_count += 1
                elif has_bearish and not has_bullish:
                    bearish_count += 1

        if total_with_sentiment > 0:
            metrics.bullish_ratio = bullish_count / total_with_sentiment
            metrics.sentiment_score = 0.5 + (metrics.bullish_ratio - 0.5) * 0.8
            metrics.sentiment_score = max(0.0, min(1.0, metrics.sentiment_score))

        total_messages = len([m for m in messages if m.text])
        if total_messages > 0:
            sentiment_messages = bullish_count + bearish_count
            metrics.engagement_score = sentiment_messages / total_messages

    async def get_token_social_features(
        self,
        telegram_url: str | None = None,
    ) -> dict[str, float]:
        """Get social features for feature vector (indices 43-49)."""
        default_features = {
            "twitter_mention_velocity": 0.0,
            "twitter_sentiment_score": 0.0,
            "telegram_user_growth": 0.0,
            "telegram_message_velocity": 0.0,
            "discord_invite_activity": 0.0,
            "influencer_mention_flag": -1.0,
            "social_authenticity_score": 0.5,
        }

        if not telegram_url:
            return default_features

        try:
            metrics = await self.get_group_metrics(telegram_url, include_messages=True)
            return metrics.to_feature_dict()
        except Exception as e:
            logger.debug(f"Failed to get social features: {e}")
            return default_features

    async def close(self):
        """Close connections."""
        if self._client:
            await self._client.disconnect()
            self._client = None
