"""
Social Sentiment Tracker.

Tracks social signals from Twitter, Telegram, and Discord for token analysis.
Provides sentiment scoring, mention velocity, and authenticity metrics.

Social signals (7 features):
- twitter_mention_velocity: Mentions per hour
- twitter_sentiment_score: -1 (negative) to 1 (positive)
- telegram_user_growth: User growth rate
- telegram_message_velocity: Messages per hour
- discord_invite_activity: Invite activity score
- influencer_mention_flag: Whether mentioned by influencers
- social_authenticity_score: 0 (fake) to 1 (authentic)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Check for optional dependencies
TWEEPY_AVAILABLE = False
TELETHON_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    import tweepy

    TWEEPY_AVAILABLE = True
except ImportError:
    pass

try:
    from telethon import TelegramClient

    TELETHON_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class TwitterSignals:
    """Twitter-specific signals."""

    mention_velocity: float = 0.0  # Mentions per hour
    sentiment_score: float = 0.0  # -1 to 1
    follower_count: int = 0
    account_age_days: int = 0
    verified_account: bool = False
    influencer_mentions: list[str] = field(default_factory=list)
    recent_mentions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_velocity": self.mention_velocity,
            "sentiment_score": self.sentiment_score,
            "follower_count": self.follower_count,
            "account_age_days": self.account_age_days,
            "verified_account": self.verified_account,
            "influencer_mention_count": len(self.influencer_mentions),
        }


@dataclass
class TelegramSignals:
    """Telegram-specific signals."""

    user_growth: float = 0.0  # Users gained per hour
    message_velocity: float = 0.0  # Messages per hour
    member_count: int = 0
    admin_count: int = 0
    bot_ratio: float = 0.0  # Estimated bot ratio
    sentiment_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_growth": self.user_growth,
            "message_velocity": self.message_velocity,
            "member_count": self.member_count,
            "admin_count": self.admin_count,
            "bot_ratio": self.bot_ratio,
            "sentiment_score": self.sentiment_score,
        }


@dataclass
class DiscordSignals:
    """Discord-specific signals."""

    invite_activity: float = 0.0  # Invites per hour
    member_count: int = 0
    online_count: int = 0
    message_velocity: float = 0.0
    channel_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "invite_activity": self.invite_activity,
            "member_count": self.member_count,
            "online_count": self.online_count,
            "message_velocity": self.message_velocity,
            "channel_count": self.channel_count,
        }


@dataclass
class SocialSignals:
    """Aggregated social signals for a token."""

    twitter: TwitterSignals = field(default_factory=TwitterSignals)
    telegram: TelegramSignals = field(default_factory=TelegramSignals)
    discord: DiscordSignals = field(default_factory=DiscordSignals)

    # Aggregated features for ML
    twitter_mention_velocity: float = 0.0
    twitter_sentiment_score: float = 0.0
    telegram_user_growth: float = 0.0
    telegram_message_velocity: float = 0.0
    discord_invite_activity: float = 0.0
    influencer_mention_flag: bool = False
    social_authenticity_score: float = 0.5

    # Metadata
    timestamp_ms: int = 0
    token_symbol: str = ""

    def to_feature_dict(self) -> dict[str, float]:
        """Convert to feature dictionary for ML."""
        return {
            "twitter_mention_velocity": self.twitter_mention_velocity,
            "twitter_sentiment_score": self.twitter_sentiment_score,
            "telegram_user_growth": self.telegram_user_growth,
            "telegram_message_velocity": self.telegram_message_velocity,
            "discord_invite_activity": self.discord_invite_activity,
            "influencer_mention_flag": 1.0 if self.influencer_mention_flag else 0.0,
            "social_authenticity_score": self.social_authenticity_score,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to full dictionary."""
        return {
            "twitter": self.twitter.to_dict(),
            "telegram": self.telegram.to_dict(),
            "discord": self.discord.to_dict(),
            "features": self.to_feature_dict(),
            "timestamp_ms": self.timestamp_ms,
            "token_symbol": self.token_symbol,
        }


class SentimentAnalyzer:
    """Sentiment analysis using transformers."""

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize sentiment analyzer.

        Args:
            model_name: HuggingFace model for sentiment analysis
        """
        self.model_name = model_name
        self._pipeline = None

        if TRANSFORMERS_AVAILABLE:
            try:
                self._pipeline = pipeline("sentiment-analysis", model=model_name)
                logger.info(f"Loaded sentiment model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentiment model: {e}")

    def analyze(self, text: str) -> float:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score from -1 (negative) to 1 (positive)
        """
        if self._pipeline is None:
            return 0.0

        try:
            result = self._pipeline(text[:512])[0]  # Truncate to max length
            label = result["label"]
            score = result["score"]

            # Convert to -1 to 1 scale
            if label == "POSITIVE":
                return score
            else:
                return -score
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0

    def analyze_batch(self, texts: list[str]) -> list[float]:
        """Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment scores
        """
        if self._pipeline is None:
            return [0.0] * len(texts)

        try:
            results = self._pipeline([t[:512] for t in texts])
            scores = []
            for result in results:
                label = result["label"]
                score = result["score"]
                if label == "POSITIVE":
                    scores.append(score)
                else:
                    scores.append(-score)
            return scores
        except Exception as e:
            logger.warning(f"Batch sentiment analysis failed: {e}")
            return [0.0] * len(texts)


class SocialProvider(ABC):
    """Abstract base class for social data providers."""

    @abstractmethod
    async def get_signals(self, token_symbol: str) -> dict[str, Any]:
        """Get social signals for a token."""
        pass


class TwitterProvider(SocialProvider):
    """Twitter/X data provider."""

    # Known crypto influencers (followers > 100k)
    INFLUENCERS = {
        "colobytesky",
        "solana",
        "paradigm",
        "a16z",
        "punk6529",
        "dydx",
        "messaricrypto",
        "theblockworks",
    }

    def __init__(
        self,
        bearer_token: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        access_token: str | None = None,
        access_secret: str | None = None,
    ):
        """Initialize Twitter provider.

        Args:
            bearer_token: Twitter API bearer token
            api_key: Twitter API key
            api_secret: Twitter API secret
            access_token: Twitter access token
            access_secret: Twitter access secret
        """
        self.client = None
        self._sentiment_analyzer = SentimentAnalyzer()

        if TWEEPY_AVAILABLE and bearer_token:
            try:
                self.client = tweepy.Client(bearer_token=bearer_token)
                logger.info("Twitter client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Twitter client: {e}")

    async def get_signals(self, token_symbol: str) -> TwitterSignals:
        """Get Twitter signals for a token.

        Args:
            token_symbol: Token symbol (e.g., $BONK)

        Returns:
            TwitterSignals with metrics
        """
        signals = TwitterSignals()

        if self.client is None:
            return signals

        try:
            # Search recent tweets
            query = f"${token_symbol} -is:retweet lang:en"
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=100,
                tweet_fields=["created_at", "public_metrics", "author_id"],
                expansions=["author_id"],
                user_fields=["username", "public_metrics", "verified"],
            )

            if not tweets.data:
                return signals

            # Process tweets
            tweet_times = []
            texts = []
            influencer_mentions = []

            users_by_id = {}
            if tweets.includes and "users" in tweets.includes:
                for user in tweets.includes["users"]:
                    users_by_id[user.id] = user

            for tweet in tweets.data:
                tweet_times.append(tweet.created_at.timestamp())
                texts.append(tweet.text)

                # Check for influencer mentions
                user = users_by_id.get(tweet.author_id)
                if user:
                    if (
                        user.username.lower() in self.INFLUENCERS
                        or user.public_metrics.get("followers_count", 0) > 100000
                    ):
                        influencer_mentions.append(user.username)

            # Calculate mention velocity (mentions per hour)
            if len(tweet_times) >= 2:
                time_span = max(tweet_times) - min(tweet_times)
                if time_span > 0:
                    signals.mention_velocity = len(tweet_times) / (time_span / 3600)

            # Analyze sentiment
            if texts:
                sentiments = self._sentiment_analyzer.analyze_batch(texts)
                signals.sentiment_score = np.mean(sentiments)

            signals.influencer_mentions = list(set(influencer_mentions))
            signals.recent_mentions = [
                {"text": t.text, "created_at": str(t.created_at)} for t in tweets.data[:5]
            ]

        except Exception as e:
            logger.warning(f"Failed to get Twitter signals: {e}")

        return signals

    def get_signals_sync(self, token_symbol: str) -> TwitterSignals:
        """Synchronous version of get_signals."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self.get_signals(token_symbol))


class TelegramProvider(SocialProvider):
    """Telegram data provider."""

    def __init__(
        self,
        api_id: str | None = None,
        api_hash: str | None = None,
        session_name: str = "coldpath_telegram",
    ):
        """Initialize Telegram provider.

        Args:
            api_id: Telegram API ID
            api_hash: Telegram API hash
            session_name: Session name for persistence
        """
        self.client = None
        self._sentiment_analyzer = SentimentAnalyzer()

        if TELETHON_AVAILABLE and api_id and api_hash:
            try:
                self.client = TelegramClient(session_name, api_id, api_hash)
                logger.info("Telegram client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Telegram client: {e}")

    async def get_signals(self, group_id: str) -> TelegramSignals:
        """Get Telegram signals for a group.

        Args:
            group_id: Telegram group ID or username

        Returns:
            TelegramSignals with metrics
        """
        signals = TelegramSignals()

        if self.client is None:
            return signals

        try:
            await self.client.connect()

            # Get group info
            entity = await self.client.get_entity(group_id)

            if hasattr(entity, "participants_count"):
                signals.member_count = entity.participants_count

            # Get recent messages
            messages = await self.client.get_messages(entity, limit=100)

            if messages:
                # Calculate message velocity
                timestamps = [m.date.timestamp() for m in messages if m.date]
                if len(timestamps) >= 2:
                    time_span = max(timestamps) - min(timestamps)
                    if time_span > 0:
                        signals.message_velocity = len(timestamps) / (time_span / 3600)

                # Analyze sentiment
                texts = [m.text for m in messages if m.text]
                if texts:
                    sentiments = self._sentiment_analyzer.analyze_batch(texts)
                    signals.sentiment_score = np.mean(sentiments)

        except Exception as e:
            logger.warning(f"Failed to get Telegram signals: {e}")
        finally:
            if self.client:
                await self.client.disconnect()

        return signals


class SentimentTracker:
    """Unified social sentiment tracker.

    Aggregates signals from Twitter, Telegram, and Discord
    into a single SocialSignals object for ML features.
    """

    def __init__(
        self,
        twitter_bearer_token: str | None = None,
        telegram_api_id: str | None = None,
        telegram_api_hash: str | None = None,
    ):
        """Initialize sentiment tracker.

        Args:
            twitter_bearer_token: Twitter API bearer token
            telegram_api_id: Telegram API ID
            telegram_api_hash: Telegram API hash
        """
        self.twitter = TwitterProvider(bearer_token=twitter_bearer_token)
        self.telegram = TelegramProvider(
            api_id=telegram_api_id,
            api_hash=telegram_api_hash,
        )

        self._cache: dict[str, tuple[SocialSignals, float]] = {}
        self._cache_ttl = 300  # 5 minutes

    async def get_signals(
        self,
        token_symbol: str,
        telegram_group: str | None = None,
        discord_server: str | None = None,
    ) -> SocialSignals:
        """Get aggregated social signals for a token.

        Args:
            token_symbol: Token symbol (e.g., BONK)
            telegram_group: Optional Telegram group ID
            discord_server: Optional Discord server ID

        Returns:
            SocialSignals with all metrics
        """
        # Check cache
        cache_key = f"{token_symbol}:{telegram_group}:{discord_server}"
        if cache_key in self._cache:
            signals, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return signals

        signals = SocialSignals(
            timestamp_ms=int(time.time() * 1000),
            token_symbol=token_symbol,
        )

        # Get Twitter signals
        signals.twitter = await self.twitter.get_signals(token_symbol)
        signals.twitter_mention_velocity = signals.twitter.mention_velocity
        signals.twitter_sentiment_score = signals.twitter.sentiment_score
        signals.influencer_mention_flag = len(signals.twitter.influencer_mentions) > 0

        # Get Telegram signals
        if telegram_group:
            signals.telegram = await self.telegram.get_signals(telegram_group)
            signals.telegram_user_growth = signals.telegram.user_growth
            signals.telegram_message_velocity = signals.telegram.message_velocity

        # Calculate authenticity score
        signals.social_authenticity_score = self._calculate_authenticity(signals)

        # Cache results
        self._cache[cache_key] = (signals, time.time())

        return signals

    def get_signals_sync(
        self,
        token_symbol: str,
        telegram_group: str | None = None,
        discord_server: str | None = None,
    ) -> SocialSignals:
        """Synchronous version of get_signals."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.get_signals(token_symbol, telegram_group, discord_server)
        )

    def _calculate_authenticity(self, signals: SocialSignals) -> float:
        """Calculate social authenticity score.

        Factors:
        - Sentiment variance (organic = more varied)
        - Mention/velocity ratio (organic = more balanced)
        - Account diversity
        - Influencer involvement
        """
        score = 0.5  # Start neutral

        # Influencer mentions add authenticity
        if signals.influencer_mention_flag:
            score += 0.2

        # High velocity with low follower count is suspicious
        if signals.twitter.mention_velocity > 50:
            if signals.twitter.follower_count < 1000:
                score -= 0.2
            else:
                score += 0.1

        # Very high Telegram activity with few members is suspicious
        if signals.telegram.member_count > 0:
            activity_ratio = signals.telegram.message_velocity / signals.telegram.member_count
            if activity_ratio > 10:  # > 10 messages per member per hour
                score -= 0.15
            elif activity_ratio > 1:
                score += 0.05

        # Account age matters
        if signals.twitter.account_age_days < 7:
            score -= 0.1
        elif signals.twitter.account_age_days > 180:
            score += 0.1

        return float(np.clip(score, 0.0, 1.0))

    def check_availability(self) -> dict[str, bool]:
        """Check which social providers are available."""
        return {
            "twitter": self.twitter.client is not None,
            "telegram": self.telegram.client is not None,
            "sentiment_analysis": TRANSFORMERS_AVAILABLE,
            "tweepy": TWEEPY_AVAILABLE,
            "telethon": TELETHON_AVAILABLE,
        }


def create_mock_signals(
    token_symbol: str,
    bullish: bool = True,
    high_activity: bool = False,
) -> SocialSignals:
    """Create mock social signals for testing.

    Args:
        token_symbol: Token symbol
        bullish: Whether to generate bullish signals
        high_activity: Whether to simulate high activity

    Returns:
        Mock SocialSignals
    """
    sentiment = 0.6 if bullish else -0.4
    velocity_mult = 5.0 if high_activity else 1.0

    return SocialSignals(
        token_symbol=token_symbol,
        timestamp_ms=int(time.time() * 1000),
        twitter=TwitterSignals(
            mention_velocity=20 * velocity_mult,
            sentiment_score=sentiment,
            follower_count=5000,
            account_age_days=90,
        ),
        telegram=TelegramSignals(
            user_growth=10 * velocity_mult,
            message_velocity=50 * velocity_mult,
            member_count=1000,
        ),
        discord=DiscordSignals(
            invite_activity=5 * velocity_mult,
            member_count=500,
        ),
        twitter_mention_velocity=20 * velocity_mult,
        twitter_sentiment_score=sentiment,
        telegram_user_growth=10 * velocity_mult,
        telegram_message_velocity=50 * velocity_mult,
        discord_invite_activity=5 * velocity_mult,
        influencer_mention_flag=bullish,
        social_authenticity_score=0.7 if bullish else 0.4,
    )
