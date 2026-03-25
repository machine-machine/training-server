"""
Social data providers for feature extraction.

Provides social sentiment and activity metrics for trading decisions.
Currently uses heuristic proxy values derived from on-chain and trading data,
with architecture ready for real API integration.

Social Features (indices 43-49):
- twitter_mention_velocity (43): Rate of social mentions
- twitter_sentiment_score (44): Overall sentiment (-1 to 1)
- telegram_user_growth (45): Growth rate of community
- telegram_message_velocity (46): Activity level in community
- discord_invite_activity (47): Discord engagement (currently proxy)
- influencer_mention_flag (48): Large account attention detected
- social_authenticity_score (49): How organic vs bot-driven (0-1)

Future Integration:
To connect real social APIs, implement SocialDataProvider class:
- Twitter API v2 for mentions and sentiment
- Telegram Bot API for channel metrics
- Discord Bot API for server activity
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SocialInput:
    """Input data for social metric calculation."""

    # Trading metrics
    volume_24h_usd: float = 0.0
    volume_to_liquidity_ratio: float = 0.0
    price_change_1h_pct: float = 0.0
    price_change_24h_pct: float = 0.0
    volatility_1h: float = 0.0

    # Holder metrics
    holder_count: int = 0
    holder_growth_rate_pct: float = 0.0
    top_holder_concentration_pct: float = 0.0

    # Activity metrics
    trade_arrival_rate: float = 0.0
    buy_volume_ratio: float = 0.5
    avg_trade_size_usd: float = 0.0

    # Token metadata
    token_age_hours: float = 0.0
    liquidity_usd: float = 0.0
    fdv_usd: float = 0.0

    # Risk indicators
    rug_risk_score: float = 0.0
    bot_to_human_ratio: float = 0.0


@dataclass
class SocialMetrics:
    """Social metrics for a token."""

    twitter_mention_velocity: float = 0.0
    twitter_sentiment_score: float = 0.0
    telegram_user_growth: float = 0.0
    telegram_message_velocity: float = 0.0
    discord_invite_activity: float = 0.0
    influencer_mention_flag: bool = False
    social_authenticity_score: float = 0.5
    source: str = "heuristic"
    collected_at_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "twitter_mention_velocity": self.twitter_mention_velocity,
            "twitter_sentiment_score": self.twitter_sentiment_score,
            "telegram_user_growth": self.telegram_user_growth,
            "telegram_message_velocity": self.telegram_message_velocity,
            "discord_invite_activity": self.discord_invite_activity,
            "influencer_mention_flag": self.influencer_mention_flag,
            "social_authenticity_score": self.social_authenticity_score,
            "source": self.source,
        }

    def to_feature_array(self) -> list:
        """Return social features as array for indices 43-49."""
        return [
            self.twitter_mention_velocity,
            self.twitter_sentiment_score,
            self.telegram_user_growth,
            self.telegram_message_velocity,
            self.discord_invite_activity,
            float(self.influencer_mention_flag),
            self.social_authenticity_score,
        ]


class HeuristicSocialProvider:
    """
    Heuristic social provider that derives metrics from trading data.

    This is the default provider when no real social APIs are connected.
    The formulas are designed to:
    1. Minimize multicollinearity with trading features
    2. Use non-linear transformations for independence
    3. Provide meaningful relative rankings
    4. Scale appropriately across different token sizes
    """

    def __init__(self, conservative: bool = False):
        """
        Initialize the heuristic provider.

        Args:
            conservative: If True, use lower variance estimates
        """
        self.conservative = conservative

    def _calc_twitter_velocity(self, input_data: SocialInput) -> float:
        """Calculate Twitter mention velocity using volume + holder engagement."""
        # Base velocity from volume (log scale for better distribution)
        volume_factor = 0.0
        if input_data.volume_24h_usd > 0:
            volume_factor = max(0.0, math.log(input_data.volume_24h_usd / 1000.0))

        # Holder engagement: more holders = more potential tweets
        holder_factor = 0.5
        if input_data.holder_count > 0:
            holder_factor = min(3.0, max(0.0, math.log(input_data.holder_count / 100.0)))

        # Price momentum amplifies social activity
        momentum_factor = 1.0 + max(-0.5, min(1.0, input_data.price_change_1h_pct / 50.0))

        # Combine factors and normalize to [0, 1]
        velocity = volume_factor * holder_factor * momentum_factor / 10.0
        return max(0.0, min(1.0, velocity))

    def _calc_twitter_sentiment(self, input_data: SocialInput) -> float:
        """Calculate Twitter sentiment score (-1 to 1)."""
        # Price-based sentiment (primary factor)
        price_sentiment = max(-1.0, min(1.0, input_data.price_change_1h_pct / 20.0)) * 0.4

        # Buy pressure sentiment
        buy_sentiment = (input_data.buy_volume_ratio - 0.5) * 2.0 * 0.3

        # Holder growth sentiment
        growth_sentiment = max(-1.0, min(1.0, input_data.holder_growth_rate_pct / 20.0)) * 0.2

        # Volatility penalty
        volatility_penalty = 0.0
        if input_data.volatility_1h > 0.1:
            volatility_penalty = -0.1 * min(2.0, input_data.volatility_1h / 0.1)

        # Rug risk penalty
        risk_penalty = -input_data.rug_risk_score * 0.3

        return max(
            -1.0,
            min(
                1.0,
                price_sentiment
                + buy_sentiment
                + growth_sentiment
                + volatility_penalty
                + risk_penalty,
            ),
        )

    def _calc_telegram_growth(self, input_data: SocialInput) -> float:
        """Calculate Telegram user growth rate."""
        # Direct holder growth
        holder_growth = input_data.holder_growth_rate_pct

        # New buyer activity
        new_buyer_activity = 0.0
        if input_data.trade_arrival_rate > 0:
            new_buyer_activity = min(1.0, input_data.trade_arrival_rate / 10.0) * 10.0

        # Age factor
        if input_data.token_age_hours < 24:
            age_factor = 1.5  # Fresh token bonus
        elif input_data.token_age_hours < 168:
            age_factor = 1.0  # Normal growth
        else:
            age_factor = 0.7  # Mature token, slower growth

        growth = (holder_growth * 0.7 + new_buyer_activity * 0.3) * age_factor
        return max(-50.0, min(100.0, growth))

    def _calc_telegram_velocity(self, input_data: SocialInput) -> float:
        """Calculate Telegram message velocity."""
        # Trade arrival rate directly correlates with chat activity
        trade_activity = input_data.trade_arrival_rate

        # Price action creates discussion
        price_discussion = abs(input_data.price_change_1h_pct) / 10.0

        # Volume spike creates excitement
        volume_excitement = 0.0
        if input_data.volume_to_liquidity_ratio > 0.5:
            volume_excitement = (input_data.volume_to_liquidity_ratio - 0.5) * 5.0

        # Holder count factor
        holder_factor = 0.1
        if input_data.holder_count > 0:
            holder_factor = min(1.0, math.log(input_data.holder_count) / 10.0)

        velocity = (trade_activity * 0.5 + price_discussion * 0.3 + volume_excitement * 0.2) * (
            0.5 + holder_factor * 0.5
        )

        return max(0.0, min(1.0, velocity))

    def _calc_discord_activity(self, input_data: SocialInput) -> float:
        """Calculate Discord invite activity (proxy)."""
        # Holder retention
        holder_stability = 1.0 - min(1.0, input_data.top_holder_concentration_pct / 100.0)

        # Community engagement
        if 0.4 < input_data.buy_volume_ratio < 0.7:
            engagement = 0.8  # Healthy mix
        elif input_data.buy_volume_ratio > 0.7:
            engagement = 0.6  # Heavy buying = FOMO
        else:
            engagement = 0.4  # Low buying = concern

        return max(0.0, min(1.0, holder_stability * engagement * 0.3))

    def _detect_influencer_mention(self, input_data: SocialInput) -> bool:
        """Detect if an influencer has mentioned this token."""
        # Large price spike
        price_spike = input_data.price_change_1h_pct > 30.0

        # Volume explosion
        volume_spike = input_data.volume_to_liquidity_ratio > 1.0

        # Holder explosion
        holder_spike = input_data.holder_growth_rate_pct > 20.0

        # Whale activity
        whale_activity = input_data.avg_trade_size_usd > 1000.0

        # Require at least 2 signals
        signals = [price_spike, volume_spike, holder_spike, whale_activity]
        return sum(signals) >= 2

    def _calc_authenticity(self, input_data: SocialInput) -> float:
        """Calculate social authenticity score (0 = bots, 1 = organic)."""
        # Bot-to-human ratio penalty
        bot_penalty = 0.8
        if input_data.bot_to_human_ratio > 0:
            bot_penalty = 1.0 - min(1.0, input_data.bot_to_human_ratio / 5.0)

        # Holder distribution score
        distribution_score = 1.0 - min(1.0, input_data.top_holder_concentration_pct / 100.0)

        # Trade pattern naturalness
        if 0.3 < input_data.buy_volume_ratio < 0.8:
            trade_naturalness = 1.0
        elif input_data.buy_volume_ratio > 0.9 or input_data.buy_volume_ratio < 0.1:
            trade_naturalness = 0.3  # Suspicious
        else:
            trade_naturalness = 0.7

        # Age confidence
        if input_data.token_age_hours < 1:
            age_confidence = 0.5
        elif input_data.token_age_hours > 24:
            age_confidence = 1.0
        else:
            age_confidence = 0.7 + 0.3 * (input_data.token_age_hours / 24.0)

        # Risk penalty
        risk_penalty = 1.0 - input_data.rug_risk_score * 0.5

        authenticity = (
            bot_penalty * 0.3
            + distribution_score * 0.2
            + trade_naturalness * 0.2
            + age_confidence * 0.15
            + risk_penalty * 0.15
        )

        return max(0.0, min(1.0, authenticity))

    def fetch_metrics(self, input_data: SocialInput) -> SocialMetrics:
        """Calculate all social metrics from input data."""
        import time

        scale = 0.8 if self.conservative else 1.0

        return SocialMetrics(
            twitter_mention_velocity=self._calc_twitter_velocity(input_data) * scale,
            twitter_sentiment_score=self._calc_twitter_sentiment(input_data) * scale,
            telegram_user_growth=self._calc_telegram_growth(input_data) * scale,
            telegram_message_velocity=self._calc_telegram_velocity(input_data) * scale,
            discord_invite_activity=self._calc_discord_activity(input_data) * scale,
            influencer_mention_flag=self._detect_influencer_mention(input_data),
            social_authenticity_score=self._calc_authenticity(input_data),
            source="heuristic",
            collected_at_ms=int(time.time() * 1000),
        )


# Default provider instance
_default_provider = HeuristicSocialProvider()


def get_social_metrics(input_data: SocialInput) -> SocialMetrics:
    """Get social metrics using the default provider."""
    return _default_provider.fetch_metrics(input_data)


def extract_social_features(data: dict[str, Any]) -> list:
    """
    Extract social features from token data.

    Args:
        data: Dictionary with token metrics

    Returns:
        List of 7 social features (indices 43-49)
    """
    input_data = SocialInput(
        volume_24h_usd=data.get("volume_24h_usd", data.get("volume_24h", 0)),
        volume_to_liquidity_ratio=data.get("volume_to_liquidity_ratio", 0.5),
        price_change_1h_pct=data.get("price_change_1h", data.get("price_change_1h_pct", 0)),
        price_change_24h_pct=data.get("price_change_24h", data.get("price_change_24h_pct", 0)),
        volatility_1h=data.get("volatility_1h", data.get("volatility", 0.05)),
        holder_count=int(data.get("holder_count", data.get("holder_count_unique", 0))),
        holder_growth_rate_pct=(
            data.get("holder_growth_rate", data.get("holder_growth_rate_pct", 0)) * 100
        ),
        top_holder_concentration_pct=(
            data.get("top_holder_pct", data.get("top_holder_concentration", 0))
        ),
        trade_arrival_rate=data.get("trade_arrival_rate", 0),
        buy_volume_ratio=data.get("buy_volume_ratio", data.get("buy_concentration", 0.5)),
        avg_trade_size_usd=data.get("avg_trade_size_usd", 0),
        token_age_hours=data.get("token_age_hours", data.get("age_hours", 0)),
        liquidity_usd=data.get("liquidity_usd", data.get("liquidity_sol", 0) * 150),
        fdv_usd=data.get("fdv_usd", 0),
        rug_risk_score=data.get("rug_risk_score", data.get("rug_pull_ml_score", 0)),
        bot_to_human_ratio=data.get("bot_to_human_ratio", 0.5),
    )

    metrics = get_social_metrics(input_data)
    return metrics.to_feature_array()
