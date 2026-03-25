"""
Synthetic Training Data Generator using FinGPT.

Generates realistic training samples when real data is unavailable,
using FinGPT for market-driven label generation and sentiment analysis.

Based on FinGPT best practices:
- Market-driven labels (stock price reactions as ground truth)
- Sentiment analysis for social features (indices 43-49)
- RLSP-inspired reward modeling
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ..ai.fingpt_client import FinGPTClient, get_fingpt_client
from ..learning.feature_engineering import FEATURE_COUNT, FeatureIndex

logger = logging.getLogger(__name__)


@dataclass
class SyntheticTokenProfile:
    """Synthetic token profile for training data generation."""

    symbol: str
    price_usd: float
    volume_24h: float
    market_cap: float
    liquidity: float
    holders: int
    top_holder_pct: float
    token_age_hours: float
    price_change_24h: float
    fdv_usd: float
    # Risk indicators
    mint_authority_revoked: bool = True
    freeze_authority_revoked: bool = True
    lp_burned_pct: float = 0.0
    # Social (currently zeros in real data)
    twitter_mentions_24h: int = 0
    telegram_members: int = 0


@dataclass
class SyntheticTrainingOutcome:
    """A synthetic training outcome with features and label."""

    mint: str
    timestamp_ms: int
    features: dict[str, Any]
    features_vector: np.ndarray
    label_binary: int
    label_return: float
    pnl_pct: float
    execution_mode: str
    model_score: float
    reasoning: str


# Realistic parameter ranges for Solana memecoins
TOKEN_PROFILES = {
    "high_quality": {
        "weight": 0.15,
        "price_range": (0.0001, 0.01),
        "volume_range": (100000, 10000000),
        "liquidity_range": (50000, 500000),
        "holders_range": (500, 5000),
        "age_range": (24, 720),  # 1-30 days
        "win_rate": 0.75,
        "avg_return": 15.0,
    },
    "medium_quality": {
        "weight": 0.35,
        "price_range": (0.00001, 0.001),
        "volume_range": (10000, 500000),
        "liquidity_range": (10000, 100000),
        "holders_range": (100, 1000),
        "age_range": (6, 168),  # 6h-7 days
        "win_rate": 0.55,
        "avg_return": 5.0,
    },
    "speculative": {
        "weight": 0.30,
        "price_range": (0.000001, 0.0001),
        "volume_range": (5000, 100000),
        "liquidity_range": (5000, 50000),
        "holders_range": (50, 500),
        "age_range": (1, 72),  # 1h-3 days
        "win_rate": 0.40,
        "avg_return": -5.0,
    },
    "rug_candidates": {
        "weight": 0.20,
        "price_range": (0.0000001, 0.00001),
        "volume_range": (1000, 50000),
        "liquidity_range": (1000, 20000),
        "holders_range": (10, 200),
        "age_range": (0.5, 24),  # 30min-1 day
        "win_rate": 0.10,
        "avg_return": -80.0,
    },
}


class SyntheticDataGenerator:
    """
    Generate synthetic training data using FinGPT for label generation.

    This addresses the cold-start problem when no real training data exists.

    Usage:
        generator = SyntheticDataGenerator()
        outcomes = await generator.generate_batch(500)
    """

    def __init__(
        self,
        fingpt_client: FinGPTClient | None = None,
        use_llm_labels: bool = True,
        seed: int | None = None,
    ):
        self.client = fingpt_client or get_fingpt_client()
        self.use_llm_labels = use_llm_labels
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self._generated_count = 0
        self._llm_label_count = 0

    def _generate_mint(self) -> str:
        """Generate a realistic-looking Solana mint address."""
        chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        return "".join(self.rng.choices(chars, k=44))

    def _generate_symbol(self) -> str:
        """Generate a memecoin-style symbol."""
        prefixes = ["DOGE", "PEPE", "WOJAK", "MOON", "SOL", "BONK", "WIF", "MYRO"]
        suffixes = ["", "2", "X", "INU", "COIN", "AI", "GPT", "GEM"]

        prefix = self.rng.choice(prefixes)
        suffix = self.rng.choice(suffixes)

        if self.rng.random() < 0.3:
            # Meme-style name
            return f"{prefix}{suffix}"
        else:
            # Random 3-5 letter symbol
            length = self.rng.randint(3, 5)
            return "".join(self.rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=length))

    def _sample_profile(self) -> tuple[str, dict[str, Any]]:
        """Sample a token profile type based on weights."""
        profiles = list(TOKEN_PROFILES.items())
        weights = [p["weight"] for _, p in profiles]
        return self.rng.choices(profiles, weights=weights, k=1)[0]

    def _generate_token_profile(self) -> SyntheticTokenProfile:
        """Generate a realistic synthetic token profile."""
        profile_name, profile = self._sample_profile()

        # Sample within profile ranges
        price = self.rng.uniform(*profile["price_range"])
        volume = self.rng.uniform(*profile["volume_range"])
        liquidity = self.rng.uniform(*profile["liquidity_range"])
        holders = self.rng.randint(*profile["holders_range"])
        age = self.rng.uniform(*profile["age_range"])

        # Derived values
        market_cap = price * (liquidity / price) * self.rng.uniform(0.5, 2.0)
        fdv = market_cap * self.rng.uniform(1.0, 5.0)
        price_change = self.rng.gauss(0, 50)  # % change, high volatility

        # Holder concentration
        top_holder_pct = self.rng.uniform(5, 50)
        if profile_name == "rug_candidates":
            top_holder_pct = self.rng.uniform(30, 80)  # High concentration

        # Security flags
        mint_revoked = profile_name in ["high_quality", "medium_quality"]
        freeze_revoked = profile_name in ["high_quality", "medium_quality"]
        lp_burned = self.rng.uniform(0, 100) if profile_name == "high_quality" else 0

        # Social (simulated)
        twitter_mentions = (
            int(self.rng.expovariate(0.01)) if profile_name != "rug_candidates" else 0
        )
        telegram_members = (
            int(self.rng.expovariate(0.0001)) if profile_name != "rug_candidates" else 0
        )

        return SyntheticTokenProfile(
            symbol=self._generate_symbol(),
            price_usd=price,
            volume_24h=volume,
            market_cap=market_cap,
            liquidity=liquidity,
            holders=holders,
            top_holder_pct=top_holder_pct,
            token_age_hours=age,
            price_change_24h=price_change,
            fdv_usd=fdv,
            mint_authority_revoked=mint_revoked,
            freeze_authority_revoked=freeze_revoked,
            lp_burned_pct=lp_burned,
            twitter_mentions_24h=twitter_mentions,
            telegram_members=telegram_members,
        )

    def _profile_to_features(self, profile: SyntheticTokenProfile) -> np.ndarray:
        """Convert token profile to 50-feature vector using FeatureIndex."""
        features = np.zeros(FEATURE_COUNT, dtype=np.float64)

        # === LIQUIDITY & POOL METRICS (0-11) ===
        features[FeatureIndex.POOL_TVL_SOL] = profile.liquidity / 150  # Convert to SOL approx
        features[FeatureIndex.POOL_AGE_SECONDS] = profile.token_age_hours * 3600
        features[FeatureIndex.LP_LOCK_PERCENTAGE] = profile.lp_burned_pct
        features[FeatureIndex.LP_CONCENTRATION] = profile.top_holder_pct / 100
        features[FeatureIndex.LP_REMOVAL_VELOCITY] = self.rng.uniform(0, 0.5)
        features[FeatureIndex.LP_ADDITION_VELOCITY] = self.rng.uniform(0, 0.3)
        features[FeatureIndex.POOL_DEPTH_IMBALANCE] = self.rng.uniform(-0.5, 0.5)
        features[FeatureIndex.SLIPPAGE_1PCT] = self.rng.uniform(0.01, 0.05)
        features[FeatureIndex.SLIPPAGE_5PCT] = self.rng.uniform(0.05, 0.2)
        features[FeatureIndex.UNIQUE_LP_PROVIDER_COUNT] = self.rng.randint(1, 20)
        features[FeatureIndex.DEPLOYER_LP_OWNERSHIP_PCT] = profile.top_holder_pct / 100
        features[FeatureIndex.EMERGENCY_LIQUIDITY_FLAG] = 0.0 if profile.liquidity > 10000 else 1.0

        # === TOKEN SUPPLY & HOLDER (12-22) ===
        features[FeatureIndex.TOTAL_SUPPLY] = profile.fdv_usd / max(profile.price_usd, 1e-12)
        features[FeatureIndex.DEPLOYER_HOLDINGS_PCT] = profile.top_holder_pct / 100
        features[FeatureIndex.TOP_10_HOLDER_CONCENTRATION] = profile.top_holder_pct / 100
        features[FeatureIndex.HOLDER_COUNT_UNIQUE] = profile.holders
        features[FeatureIndex.HOLDER_GROWTH_VELOCITY] = self.rng.uniform(-0.1, 0.3)
        features[FeatureIndex.TRANSFER_CONCENTRATION] = self.rng.uniform(0.1, 0.5)
        features[FeatureIndex.SNIPER_BOT_COUNT_T0] = self.rng.randint(0, 5)
        features[FeatureIndex.BOT_TO_HUMAN_RATIO] = self.rng.uniform(0.1, 0.8)
        features[FeatureIndex.LARGE_HOLDER_CHURN] = self.rng.uniform(0, 0.2)
        features[FeatureIndex.MINT_AUTHORITY_REVOKED] = (
            1.0 if profile.mint_authority_revoked else 0.0
        )
        features[FeatureIndex.TOKEN_FREEZEABLE] = (
            1.0 if not profile.freeze_authority_revoked else 0.0
        )

        # === PRICE & VOLUME (23-32) ===
        features[FeatureIndex.PRICE_MOMENTUM_30S] = profile.price_change_24h / 2880  # 30s in 24h
        features[FeatureIndex.PRICE_MOMENTUM_5M] = profile.price_change_24h / 288  # 5m in 24h
        features[FeatureIndex.VOLATILITY_5M] = self.rng.uniform(0.1, 0.5)
        features[FeatureIndex.VOLUME_ACCELERATION] = profile.volume_24h / 1e6  # Normalized
        features[FeatureIndex.BUY_VOLUME_RATIO] = self.rng.uniform(0.4, 0.7)
        features[FeatureIndex.TRADE_SIZE_VARIANCE] = self.rng.uniform(0.5, 2.0)
        features[FeatureIndex.VWAP_DEVIATION] = self.rng.uniform(-0.1, 0.1)
        features[FeatureIndex.PRICE_IMPACT_1PCT] = self.rng.uniform(0.01, 0.1)
        features[FeatureIndex.CONSECUTIVE_BUYS] = self.rng.randint(0, 10)
        features[FeatureIndex.MAX_BUY_IN_WINDOW] = self.rng.uniform(0.01, 0.1)

        # === ON-CHAIN RISK (33-42) ===
        features[FeatureIndex.CONTRACT_IS_MINTABLE] = 0.0 if profile.mint_authority_revoked else 1.0
        features[FeatureIndex.CONTRACT_TRANSFER_FEE] = 0.0  # Assume no transfer fee
        features[FeatureIndex.HIDDEN_FEE_DETECTED] = 0.0  # Assume no hidden fee
        features[FeatureIndex.CIRCULAR_TRADING_SCORE] = self.rng.uniform(0, 0.5)
        features[FeatureIndex.BENFORD_LAW_PVALUE] = self.rng.uniform(0.01, 0.5)
        features[FeatureIndex.ADDRESS_CLUSTERING_RISK] = self.rng.uniform(0, 0.5)
        features[FeatureIndex.PROXY_CONTRACT_FLAG] = 0.0  # Assume no proxy
        features[FeatureIndex.UNVERIFIED_CODE_FLAG] = 0.0 if profile.mint_authority_revoked else 1.0
        features[FeatureIndex.EXTERNAL_TRANSFER_FLAG] = self.rng.uniform(0, 0.3)
        features[FeatureIndex.RUG_PULL_ML_SCORE] = self.rng.uniform(0, 1)

        # === SOCIAL & SENTIMENT (43-49) - Now populated! ===
        features[FeatureIndex.TWITTER_MENTION_VELOCITY] = min(
            profile.twitter_mentions_24h / 100, 100
        )
        features[FeatureIndex.TWITTER_SENTIMENT_SCORE] = self.rng.uniform(-1, 1)
        features[FeatureIndex.TELEGRAM_USER_GROWTH] = min(profile.telegram_members / 1000, 100)
        features[FeatureIndex.TELEGRAM_MESSAGE_VELOCITY] = self.rng.uniform(0, 10)
        features[FeatureIndex.DISCORD_INVITE_ACTIVITY] = self.rng.uniform(0, 5)
        features[FeatureIndex.INFLUENCER_MENTION_FLAG] = 1.0 if self.rng.random() > 0.8 else 0.0
        features[FeatureIndex.SOCIAL_AUTHENTICITY_SCORE] = self.rng.uniform(0.3, 1.0)

        return features

    async def _get_llm_label(
        self,
        profile: SyntheticTokenProfile,
        features: np.ndarray,
    ) -> tuple[int, float, float, str]:
        """
        Get label from FinGPT analysis.

        Returns:
            (label_binary, label_return, model_score, reasoning)
        """
        prompt = """Analyze this Solana token and predict if it would be
profitable for a SHORT-TERM trade (1-4 hour hold).

Token Profile:
- Symbol: {profile.symbol}
- Price: ${profile.price_usd:.8f}
- 24h Volume: ${profile.volume_24h:,.0f}
- Market Cap: ${profile.market_cap:,.0f}
- Liquidity: ${profile.liquidity:,.0f}
- Holders: {profile.holders}
- Top Holder %: {profile.top_holder_pct:.1f}%
- Token Age: {profile.token_age_hours:.1f} hours
- 24h Price Change: {profile.price_change_24h:.1f}%
- Mint Authority Revoked: {profile.mint_authority_revoked}
- Freeze Authority Revoked: {profile.freeze_authority_revoked}

Predict the trade outcome. Be realistic - most memecoins lose money.

Respond in JSON format:
{{
  "profitable": true/false,
  "expected_return_pct": -50.0 to 100.0,
  "confidence": 0.0 to 1.0,
  "key_risk": "brief risk description",
  "key_catalyst": "brief potential catalyst"
}}"""

        try:
            response = await self.client.generate(prompt, json_mode=True, max_tokens=300)

            if response.parsed_json:
                data = response.parsed_json
                profitable = data.get("profitable", False)
                expected_return = float(data.get("expected_return_pct", 0.0))
                confidence = float(data.get("confidence", 0.5))

                label_binary = 1 if profitable else 0
                model_score = confidence if profitable else 1 - confidence

                risk = data.get("key_risk", "unknown")
                catalyst = data.get("key_catalyst", "none")
                reasoning = f"Risk: {risk}; Catalyst: {catalyst}"

                self._llm_label_count += 1
                return label_binary, expected_return, model_score, reasoning

        except Exception as e:
            logger.debug(f"LLM labeling failed: {e}")

        # Fallback to rule-based labeling
        return self._rule_based_label(profile)

    def _get_profile_config(self, profile: SyntheticTokenProfile) -> tuple[str, dict[str, Any]]:
        """Determine which profile config matches the given profile.

        Uses heuristics based on liquidity and holder count to classify.
        """
        # Classify based on liquidity
        if profile.liquidity >= 50000 and profile.holders >= 500:
            return "high_quality", TOKEN_PROFILES["high_quality"]
        elif profile.liquidity >= 10000 and profile.holders >= 100:
            return "medium_quality", TOKEN_PROFILES["medium_quality"]
        elif profile.liquidity >= 5000:
            return "speculative", TOKEN_PROFILES["speculative"]
        else:
            return "rug_candidates", TOKEN_PROFILES["rug_candidates"]

    def _rule_based_label(
        self,
        profile: SyntheticTokenProfile,
    ) -> tuple[int, float, float, str]:
        """
        Rule-based label generation as fallback.

        Uses market-driven heuristics based on FinGPT paper:
        - Labels derived from expected price reaction
        """
        profile_name, profile_config = self._get_profile_config(profile)

        # Base probability from profile
        win_rate = profile_config["win_rate"]
        avg_return = profile_config["avg_return"]

        # Adjustments based on specific factors
        adjustment = 0.0

        # Liquidity boost
        if profile.liquidity > 100000:
            adjustment += 0.05
        elif profile.liquidity < 10000:
            adjustment -= 0.10

        # Holder concentration penalty
        if profile.top_holder_pct > 50:
            adjustment -= 0.15
        elif profile.top_holder_pct > 30:
            adjustment -= 0.05

        # Security bonus
        if profile.mint_authority_revoked and profile.freeze_authority_revoked:
            adjustment += 0.05

        # Age factor
        if profile.token_age_hours < 1:
            adjustment -= 0.10
        elif profile.token_age_hours > 72:
            adjustment += 0.03

        # Calculate final probability
        final_win_prob = max(0, min(1, win_rate + adjustment + self.rng.gauss(0, 0.05)))

        # Determine outcome
        is_win = self.rng.random() < final_win_prob

        # Generate return based on outcome
        if is_win:
            # Winning trade: positive return
            base_return = abs(avg_return) * self.rng.uniform(0.5, 2.0)
            pnl_pct = base_return * self.rng.uniform(0.3, 1.0)  # Partial capture
        else:
            # Losing trade: negative return
            base_loss = abs(avg_return) * 0.5 if avg_return < 0 else 20
            pnl_pct = -base_loss * self.rng.uniform(0.5, 1.5)  # Stop loss helps

        label_binary = 1 if pnl_pct > 0 else 0
        model_score = final_win_prob

        reasoning = f"Rule-based: profile={profile_name}, liquidity_adj={adjustment:.2f}"

        return label_binary, pnl_pct, model_score, reasoning

    async def generate_outcome(self) -> SyntheticTrainingOutcome:
        """Generate a single synthetic training outcome."""
        profile = self._generate_token_profile()
        features = self._profile_to_features(profile)

        # Get label (LLM or rule-based)
        if self.use_llm_labels:
            label_binary, pnl_pct, model_score, reasoning = await self._get_llm_label(
                profile, features
            )
        else:
            label_binary, pnl_pct, model_score, reasoning = self._rule_based_label(profile)

        # Generate metadata
        mint = self._generate_mint()
        timestamp = int(
            (datetime.now() - timedelta(hours=self.rng.uniform(0, 168))).timestamp() * 1000
        )

        # Features dict for storage
        features_dict = {
            "symbol": profile.symbol,
            "price_usd": profile.price_usd,
            "volume_24h": profile.volume_24h,
            "liquidity": profile.liquidity,
            "holders": profile.holders,
            "top_holder_pct": profile.top_holder_pct,
            "token_age_hours": profile.token_age_hours,
            "social_mentions": profile.twitter_mentions_24h,
            "community_size": profile.telegram_members,
        }

        self._generated_count += 1

        return SyntheticTrainingOutcome(
            mint=mint,
            timestamp_ms=timestamp,
            features=features_dict,
            features_vector=features,
            label_binary=label_binary,
            label_return=pnl_pct / 100,  # As decimal
            pnl_pct=pnl_pct,
            execution_mode=self.rng.choice(["paper", "paper", "paper", "live"]),
            model_score=model_score,
            reasoning=reasoning,
        )

    async def generate_batch(
        self,
        count: int,
        batch_size: int = 10,
    ) -> list[SyntheticTrainingOutcome]:
        """
        Generate a batch of synthetic training outcomes.

        Args:
            count: Number of outcomes to generate
            batch_size: Concurrent generation batch size (for LLM rate limiting)

        Returns:
            List of SyntheticTrainingOutcome
        """
        outcomes = []

        for i in range(0, count, batch_size):
            batch_count = min(batch_size, count - i)

            # Generate batch concurrently
            batch = await asyncio.gather(*[self.generate_outcome() for _ in range(batch_count)])

            outcomes.extend(batch)

            if (i + batch_count) % 100 == 0:
                logger.info(f"Generated {len(outcomes)}/{count} synthetic outcomes")

            # Small delay between batches for LLM rate limiting
            if self.use_llm_labels and i + batch_count < count:
                await asyncio.sleep(0.1)

        logger.info(
            f"Generated {len(outcomes)} synthetic outcomes (LLM-labeled: {self._llm_label_count})"
        )

        return outcomes

    def outcome_to_dict(self, outcome: SyntheticTrainingOutcome) -> dict[str, Any]:
        """Convert outcome to dict format compatible with ProfitabilityLearner."""
        return {
            "mint": outcome.mint,
            "timestamp_ms": outcome.timestamp_ms,
            "decision_timestamp_ms": outcome.timestamp_ms,
            "outcome_type": "synthetic",
            "pnl_pct": outcome.pnl_pct,
            "pnl_sol": outcome.pnl_pct * 0.01,  # Approximate
            "label_binary": outcome.label_binary,
            "was_profitable_counterfactual": outcome.label_binary,
            "execution_mode": outcome.execution_mode,
            "model_score": outcome.model_score,
            **outcome.features,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get generation statistics."""
        return {
            "total_generated": self._generated_count,
            "llm_labeled": self._llm_label_count,
            "rule_labeled": self._generated_count - self._llm_label_count,
        }


async def generate_synthetic_training_data(
    count: int = 500,
    use_llm_labels: bool = True,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Convenience function to generate synthetic training data.

    Args:
        count: Number of samples to generate
        use_llm_labels: Use FinGPT for labeling (slower but more realistic)
        seed: Random seed for reproducibility

    Returns:
        List of outcome dicts compatible with training pipeline
    """
    generator = SyntheticDataGenerator(use_llm_labels=use_llm_labels, seed=seed)
    outcomes = await generator.generate_batch(count)
    return [generator.outcome_to_dict(o) for o in outcomes]
