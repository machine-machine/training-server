"""
Feature Engineering - Extract and transform features for ML training.

Implements 50-feature unified vector across 5 categories:
- Liquidity & Pool Metrics (12 features)
- Token Supply & Holder (11 features)
- Price & Volume (10 features)
- On-Chain Risk (10 features)
- Social & Sentiment (7 features)

Matches the Hot Path scoring feature vector for production inference.
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureIndex(IntEnum):
    """Feature indices for the 50-feature unified vector."""

    # === LIQUIDITY & POOL METRICS (0-11) ===
    POOL_TVL_SOL = 0
    POOL_AGE_SECONDS = 1
    LP_LOCK_PERCENTAGE = 2
    LP_CONCENTRATION = 3
    LP_REMOVAL_VELOCITY = 4
    LP_ADDITION_VELOCITY = 5
    POOL_DEPTH_IMBALANCE = 6
    SLIPPAGE_1PCT = 7
    SLIPPAGE_5PCT = 8
    UNIQUE_LP_PROVIDER_COUNT = 9
    DEPLOYER_LP_OWNERSHIP_PCT = 10
    EMERGENCY_LIQUIDITY_FLAG = 11

    # === TOKEN SUPPLY & HOLDER (12-22) ===
    TOTAL_SUPPLY = 12
    DEPLOYER_HOLDINGS_PCT = 13
    TOP_10_HOLDER_CONCENTRATION = 14
    HOLDER_COUNT_UNIQUE = 15
    HOLDER_GROWTH_VELOCITY = 16
    TRANSFER_CONCENTRATION = 17
    SNIPER_BOT_COUNT_T0 = 18
    BOT_TO_HUMAN_RATIO = 19
    LARGE_HOLDER_CHURN = 20
    MINT_AUTHORITY_REVOKED = 21
    TOKEN_FREEZEABLE = 22

    # === PRICE & VOLUME (23-32) ===
    PRICE_MOMENTUM_30S = 23
    PRICE_MOMENTUM_5M = 24
    VOLATILITY_5M = 25
    VOLUME_ACCELERATION = 26
    BUY_VOLUME_RATIO = 27
    TRADE_SIZE_VARIANCE = 28
    VWAP_DEVIATION = 29
    PRICE_IMPACT_1PCT = 30
    CONSECUTIVE_BUYS = 31
    MAX_BUY_IN_WINDOW = 32

    # === ON-CHAIN RISK (33-42) ===
    CONTRACT_IS_MINTABLE = 33
    CONTRACT_TRANSFER_FEE = 34
    HIDDEN_FEE_DETECTED = 35
    CIRCULAR_TRADING_SCORE = 36
    BENFORD_LAW_PVALUE = 37
    ADDRESS_CLUSTERING_RISK = 38
    PROXY_CONTRACT_FLAG = 39
    UNVERIFIED_CODE_FLAG = 40
    EXTERNAL_TRANSFER_FLAG = 41
    RUG_PULL_ML_SCORE = 42

    # === SOCIAL & SENTIMENT (43-49) ===
    TWITTER_MENTION_VELOCITY = 43
    TWITTER_SENTIMENT_SCORE = 44
    TELEGRAM_USER_GROWTH = 45
    TELEGRAM_MESSAGE_VELOCITY = 46
    DISCORD_INVITE_ACTIVITY = 47
    INFLUENCER_MENTION_FLAG = 48
    SOCIAL_AUTHENTICITY_SCORE = 49


# Total number of features
FEATURE_COUNT = 50


@dataclass
class FeatureSet:
    """Container for extracted 50 features."""

    # Feature names (must match Hot Path order)
    FEATURE_NAMES: list[str] = field(
        default_factory=lambda: [
            # Liquidity & Pool Metrics (12)
            "pool_tvl_sol",
            "pool_age_seconds",
            "lp_lock_percentage",
            "lp_concentration",
            "lp_removal_velocity",
            "lp_addition_velocity",
            "pool_depth_imbalance",
            "slippage_1pct",
            "slippage_5pct",
            "unique_lp_provider_count",
            "deployer_lp_ownership_pct",
            "emergency_liquidity_flag",
            # Token Supply & Holder (11)
            "total_supply",
            "deployer_holdings_pct",
            "top_10_holder_concentration",
            "holder_count_unique",
            "holder_growth_velocity",
            "transfer_concentration",
            "sniper_bot_count_t0",
            "bot_to_human_ratio",
            "large_holder_churn",
            "mint_authority_revoked",
            "token_freezeable",
            # Price & Volume (10)
            "price_momentum_30s",
            "price_momentum_5m",
            "volatility_5m",
            "volume_acceleration",
            "buy_volume_ratio",
            "trade_size_variance",
            "vwap_deviation",
            "price_impact_1pct",
            "consecutive_buys",
            "max_buy_in_window",
            # On-Chain Risk (10)
            "contract_is_mintable",
            "contract_transfer_fee",
            "hidden_fee_detected",
            "circular_trading_score",
            "benford_law_pvalue",
            "address_clustering_risk",
            "proxy_contract_flag",
            "unverified_code_flag",
            "external_transfer_flag",
            "rug_pull_ml_score",
            # Social & Sentiment (7)
            "twitter_mention_velocity",
            "twitter_sentiment_score",
            "telegram_user_growth",
            "telegram_message_velocity",
            "discord_invite_activity",
            "influencer_mention_flag",
            "social_authenticity_score",
        ]
    )

    # Legacy 15-feature names for backward compatibility
    LEGACY_FEATURE_NAMES: list[str] = field(
        default_factory=lambda: [
            "liquidity_usd",
            "volume_24h_usd",
            "vol_liq_ratio",
            "holder_count",
            "top_holder_pct",
            "holder_growth",
            "price_change_1h",
            "price_change_24h",
            "volatility_1h",
            "age_seconds",
            "fdv_usd",
            "lp_burn_pct",
            "mint_authority",
            "freeze_authority",
            "risk_score",
        ]
    )

    # Additional training-only features
    EXTRA_FEATURES: list[str] = field(
        default_factory=lambda: [
            "source_priority",
            "time_of_day",
        ]
    )

    features: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_COUNT))
    extra_features: np.ndarray = field(default_factory=lambda: np.zeros(2))
    feature_dict: dict[str, float] = field(default_factory=dict)
    version: int = 3  # v3 = 50 features

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        result = {}
        for i, name in enumerate(self.FEATURE_NAMES):
            result[name] = float(self.features[i])
        for i, name in enumerate(self.EXTRA_FEATURES):
            result[name] = float(self.extra_features[i])
        return result

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "FeatureSet":
        """Create from dictionary."""
        fs = cls()
        fs.feature_dict = data

        for i, name in enumerate(fs.FEATURE_NAMES):
            fs.features[i] = data.get(name, 0.0)

        for i, name in enumerate(fs.EXTRA_FEATURES):
            fs.extra_features[i] = data.get(name, 0.0)

        return fs

    def to_legacy_array(self) -> np.ndarray:
        """Convert to legacy 15-feature array for backward compatibility.

        Maps the new 50 features to the closest legacy equivalents.
        """
        legacy = np.zeros(15)

        # Map pool_tvl_sol -> liquidity_usd (approximate)
        legacy[0] = self.features[FeatureIndex.POOL_TVL_SOL]
        # volume approximation from volume_acceleration
        legacy[1] = self.features[FeatureIndex.VOLUME_ACCELERATION]
        # vol/liq ratio from buy_volume_ratio
        legacy[2] = self.features[FeatureIndex.BUY_VOLUME_RATIO]
        # holder count
        legacy[3] = self.features[FeatureIndex.HOLDER_COUNT_UNIQUE]
        # top holder %
        legacy[4] = self.features[FeatureIndex.TOP_10_HOLDER_CONCENTRATION]
        # holder growth
        legacy[5] = self.features[FeatureIndex.HOLDER_GROWTH_VELOCITY]
        # price change (from momentum)
        legacy[6] = self.features[FeatureIndex.PRICE_MOMENTUM_30S]
        legacy[7] = self.features[FeatureIndex.PRICE_MOMENTUM_5M]
        # volatility
        legacy[8] = self.features[FeatureIndex.VOLATILITY_5M]
        # age
        legacy[9] = self.features[FeatureIndex.POOL_AGE_SECONDS]
        # FDV (from total_supply)
        legacy[10] = self.features[FeatureIndex.TOTAL_SUPPLY]
        # LP burn (from lock percentage)
        legacy[11] = self.features[FeatureIndex.LP_LOCK_PERCENTAGE]
        # mint authority
        legacy[12] = self.features[FeatureIndex.MINT_AUTHORITY_REVOKED]
        # freeze authority
        legacy[13] = self.features[FeatureIndex.TOKEN_FREEZEABLE]
        # risk score
        legacy[14] = self.features[FeatureIndex.RUG_PULL_ML_SCORE]

        return legacy

    def get_rug_detection_features(self) -> np.ndarray:
        """Get the 33 features used for rug detection (excluding social).

        Returns features for categories: Liquidity (12) + Holder (11) + On-Chain Risk (10).
        """
        indices = list(range(12)) + list(range(12, 23)) + list(range(33, 43))
        return self.features[indices]

    def get_category_features(self, category: str) -> np.ndarray:
        """Get features for a specific category.

        Args:
            category: One of 'liquidity', 'holder', 'price', 'risk', 'social'

        Returns:
            Feature array for that category
        """
        ranges = {
            "liquidity": (0, 12),
            "holder": (12, 23),
            "price": (23, 33),
            "risk": (33, 43),
            "social": (43, 50),
        }
        if category not in ranges:
            raise ValueError(f"Unknown category: {category}")
        start, end = ranges[category]
        return self.features[start:end]


@dataclass
class NormalizationParams:
    """Parameters for feature normalization across all 50 features."""

    # Liquidity & Pool params
    tvl_log_mean: float = 10.0
    tvl_log_std: float = 2.0
    age_log_mean: float = 10.0
    age_log_std: float = 2.0
    lp_count_log_mean: float = 2.0
    lp_count_log_std: float = 1.0

    # Token Supply & Holder params
    supply_log_mean: float = 20.0
    supply_log_std: float = 3.0
    holder_log_mean: float = 5.0
    holder_log_std: float = 1.5

    # Price & Volume params
    volume_log_mean: float = 9.0
    volume_log_std: float = 2.5

    # Legacy params for backward compatibility
    liquidity_log_mean: float = 10.0
    liquidity_log_std: float = 2.0
    fdv_log_mean: float = 13.0
    fdv_log_std: float = 2.0

    def to_dict(self) -> dict[str, float]:
        """Export to dictionary."""
        return {
            "tvl_log_mean": self.tvl_log_mean,
            "tvl_log_std": self.tvl_log_std,
            "age_log_mean": self.age_log_mean,
            "age_log_std": self.age_log_std,
            "lp_count_log_mean": self.lp_count_log_mean,
            "lp_count_log_std": self.lp_count_log_std,
            "supply_log_mean": self.supply_log_mean,
            "supply_log_std": self.supply_log_std,
            "holder_log_mean": self.holder_log_mean,
            "holder_log_std": self.holder_log_std,
            "volume_log_mean": self.volume_log_mean,
            "volume_log_std": self.volume_log_std,
            "liquidity_log_mean": self.liquidity_log_mean,
            "liquidity_log_std": self.liquidity_log_std,
            "fdv_log_mean": self.fdv_log_mean,
            "fdv_log_std": self.fdv_log_std,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "NormalizationParams":
        """Create from dictionary."""
        return cls(
            tvl_log_mean=data.get("tvl_log_mean", 10.0),
            tvl_log_std=data.get("tvl_log_std", 2.0),
            age_log_mean=data.get("age_log_mean", 10.0),
            age_log_std=data.get("age_log_std", 2.0),
            lp_count_log_mean=data.get("lp_count_log_mean", 2.0),
            lp_count_log_std=data.get("lp_count_log_std", 1.0),
            supply_log_mean=data.get("supply_log_mean", 20.0),
            supply_log_std=data.get("supply_log_std", 3.0),
            holder_log_mean=data.get("holder_log_mean", 5.0),
            holder_log_std=data.get("holder_log_std", 1.5),
            volume_log_mean=data.get("volume_log_mean", 9.0),
            volume_log_std=data.get("volume_log_std", 2.5),
            liquidity_log_mean=data.get("liquidity_log_mean", 10.0),
            liquidity_log_std=data.get("liquidity_log_std", 2.0),
            fdv_log_mean=data.get("fdv_log_mean", 13.0),
            fdv_log_std=data.get("fdv_log_std", 2.0),
        )


class FeatureEngineer:
    """Feature extraction and transformation for ML training.

    Implements the unified 50-feature extraction pipeline:
    - Liquidity & Pool Metrics (12 features)
    - Token Supply & Holder (11 features)
    - Price & Volume (10 features)
    - On-Chain Risk (10 features)
    - Social & Sentiment (7 features)
    """

    def __init__(self, params: NormalizationParams | None = None):
        self.params = params or NormalizationParams()
        self._fitted = False

    def fit(self, data: pd.DataFrame) -> "FeatureEngineer":
        """Fit normalization parameters from training data."""
        # Log-transform numeric columns and compute mean/std

        # Pool TVL
        if "pool_tvl_sol" in data.columns:
            log_tvl = np.log1p(data["pool_tvl_sol"].fillna(0).clip(lower=0))
            self.params.tvl_log_mean = log_tvl.mean()
            self.params.tvl_log_std = max(log_tvl.std(), 0.1)
        elif "liquidity_usd" in data.columns:
            log_liq = np.log1p(data["liquidity_usd"].fillna(0).clip(lower=0))
            self.params.tvl_log_mean = log_liq.mean()
            self.params.tvl_log_std = max(log_liq.std(), 0.1)
            self.params.liquidity_log_mean = log_liq.mean()
            self.params.liquidity_log_std = max(log_liq.std(), 0.1)

        # Pool age
        if "pool_age_seconds" in data.columns:
            log_age = np.log1p(data["pool_age_seconds"].fillna(0).clip(lower=0))
            self.params.age_log_mean = log_age.mean()
            self.params.age_log_std = max(log_age.std(), 0.1)
        elif "age_seconds" in data.columns:
            log_age = np.log1p(data["age_seconds"].fillna(0).clip(lower=0))
            self.params.age_log_mean = log_age.mean()
            self.params.age_log_std = max(log_age.std(), 0.1)

        # Total supply
        if "total_supply" in data.columns:
            log_supply = np.log1p(data["total_supply"].fillna(0).clip(lower=0))
            self.params.supply_log_mean = log_supply.mean()
            self.params.supply_log_std = max(log_supply.std(), 0.1)
        elif "fdv_usd" in data.columns:
            log_fdv = np.log1p(data["fdv_usd"].fillna(0).clip(lower=0))
            self.params.fdv_log_mean = log_fdv.mean()
            self.params.fdv_log_std = max(log_fdv.std(), 0.1)

        # Holder count
        if "holder_count_unique" in data.columns:
            log_holders = np.log1p(data["holder_count_unique"].fillna(0).clip(lower=0))
            self.params.holder_log_mean = log_holders.mean()
            self.params.holder_log_std = max(log_holders.std(), 0.1)
        elif "holder_count" in data.columns:
            log_holders = np.log1p(data["holder_count"].fillna(0).clip(lower=0))
            self.params.holder_log_mean = log_holders.mean()
            self.params.holder_log_std = max(log_holders.std(), 0.1)

        # Volume
        if "volume_24h_usd" in data.columns:
            log_vol = np.log1p(data["volume_24h_usd"].fillna(0).clip(lower=0))
            self.params.volume_log_mean = log_vol.mean()
            self.params.volume_log_std = max(log_vol.std(), 0.1)

        self._fitted = True
        logger.info("Feature engineer fitted with params: %s", self.params.to_dict())

        return self

    def extract(self, raw: dict[str, Any]) -> FeatureSet:
        """Extract and normalize all 50 features from raw data."""
        features = np.zeros(FEATURE_COUNT)

        # === LIQUIDITY & POOL METRICS (0-11) ===
        features[FeatureIndex.POOL_TVL_SOL] = self._log_normalize(
            raw.get("pool_tvl_sol", raw.get("liquidity_usd", 0)),
            self.params.tvl_log_mean,
            self.params.tvl_log_std,
        )

        features[FeatureIndex.POOL_AGE_SECONDS] = self._log_normalize(
            raw.get("pool_age_seconds", raw.get("age_seconds", 0)),
            self.params.age_log_mean,
            self.params.age_log_std,
        )

        features[FeatureIndex.LP_LOCK_PERCENTAGE] = (
            raw.get("lp_lock_percentage", raw.get("lp_burn_pct", 0)) / 100.0
        )

        features[FeatureIndex.LP_CONCENTRATION] = self._sigmoid_normalize(
            raw.get("lp_concentration", 0), 1.0
        )

        features[FeatureIndex.LP_REMOVAL_VELOCITY] = np.tanh(
            raw.get("lp_removal_velocity", 0) / 10.0
        )

        features[FeatureIndex.LP_ADDITION_VELOCITY] = np.tanh(
            raw.get("lp_addition_velocity", 0) / 10.0
        )

        features[FeatureIndex.POOL_DEPTH_IMBALANCE] = np.clip(
            raw.get("pool_depth_imbalance", 0), -1.0, 1.0
        )

        features[FeatureIndex.SLIPPAGE_1PCT] = self._sigmoid_normalize(
            raw.get("slippage_1pct", 0), 100.0
        )

        features[FeatureIndex.SLIPPAGE_5PCT] = self._sigmoid_normalize(
            raw.get("slippage_5pct", 0), 500.0
        )

        features[FeatureIndex.UNIQUE_LP_PROVIDER_COUNT] = self._log_normalize(
            raw.get("unique_lp_provider_count", 1),
            self.params.lp_count_log_mean,
            self.params.lp_count_log_std,
        )

        features[FeatureIndex.DEPLOYER_LP_OWNERSHIP_PCT] = (
            raw.get("deployer_lp_ownership_pct", 100) / 50.0 - 1.0
        )

        features[FeatureIndex.EMERGENCY_LIQUIDITY_FLAG] = (
            1.0 if raw.get("emergency_liquidity_flag", False) else -1.0
        )

        # === TOKEN SUPPLY & HOLDER (12-22) ===
        features[FeatureIndex.TOTAL_SUPPLY] = self._log_normalize(
            raw.get("total_supply", raw.get("fdv_usd", 0)),
            self.params.supply_log_mean,
            self.params.supply_log_std,
        )

        features[FeatureIndex.DEPLOYER_HOLDINGS_PCT] = (
            raw.get("deployer_holdings_pct", raw.get("top_holder_pct", 0)) / 50.0 - 1.0
        )

        features[FeatureIndex.TOP_10_HOLDER_CONCENTRATION] = (
            raw.get("top_10_holder_concentration", raw.get("top_holder_pct", 0)) / 50.0 - 1.0
        )

        features[FeatureIndex.HOLDER_COUNT_UNIQUE] = self._log_normalize(
            raw.get("holder_count_unique", raw.get("holder_count", 0)),
            self.params.holder_log_mean,
            self.params.holder_log_std,
        )

        features[FeatureIndex.HOLDER_GROWTH_VELOCITY] = np.tanh(
            raw.get("holder_growth_velocity", raw.get("holder_growth_rate", 0))
        )

        features[FeatureIndex.TRANSFER_CONCENTRATION] = self._sigmoid_normalize(
            raw.get("transfer_concentration", 0), 1.0
        )

        features[FeatureIndex.SNIPER_BOT_COUNT_T0] = self._log_normalize(
            raw.get("sniper_bot_count_t0", 0), 2.0, 1.0
        )

        features[FeatureIndex.BOT_TO_HUMAN_RATIO] = self._sigmoid_normalize(
            raw.get("bot_to_human_ratio", 0), 0.5
        )

        features[FeatureIndex.LARGE_HOLDER_CHURN] = np.tanh(raw.get("large_holder_churn", 0) / 10.0)

        mint_revoked = raw.get(
            "mint_authority_revoked", not raw.get("mint_authority_enabled", False)
        )
        features[FeatureIndex.MINT_AUTHORITY_REVOKED] = 1.0 if mint_revoked else -1.0

        features[FeatureIndex.TOKEN_FREEZEABLE] = (
            -1.0 if raw.get("token_freezeable", raw.get("freeze_authority_enabled", False)) else 1.0
        )

        # === PRICE & VOLUME (23-32) ===
        features[FeatureIndex.PRICE_MOMENTUM_30S] = np.tanh(
            raw.get("price_momentum_30s", raw.get("price_change_1h", 0)) / 50.0
        )

        features[FeatureIndex.PRICE_MOMENTUM_5M] = np.tanh(
            raw.get("price_momentum_5m", raw.get("price_change_24h", 0)) / 100.0
        )

        features[FeatureIndex.VOLATILITY_5M] = self._sigmoid_normalize(
            raw.get("volatility_5m", raw.get("volatility_1h", 0)), 20.0
        )

        features[FeatureIndex.VOLUME_ACCELERATION] = np.tanh(
            raw.get("volume_acceleration", 0) / 100.0
        )

        features[FeatureIndex.BUY_VOLUME_RATIO] = raw.get("buy_volume_ratio", 0.5) * 2.0 - 1.0

        features[FeatureIndex.TRADE_SIZE_VARIANCE] = self._sigmoid_normalize(
            raw.get("trade_size_variance", 0), 10.0
        )

        features[FeatureIndex.VWAP_DEVIATION] = np.tanh(raw.get("vwap_deviation", 0) / 10.0)

        features[FeatureIndex.PRICE_IMPACT_1PCT] = self._sigmoid_normalize(
            raw.get("price_impact_1pct", 0), 5.0
        )

        features[FeatureIndex.CONSECUTIVE_BUYS] = self._log_normalize(
            raw.get("consecutive_buys", 0), 2.0, 1.0
        )

        features[FeatureIndex.MAX_BUY_IN_WINDOW] = self._log_normalize(
            raw.get("max_buy_in_window", 0), 5.0, 2.0
        )

        # === ON-CHAIN RISK (33-42) ===
        is_mintable = raw.get("contract_is_mintable", raw.get("mint_authority_enabled", False))
        features[FeatureIndex.CONTRACT_IS_MINTABLE] = -1.0 if is_mintable else 1.0

        features[FeatureIndex.CONTRACT_TRANSFER_FEE] = self._sigmoid_normalize(
            raw.get("contract_transfer_fee", 0), 5.0
        )

        features[FeatureIndex.HIDDEN_FEE_DETECTED] = (
            -1.0 if raw.get("hidden_fee_detected", False) else 1.0
        )

        features[FeatureIndex.CIRCULAR_TRADING_SCORE] = (
            raw.get("circular_trading_score", 0) * 2.0 - 1.0
        )

        # Benford's law p-value (higher = more natural, mapped to [-1, 1])
        features[FeatureIndex.BENFORD_LAW_PVALUE] = raw.get("benford_law_pvalue", 0.5) * 2.0 - 1.0

        features[FeatureIndex.ADDRESS_CLUSTERING_RISK] = (
            raw.get("address_clustering_risk", 0) * 2.0 - 1.0
        )

        features[FeatureIndex.PROXY_CONTRACT_FLAG] = (
            -1.0 if raw.get("proxy_contract_flag", False) else 1.0
        )

        features[FeatureIndex.UNVERIFIED_CODE_FLAG] = (
            -1.0 if raw.get("unverified_code_flag", False) else 1.0
        )

        features[FeatureIndex.EXTERNAL_TRANSFER_FLAG] = (
            -1.0 if raw.get("external_transfer_flag", False) else 1.0
        )

        # Rug pull ML score (0 = safe, 1 = rug, inverted so positive is good)
        features[FeatureIndex.RUG_PULL_ML_SCORE] = 1.0 - 2.0 * raw.get(
            "rug_pull_ml_score", raw.get("goplus_risk_score", 0)
        )

        # === SOCIAL & SENTIMENT (43-49) ===
        features[FeatureIndex.TWITTER_MENTION_VELOCITY] = self._log_normalize(
            raw.get("twitter_mention_velocity", 0), 3.0, 1.5
        )

        # Sentiment score: -1 (negative) to 1 (positive)
        features[FeatureIndex.TWITTER_SENTIMENT_SCORE] = np.clip(
            raw.get("twitter_sentiment_score", 0), -1.0, 1.0
        )

        features[FeatureIndex.TELEGRAM_USER_GROWTH] = np.tanh(
            raw.get("telegram_user_growth", 0) / 100.0
        )

        features[FeatureIndex.TELEGRAM_MESSAGE_VELOCITY] = self._log_normalize(
            raw.get("telegram_message_velocity", 0), 4.0, 2.0
        )

        features[FeatureIndex.DISCORD_INVITE_ACTIVITY] = self._log_normalize(
            raw.get("discord_invite_activity", 0), 3.0, 1.5
        )

        features[FeatureIndex.INFLUENCER_MENTION_FLAG] = (
            1.0 if raw.get("influencer_mention_flag", False) else -1.0
        )

        # Social authenticity: 0 = fake, 1 = authentic
        features[FeatureIndex.SOCIAL_AUTHENTICITY_SCORE] = (
            raw.get("social_authenticity_score", 0.5) * 2.0 - 1.0
        )

        # Extra features for training
        extra = np.zeros(2)
        extra[0] = raw.get("source_priority", 0) / 100.0  # Normalize to ~0-1
        extra[1] = raw.get("time_of_day_hour", 12) / 24.0  # Hour as fraction

        fs = FeatureSet(features=features, extra_features=extra)
        fs.feature_dict = fs.to_dict()

        return fs

    def extract_batch(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Extract features from a DataFrame.

        Returns:
            Tuple of (features array, labels array)
        """
        features_list = []
        labels = []

        for _, row in df.iterrows():
            raw = row.to_dict()
            fs = self.extract(raw)
            features_list.append(fs.features)

            # Determine label (profitable or not)
            if "pnl_pct" in raw and raw["pnl_pct"] is not None:
                labels.append(1 if raw["pnl_pct"] > 0 else 0)
            elif "was_profitable_counterfactual" in raw:
                labels.append(1 if raw["was_profitable_counterfactual"] else 0)
            else:
                labels.append(0)

        X = np.array(features_list)
        y = np.array(labels)

        # CRITICAL: Replace NaN/Inf values with 0 to prevent sklearn crashes
        # This handles missing enrichment data (e.g., liquidity_usd = None -> NaN after transform)
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0 or inf_count > 0:
            logger.warning(
                f"Feature matrix contains {nan_count} NaN and {inf_count} Inf values, "
                f"replacing with 0 (likely missing enrichment data)"
            )
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        return X, y

    def extract_for_signal_classification(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Extract features with 3-class labels for signal generation.

        Labels:
            0 = SELL (price drops >2% in 15 min)
            1 = HOLD (neither)
            2 = BUY (price rises >6% in 15 min)

        Returns:
            Tuple of (features array, labels array)
        """
        features_list = []
        labels = []

        for _, row in df.iterrows():
            raw = row.to_dict()
            fs = self.extract(raw)
            features_list.append(fs.features)

            # Determine 3-class label
            future_return = raw.get("future_return_15m", raw.get("pnl_pct", 0))
            if future_return >= 6.0:
                labels.append(2)  # BUY
            elif future_return <= -2.0:
                labels.append(0)  # SELL
            else:
                labels.append(1)  # HOLD

        X = np.array(features_list)
        y = np.array(labels)

        # CRITICAL: Replace NaN/Inf values with 0 to prevent sklearn crashes
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0 or inf_count > 0:
            logger.warning(
                f"Feature matrix contains {nan_count} NaN and {inf_count} Inf values, "
                f"replacing with 0 (likely missing enrichment data)"
            )
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        return X, y

    def _log_normalize(self, value: float, mean: float, std: float) -> float:
        """Log-normalize a value."""
        log_value = np.log1p(max(0, value))
        normalized = (log_value - mean) / std
        return np.clip(normalized / 3.0, -1.0, 1.0)

    def _sigmoid_normalize(self, value: float, scale: float) -> float:
        """Sigmoid-like normalization."""
        x = value / scale
        return 2.0 / (1.0 + np.exp(-x)) - 1.0

    def get_feature_importance_names(self) -> list[str]:
        """Get feature names for importance analysis."""
        return FeatureSet().FEATURE_NAMES

    def export_params(self) -> dict[str, Any]:
        """Export normalization parameters for Hot Path."""
        return self.params.to_dict()

    def import_params(self, params_dict: dict[str, Any]):
        """Import normalization parameters."""
        self.params = NormalizationParams.from_dict(params_dict)
        self._fitted = True


def compute_sample_weights(
    outcomes: list[dict[str, Any]],
    traded_weight: float = 1.0,
    counterfactual_weight: float = 0.5,
) -> np.ndarray:
    """Compute sample weights for training.

    Real trades are weighted higher than counterfactuals.
    """
    weights = []
    for outcome in outcomes:
        if outcome.get("outcome_type") == "traded":
            weights.append(traded_weight)
        else:
            weights.append(counterfactual_weight)
    return np.array(weights)


def balance_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    max_ratio: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balance dataset by undersampling majority class.

    Args:
        features: Feature matrix
        labels: Label array
        weights: Sample weights
        max_ratio: Maximum ratio of majority to minority class

    Returns:
        Balanced features, labels, weights
    """
    positive_idx = np.where(labels == 1)[0]
    negative_idx = np.where(labels == 0)[0]

    minority_count = min(len(positive_idx), len(negative_idx))
    majority_count = max(len(positive_idx), len(negative_idx))

    if majority_count <= minority_count * max_ratio:
        return features, labels, weights

    # Undersample majority class
    target_majority = int(minority_count * max_ratio)

    if len(positive_idx) > len(negative_idx):
        # Positive is majority
        np.random.shuffle(positive_idx)
        positive_idx = positive_idx[:target_majority]
    else:
        # Negative is majority
        np.random.shuffle(negative_idx)
        negative_idx = negative_idx[:target_majority]

    selected_idx = np.concatenate([positive_idx, negative_idx])
    np.random.shuffle(selected_idx)

    return features[selected_idx], labels[selected_idx], weights[selected_idx]


def get_feature_category_names() -> dict[str, list[str]]:
    """Get feature names grouped by category."""
    fs = FeatureSet()
    return {
        "liquidity": fs.FEATURE_NAMES[0:12],
        "holder": fs.FEATURE_NAMES[12:23],
        "price": fs.FEATURE_NAMES[23:33],
        "risk": fs.FEATURE_NAMES[33:43],
        "social": fs.FEATURE_NAMES[43:50],
    }
