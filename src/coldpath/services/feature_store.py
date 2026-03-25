"""
Feast Feature Store for 2DEXY.

Provides real-time feature serving for ML inference with:
- Low-latency feature retrieval (<10ms)
- Feature versioning and lineage
- Offline/online store synchronization
- Feature drift monitoring

Architecture:
    ┌─────────────────┐     ┌──────────────────┐
    │ Token Events    │────►│ Offline Store    │
    │ (PostgreSQL)    │     │ (SQLite/Parquet) │
    └─────────────────┘     └──────────────────┘
                                    │
                                    ▼ Materialization
    ┌─────────────────┐     ┌──────────────────┐
    │ HotPath/Rust    │◄────│ Online Store     │
    │ Inference       │     │ (Redis/SQLite)   │
    └─────────────────┘     └──────────────────┘

Usage:
    from coldpath.services.feature_store import FeatureStore

    store = FeatureStore()

    # Get features for inference
    features = await store.get_online_features(
        entity_ids=["So11111111111111111111111111111111111111112"],
        feature_names=["liquidity_usd", "volume_24h", "fraud_score"],
    )

    # Materialize features from offline to online
    await store.materialize_incremental()
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FEAST_AVAILABLE = False
try:
    from feast import (
        Entity,  # noqa: F401
        Feature,  # noqa: F401
        FeatureService,  # noqa: F401
        FeatureView,  # noqa: F401
        FileSource,  # noqa: F401
    )
    from feast.data_source import DataSource  # noqa: F401
    from feast.feature_store import FeatureStore as FeastStore
    from feast.types import (
        Float32,  # noqa: F401
        Int64,  # noqa: F401
        String,  # noqa: F401
    )

    FEAST_AVAILABLE = True
    logger.info("Feast feature store available")
except ImportError:
    logger.debug("Feast not available, using native implementation")


@dataclass
class TokenFeatures:
    """Features for a Solana token."""

    token_mint: str
    timestamp: datetime

    # Liquidity features
    liquidity_usd: float = 0.0
    fdv_usd: float = 0.0
    liquidity_to_fdv_ratio: float = 0.0

    # Volume features
    volume_24h: float = 0.0
    volume_1h: float = 0.0
    volume_change_24h: float = 0.0
    buy_volume_24h: float = 0.0
    sell_volume_24h: float = 0.0
    buy_sell_ratio: float = 0.0

    # Price features
    price_usd: float = 0.0
    price_change_1h: float = 0.0
    price_change_24h: float = 0.0
    price_high_24h: float = 0.0
    price_low_24h: float = 0.0
    price_volatility_24h: float = 0.0

    # Holder features
    holder_count: int = 0
    holder_change_24h: int = 0
    top_holder_pct: float = 0.0
    whale_count: int = 0  # Holders > 1% supply

    # Risk features
    fraud_score: float = 0.0
    rug_pull_probability: float = 0.0
    mint_authority_enabled: bool = False
    freeze_authority_enabled: bool = False
    lp_locked: bool = False
    lp_locked_pct: float = 0.0

    # Age features
    token_age_hours: float = 0.0
    first_swap_timestamp: datetime | None = None

    # Social features (from external APIs)
    twitter_mentions_24h: int = 0
    telegram_members: int = 0
    sentiment_score: float = 0.5

    # Derived features
    momentum_score: float = 0.0
    trend_direction: str = "neutral"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_mint": self.token_mint,
            "timestamp": self.timestamp.isoformat(),
            "liquidity_usd": self.liquidity_usd,
            "fdv_usd": self.fdv_usd,
            "liquidity_to_fdv_ratio": self.liquidity_to_fdv_ratio,
            "volume_24h": self.volume_24h,
            "volume_1h": self.volume_1h,
            "volume_change_24h": self.volume_change_24h,
            "buy_volume_24h": self.buy_volume_24h,
            "sell_volume_24h": self.sell_volume_24h,
            "buy_sell_ratio": self.buy_sell_ratio,
            "price_usd": self.price_usd,
            "price_change_1h": self.price_change_1h,
            "price_change_24h": self.price_change_24h,
            "price_high_24h": self.price_high_24h,
            "price_low_24h": self.price_low_24h,
            "price_volatility_24h": self.price_volatility_24h,
            "holder_count": self.holder_count,
            "holder_change_24h": self.holder_change_24h,
            "top_holder_pct": self.top_holder_pct,
            "whale_count": self.whale_count,
            "fraud_score": self.fraud_score,
            "rug_pull_probability": self.rug_pull_probability,
            "mint_authority_enabled": self.mint_authority_enabled,
            "freeze_authority_enabled": self.freeze_authority_enabled,
            "lp_locked": self.lp_locked,
            "lp_locked_pct": self.lp_locked_pct,
            "token_age_hours": self.token_age_hours,
            "twitter_mentions_24h": self.twitter_mentions_24h,
            "telegram_members": self.telegram_members,
            "sentiment_score": self.sentiment_score,
            "momentum_score": self.momentum_score,
            "trend_direction": self.trend_direction,
        }

    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML inference."""
        return np.array(
            [
                self.liquidity_usd,
                self.fdv_usd,
                self.volume_24h,
                float(self.holder_count),
                self.top_holder_pct,
                self.fraud_score,
                self.token_age_hours,
                self.liquidity_to_fdv_ratio,
                self.volume_change_24h,
                self.price_change_24h,
                self.price_volatility_24h,
                float(self.holder_change_24h),
                self.rug_pull_probability,
                float(self.lp_locked_pct),
                self.momentum_score,
            ]
        )


@dataclass
class FeatureStoreConfig:
    """Configuration for feature store."""

    repo_path: Path = Path("feature_repo")
    online_store_type: str = "sqlite"  # sqlite, redis, postgres
    offline_store_type: str = "sqlite"  # sqlite, postgres, parquet

    redis_host: str = "localhost"
    redis_port: int = 6379

    postgres_host: str = "localhost"
    redis_port: int = 5432
    postgres_database: str = "feast"

    materialization_interval_minutes: int = 5
    feature_ttl_hours: int = 24


class FeatureStore:
    """Feature store for real-time ML features.

    Supports both Feast (when available) and native implementation.
    """

    def __init__(self, config: FeatureStoreConfig | None = None):
        self.config = config or FeatureStoreConfig()
        self._store: Any | None = None
        self._cache: dict[str, TokenFeatures] = {}
        self._cache_timestamps: dict[str, datetime] = {}

        if FEAST_AVAILABLE:
            self._init_feast()
        else:
            self._init_native()

    def _init_feast(self):
        """Initialize Feast feature store."""
        repo_path = self.config.repo_path
        repo_path.mkdir(parents=True, exist_ok=True)

        self._create_feast_repo(repo_path)

        self._store = FeastStore(repo_path=str(repo_path))
        logger.info(f"Feast feature store initialized at {repo_path}")

    def _create_feast_repo(self, repo_path: Path):
        """Create Feast repository structure."""
        feature_file = repo_path / "token_features.py"
        if not feature_file.exists():
            feature_file.write_text(self._get_feast_definitions())

        config_file = repo_path / "feature_store.yaml"
        if not config_file.exists():
            config_file.write_text(self._get_feast_config())

    def _get_feast_definitions(self) -> str:
        """Get Feast feature definitions."""
        return """
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.types import Float32, Int64, String

# Token entity
token = Entity(
    name="token",
    value_type=ValueType.STRING,
    description="Solana token mint address",
)

# Token stats feature view
token_stats_source = FileSource(
    path="data/token_stats.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

token_stats_fv = FeatureView(
    name="token_stats",
    entities=["token"],
    ttl=timedelta(hours=24),
    features=[
        Feature(name="liquidity_usd", dtype=Float32),
        Feature(name="fdv_usd", dtype=Float32),
        Feature(name="volume_24h", dtype=Float32),
        Feature(name="holder_count", dtype=Int64),
        Feature(name="top_holder_pct", dtype=Float32),
        Feature(name="fraud_score", dtype=Float32),
        Feature(name="token_age_hours", dtype=Float32),
        Feature(name="momentum_score", dtype=Float32),
        Feature(name="price_change_24h", dtype=Float32),
        Feature(name="volatility_24h", dtype=Float32),
    ],
    online=True,
    batch_source=token_stats_source,
    tags={"team": "trading"},
)

# Risk features view
risk_source = FileSource(
    path="data/risk_features.parquet",
    event_timestamp_column="event_timestamp",
)

risk_fv = FeatureView(
    name="risk_features",
    entities=["token"],
    ttl=timedelta(hours=6),
    features=[
        Feature(name="rug_pull_probability", dtype=Float32),
        Feature(name="mint_authority_enabled", dtype=Int64),
        Feature(name="freeze_authority_enabled", dtype=Int64),
        Feature(name="lp_locked_pct", dtype=Float32),
    ],
    online=True,
    batch_source=risk_source,
    tags={"team": "risk"},
)
"""

    def _get_feast_config(self) -> str:
        """Get Feast configuration."""
        return """
project: dexy_features
registry: data/registry.db
provider: local
online_store:
    type: sqlite
    path: data/online_store.db
offline_store:
    type: sqlite
    path: data/offline_store.db
entity_key_serialization_version: 2
"""

    def _init_native(self):
        """Initialize native feature store (without Feast)."""
        data_dir = self.config.repo_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        self._online_store_path = data_dir / "online_store.json"
        if self._online_store_path.exists():
            with open(self._online_store_path) as f:
                raw_data = json.load(f)
                # Convert timestamp strings back to datetime objects
                for _k, v in raw_data.items():
                    if "timestamp" in v and isinstance(v["timestamp"], str):
                        v["timestamp"] = datetime.fromisoformat(v["timestamp"])
                    if (
                        "first_swap_timestamp" in v
                        and v["first_swap_timestamp"] is not None
                        and isinstance(v["first_swap_timestamp"], str)
                    ):
                        v["first_swap_timestamp"] = datetime.fromisoformat(
                            v["first_swap_timestamp"]
                        )
                self._cache = {k: TokenFeatures(**v) for k, v in raw_data.items()}

        logger.info("Native feature store initialized")

    async def get_online_features(
        self,
        entity_ids: list[str],
        feature_names: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Get features for online inference.

        Args:
            entity_ids: List of token mint addresses
            feature_names: Specific features to retrieve (None = all)

        Returns:
            Dict mapping entity_id -> feature_name -> value
        """
        if FEAST_AVAILABLE and self._store:
            return await self._get_feast_online_features(entity_ids, feature_names)
        else:
            return self._get_native_online_features(entity_ids, feature_names)

    async def _get_feast_online_features(
        self,
        entity_ids: list[str],
        feature_names: list[str] | None,
    ) -> dict[str, dict[str, Any]]:
        """Get features from Feast online store."""
        entity_rows = [{"token": tid} for tid in entity_ids]

        if feature_names is None:
            features = [
                "token_stats:liquidity_usd",
                "token_stats:fdv_usd",
                "token_stats:volume_24h",
                "token_stats:holder_count",
                "token_stats:top_holder_pct",
                "token_stats:fraud_score",
                "token_stats:momentum_score",
                "risk_features:rug_pull_probability",
            ]
        else:
            features = [f"token_stats:{f}" for f in feature_names]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._store.get_online_features(
                features=features,
                entity_rows=entity_rows,
            ).to_dict(),
        )

        output = {}
        for i, entity_id in enumerate(entity_ids):
            output[entity_id] = {k: v[i] if isinstance(v, list) else v for k, v in result.items()}

        return output

    def _get_native_online_features(
        self,
        entity_ids: list[str],
        feature_names: list[str] | None,
    ) -> dict[str, dict[str, Any]]:
        """Get features from native store."""
        output = {}

        for entity_id in entity_ids:
            if entity_id in self._cache:
                features = self._cache[entity_id].to_dict()
                if feature_names:
                    features = {k: v for k, v in features.items() if k in feature_names}
                output[entity_id] = features
            else:
                output[entity_id] = {}

        return output

    async def get_token_features(self, token_mint: str) -> TokenFeatures | None:
        """Get all features for a token.

        Args:
            token_mint: Token mint address

        Returns:
            TokenFeatures or None if not found
        """
        result = await self.get_online_features([token_mint])

        if token_mint in result and result[token_mint]:
            data = result[token_mint]
            return TokenFeatures(
                token_mint=token_mint,
                timestamp=datetime.now(),
                liquidity_usd=data.get("liquidity_usd", 0.0),
                fdv_usd=data.get("fdv_usd", 0.0),
                volume_24h=data.get("volume_24h", 0.0),
                holder_count=int(data.get("holder_count", 0)),
                top_holder_pct=data.get("top_holder_pct", 0.0),
                fraud_score=data.get("fraud_score", 0.0),
                token_age_hours=data.get("token_age_hours", 0.0),
                momentum_score=data.get("momentum_score", 0.0),
            )

        return self._cache.get(token_mint)

    async def write_token_features(self, features: TokenFeatures):
        """Write token features to online store.

        Args:
            features: TokenFeatures to write
        """
        self._cache[features.token_mint] = features
        self._cache_timestamps[features.token_mint] = datetime.now()

        await self._persist_native_store()

    async def write_batch(self, features_list: list[TokenFeatures]):
        """Write multiple token features.

        Args:
            features_list: List of TokenFeatures to write
        """
        for features in features_list:
            self._cache[features.token_mint] = features
            self._cache_timestamps[features.token_mint] = datetime.now()

        await self._persist_native_store()

    async def _persist_native_store(self):
        """Persist native store to disk."""
        if not hasattr(self, "_online_store_path"):
            return

        data = {k: v.to_dict() for k, v in self._cache.items()}

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: self._online_store_path.write_text(json.dumps(data, default=str))
        )

    async def materialize_incremental(
        self,
        end_date: datetime | None = None,
    ):
        """Materialize incremental features from offline to online store.

        Args:
            end_date: End date for materialization (default: now)
        """
        if not FEAST_AVAILABLE or not self._store:
            logger.warning("Feast not available, skipping materialization")
            return

        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(minutes=self.config.materialization_interval_minutes)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._store.materialize_incremental(
                start_date=start_date,
                end_date=end_date,
            ),
        )

        logger.info(f"Materialized features from {start_date} to {end_date}")

    def get_feature_vector(self, token_mint: str) -> np.ndarray | None:
        """Get feature vector for ML inference (synchronous).

        Fast path for HotPath inference.

        Args:
            token_mint: Token mint address

        Returns:
            Feature vector as numpy array or None
        """
        if token_mint in self._cache:
            return self._cache[token_mint].to_feature_vector()
        return None

    def clear_expired(self, max_age_hours: int = 24):
        """Clear expired features from cache.

        Args:
            max_age_hours: Maximum age in hours
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        expired = [k for k, v in self._cache_timestamps.items() if v < cutoff]

        for k in expired:
            del self._cache[k]
            del self._cache_timestamps[k]

        if expired:
            logger.info(f"Cleared {len(expired)} expired features")

    def get_stats(self) -> dict[str, Any]:
        """Get feature store statistics."""
        return {
            "total_tokens": len(self._cache),
            "backend": "feast" if FEAST_AVAILABLE and self._store else "native",
            "config": {
                "online_store_type": self.config.online_store_type,
                "materialization_interval_minutes": self.config.materialization_interval_minutes,
            },
        }


FEATURE_DEFINITIONS = {
    "token_stats": [
        {"name": "liquidity_usd", "dtype": "float", "description": "Token liquidity in USD"},
        {"name": "fdv_usd", "dtype": "float", "description": "Fully diluted valuation in USD"},
        {"name": "volume_24h", "dtype": "float", "description": "24-hour trading volume"},
        {"name": "holder_count", "dtype": "int", "description": "Number of token holders"},
        {
            "name": "top_holder_pct",
            "dtype": "float",
            "description": "Percentage held by top holder",
        },
        {"name": "fraud_score", "dtype": "float", "description": "Risk score (0-1)"},
        {"name": "token_age_hours", "dtype": "float", "description": "Token age in hours"},
        {"name": "momentum_score", "dtype": "float", "description": "Price momentum indicator"},
    ],
    "risk_features": [
        {"name": "rug_pull_probability", "dtype": "float", "description": "Rug pull probability"},
        {"name": "mint_authority_enabled", "dtype": "bool", "description": "Mint authority status"},
        {
            "name": "freeze_authority_enabled",
            "dtype": "bool",
            "description": "Freeze authority status",
        },
        {"name": "lp_locked_pct", "dtype": "float", "description": "LP locked percentage"},
    ],
    "social_features": [
        {"name": "twitter_mentions_24h", "dtype": "int", "description": "Twitter mentions"},
        {"name": "telegram_members", "dtype": "int", "description": "Telegram member count"},
        {"name": "sentiment_score", "dtype": "float", "description": "Sentiment (0-1)"},
    ],
}
