"""
Dynamic Survivorship Model - Per-token survivorship bias discount.

Calculates survivorship bias discount dynamically based on token characteristics:
- Token age and maturity
- Deployer history and reputation
- LP lock percentage
- Holder concentration
- Historical rug rate for similar tokens

Replaces static 15% discount with dynamic 5-50% based on risk factors.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DeployerStats:
    """Statistics for a token deployer address."""
    address: str
    total_tokens: int = 0
    successful_tokens: int = 0  # Tokens that lasted > 24h
    rug_count: int = 0
    avg_token_lifespan_hours: float = 0.0

    @property
    def rug_rate(self) -> float:
        """Historical rug rate."""
        if self.total_tokens == 0:
            return 0.0
        return self.rug_count / self.total_tokens

    @property
    def success_rate(self) -> float:
        """Token success rate (> 24h lifespan)."""
        if self.total_tokens == 0:
            return 0.0
        return self.successful_tokens / self.total_tokens


@dataclass
class TokenRiskFeatures:
    """Risk features for survivorship discount calculation."""
    # Token identity
    mint: str
    symbol: str = ""

    # Age and maturity
    age_hours: float = 0.0
    first_trade_slot: int = 0

    # Deployer risk
    deployer_address: Optional[str] = None
    deployer_total_tokens: int = 0
    deployer_rug_count: int = 0
    deployer_avg_lifespan_hours: float = 0.0

    # Liquidity
    lp_locked_pct: float = 0.0
    lp_lock_duration_days: float = 0.0

    # Holder distribution
    holder_count: int = 0
    top_holder_pct: float = 0.0  # Top 10 holders %
    unique_buyers_24h: int = 0

    # Trading activity
    volume_24h_usd: float = 0.0
    trades_24h: int = 0
    buy_sell_ratio: float = 1.0

    # Authority status
    mint_authority_enabled: bool = False
    freeze_authority_enabled: bool = False

    # ML scores
    fraud_score: float = 0.0
    regime: str = "chop"


@dataclass
class SurvivorshipConfig:
    """Configuration for survivorship discount calculation."""
    # Base discount range
    min_discount: float = 0.05  # 5% minimum
    max_discount: float = 0.50  # 50% maximum
    base_discount: float = 0.08  # 8% base

    # Age factors (token maturity)
    age_maturity_hours: float = 24.0  # Hours to reach "mature"
    age_discount_reduction: float = 0.15  # Max reduction from age

    # Deployer factors
    unknown_deployer_penalty: float = 0.10  # Unknown deployer
    new_deployer_penalty: float = 0.08  # < 3 tokens
    high_rug_rate_penalty: float = 0.20  # Rug rate > 30%

    # LP factors
    no_lp_lock_penalty: float = 0.10
    short_lp_lock_penalty: float = 0.05  # < 30 days

    # Holder factors
    low_holder_penalty: float = 0.08  # < 50 holders
    concentrated_holders_penalty: float = 0.10  # Top 10 > 80%

    # Authority factors
    mint_authority_penalty: float = 0.12
    freeze_authority_penalty: float = 0.08

    # Fraud integration
    fraud_score_weight: float = 0.20  # Max penalty from fraud score

    # Regime adjustments
    bear_regime_penalty: float = 0.05
    mev_heavy_regime_penalty: float = 0.07


class DynamicSurvivorshipModel:
    """Calculates dynamic per-token survivorship discount.

    The discount represents the probability that historical backtest
    performance is overstated due to survivorship bias - tokens that
    failed quickly are underrepresented in historical data.

    Higher discount = more skeptical of historical returns
    """

    def __init__(self, config: Optional[SurvivorshipConfig] = None):
        self.config = config or SurvivorshipConfig()

        # Cache for recent calculations
        self._cache: Dict[str, float] = {}
        self._cache_timestamps: Dict[str, int] = {}
        self._cache_ttl_ms = 60000  # 1 minute cache

    def calculate_discount(self, features: TokenRiskFeatures) -> float:
        """Calculate survivorship discount for a token.

        Args:
            features: Token risk features

        Returns:
            Discount factor (0.05 to 0.50)
        """
        # Check cache
        cached = self._get_cached(features.mint)
        if cached is not None:
            return cached

        # Start with base discount
        discount = self.config.base_discount

        # 1. Age factor (older tokens are more reliable)
        age_factor = self._calculate_age_factor(features.age_hours)
        discount -= age_factor  # Reduce discount for mature tokens

        # 2. Deployer factor
        deployer_penalty = self._calculate_deployer_penalty(features)
        discount += deployer_penalty

        # 3. LP lock factor
        lp_penalty = self._calculate_lp_penalty(features)
        discount += lp_penalty

        # 4. Holder distribution factor
        holder_penalty = self._calculate_holder_penalty(features)
        discount += holder_penalty

        # 5. Authority factor
        authority_penalty = self._calculate_authority_penalty(features)
        discount += authority_penalty

        # 6. Fraud score integration
        fraud_penalty = features.fraud_score * self.config.fraud_score_weight
        discount += fraud_penalty

        # 7. Regime adjustment
        regime_penalty = self._calculate_regime_penalty(features.regime)
        discount += regime_penalty

        # Clamp to valid range
        discount = max(self.config.min_discount, min(self.config.max_discount, discount))

        # Cache result
        self._set_cached(features.mint, discount)

        return discount

    def _calculate_age_factor(self, age_hours: float) -> float:
        """Calculate age-based discount reduction.

        Older tokens have proven survivorship, so reduce discount.
        """
        if age_hours <= 0:
            return 0.0

        # Logistic curve: rapid reduction early, plateaus later
        maturity_ratio = age_hours / self.config.age_maturity_hours
        age_factor = self.config.age_discount_reduction * (1 - math.exp(-2 * maturity_ratio))

        return min(age_factor, self.config.age_discount_reduction)

    def _calculate_deployer_penalty(self, features: TokenRiskFeatures) -> float:
        """Calculate deployer-based penalty."""
        if features.deployer_address is None:
            # Unknown deployer
            return self.config.unknown_deployer_penalty

        # New deployer (< 3 tokens)
        if features.deployer_total_tokens < 3:
            return self.config.new_deployer_penalty

        # High rug rate
        if features.deployer_total_tokens > 0:
            rug_rate = features.deployer_rug_count / features.deployer_total_tokens
            if rug_rate > 0.3:
                return self.config.high_rug_rate_penalty * min(1.0, rug_rate / 0.5)

        # Short-lived tokens
        if features.deployer_avg_lifespan_hours < 12.0:
            return 0.05

        return 0.0

    def _calculate_lp_penalty(self, features: TokenRiskFeatures) -> float:
        """Calculate LP lock penalty."""
        if features.lp_locked_pct < 50:
            return self.config.no_lp_lock_penalty

        if features.lp_lock_duration_days < 30:
            return self.config.short_lp_lock_penalty

        return 0.0

    def _calculate_holder_penalty(self, features: TokenRiskFeatures) -> float:
        """Calculate holder distribution penalty."""
        penalty = 0.0

        # Low holder count
        if features.holder_count < 50:
            penalty += self.config.low_holder_penalty * (1 - features.holder_count / 50)

        # Concentrated holdings
        if features.top_holder_pct > 80:
            concentration_factor = (features.top_holder_pct - 80) / 20
            penalty += self.config.concentrated_holders_penalty * concentration_factor

        return min(penalty, 0.15)  # Cap combined holder penalty

    def _calculate_authority_penalty(self, features: TokenRiskFeatures) -> float:
        """Calculate authority-based penalty."""
        penalty = 0.0

        if features.mint_authority_enabled:
            penalty += self.config.mint_authority_penalty

        if features.freeze_authority_enabled:
            penalty += self.config.freeze_authority_penalty

        return penalty

    def _calculate_regime_penalty(self, regime: str) -> float:
        """Calculate regime-based penalty."""
        regime_lower = regime.lower()

        if regime_lower == "bear":
            return self.config.bear_regime_penalty
        elif regime_lower == "mev_heavy":
            return self.config.mev_heavy_regime_penalty

        return 0.0

    def _get_cached(self, mint: str) -> Optional[float]:
        """Get cached discount if still valid."""
        if mint not in self._cache:
            return None

        timestamp = self._cache_timestamps.get(mint, 0)
        now_ms = int(datetime.now().timestamp() * 1000)

        if now_ms - timestamp > self._cache_ttl_ms:
            del self._cache[mint]
            del self._cache_timestamps[mint]
            return None

        return self._cache[mint]

    def _set_cached(self, mint: str, discount: float) -> None:
        """Cache a discount value."""
        self._cache[mint] = discount
        self._cache_timestamps[mint] = int(datetime.now().timestamp() * 1000)

        # Limit cache size
        if len(self._cache) > 1000:
            # Remove oldest entries
            oldest = sorted(self._cache_timestamps.items(), key=lambda x: x[1])[:100]
            for mint, _ in oldest:
                del self._cache[mint]
                del self._cache_timestamps[mint]

    def get_discount_breakdown(self, features: TokenRiskFeatures) -> Dict[str, float]:
        """Get detailed breakdown of discount components."""
        age_factor = self._calculate_age_factor(features.age_hours)
        deployer_penalty = self._calculate_deployer_penalty(features)
        lp_penalty = self._calculate_lp_penalty(features)
        holder_penalty = self._calculate_holder_penalty(features)
        authority_penalty = self._calculate_authority_penalty(features)
        fraud_penalty = features.fraud_score * self.config.fraud_score_weight
        regime_penalty = self._calculate_regime_penalty(features.regime)

        total = (
            self.config.base_discount
            - age_factor
            + deployer_penalty
            + lp_penalty
            + holder_penalty
            + authority_penalty
            + fraud_penalty
            + regime_penalty
        )

        return {
            "base_discount": self.config.base_discount,
            "age_reduction": -age_factor,
            "deployer_penalty": deployer_penalty,
            "lp_penalty": lp_penalty,
            "holder_penalty": holder_penalty,
            "authority_penalty": authority_penalty,
            "fraud_penalty": fraud_penalty,
            "regime_penalty": regime_penalty,
            "total_discount": max(self.config.min_discount, min(self.config.max_discount, total)),
        }

    def clear_cache(self) -> int:
        """Clear the discount cache. Returns number of entries cleared."""
        count = len(self._cache)
        self._cache.clear()
        self._cache_timestamps.clear()
        return count


def create_features_from_dict(data: Dict[str, Any]) -> TokenRiskFeatures:
    """Create TokenRiskFeatures from a dictionary.

    Useful for creating features from enrichment data or API responses.
    """
    return TokenRiskFeatures(
        mint=data.get("mint", ""),
        symbol=data.get("symbol", ""),
        age_hours=data.get("age_hours", 0.0),
        first_trade_slot=data.get("first_trade_slot", 0),
        deployer_address=data.get("deployer_address"),
        deployer_total_tokens=data.get("deployer_total_tokens", 0),
        deployer_rug_count=data.get("deployer_rug_count", 0),
        deployer_avg_lifespan_hours=data.get("deployer_avg_lifespan_hours", 0.0),
        lp_locked_pct=data.get("lp_locked_pct", 0.0),
        lp_lock_duration_days=data.get("lp_lock_duration_days", 0.0),
        holder_count=data.get("holder_count", 0),
        top_holder_pct=data.get("top_holder_pct", 0.0),
        unique_buyers_24h=data.get("unique_buyers_24h", 0),
        volume_24h_usd=data.get("volume_24h_usd", 0.0),
        trades_24h=data.get("trades_24h", 0),
        buy_sell_ratio=data.get("buy_sell_ratio", 1.0),
        mint_authority_enabled=data.get("mint_authority_enabled", False),
        freeze_authority_enabled=data.get("freeze_authority_enabled", False),
        fraud_score=data.get("fraud_score", 0.0),
        regime=data.get("regime", "chop"),
    )
