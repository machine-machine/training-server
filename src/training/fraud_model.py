"""
Fraud/Rug prediction model.

Uses historical rug data to train a classifier for detecting potential rugs.
Supports AutoML (TPOT, Auto-sklearn) with graceful fallback to RandomForest.

Features:
- Mutual information-based feature selection to identify predictive characteristics
- Synthetic data augmentation (SMOTE, adversarial, feature-conditional) for class balancing
"""

import logging
import os
import pickle
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Import advanced MI selector
try:
    from ..learning.mutual_information import (
        MutualInformationSelector,
        MIConfig,
        SelectionMethod,
        SelectionResult,
    )
    ADVANCED_MI_AVAILABLE = True
except ImportError:
    ADVANCED_MI_AVAILABLE = False
    SelectionMethod = None  # type: ignore

# Import temporal synthetic generator
try:
    from .synthetic_temporal import (
        TemporalSyntheticGenerator,
        TokenLifecycleType,
    )
    TEMPORAL_SYNTHETIC_AVAILABLE = True
except ImportError:
    TEMPORAL_SYNTHETIC_AVAILABLE = False
    TemporalSyntheticGenerator = None  # type: ignore
    TokenLifecycleType = None  # type: ignore

logger = logging.getLogger(__name__)

# Check for optional SMOTE dependency
SMOTE_AVAILABLE = False
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.combine import SMOTETomek
    SMOTE_AVAILABLE = True
    logger.info("imbalanced-learn SMOTE available")
except Exception as e:
    logger.warning(f"SMOTE dependencies unavailable, continuing without SMOTE: {e}")
    pass

# Check for optional AutoML dependencies
TPOT_AVAILABLE = False
AUTOSKLEARN_AVAILABLE = False

try:
    from tpot import TPOTClassifier
    TPOT_AVAILABLE = True
    logger.info("TPOT AutoML available")
except Exception as e:
    logger.warning(f"TPOT unavailable, continuing without TPOT AutoML: {e}")
    pass

try:
    # Auto-sklearn only works on Linux
    import platform
    if platform.system() == "Linux":
        from autosklearn.classification import AutoSklearnClassifier
        AUTOSKLEARN_AVAILABLE = True
        logger.info("Auto-sklearn AutoML available")
except Exception as e:
    logger.warning(f"Auto-sklearn unavailable, continuing without Auto-sklearn: {e}")
    pass

# Check for GPU-accelerated XGBoost training
XGBOOST_GPU_AVAILABLE = False
try:
    import xgboost as xgb
    import subprocess
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        XGBOOST_GPU_AVAILABLE = True
        logger.info("XGBoost GPU training available (CUDA detected)")
except (ImportError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
    pass


@dataclass
class TokenFeatures:
    """Features for fraud prediction."""
    liquidity_usd: float
    fdv_usd: float
    holder_count: int
    top_holder_pct: float
    mint_authority_enabled: bool
    freeze_authority_enabled: bool
    lp_burn_pct: float
    age_seconds: int
    volume_24h: float
    price_change_1h: float


@dataclass
class EnhancedTokenFeatures(TokenFeatures):
    """Extended features including real-time behavioral signals for enhanced rug detection.

    Adds:
    - Deployer historical performance (critical for pattern detection)
    - Buy/sell ratio analysis (early warning indicator)
    - LP removal monitoring (critical rug signal)
    - Social signal analysis (hype vs reality)
    """
    # Deployer behavior features
    deployer_previous_tokens: int = 0  # How many tokens this deployer has created
    deployer_rug_count: int = 0  # How many of their tokens rugged
    deployer_rug_rate: float = 0.0  # Percentage of deployer's tokens that rugged
    deployer_avg_token_lifespan_hours: float = 0.0  # Avg lifespan of deployer's tokens
    deployer_wallet_age_days: int = 0  # How old is the deployer's wallet
    deployer_total_volume_sol: float = 0.0  # Total trading volume from deployer's tokens

    # Buy/sell ratio in first N minutes (critical early indicator)
    buy_sell_ratio_5min: float = 1.0  # Ratio of buy vs sell volume in first 5 minutes
    buy_sell_ratio_15min: float = 1.0  # Ratio in first 15 minutes
    unique_buyers_5min: int = 0  # Number of unique buyers in first 5 minutes
    unique_sellers_5min: int = 0  # Number of unique sellers in first 5 minutes
    avg_buy_size_sol: float = 0.0  # Average buy transaction size
    avg_sell_size_sol: float = 0.0  # Average sell transaction size
    large_sell_count: int = 0  # Number of sells > 10% of liquidity

    # LP behavior monitoring
    lp_removal_attempts: int = 0  # Number of attempted LP removals
    lp_removal_amount_pct: float = 0.0  # Total LP removed as percentage
    lp_unlock_timestamp: Optional[int] = None  # When LP unlocks (None = unlocked)
    lp_lock_duration_days: int = 0  # How long LP is locked
    has_lp_lock: bool = False  # Whether LP is locked at all

    # Social signals
    twitter_followers: int = 0  # Token's Twitter followers
    twitter_account_age_days: int = 0  # Age of Twitter account
    telegram_members: int = 0  # Telegram group members
    website_exists: bool = False  # Whether token has a website
    github_exists: bool = False  # Whether token has a GitHub

    # Transaction pattern analysis
    insider_buy_detected: bool = False  # Suspicious early buying pattern
    coordinated_selling: bool = False  # Multiple wallets selling simultaneously
    sybil_wallet_pct: float = 0.0  # Percentage of holders that appear to be sybil
    bundle_buy_detected: bool = False  # Jito bundle buying at launch

    # Market making signals
    fake_volume_score: float = 0.0  # 0-1 score indicating likely wash trading
    organic_volume_pct: float = 100.0  # Estimated organic volume percentage


@dataclass
class DeployerProfile:
    """Profile of a token deployer for historical analysis."""
    wallet_address: str
    first_seen_timestamp: int
    total_tokens_deployed: int
    tokens_rugged: int
    tokens_successful: int  # Survived > 7 days with positive price
    avg_token_lifespan_seconds: float
    total_volume_sol: float
    avg_max_mcap_sol: float  # Average peak market cap of their tokens

    @property
    def rug_rate(self) -> float:
        """Calculate rug rate as percentage."""
        if self.total_tokens_deployed == 0:
            return 0.0
        return self.tokens_rugged / self.total_tokens_deployed

    @property
    def wallet_age_days(self) -> int:
        """Calculate wallet age in days."""
        import time
        now_ms = int(time.time() * 1000)
        return (now_ms - self.first_seen_timestamp) // (86400 * 1000)


class DeployerTracker:
    """Track deployer history for pattern detection.

    This is a critical component for rug detection - many rug operators
    create multiple tokens in sequence.
    """

    def __init__(self):
        self._profiles: Dict[str, DeployerProfile] = {}
        self._token_deployers: Dict[str, str] = {}  # token_mint -> deployer_wallet

    def record_token_launch(
        self,
        deployer_wallet: str,
        token_mint: str,
        timestamp_ms: int,
    ):
        """Record a new token launch by a deployer."""
        self._token_deployers[token_mint] = deployer_wallet

        if deployer_wallet not in self._profiles:
            self._profiles[deployer_wallet] = DeployerProfile(
                wallet_address=deployer_wallet,
                first_seen_timestamp=timestamp_ms,
                total_tokens_deployed=0,
                tokens_rugged=0,
                tokens_successful=0,
                avg_token_lifespan_seconds=0,
                total_volume_sol=0,
                avg_max_mcap_sol=0,
            )

        self._profiles[deployer_wallet].total_tokens_deployed += 1
        logger.debug(f"Recorded token {token_mint} by deployer {deployer_wallet}")

    def record_rug(self, token_mint: str, lifespan_seconds: float, volume_sol: float):
        """Record that a token rugged."""
        deployer = self._token_deployers.get(token_mint)
        if deployer and deployer in self._profiles:
            profile = self._profiles[deployer]
            profile.tokens_rugged += 1
            self._update_avg_lifespan(profile, lifespan_seconds)
            profile.total_volume_sol += volume_sol

    def record_success(self, token_mint: str, lifespan_seconds: float, volume_sol: float, max_mcap_sol: float):
        """Record that a token was successful (survived > 7 days)."""
        deployer = self._token_deployers.get(token_mint)
        if deployer and deployer in self._profiles:
            profile = self._profiles[deployer]
            profile.tokens_successful += 1
            self._update_avg_lifespan(profile, lifespan_seconds)
            profile.total_volume_sol += volume_sol
            self._update_avg_mcap(profile, max_mcap_sol)

    def get_profile(self, deployer_wallet: str) -> Optional[DeployerProfile]:
        """Get deployer profile if exists."""
        return self._profiles.get(deployer_wallet)

    def get_deployer_for_token(self, token_mint: str) -> Optional[str]:
        """Get the deployer wallet for a token."""
        return self._token_deployers.get(token_mint)

    def get_risk_score(self, deployer_wallet: str) -> float:
        """Calculate a risk score (0-1) based on deployer history.

        Higher score = more risky.
        """
        profile = self._profiles.get(deployer_wallet)
        if not profile:
            return 0.3  # Unknown deployer - moderate baseline risk

        score = 0.0

        # Rug rate is the strongest signal
        if profile.rug_rate > 0.7:
            score += 0.5
        elif profile.rug_rate > 0.5:
            score += 0.35
        elif profile.rug_rate > 0.3:
            score += 0.2
        elif profile.rug_rate > 0.1:
            score += 0.1

        # Serial deployers are risky
        if profile.total_tokens_deployed > 10:
            score += 0.15
        elif profile.total_tokens_deployed > 5:
            score += 0.1
        elif profile.total_tokens_deployed > 2:
            score += 0.05

        # Short average lifespan is a red flag
        avg_hours = profile.avg_token_lifespan_seconds / 3600
        if avg_hours < 1:
            score += 0.2
        elif avg_hours < 6:
            score += 0.1
        elif avg_hours < 24:
            score += 0.05

        # New wallets are riskier
        if profile.wallet_age_days < 7:
            score += 0.1
        elif profile.wallet_age_days < 30:
            score += 0.05

        return min(1.0, score)

    def _update_avg_lifespan(self, profile: DeployerProfile, new_lifespan: float):
        """Update average lifespan with new data point."""
        n = profile.total_tokens_deployed
        if n <= 1:
            profile.avg_token_lifespan_seconds = new_lifespan
        else:
            profile.avg_token_lifespan_seconds = (
                profile.avg_token_lifespan_seconds * (n - 1) + new_lifespan
            ) / n

    def _update_avg_mcap(self, profile: DeployerProfile, new_mcap: float):
        """Update average max mcap with new data point."""
        n = profile.tokens_successful
        if n <= 1:
            profile.avg_max_mcap_sol = new_mcap
        else:
            profile.avg_max_mcap_sol = (
                profile.avg_max_mcap_sol * (n - 1) + new_mcap
            ) / n


class BuySellAnalyzer:
    """Analyze buy/sell patterns for early rug warning.

    Key indicators:
    - High sell ratio in first minutes = likely insider dumping
    - Few unique buyers, many sellers = pump and dump
    - Large sells relative to liquidity = whale exit
    """

    def __init__(self):
        self._token_trades: Dict[str, List[Dict]] = {}

    def record_trade(
        self,
        token_mint: str,
        is_buy: bool,
        amount_sol: float,
        wallet: str,
        timestamp_ms: int,
        token_age_seconds: int,
    ):
        """Record a trade for analysis."""
        if token_mint not in self._token_trades:
            self._token_trades[token_mint] = []

        self._token_trades[token_mint].append({
            "is_buy": is_buy,
            "amount_sol": amount_sol,
            "wallet": wallet,
            "timestamp_ms": timestamp_ms,
            "age_seconds": token_age_seconds,
        })

    def analyze(self, token_mint: str, current_liquidity_sol: float) -> Dict[str, Any]:
        """Analyze buy/sell patterns for a token.

        Returns:
            Dictionary with analysis results including risk indicators.
        """
        trades = self._token_trades.get(token_mint, [])

        if not trades:
            return {
                "buy_sell_ratio_5min": 1.0,
                "buy_sell_ratio_15min": 1.0,
                "unique_buyers_5min": 0,
                "unique_sellers_5min": 0,
                "avg_buy_size_sol": 0.0,
                "avg_sell_size_sol": 0.0,
                "large_sell_count": 0,
                "risk_score": 0.3,
            }

        # Filter trades by time window
        trades_5min = [t for t in trades if t["age_seconds"] <= 300]
        trades_15min = [t for t in trades if t["age_seconds"] <= 900]

        # Calculate buy/sell volumes
        def calc_ratio(trade_list):
            buy_vol = sum(t["amount_sol"] for t in trade_list if t["is_buy"])
            sell_vol = sum(t["amount_sol"] for t in trade_list if not t["is_buy"])
            if sell_vol == 0:
                return 10.0  # Cap at 10 to avoid infinity
            return min(10.0, buy_vol / sell_vol)

        ratio_5min = calc_ratio(trades_5min)
        ratio_15min = calc_ratio(trades_15min)

        # Count unique participants
        buyers_5min = set(t["wallet"] for t in trades_5min if t["is_buy"])
        sellers_5min = set(t["wallet"] for t in trades_5min if not t["is_buy"])

        # Calculate average sizes
        buys = [t["amount_sol"] for t in trades if t["is_buy"]]
        sells = [t["amount_sol"] for t in trades if not t["is_buy"]]
        avg_buy = np.mean(buys) if buys else 0.0
        avg_sell = np.mean(sells) if sells else 0.0

        # Count large sells (> 10% of liquidity)
        large_threshold = current_liquidity_sol * 0.1
        large_sells = sum(1 for t in trades if not t["is_buy"] and t["amount_sol"] > large_threshold)

        # Calculate risk score based on patterns
        risk_score = 0.0

        # Very low buy/sell ratio is a red flag
        if ratio_5min < 0.3:
            risk_score += 0.3
        elif ratio_5min < 0.5:
            risk_score += 0.2
        elif ratio_5min < 0.8:
            risk_score += 0.1

        # Few buyers, many sellers = bad
        if len(sellers_5min) > len(buyers_5min) * 2 and len(sellers_5min) > 3:
            risk_score += 0.2

        # Large sells are concerning
        if large_sells > 3:
            risk_score += 0.2
        elif large_sells > 1:
            risk_score += 0.1

        # Sells much larger than buys on average
        if avg_sell > avg_buy * 3 and avg_sell > 0:
            risk_score += 0.15

        return {
            "buy_sell_ratio_5min": ratio_5min,
            "buy_sell_ratio_15min": ratio_15min,
            "unique_buyers_5min": len(buyers_5min),
            "unique_sellers_5min": len(sellers_5min),
            "avg_buy_size_sol": avg_buy,
            "avg_sell_size_sol": avg_sell,
            "large_sell_count": large_sells,
            "risk_score": min(1.0, risk_score),
        }


class LPMonitor:
    """Monitor LP (Liquidity Pool) for removal attempts.

    LP removal is the primary mechanism for rugs - detecting
    removal attempts before they complete is critical.
    """

    def __init__(self):
        self._token_lp_events: Dict[str, List[Dict]] = {}
        self._token_lp_locks: Dict[str, Dict] = {}

    def record_lp_lock(
        self,
        token_mint: str,
        lock_amount_pct: float,
        unlock_timestamp_ms: int,
    ):
        """Record that LP was locked."""
        self._token_lp_locks[token_mint] = {
            "lock_amount_pct": lock_amount_pct,
            "unlock_timestamp_ms": unlock_timestamp_ms,
            "recorded_at": int(np.datetime64('now', 'ms').astype(int)),
        }

    def record_lp_event(
        self,
        token_mint: str,
        event_type: str,  # "add", "remove", "remove_attempt"
        amount_pct: float,
        timestamp_ms: int,
    ):
        """Record an LP event."""
        if token_mint not in self._token_lp_events:
            self._token_lp_events[token_mint] = []

        self._token_lp_events[token_mint].append({
            "type": event_type,
            "amount_pct": amount_pct,
            "timestamp_ms": timestamp_ms,
        })

        if event_type in ["remove", "remove_attempt"]:
            logger.warning(
                f"LP removal {'attempt' if 'attempt' in event_type else 'detected'} "
                f"for {token_mint}: {amount_pct:.1f}%"
            )

    def analyze(self, token_mint: str) -> Dict[str, Any]:
        """Analyze LP status for a token.

        Returns:
            Dictionary with LP analysis including risk score.
        """
        events = self._token_lp_events.get(token_mint, [])
        lock_info = self._token_lp_locks.get(token_mint)

        removal_attempts = sum(1 for e in events if "remove" in e["type"])
        total_removed = sum(e["amount_pct"] for e in events if e["type"] == "remove")

        has_lock = lock_info is not None
        lock_duration_days = 0
        lp_unlock_timestamp = None

        if has_lock:
            lp_unlock_timestamp = lock_info["unlock_timestamp_ms"]
            now_ms = int(np.datetime64('now', 'ms').astype(int))
            remaining_ms = lp_unlock_timestamp - now_ms
            lock_duration_days = max(0, remaining_ms // (86400 * 1000))

        # Calculate risk score
        risk_score = 0.0

        # LP removal attempts are major red flags
        if removal_attempts > 0:
            risk_score += 0.4

        # Significant LP already removed
        if total_removed > 50:
            risk_score += 0.4
        elif total_removed > 20:
            risk_score += 0.25
        elif total_removed > 5:
            risk_score += 0.1

        # No lock or short lock
        if not has_lock:
            risk_score += 0.2
        elif lock_duration_days < 7:
            risk_score += 0.1
        elif lock_duration_days < 30:
            risk_score += 0.05

        return {
            "lp_removal_attempts": removal_attempts,
            "lp_removal_amount_pct": total_removed,
            "has_lp_lock": has_lock,
            "lp_lock_duration_days": lock_duration_days,
            "lp_unlock_timestamp": lp_unlock_timestamp,
            "risk_score": min(1.0, risk_score),
        }


@dataclass
class MutualInfoResult:
    """Result of mutual information feature selection."""
    feature_scores: Dict[str, float]  # MI score per feature
    selected_features: List[str]  # Features above threshold
    dropped_features: List[str]  # Features below threshold
    threshold: float
    total_mi: float  # Sum of MI for selected features
    feature_score_std: Optional[Dict[str, float]] = None  # Optional score std dev

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["Mutual Information Feature Selection:"]
        lines.append(f"  Threshold: {self.threshold:.4f}")
        lines.append(f"  Selected: {len(self.selected_features)}/{len(self.feature_scores)} features")
        lines.append(f"  Total MI: {self.total_mi:.4f}")
        if self.feature_score_std:
            avg_std = np.mean(list(self.feature_score_std.values()))
            lines.append(f"  Avg MI std: {avg_std:.4f}")
        lines.append("  Ranking:")
        for name, score in sorted(self.feature_scores.items(), key=lambda x: -x[1]):
            status = "+" if name in self.selected_features else "-"
            lines.append(f"    {status} {name}: {score:.4f}")
        return "\n".join(lines)


class SyntheticMethod(Enum):
    """Synthetic data generation methods."""
    NONE = "none"
    SMOTE = "smote"  # Standard SMOTE
    BORDERLINE_SMOTE = "borderline_smote"  # Focus on borderline cases
    ADASYN = "adasyn"  # Adaptive synthetic sampling
    SMOTE_TOMEK = "smote_tomek"  # SMOTE + Tomek links cleanup
    ADVERSARIAL = "adversarial"  # Custom adversarial generation
    FEATURE_CONDITIONAL = "feature_conditional"  # Realistic feature generation
    HYBRID = "hybrid"  # Combine SMOTE + adversarial


@dataclass
class SyntheticDataResult:
    """Result of synthetic data augmentation."""
    method: str
    original_samples: int
    original_class_ratio: float  # minority / majority
    synthetic_samples_added: int
    final_samples: int
    final_class_ratio: float
    adversarial_samples: int = 0  # Edge cases near decision boundary

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["Synthetic Data Augmentation:"]
        lines.append(f"  Method: {self.method}")
        lines.append(f"  Original: {self.original_samples} samples (ratio: {self.original_class_ratio:.2%})")
        lines.append(f"  Synthetic added: {self.synthetic_samples_added}")
        if self.adversarial_samples > 0:
            lines.append(f"  Adversarial samples: {self.adversarial_samples}")
        lines.append(f"  Final: {self.final_samples} samples (ratio: {self.final_class_ratio:.2%})")
        return "\n".join(lines)


# Feature bounds for realistic synthetic data generation
# Based on typical Solana meme token characteristics
FEATURE_BOUNDS = {
    "liquidity_usd": (100, 1_000_000),
    "fdv_usd": (1_000, 100_000_000),
    "holder_count": (10, 10_000),
    "top_holder_pct": (5, 95),
    "mint_authority_enabled": (0, 1),
    "freeze_authority_enabled": (0, 1),
    "lp_burn_pct": (0, 100),
    "age_seconds": (60, 86400 * 30),  # 1 min to 30 days
    "volume_24h": (0, 10_000_000),
    "price_change_1h": (-99, 1000),  # -99% to +1000%
}

# Feature correlations for rugs (used in conditional generation)
# Higher values = more likely to be a rug
RUG_FEATURE_PROFILES = {
    "high_risk": {
        "liquidity_usd": (100, 5000),
        "top_holder_pct": (50, 95),
        "mint_authority_enabled": 0.8,  # 80% chance enabled
        "freeze_authority_enabled": 0.6,
        "lp_burn_pct": (0, 30),
        "age_seconds": (60, 3600),
    },
    "medium_risk": {
        "liquidity_usd": (5000, 20000),
        "top_holder_pct": (30, 60),
        "mint_authority_enabled": 0.4,
        "freeze_authority_enabled": 0.3,
        "lp_burn_pct": (20, 60),
        "age_seconds": (1800, 14400),
    },
}


@dataclass
class TrainingResult:
    """Result of model training."""
    model_type: str
    accuracy: float
    cv_score: float
    feature_importance: Dict[str, float]
    training_time_secs: float
    automl_used: bool
    pipeline_str: Optional[str] = None
    mi_result: Optional[MutualInfoResult] = None  # MI feature selection results
    synthetic_result: Optional[SyntheticDataResult] = None  # Synthetic data augmentation results


class FraudModel:
    """Classifier for rug detection with AutoML support.

    Training order:
    1. TPOT (if available and USE_AUTOML=true)
    2. Auto-sklearn (if available, Linux only, and USE_AUTOML=true)
    3. RandomForest (always available fallback)
    """

    def __init__(
        self,
        use_automl: bool = True,
        automl_time_mins: int = 10,
        random_state: int = 42,
        use_mi_selection: bool = True,
        mi_threshold: float = 0.02,
        mi_min_features: int = 3,
        mi_bootstrap_runs: int = 5,
        mi_bootstrap_frac: float = 0.8,
        synthetic_method: Union[SyntheticMethod, str] = SyntheticMethod.SMOTE,
        synthetic_ratio: float = 1.0,
        adversarial_ratio: float = 0.1,
        # Advanced MI selector options
        use_advanced_mi: bool = True,
        mi_method: Optional[str] = "mrmr",  # "univariate_mi", "mrmr", "jmi", "cmim"
        mi_max_features: int = 25,
        mi_redundancy_threshold: float = 0.85,
        mi_per_regime: bool = True,
        # Temporal synthetic generation options
        use_temporal_synthetic: bool = True,
        temporal_augment_ratio: float = 0.2,  # Fraction of temporal synthetic to add
    ):
        """Initialize the fraud model.

        Args:
            use_automl: Whether to attempt AutoML training
            automl_time_mins: Maximum time for AutoML search
            random_state: Random seed for reproducibility
            use_mi_selection: Whether to use mutual information feature selection
            mi_threshold: Minimum MI score for feature inclusion (default 0.02)
            mi_min_features: Minimum features to keep regardless of threshold
            mi_bootstrap_runs: Number of bootstrap runs to stabilize MI scores
            mi_bootstrap_frac: Fraction of each class to sample per bootstrap
            synthetic_method: Method for synthetic data generation (SMOTE, ADASYN, etc.)
            synthetic_ratio: Target ratio of minority/majority after augmentation (1.0 = balanced)
            adversarial_ratio: Fraction of synthetic samples to generate as adversarial edge cases
            use_advanced_mi: Whether to use advanced MI selector (mRMR/JMI/CMIM)
            mi_method: Selection method - "univariate_mi", "mrmr", "jmi", "cmim"
            mi_max_features: Maximum features to select
            mi_redundancy_threshold: Max feature correlation before filtering
            mi_per_regime: Whether to compute regime-specific feature selection
            use_temporal_synthetic: Whether to use temporal lifecycle synthetic data
            temporal_augment_ratio: Fraction of temporal synthetic samples to add
        """
        self.use_automl = use_automl and (TPOT_AVAILABLE or AUTOSKLEARN_AVAILABLE)
        self.automl_time_mins = automl_time_mins
        self.random_state = random_state

        # Mutual information feature selection config
        self.use_mi_selection = use_mi_selection
        self.mi_threshold = mi_threshold
        self.mi_min_features = mi_min_features
        self.mi_bootstrap_runs = mi_bootstrap_runs
        self.mi_bootstrap_frac = mi_bootstrap_frac

        # Advanced MI selector config
        self.use_advanced_mi = use_advanced_mi and ADVANCED_MI_AVAILABLE
        self.mi_method = mi_method
        self.mi_max_features = mi_max_features
        self.mi_redundancy_threshold = mi_redundancy_threshold
        self.mi_per_regime = mi_per_regime
        self.advanced_mi_selector: Optional[MutualInformationSelector] = None
        self.advanced_mi_result: Optional[SelectionResult] = None

        if self.use_advanced_mi:
            mi_config = MIConfig(
                min_features=mi_min_features,
                max_features=mi_max_features,
                mi_threshold=mi_threshold,
                redundancy_threshold=mi_redundancy_threshold,
                per_regime_selection=mi_per_regime,
                random_state=random_state,
            )
            self.advanced_mi_selector = MutualInformationSelector(mi_config)
            logger.info(f"Advanced MI selector enabled: method={mi_method}")

        # Synthetic data augmentation config
        if isinstance(synthetic_method, str):
            try:
                self.synthetic_method = SyntheticMethod(synthetic_method)
            except ValueError:
                self.synthetic_method = SyntheticMethod.SMOTE
        else:
            self.synthetic_method = synthetic_method
        self.synthetic_ratio = synthetic_ratio
        self.adversarial_ratio = adversarial_ratio
        self.synthetic_result: Optional[SyntheticDataResult] = None

        # Temporal synthetic generation config
        self.use_temporal_synthetic = use_temporal_synthetic and TEMPORAL_SYNTHETIC_AVAILABLE
        self.temporal_augment_ratio = temporal_augment_ratio
        self.temporal_generator: Optional[TemporalSyntheticGenerator] = None

        self.model: Any = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_type = "none"

        # Feature selection state
        self.feature_mask: Optional[np.ndarray] = None  # Boolean mask for selected features
        self.selected_feature_names: Optional[List[str]] = None
        self.mi_scores: Optional[Dict[str, float]] = None
        self.mi_score_std: Optional[Dict[str, float]] = None
        self.mi_result: Optional[MutualInfoResult] = None

        self.feature_names = [
            "liquidity_usd",
            "fdv_usd",
            "holder_count",
            "top_holder_pct",
            "mint_authority_enabled",
            "freeze_authority_enabled",
            "lp_burn_pct",
            "age_seconds",
            "volume_24h",
            "price_change_1h",
        ]

        # Load config from environment
        if os.environ.get("USE_AUTOML", "").lower() == "false":
            self.use_automl = False
        if time_mins := os.environ.get("AUTOML_TIME_MINS"):
            try:
                self.automl_time_mins = int(time_mins)
            except ValueError:
                pass
        if os.environ.get("USE_MI_SELECTION", "").lower() == "false":
            self.use_mi_selection = False
        if mi_thresh := os.environ.get("MI_THRESHOLD"):
            try:
                self.mi_threshold = float(mi_thresh)
            except ValueError:
                pass
        if mi_runs := os.environ.get("MI_BOOTSTRAP_RUNS"):
            try:
                self.mi_bootstrap_runs = int(mi_runs)
            except ValueError:
                pass
        if mi_frac := os.environ.get("MI_BOOTSTRAP_FRAC"):
            try:
                self.mi_bootstrap_frac = float(mi_frac)
            except ValueError:
                pass
        if synth_method := os.environ.get("SYNTHETIC_METHOD"):
            try:
                self.synthetic_method = SyntheticMethod(synth_method.lower())
            except ValueError:
                pass
        if synth_ratio := os.environ.get("SYNTHETIC_RATIO"):
            try:
                self.synthetic_ratio = float(synth_ratio)
            except ValueError:
                pass

    def _extract_features(self, token: TokenFeatures) -> np.ndarray:
        """Convert token features to numpy array."""
        return np.array([
            token.liquidity_usd,
            token.fdv_usd,
            token.holder_count,
            token.top_holder_pct,
            int(token.mint_authority_enabled),
            int(token.freeze_authority_enabled),
            token.lp_burn_pct,
            token.age_seconds,
            token.volume_24h,
            token.price_change_1h,
        ])

    def _compute_mi_scores(
        self, X: np.ndarray, y: np.ndarray, n_neighbors: int
    ) -> np.ndarray:
        """Compute MI scores for a single sample of data."""
        return mutual_info_classif(
            X, y,
            discrete_features=[4, 5],  # mint_authority, freeze_authority are binary
            random_state=self.random_state,
            n_neighbors=n_neighbors,
        )

    def _stratified_bootstrap(
        self, X: np.ndarray, y: np.ndarray, frac: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bootstrap sample while preserving class ratios."""
        frac = min(max(frac, 0.1), 1.0)
        idx_rug = np.where(y == 1)[0]
        idx_safe = np.where(y == 0)[0]

        n_rug = max(2, int(len(idx_rug) * frac))
        n_safe = max(2, int(len(idx_safe) * frac))

        rug_sample = np.random.choice(idx_rug, size=n_rug, replace=True)
        safe_sample = np.random.choice(idx_safe, size=n_safe, replace=True)
        sample_idx = np.concatenate([rug_sample, safe_sample])
        np.random.shuffle(sample_idx)
        return X[sample_idx], y[sample_idx]

    def _compute_mutual_information(
        self, X: np.ndarray, y: np.ndarray
    ) -> MutualInfoResult:
        """Compute mutual information scores for feature selection.

        MI measures the mutual dependence between each feature and the target.
        Unlike correlation, MI captures nonlinear relationships.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (0=safe, 1=rug)

        Returns:
            MutualInfoResult with scores and selected features
        """
        logger.info("Computing mutual information scores for feature selection")

        n_samples = len(y)
        minority_count = min(np.sum(y == 1), np.sum(y == 0))
        if n_samples < 10 or minority_count < 2:
            logger.warning(
                "Insufficient samples for stable MI; keeping all features"
            )
            mi_scores = np.zeros(len(self.feature_names), dtype=float)
            mi_std = None
        else:
            base_neighbors = max(1, min(10, int(np.sqrt(n_samples))))
            n_neighbors = min(base_neighbors, max(1, minority_count - 1))
            runs = max(1, self.mi_bootstrap_runs)

            if runs == 1:
                mi_scores = self._compute_mi_scores(X, y, n_neighbors=n_neighbors)
                mi_std = None
            else:
                score_runs = []
                np.random.seed(self.random_state)
                for _ in range(runs):
                    X_sample, y_sample = self._stratified_bootstrap(
                        X, y, self.mi_bootstrap_frac
                    )
                    sample_minority = min(np.sum(y_sample == 1), np.sum(y_sample == 0))
                    sample_neighbors = min(n_neighbors, max(1, sample_minority - 1))
                    score_runs.append(
                        self._compute_mi_scores(X_sample, y_sample, n_neighbors=sample_neighbors)
                    )
                mi_scores = np.mean(score_runs, axis=0)
                mi_std = np.std(score_runs, axis=0)

        # Create score dictionary
        self.mi_scores = dict(zip(self.feature_names, mi_scores))
        self.mi_score_std = dict(zip(self.feature_names, mi_std)) if mi_std is not None else None

        # Determine which features to select
        # Sort by MI score descending
        ranked = sorted(
            zip(self.feature_names, mi_scores, range(len(self.feature_names))),
            key=lambda x: -x[1]
        )

        # Select features above threshold, but keep at least mi_min_features
        selected_indices = []
        selected_names = []
        dropped_names = []

        for name, score, idx in ranked:
            if score >= self.mi_threshold or len(selected_indices) < self.mi_min_features:
                selected_indices.append(idx)
                selected_names.append(name)
            else:
                dropped_names.append(name)

        # Create boolean mask
        self.feature_mask = np.zeros(len(self.feature_names), dtype=bool)
        self.feature_mask[selected_indices] = True
        self.selected_feature_names = selected_names

        # Calculate total MI of selected features
        total_mi = float(sum(mi_scores[i] for i in selected_indices))

        # Create result object
        self.mi_result = MutualInfoResult(
            feature_scores=self.mi_scores.copy(),
            feature_score_std=self.mi_score_std.copy() if self.mi_score_std else None,
            selected_features=selected_names,
            dropped_features=dropped_names,
            threshold=self.mi_threshold,
            total_mi=total_mi,
        )

        # Log results
        logger.info(self.mi_result.summary())

        return self.mi_result

    def _compute_advanced_mi(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime: Optional[str] = None,
    ) -> MutualInfoResult:
        """Compute mutual information using advanced mRMR/JMI/CMIM methods.

        Uses the MutualInformationSelector for more sophisticated feature selection
        that accounts for feature redundancy and joint information.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (0=safe, 1=rug)
            regime: Optional market regime for regime-specific selection

        Returns:
            MutualInfoResult with scores and selected features
        """
        if not self.use_advanced_mi or self.advanced_mi_selector is None:
            return self._compute_mutual_information(X, y)

        logger.info(f"Computing advanced MI scores using {self.mi_method} method")

        # Map string method to enum
        method_map = {
            "univariate_mi": SelectionMethod.UNIVARIATE_MI,
            "mrmr": SelectionMethod.MRMR,
            "jmi": SelectionMethod.JMI,
            "cmim": SelectionMethod.CMIM,
        }
        method = method_map.get(self.mi_method, SelectionMethod.MRMR)

        try:
            # Run advanced MI selection
            result = self.advanced_mi_selector.select_features(
                X=X,
                y=y,
                feature_names=self.feature_names,
                method=method,
                regime=regime,
                is_classification=True,
            )

            # Store the advanced result
            self.advanced_mi_result = result

            # Update feature mask and selected names
            self.feature_mask = np.zeros(len(self.feature_names), dtype=bool)
            self.feature_mask[result.selected_indices] = True
            self.selected_feature_names = result.selected_features

            # Build MI scores dict
            self.mi_scores = {}
            for score in result.feature_scores:
                self.mi_scores[score.feature_name] = score.mi_score

            # Create MutualInfoResult for compatibility
            self.mi_result = MutualInfoResult(
                feature_scores=self.mi_scores.copy(),
                feature_score_std=None,  # Advanced method doesn't do bootstrap
                selected_features=result.selected_features,
                dropped_features=result.redundancy_filtered + result.low_mi_filtered,
                threshold=self.mi_threshold,
                total_mi=result.total_mi,
            )

            logger.info(
                f"Advanced MI selection ({method.value}): "
                f"{len(self.feature_names)} -> {len(result.selected_features)} features"
            )
            logger.info(f"  Selected: {result.selected_features}")
            if result.redundancy_filtered:
                logger.info(f"  Redundancy filtered: {result.redundancy_filtered}")

            return self.mi_result

        except Exception as e:
            logger.warning(f"Advanced MI selection failed: {e}, falling back to basic MI")
            return self._compute_mutual_information(X, y)

    def _apply_feature_selection(self, X: np.ndarray) -> np.ndarray:
        """Apply feature mask to reduce dimensionality.

        Args:
            X: Full feature matrix

        Returns:
            Reduced feature matrix with only selected features
        """
        if self.feature_mask is None or not self.use_mi_selection:
            return X
        return X[:, self.feature_mask]

    def _augment_with_synthetic_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, SyntheticDataResult]:
        """Augment training data with synthetic samples.

        Handles class imbalance by generating synthetic minority samples
        using various techniques (SMOTE, ADASYN, adversarial, etc.)

        Args:
            X: Feature matrix (scaled)
            y: Target labels (0=safe, 1=rug)

        Returns:
            Tuple of (augmented_X, augmented_y, SyntheticDataResult)
        """
        original_samples = len(y)
        minority_count = np.sum(y == 1)
        majority_count = np.sum(y == 0)
        original_ratio = minority_count / majority_count if majority_count > 0 else 0

        logger.info(
            f"Class distribution: {minority_count} rugs, {majority_count} safe "
            f"(ratio: {original_ratio:.2%})"
        )

        # Skip augmentation if already balanced or no method specified
        if self.synthetic_method == SyntheticMethod.NONE or original_ratio >= 0.8:
            return X, y, SyntheticDataResult(
                method="none",
                original_samples=original_samples,
                original_class_ratio=original_ratio,
                synthetic_samples_added=0,
                final_samples=original_samples,
                final_class_ratio=original_ratio,
            )

        X_augmented, y_augmented = X.copy(), y.copy()
        adversarial_count = 0

        # Apply SMOTE-based methods if available
        if self.synthetic_method in [
            SyntheticMethod.SMOTE,
            SyntheticMethod.BORDERLINE_SMOTE,
            SyntheticMethod.ADASYN,
            SyntheticMethod.SMOTE_TOMEK,
            SyntheticMethod.HYBRID,
        ]:
            X_augmented, y_augmented = self._apply_smote(X, y)

        # Apply adversarial generation for ADVERSARIAL or HYBRID methods
        if self.synthetic_method in [SyntheticMethod.ADVERSARIAL, SyntheticMethod.HYBRID]:
            X_adv, y_adv = self._generate_adversarial_samples(X, y)
            if len(X_adv) > 0:
                X_augmented = np.vstack([X_augmented, X_adv])
                y_augmented = np.concatenate([y_augmented, y_adv])
                adversarial_count = len(X_adv)

        # Apply feature-conditional generation
        if self.synthetic_method == SyntheticMethod.FEATURE_CONDITIONAL:
            X_cond, y_cond = self._generate_conditional_samples(X, y)
            if len(X_cond) > 0:
                X_augmented = np.vstack([X_augmented, X_cond])
                y_augmented = np.concatenate([y_augmented, y_cond])

        # Calculate final statistics
        final_samples = len(y_augmented)
        final_minority = np.sum(y_augmented == 1)
        final_majority = np.sum(y_augmented == 0)
        final_ratio = final_minority / final_majority if final_majority > 0 else 0

        result = SyntheticDataResult(
            method=self.synthetic_method.value,
            original_samples=original_samples,
            original_class_ratio=original_ratio,
            synthetic_samples_added=final_samples - original_samples,
            final_samples=final_samples,
            final_class_ratio=final_ratio,
            adversarial_samples=adversarial_count,
        )

        logger.info(result.summary())
        self.synthetic_result = result

        return X_augmented, y_augmented, result

    def _apply_smote(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE-based oversampling.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        if not SMOTE_AVAILABLE:
            logger.warning("imbalanced-learn not installed, skipping SMOTE")
            return X, y

        try:
            # Select SMOTE variant
            if self.synthetic_method == SyntheticMethod.BORDERLINE_SMOTE:
                sampler = BorderlineSMOTE(
                    sampling_strategy=self.synthetic_ratio,
                    random_state=self.random_state,
                    k_neighbors=min(5, np.sum(y == 1) - 1),
                )
            elif self.synthetic_method == SyntheticMethod.ADASYN:
                sampler = ADASYN(
                    sampling_strategy=self.synthetic_ratio,
                    random_state=self.random_state,
                    n_neighbors=min(5, np.sum(y == 1) - 1),
                )
            elif self.synthetic_method == SyntheticMethod.SMOTE_TOMEK:
                sampler = SMOTETomek(
                    sampling_strategy=self.synthetic_ratio,
                    random_state=self.random_state,
                )
            else:  # Default SMOTE or HYBRID
                sampler = SMOTE(
                    sampling_strategy=self.synthetic_ratio,
                    random_state=self.random_state,
                    k_neighbors=min(5, np.sum(y == 1) - 1),
                )

            X_resampled, y_resampled = sampler.fit_resample(X, y)
            logger.info(
                f"SMOTE: {len(y)} -> {len(y_resampled)} samples "
                f"({self.synthetic_method.value})"
            )
            return X_resampled, y_resampled

        except Exception as e:
            logger.warning(f"SMOTE failed: {e}, using original data")
            return X, y

    def _generate_adversarial_samples(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate adversarial samples near the decision boundary.

        These are edge cases that challenge the model - tokens that look
        safe but are rugs, or vice versa. This hardens the model against
        sophisticated attacks.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Tuple of (adversarial_X, adversarial_y)
        """
        np.random.seed(self.random_state)

        # Calculate how many adversarial samples to generate
        minority_count = np.sum(y == 1)
        n_adversarial = max(1, int(minority_count * self.adversarial_ratio))

        # Get minority (rug) and majority (safe) samples
        X_rug = X[y == 1]
        X_safe = X[y == 0]

        if len(X_rug) < 2 or len(X_safe) < 2:
            return np.array([]).reshape(0, X.shape[1]), np.array([])

        adversarial_samples = []
        adversarial_labels = []

        for _ in range(n_adversarial):
            # Strategy 1: Interpolate between rug and safe (boundary samples)
            if np.random.random() < 0.5:
                rug_idx = np.random.randint(len(X_rug))
                safe_idx = np.random.randint(len(X_safe))

                # Create sample closer to boundary (70-90% toward the other class)
                alpha = np.random.uniform(0.7, 0.9)

                # "Deceptive rug" - looks safe but is a rug
                if np.random.random() < 0.5:
                    sample = X_rug[rug_idx] * (1 - alpha) + X_safe[safe_idx] * alpha
                    label = 1  # Still a rug
                else:
                    # "Suspicious safe" - looks risky but is safe
                    sample = X_safe[safe_idx] * (1 - alpha) + X_rug[rug_idx] * alpha
                    label = 0  # Still safe

            # Strategy 2: Perturb existing minority samples
            else:
                rug_idx = np.random.randint(len(X_rug))
                sample = X_rug[rug_idx].copy()

                # Perturb 2-4 features to make it look more legitimate
                n_perturb = np.random.randint(2, 5)
                perturb_indices = np.random.choice(X.shape[1], n_perturb, replace=False)

                for idx in perturb_indices:
                    # Move feature value toward safe distribution
                    safe_mean = X_safe[:, idx].mean()
                    safe_std = X_safe[:, idx].std()
                    sample[idx] = np.random.normal(safe_mean, safe_std * 0.5)

                label = 1  # Still a rug (deceptive)

            adversarial_samples.append(sample)
            adversarial_labels.append(label)

        logger.info(f"Generated {len(adversarial_samples)} adversarial samples")

        return np.array(adversarial_samples), np.array(adversarial_labels)

    def _generate_conditional_samples(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic samples with realistic feature correlations.

        Uses domain knowledge about rug characteristics to generate
        plausible synthetic rugs with correlated features.

        Args:
            X: Feature matrix (may be reduced by MI selection)
            y: Target labels

        Returns:
            Tuple of (conditional_X, conditional_y) matching X's dimensionality
        """
        np.random.seed(self.random_state)

        minority_count = np.sum(y == 1)
        majority_count = np.sum(y == 0)
        target_count = int(majority_count * self.synthetic_ratio) - minority_count

        if target_count <= 0:
            return np.array([]).reshape(0, X.shape[1]), np.array([])

        synthetic_samples = []

        for i in range(target_count):
            # Alternate between high-risk and medium-risk profiles
            profile = RUG_FEATURE_PROFILES["high_risk" if i % 2 == 0 else "medium_risk"]

            # Generate full feature vector (all 10 features)
            full_sample = np.zeros(len(self.feature_names))

            for j, feature_name in enumerate(self.feature_names):
                if feature_name in profile:
                    constraint = profile[feature_name]
                    if isinstance(constraint, tuple):
                        # Range constraint
                        full_sample[j] = np.random.uniform(constraint[0], constraint[1])
                    else:
                        # Probability constraint (for binary features)
                        full_sample[j] = 1 if np.random.random() < constraint else 0
                else:
                    # Use bounds with slight bias toward risky values
                    bounds = FEATURE_BOUNDS[feature_name]
                    if feature_name in ["fdv_usd", "volume_24h"]:
                        # Log-uniform for wide-range features
                        log_min = np.log(max(bounds[0], 1))
                        log_max = np.log(bounds[1])
                        full_sample[j] = np.exp(np.random.uniform(log_min, log_max))
                    else:
                        full_sample[j] = np.random.uniform(bounds[0], bounds[1])

            synthetic_samples.append(full_sample)

        if not synthetic_samples:
            return np.array([]).reshape(0, X.shape[1]), np.array([])

        # Scale synthetic samples using the fitted scaler
        synthetic_full = np.array(synthetic_samples)
        synthetic_scaled = self.scaler.transform(synthetic_full)

        # Apply feature selection mask if active
        if self.feature_mask is not None and self.use_mi_selection:
            synthetic_X = synthetic_scaled[:, self.feature_mask]
        else:
            synthetic_X = synthetic_scaled

        synthetic_y = np.ones(len(synthetic_samples), dtype=int)  # All rugs

        logger.info(f"Generated {len(synthetic_samples)} feature-conditional samples")

        return synthetic_X, synthetic_y

    def _augment_with_temporal_synthetic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment with temporal lifecycle sequences.

        Uses TemporalSyntheticGenerator to create realistic token lifecycle
        sequences (honeypot evolution, fast rug, etc.) and extracts the final
        timestep for training non-LSTM models.

        Args:
            X: Feature matrix (already scaled and selected)
            y: Target labels
            regime: Optional market regime for conditioning

        Returns:
            Tuple of (augmented_X, augmented_y)
        """
        if not TEMPORAL_SYNTHETIC_AVAILABLE:
            logger.warning("Temporal synthetic not available, skipping")
            return X, y

        # Initialize generator if needed
        if self.temporal_generator is None:
            # Use unscaled features for reference distribution
            # We'll scale the synthetic samples after generation
            self.temporal_generator = TemporalSyntheticGenerator(
                base_features=X,  # Use current data as reference
                n_timesteps=60,
                timestep_minutes=5,
                random_state=self.random_state,
            )

        n_synthetic = int(len(y) * self.temporal_augment_ratio)
        if n_synthetic < 1:
            return X, y

        try:
            # Generate honeypot evolution sequences (most valuable)
            honeypot_seqs, honeypot_y = self.temporal_generator.generate_batch(
                TokenLifecycleType.HONEYPOT_EVOLUTION,
                n_samples=max(1, n_synthetic // 3),
                regime=regime,
            )

            # Generate fast rug sequences
            rug_seqs, rug_y = self.temporal_generator.generate_batch(
                TokenLifecycleType.FAST_RUG,
                n_samples=max(1, n_synthetic // 3),
                regime=regime,
            )

            # Generate pump and dump sequences
            pnd_seqs, pnd_y = self.temporal_generator.generate_batch(
                TokenLifecycleType.PUMP_AND_DUMP,
                n_samples=max(1, n_synthetic // 3),
                regime=regime,
            )

            # Flatten sequences (take last timestep for non-LSTM models)
            honeypot_flat = honeypot_seqs[:, -1, :]
            rug_flat = rug_seqs[:, -1, :]
            pnd_flat = pnd_seqs[:, -1, :]

            # Ensure dimensions match
            if honeypot_flat.shape[1] != X.shape[1]:
                # Adjust feature count if needed
                min_features = min(honeypot_flat.shape[1], X.shape[1])
                honeypot_flat = honeypot_flat[:, :min_features]
                rug_flat = rug_flat[:, :min_features]
                pnd_flat = pnd_flat[:, :min_features]
                if X.shape[1] > min_features:
                    X = X[:, :min_features]

            # Combine all synthetic samples
            synthetic_X = np.vstack([honeypot_flat, rug_flat, pnd_flat])
            synthetic_y = np.concatenate([honeypot_y, rug_y, pnd_y])

            # Combine with original
            X_aug = np.vstack([X, synthetic_X])
            y_aug = np.concatenate([y, synthetic_y])

            logger.info(
                f"Temporal synthetic: {len(y)} -> {len(y_aug)} samples "
                f"(+{len(honeypot_y)} honeypot, +{len(rug_y)} rug, +{len(pnd_y)} pnd, "
                f"regime={regime})"
            )

            return X_aug, y_aug

        except Exception as e:
            logger.warning(f"Temporal synthetic augmentation failed: {e}")
            return X, y

    def train(
        self,
        tokens: List[TokenFeatures],
        labels: List[int],
        regime: Optional[str] = None,
    ) -> TrainingResult:
        """
        Train the model using AutoML with fallback chain.

        Pipeline:
        1. Extract features from tokens
        2. Scale features
        3. Compute mutual information and select informative features
        4. Augment with synthetic data for class balancing
        5. Train model on augmented, selected features

        Args:
            tokens: List of token features
            labels: 1 = rug, 0 = safe
            regime: Optional market regime for regime-specific MI selection

        Returns:
            TrainingResult with model info and metrics
        """
        import time

        start_time = time.time()

        X = np.array([self._extract_features(t) for t in tokens])
        y = np.array(labels)

        # Scale features (before MI computation for consistency)
        X_scaled = self.scaler.fit_transform(X)

        # Compute mutual information for feature selection
        mi_result = None
        if self.use_mi_selection:
            # Use advanced MI selector if available
            if self.use_advanced_mi and self.advanced_mi_selector is not None:
                mi_result = self._compute_advanced_mi(X_scaled, y, regime)
            else:
                mi_result = self._compute_mutual_information(X_scaled, y)
            X_selected = self._apply_feature_selection(X_scaled)
            logger.info(
                f"Feature selection: {X_scaled.shape[1]} -> {X_selected.shape[1]} features"
            )
        else:
            X_selected = X_scaled
            # Use all features if MI selection disabled
            self.feature_mask = np.ones(len(self.feature_names), dtype=bool)
            self.selected_feature_names = self.feature_names.copy()

        # Augment with synthetic data for class balancing
        synthetic_result = None
        if self.synthetic_method != SyntheticMethod.NONE:
            X_augmented, y_augmented, synthetic_result = self._augment_with_synthetic_data(
                X_selected, y
            )
        else:
            X_augmented, y_augmented = X_selected, y

        # Augment with temporal synthetic data (lifecycle sequences)
        if self.use_temporal_synthetic and self.temporal_augment_ratio > 0:
            X_augmented, y_augmented = self._augment_with_temporal_synthetic(
                X_augmented, y_augmented, regime
            )

        # Try AutoML methods in order
        result = None

        if self.use_automl:
            # 1. Try TPOT
            result = self._try_tpot(X_augmented, y_augmented)
            if result:
                self.is_trained = True
                result.training_time_secs = time.time() - start_time
                result.mi_result = mi_result
                result.synthetic_result = synthetic_result
                return result

            # 2. Try Auto-sklearn (Linux only)
            result = self._try_autosklearn(X_augmented, y_augmented)
            if result:
                self.is_trained = True
                result.training_time_secs = time.time() - start_time
                result.mi_result = mi_result
                result.synthetic_result = synthetic_result
                return result

        # 3. Fallback to RandomForest (always works)
        result = self._train_random_forest(X_augmented, y_augmented)
        self.is_trained = True
        result.training_time_secs = time.time() - start_time
        result.mi_result = mi_result
        result.synthetic_result = synthetic_result
        return result

    def _try_tpot(self, X: np.ndarray, y: np.ndarray) -> Optional[TrainingResult]:
        """Try training with TPOT AutoML."""
        if not TPOT_AVAILABLE:
            return None

        logger.info(f"Attempting TPOT AutoML training (max {self.automl_time_mins} mins)")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                tpot = TPOTClassifier(
                    generations=5,
                    population_size=20,
                    cv=5,
                    random_state=self.random_state,
                    verbosity=1,
                    max_time_mins=self.automl_time_mins,
                    n_jobs=-1,
                )

                tpot.fit(X, y)

            self.model = tpot.fitted_pipeline_
            self.model_type = "tpot"

            # Calculate metrics
            cv_scores = cross_val_score(self.model, X, y, cv=5)

            logger.info(f"TPOT training successful, CV score: {cv_scores.mean():.3f}")

            return TrainingResult(
                model_type="tpot",
                accuracy=tpot.score(X, y),
                cv_score=cv_scores.mean(),
                feature_importance=self._get_feature_importance(),
                training_time_secs=0,  # Will be filled in by caller
                automl_used=True,
                pipeline_str=str(tpot.fitted_pipeline_),
            )

        except Exception as e:
            logger.warning(f"TPOT training failed: {e}")
            return None

    def _try_autosklearn(self, X: np.ndarray, y: np.ndarray) -> Optional[TrainingResult]:
        """Try training with Auto-sklearn (Linux only)."""
        if not AUTOSKLEARN_AVAILABLE:
            return None

        logger.info(f"Attempting Auto-sklearn training (max {self.automl_time_mins} mins)")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                automl = AutoSklearnClassifier(
                    time_left_for_this_task=self.automl_time_mins * 60,
                    per_run_time_limit=60,
                    n_jobs=-1,
                    memory_limit=4096,
                    seed=self.random_state,
                )

                automl.fit(X, y)

            self.model = automl
            self.model_type = "autosklearn"

            # Calculate metrics
            cv_scores = cross_val_score(self.model, X, y, cv=5)

            logger.info(f"Auto-sklearn training successful, CV score: {cv_scores.mean():.3f}")

            return TrainingResult(
                model_type="autosklearn",
                accuracy=automl.score(X, y),
                cv_score=cv_scores.mean(),
                feature_importance=self._get_feature_importance(),
                training_time_secs=0,
                automl_used=True,
                pipeline_str=str(automl.show_models()),
            )

        except Exception as e:
            logger.warning(f"Auto-sklearn training failed: {e}")
            return None

    def _train_random_forest(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """Train with XGBoost GPU or RandomForest CPU fallback."""
        # Try XGBoost GPU first (5-10x faster)
        if XGBOOST_GPU_AVAILABLE:
            logger.info("Training with XGBoost GPU classifier")
            try:
                self.model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    tree_method="gpu_hist",
                    device="cuda",
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                )
                self.model.fit(X, y)
                self.model_type = "xgboost_gpu"

                cv_scores = cross_val_score(self.model, X, y, cv=5)
                logger.info(f"XGBoost GPU training successful, CV score: {cv_scores.mean():.3f}")

                return TrainingResult(
                    model_type="xgboost_gpu",
                    accuracy=self.model.score(X, y),
                    cv_score=cv_scores.mean(),
                    feature_importance=self._get_feature_importance(),
                    training_time_secs=0,
                    automl_used=False,
                )
            except Exception as e:
                logger.warning(f"XGBoost GPU failed, falling back to CPU: {e}")

        # Fallback to RandomForest CPU
        logger.info("Training with RandomForest classifier (CPU)")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.model.fit(X, y)
        self.model_type = "random_forest"

        # Calculate metrics
        cv_scores = cross_val_score(self.model, X, y, cv=5)

        logger.info(f"RandomForest training successful, CV score: {cv_scores.mean():.3f}")

        return TrainingResult(
            model_type="random_forest",
            accuracy=self.model.score(X, y),
            cv_score=cv_scores.mean(),
            feature_importance=self._get_feature_importance(),
            training_time_secs=0,
            automl_used=False,
        )

    def predict_proba(self, token: TokenFeatures) -> float:
        """Predict probability of being a rug."""
        if not self.is_trained:
            # Default heuristic when model not trained
            return self._heuristic_score(token)

        X = self._extract_features(token).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        X_selected = self._apply_feature_selection(X_scaled)
        return self.model.predict_proba(X_selected)[0][1]

    def predict_batch(self, tokens: List[TokenFeatures]) -> np.ndarray:
        """Predict probabilities for multiple tokens."""
        if not self.is_trained:
            return np.array([self._heuristic_score(t) for t in tokens])

        X = np.array([self._extract_features(t) for t in tokens])
        X_scaled = self.scaler.transform(X)
        X_selected = self._apply_feature_selection(X_scaled)
        return self.model.predict_proba(X_selected)[:, 1]

    def _heuristic_score(self, token: TokenFeatures) -> float:
        """Fallback heuristic-based score."""
        score = 0.0

        # Authority checks (major red flags)
        if token.mint_authority_enabled:
            score += 0.3
        if token.freeze_authority_enabled:
            score += 0.2

        # Liquidity check
        if token.liquidity_usd < 5000:
            score += 0.1
        elif token.liquidity_usd < 10000:
            score += 0.05

        # Holder concentration
        if token.top_holder_pct > 50:
            score += 0.15
        elif token.top_holder_pct > 30:
            score += 0.08

        # LP burn
        if token.lp_burn_pct < 50:
            score += 0.1
        elif token.lp_burn_pct < 80:
            score += 0.05

        # Age (newer = riskier)
        if token.age_seconds < 300:  # < 5 min
            score += 0.1
        elif token.age_seconds < 3600:  # < 1 hour
            score += 0.05

        return min(1.0, score)

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns importance for selected features only (after MI filtering).
        If MI selection was used, also includes MI scores in the result.
        """
        if not self.is_trained or self.model is None:
            return {}

        # Use selected feature names if available
        feature_names = self.selected_feature_names or self.feature_names

        # Try to get feature importance from different model types
        try:
            if hasattr(self.model, "feature_importances_"):
                importance = self.model.feature_importances_
                return dict(zip(feature_names, importance))
            elif hasattr(self.model, "coef_"):
                importance = np.abs(self.model.coef_[0])
                return dict(zip(feature_names, importance))
        except Exception:
            pass

        return {}

    def get_mi_scores(self) -> Dict[str, float]:
        """Get mutual information scores for all features.

        Returns:
            Dictionary mapping feature names to MI scores, or empty if
            MI selection wasn't performed.
        """
        return self.mi_scores.copy() if self.mi_scores else {}

    def get_combined_importance(self) -> Dict[str, Dict[str, float]]:
        """Get combined view of MI scores and model feature importance.

        Returns:
            Dictionary with 'mi_score', 'model_importance', and 'selected' for each feature.
        """
        result = {}
        model_importance = self._get_feature_importance()

        for name in self.feature_names:
            result[name] = {
                "mi_score": self.mi_scores.get(name, 0.0) if self.mi_scores else 0.0,
                "model_importance": model_importance.get(name, 0.0),
                "selected": name in (self.selected_feature_names or self.feature_names),
            }

        return result

    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (public API)."""
        return self._get_feature_importance()

    def save(self, path: str):
        """Save model to file, including MI and synthetic data state."""
        # Serialize MI result if present
        mi_result_dict = None
        if self.mi_result:
            mi_result_dict = {
                "feature_scores": self.mi_result.feature_scores,
                "feature_score_std": self.mi_result.feature_score_std,
                "selected_features": self.mi_result.selected_features,
                "dropped_features": self.mi_result.dropped_features,
                "threshold": self.mi_result.threshold,
                "total_mi": self.mi_result.total_mi,
            }

        # Serialize synthetic result if present
        synthetic_result_dict = None
        if self.synthetic_result:
            synthetic_result_dict = {
                "method": self.synthetic_result.method,
                "original_samples": self.synthetic_result.original_samples,
                "original_class_ratio": self.synthetic_result.original_class_ratio,
                "synthetic_samples_added": self.synthetic_result.synthetic_samples_added,
                "final_samples": self.synthetic_result.final_samples,
                "final_class_ratio": self.synthetic_result.final_class_ratio,
                "adversarial_samples": self.synthetic_result.adversarial_samples,
            }

        # Serialize advanced MI result if present
        advanced_mi_dict = None
        if self.advanced_mi_result:
            advanced_mi_dict = self.advanced_mi_result.to_dict()

        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "is_trained": self.is_trained,
                "model_type": self.model_type,
                "feature_names": self.feature_names,
                # MI feature selection state
                "feature_mask": self.feature_mask,
                "selected_feature_names": self.selected_feature_names,
                "mi_scores": self.mi_scores,
                "mi_result": mi_result_dict,
                "use_mi_selection": self.use_mi_selection,
                "mi_threshold": self.mi_threshold,
                # Advanced MI state
                "use_advanced_mi": self.use_advanced_mi,
                "mi_method": self.mi_method,
                "mi_max_features": self.mi_max_features,
                "advanced_mi_result": advanced_mi_dict,
                # Synthetic data state
                "synthetic_method": self.synthetic_method.value,
                "synthetic_ratio": self.synthetic_ratio,
                "adversarial_ratio": self.adversarial_ratio,
                "synthetic_result": synthetic_result_dict,
            }, f)
        logger.info(f"Saved {self.model_type} model to {path}")
        if self.selected_feature_names:
            logger.info(f"  Selected features: {self.selected_feature_names}")
        if self.use_advanced_mi:
            logger.info(f"  Advanced MI method: {self.mi_method}")
        if self.synthetic_result:
            logger.info(f"  Synthetic method: {self.synthetic_result.method}")

    def load(self, path: str):
        """Load model from file, including MI and synthetic data state."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.is_trained = data["is_trained"]
            self.model_type = data.get("model_type", "unknown")
            if "feature_names" in data:
                self.feature_names = data["feature_names"]

            # Load MI feature selection state
            self.feature_mask = data.get("feature_mask")
            self.selected_feature_names = data.get("selected_feature_names")
            self.mi_scores = data.get("mi_scores")
            self.mi_score_std = data.get("mi_score_std")
            self.use_mi_selection = data.get("use_mi_selection", False)
            self.mi_threshold = data.get("mi_threshold", 0.02)

            # Reconstruct MI result if available
            if mi_data := data.get("mi_result"):
                self.mi_result = MutualInfoResult(
                    feature_scores=mi_data["feature_scores"],
                    feature_score_std=mi_data.get("feature_score_std"),
                    selected_features=mi_data["selected_features"],
                    dropped_features=mi_data["dropped_features"],
                    threshold=mi_data["threshold"],
                    total_mi=mi_data["total_mi"],
                )

            # Load advanced MI state
            self.use_advanced_mi = data.get("use_advanced_mi", False)
            self.mi_method = data.get("mi_method", "mrmr")
            self.mi_max_features = data.get("mi_max_features", 25)
            # Note: advanced_mi_result is loaded as dict; selector is not persisted

            # Load synthetic data state
            if synth_method := data.get("synthetic_method"):
                try:
                    self.synthetic_method = SyntheticMethod(synth_method)
                except ValueError:
                    self.synthetic_method = SyntheticMethod.NONE
            self.synthetic_ratio = data.get("synthetic_ratio", 1.0)
            self.adversarial_ratio = data.get("adversarial_ratio", 0.1)

            # Reconstruct synthetic result if available
            if synth_data := data.get("synthetic_result"):
                self.synthetic_result = SyntheticDataResult(
                    method=synth_data["method"],
                    original_samples=synth_data["original_samples"],
                    original_class_ratio=synth_data["original_class_ratio"],
                    synthetic_samples_added=synth_data["synthetic_samples_added"],
                    final_samples=synth_data["final_samples"],
                    final_class_ratio=synth_data["final_class_ratio"],
                    adversarial_samples=synth_data.get("adversarial_samples", 0),
                )

        logger.info(f"Loaded {self.model_type} model from {path}")
        if self.selected_feature_names:
            logger.info(f"  Selected features: {self.selected_feature_names}")
        if self.synthetic_result:
            logger.info(f"  Synthetic method: {self.synthetic_result.method}")

    @staticmethod
    def check_automl_availability() -> Dict[str, bool]:
        """Check which AutoML backends are available."""
        return {
            "tpot": TPOT_AVAILABLE,
            "autosklearn": AUTOSKLEARN_AVAILABLE,
            "random_forest": True,  # Always available
            "mutual_info_selection": True,  # Always available via sklearn
            "advanced_mi_selection": ADVANCED_MI_AVAILABLE,  # mRMR/JMI/CMIM
            "smote": SMOTE_AVAILABLE,
            "adversarial_generation": True,  # Custom implementation
            "feature_conditional_generation": True,  # Custom implementation
            "temporal_synthetic_generation": TEMPORAL_SYNTHETIC_AVAILABLE,  # Lifecycle sequences
        }

    def analyze_feature_selection_impact(
        self,
        tokens: List[TokenFeatures],
        labels: List[int],
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Compare model performance with and without MI feature selection.

        Useful for validating that MI selection improves generalization.

        Args:
            tokens: List of token features
            labels: 1 = rug, 0 = safe
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with comparison metrics
        """
        X = np.array([self._extract_features(t) for t in tokens])
        y = np.array(labels)
        X_scaled = self.scaler.fit_transform(X)

        # Train baseline model (all features)
        baseline_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
        )
        baseline_scores = cross_val_score(baseline_model, X_scaled, y, cv=cv_folds)

        # Compute MI and select features
        mi_result = self._compute_mutual_information(X_scaled, y)
        X_selected = self._apply_feature_selection(X_scaled)

        # Train MI-selected model
        mi_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
        )
        mi_scores = cross_val_score(mi_model, X_selected, y, cv=cv_folds)

        # Compute improvement
        baseline_mean = baseline_scores.mean()
        mi_mean = mi_scores.mean()
        improvement = (mi_mean - baseline_mean) / baseline_mean * 100

        result = {
            "baseline": {
                "features": len(self.feature_names),
                "cv_mean": baseline_mean,
                "cv_std": baseline_scores.std(),
            },
            "mi_selected": {
                "features": len(mi_result.selected_features),
                "selected": mi_result.selected_features,
                "dropped": mi_result.dropped_features,
                "cv_mean": mi_mean,
                "cv_std": mi_scores.std(),
            },
            "improvement_pct": improvement,
            "mi_scores": mi_result.feature_scores,
        }

        logger.info(
            f"Feature selection impact: baseline={baseline_mean:.4f}, "
            f"mi_selected={mi_mean:.4f}, improvement={improvement:+.2f}%"
        )

        return result

    def analyze_synthetic_data_impact(
        self,
        tokens: List[TokenFeatures],
        labels: List[int],
        methods: Optional[List[SyntheticMethod]] = None,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Compare model performance with different synthetic data methods.

        Evaluates how different augmentation strategies affect recall, precision,
        and F1 score for rug detection.

        Args:
            tokens: List of token features
            labels: 1 = rug, 0 = safe
            methods: List of methods to compare (default: all available)
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with comparison metrics for each method
        """
        from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

        if methods is None:
            methods = [
                SyntheticMethod.NONE,
                SyntheticMethod.SMOTE,
                SyntheticMethod.BORDERLINE_SMOTE,
                SyntheticMethod.ADVERSARIAL,
                SyntheticMethod.HYBRID,
            ]

        X = np.array([self._extract_features(t) for t in tokens])
        y = np.array(labels)
        X_scaled = self.scaler.fit_transform(X)

        # Apply MI selection if enabled
        if self.use_mi_selection:
            self._compute_mutual_information(X_scaled, y)
            X_selected = self._apply_feature_selection(X_scaled)
        else:
            X_selected = X_scaled

        results = {}

        for method in methods:
            method_name = method.value

            # Skip SMOTE methods if not available
            if method in [
                SyntheticMethod.SMOTE,
                SyntheticMethod.BORDERLINE_SMOTE,
                SyntheticMethod.ADASYN,
                SyntheticMethod.SMOTE_TOMEK,
            ] and not SMOTE_AVAILABLE:
                logger.warning(f"Skipping {method_name}: imbalanced-learn not installed")
                continue

            # Temporarily set method
            original_method = self.synthetic_method
            self.synthetic_method = method

            try:
                # Augment data
                if method != SyntheticMethod.NONE:
                    X_aug, y_aug, synth_result = self._augment_with_synthetic_data(
                        X_selected, y
                    )
                else:
                    X_aug, y_aug = X_selected, y
                    synth_result = None

                # Train and evaluate
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1,
                    class_weight="balanced" if method == SyntheticMethod.NONE else None,
                )

                # Calculate multiple metrics
                accuracy_scores = cross_val_score(model, X_aug, y_aug, cv=cv_folds)

                # For precision/recall/F1, we need custom scoring
                precision_scorer = make_scorer(precision_score, zero_division=0)
                recall_scorer = make_scorer(recall_score, zero_division=0)
                f1_scorer = make_scorer(f1_score, zero_division=0)

                precision_scores = cross_val_score(
                    model, X_aug, y_aug, cv=cv_folds, scoring=precision_scorer
                )
                recall_scores = cross_val_score(
                    model, X_aug, y_aug, cv=cv_folds, scoring=recall_scorer
                )
                f1_scores = cross_val_score(
                    model, X_aug, y_aug, cv=cv_folds, scoring=f1_scorer
                )

                results[method_name] = {
                    "samples": len(y_aug),
                    "synthetic_added": len(y_aug) - len(y) if synth_result else 0,
                    "accuracy": {
                        "mean": accuracy_scores.mean(),
                        "std": accuracy_scores.std(),
                    },
                    "precision": {
                        "mean": precision_scores.mean(),
                        "std": precision_scores.std(),
                    },
                    "recall": {
                        "mean": recall_scores.mean(),
                        "std": recall_scores.std(),
                    },
                    "f1": {
                        "mean": f1_scores.mean(),
                        "std": f1_scores.std(),
                    },
                }

                logger.info(
                    f"{method_name}: accuracy={accuracy_scores.mean():.3f}, "
                    f"recall={recall_scores.mean():.3f}, F1={f1_scores.mean():.3f}"
                )

            except Exception as e:
                logger.warning(f"Failed to evaluate {method_name}: {e}")
                results[method_name] = {"error": str(e)}

            finally:
                self.synthetic_method = original_method

        # Find best method by F1 score
        valid_results = {k: v for k, v in results.items() if "f1" in v}
        if valid_results:
            best_method = max(valid_results.keys(), key=lambda k: valid_results[k]["f1"]["mean"])
            best_f1 = valid_results[best_method]["f1"]["mean"]
            baseline_f1 = valid_results.get("none", {}).get("f1", {}).get("mean", 0)

            improvement = ((best_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0

            results["_summary"] = {
                "best_method": best_method,
                "best_f1": best_f1,
                "baseline_f1": baseline_f1,
                "improvement_pct": improvement,
            }

            logger.info(
                f"Best method: {best_method} (F1={best_f1:.3f}, "
                f"improvement={improvement:+.1f}% over baseline)"
            )

        return results
