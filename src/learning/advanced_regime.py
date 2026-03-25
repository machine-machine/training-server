"""
Advanced 5-State HMM Regime Detection.

Implements 5-state Hidden Markov Model for market regime detection:
- MEME_SEASON: High new tokens, volume spike
- NORMAL: Baseline conditions
- BEAR_VOLATILE: High vol, negative momentum
- HIGH_MEV: Sandwich activity spike
- LOW_LIQUIDITY: TVL drawdown

Features (8): volatility, momentum, mev_activity, liquidity_delta,
              volume_ratio, new_token_rate, sandwich_rate, avg_trade_size

Per-regime model weights for adaptive scoring.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class AdvancedRegime(Enum):
    """5-state market regime classifications."""
    MEME_SEASON = "meme_season"       # High new tokens, viral activity
    NORMAL = "normal"                  # Baseline market conditions
    BEAR_VOLATILE = "bear_volatile"    # Downtrend with high volatility
    HIGH_MEV = "high_mev"              # Elevated MEV/sandwich activity
    LOW_LIQUIDITY = "low_liquidity"    # TVL drawdown, thin markets

    @classmethod
    def from_index(cls, idx: int) -> "AdvancedRegime":
        """Convert HMM state index to regime."""
        mapping = {
            0: cls.MEME_SEASON,
            1: cls.NORMAL,
            2: cls.BEAR_VOLATILE,
            3: cls.HIGH_MEV,
            4: cls.LOW_LIQUIDITY,
        }
        return mapping.get(idx, cls.NORMAL)

    def to_index(self) -> int:
        """Convert regime to index."""
        mapping = {
            self.MEME_SEASON: 0,
            self.NORMAL: 1,
            self.BEAR_VOLATILE: 2,
            self.HIGH_MEV: 3,
            self.LOW_LIQUIDITY: 4,
        }
        return mapping.get(self, 1)


@dataclass
class RegimeFeatures8D:
    """8-dimensional feature vector for regime detection."""
    volatility: float              # Price volatility (30-day rolling)
    momentum: float                # Price momentum (1-week return)
    mev_activity: float            # MEV activity level (normalized)
    liquidity_delta: float         # Change in aggregate liquidity
    volume_ratio: float            # Volume vs average ratio
    new_token_rate: float          # New token creation rate
    sandwich_rate: float           # Sandwich attack frequency
    avg_trade_size: float          # Average trade size (normalized)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.volatility,
            self.momentum,
            self.mev_activity,
            self.liquidity_delta,
            self.volume_ratio,
            self.new_token_rate,
            self.sandwich_rate,
            self.avg_trade_size,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "RegimeFeatures8D":
        """Create from numpy array."""
        return cls(
            volatility=float(arr[0]),
            momentum=float(arr[1]),
            mev_activity=float(arr[2]),
            liquidity_delta=float(arr[3]),
            volume_ratio=float(arr[4]),
            new_token_rate=float(arr[5]),
            sandwich_rate=float(arr[6]),
            avg_trade_size=float(arr[7]),
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "volatility": self.volatility,
            "momentum": self.momentum,
            "mev_activity": self.mev_activity,
            "liquidity_delta": self.liquidity_delta,
            "volume_ratio": self.volume_ratio,
            "new_token_rate": self.new_token_rate,
            "sandwich_rate": self.sandwich_rate,
            "avg_trade_size": self.avg_trade_size,
        }


@dataclass
class AdvancedRegimeState:
    """Current advanced regime state."""
    regime: AdvancedRegime
    probabilities: Dict[str, float]
    confidence: float
    features: RegimeFeatures8D
    time_in_regime: int
    transition_matrix: Optional[np.ndarray]
    expected_duration: float  # Expected remaining time in regime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "features": self.features.to_dict(),
            "time_in_regime": self.time_in_regime,
            "expected_duration": self.expected_duration,
        }


@dataclass
class RegimeModelWeights:
    """Model weights specific to a regime."""
    regime: AdvancedRegime
    feature_weights: Dict[str, float]
    confidence_multiplier: float
    position_limit: float
    slippage_multiplier: float
    take_profit_multiplier: float
    stop_loss_multiplier: float
    preferred_strategies: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "feature_weights": self.feature_weights,
            "confidence_multiplier": self.confidence_multiplier,
            "position_limit": self.position_limit,
            "slippage_multiplier": self.slippage_multiplier,
            "take_profit_multiplier": self.take_profit_multiplier,
            "stop_loss_multiplier": self.stop_loss_multiplier,
            "preferred_strategies": self.preferred_strategies,
        }


# Default regime-specific model weights
DEFAULT_REGIME_WEIGHTS: Dict[str, RegimeModelWeights] = {
    "meme_season": RegimeModelWeights(
        regime=AdvancedRegime.MEME_SEASON,
        feature_weights={
            "holder_growth_velocity": 1.5,
            "social_authenticity_score": 1.3,
            "volume_acceleration": 1.4,
            "twitter_mention_velocity": 1.3,
            "buy_volume_ratio": 1.2,
        },
        confidence_multiplier=1.3,
        position_limit=0.8,
        slippage_multiplier=1.2,
        take_profit_multiplier=1.5,
        stop_loss_multiplier=0.8,
        preferred_strategies=["momentum", "announcement"],
    ),
    "normal": RegimeModelWeights(
        regime=AdvancedRegime.NORMAL,
        feature_weights={
            "pool_tvl_sol": 1.0,
            "rug_pull_ml_score": 1.0,
            "holder_count_unique": 1.0,
            "lp_lock_percentage": 1.0,
            "price_momentum_5m": 1.0,
        },
        confidence_multiplier=1.0,
        position_limit=1.0,
        slippage_multiplier=1.0,
        take_profit_multiplier=1.0,
        stop_loss_multiplier=1.0,
        preferred_strategies=["ml_ensemble", "momentum"],
    ),
    "bear_volatile": RegimeModelWeights(
        regime=AdvancedRegime.BEAR_VOLATILE,
        feature_weights={
            "rug_pull_ml_score": 1.8,
            "volatility_5m": 1.5,
            "top_10_holder_concentration": 1.4,
            "mint_authority_revoked": 1.5,
            "hidden_fee_detected": 1.6,
        },
        confidence_multiplier=0.7,
        position_limit=0.5,
        slippage_multiplier=1.5,
        take_profit_multiplier=0.6,
        stop_loss_multiplier=0.7,
        preferred_strategies=["mean_reversion"],
    ),
    "high_mev": RegimeModelWeights(
        regime=AdvancedRegime.HIGH_MEV,
        feature_weights={
            "slippage_1pct": 1.6,
            "slippage_5pct": 1.5,
            "price_impact_1pct": 1.4,
            "trade_size_variance": 1.3,
            "pool_depth_imbalance": 1.3,
        },
        confidence_multiplier=0.85,
        position_limit=0.6,
        slippage_multiplier=2.0,
        take_profit_multiplier=1.3,
        stop_loss_multiplier=1.2,
        preferred_strategies=["liquidity"],
    ),
    "low_liquidity": RegimeModelWeights(
        regime=AdvancedRegime.LOW_LIQUIDITY,
        feature_weights={
            "pool_tvl_sol": 2.0,
            "unique_lp_provider_count": 1.5,
            "lp_removal_velocity": 1.6,
            "emergency_liquidity_flag": 2.0,
            "deployer_lp_ownership_pct": 1.4,
        },
        confidence_multiplier=0.6,
        position_limit=0.3,
        slippage_multiplier=2.5,
        take_profit_multiplier=0.5,
        stop_loss_multiplier=0.6,
        preferred_strategies=[],
    ),
}


class AdvancedRegimeDetector:
    """5-state HMM-based regime detection with per-regime model weights.

    Uses Gaussian HMM to detect market regimes from 8-dimensional
    feature vectors. Each regime has custom model weights for
    adaptive scoring.

    Example:
        detector = AdvancedRegimeDetector()
        detector.fit(historical_features)
        state = detector.get_current_state(current_features)
        weights = detector.get_regime_weights()
    """

    N_REGIMES = 5
    N_FEATURES = 8

    def __init__(
        self,
        n_iter: int = 100,
        covariance_type: str = "full",
        min_samples: int = 100,
    ):
        """Initialize detector.

        Args:
            n_iter: HMM training iterations
            covariance_type: HMM covariance type
            min_samples: Minimum samples for training
        """
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.min_samples = min_samples

        self._model: Optional[hmm.GaussianHMM] = None
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted = False

        # State mapping (HMM state -> regime)
        self._state_to_regime: Dict[int, AdvancedRegime] = {}

        # Tracking
        self._current_regime = AdvancedRegime.NORMAL
        self._time_in_regime = 0
        self._history: List[AdvancedRegime] = []
        self._last_features: Optional[RegimeFeatures8D] = None

        # Per-regime weights
        self._regime_weights = DEFAULT_REGIME_WEIGHTS.copy()

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, observations: np.ndarray) -> "AdvancedRegimeDetector":
        """Fit HMM on historical feature observations.

        Args:
            observations: Array of shape (n_samples, 8) with features

        Returns:
            Self for chaining
        """
        if len(observations) < self.min_samples:
            logger.warning(
                f"Insufficient samples: {len(observations)} < {self.min_samples}"
            )
            self._is_fitted = False
            return self

        # Standardize features
        self._scaler = StandardScaler()
        scaled_obs = self._scaler.fit_transform(observations)

        if HMM_AVAILABLE:
            # Train Gaussian HMM
            self._model = hmm.GaussianHMM(
                n_components=self.N_REGIMES,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=42,
            )
            self._model.fit(scaled_obs)

            # Label states based on emission means
            self._label_states()
        elif SKLEARN_AVAILABLE:
            # Fallback to GMM
            self._model = GaussianMixture(
                n_components=self.N_REGIMES,
                covariance_type=self.covariance_type,
                n_init=3,
                random_state=42,
            )
            self._model.fit(scaled_obs)
            self._label_states_gmm()
        else:
            logger.warning("Neither hmmlearn nor sklearn available")
            self._is_fitted = False
            return self

        self._is_fitted = True
        logger.info(f"Regime detector fitted on {len(observations)} samples")
        return self

    def _label_states(self):
        """Label HMM states based on emission characteristics.

        Feature indices:
        0: volatility, 1: momentum, 2: mev_activity, 3: liquidity_delta,
        4: volume_ratio, 5: new_token_rate, 6: sandwich_rate, 7: avg_trade_size
        """
        if not HMM_AVAILABLE or self._model is None:
            return

        means = self._model.means_

        for state_idx in range(self.N_REGIMES):
            m = means[state_idx]
            regime = self._classify_state_by_features(m)
            self._state_to_regime[state_idx] = regime

    def _label_states_gmm(self):
        """Label GMM states (fallback)."""
        if self._model is None:
            return

        means = self._model.means_

        for state_idx in range(self.N_REGIMES):
            m = means[state_idx]
            regime = self._classify_state_by_features(m)
            self._state_to_regime[state_idx] = regime

    def _classify_state_by_features(self, m: np.ndarray) -> AdvancedRegime:
        """Classify a state based on feature means.

        Args:
            m: Mean feature vector for state

        Returns:
            AdvancedRegime classification
        """
        volatility = m[0]
        momentum = m[1]
        mev_activity = m[2]
        liquidity_delta = m[3]
        volume_ratio = m[4]
        new_token_rate = m[5]
        sandwich_rate = m[6]

        # Classification rules
        if new_token_rate > 0.5 and volume_ratio > 1.0:
            return AdvancedRegime.MEME_SEASON
        elif mev_activity > 0.5 or sandwich_rate > 0.5:
            return AdvancedRegime.HIGH_MEV
        elif liquidity_delta < -0.3 or volume_ratio < 0.5:
            return AdvancedRegime.LOW_LIQUIDITY
        elif volatility > 0.5 and momentum < -0.3:
            return AdvancedRegime.BEAR_VOLATILE
        else:
            return AdvancedRegime.NORMAL

    def predict(self, features: RegimeFeatures8D) -> AdvancedRegime:
        """Predict regime from current features.

        Args:
            features: Current 8D feature vector

        Returns:
            Predicted regime
        """
        if not self._is_fitted:
            return self._threshold_detection(features)

        obs = features.to_array().reshape(1, -1)

        if self._scaler is not None:
            obs = self._scaler.transform(obs)

        if HMM_AVAILABLE and isinstance(self._model, hmm.GaussianHMM):
            state = self._model.predict(obs)[0]
            regime = self._state_to_regime.get(state, AdvancedRegime.NORMAL)
        elif SKLEARN_AVAILABLE and hasattr(self._model, "predict"):
            state = self._model.predict(obs)[0]
            regime = self._state_to_regime.get(state, AdvancedRegime.NORMAL)
        else:
            regime = self._threshold_detection(features)

        # Update tracking
        if regime == self._current_regime:
            self._time_in_regime += 1
        else:
            self._current_regime = regime
            self._time_in_regime = 1

        self._history.append(regime)
        self._last_features = features

        return regime

    def predict_proba(self, features: RegimeFeatures8D) -> Dict[str, float]:
        """Get probability distribution over regimes.

        Args:
            features: Current features

        Returns:
            Dict mapping regime name to probability
        """
        if not self._is_fitted:
            return {r.value: 0.2 for r in AdvancedRegime}

        obs = features.to_array().reshape(1, -1)

        if self._scaler is not None:
            obs = self._scaler.transform(obs)

        if HMM_AVAILABLE and isinstance(self._model, hmm.GaussianHMM):
            posteriors = self._model.predict_proba(obs)[0]

            probs = {}
            for state_idx, prob in enumerate(posteriors):
                regime = self._state_to_regime.get(state_idx, AdvancedRegime.NORMAL)
                probs[regime.value] = probs.get(regime.value, 0) + prob

            return probs

        # Fallback
        return {r.value: 0.2 for r in AdvancedRegime}

    def _threshold_detection(self, features: RegimeFeatures8D) -> AdvancedRegime:
        """Simple threshold-based detection (fallback)."""
        if features.new_token_rate > 0.5 and features.volume_ratio > 1.5:
            return AdvancedRegime.MEME_SEASON
        elif features.mev_activity > 0.3 or features.sandwich_rate > 0.2:
            return AdvancedRegime.HIGH_MEV
        elif features.liquidity_delta < -0.2:
            return AdvancedRegime.LOW_LIQUIDITY
        elif features.volatility > 0.3 and features.momentum < -0.1:
            return AdvancedRegime.BEAR_VOLATILE
        else:
            return AdvancedRegime.NORMAL

    def get_current_state(
        self,
        features: Optional[RegimeFeatures8D] = None,
    ) -> AdvancedRegimeState:
        """Get full current regime state.

        Args:
            features: Current features (uses last if not provided)

        Returns:
            AdvancedRegimeState with all details
        """
        if features is not None:
            self.predict(features)

        features = features or self._last_features or RegimeFeatures8D(
            0, 0, 0, 0, 1, 0, 0, 1
        )

        probs = self.predict_proba(features)
        confidence = probs.get(self._current_regime.value, 0.2)

        # Expected duration from transition matrix
        expected_duration = self._estimate_expected_duration()

        # Get transition matrix if available
        trans_matrix = None
        if HMM_AVAILABLE and hasattr(self._model, "transmat_"):
            trans_matrix = self._model.transmat_

        return AdvancedRegimeState(
            regime=self._current_regime,
            probabilities=probs,
            confidence=confidence,
            features=features,
            time_in_regime=self._time_in_regime,
            transition_matrix=trans_matrix,
            expected_duration=expected_duration,
        )

    def _estimate_expected_duration(self) -> float:
        """Estimate expected remaining time in current regime."""
        if not HMM_AVAILABLE or not hasattr(self._model, "transmat_"):
            return 10.0  # Default expectation

        state_idx = self._current_regime.to_index()
        if state_idx < len(self._model.transmat_):
            stay_prob = self._model.transmat_[state_idx, state_idx]
            # Geometric distribution: E[duration] = 1 / (1 - p_stay)
            return 1 / (1 - stay_prob + 1e-10)

        return 10.0

    def get_regime_weights(
        self,
        regime: Optional[AdvancedRegime] = None,
    ) -> RegimeModelWeights:
        """Get model weights for a regime.

        Args:
            regime: Regime (default: current)

        Returns:
            RegimeModelWeights for the regime
        """
        regime = regime or self._current_regime
        return self._regime_weights.get(regime.value, DEFAULT_REGIME_WEIGHTS["normal"])

    def set_regime_weights(self, regime: AdvancedRegime, weights: RegimeModelWeights):
        """Set custom weights for a regime."""
        self._regime_weights[regime.value] = weights

    def get_all_regime_weights(self) -> Dict[str, RegimeModelWeights]:
        """Get weights for all regimes."""
        return self._regime_weights

    def get_feature_adjustments(
        self,
        base_weights: Dict[str, float],
        regime: Optional[AdvancedRegime] = None,
    ) -> Dict[str, float]:
        """Get adjusted feature weights for current regime.

        Args:
            base_weights: Base feature weights
            regime: Regime (default: current)

        Returns:
            Adjusted weights
        """
        regime_weights = self.get_regime_weights(regime)

        adjusted = base_weights.copy()
        for feature, multiplier in regime_weights.feature_weights.items():
            if feature in adjusted:
                adjusted[feature] *= multiplier

        return adjusted

    def export_state(self) -> Dict[str, Any]:
        """Export full detector state for persistence."""
        export = {
            "current_regime": self._current_regime.value,
            "time_in_regime": self._time_in_regime,
            "is_fitted": self._is_fitted,
            "regime_weights": {
                k: v.to_dict() for k, v in self._regime_weights.items()
            },
        }

        if self._is_fitted and HMM_AVAILABLE and hasattr(self._model, "means_"):
            export["hmm_params"] = {
                "means": self._model.means_.tolist(),
                "covars": (
                    self._model.covars_.tolist()
                    if hasattr(self._model, "covars_") else None
                ),
                "transmat": self._model.transmat_.tolist(),
                "startprob": self._model.startprob_.tolist(),
            }
            export["state_mapping"] = {
                str(k): v.value for k, v in self._state_to_regime.items()
            }

        if self._scaler is not None:
            export["scaler"] = {
                "mean": self._scaler.mean_.tolist(),
                "scale": self._scaler.scale_.tolist(),
            }

        return export

    def load_state(self, state: Dict[str, Any]):
        """Load detector state from export."""
        self._current_regime = AdvancedRegime(
            state.get("current_regime", "normal")
        )
        self._time_in_regime = state.get("time_in_regime", 0)

        if "regime_weights" in state:
            for regime_name, weight_dict in state["regime_weights"].items():
                self._regime_weights[regime_name] = RegimeModelWeights(
                    regime=AdvancedRegime(weight_dict["regime"]),
                    feature_weights=weight_dict["feature_weights"],
                    confidence_multiplier=weight_dict["confidence_multiplier"],
                    position_limit=weight_dict["position_limit"],
                    slippage_multiplier=weight_dict["slippage_multiplier"],
                    take_profit_multiplier=weight_dict["take_profit_multiplier"],
                    stop_loss_multiplier=weight_dict["stop_loss_multiplier"],
                    preferred_strategies=weight_dict["preferred_strategies"],
                )

        if "state_mapping" in state:
            self._state_to_regime = {
                int(k): AdvancedRegime(v)
                for k, v in state["state_mapping"].items()
            }

        self._is_fitted = state.get("is_fitted", False)


def compute_8d_features(
    prices: np.ndarray,
    volumes: np.ndarray,
    mev_counts: np.ndarray,
    liquidity: np.ndarray,
    new_tokens: np.ndarray,
    sandwich_counts: np.ndarray,
    trade_sizes: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Compute 8D regime features from market data.

    Args:
        prices: Price array
        volumes: Volume array
        mev_counts: MEV event counts
        liquidity: Aggregate liquidity
        new_tokens: New token count per period
        sandwich_counts: Sandwich attack counts
        trade_sizes: Average trade sizes
        window: Rolling window size

    Returns:
        Array of shape (n_samples, 8)
    """
    n = len(prices)
    features = np.zeros((n, 8))

    # Returns for calculations
    returns = np.diff(prices) / (prices[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)

    for i in range(window, n):
        # Volatility: rolling std of returns
        window_returns = returns[i - window:i]
        features[i, 0] = np.std(window_returns)

        # Momentum: return over window
        features[i, 1] = (prices[i] - prices[i - window]) / (prices[i - window] + 1e-10)

        # MEV activity: normalized
        window_mev = mev_counts[i - window:i]
        features[i, 2] = np.mean(window_mev) / (np.mean(mev_counts) + 1e-10)

        # Liquidity delta
        if i > 0:
            features[i, 3] = (liquidity[i] - liquidity[i - 1]) / (liquidity[i - 1] + 1e-10)

        # Volume ratio
        avg_vol = np.mean(volumes[i - window:i])
        features[i, 4] = volumes[i] / (avg_vol + 1e-10)

        # New token rate
        features[i, 5] = np.mean(new_tokens[i - window:i]) / (np.mean(new_tokens) + 1e-10)

        # Sandwich rate
        features[i, 6] = np.mean(sandwich_counts[i - window:i]) / (np.mean(sandwich_counts) + 1e-10)

        # Average trade size (normalized)
        features[i, 7] = trade_sizes[i] / (np.mean(trade_sizes) + 1e-10)

    return features
