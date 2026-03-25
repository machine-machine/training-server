"""
HMM-based regime detection for adaptive trading strategies.

Classifies market conditions into distinct regimes:
- Bull: Strong upward momentum, low volatility
- Bear: Downward trend, increasing volatility
- Chop: Sideways movement, mean-reverting
- MEV Heavy: High MEV activity affecting execution

Trains separate model weights per regime for adaptive scoring.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

try:
    from hmmlearn import hmm

    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""

    BULL = "bull"  # Strong upward momentum
    BEAR = "bear"  # Downward trend
    CHOP = "chop"  # Sideways, mean-reverting
    MEV_HEAVY = "mev_heavy"  # High MEV activity

    @classmethod
    def from_index(cls, idx: int) -> "MarketRegime":
        """Convert HMM state index to regime."""
        mapping = {0: cls.BULL, 1: cls.BEAR, 2: cls.CHOP, 3: cls.MEV_HEAVY}
        return mapping.get(idx, cls.CHOP)

    def to_index(self) -> int:
        """Convert regime to index."""
        mapping = {self.BULL: 0, self.BEAR: 1, self.CHOP: 2, self.MEV_HEAVY: 3}
        return mapping.get(self, 2)


@dataclass
class RegimeFeatures:
    """Features used for regime detection."""

    volatility: float  # Price volatility (std of returns)
    momentum: float  # Price momentum (return over period)
    mev_activity: float  # MEV activity level (sandwich attacks, frontrunning)
    liquidity_delta: float  # Change in liquidity (LP additions/removals)
    volume_ratio: float = 1.0  # Volume relative to average

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array(
            [
                self.volatility,
                self.momentum,
                self.mev_activity,
                self.liquidity_delta,
                self.volume_ratio,
            ]
        )


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""

    # HMM parameters
    n_regimes: int = 4  # Number of regimes (states)
    n_iter: int = 100  # HMM training iterations
    covariance_type: str = "full"  # Covariance type

    # Feature calculation
    volatility_window: int = 20  # Window for volatility calculation
    momentum_window: int = 10  # Window for momentum calculation
    mev_threshold: float = 0.1  # Threshold for high MEV activity

    # Regime-specific adjustments (less punishing)
    bull_confidence_boost: float = 1.2  # Boost confidence in bull regime
    bear_confidence_penalty: float = 0.85  # Reduced from 0.8 - less punishing
    chop_position_limit: float = 0.70  # Increased from 0.5 - less restrictive
    mev_slippage_multiplier: float = 1.5  # Increase slippage in MEV heavy

    # Minimum data requirements
    min_samples: int = 100
    min_regime_samples: int = 10  # Minimum samples per regime

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_regimes": self.n_regimes,
            "n_iter": self.n_iter,
            "volatility_window": self.volatility_window,
            "momentum_window": self.momentum_window,
            "regime_adjustments": {
                "bull_confidence_boost": self.bull_confidence_boost,
                "bear_confidence_penalty": self.bear_confidence_penalty,
                "chop_position_limit": self.chop_position_limit,
                "mev_slippage_multiplier": self.mev_slippage_multiplier,
            },
        }


@dataclass
class RegimeState:
    """Current regime state for export to HotPath."""

    current_regime: MarketRegime
    regime_probabilities: dict[str, float]  # Probability of each regime
    confidence: float  # Confidence in current regime
    time_in_regime: int  # Time steps in current regime
    transition_probability: float  # Probability of regime change

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_regime": self.current_regime.value,
            "probabilities": self.regime_probabilities,
            "confidence": self.confidence,
            "time_in_regime": self.time_in_regime,
            "transition_probability": self.transition_probability,
        }


class RegimeDetector:
    """HMM-based regime classification for adaptive trading.

    Detects market regimes and provides regime-specific adjustments
    for trading strategy parameters.

    Example:
        detector = RegimeDetector(config)
        detector.fit(historical_features)
        current_regime = detector.predict(current_features)
        adjustments = detector.get_regime_adjustments()
    """

    REGIMES = [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.CHOP, MarketRegime.MEV_HEAVY]

    def __init__(self, config: RegimeConfig | None = None):
        if not HMM_AVAILABLE:
            logger.warning(
                "hmmlearn not available. Install with: pip install hmmlearn. "
                "Falling back to simple threshold-based detection."
            )

        self.config = config or RegimeConfig()
        self._model: hmm.GaussianHMM | None = None
        self._scaler: StandardScaler | None = None
        self._is_fitted = False

        # State tracking
        self._current_regime = MarketRegime.CHOP
        self._time_in_regime = 0
        self._regime_history: list[MarketRegime] = []

        # Regime-specific model weights (trained separately per regime)
        self._regime_weights: dict[str, dict[str, Any]] = {}

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, observations: np.ndarray) -> "RegimeDetector":
        """Fit the HMM model on historical observations.

        Args:
            observations: Array of shape (n_samples, n_features)
                Features: [volatility, momentum, mev_activity, liquidity_delta, volume_ratio]

        Returns:
            Self for chaining
        """
        n_obs = len(observations)
        if n_obs < self.config.min_samples:
            logger.warning(
                f"Insufficient samples for regime detection: {n_obs} < {self.config.min_samples}"
            )
            self._is_fitted = False
            return self

        # Standardize features
        self._scaler = StandardScaler()
        scaled_obs = self._scaler.fit_transform(observations)

        if HMM_AVAILABLE:
            # Fit Gaussian HMM
            self._model = hmm.GaussianHMM(
                n_components=self.config.n_regimes,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                random_state=42,
            )
            self._model.fit(scaled_obs)

            # Label regimes based on emission means
            self._label_regimes()
        else:
            # Fallback: Use KMeans clustering
            if SKLEARN_AVAILABLE:
                self._model = KMeans(
                    n_clusters=self.config.n_regimes,
                    random_state=42,
                    n_init=10,
                )
                self._model.fit(scaled_obs)

        self._is_fitted = True
        logger.info(f"Regime detector fitted on {len(observations)} samples")

        return self

    def _label_regimes(self):
        """Label HMM states based on emission characteristics.

        Assigns regime labels based on the mean emission for each state:
        - High momentum, low volatility -> BULL
        - Negative momentum, high volatility -> BEAR
        - Low momentum, low volatility -> CHOP
        - High MEV activity -> MEV_HEAVY
        """
        if not HMM_AVAILABLE or self._model is None:
            return

        means = self._model.means_
        n_states = len(means)

        # Feature indices: [volatility, momentum, mev_activity, liquidity_delta, volume_ratio]
        VOL_IDX = 0
        MOM_IDX = 1
        MEV_IDX = 2

        self._state_to_regime = {}

        for state_idx in range(n_states):
            state_mean = means[state_idx]
            vol = state_mean[VOL_IDX]
            mom = state_mean[MOM_IDX]
            mev = state_mean[MEV_IDX]

            # Classification logic
            if mev > 0.5:  # High MEV activity
                regime = MarketRegime.MEV_HEAVY
            elif mom > 0.3 and vol < 0.5:  # Strong positive momentum, low vol
                regime = MarketRegime.BULL
            elif mom < -0.3 or vol > 0.5:  # Negative momentum or high vol
                regime = MarketRegime.BEAR
            else:  # Everything else
                regime = MarketRegime.CHOP

            self._state_to_regime[state_idx] = regime

    def predict(self, features: RegimeFeatures) -> MarketRegime:
        """Predict current market regime.

        Args:
            features: Current market features

        Returns:
            Predicted market regime
        """
        if not self._is_fitted:
            return MarketRegime.CHOP

        obs = features.to_array().reshape(1, -1)

        if self._scaler is not None:
            obs = self._scaler.transform(obs)

        if HMM_AVAILABLE and isinstance(self._model, hmm.GaussianHMM):
            state = self._model.predict(obs)[0]
            regime = self._state_to_regime.get(state, MarketRegime.CHOP)
        elif SKLEARN_AVAILABLE and hasattr(self._model, "predict"):
            state = self._model.predict(obs)[0]
            regime = MarketRegime.from_index(state)
        else:
            regime = self._simple_threshold_detection(features)

        # Update state tracking
        if regime == self._current_regime:
            self._time_in_regime += 1
        else:
            self._current_regime = regime
            self._time_in_regime = 1

        self._regime_history.append(regime)

        return regime

    def predict_proba(self, features: RegimeFeatures) -> dict[str, float]:
        """Get probability distribution over regimes.

        Args:
            features: Current market features

        Returns:
            Dictionary mapping regime name to probability
        """
        if not self._is_fitted:
            return {r.value: 0.25 for r in self.REGIMES}

        obs = features.to_array().reshape(1, -1)

        if self._scaler is not None:
            obs = self._scaler.transform(obs)

        if HMM_AVAILABLE and isinstance(self._model, hmm.GaussianHMM):
            # Get posterior probabilities
            posteriors = self._model.predict_proba(obs)[0]

            probs = {}
            for state_idx, prob in enumerate(posteriors):
                regime = self._state_to_regime.get(state_idx, MarketRegime.CHOP)
                if regime.value in probs:
                    probs[regime.value] += prob
                else:
                    probs[regime.value] = prob

            return probs
        else:
            # Fallback: Return uniform distribution
            return {r.value: 0.25 for r in self.REGIMES}

    def _simple_threshold_detection(self, features: RegimeFeatures) -> MarketRegime:
        """Simple threshold-based regime detection (fallback)."""
        if features.mev_activity > self.config.mev_threshold:
            return MarketRegime.MEV_HEAVY
        elif features.momentum > 0.05 and features.volatility < 0.3:
            return MarketRegime.BULL
        elif features.momentum < -0.05 or features.volatility > 0.5:
            return MarketRegime.BEAR
        else:
            return MarketRegime.CHOP

    def get_regime_state(self) -> RegimeState:
        """Get current regime state for export."""
        probs = self.predict_proba(RegimeFeatures(0, 0, 0, 0, 1))

        current_prob = probs.get(self._current_regime.value, 0.25)

        return RegimeState(
            current_regime=self._current_regime,
            regime_probabilities=probs,
            confidence=current_prob,
            time_in_regime=self._time_in_regime,
            transition_probability=1.0 - current_prob,
        )

    def get_regime_adjustments(self, regime: MarketRegime | None = None) -> dict[str, float]:
        """Get trading parameter adjustments for a regime.

        Args:
            regime: Regime to get adjustments for (default: current)

        Returns:
            Dictionary of parameter adjustments
        """
        regime = regime or self._current_regime

        if regime == MarketRegime.BULL:
            return {
                "confidence_multiplier": self.config.bull_confidence_boost,
                "position_limit_multiplier": 1.0,
                "slippage_multiplier": 1.0,
                "take_profit_multiplier": 1.2,  # Higher targets in bull
                "stop_loss_multiplier": 1.0,
            }
        elif regime == MarketRegime.BEAR:
            return {
                "confidence_multiplier": self.config.bear_confidence_penalty,
                "position_limit_multiplier": 0.80,  # Increased from 0.7
                "slippage_multiplier": 1.2,
                "take_profit_multiplier": 0.8,  # Lower targets in bear
                "stop_loss_multiplier": 0.8,  # Tighter stops
            }
        elif regime == MarketRegime.CHOP:
            return {
                "confidence_multiplier": 1.0,
                "position_limit_multiplier": self.config.chop_position_limit,
                "slippage_multiplier": 1.0,
                "take_profit_multiplier": 0.7,  # Quick profits in chop
                "stop_loss_multiplier": 0.9,
            }
        elif regime == MarketRegime.MEV_HEAVY:
            return {
                "confidence_multiplier": 0.92,  # Increased from 0.9
                "position_limit_multiplier": 0.75,  # Increased from 0.6
                "slippage_multiplier": self.config.mev_slippage_multiplier,
                "take_profit_multiplier": 1.3,  # Need bigger moves to overcome MEV
                "stop_loss_multiplier": 1.2,
            }
        else:
            return {
                "confidence_multiplier": 1.0,
                "position_limit_multiplier": 1.0,
                "slippage_multiplier": 1.0,
                "take_profit_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
            }

    def set_regime_weights(self, regime: MarketRegime, weights: dict[str, Any]):
        """Set model weights for a specific regime.

        Allows training separate models per regime for adaptive scoring.

        Args:
            regime: The regime these weights apply to
            weights: Model weights dictionary (same format as profitability learner export)
        """
        self._regime_weights[regime.value] = weights

    def get_regime_weights(self, regime: MarketRegime | None = None) -> dict[str, Any] | None:
        """Get model weights for a specific regime.

        Args:
            regime: Regime to get weights for (default: current)

        Returns:
            Model weights dictionary or None if not set
        """
        regime = regime or self._current_regime
        return self._regime_weights.get(regime.value)

    def get_export_data(self) -> dict[str, Any]:
        """Export regime detector state for HotPath."""
        export = {
            "config": self.config.to_dict(),
            "current_state": self.get_regime_state().to_dict(),
            "adjustments": {r.value: self.get_regime_adjustments(r) for r in self.REGIMES},
        }

        # Include regime-specific weights if available
        if self._regime_weights:
            export["regime_weights"] = self._regime_weights

        # Include HMM parameters if fitted
        if self._is_fitted and HMM_AVAILABLE and isinstance(self._model, hmm.GaussianHMM):
            export["hmm_params"] = {
                "means": self._model.means_.tolist(),
                "covars": self._model.covars_.tolist() if hasattr(self._model, "covars_") else None,
                "transmat": self._model.transmat_.tolist(),
                "startprob": self._model.startprob_.tolist(),
            }

        return export

    def load_from_export(self, data: dict[str, Any]):
        """Load regime detector state from export data."""
        if "config" in data:
            config_data = data["config"]
            self.config = RegimeConfig(
                n_regimes=config_data.get("n_regimes", 4),
                volatility_window=config_data.get("volatility_window", 20),
                momentum_window=config_data.get("momentum_window", 10),
            )

        if "regime_weights" in data:
            self._regime_weights = data["regime_weights"]

        if "current_state" in data:
            state = data["current_state"]
            self._current_regime = MarketRegime(state.get("current_regime", "chop"))
            self._time_in_regime = state.get("time_in_regime", 0)

        self._is_fitted = "hmm_params" in data or "regime_weights" in data


def compute_regime_features(
    prices: np.ndarray,
    volumes: np.ndarray,
    mev_counts: np.ndarray | None = None,
    liquidity: np.ndarray | None = None,
    volatility_window: int = 20,
    momentum_window: int = 10,
) -> np.ndarray:
    """Compute regime detection features from market data.

    Args:
        prices: Price array
        volumes: Volume array
        mev_counts: MEV event counts (optional)
        liquidity: Liquidity array (optional)
        volatility_window: Window for volatility calculation
        momentum_window: Window for momentum calculation

    Returns:
        Array of shape (n_samples, 5) with features:
        [volatility, momentum, mev_activity, liquidity_delta, volume_ratio]
    """
    n = len(prices)
    features = np.zeros((n, 5))

    # Returns for calculations
    returns = np.diff(prices) / prices[:-1]
    returns = np.insert(returns, 0, 0)

    for i in range(volatility_window, n):
        # Volatility: rolling std of returns
        window_returns = returns[i - volatility_window : i]
        features[i, 0] = np.std(window_returns)

        # Momentum: return over momentum window
        if i >= momentum_window:
            features[i, 1] = (prices[i] - prices[i - momentum_window]) / prices[i - momentum_window]

        # MEV activity: normalized count
        if mev_counts is not None:
            window_mev = mev_counts[i - volatility_window : i]
            features[i, 2] = np.mean(window_mev) / (np.mean(mev_counts) + 1e-10)

        # Liquidity delta: change from previous
        if liquidity is not None and i > 0:
            features[i, 3] = (liquidity[i] - liquidity[i - 1]) / (liquidity[i - 1] + 1e-10)

        # Volume ratio: current vs rolling average
        avg_volume = np.mean(volumes[i - volatility_window : i])
        features[i, 4] = volumes[i] / (avg_volume + 1e-10)

    return features
