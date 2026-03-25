"""
Synthetic Data Generation for Training ML Models

Generates realistic memecoin market data for training when historical data is limited.
Uses regime-aware simulation to create diverse market conditions for robust training.

Features:
- Geometric Brownian Motion (GBM) for price paths
- Regime-specific volatility and drift parameters
- Liquidity depth and volume simulation
- MEV activity simulation
- Feature engineering aligned with real data
- Label generation for rug detection, signals, and price prediction

Architecture:
1. Regime Selection: Choose market regime (BULL, BEAR, CHOP, MEV_HEAVY)
2. Price Simulation: GBM with regime-specific parameters
3. Market Features: Liquidity, volume, holder patterns
4. Risk Features: Honeypot indicators, MEV activity
5. Labels: Generate ground truth for models
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def sanitize_features(features: np.ndarray, context: str = "features") -> np.ndarray:
    """P1-4 Fix: Replace NaN/Inf with logging to detect data quality issues.

    Args:
        features: Feature array to sanitize
        context: Description of the data being sanitized for logging

    Returns:
        Sanitized array with NaN/Inf replaced by 0.0
    """
    nan_count = np.sum(np.isnan(features))
    posinf_count = np.sum(np.isposinf(features))
    neginf_count = np.sum(np.isneginf(features))
    total_invalid = nan_count + posinf_count + neginf_count

    if total_invalid > 0:
        total_elements = features.size
        pct_invalid = (total_invalid / total_elements) * 100

        # Log at WARNING for significant issues (>1% invalid), DEBUG otherwise
        log_level = logging.WARNING if pct_invalid > 1.0 else logging.DEBUG
        logger.log(
            log_level,
            "P1-4: Sanitized %s - %d NaN, %d +Inf, %d -Inf (%.2f%% of %d elements)",
            context,
            nan_count,
            posinf_count,
            neginf_count,
            pct_invalid,
            total_elements,
        )

        # Log feature-wise breakdown for significant issues
        if pct_invalid > 5.0 and features.ndim == 2:
            invalid_per_feature = np.sum(~np.isfinite(features), axis=0)
            worst_features = np.argsort(invalid_per_feature)[-3:][::-1]
            for idx in worst_features:
                if invalid_per_feature[idx] > 0:
                    logger.warning(
                        "  Feature %d: %d invalid values",
                        idx,
                        invalid_per_feature[idx],
                    )

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class SyntheticRegime(Enum):
    """Market regimes for synthetic data generation."""

    BULL = "bull"  # Strong upward momentum
    BEAR = "bear"  # Downward trend
    CHOP = "chop"  # Sideways, mean-reverting
    MEV_HEAVY = "mev_heavy"  # High MEV activity
    RUG_PULL = "rug_pull"  # Rug pull simulation


@dataclass
class RegimeParams:
    """Parameters for regime-specific simulation."""

    drift: float  # Annual drift rate
    volatility: float  # Annual volatility
    mev_intensity: float  # MEV event frequency
    liquidity_stability: float  # Liquidity change volatility
    holder_concentration: float  # Top holder percentage
    rug_probability: float  # Probability of rug pull

    # Price action characteristics
    pump_probability: float = 0.1  # Probability of pump event
    dump_probability: float = 0.05  # Probability of dump event
    pump_magnitude: float = 0.5  # Average pump size (50%)
    dump_magnitude: float = -0.3  # Average dump size (-30%)


# Regime-specific parameters calibrated from historical memecoin data
REGIME_PARAMETERS = {
    SyntheticRegime.BULL: RegimeParams(
        drift=2.5,  # Strong upward drift
        volatility=0.8,  # Moderate volatility
        mev_intensity=0.05,  # Low MEV
        liquidity_stability=0.3,  # Stable liquidity
        holder_concentration=0.35,  # Distributed holders
        rug_probability=0.02,  # Low rug risk
        pump_probability=0.15,
        pump_magnitude=0.6,
        dump_probability=0.03,
    ),
    SyntheticRegime.BEAR: RegimeParams(
        drift=-1.5,  # Downward drift
        volatility=1.2,  # High volatility
        mev_intensity=0.08,  # Elevated MEV
        liquidity_stability=0.6,  # Unstable liquidity
        holder_concentration=0.45,  # Concentrated holders
        rug_probability=0.08,  # Higher rug risk
        pump_probability=0.05,
        dump_probability=0.12,
        dump_magnitude=-0.4,
    ),
    SyntheticRegime.CHOP: RegimeParams(
        drift=0.0,  # No drift
        volatility=0.5,  # Low volatility
        mev_intensity=0.03,  # Low MEV
        liquidity_stability=0.2,  # Very stable
        holder_concentration=0.40,  # Moderate concentration
        rug_probability=0.04,  # Moderate rug risk
        pump_probability=0.08,
        pump_magnitude=0.3,
        dump_probability=0.08,
        dump_magnitude=-0.25,
    ),
    SyntheticRegime.MEV_HEAVY: RegimeParams(
        drift=0.5,  # Slight upward
        volatility=1.5,  # Very high volatility
        mev_intensity=0.25,  # Very high MEV
        liquidity_stability=0.8,  # Very unstable
        holder_concentration=0.38,  # Moderate concentration
        rug_probability=0.06,  # Elevated rug risk
        pump_probability=0.12,
        pump_magnitude=0.7,
        dump_probability=0.15,
        dump_magnitude=-0.5,
    ),
    SyntheticRegime.RUG_PULL: RegimeParams(
        drift=3.0,  # Initial pump
        volatility=0.6,  # Moderate volatility initially
        mev_intensity=0.10,  # Moderate MEV
        liquidity_stability=0.4,  # Moderate stability
        holder_concentration=0.65,  # Highly concentrated
        rug_probability=0.95,  # Near certain rug
        pump_probability=0.20,
        pump_magnitude=1.0,
        dump_probability=0.05,  # Dump comes from rug
    ),
}


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""

    # Simulation parameters
    n_samples: int = 1000  # Number of time steps (min 100 recommended)
    dt: float = 1 / 525600  # Time step (1 minute in years)
    initial_price: float = 1.0  # Starting price

    # Market parameters
    initial_liquidity: float = 50_000.0  # Initial liquidity (USD)
    initial_volume_24h: float = 100_000.0  # Initial 24h volume
    base_holders: int = 500  # Base number of holders

    # Feature engineering
    volatility_window: int = 20  # Window for volatility calculation
    momentum_window: int = 10  # Window for momentum calculation

    # Random seed for reproducibility
    seed: int | None = None


@dataclass
class SyntheticDataset:
    """Complete synthetic dataset with features and labels."""

    # Time series
    timestamps: np.ndarray  # Unix timestamps
    prices: np.ndarray  # Price path
    volumes: np.ndarray  # Trading volumes
    liquidities: np.ndarray  # Liquidity depth
    mev_counts: np.ndarray  # MEV event counts

    # Features (50-dimensional)
    features_50: np.ndarray  # Complete feature matrix

    # Labels
    signal_labels: np.ndarray  # 0=SELL, 1=HOLD, 2=BUY
    rug_labels: np.ndarray  # 0=clean, 1=rug
    future_returns_t5: np.ndarray  # Returns at T+5min
    future_returns_t10: np.ndarray  # Returns at T+10min
    future_returns_t15: np.ndarray  # Returns at T+15min

    # LSTM sequences (60 timesteps x 5 features)
    price_sequences: np.ndarray  # For LSTM training

    # Metadata
    regime: SyntheticRegime
    config: SyntheticConfig

    def __len__(self) -> int:
        return len(self.prices)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "n_samples": len(self),
            "regime": self.regime.value,
            "timestamps": self.timestamps.tolist(),
            "prices": self.prices.tolist(),
            "volumes": self.volumes.tolist(),
            "liquidities": self.liquidities.tolist(),
            "features_50_shape": self.features_50.shape,
            "signal_labels_distribution": {
                "sell": int((self.signal_labels == 0).sum()),
                "hold": int((self.signal_labels == 1).sum()),
                "buy": int((self.signal_labels == 2).sum()),
            },
            "rug_labels_distribution": {
                "clean": int((self.rug_labels == 0).sum()),
                "rug": int((self.rug_labels == 1).sum()),
            },
        }


class SyntheticDataGenerator:
    """Generate synthetic memecoin market data for training.

    Example:
        generator = SyntheticDataGenerator()
        dataset = generator.generate(
            regime=SyntheticRegime.BULL,
            n_samples=1000
        )
    """

    def __init__(self, config: SyntheticConfig | None = None):
        self.config = config or SyntheticConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

    def generate(
        self,
        regime: SyntheticRegime,
        n_samples: int | None = None,
        include_rug: bool = False,
    ) -> SyntheticDataset:
        """Generate complete synthetic dataset for a regime.

        Args:
            regime: Market regime to simulate
            n_samples: Number of time steps (overrides config)
            include_rug: Whether to simulate a rug pull event

        Returns:
            SyntheticDataset with all features and labels
        """
        n_samples = n_samples or self.config.n_samples
        params = REGIME_PARAMETERS[regime]

        logger.info(f"Generating {n_samples} samples for regime: {regime.value}")

        # Step 1: Simulate price path using GBM
        prices, timestamps = self._simulate_price_path(
            n_samples=n_samples,
            params=params,
            include_rug=include_rug or regime == SyntheticRegime.RUG_PULL,
        )

        # Step 2: Simulate market microstructure
        volumes = self._simulate_volumes(prices, params)
        liquidities = self._simulate_liquidity(prices, params)
        mev_counts = self._simulate_mev(n_samples, params)

        # Step 3: Engineer features (50-dimensional)
        features_50 = self._engineer_features(
            prices=prices,
            volumes=volumes,
            liquidities=liquidities,
            mev_counts=mev_counts,
            params=params,
        )

        # Step 4: Generate labels
        signal_labels = self._generate_signal_labels(prices)
        rug_labels = self._generate_rug_labels(
            prices=prices,
            features=features_50,
            params=params,
            forced_rug=include_rug or regime == SyntheticRegime.RUG_PULL,
        )

        # Step 5: Calculate future returns
        future_returns_t5 = self._calculate_future_returns(prices, horizon=5)
        future_returns_t10 = self._calculate_future_returns(prices, horizon=10)
        future_returns_t15 = self._calculate_future_returns(prices, horizon=15)

        # Step 6: Create LSTM sequences
        price_sequences = self._create_lstm_sequences(
            prices=prices,
            volumes=volumes,
            liquidities=liquidities,
        )

        dataset = SyntheticDataset(
            timestamps=timestamps,
            prices=prices,
            volumes=volumes,
            liquidities=liquidities,
            mev_counts=mev_counts,
            features_50=features_50,
            signal_labels=signal_labels,
            rug_labels=rug_labels,
            future_returns_t5=future_returns_t5,
            future_returns_t10=future_returns_t10,
            future_returns_t15=future_returns_t15,
            price_sequences=price_sequences,
            regime=regime,
            config=self.config,
        )

        logger.info(f"Generated dataset: {dataset.to_dict()}")
        return dataset

    def generate_mixed_regimes(
        self,
        regime_distribution: dict[SyntheticRegime, float],
        total_samples: int,
    ) -> SyntheticDataset:
        """Generate dataset with mixed regimes.

        Args:
            regime_distribution: Dict mapping regime to fraction (should sum to 1.0)
            total_samples: Total number of samples to generate

        Returns:
            Combined SyntheticDataset
        """
        datasets = []

        for regime, fraction in regime_distribution.items():
            n_samples = int(total_samples * fraction)
            if n_samples > 0:
                dataset = self.generate(regime=regime, n_samples=n_samples)
                datasets.append(dataset)

        # Combine datasets
        return self._combine_datasets(datasets)

    def _simulate_price_path(
        self,
        n_samples: int,
        params: RegimeParams,
        include_rug: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate price path using Geometric Brownian Motion with jumps.

        dS/S = μ dt + σ dW + J
        where J represents jump events (pumps/dumps)
        """
        dt = self.config.dt
        prices = np.zeros(n_samples)
        prices[0] = self.config.initial_price

        # Time in minutes
        timestamps = np.arange(n_samples) * 60  # 1 minute intervals

        # Rug pull detection point (if applicable)
        rug_point = None
        if include_rug and np.random.random() < params.rug_probability:
            rug_point = int(n_samples * np.random.uniform(0.3, 0.8))

        for i in range(1, n_samples):
            # Base GBM
            drift = params.drift
            vol = params.volatility

            # Adjust parameters near rug pull
            if rug_point and i >= rug_point - 50:
                if i < rug_point:
                    # Pump before rug
                    drift = params.drift * 2.0
                    vol = params.volatility * 0.7
                else:
                    # Crash after rug
                    drift = -5.0
                    vol = params.volatility * 2.0

            # Brownian motion
            dW = np.random.normal(0, np.sqrt(dt))
            dS = prices[i - 1] * (drift * dt + vol * dW)

            # Add jump events (pumps/dumps) - scaled properly for minute-level
            # Probability per minute should be ~0.1% to 1%, not 15%
            pump_prob_per_min = params.pump_probability / 1000.0  # Scale down
            dump_prob_per_min = params.dump_probability / 1000.0

            if np.random.random() < pump_prob_per_min:
                jump = prices[i - 1] * params.pump_magnitude * 0.1 * np.random.uniform(0.5, 1.5)
                dS += jump
            elif np.random.random() < dump_prob_per_min:
                jump = prices[i - 1] * params.dump_magnitude * 0.1 * np.random.uniform(0.5, 1.5)
                dS += jump

            prices[i] = max(prices[i - 1] + dS, 0.001)  # Prevent negative prices

        return prices, timestamps

    def _simulate_volumes(
        self,
        prices: np.ndarray,
        params: RegimeParams,
    ) -> np.ndarray:
        """Simulate trading volumes correlated with price volatility."""
        n = len(prices)
        base_volume = self.config.initial_volume_24h / (24 * 60)  # Per minute

        # Volume increases with volatility
        returns = np.diff(prices) / prices[:-1]
        returns = np.insert(returns, 0, 0)
        volatility = np.abs(returns)

        # Add autocorrelation
        volumes = np.zeros(n)
        volumes[0] = base_volume

        for i in range(1, n):
            # Base volume with volatility scaling
            vol_scale = 1.0 + 5.0 * volatility[i]

            # AR(1) process for smoothness
            volumes[i] = 0.7 * volumes[i - 1] + 0.3 * base_volume * vol_scale

            # Add noise
            volumes[i] *= np.random.lognormal(0, 0.3)

        return volumes

    def _simulate_liquidity(
        self,
        prices: np.ndarray,
        params: RegimeParams,
    ) -> np.ndarray:
        """Simulate liquidity depth with regime-specific stability."""
        n = len(prices)
        base_liq = self.config.initial_liquidity

        liquidities = np.zeros(n)
        liquidities[0] = base_liq

        for i in range(1, n):
            # Random walk with mean reversion
            change = np.random.normal(0, params.liquidity_stability * 0.01)
            mean_reversion = 0.1 * (base_liq - liquidities[i - 1]) / base_liq

            liquidities[i] = liquidities[i - 1] * (1 + change + mean_reversion)
            liquidities[i] = max(liquidities[i], base_liq * 0.1)  # Floor

        return liquidities

    def _simulate_mev(
        self,
        n_samples: int,
        params: RegimeParams,
    ) -> np.ndarray:
        """Simulate MEV events (sandwich attacks, front-running)."""
        # Poisson process for MEV events
        mev_counts = np.random.poisson(params.mev_intensity, n_samples)
        return mev_counts.astype(float)

    def _engineer_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        liquidities: np.ndarray,
        mev_counts: np.ndarray,
        params: RegimeParams,
    ) -> np.ndarray:
        """Engineer 50-dimensional feature vectors.

        Feature breakdown:
        - Liquidity (0-11): 12 features
        - Holder (12-22): 11 features
        - Price (23-32): 10 features
        - Risk (33-42): 10 features
        - MEV (43-49): 7 features
        """
        n = len(prices)
        features = np.zeros((n, 50))

        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        returns = np.insert(returns, 0, 0)

        for i in range(max(self.config.volatility_window, 1), n):
            # === LIQUIDITY FEATURES (0-11) ===
            features[i, 0] = liquidities[i]  # Current liquidity
            features[i, 1] = np.mean(liquidities[max(0, i - 20) : i])  # Avg liquidity
            features[i, 2] = np.std(liquidities[max(0, i - 20) : i])  # Liquidity volatility
            features[i, 3] = liquidities[i] / (
                np.mean(liquidities[max(0, i - 60) : i]) + 1
            )  # Relative
            features[i, 4] = np.max(liquidities[max(0, i - 60) : i])  # Peak liquidity
            features[i, 5] = np.min(liquidities[max(0, i - 60) : i])  # Min liquidity
            features[i, 6] = (liquidities[i] - liquidities[i - 1]) / (
                liquidities[i - 1] + 1
            )  # Change
            features[i, 7] = volumes[i] / (liquidities[i] + 1)  # Volume/Liquidity ratio
            features[i, 8] = np.percentile(liquidities[max(0, i - 100) : i], 25)  # P25
            features[i, 9] = np.percentile(liquidities[max(0, i - 100) : i], 75)  # P75
            features[i, 10] = np.sum(liquidities[max(0, i - 10) : i] < liquidities[i])  # Rank
            features[i, 11] = (
                np.corrcoef(prices[max(0, i - 30) : i], liquidities[max(0, i - 30) : i])[0, 1]
                if i >= 30
                else 0
            )

            # === HOLDER FEATURES (12-22) ===
            # Simulated based on regime parameters
            features[i, 12] = self.config.base_holders + np.random.randint(-50, 50)  # Holder count
            features[i, 13] = params.holder_concentration  # Top holder %
            features[i, 14] = params.holder_concentration * 0.8  # Top 3 %
            features[i, 15] = params.holder_concentration * 0.5  # Top 10 %
            features[i, 16] = np.random.uniform(0.7, 0.95)  # Gini coefficient
            features[i, 17] = np.random.uniform(0.05, 0.15)  # New holders rate
            features[i, 18] = np.random.uniform(0.02, 0.10)  # Exit rate
            features[i, 19] = np.random.uniform(50, 200)  # Avg holding time
            features[i, 20] = np.random.randint(5, 20)  # Whale count
            features[i, 21] = np.random.uniform(0.3, 0.7)  # Whale activity
            features[i, 22] = np.random.uniform(0.1, 0.4)  # Smart money %

            # === PRICE FEATURES (23-32) ===
            window = min(i, self.config.volatility_window)
            features[i, 23] = np.std(returns[i - window : i])  # Volatility
            features[i, 24] = np.mean(returns[i - window : i])  # Mean return
            features[i, 25] = (
                (prices[i] - prices[i - 10]) / prices[i - 10] if i >= 10 else 0
            )  # Momentum
            features[i, 26] = (
                np.max(prices[max(0, i - 60) : i]) / prices[i] - 1
            )  # Distance from peak
            features[i, 27] = (
                prices[i] / np.min(prices[max(0, i - 60) : i]) - 1
            )  # Distance from trough
            features[i, 28] = np.mean(prices[max(0, i - 20) : i])  # SMA20
            features[i, 29] = np.mean(prices[max(0, i - 50) : i])  # SMA50
            features[i, 30] = (
                features[i, 28] / features[i, 29] - 1 if features[i, 29] > 0 else 0
            )  # SMA crossover
            features[i, 31] = (
                (np.sum(returns[max(0, i - 20) : i] > 0) / 20.0) if i >= 20 else 0.5
            )  # Win rate
            features[i, 32] = (
                np.percentile(returns[max(0, i - 100) : i], 95) if i >= 100 else returns[i]
            )  # 95th percentile

            # === RISK FEATURES (33-42) ===
            features[i, 33] = params.rug_probability  # Base rug risk
            features[i, 34] = np.random.uniform(0, 0.5)  # Honeypot score
            features[i, 35] = np.random.uniform(0, 1)  # Contract verified
            features[i, 36] = np.random.uniform(0, 1)  # LP locked
            features[i, 37] = np.random.uniform(0, 100)  # Days until unlock
            features[i, 38] = np.random.uniform(0, 1)  # Mint authority
            features[i, 39] = np.random.uniform(0, 1)  # Freeze authority
            features[i, 40] = params.holder_concentration  # Centralization risk
            features[i, 41] = np.sum(mev_counts[max(0, i - 60) : i]) / 60.0  # MEV rate
            features[i, 42] = np.random.uniform(0, 1)  # Social sentiment

            # === MEV FEATURES (43-49) ===
            features[i, 43] = mev_counts[i]  # Current MEV count
            features[i, 44] = np.mean(mev_counts[max(0, i - 20) : i])  # Avg MEV
            features[i, 45] = np.max(mev_counts[max(0, i - 60) : i])  # Max MEV
            features[i, 46] = np.sum(mev_counts[max(0, i - 10) : i])  # Recent MEV
            features[i, 47] = params.mev_intensity  # MEV intensity
            features[i, 48] = np.random.uniform(0, 1)  # Sandwich attack rate
            features[i, 49] = np.random.uniform(0, 1)  # Front-run rate

        # P1-4 Fix: Replace NaN/Inf with zeros + logging
        features = sanitize_features(features, context="synthetic_features")

        return features

    def _generate_signal_labels(
        self,
        prices: np.ndarray,
        buy_threshold: float = 0.0033,  # +0.33% in 15 min (75th percentile)
        sell_threshold: float = -0.0024,  # -0.24% in 15 min (25th percentile)
    ) -> np.ndarray:
        """Generate signal labels based on future returns.

        Returns:
            Array of labels (0=SELL, 1=HOLD, 2=BUY)
        """
        n = len(prices)
        labels = np.ones(n, dtype=int)  # Default to HOLD

        for i in range(n - 15):
            future_return = (prices[i + 15] - prices[i]) / prices[i]  # Decimal return

            if future_return >= buy_threshold:
                labels[i] = 2  # BUY
            elif future_return <= sell_threshold:
                labels[i] = 0  # SELL

        return labels

    def _generate_rug_labels(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        params: RegimeParams,
        forced_rug: bool = False,
    ) -> np.ndarray:
        """Generate rug pull labels based on probability and price action.

        Returns:
            Binary array (0=clean, 1=rug)
        """
        n = len(prices)
        labels = np.zeros(n, dtype=int)

        rug_prob = params.rug_probability

        # Use probability-based assignment for realistic rug rates
        # Higher concentration = higher rug probability
        concentration_boost = params.holder_concentration * 0.5

        for i in range(n):
            # Sample-based rug assignment
            if np.random.random() < (rug_prob + concentration_boost) * 0.1:
                labels[i] = 1

        # For forced rug scenario, also mark after large drops
        if forced_rug:
            returns = np.diff(prices) / prices[:-1]
            returns = np.insert(returns, 0, 0)

            for i in range(10, n):
                drop_10min = (prices[i] - prices[i - 10]) / prices[i - 10]
                if drop_10min < -0.30:
                    labels[i:] = 1
                    break

        return labels

    def _calculate_future_returns(
        self,
        prices: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """Calculate future returns at given horizon.

        Returns decimal values (e.g., 0.05 = 5% return).
        """
        n = len(prices)
        returns = np.zeros(n)

        for i in range(n - horizon):
            returns[i] = (prices[i + horizon] - prices[i]) / prices[i]

        return returns

    def _create_lstm_sequences(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        liquidities: np.ndarray,
        sequence_length: int = 60,
    ) -> np.ndarray:
        """Create LSTM input sequences (60 timesteps x 5 features).

        Features:
        - Normalized price
        - Returns
        - Volume (log-scaled)
        - Liquidity (log-scaled)
        - Volatility (rolling std)
        """
        n = len(prices)

        # Handle case where n_samples < sequence_length
        if n < sequence_length:
            logger.warning(
                f"Not enough samples ({n}) for sequence length ({sequence_length}). "
                f"Returning empty array."
            )
            return np.zeros((0, sequence_length, 5))

        n_sequences = n - sequence_length + 1
        sequences = np.zeros((n_sequences, sequence_length, 5))

        # Calculate features
        returns = np.diff(prices) / prices[:-1]
        returns = np.insert(returns, 0, 0)

        for i in range(n_sequences):
            seq_prices = prices[i : i + sequence_length]
            seq_volumes = volumes[i : i + sequence_length]
            seq_liquidities = liquidities[i : i + sequence_length]
            seq_returns = returns[i : i + sequence_length]

            # Normalize price (z-score)
            price_mean = np.mean(seq_prices)
            price_std = np.std(seq_prices) + 1e-8
            sequences[i, :, 0] = (seq_prices - price_mean) / price_std

            # Returns
            sequences[i, :, 1] = seq_returns

            # Log-scaled volume
            sequences[i, :, 2] = np.log1p(seq_volumes)

            # Log-scaled liquidity
            sequences[i, :, 3] = np.log1p(seq_liquidities)

            # Rolling volatility
            for j in range(sequence_length):
                if j >= 10:
                    sequences[i, j, 4] = np.std(seq_returns[j - 10 : j])
                else:
                    sequences[i, j, 4] = 0

        return sequences

    def _combine_datasets(
        self,
        datasets: list[SyntheticDataset],
    ) -> SyntheticDataset:
        """Combine multiple datasets into one."""
        if len(datasets) == 1:
            return datasets[0]

        # Concatenate all arrays
        combined = SyntheticDataset(
            timestamps=np.concatenate([d.timestamps for d in datasets]),
            prices=np.concatenate([d.prices for d in datasets]),
            volumes=np.concatenate([d.volumes for d in datasets]),
            liquidities=np.concatenate([d.liquidities for d in datasets]),
            mev_counts=np.concatenate([d.mev_counts for d in datasets]),
            features_50=np.vstack([d.features_50 for d in datasets]),
            signal_labels=np.concatenate([d.signal_labels for d in datasets]),
            rug_labels=np.concatenate([d.rug_labels for d in datasets]),
            future_returns_t5=np.concatenate([d.future_returns_t5 for d in datasets]),
            future_returns_t10=np.concatenate([d.future_returns_t10 for d in datasets]),
            future_returns_t15=np.concatenate([d.future_returns_t15 for d in datasets]),
            price_sequences=np.vstack([d.price_sequences for d in datasets]),
            regime=SyntheticRegime.CHOP,  # Mixed regime
            config=datasets[0].config,
        )

        return combined


def generate_training_dataset(
    n_samples_per_regime: int = 2000,
    regimes: list[SyntheticRegime] | None = None,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate balanced training dataset across multiple regimes.

    Args:
        n_samples_per_regime: Samples to generate per regime
        regimes: List of regimes (default: BULL, BEAR, CHOP, MEV_HEAVY)
        seed: Random seed

    Returns:
        Combined SyntheticDataset
    """
    if regimes is None:
        regimes = [
            SyntheticRegime.BULL,
            SyntheticRegime.BEAR,
            SyntheticRegime.CHOP,
            SyntheticRegime.MEV_HEAVY,
        ]

    config = SyntheticConfig(seed=seed)
    generator = SyntheticDataGenerator(config)

    datasets = []
    for regime in regimes:
        logger.info(f"Generating {n_samples_per_regime} samples for {regime.value}")
        dataset = generator.generate(regime=regime, n_samples=n_samples_per_regime)
        datasets.append(dataset)

    combined = generator._combine_datasets(datasets)
    logger.info(f"Combined dataset: {len(combined)} total samples")

    return combined
