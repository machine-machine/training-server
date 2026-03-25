"""
Temporal Synthetic Data Generation for LSTM Training.

Generates synthetic token lifecycle sequences for training temporal models.
Each lifecycle represents a realistic evolution of token metrics over time,
enabling the model to learn patterns that precede rugs/honeypots.

Lifecycle Types:
- LEGITIMATE: Normal token growth and maturation
- SLOW_RUG: Gradual extraction over hours/days
- FAST_RUG: Rapid dump within minutes
- HONEYPOT_EVOLUTION: Initially tradeable, then blocked
- PUMP_AND_DUMP: Artificial pump followed by dump
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class TokenLifecycleType(Enum):
    """Types of token lifecycles for synthetic generation."""

    LEGITIMATE = "legitimate"
    SLOW_RUG = "slow_rug"
    FAST_RUG = "fast_rug"
    HONEYPOT_EVOLUTION = "honeypot_evolution"
    PUMP_AND_DUMP = "pump_and_dump"
    ORGANIC_GROWTH = "organic_growth"
    WHALE_MANIPULATION = "whale_manipulation"


@dataclass
class LifecyclePhase:
    """A phase within a token lifecycle."""

    name: str
    duration_hours_range: tuple[float, float]
    # feature_name -> (multiplier_min, multiplier_max) or absolute (min, max)
    feature_modifiers: dict[str, tuple[float, float]]
    # Whether modifiers are multipliers (True) or absolute values (False)
    is_multiplier: bool = True


# Regime-specific adjustments for synthetic generation
REGIME_ADJUSTMENTS = {
    "MEME_SEASON": {
        "volume_multiplier": 3.0,
        "volatility_multiplier": 2.0,
        "holder_growth_rate": 1.5,
    },
    "BEAR_VOLATILE": {
        "volume_multiplier": 0.5,
        "volatility_multiplier": 1.5,
        "holder_growth_rate": 0.3,
    },
    "NORMAL": {
        "volume_multiplier": 1.0,
        "volatility_multiplier": 1.0,
        "holder_growth_rate": 1.0,
    },
    "LOW_LIQUIDITY": {
        "volume_multiplier": 0.3,
        "volatility_multiplier": 0.8,
        "holder_growth_rate": 0.5,
    },
}


# Lifecycle templates defining how different token types evolve
LIFECYCLE_TEMPLATES: dict[TokenLifecycleType, list[LifecyclePhase]] = {
    TokenLifecycleType.LEGITIMATE: [
        LifecyclePhase(
            name="launch",
            duration_hours_range=(0.1, 1.0),
            feature_modifiers={
                "volume_24h": (0.5, 2.0),
                "holder_count": (10, 50),
                "price_change_1h": (-10, 50),
                "liquidity_usd": (5000, 50000),
            },
            is_multiplier=False,
        ),
        LifecyclePhase(
            name="growth",
            duration_hours_range=(6, 48),
            feature_modifiers={
                "volume_24h": (1.0, 5.0),
                "holder_count": (1.5, 3.0),
                "price_change_1h": (-5, 20),
                "liquidity_usd": (1.2, 2.5),
            },
        ),
        LifecyclePhase(
            name="mature",
            duration_hours_range=(48, 168),
            feature_modifiers={
                "volume_24h": (0.8, 1.2),
                "holder_count": (1.0, 1.2),
                "price_change_1h": (-3, 3),
                "liquidity_usd": (0.9, 1.1),
            },
        ),
    ],
    TokenLifecycleType.HONEYPOT_EVOLUTION: [
        LifecyclePhase(
            name="open_trading",
            duration_hours_range=(0.1, 0.5),
            feature_modifiers={
                "volume_24h": (2.0, 10.0),
                "holder_count": (50, 200),
                "buy_sell_ratio_5min": (5.0, 20.0),  # Many buys, few sells
                "price_change_1h": (50, 200),
                "unique_sellers_5min": (0.1, 0.3),  # Very few sellers
            },
            is_multiplier=False,
        ),
        LifecyclePhase(
            name="restriction_active",
            duration_hours_range=(0.5, 2.0),
            feature_modifiers={
                "volume_24h": (0.1, 0.3),  # Volume crashes
                "unique_sellers_5min": (0, 2),  # Almost no sellers
                "buy_sell_ratio_5min": (50, 1000),  # Extreme ratio
                "price_change_1h": (-50, -90),
            },
            is_multiplier=False,
        ),
        LifecyclePhase(
            name="locked",
            duration_hours_range=(2.0, 24.0),
            feature_modifiers={
                "volume_24h": (0, 100),
                "unique_sellers_5min": (0, 0),
                "lp_removal_attempts": (1, 10),
            },
            is_multiplier=False,
        ),
    ],
    TokenLifecycleType.FAST_RUG: [
        LifecyclePhase(
            name="accumulation",
            duration_hours_range=(0.05, 0.2),  # 3-12 minutes
            feature_modifiers={
                "volume_24h": (5.0, 20.0),
                "holder_count": (20, 100),
                "price_change_1h": (100, 500),
                "buy_sell_ratio_5min": (3.0, 10.0),
            },
            is_multiplier=False,
        ),
        LifecyclePhase(
            name="dump",
            duration_hours_range=(0.01, 0.05),  # 30 seconds to 3 minutes
            feature_modifiers={
                "volume_24h": (0.5, 1.0),
                "price_change_1h": (-80, -99),
                "liquidity_usd": (0.01, 0.1),  # 90-99% drained
                "buy_sell_ratio_5min": (0.01, 0.1),
            },
        ),
        LifecyclePhase(
            name="dead",
            duration_hours_range=(1.0, 24.0),
            feature_modifiers={
                "volume_24h": (0, 10),
                "holder_count": (0.9, 1.0),  # Holders stuck
                "liquidity_usd": (0, 100),
            },
            is_multiplier=False,
        ),
    ],
    TokenLifecycleType.SLOW_RUG: [
        LifecyclePhase(
            name="build_trust",
            duration_hours_range=(24, 72),
            feature_modifiers={
                "volume_24h": (1.0, 3.0),
                "holder_count": (1.2, 2.0),
                "price_change_1h": (-5, 15),
                "liquidity_usd": (1.0, 1.5),
            },
        ),
        LifecyclePhase(
            name="extraction",
            duration_hours_range=(12, 48),
            feature_modifiers={
                "volume_24h": (0.5, 0.8),
                "price_change_1h": (-10, -30),
                "liquidity_usd": (0.7, 0.9),  # Gradual drain
                "top_holder_pct": (1.1, 1.3),  # Concentration increasing
            },
        ),
        LifecyclePhase(
            name="final_exit",
            duration_hours_range=(0.5, 2.0),
            feature_modifiers={
                "volume_24h": (0.1, 0.3),
                "price_change_1h": (-50, -80),
                "liquidity_usd": (0.05, 0.2),
            },
        ),
    ],
    TokenLifecycleType.PUMP_AND_DUMP: [
        LifecyclePhase(
            name="accumulation",
            duration_hours_range=(1, 6),
            feature_modifiers={
                "volume_24h": (0.5, 1.5),
                "holder_count": (1.0, 1.2),
                "price_change_1h": (-2, 5),
            },
        ),
        LifecyclePhase(
            name="pump",
            duration_hours_range=(0.5, 2.0),
            feature_modifiers={
                "volume_24h": (5.0, 20.0),
                "holder_count": (2.0, 5.0),
                "price_change_1h": (50, 300),
                "buy_sell_ratio_5min": (3.0, 10.0),
            },
        ),
        LifecyclePhase(
            name="dump",
            duration_hours_range=(0.1, 0.5),
            feature_modifiers={
                "volume_24h": (2.0, 5.0),
                "price_change_1h": (-60, -90),
                "buy_sell_ratio_5min": (0.1, 0.3),
                "large_sell_count": (5, 20),
            },
            is_multiplier=False,
        ),
        LifecyclePhase(
            name="aftermath",
            duration_hours_range=(6, 24),
            feature_modifiers={
                "volume_24h": (0.1, 0.3),
                "price_change_1h": (-5, 5),
                "holder_count": (0.8, 0.95),  # Some exit
            },
        ),
    ],
    TokenLifecycleType.ORGANIC_GROWTH: [
        LifecyclePhase(
            name="discovery",
            duration_hours_range=(6, 24),
            feature_modifiers={
                "volume_24h": (1.0, 3.0),
                "holder_count": (1.2, 1.8),
                "price_change_1h": (-5, 20),
            },
        ),
        LifecyclePhase(
            name="adoption",
            duration_hours_range=(24, 72),
            feature_modifiers={
                "volume_24h": (1.5, 4.0),
                "holder_count": (1.5, 3.0),
                "price_change_1h": (0, 30),
                "liquidity_usd": (1.2, 2.0),
            },
        ),
        LifecyclePhase(
            name="stabilization",
            duration_hours_range=(72, 168),
            feature_modifiers={
                "volume_24h": (0.6, 1.0),
                "holder_count": (1.0, 1.1),
                "price_change_1h": (-3, 3),
            },
        ),
    ],
    TokenLifecycleType.WHALE_MANIPULATION: [
        LifecyclePhase(
            name="quiet_accumulation",
            duration_hours_range=(2, 12),
            feature_modifiers={
                "volume_24h": (0.8, 1.2),
                "top_holder_pct": (1.2, 1.5),  # Whale accumulating
                "price_change_1h": (-2, 5),
            },
        ),
        LifecyclePhase(
            name="price_spike",
            duration_hours_range=(0.2, 1.0),
            feature_modifiers={
                "volume_24h": (3.0, 10.0),
                "price_change_1h": (30, 100),
                "buy_sell_ratio_5min": (2.0, 5.0),
            },
        ),
        LifecyclePhase(
            name="whale_exit",
            duration_hours_range=(0.1, 0.5),
            feature_modifiers={
                "volume_24h": (2.0, 5.0),
                "price_change_1h": (-30, -60),
                "top_holder_pct": (0.5, 0.7),  # Whale sold
                "large_sell_count": (1, 5),
            },
            is_multiplier=False,
        ),
        LifecyclePhase(
            name="recovery_or_death",
            duration_hours_range=(6, 48),
            feature_modifiers={
                "volume_24h": (0.3, 0.8),
                "price_change_1h": (-10, 10),
            },
        ),
    ],
}


# Feature name to index mapping (based on fraud_model.py feature order)
FEATURE_INDEX_MAP = {
    "liquidity_usd": 0,
    "fdv_usd": 1,
    "holder_count": 2,
    "top_holder_pct": 3,
    "mint_authority_enabled": 4,
    "freeze_authority_enabled": 5,
    "lp_burn_pct": 6,
    "age_seconds": 7,
    "volume_24h": 8,
    "price_change_1h": 9,
    # Extended features (if using EnhancedTokenFeatures)
    "buy_sell_ratio_5min": 10,
    "unique_sellers_5min": 11,
    "large_sell_count": 12,
    "lp_removal_attempts": 13,
}


class TemporalSyntheticGenerator:
    """Generate synthetic token lifecycle sequences for LSTM training.

    Produces sequences of feature vectors representing token evolution
    over time, labeled as safe (0) or malicious (1).

    Example:
        generator = TemporalSyntheticGenerator(base_features)
        X_seq, y = generator.generate_batch(
            TokenLifecycleType.HONEYPOT_EVOLUTION,
            n_samples=100,
            regime="MEME_SEASON",
        )
        # X_seq.shape = (100, 60, n_features)
        # y.shape = (100,)
    """

    def __init__(
        self,
        base_features: np.ndarray,
        n_timesteps: int = 60,
        timestep_minutes: int = 5,
        random_state: int = 42,
    ):
        """Initialize the generator.

        Args:
            base_features: Reference real data for distribution estimation
                           Shape: (n_samples, n_features)
            n_timesteps: Number of timesteps per sequence
            timestep_minutes: Minutes per timestep
            random_state: Random seed for reproducibility
        """
        self.base_features = base_features
        self.n_timesteps = n_timesteps
        self.timestep_minutes = timestep_minutes
        self.random_state = random_state

        # Compute feature statistics from real data
        self.feature_means = np.mean(base_features, axis=0)
        self.feature_stds = np.std(base_features, axis=0)
        self.feature_mins = np.min(base_features, axis=0)
        self.feature_maxs = np.max(base_features, axis=0)

        self.n_features = base_features.shape[1]

        np.random.seed(random_state)

    def generate_batch(
        self,
        lifecycle_type: TokenLifecycleType,
        n_samples: int,
        regime: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic lifecycle sequences.

        Args:
            lifecycle_type: Type of lifecycle to generate
            n_samples: Number of sequences to generate
            regime: Optional market regime for conditioning

        Returns:
            sequences: Shape (n_samples, n_timesteps, n_features)
            labels: Shape (n_samples,) - 0=safe, 1=rug/honeypot
        """
        sequences = []
        labels = []

        template = LIFECYCLE_TEMPLATES.get(lifecycle_type)
        if template is None:
            logger.warning(f"Unknown lifecycle type: {lifecycle_type}, using LEGITIMATE")
            template = LIFECYCLE_TEMPLATES[TokenLifecycleType.LEGITIMATE]

        is_malicious = lifecycle_type in [
            TokenLifecycleType.SLOW_RUG,
            TokenLifecycleType.FAST_RUG,
            TokenLifecycleType.HONEYPOT_EVOLUTION,
            TokenLifecycleType.PUMP_AND_DUMP,
            TokenLifecycleType.WHALE_MANIPULATION,
        ]

        for _ in range(n_samples):
            sequence = self._generate_single_sequence(template, regime)
            sequences.append(sequence)
            labels.append(1 if is_malicious else 0)

        return np.array(sequences), np.array(labels)

    def generate_mixed_batch(
        self,
        n_samples: int,
        malicious_ratio: float = 0.5,
        regime: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a mixed batch of safe and malicious sequences.

        Args:
            n_samples: Total number of sequences
            malicious_ratio: Fraction of malicious samples
            regime: Optional market regime

        Returns:
            sequences: Shape (n_samples, n_timesteps, n_features)
            labels: Shape (n_samples,)
        """
        n_malicious = int(n_samples * malicious_ratio)
        n_safe = n_samples - n_malicious

        all_sequences = []
        all_labels = []

        # Generate safe samples
        safe_types = [TokenLifecycleType.LEGITIMATE, TokenLifecycleType.ORGANIC_GROWTH]
        for i in range(n_safe):
            lifecycle = safe_types[i % len(safe_types)]
            seq, lab = self.generate_batch(lifecycle, 1, regime)
            all_sequences.append(seq[0])
            all_labels.append(lab[0])

        # Generate malicious samples
        malicious_types = [
            TokenLifecycleType.FAST_RUG,
            TokenLifecycleType.SLOW_RUG,
            TokenLifecycleType.HONEYPOT_EVOLUTION,
            TokenLifecycleType.PUMP_AND_DUMP,
        ]
        for i in range(n_malicious):
            lifecycle = malicious_types[i % len(malicious_types)]
            seq, lab = self.generate_batch(lifecycle, 1, regime)
            all_sequences.append(seq[0])
            all_labels.append(lab[0])

        # Shuffle
        indices = np.random.permutation(n_samples)
        sequences = np.array(all_sequences)[indices]
        labels = np.array(all_labels)[indices]

        return sequences, labels

    def _generate_single_sequence(
        self,
        phases: list[LifecyclePhase],
        regime: str | None,
    ) -> np.ndarray:
        """Generate a single lifecycle sequence.

        Args:
            phases: List of lifecycle phases
            regime: Optional market regime

        Returns:
            Sequence of shape (n_timesteps, n_features)
        """
        sequence = np.zeros((self.n_timesteps, self.n_features))

        # Get regime adjustments
        regime_adj = REGIME_ADJUSTMENTS.get(regime, REGIME_ADJUSTMENTS["NORMAL"])

        # Initialize with base features (random sample from real data)
        base_idx = np.random.randint(len(self.base_features))
        current_features = self.base_features[base_idx].copy()

        current_timestep = 0

        for phase in phases:
            if current_timestep >= self.n_timesteps:
                break

            # Determine phase duration in timesteps
            duration_hours = np.random.uniform(*phase.duration_hours_range)
            duration_timesteps = int(duration_hours * 60 / self.timestep_minutes)
            duration_timesteps = max(
                1, min(duration_timesteps, self.n_timesteps - current_timestep)
            )

            # Generate features for each timestep in this phase
            for t in range(current_timestep, current_timestep + duration_timesteps):
                # Start from current features
                features = current_features.copy()

                # Apply phase modifiers
                for feat_name, (mod_min, mod_max) in phase.feature_modifiers.items():
                    feat_idx = self._get_feature_index(feat_name)
                    if feat_idx is None or feat_idx >= self.n_features:
                        continue

                    # Calculate progress through phase (0 to 1)
                    progress = (t - current_timestep) / max(1, duration_timesteps - 1)

                    if phase.is_multiplier:
                        # Interpolate multiplier based on progress
                        multiplier = mod_min + (mod_max - mod_min) * progress
                        # Apply regime adjustment for volume
                        if feat_name == "volume_24h":
                            multiplier *= regime_adj.get("volume_multiplier", 1.0)
                        features[feat_idx] *= multiplier
                    else:
                        # Absolute value - interpolate
                        value = mod_min + (mod_max - mod_min) * progress
                        # Apply regime adjustment
                        if feat_name == "volume_24h":
                            value *= regime_adj.get("volume_multiplier", 1.0)
                        features[feat_idx] = value

                # Add noise (scaled by feature std)
                noise_scale = 0.05  # 5% noise
                noise = np.random.normal(0, self.feature_stds * noise_scale)
                features += noise

                # Clip to reasonable bounds
                features = np.clip(features, self.feature_mins * 0.5, self.feature_maxs * 2.0)

                # Update age_seconds if present
                age_idx = self._get_feature_index("age_seconds")
                if age_idx is not None and age_idx < self.n_features:
                    features[age_idx] = t * self.timestep_minutes * 60

                sequence[t] = features

            # Update current features for next phase
            if current_timestep + duration_timesteps > 0:
                current_features = sequence[current_timestep + duration_timesteps - 1].copy()

            current_timestep += duration_timesteps

        # Fill remaining timesteps with last state (if any)
        if current_timestep < self.n_timesteps and current_timestep > 0:
            sequence[current_timestep:] = sequence[current_timestep - 1]

        return sequence

    def _get_feature_index(self, name: str) -> int | None:
        """Map feature name to index."""
        return FEATURE_INDEX_MAP.get(name)

    def get_flattened_samples(
        self,
        lifecycle_type: TokenLifecycleType,
        n_samples: int,
        regime: str | None = None,
        timestep: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get flattened (non-sequential) samples for non-LSTM models.

        Takes a specific timestep from each sequence (default: last).

        Args:
            lifecycle_type: Type of lifecycle
            n_samples: Number of samples
            regime: Optional market regime
            timestep: Which timestep to extract (-1 for last)

        Returns:
            X: Shape (n_samples, n_features)
            y: Shape (n_samples,)
        """
        sequences, labels = self.generate_batch(lifecycle_type, n_samples, regime)
        X = sequences[:, timestep, :]
        return X, labels

    def augment_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment_ratio: float = 0.3,
        regime: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Augment existing dataset with synthetic samples.

        Args:
            X: Original features (n_samples, n_features)
            y: Original labels
            augment_ratio: Fraction of synthetic samples to add
            regime: Optional market regime

        Returns:
            Augmented X and y
        """
        n_synthetic = int(len(y) * augment_ratio)

        # Generate synthetic samples (mixed)
        temp_generator = TemporalSyntheticGenerator(
            X, self.n_timesteps, self.timestep_minutes, self.random_state + 1
        )

        synthetic_sequences, synthetic_labels = temp_generator.generate_mixed_batch(
            n_synthetic, malicious_ratio=0.6, regime=regime
        )

        # Flatten to last timestep
        synthetic_X = synthetic_sequences[:, -1, :]

        # Combine
        X_aug = np.vstack([X, synthetic_X])
        y_aug = np.concatenate([y, synthetic_labels])

        logger.info(
            f"Temporal augmentation: {len(y)} -> {len(y_aug)} samples "
            f"(+{n_synthetic} synthetic, regime={regime})"
        )

        return X_aug, y_aug


def create_temporal_generator(
    reference_data: np.ndarray,
    n_timesteps: int = 60,
    timestep_minutes: int = 5,
) -> TemporalSyntheticGenerator:
    """Factory function to create a temporal generator.

    Args:
        reference_data: Real data for distribution estimation
        n_timesteps: Timesteps per sequence
        timestep_minutes: Minutes per timestep

    Returns:
        Configured TemporalSyntheticGenerator
    """
    return TemporalSyntheticGenerator(
        base_features=reference_data,
        n_timesteps=n_timesteps,
        timestep_minutes=timestep_minutes,
    )
