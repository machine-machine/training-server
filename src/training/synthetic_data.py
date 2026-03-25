"""
Synthetic dataset utilities for ensemble training.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence

import numpy as np


class SyntheticRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    CHOP = "chop"
    MEV_HEAVY = "mev_heavy"


@dataclass
class SyntheticDataset:
    features_50: np.ndarray
    signal_labels: np.ndarray
    price_sequences: np.ndarray
    future_returns_t15: np.ndarray
    rug_labels: np.ndarray

    def __len__(self) -> int:
        return int(self.features_50.shape[0])


class SyntheticDataGenerator:
    """Convenience wrapper for synthetic dataset generation."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate(
        self,
        n_samples_per_regime: int = 2000,
        regimes: Sequence[SyntheticRegime] | None = None,
    ) -> SyntheticDataset:
        return generate_training_dataset(
            n_samples_per_regime=n_samples_per_regime,
            regimes=regimes,
            seed=self.seed,
        )


def _regime_params(regime: SyntheticRegime) -> tuple[float, float, float]:
    if regime == SyntheticRegime.BULL:
        return 0.020, 0.040, 0.03
    if regime == SyntheticRegime.BEAR:
        return -0.020, 0.060, 0.08
    if regime == SyntheticRegime.MEV_HEAVY:
        return 0.000, 0.080, 0.12
    return 0.000, 0.035, 0.05  # CHOP/default


def _single_regime_block(
    rng: np.random.Generator,
    regime: SyntheticRegime,
    n: int,
    sequence_length: int,
) -> SyntheticDataset:
    drift, vol, rug_rate = _regime_params(regime)

    features = rng.normal(0.0, 1.0, size=(n, 50)).astype(np.float64)
    # Seed regime-relevant feature slots used by trainers.
    features[:, 23] = np.abs(rng.normal(vol, vol * 0.5, size=n))  # momentum proxy
    features[:, 25] = rng.normal(drift, vol, size=n)  # return/momentum proxy
    features[:, 41] = np.clip(rng.normal(rug_rate, 0.05, size=n), 0.0, 1.0)  # MEV/risk proxy
    features[:, 2] = np.clip(rng.normal(0.4, 0.2, size=n), 0.0, 1.0)  # liquidity quality
    features[:, 7] = np.clip(rng.normal(0.6, 0.3, size=n), 0.0, 2.0)  # volume/liquidity ratio

    future_returns = rng.normal(drift, vol, size=n).astype(np.float64)
    signal_labels = (future_returns > 0.0).astype(np.int32)
    rug_labels = (rng.random(n) < rug_rate).astype(np.int32)

    # Generate simple random-walk price windows.
    noise = rng.normal(drift / max(sequence_length, 1), vol / np.sqrt(max(sequence_length, 1)), size=(n, sequence_length))
    prices = 1.0 + np.cumsum(noise, axis=1)
    prices = np.clip(prices, 1e-4, None).astype(np.float64)

    return SyntheticDataset(
        features_50=features,
        signal_labels=signal_labels,
        price_sequences=prices,
        future_returns_t15=future_returns,
        rug_labels=rug_labels,
    )


def generate_training_dataset(
    n_samples_per_regime: int = 2000,
    regimes: Sequence[SyntheticRegime] | None = None,
    seed: int = 42,
    sequence_length: int = 60,
) -> SyntheticDataset:
    active_regimes: List[SyntheticRegime] = list(regimes) if regimes else [
        SyntheticRegime.BULL,
        SyntheticRegime.BEAR,
        SyntheticRegime.CHOP,
        SyntheticRegime.MEV_HEAVY,
    ]
    rng = np.random.default_rng(seed)

    blocks = [
        _single_regime_block(rng, regime=r, n=n_samples_per_regime, sequence_length=sequence_length)
        for r in active_regimes
    ]
    features = np.vstack([b.features_50 for b in blocks])
    labels = np.concatenate([b.signal_labels for b in blocks])
    sequences = np.vstack([b.price_sequences for b in blocks])
    returns = np.concatenate([b.future_returns_t15 for b in blocks])
    rugs = np.concatenate([b.rug_labels for b in blocks])

    # Shuffle globally for realistic train/val/test splitting.
    idx = rng.permutation(len(labels))
    return SyntheticDataset(
        features_50=features[idx],
        signal_labels=labels[idx],
        price_sequences=sequences[idx],
        future_returns_t15=returns[idx],
        rug_labels=rugs[idx],
    )
