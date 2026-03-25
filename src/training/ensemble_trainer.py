"""
Ensemble Trainer with Synthetic Data Integration

Trains the complete 4-model ensemble using either real or synthetic data.
Supports mixed training strategies and validation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..learning.ensemble import ModelEnsemble, create_ensemble_from_training
from ..learning.regime_detector import RegimeDetector, compute_regime_features
from .synthetic_data import (
    SyntheticDataGenerator,
    SyntheticDataset,
    SyntheticRegime,
    generate_training_dataset,
)
from .synthetic_validator import SyntheticDataValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for ensemble training."""
    # Data generation
    use_synthetic: bool = True
    n_samples_per_regime: int = 2000
    synthetic_regimes: List[SyntheticRegime] = None

    # Training split
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    # Model training
    xgboost_n_estimators: int = 500
    lstm_epochs: int = 50
    isolation_forest_contamination: float = 0.1

    # Validation
    validate_synthetic: bool = True
    min_validation_score: float = 0.6

    # Random seed
    seed: int = 42

    def __post_init__(self):
        if self.synthetic_regimes is None:
            self.synthetic_regimes = [
                SyntheticRegime.BULL,
                SyntheticRegime.BEAR,
                SyntheticRegime.CHOP,
                SyntheticRegime.MEV_HEAVY,
            ]


@dataclass
class TrainingResult:
    """Result of ensemble training."""
    ensemble: ModelEnsemble
    training_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    synthetic_validation: Optional[ValidationResult]
    regime_detector: Optional[RegimeDetector]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
        }
        if self.synthetic_validation:
            result["synthetic_validation"] = self.synthetic_validation.to_dict()
        return result


class EnsembleTrainer:
    """Train complete ensemble with synthetic or real data.

    Example:
        trainer = EnsembleTrainer()
        result = trainer.train(use_synthetic=True)
        ensemble = result.ensemble
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        np.random.seed(self.config.seed)

    def train(
        self,
        real_data: Optional[Dict[str, np.ndarray]] = None,
        synthetic_only: bool = False,
        mixed_ratio: float = 0.5,
    ) -> TrainingResult:
        """Train ensemble on synthetic and/or real data.

        Args:
            real_data: Optional real market data dict with keys:
                      'features_50', 'signal_labels', 'price_sequences',
                      'price_targets', 'rug_labels'
            synthetic_only: Use only synthetic data
            mixed_ratio: Ratio of synthetic to real data (0.5 = 50/50)

        Returns:
            TrainingResult with trained ensemble and metrics
        """
        logger.info("Starting ensemble training")

        # Step 1: Generate or load data
        if synthetic_only or real_data is None:
            dataset = self._generate_synthetic_data()
            train_data, val_data, test_data = self._split_dataset(dataset)
            synthetic_validation = self._validate_synthetic(dataset)
        else:
            # Mix real and synthetic
            synthetic_dataset = self._generate_synthetic_data()
            mixed_dataset = self._mix_data(real_data, synthetic_dataset, mixed_ratio)
            train_data, val_data, test_data = self._split_mixed_data(mixed_dataset)
            synthetic_validation = self._validate_synthetic(synthetic_dataset)

        # Step 2: Train regime detector
        regime_detector = self._train_regime_detector(train_data)

        # Step 3: Train ensemble models
        ensemble = self._train_ensemble(train_data)

        # Step 4: Validate
        training_metrics = self._evaluate(ensemble, train_data, "train")
        validation_metrics = self._evaluate(ensemble, val_data, "validation")
        test_metrics = self._evaluate(ensemble, test_data, "test")

        logger.info(f"Training complete: {training_metrics}")
        logger.info(f"Validation: {validation_metrics}")
        logger.info(f"Test: {test_metrics}")

        return TrainingResult(
            ensemble=ensemble,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            synthetic_validation=synthetic_validation,
            regime_detector=regime_detector,
        )

    def _generate_synthetic_data(self) -> SyntheticDataset:
        """Generate synthetic training dataset."""
        logger.info(f"Generating synthetic data: {self.config.n_samples_per_regime} samples/regime")

        dataset = generate_training_dataset(
            n_samples_per_regime=self.config.n_samples_per_regime,
            regimes=self.config.synthetic_regimes,
            seed=self.config.seed,
        )

        logger.info(f"Generated {len(dataset)} total samples")
        return dataset

    def _validate_synthetic(
        self,
        dataset: SyntheticDataset,
    ) -> Optional[ValidationResult]:
        """Validate synthetic dataset quality."""
        if not self.config.validate_synthetic:
            return None

        validator = SyntheticDataValidator()
        result = validator.validate(dataset)

        if not result.passed:
            logger.error(f"Synthetic validation failed: {result.errors}")
        if result.warnings:
            logger.warning(f"Synthetic validation warnings: {result.warnings}")

        return result

    def _split_dataset(
        self,
        dataset: SyntheticDataset,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Split dataset into train/val/test."""
        n = len(dataset)
        n_train = int(n * self.config.train_fraction)
        n_val = int(n * self.config.val_fraction)

        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        # Handle price_sequences having fewer samples due to LSTM windowing
        # LSTM sequences are: n - sequence_length + 1 (default sequence_length=60)
        n_price_seq = len(dataset.price_sequences) if len(dataset.price_sequences) > 0 else n
        
        def safe_index(arr, idx, n_arr):
            """Index array safely, clipping indices to valid range."""
            valid_mask = idx < n_arr
            return arr[idx[valid_mask]]
        
        train_data = {
            "features_50": dataset.features_50[train_idx],
            "signal_labels": dataset.signal_labels[train_idx],
            "price_sequences": safe_index(dataset.price_sequences, train_idx, n_price_seq),
            "future_returns_t15": dataset.future_returns_t15[train_idx],
            "rug_labels": dataset.rug_labels[train_idx],
        }

        val_data = {
            "features_50": dataset.features_50[val_idx],
            "signal_labels": dataset.signal_labels[val_idx],
            "price_sequences": safe_index(dataset.price_sequences, val_idx, n_price_seq),
            "future_returns_t15": dataset.future_returns_t15[val_idx],
            "rug_labels": dataset.rug_labels[val_idx],
        }

        test_data = {
            "features_50": dataset.features_50[test_idx],
            "signal_labels": dataset.signal_labels[test_idx],
            "price_sequences": safe_index(dataset.price_sequences, test_idx, n_price_seq),
            "future_returns_t15": dataset.future_returns_t15[test_idx],
            "rug_labels": dataset.rug_labels[test_idx],
        }

        return train_data, val_data, test_data

    def _train_regime_detector(
        self,
        train_data: Dict[str, np.ndarray],
    ) -> RegimeDetector:
        """Train regime detector on feature data."""
        logger.info("Training regime detector")

        # Extract regime features from full feature set
        # Use liquidity, price, and MEV features
        features = train_data["features_50"]

        # Simplified: Use key features for regime detection
        regime_features = np.column_stack([
            features[:, 23],  # Volatility
            features[:, 25],  # Momentum
            features[:, 41],  # MEV rate
            features[:, 2],   # Liquidity volatility
            features[:, 7],   # Volume/Liquidity ratio
        ])

        detector = RegimeDetector()
        detector.fit(regime_features)

        logger.info("Regime detector trained")
        return detector

    def _train_ensemble(
        self,
        train_data: Dict[str, np.ndarray],
    ) -> ModelEnsemble:
        """Train complete ensemble."""
        logger.info("Training ensemble models")

        ensemble = create_ensemble_from_training(
            features_train=train_data["features_50"],
            labels_train=train_data["signal_labels"],
            price_sequences=train_data["price_sequences"],
            price_targets=train_data["future_returns_t15"],
            rug_labels=train_data.get("rug_labels"),
        )

        logger.info("Ensemble training complete")
        return ensemble

    def _evaluate(
        self,
        ensemble: ModelEnsemble,
        data: Dict[str, np.ndarray],
        split_name: str,
    ) -> Dict[str, Any]:
        """Evaluate ensemble on dataset."""
        features = data["features_50"]
        signal_labels = data["signal_labels"]
        sequences = data["price_sequences"]

        # Simplified evaluation (no price context for now)
        # In production, would use real current prices
        dummy_price = 1.0

        decisions = []
        for i in range(min(100, len(features))):  # Sample 100 for speed
            decision = ensemble.evaluate(
                features_50=features[i],
                price_sequence=sequences[i] if i < len(sequences) else sequences[0],
                current_price=dummy_price,
            )
            decisions.append(decision)

        # Calculate metrics
        confidences = [d.confidence for d in decisions]
        go_rate = sum(1 for d in decisions if d.should_trade) / len(decisions)

        return {
            "split": split_name,
            "n_samples": len(features),
            "avg_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences)),
            "go_rate": float(go_rate),
            "model_status": ensemble.get_model_status(),
        }

    def _mix_data(
        self,
        real_data: Dict[str, np.ndarray],
        synthetic: SyntheticDataset,
        ratio: float,
    ) -> Dict[str, np.ndarray]:
        """Mix real and synthetic data."""
        # Calculate sizes
        n_real = len(real_data["features_50"])
        n_synthetic = int(n_real * ratio / (1 - ratio))

        # Sample from synthetic
        indices = np.random.choice(len(synthetic), n_synthetic, replace=False)

        mixed = {
            "features_50": np.vstack([
                real_data["features_50"],
                synthetic.features_50[indices],
            ]),
            "signal_labels": np.concatenate([
                real_data["signal_labels"],
                synthetic.signal_labels[indices],
            ]),
            "price_sequences": np.vstack([
                real_data["price_sequences"],
                synthetic.price_sequences[indices],
            ]),
            "future_returns_t15": np.concatenate([
                real_data["future_returns_t15"],
                synthetic.future_returns_t15[indices],
            ]),
            "rug_labels": np.concatenate([
                real_data.get("rug_labels", np.zeros(n_real)),
                synthetic.rug_labels[indices],
            ]),
        }

        logger.info(f"Mixed dataset: {n_real} real + {n_synthetic} synthetic")
        return mixed

    def _split_mixed_data(
        self,
        data: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Split mixed dataset."""
        n = len(data["features_50"])
        n_train = int(n * self.config.train_fraction)
        n_val = int(n * self.config.val_fraction)

        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        def split_dict(d: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
            return {k: v[idx] for k, v in d.items()}

        return (
            split_dict(data, train_idx),
            split_dict(data, val_idx),
            split_dict(data, test_idx),
        )

    def save_trained_ensemble(
        self,
        result: TrainingResult,
        output_dir: str,
    ) -> None:
        """Save trained ensemble to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save ensemble
        result.ensemble.save(str(output_path))

        # Save regime detector
        if result.regime_detector:
            import pickle
            with open(output_path / "regime_detector.pkl", "wb") as f:
                pickle.dump(result.regime_detector, f)

        # Save metrics
        import json
        with open(output_path / "metrics.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved ensemble to {output_dir}")
