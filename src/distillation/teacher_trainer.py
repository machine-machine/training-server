"""
Teacher Trainer - Wraps ProfitabilityLearner to produce soft labels for distillation.

Orchestrates the existing ensemble (LGB/XGB/Logistic) training and generates
teacher probability predictions for knowledge distillation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..learning.profitability_learner import (
    ProfitabilityLearner,
    TrainingConfig,
    TrainingMetrics,
)
from ..learning.feature_engineering import FeatureEngineer
from ..storage import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class TeacherConfig:
    """Configuration for teacher training."""

    # Data selection
    lookback_hours: int = 168  # 1 week
    min_outcomes: int = 200
    outcome_type: Optional[str] = None  # None = all types

    # Training config passthrough
    training_config: Optional[TrainingConfig] = None


@dataclass
class TeacherResult:
    """Result of teacher training including soft labels."""

    learner: ProfitabilityLearner
    soft_labels: np.ndarray  # Teacher probability predictions on full training data
    features: np.ndarray  # Full feature matrix (N x 50)
    labels: np.ndarray  # Hard binary labels
    metrics: TrainingMetrics
    feature_engineer: FeatureEngineer
    outcomes: List[Dict[str, Any]]  # Raw outcome dicts for feature mapping


class TeacherTrainer:
    """Train teacher ensemble and produce soft labels for distillation."""

    def __init__(self, db: DatabaseManager, config: Optional[TeacherConfig] = None):
        self.db = db
        self.config = config or TeacherConfig()

    async def train(self) -> Optional[TeacherResult]:
        """Train teacher model and generate soft labels.

        Returns:
            TeacherResult with trained model and soft labels, or None if
            insufficient data.
        """
        # 1. Load training outcomes from DB
        outcomes = await self.db.get_training_outcomes(
            since_hours=self.config.lookback_hours,
            limit=10000,
        )

        if len(outcomes) < self.config.min_outcomes:
            logger.warning(
                f"Insufficient training outcomes: {len(outcomes)} < "
                f"{self.config.min_outcomes}"
            )
            return None

        # Also load scan outcomes as fallback/supplement
        scan_outcomes = await self.db.get_scan_outcomes(
            since_hours=self.config.lookback_hours,
            limit=10000,
        )

        # Merge: prefer training_outcomes (richer labels), supplement with scan_outcomes
        all_outcomes = self._merge_outcomes(outcomes, scan_outcomes)

        if len(all_outcomes) < self.config.min_outcomes:
            logger.warning(
                f"Insufficient merged outcomes: {len(all_outcomes)} < "
                f"{self.config.min_outcomes}"
            )
            return None

        logger.info(f"Training teacher on {len(all_outcomes)} outcomes")

        # 2. Instantiate ProfitabilityLearner
        training_config = self.config.training_config or TrainingConfig()
        learner = ProfitabilityLearner(config=training_config)

        # 3. Train the ensemble
        metrics = learner.train(all_outcomes)

        if not learner.is_trained:
            logger.warning("Teacher training failed - model not trained")
            return None

        # 4. Generate soft labels on full dataset
        import pandas as pd

        df = pd.DataFrame(all_outcomes)
        learner.feature_engineer.fit(df)
        X_all, y_all = learner.feature_engineer.extract_batch(df)

        # Apply MI feature selection if the learner used it
        X_for_predict = X_all
        if learner._selected_feature_indices is not None:
            X_for_predict = X_all[:, learner._selected_feature_indices]

        soft_labels = learner._get_ensemble_predictions(X_for_predict)

        logger.info(
            f"Teacher trained: accuracy={metrics.accuracy:.3f}, "
            f"AUC={metrics.auc_roc:.3f}, "
            f"soft_labels shape={soft_labels.shape}"
        )

        return TeacherResult(
            learner=learner,
            soft_labels=soft_labels,
            features=X_all,  # Full 50-feature matrix (pre-MI-selection)
            labels=y_all,
            metrics=metrics,
            feature_engineer=learner.feature_engineer,
            outcomes=all_outcomes,
        )

    def _merge_outcomes(
        self,
        training_outcomes: List[Dict[str, Any]],
        scan_outcomes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge training outcomes and scan outcomes, deduplicating by mint+timestamp."""
        seen = set()
        merged = []

        # Prefer training_outcomes (have richer labels)
        for outcome in training_outcomes:
            key = (outcome.get("mint", ""), outcome.get("decision_timestamp_ms", 0))
            if key not in seen:
                seen.add(key)
                # Normalize fields for ProfitabilityLearner compatibility
                normalized = self._normalize_training_outcome(outcome)
                if normalized:
                    merged.append(normalized)

        # Supplement with scan_outcomes
        for outcome in scan_outcomes:
            key = (outcome.get("mint", ""), outcome.get("timestamp_ms", 0))
            if key not in seen:
                seen.add(key)
                merged.append(outcome)

        return merged

    def _normalize_training_outcome(self, outcome: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a training_outcome record to look like a scan_outcome for the learner."""
        import json

        features_json = outcome.get("features_json", "{}")
        try:
            features = json.loads(features_json) if isinstance(features_json, str) else features_json
        except (json.JSONDecodeError, TypeError):
            features = {}

        # Build a scan_outcome-compatible dict
        normalized = {
            "mint": outcome.get("mint", ""),
            "timestamp_ms": outcome.get("decision_timestamp_ms", 0),
            "outcome_type": outcome.get("execution_mode", "traded"),
            "pnl_pct": outcome.get("pnl_pct"),
            "pnl_sol": outcome.get("pnl_sol"),
            "label_binary": outcome.get("label_binary"),
            "was_profitable_counterfactual": outcome.get("label_binary"),
        }

        # Merge feature fields
        normalized.update(features)

        return normalized
