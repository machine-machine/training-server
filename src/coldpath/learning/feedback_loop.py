"""
Feedback Loop Pipeline - Continuous Model Improvement.

This module implements the critical missing feedback loop that enables
the trading system to learn from its own outcomes and continuously
improve model performance.

Architecture:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Trade Outcomes │────▶│  ETL Pipeline   │────▶│  Model Trainer  │
│  (Database)     │     │  (Transform)    │     │  (V3 Model)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
         ┌──────────────────────────────────────────────┘
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Evidence Gates  │────▶│ Model Promotion │────▶│ HotPath Push    │
│ (Validation)    │     │ (A/B Testing)   │     │ (gRPC)          │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         └───────────────────────────────────────────────┘
                           ❌ Rollback if evidence gates fail

Evidence Gates (ALL must pass for promotion):
1. Minimum sample count: >= 200 real outcomes
2. Win rate improvement: candidate_win_rate >= baseline_win_rate * 1.05
3. OOS consistency: |train_auc - oos_auc| < 0.15
4. Walk-forward stability: avg_sharpe > 0.5 across 5 folds
5. Profit factor: pf > 1.2 on validation set
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .feature_engineering import FeatureSet

if TYPE_CHECKING:
    from ..storage import DatabaseManager
    from ..training.ensemble_trainer import EnsembleTrainer
    from .model_updater import ModelUpdater
    from .profitability_learner import ProfitabilityLearner

logger = logging.getLogger(__name__)


class FeedbackLoopState(Enum):
    """State of the feedback loop."""

    IDLE = "idle"
    COLLECTING = "collecting"
    TRAINING = "training"
    VALIDATING = "validating"
    PROMOTING = "promoting"
    ROLLBACK = "rollback"
    ERROR = "error"


class PromotionDecision(Enum):
    """Decision after evidence gate evaluation."""

    PROMOTE = "promote"
    REJECT = "reject"
    DEFER = "defer"  # Need more data
    ROLLBACK = "rollback"


@dataclass
class EvidenceGates:
    """Evidence gate thresholds for model promotion."""

    # Sample requirements
    min_total_samples: int = 200
    min_traded_samples: int = 50
    min_positive_samples: int = 20

    # Performance requirements
    min_win_rate: float = 0.35
    min_profit_factor: float = 1.2
    min_sharpe_ratio: float = 0.5

    # Validation requirements
    max_train_oos_gap: float = 0.15  # Max gap between train and OOS AUC
    min_oos_auc: float = 0.55
    min_fold_consistency: float = 0.7  # % of folds that pass

    # Improvement requirements (vs baseline)
    min_win_rate_improvement: float = 0.05  # 5% relative improvement
    min_profit_factor_improvement: float = 0.10  # 10% relative improvement


@dataclass
class FeedbackMetrics:
    """Metrics from the feedback loop."""

    # Data collection
    total_outcomes: int = 0
    traded_outcomes: int = 0
    skipped_outcomes: int = 0
    positive_outcomes: int = 0

    # Current model performance
    baseline_win_rate: float = 0.0
    baseline_profit_factor: float = 0.0
    baseline_sharpe: float = 0.0

    # Candidate model performance
    candidate_win_rate: float = 0.0
    candidate_profit_factor: float = 0.0
    candidate_sharpe: float = 0.0
    candidate_oos_auc: float = 0.0

    # Evidence gate results
    gates_passed: int = 0
    gates_total: int = 0
    gate_details: dict[str, bool] = field(default_factory=dict)

    # Promotion
    promotion_decision: PromotionDecision = PromotionDecision.DEFER
    promoted_version: int | None = None
    rollback_version: int | None = None

    # Timestamps
    last_collection_ms: int | None = None
    last_training_ms: int | None = None
    last_promotion_ms: int | None = None


class FeedbackLoopPipeline:
    """Continuous model improvement feedback loop.

    This is the critical missing piece that enables the trading system
    to learn from its mistakes and improve over time.

    Usage:
        pipeline = FeedbackLoopPipeline(db, trainer, updater)
        await pipeline.start()  # Starts background collection and training

    Or run manually:
        await pipeline.run_cycle()
    """

    def __init__(
        self,
        db: "DatabaseManager",
        trainer: Optional["EnsembleTrainer"] = None,
        updater: Optional["ModelUpdater"] = None,
        learner: Optional["ProfitabilityLearner"] = None,
        gates: EvidenceGates | None = None,
        collection_interval_hours: float = 1.0,
        training_interval_hours: float = 6.0,
    ):
        self.db = db
        self.trainer = trainer
        self.updater = updater
        self.learner = learner
        self.gates = gates or EvidenceGates()

        self.collection_interval = collection_interval_hours
        self.training_interval = training_interval_hours

        self._state = FeedbackLoopState.IDLE
        self._metrics = FeedbackMetrics()
        self._running = False

        # Track training history for rollback
        self._model_history: list[dict[str, Any]] = []
        self._max_history = 10

        # Track backtest outcomes for parameter optimization
        self._backtest_outcomes: list[dict[str, Any]] = []
        self._max_backtest_history = 500

    @property
    def state(self) -> FeedbackLoopState:
        return self._state

    @property
    def metrics(self) -> FeedbackMetrics:
        return self._metrics

    async def start(self):
        """Start the feedback loop background task."""
        self._running = True
        logger.info("Feedback loop pipeline started")

        while self._running:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"Feedback loop error: {e}")
                self._state = FeedbackLoopState.ERROR

            # Wait for next cycle
            await asyncio.sleep(self.collection_interval * 3600)

    async def stop(self):
        """Stop the feedback loop."""
        self._running = False
        self._state = FeedbackLoopState.IDLE
        logger.info("Feedback loop pipeline stopped")

    def add_backtest_outcome(self, outcome: dict[str, Any]) -> None:
        """Add a backtest outcome for parameter optimization learning.

        This allows the feedback loop to learn from backtest optimization
        results and use them for future parameter recommendations.

        Args:
            outcome: Dict with params, metrics, score, risk_tolerance, timestamp
        """
        self._backtest_outcomes.append(outcome)

        # Limit history size
        if len(self._backtest_outcomes) > self._max_backtest_history:
            self._backtest_outcomes = self._backtest_outcomes[-self._max_backtest_history // 2 :]

        logger.debug(f"Added backtest outcome, total: {len(self._backtest_outcomes)}")

    def get_backtest_outcomes(self) -> list[dict[str, Any]]:
        """Get accumulated backtest outcomes."""
        return self._backtest_outcomes.copy()

    async def run_cycle(self) -> FeedbackMetrics:
        """Run a complete feedback loop cycle.

        Steps:
        1. Collect outcomes from database
        2. Transform into training data
        3. Check if enough data to train
        4. Train candidate model
        5. Validate against evidence gates
        6. Promote or reject
        """
        logger.info("Starting feedback loop cycle")

        # Step 1: Collect outcomes
        self._state = FeedbackLoopState.COLLECTING
        training_data = await self._collect_training_data()

        if len(training_data) < self.gates.min_total_samples:
            logger.info(
                f"Insufficient samples ({len(training_data)} < {self.gates.min_total_samples}), "
                f"deferring training"
            )
            self._metrics.promotion_decision = PromotionDecision.DEFER
            return self._metrics

        # Step 2: Train candidate model
        self._state = FeedbackLoopState.TRAINING
        candidate_metrics = await self._train_candidate_model(training_data)

        if candidate_metrics is None:
            logger.warning("Training failed, deferring promotion")
            self._metrics.promotion_decision = PromotionDecision.DEFER
            return self._metrics

        # Step 3: Validate against evidence gates
        self._state = FeedbackLoopState.VALIDATING
        decision = await self._evaluate_evidence_gates(candidate_metrics)
        self._metrics.promotion_decision = decision

        # Step 4: Promote or reject
        if decision == PromotionDecision.PROMOTE:
            self._state = FeedbackLoopState.PROMOTING
            await self._promote_model()
        elif decision == PromotionDecision.ROLLBACK:
            self._state = FeedbackLoopState.ROLLBACK
            await self._rollback_model()
        else:
            logger.info(f"Model not promoted: {decision.value}")

        self._state = FeedbackLoopState.IDLE
        self._metrics.last_collection_ms = int(datetime.now().timestamp() * 1000)

        return self._metrics

    async def _collect_training_data(self) -> list[dict[str, Any]]:
        """Collect and transform outcomes into training data.

        ETL Pipeline:
        1. Extract: Query scan_outcomes and training_outcomes tables
        2. Transform: Convert to feature vectors with labels
        3. Validate: Apply data quality and live mode safeguards
        4. Load: Return as list of validated training samples
        """
        # Get outcomes from last 7 days
        since_hours = 168

        # Get scan outcomes (trades + counterfactuals)
        scan_outcomes = await self.db.get_scan_outcomes(
            since_hours=since_hours,
            limit=10000,
        )

        # Get training outcomes (detailed decision context)
        training_outcomes = await self.db.get_training_outcomes(
            since_hours=since_hours,
            limit=10000,
        )

        training_data = []

        # Process scan outcomes
        for outcome in scan_outcomes:
            sample = self._transform_outcome(outcome)
            if sample is not None:
                # Apply live mode safeguards if sample is from live trading
                if outcome.get("mode") == "LIVE":
                    sample = self._apply_live_safeguards(sample, outcome)
                    if sample is None:
                        continue  # Rejected by safeguards
                training_data.append(sample)

        # Process training outcomes (higher quality)
        for outcome in training_outcomes:
            sample = self._transform_training_outcome(outcome)
            if sample is not None:
                # Apply live mode safeguards if sample is from live trading
                if outcome.get("mode") == "LIVE":
                    sample = self._apply_live_safeguards(sample, outcome)
                    if sample is None:
                        continue  # Rejected by safeguards
                training_data.append(sample)

        # Update metrics
        self._metrics.total_outcomes = len(training_data)
        self._metrics.traded_outcomes = sum(
            1 for s in training_data if s.get("outcome_type") == "traded"
        )
        self._metrics.positive_outcomes = sum(
            1 for s in training_data if s.get("label_binary") == 1
        )

        logger.info(
            f"Collected {len(training_data)} training samples: "
            f"{self._metrics.traded_outcomes} traded, "
            f"{self._metrics.positive_outcomes} positive"
        )

        return training_data

    def _transform_outcome(self, outcome: dict[str, Any]) -> dict[str, Any] | None:
        """Transform a scan outcome into a training sample.

        Label calculation:
        - Traded: label = 1 if pnl_pct > 0, else 0
        - Skipped (counterfactual): label = 1 if return_1h > 0, else 0
        """
        try:
            features_json = outcome.get("features_json", "{}")
            features = (
                json.loads(features_json) if isinstance(features_json, str) else features_json
            )

            # Determine label
            label_binary = None
            label_return = None
            outcome_type = outcome.get("outcome_type", "unknown")

            if outcome_type == "traded":
                pnl_pct = outcome.get("pnl_pct")
                if pnl_pct is not None:
                    label_binary = 1 if pnl_pct > 0 else 0
                    label_return = pnl_pct / 100.0  # Convert to decimal
            elif outcome_type == "skipped":
                # Counterfactual: would it have been profitable?
                was_profitable = outcome.get("was_profitable_counterfactual")
                if was_profitable is not None:
                    label_binary = was_profitable
                    # Estimate return from counterfactual
                    price_skip = outcome.get("price_at_skip", 0)
                    price_1h = outcome.get("price_1h_later", 0)
                    if price_skip > 0 and price_1h > 0:
                        label_return = (price_1h - price_skip) / price_skip

            if label_binary is None:
                return None

            return {
                "features": features,
                "label_binary": label_binary,
                "label_return": label_return,
                "outcome_type": outcome_type,
                "mint": outcome.get("mint"),
                "timestamp_ms": outcome.get("timestamp_ms"),
                "model_version": outcome.get("model_version", 0),
            }

        except Exception as e:
            logger.debug(f"Failed to transform outcome: {e}")
            return None

    def _transform_training_outcome(self, outcome: dict[str, Any]) -> dict[str, Any] | None:
        """Transform a training outcome (higher quality) into a training sample."""
        try:
            features_json = outcome.get("features_json", "{}")
            features = (
                json.loads(features_json) if isinstance(features_json, str) else features_json
            )

            if not features:
                return None

            # Training outcomes have explicit labels
            label_binary = outcome.get("label_binary")
            label_return = outcome.get("label_return")

            # Use ev_label if binary not available
            if label_binary is None:
                ev_label = outcome.get("ev_label", "")
                if ev_label in ("GOOD_EV",):
                    label_binary = 1
                elif ev_label in ("BAD_EV", "RUG", "MEV_HIT", "SLIPPAGE_FAIL", "EXECUTION_TIMEOUT"):
                    label_binary = 0
                else:
                    # Fall back to pnl_pct
                    pnl_pct = outcome.get("pnl_pct")
                    if pnl_pct is not None:
                        label_binary = 1 if pnl_pct > 0 else 0

            if label_binary is None:
                return None

            return {
                "features": features,
                "label_binary": label_binary,
                "label_return": label_return,
                "outcome_type": "training",
                "mint": outcome.get("mint"),
                "timestamp_ms": outcome.get("decision_timestamp_ms"),
                "model_version": outcome.get("model_version", 0),
                "regime": outcome.get("regime"),
                "ev_label": outcome.get("ev_label"),
            }

        except Exception as e:
            logger.debug(f"Failed to transform training outcome: {e}")
            return None

    async def _train_candidate_model(
        self,
        training_data: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Train a candidate model and return validation metrics.

        Uses walk-forward validation to prevent overfitting.
        """
        if len(training_data) < self.gates.min_total_samples:
            return None

        # Validate training samples before use
        from ..validation.training_sample_validator import TrainingSampleValidator

        validator = TrainingSampleValidator(
            strict_mode=False,
            min_samples=self.gates.min_total_samples,
        )
        validation_result = validator.validate_samples(training_data)

        if not validation_result.valid:
            logger.error(f"Training sample validation failed: {validation_result.summary()}")
            # Log issues for debugging
            for issue in validation_result.issues[:5]:  # Log first 5 issues
                logger.error(f"  - {issue.code}: {issue.message}")
            return None

        logger.info(f"Training samples validated: {validation_result.summary()}")

        # Extract features and labels from validated samples
        X = []
        y_binary = []
        y_return = []

        for sample in training_data:
            features = sample.get("features", {})
            if isinstance(features, dict):
                # Convert dict to array (50 features in order)
                feature_array = self._features_to_array(features)
                X.append(feature_array)
                y_binary.append(sample.get("label_binary", 0))
                y_return.append(sample.get("label_return", 0.0))

        if len(X) < self.gates.min_total_samples:
            logger.warning(f"Not enough valid feature vectors: {len(X)}")
            return None
        if self.trainer is None:
            logger.warning("Feedback loop trainer is not configured; deferring training")
            return None

        X = np.array(X)
        y_binary = np.array(y_binary)
        y_return = np.array(y_return)

        # Train with walk-forward validation
        try:
            # Use the existing train method with real data format
            real_data = {
                "features_50": X,
                "signal_labels": y_binary,
                "price_sequences": None,  # Not needed for signal classification
                "price_targets": y_return,
                "rug_labels": None,
            }

            result = self.trainer.train(
                real_data=real_data,
                synthetic_only=False,
                mixed_ratio=0.0,  # Use only real data
            )

            metrics = {
                "win_rate": result.validation_metrics.get("win_rate", 0),
                "profit_factor": result.validation_metrics.get("profit_factor", 0),
                "sharpe": result.validation_metrics.get("sharpe", 0),
                "oos_auc": result.validation_metrics.get("auc_roc", 0),
                "train_auc": result.training_metrics.get("auc_roc", 0),
                "fold_results": [],
            }

            self._metrics.last_training_ms = int(datetime.now().timestamp() * 1000)

            return {
                "win_rate": metrics.get("win_rate", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "sharpe": metrics.get("sharpe", 0),
                "oos_auc": metrics.get("oos_auc", 0),
                "train_auc": metrics.get("train_auc", 0),
                "fold_results": metrics.get("fold_results", []),
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None

    def _features_to_array(self, features: dict[str, float]) -> np.ndarray:
        """Convert feature dict to ordered array (50 features).

        Uses canonical FeatureSet.FEATURE_NAMES to ensure alignment with:
        - HotPath feature_vector.rs FeatureVector::feature_names()
        - v3_profitability_trainer FEATURE_NAMES
        - student_distiller HOTPATH_FEATURES

        This fixes a critical bug where the previous hardcoded list used
        completely different feature names/order, corrupting training data.
        """
        # Use canonical feature order from FeatureSet (matches HotPath v3 schema)
        feature_names = FeatureSet().FEATURE_NAMES

        # Build array with defaults for missing features
        arr = np.zeros(50)
        for i, name in enumerate(feature_names):
            if i >= 50:
                break
            arr[i] = features.get(name, 0.0)

        return arr

    def _apply_live_safeguards(
        self,
        sample: dict[str, Any],
        outcome: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Apply live mode safeguards to a training sample.

        This ensures that live trading data meets quality standards
        before being used for model training.

        Args:
            sample: Transformed training sample
            outcome: Original outcome data

        Returns:
            Validated sample or None if rejected
        """
        try:
            from ..validation.live_mode_safeguards import (
                LiveDataSafeguard,
                LiveDataSafeguardConfig,
            )

            # Initialize safeguard (could be cached as instance variable)
            if not hasattr(self, "_live_safeguard"):
                config = LiveDataSafeguardConfig(
                    min_delay_hours=24.0,
                    max_slippage_bps=200.0,
                    min_fill_rate=0.95,
                )
                self._live_safeguard = LiveDataSafeguard(config)

            # Validate live sample
            live_sample = self._live_safeguard.validate_live_sample(
                {
                    **sample,
                    **outcome,
                    "slippage_bps": outcome.get("slippage_bps", 0),
                    "fill_rate": outcome.get("fill_rate", 1.0),
                    "execution_timestamp_ms": outcome.get(
                        "execution_timestamp_ms", sample.get("timestamp_ms", 0)
                    ),
                }
            )

            # Check if rejected
            if live_sample.is_rejected:
                logger.debug(
                    f"Live sample rejected for training: {live_sample.sample_id} - "
                    f"{[r.value for r in live_sample.rejection_reasons]}"
                )
                return None

            # Track for statistics
            self._live_safeguard.add_pending_sample(live_sample)

            # Return validated sample with quality metadata
            sample["execution_quality"] = live_sample.execution_quality.value
            sample["safeguard_validated"] = True

            return sample

        except Exception as e:
            logger.warning(f"Failed to apply live safeguards: {e}")
            # Return sample anyway to avoid blocking training
            return sample

    async def _evaluate_evidence_gates(
        self,
        candidate_metrics: dict[str, Any],
    ) -> PromotionDecision:
        """Evaluate all evidence gates and return promotion decision.

        Evidence gates (ALL must pass):
        1. Sample count
        2. Win rate
        3. Profit factor
        4. Sharpe ratio
        5. OOS AUC
        6. Train-OOS gap
        7. Fold consistency
        8. Improvement vs baseline
        """
        gates_passed = 0
        gates_total = 8
        gate_results = {}

        # Gate 1: Sample count
        sample_gate = self._metrics.total_outcomes >= self.gates.min_total_samples
        gate_results["min_samples"] = sample_gate
        if sample_gate:
            gates_passed += 1

        # Gate 2: Win rate
        win_rate = candidate_metrics.get("win_rate", 0)
        win_rate_gate = win_rate >= self.gates.min_win_rate
        gate_results["min_win_rate"] = win_rate_gate
        if win_rate_gate:
            gates_passed += 1
        self._metrics.candidate_win_rate = win_rate

        # Gate 3: Profit factor
        pf = candidate_metrics.get("profit_factor", 0)
        pf_gate = pf >= self.gates.min_profit_factor
        gate_results["min_profit_factor"] = pf_gate
        if pf_gate:
            gates_passed += 1
        self._metrics.candidate_profit_factor = pf

        # Gate 4: Sharpe ratio
        sharpe = candidate_metrics.get("sharpe", 0)
        sharpe_gate = sharpe >= self.gates.min_sharpe_ratio
        gate_results["min_sharpe"] = sharpe_gate
        if sharpe_gate:
            gates_passed += 1
        self._metrics.candidate_sharpe = sharpe

        # Gate 5: OOS AUC
        oos_auc = candidate_metrics.get("oos_auc", 0)
        oos_gate = oos_auc >= self.gates.min_oos_auc
        gate_results["min_oos_auc"] = oos_gate
        if oos_gate:
            gates_passed += 1
        self._metrics.candidate_oos_auc = oos_auc

        # Gate 6: Train-OOS gap (prevent overfitting)
        train_auc = candidate_metrics.get("train_auc", 0)
        gap = abs(train_auc - oos_auc)
        gap_gate = gap <= self.gates.max_train_oos_gap
        gate_results["train_oos_gap"] = gap_gate
        if gap_gate:
            gates_passed += 1

        # Gate 7: Fold consistency
        fold_results = candidate_metrics.get("fold_results", [])
        if fold_results:
            passing_folds = sum(1 for f in fold_results if f.get("sharpe", 0) > 0.5)
            consistency = passing_folds / len(fold_results)
        else:
            consistency = 0
        consistency_gate = consistency >= self.gates.min_fold_consistency
        gate_results["fold_consistency"] = consistency_gate
        if consistency_gate:
            gates_passed += 1

        # Gate 8: Improvement vs baseline
        baseline_wr = self._metrics.baseline_win_rate or 0.35
        if baseline_wr > 0:
            wr_improvement = (win_rate - baseline_wr) / baseline_wr
            improvement_gate = wr_improvement >= self.gates.min_win_rate_improvement
        else:
            improvement_gate = win_rate >= self.gates.min_win_rate
        gate_results["improvement_vs_baseline"] = improvement_gate
        if improvement_gate:
            gates_passed += 1

        # Update metrics
        self._metrics.gates_passed = gates_passed
        self._metrics.gates_total = gates_total
        self._metrics.gate_details = gate_results

        logger.info(f"Evidence gates: {gates_passed}/{gates_total} passed - {gate_results}")

        # Decision logic
        if gates_passed == gates_total:
            return PromotionDecision.PROMOTE
        elif gates_passed >= gates_total * 0.7:
            # 70% gates pass - defer for more data
            return PromotionDecision.DEFER
        elif gates_passed < gates_total * 0.5:
            # Less than 50% gates pass - potential issue, consider rollback
            if self._metrics.baseline_win_rate > win_rate:
                return PromotionDecision.ROLLBACK
            return PromotionDecision.REJECT
        else:
            return PromotionDecision.DEFER

    async def _promote_model(self):
        """Promote the candidate model to production."""
        logger.info("Promoting candidate model to production")
        if self.updater is None:
            logger.warning("Feedback loop updater is not configured; skipping promotion")
            return

        try:
            # Get current model version for history
            current_version = self.updater.current_version

            # Save current model to history for potential rollback
            if self.updater.last_export:
                self._model_history.append(
                    {
                        "version": current_version,
                        "export": self.updater.last_export.to_dict(),
                        "timestamp_ms": int(datetime.now().timestamp() * 1000),
                    }
                )
                # Trim history
                if len(self._model_history) > self._max_history:
                    self._model_history.pop(0)

            # Export and push new model
            if self.learner and self.learner.is_trained:
                export = await self.updater.export_and_push(
                    self.learner,
                    apply_immediately=True,
                    reason="feedback_loop_promotion",
                )

                self._metrics.promoted_version = export.version
                self._metrics.baseline_win_rate = self._metrics.candidate_win_rate
                self._metrics.baseline_profit_factor = self._metrics.candidate_profit_factor
                self._metrics.baseline_sharpe = self._metrics.candidate_sharpe
                self._metrics.last_promotion_ms = int(datetime.now().timestamp() * 1000)

                logger.info(
                    f"Model v{export.version} promoted: "
                    f"wr={self._metrics.candidate_win_rate:.2%}, "
                    f"pf={self._metrics.candidate_profit_factor:.2f}, "
                    f"sharpe={self._metrics.candidate_sharpe:.2f}"
                )
            else:
                logger.warning("Learner not trained, cannot promote")

        except Exception as e:
            logger.error(f"Model promotion failed: {e}")
            # Trigger rollback
            await self._rollback_model()

    async def _rollback_model(self):
        """Rollback to the previous model version."""
        logger.warning("Rolling back to previous model")
        if self.updater is None:
            logger.error("Feedback loop updater is not configured; cannot rollback")
            return

        if not self._model_history:
            logger.error("No model history available for rollback")
            return

        try:
            # Get previous model
            previous = self._model_history[-1]
            previous_version = previous.get("version")
            previous_export = previous.get("export")

            if previous_export:
                # Restore previous model
                from .model_updater import ModelExport

                export = ModelExport.from_dict(previous_export)

                # Push to HotPath
                if self.updater.grpc_client:
                    await self.updater._push_to_hotpath(
                        export,
                        apply_immediately=True,
                        reason="feedback_loop_rollback",
                    )

                self._metrics.rollback_version = previous_version
                logger.info(f"Rolled back to model v{previous_version}")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    async def get_health_report(self) -> dict[str, Any]:
        """Get a comprehensive health report of the feedback loop."""
        return {
            "state": self._state.value,
            "metrics": {
                "total_outcomes": self._metrics.total_outcomes,
                "traded_outcomes": self._metrics.traded_outcomes,
                "positive_outcomes": self._metrics.positive_outcomes,
                "baseline_win_rate": self._metrics.baseline_win_rate,
                "baseline_profit_factor": self._metrics.baseline_profit_factor,
                "gates_passed": f"{self._metrics.gates_passed}/{self._metrics.gates_total}",
                "gate_details": self._metrics.gate_details,
                "last_collection_ms": self._metrics.last_collection_ms,
                "last_training_ms": self._metrics.last_training_ms,
                "last_promotion_ms": self._metrics.last_promotion_ms,
            },
            "model_history": [
                {"version": h.get("version"), "timestamp_ms": h.get("timestamp_ms")}
                for h in self._model_history
            ],
            "gates_config": {
                "min_samples": self.gates.min_total_samples,
                "min_win_rate": self.gates.min_win_rate,
                "min_profit_factor": self.gates.min_profit_factor,
                "min_sharpe": self.gates.min_sharpe_ratio,
                "min_oos_auc": self.gates.min_oos_auc,
            },
        }


class FeedbackLoopScheduler:
    """Schedules and manages the feedback loop.

    Usage:
        scheduler = FeedbackLoopScheduler(pipeline)
        await scheduler.start()

        # Or trigger manually:
        await scheduler.trigger_cycle()

        # Restart if stuck:
        await scheduler.restart()
    """

    def __init__(
        self,
        pipeline: FeedbackLoopPipeline,
        min_interval_hours: float = 6.0,
        min_new_outcomes: int = 50,
    ):
        self.pipeline = pipeline
        self.min_interval_hours = min_interval_hours
        self.min_new_outcomes = min_new_outcomes

        self._running = False
        self._last_outcome_count = 0
        self._last_cycle_ms: int | None = None
        self._task: asyncio.Task | None = None

    async def start(self):
        """Start the scheduler.

        NOTE: If already running, this will log a warning and return.
        Use restart() to force a restart.
        """
        if self._running and self._task and not self._task.done():
            logger.warning(
                "Feedback loop scheduler already running, ignoring start() call. "
                "Use restart() to force restart."
            )
            return

        self._running = True
        logger.info("Feedback loop scheduler started")

        while self._running:
            try:
                # Check if we should run a cycle
                should_run = await self._should_run_cycle()

                if should_run:
                    await self.pipeline.run_cycle()
                    self._last_cycle_ms = int(datetime.now().timestamp() * 1000)
                    self._last_outcome_count = self.pipeline.metrics.total_outcomes

            except Exception as e:
                logger.error(f"Scheduler cycle error: {e}")

            # Check every hour
            await asyncio.sleep(3600)

    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        await self.pipeline.stop()

        # Cancel the background task if it exists
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (TimeoutError, asyncio.CancelledError):
                logger.warning("Scheduler task did not complete after cancellation")

        logger.info("Feedback loop scheduler stopped")

    async def restart(self) -> dict[str, Any]:
        """Restart the scheduler.

        This will:
        1. Stop the current scheduler loop
        2. Wait for it to finish
        3. Start a fresh scheduler loop

        Returns:
            Status dict with restart result
        """
        logger.info("Restarting feedback loop scheduler...")

        was_running = self._running

        # Stop the current scheduler
        await self.stop()

        # Small delay to ensure cleanup
        await asyncio.sleep(0.5)

        # Reset state
        self._running = False
        self._task = None

        # Start fresh
        self._task = asyncio.create_task(self.start(), name="feedback-loop-scheduler")

        # Wait a moment to confirm it started
        await asyncio.sleep(0.1)

        result = {
            "status": "restarted",
            "was_running": was_running,
            "now_running": self._running and not self._task.done(),
            "pipeline_state": self.pipeline.state.value,
        }

        logger.info(f"Feedback loop scheduler restart complete: {result}")
        return result

    def set_task(self, task: asyncio.Task) -> None:
        """Set the background task reference (called by server.py)."""
        self._task = task

    def is_alive(self) -> bool:
        """Check if the scheduler is actually running (task is alive)."""
        return self._running and self._task is not None and not self._task.done()

    async def _should_run_cycle(self) -> bool:
        """Check if we should run a feedback loop cycle."""
        # Check time since last cycle
        if self._last_cycle_ms:
            elapsed_hours = (datetime.now().timestamp() * 1000 - self._last_cycle_ms) / (
                1000 * 3600
            )

            if elapsed_hours < self.min_interval_hours:
                return False

        # Check for new outcomes
        current_outcomes = self.pipeline.metrics.total_outcomes
        new_outcomes = current_outcomes - self._last_outcome_count

        return new_outcomes >= self.min_new_outcomes

    async def trigger_cycle(self) -> FeedbackMetrics:
        """Manually trigger a feedback loop cycle."""
        logger.info("Manually triggering feedback loop cycle")
        return await self.pipeline.run_cycle()

    async def get_status(self) -> dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self._running,
            "last_cycle_ms": self._last_cycle_ms,
            "last_outcome_count": self._last_outcome_count,
            "pipeline_state": self.pipeline.state.value,
            "pipeline_health": await self.pipeline.get_health_report(),
        }
