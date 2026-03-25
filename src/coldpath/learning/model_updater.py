"""
Model Updater - Export trained models to Hot Path.

Manages model versioning and pushes updates via gRPC.

Includes OnlineLearningIntegrator for real-time model adaptation:
- Connects OnlineLearner to production ensemble
- Provides continuous coefficient updates from paper/live outcomes
- Integrates with trading_losses for trading-aware optimization
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..storage import ModelArtifactStore
    from .online_learner import OnlineLearner, TradeOutcome
    from .profitability_learner import ProfitabilityLearner

logger = logging.getLogger(__name__)


@dataclass
class ModelExport:
    """Exported model data for Hot Path."""

    model_type: str  # "linear" or "lightgbm"
    version: int
    trained_at_ms: int

    # Linear weights (always present as fallback)
    linear_coefficients: list[float]
    linear_bias: float

    # LightGBM (optional)
    lightgbm_trees: str | None = None  # JSON-encoded tree structure
    lightgbm_learning_rate: float = 0.1
    lightgbm_base_score: float = 0.0

    # Calibration
    isotonic_points: list[tuple] | None = None
    platt_a: float = -1.0
    platt_b: float = 0.0

    # Normalization
    normalization_params: dict[str, float] | None = None

    # MI Feature Selection (for Hot Path to apply same selection)
    mi_selection_indices: list[int] | None = None
    mi_selection_method: str | None = None
    feature_count: int = 50

    # Metrics
    accuracy: float = 0.0
    auc_roc: float = 0.0
    training_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_type": self.model_type,
            "version": self.version,
            "trained_at_ms": self.trained_at_ms,
            "linear": {
                "coefficients": self.linear_coefficients,
                "bias": self.linear_bias,
            },
            "lightgbm": {
                "trees_json": self.lightgbm_trees,
                "learning_rate": self.lightgbm_learning_rate,
                "base_score": self.lightgbm_base_score,
            }
            if self.lightgbm_trees
            else None,
            "calibration": {
                "isotonic_points": self.isotonic_points,
                "platt_a": self.platt_a,
                "platt_b": self.platt_b,
            },
            "normalization": self.normalization_params,
            "mi_selection": {
                "indices": self.mi_selection_indices,
                "method": self.mi_selection_method,
            }
            if self.mi_selection_indices
            else None,
            "feature_count": self.feature_count,
            "metrics": {
                "accuracy": self.accuracy,
                "auc_roc": self.auc_roc,
                "training_samples": self.training_samples,
            },
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelExport":
        """Create from dictionary."""
        linear = data.get("linear", {})
        lgb = data.get("lightgbm") or {}
        calib = data.get("calibration", {})
        metrics = data.get("metrics", {})

        mi_data = data.get("mi_selection") or {}

        return cls(
            model_type=data.get("model_type", "linear"),
            version=data.get("version", 0),
            trained_at_ms=data.get("trained_at_ms", 0),
            linear_coefficients=linear.get("coefficients", [0.0] * 50),
            linear_bias=linear.get("bias", 0.0),
            lightgbm_trees=lgb.get("trees_json"),
            lightgbm_learning_rate=lgb.get("learning_rate", 0.1),
            lightgbm_base_score=lgb.get("base_score", 0.0),
            isotonic_points=calib.get("isotonic_points"),
            platt_a=calib.get("platt_a", -1.0),
            platt_b=calib.get("platt_b", 0.0),
            normalization_params=data.get("normalization"),
            mi_selection_indices=mi_data.get("indices"),
            mi_selection_method=mi_data.get("method"),
            feature_count=data.get("feature_count", 50),
            accuracy=metrics.get("accuracy", 0.0),
            auc_roc=metrics.get("auc_roc", 0.0),
            training_samples=metrics.get("training_samples", 0),
        )


class ModelUpdater:
    """Manages model updates and exports to Hot Path."""

    def __init__(
        self,
        artifact_store: "ModelArtifactStore",
        grpc_client: Any | None = None,  # HotPathClient
    ):
        self.artifact_store = artifact_store
        self.grpc_client = grpc_client
        self._current_version: int = 0
        self._last_export: ModelExport | None = None

    async def export_and_push(
        self,
        learner: "ProfitabilityLearner",
        apply_immediately: bool = True,
        reason: str = "scheduled_update",
    ) -> ModelExport:
        """Export trained model and push to Hot Path.

        Args:
            learner: Trained profitability learner
            apply_immediately: Whether Hot Path should apply immediately
            reason: Reason for the update

        Returns:
            ModelExport instance
        """
        if not learner.is_trained:
            raise RuntimeError("Learner is not trained")

        # Get export data from learner
        export_data = learner.get_export_data()

        # Create ModelExport
        export = self._create_export(export_data)

        # Save to artifact store
        version = self.artifact_store.save("profitability_model", export.to_dict())
        export.version = version
        self._current_version = version
        self._last_export = export

        logger.info(f"Saved model version {version}")

        # Push to Hot Path via gRPC
        if self.grpc_client:
            await self._push_to_hotpath(export, apply_immediately, reason)

        return export

    def _create_export(self, export_data: dict[str, Any]) -> ModelExport:
        """Create ModelExport from learner export data."""
        linear = export_data.get("linear", {})
        lgb = export_data.get("lightgbm", {})
        metrics_data = export_data.get("metrics", {})

        # Parse LightGBM trees if present
        lightgbm_trees = None
        if lgb and lgb.get("model_json"):
            lightgbm_trees = lgb["model_json"]

        # Extract MI selection if present
        mi_data = export_data.get("mi_selection")
        mi_indices = mi_data["indices"] if mi_data else None
        mi_method = mi_data.get("method") if mi_data else None

        return ModelExport(
            model_type=export_data.get("model_type", "linear"),
            version=export_data.get("version", self._current_version + 1),
            trained_at_ms=export_data.get("trained_at_ms", int(datetime.now().timestamp() * 1000)),
            linear_coefficients=linear.get("coefficients", [0.0] * 50),
            linear_bias=linear.get("bias", 0.0),
            lightgbm_trees=lightgbm_trees,
            lightgbm_learning_rate=lgb.get("learning_rate", 0.1) if lgb else 0.1,
            lightgbm_base_score=lgb.get("base_score", 0.0) if lgb else 0.0,
            isotonic_points=export_data.get("isotonic_points"),
            normalization_params=export_data.get("normalization"),
            mi_selection_indices=mi_indices,
            mi_selection_method=mi_method,
            feature_count=50,
            accuracy=metrics_data.get("accuracy", 0.0),
            auc_roc=metrics_data.get("auc_roc", 0.0),
            training_samples=metrics_data.get("training_samples", 0),
        )

    async def _push_to_hotpath(
        self,
        export: ModelExport,
        apply_immediately: bool,
        reason: str,
    ):
        """Push model update to Hot Path via gRPC."""
        try:
            # Construct the update message
            message = {
                "weights": export.to_dict(),
                "apply_immediately": apply_immediately,
                "reason": reason,
            }

            # Call gRPC method
            # Assuming grpc_client has an update_model_weights method
            client = self.grpc_client
            if client is not None and hasattr(client, "update_model_weights"):
                func = client.update_model_weights
                await func(message)
                logger.info(f"Pushed model v{export.version} to Hot Path: {reason}")
            else:
                logger.warning("gRPC client does not have update_model_weights method")

        except Exception as e:
            logger.error(f"Failed to push model to Hot Path: {e}")
            raise

    async def load_latest(self) -> ModelExport | None:
        """Load the latest model version from artifact store."""
        data = self.artifact_store.load_latest("profitability_model")
        if data:
            export = ModelExport.from_dict(data)
            self._current_version = export.version
            self._last_export = export
            return export
        return None

    async def get_version_history(self) -> list[dict[str, Any]]:
        """Get list of model versions."""
        return self.artifact_store.list_versions("profitability_model")

    @property
    def current_version(self) -> int:
        """Get current model version."""
        return self._current_version

    @property
    def last_export(self) -> ModelExport | None:
        """Get last exported model."""
        return self._last_export


class LearningScheduler:
    """Schedule periodic learning updates."""

    def __init__(
        self,
        updater: ModelUpdater,
        learner: "ProfitabilityLearner",
        outcome_tracker: Any,  # OutcomeTracker
        interval_hours: float = 6.0,
        min_new_samples: int = 50,
    ):
        self.updater = updater
        self.learner = learner
        self.outcome_tracker = outcome_tracker
        self.interval_hours = interval_hours
        self.min_new_samples = min_new_samples

        self._running = False
        self._last_training_samples = 0

    async def start(self):
        """Start the learning scheduler."""
        self._running = True
        logger.info(f"Learning scheduler started (interval: {self.interval_hours}h)")

        while self._running:
            try:
                await self._run_learning_cycle()
            except Exception as e:
                logger.error(f"Learning cycle error: {e}")

            # Wait for next cycle
            await asyncio.sleep(self.interval_hours * 3600)

    async def stop(self):
        """Stop the learning scheduler."""
        self._running = False
        logger.info("Learning scheduler stopped")

    async def _run_learning_cycle(self):
        """Run a single learning cycle."""
        logger.info("Starting learning cycle")

        # Get training data
        outcomes = await self.outcome_tracker.get_training_data(
            since_hours=int(self.interval_hours * 2),  # Overlap with previous data
            include_counterfactuals=True,
        )

        # Convert to dictionaries
        outcomes_data = [o.to_dict() for o in outcomes]

        # Check if we have enough new samples
        new_samples = len(outcomes_data) - self._last_training_samples
        if new_samples < self.min_new_samples:
            logger.info(
                f"Insufficient new samples ({new_samples} < "
                f"{self.min_new_samples}), skipping training"
            )
            return

        # Train model
        metrics = self.learner.train(outcomes_data)

        if metrics.accuracy > 0:
            # Export and push
            export = await self.updater.export_and_push(
                self.learner,
                apply_immediately=True,
                reason=f"scheduled_update_{int(datetime.now().timestamp())}",
            )

            self._last_training_samples = len(outcomes_data)

            logger.info(
                f"Learning cycle complete: v{export.version}, "
                f"accuracy={metrics.accuracy:.3f}, "
                f"samples={metrics.training_samples}"
            )
        else:
            logger.warning("Training failed, no model exported")

    async def run_once(self) -> ModelExport | None:
        """Run a single learning cycle immediately."""
        await self._run_learning_cycle()
        return self.updater.last_export


class OnlineLearningIntegrator:
    """Integrates OnlineLearner with production models for real-time adaptation.

    Connects online learning outcomes to the ensemble and HotPath:
    1. Receives trade outcomes from paper/live trading
    2. Updates OnlineLearner coefficients incrementally
    3. Pushes coefficient updates to production ensemble
    4. Optionally pushes to HotPath for immediate deployment

    Usage:
        integrator = OnlineLearningIntegrator(
            online_learner=learner,
            model_updater=updater,
        )

        # After each trade
        await integrator.process_outcome(trade_outcome)

        # Periodic sync to HotPath
        await integrator.sync_to_hotpath()
    """

    def __init__(
        self,
        online_learner: "OnlineLearner",
        model_updater: ModelUpdater | None = None,
        ensemble: Any | None = None,
        sync_interval_seconds: float = 300.0,
        min_updates_before_sync: int = 10,
    ):
        self.online_learner = online_learner
        self.model_updater = model_updater
        self.ensemble = ensemble
        self.sync_interval_seconds = sync_interval_seconds
        self.min_updates_before_sync = min_updates_before_sync

        self._last_sync_time = datetime.now()
        self._pending_sync = False
        self._total_outcomes_processed = 0

    async def process_outcome(self, outcome: "TradeOutcome") -> dict[str, Any]:
        """Process a single trade outcome through online learning.

        Args:
            outcome: Trade outcome with features and result

        Returns:
            Processing result with update status
        """
        self.online_learner.add_outcome(outcome)
        self._total_outcomes_processed += 1

        result = await self.online_learner.update_if_ready()

        if result and result.get("status") == "updated":
            self._pending_sync = True

            if self._should_sync_to_hotpath():
                await self.sync_to_hotpath()

        return {
            "processed": True,
            "total_outcomes": self._total_outcomes_processed,
            "update_result": result,
        }

    async def process_outcomes_batch(
        self,
        outcomes: list["TradeOutcome"],
    ) -> dict[str, Any]:
        """Process multiple outcomes at once.

        Args:
            outcomes: List of trade outcomes

        Returns:
            Batch processing result
        """
        for outcome in outcomes:
            self.online_learner.add_outcome(outcome)
            self._total_outcomes_processed += 1

        result = await self.online_learner.update()

        if result.get("status") == "updated":
            self._pending_sync = True
            await self.sync_to_hotpath()

        return {
            "batch_size": len(outcomes),
            "total_outcomes": self._total_outcomes_processed,
            "update_result": result,
        }

    def _should_sync_to_hotpath(self) -> bool:
        """Check if we should sync coefficients to HotPath."""
        if not self.model_updater:
            return False

        elapsed = (datetime.now() - self._last_sync_time).total_seconds()
        if elapsed < self.sync_interval_seconds:
            return False

        stats = self.online_learner.get_stats()
        if stats.get("n_updates", 0) < self.min_updates_before_sync:
            return False

        return True

    async def sync_to_hotpath(self) -> dict[str, Any]:
        """Sync current online learner coefficients to HotPath.

        Returns:
            Sync result with status
        """
        if not self.model_updater:
            return {"synced": False, "reason": "no_model_updater"}

        if not self._pending_sync:
            return {"synced": False, "reason": "no_pending_updates"}

        coefficients = self.online_learner.get_coefficients()

        export = ModelExport(
            model_type="online_linear",
            version=self.model_updater.current_version + 1,
            trained_at_ms=int(datetime.now().timestamp() * 1000),
            linear_coefficients=coefficients.tolist(),
            linear_bias=0.0,
            feature_count=len(coefficients),
            accuracy=self.online_learner.state.recent_accuracy,
            training_samples=self.online_learner.state.cumulative_samples,
        )

        if self.model_updater.grpc_client:
            await self.model_updater._push_to_hotpath(
                export,
                apply_immediately=True,
                reason="online_learning_update",
            )

        self._last_sync_time = datetime.now()
        self._pending_sync = False

        logger.info(
            f"Synced online learner to HotPath: "
            f"samples={self.online_learner.state.cumulative_samples}, "
            f"accuracy={self.online_learner.state.recent_accuracy:.3f}"
        )

        return {
            "synced": True,
            "version": export.version,
            "coefficients_shape": coefficients.shape,
            "recent_accuracy": self.online_learner.state.recent_accuracy,
        }

    def update_ensemble_coefficients(self) -> bool:
        """Update the production ensemble with current online coefficients.

        Returns:
            True if ensemble was updated
        """
        if not self.ensemble:
            return False

        coefficients = self.online_learner.get_coefficients()

        if hasattr(self.ensemble, "signal_generator") and hasattr(
            self.ensemble.signal_generator, "update_coefficients"
        ):
            self.ensemble.signal_generator.update_coefficients(coefficients)
            logger.info("Updated ensemble signal generator with online coefficients")
            return True

        return False

    def get_integrator_stats(self) -> dict[str, Any]:
        """Get statistics about the integrator."""
        return {
            "total_outcomes_processed": self._total_outcomes_processed,
            "pending_sync": self._pending_sync,
            "last_sync_time": self._last_sync_time.isoformat() if self._last_sync_time else None,
            "online_learner_stats": self.online_learner.get_stats(),
            "has_model_updater": self.model_updater is not None,
            "has_ensemble": self.ensemble is not None,
        }
