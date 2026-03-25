"""
Training Pipeline Integration - Connects AutoTrader to the training pipeline.

This module provides the glue between:
- AutoTrader outcome tracking
- TrainingPipelineMonitor
- TrainingPipelineOrchestrator
- HotPath data collection

When AutoTrader starts:
1. Connects to HotPath for candidate events
2. Records outcomes to the training monitor
3. Triggers training when thresholds are met
4. Uploads data to ColdPath for distillation
"""

import asyncio
import json
import logging
import math
from datetime import datetime
from typing import Any

from ..autotrader.monitoring import TrainingPipelineMonitor
from ..distillation.pipeline_orchestrator import PipelineConfig, TrainingPipelineOrchestrator
from ..storage import DatabaseManager

logger = logging.getLogger(__name__)


class TrainingIntegration:
    """Integrates AutoTrader with the training pipeline.

    Usage:
        integration = TrainingIntegration(db, monitor, orchestrator)
        await integration.start()  # Start background tasks

        # When outcome recorded:
        integration.record_outcome(is_win=True, pnl_pct=5.0)

        # Check if training should run:
        if integration.should_train():
            result = await integration.trigger_training()
    """

    def __init__(
        self,
        db: DatabaseManager,
        monitor: TrainingPipelineMonitor | None = None,
        orchestrator: TrainingPipelineOrchestrator | None = None,
        config: PipelineConfig | None = None,
        enable_synthetic_data: bool = True,
        synthetic_data_threshold: int = 50,
    ):
        self.db = db
        self.config = config or PipelineConfig()
        self.monitor = monitor or TrainingPipelineMonitor(
            min_outcomes_for_training=self.config.min_outcomes,
            training_interval_hours=self.config.training_interval_hours,
        )
        self.orchestrator = orchestrator or TrainingPipelineOrchestrator(
            db=db,
            monitor=self.monitor,
            config=self.config,
        )

        # Synthetic data generation settings
        self.enable_synthetic_data = enable_synthetic_data
        self.synthetic_data_threshold = synthetic_data_threshold

        # Background tasks
        self._scheduler_task: asyncio.Task | None = None
        self._running = False

        # Outcome buffer for batch recording
        self._outcome_buffer: list = []
        self._buffer_lock = asyncio.Lock()
        self._buffer_flush_interval = 60  # seconds
        self._scheduler_interval_seconds = 60
        self._last_training_sample_count = 0

        # Stats
        self._outcomes_recorded = 0
        self._trainings_triggered = 0
        self._last_training_at: datetime | None = None
        self._synthetic_samples_generated = 0

    async def start(self) -> None:
        """Start the training integration background tasks."""
        if self._running:
            return

        self._running = True

        # Start the training scheduler
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        logger.info("Training integration started")

    async def stop(self) -> None:
        """Stop the training integration."""
        self._running = False

        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        logger.info("Training integration stopped")

    def record_outcome(
        self,
        is_win: bool,
        pnl_pct: float = 0.0,
        pnl_sol: float = 0.0,
        mint: str | None = None,
        signal_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a trade outcome for training tracking.

        This increments the outcome counter in the training monitor,
        AND persists to database so _count_training_samples() can see it.

        Args:
            is_win: Whether the trade was profitable
            pnl_pct: P&L as percentage
            pnl_sol: P&L in SOL
            mint: Token mint (for logging and DB persistence)
            signal_id: Signal ID (for logging and DB persistence)
            metadata: Additional metadata
        """
        self.monitor.record_outcome(count=1)
        self._outcomes_recorded += 1

        # FIX: Persist to database so samples are counted for training
        if self.db and mint:
            try:
                timestamp_ms = int(datetime.now().timestamp() * 1000)
                outcome_data = {
                    "record_id": f"ti_{mint}_{timestamp_ms}",
                    "mint": mint,
                    "signal_id": signal_id or "",
                    "decision_timestamp_ms": timestamp_ms,
                    "regime": metadata.get("regime", "paper") if metadata else "paper",
                    "is_win": is_win,
                    "pnl_pct": pnl_pct,
                    "pnl_sol": pnl_sol,
                    "label_binary": 1 if is_win else 0,
                    "label_return": pnl_pct / 100.0 if pnl_pct else 0.0,
                    "label_quality": 1.0,
                    "token_status": "labeled",
                    "execution_mode": metadata.get("mode", "paper") if metadata else "paper",
                    "metadata": metadata or {},
                }
                # Fire-and-forget persistence (don't block trading)
                asyncio.create_task(self._persist_outcome(outcome_data))
            except Exception as e:
                logger.warning(f"Failed to queue outcome persistence: {e}")

        # Log for debugging
        logger.debug(
            f"Outcome recorded: win={is_win}, pnl_pct={pnl_pct:.2f}%, "
            f"total_outcomes={self._outcomes_recorded}, "
            f"since_last_train={self.monitor._outcomes_since_last_train}"
        )

    async def _persist_outcome(self, outcome_data: dict) -> None:
        """Persist outcome to database (async helper)."""
        try:
            await self.db.insert_training_outcome(outcome_data)
            logger.debug(f"Persisted outcome for {outcome_data.get('mint', '?')[:16]}")
        except Exception as e:
            logger.error(f"Failed to persist outcome to DB: {e}")

    def should_train(self) -> bool:
        """Check if training should be triggered."""
        return self.monitor.should_train()

    def _count_jsonl_samples(self) -> int:
        """Count labeled samples from JSONL files.

        Checks both Swift app and ColdPath data directories.
        """
        import json
        from pathlib import Path

        count = 0
        seen_keys = set()

        # Locations to check
        home = Path.home()
        swift_dir = home / "Library" / "Application Support" / "2DEXY" / "TrainingData"
        project_root = Path(__file__).resolve().parents[3]  # EngineColdPath/
        coldpath_dir = project_root / "data" / "training"

        locations = [
            swift_dir / "training_samples.jsonl",
            coldpath_dir / "training_samples.jsonl",
        ]

        for samples_path in locations:
            if not samples_path.exists():
                continue

            try:
                content = samples_path.read_text(encoding="utf-8")
                for line in content.strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        sample = json.loads(line)
                        # Only count labeled samples (with outcome)
                        if sample.get("outcome") and sample.get("mint"):
                            key = (sample.get("mint"), sample.get("timestamp_ms", 0))
                            if key not in seen_keys:
                                seen_keys.add(key)
                                count += 1
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                logger.debug(f"Error counting samples from {samples_path}: {e}")

        return count

    async def _count_training_samples(self) -> int:
        """Count persisted samples eligible for the training pipeline.

        Counts from:
        1. SQLite tables (training_outcomes, scan_outcomes, hotpath trades)
        2. JSONL files (Swift app, ColdPath data directory)
        """
        lookback_hours = max(1, int(math.ceil(self.config.teacher_lookback_hours)))
        try:
            training_outcomes, scan_outcomes, hotpath_trade_outcomes = await asyncio.gather(
                self.db.get_training_outcomes(
                    since_hours=lookback_hours,
                    limit=10000,
                ),
                self.db.get_scan_outcomes(
                    since_hours=lookback_hours,
                    limit=10000,
                ),
                self.db.get_hotpath_trade_outcomes(
                    since_hours=lookback_hours,
                    limit=10000,
                ),
            )
        except Exception as exc:
            logger.error("Failed to count training samples: %s", exc, exc_info=True)
            return 0

        sample_ids = set()
        for outcome in training_outcomes:
            sample_ids.add(
                (
                    "training",
                    outcome.get("record_id") or outcome.get("mint"),
                    outcome.get("decision_timestamp_ms", 0),
                )
            )
        for outcome in scan_outcomes:
            sample_ids.add(
                (
                    "scan",
                    outcome.get("mint"),
                    outcome.get("timestamp_ms", 0),
                )
            )
        for outcome in hotpath_trade_outcomes:
            sample_ids.add(
                (
                    "hotpath",
                    outcome.get("mint"),
                    outcome.get("timestamp_ms", 0),
                )
            )

        db_count = len(sample_ids)

        # Also count samples from JSONL files
        jsonl_count = self._count_jsonl_samples()

        total_count = db_count + jsonl_count

        if jsonl_count > 0:
            logger.debug(f"Training samples: {db_count} DB + {jsonl_count} JSONL = {total_count}")

        return total_count

    async def evaluate_training_readiness(self) -> dict[str, Any]:
        """Evaluate whether the training threshold has been met and log the reason."""
        import os

        observations = await self._count_training_samples()
        new_observations = max(0, observations - self._last_training_sample_count)

        # FIX: Allow MIN_OBSERVATIONS_TRAINING env override
        env_threshold = os.environ.get("MIN_OBSERVATIONS_TRAINING")
        if env_threshold:
            threshold = max(1, int(env_threshold))
        else:
            threshold = max(1, int(self.config.min_outcomes))

        self.monitor._outcomes_since_last_train = new_observations

        if self.monitor._current_status.name == "RUNNING":
            reason = "training_in_progress"
            can_train = False
        elif new_observations < threshold:
            reason = f"below_threshold({new_observations}<{threshold})"
            can_train = False
        elif self.monitor.should_train():
            reason = "threshold_met"
            can_train = True
        elif self.monitor._consecutive_failures >= self.monitor.max_consecutive_failures:
            reason = "failure_cooldown"
            can_train = False
        elif self.monitor._last_train_at is not None:
            # FIX: Check if interval elapsed OR first training
            min_interval_hours = float(os.environ.get("TRAINING_INTERVAL_HOURS", "1.0"))
            elapsed_hours = (datetime.now() - self.monitor._last_train_at).total_seconds() / 3600

            if elapsed_hours >= min_interval_hours:
                reason = "interval_elapsed"
                can_train = self.monitor.should_train()
            elif self._trainings_triggered == 0:
                # First training attempt — allow it
                reason = "first_training_attempt"
                can_train = self.monitor.should_train()
            else:
                reason = (
                    f"training_interval_not_elapsed({elapsed_hours:.1f}h < {min_interval_hours}h)"
                )
                can_train = False
        else:
            reason = "monitor_blocked"
            can_train = False

        # FIX: Detailed logging for debugging
        logger.info(
            f"Training readiness: observations={observations}, "
            f"new_observations={new_observations}, threshold={threshold}, "
            f"can_train={can_train}, reason={reason}, "
            f"trainings_triggered={self._trainings_triggered}"
        )

        if not can_train and "below_threshold" in reason:
            missing = threshold - new_observations
            logger.info(
                f"Training blocked: need {missing} more observations. "
                f"Keep autotrader running to collect samples."
            )

        return {
            "observations": observations,
            "new_observations": new_observations,
            "threshold": threshold,
            "can_train": can_train,
            "reason": reason,
        }

    async def trigger_training(self, force: bool = False) -> dict[str, Any]:
        """Trigger a training run.

        Args:
            force: Bypass threshold checks

        Returns:
            Training result dict
        """
        # Check if we need synthetic data
        observations = await self._count_training_samples()

        if self.enable_synthetic_data and observations < self.config.min_outcomes:
            logger.info(
                "Insufficient real data (%d < %d), generating synthetic samples...",
                observations,
                self.config.min_outcomes,
            )
            synthetic_count = await self._generate_and_store_synthetic_data(
                target_count=self.config.min_outcomes - observations
            )
            logger.info("Generated %d synthetic training samples", synthetic_count)

        result = await self.orchestrator.run_training(force=force)

        if result.get("success"):
            self._trainings_triggered += 1
            self._last_training_at = datetime.now()
            self._last_training_sample_count = await self._count_training_samples()
            logger.info(f"Training completed: v{result.get('model_version')}")
        else:
            logger.error("Training failed: %s", result.get("message"))

        return result

    async def _generate_and_store_synthetic_data(
        self,
        target_count: int,
        use_llm_labels: bool = False,  # Default to rule-based for speed
    ) -> int:
        """Generate synthetic training data and store in database.

        Args:
            target_count: Number of samples to generate
            use_llm_labels: Use FinGPT for labeling (slower but more realistic)

        Returns:
            Number of samples generated and stored
        """
        from ..training.synthetic_data_generator import SyntheticDataGenerator

        try:
            generator = SyntheticDataGenerator(use_llm_labels=use_llm_labels)
            outcomes = await generator.generate_batch(target_count)

            # Store in database
            stored_count = 0
            for outcome in outcomes:
                try:
                    generator.outcome_to_dict(outcome)
                    # Store as training outcome
                    await self.db.insert_training_outcome(
                        {
                            "record_id": f"synth_{outcome.mint[:16]}_{outcome.timestamp_ms}",
                            "mint": outcome.mint,
                            "decision_timestamp_ms": outcome.timestamp_ms,
                            "regime": "synthetic",
                            "features_json": json.dumps(outcome.features),
                            "model_version": 0,
                            "model_score": outcome.model_score,
                            "execution_mode": outcome.execution_mode,
                            "pnl_sol": outcome.pnl_pct * 0.01,
                            "pnl_pct": outcome.pnl_pct,
                            "token_status": "synthetic",
                            "label_binary": outcome.label_binary,
                            "label_return": outcome.label_return,
                            "label_quality": 0.5,  # Medium quality for synthetic
                        }
                    )
                    stored_count += 1
                except Exception as e:
                    logger.debug(f"Failed to store synthetic outcome: {e}")

            self._synthetic_samples_generated += stored_count
            self.monitor.record_outcome(count=stored_count)

            return stored_count

        except Exception as e:
            logger.error("Failed to generate synthetic data: %s", e, exc_info=True)
            return 0

    async def _scheduler_loop(self) -> None:
        """Background loop that checks for training triggers."""
        while self._running:
            try:
                await asyncio.sleep(self._scheduler_interval_seconds)

                evaluation = await self.evaluate_training_readiness()
                logger.info(
                    "Train check: observations=%s, threshold=%s, can_train=%s, reason=%s",
                    evaluation["observations"],
                    evaluation["threshold"],
                    evaluation["can_train"],
                    evaluation["reason"],
                )

                if evaluation["can_train"]:
                    logger.info(
                        f"Auto-training triggered: {evaluation['observations']} samples collected"
                    )
                    await self.trigger_training()
                elif (
                    evaluation["reason"].startswith("below_threshold")
                    and self.enable_synthetic_data
                ):
                    missing = max(0, evaluation["threshold"] - evaluation["observations"])
                    if missing > 0:
                        logger.info(
                            "Generating %s synthetic samples to reach training threshold",
                            missing,
                        )
                        generated = await self._generate_and_store_synthetic_data(
                            target_count=missing
                        )
                        logger.info("Synthetic samples generated: %s", generated)

                    # Force training after synthetic backfill so the loop never stalls
                    await self.trigger_training(force=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduler error: %s", e, exc_info=True)
                await asyncio.sleep(60)

    def get_status(self) -> dict[str, Any]:
        """Get training integration status."""
        return {
            "running": self._running,
            "outcomes_recorded": self._outcomes_recorded,
            "trainings_triggered": self._trainings_triggered,
            "synthetic_samples_generated": self._synthetic_samples_generated,
            "last_training_at": self._last_training_at.isoformat()
            if self._last_training_at
            else None,
            "monitor": self.monitor.get_status().to_dict(),
            "should_train_now": self.should_train(),
            "config": {
                "min_outcomes": self.config.min_outcomes,
                "training_interval_hours": self.config.training_interval_hours,
                "min_correlation": self.config.min_correlation,
                "max_calibration_error": self.config.max_calibration_error,
                "enable_synthetic_data": self.enable_synthetic_data,
            },
        }

    async def check_and_train(self) -> dict[str, Any] | None:
        """Check if training should run and trigger if so.

        Returns:
            Training result if triggered, None otherwise
        """
        evaluation = await self.evaluate_training_readiness()
        logger.info(
            "Train check: observations=%s, threshold=%s, can_train=%s, reason=%s",
            evaluation["observations"],
            evaluation["threshold"],
            evaluation["can_train"],
            evaluation["reason"],
        )
        if evaluation["can_train"]:
            return await self.trigger_training()
        return None


# Global singleton
_training_integration: TrainingIntegration | None = None


def get_training_integration(db: DatabaseManager | None = None) -> TrainingIntegration:
    """Get or create the training integration singleton."""
    global _training_integration

    if _training_integration is None:
        if db is None:
            raise ValueError("Database required for first initialization")
        _training_integration = TrainingIntegration(db=db)

    return _training_integration


async def setup_training_integration(db: DatabaseManager) -> TrainingIntegration:
    """Set up and start the training integration.

    This should be called during ColdPath startup.
    """
    integration = get_training_integration(db)
    await integration.start()
    return integration
