"""
Training Pipeline Orchestrator - Coordinates the full training pipeline with monitoring.

Orchestrates:
1. Data collection from trade outcomes
2. Teacher model training (50-feature ensemble)
3. Student distillation (7-feature logistic)
4. Quality validation (correlation, calibration)
5. Artifact export and push to HotPath

Integrates with TrainingPipelineMonitor for status tracking and alerting.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..autotrader.monitoring import (
    TrainingPipelineMonitor,
    TrainingStage,
    TrainingStatus,
)
from ..storage import DatabaseManager
from .artifact_exporter import ArtifactExporter
from .student_distiller import StudentConfig, StudentDistiller, StudentResult
from .teacher_trainer import TeacherConfig, TeacherResult, TeacherTrainer

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""

    # Minimum outcomes to trigger training
    min_outcomes: int = 200

    # Training interval (hours)
    training_interval_hours: float = 1.0

    # Teacher config
    teacher_lookback_hours: int = 168  # 1 week

    # Quality gates
    min_correlation: float = 0.85
    max_calibration_error: float = 0.05

    # HotPath push
    push_to_hotpath: bool = True
    hotpath_model_dir: str = "models/profitability"


class TrainingPipelineOrchestrator:
    """Orchestrates the full ML training pipeline with monitoring.

    Usage:
        orchestrator = TrainingPipelineOrchestrator(db, monitor)

        # Check if training should run
        if orchestrator.should_train():
            result = await orchestrator.run_training()

    The orchestrator updates the TrainingPipelineMonitor at each stage,
    enabling real-time monitoring via API endpoints.
    """

    def __init__(
        self,
        db: DatabaseManager,
        monitor: TrainingPipelineMonitor | None = None,
        config: PipelineConfig | None = None,
    ):
        self.db = db
        self.monitor = monitor or TrainingPipelineMonitor()
        self.config = config or PipelineConfig()

        # Pipeline state
        self._teacher_result: TeacherResult | None = None
        self._student_result: StudentResult | None = None
        self._current_model_version: int = 0
        self._hotpath_push_state: dict[str, Any] = {
            "state": "idle",
            "updated_at": None,
            "version": None,
            "error": None,
        }

    def should_train(self) -> bool:
        """Check if training should be triggered."""
        return self.monitor.should_train()

    async def run_training(self, force: bool = False) -> dict[str, Any]:
        """Run the full training pipeline.

        Args:
            force: Bypass threshold checks

        Returns:
            Dict with training results and status
        """
        # Check if already training
        if self.monitor._current_status == TrainingStatus.RUNNING:
            return {
                "success": False,
                "message": "Training already in progress",
                "run_id": self.monitor._current_run.run_id if self.monitor._current_run else None,
            }

        # Check thresholds unless forced
        if not force and not self.should_train():
            return {
                "success": False,
                "message": "Training thresholds not met",
                "outcomes_available": self.monitor._outcomes_since_last_train,
                "min_outcomes": self.config.min_outcomes,
            }

        # Get outcome count
        try:
            outcomes = await self.db.get_training_outcomes(
                since_hours=self.config.teacher_lookback_hours,
                limit=10000,
            )
            outcomes_available = len(outcomes)
        except Exception as e:
            logger.error(f"Failed to get outcome count: {e}")
            outcomes_available = 0

        # Start training run
        run_id = self.monitor.start_training(outcomes_available)

        try:
            # Stage 1: Data Collection (already done above)
            self.monitor.update_stage(
                TrainingStage.DATA_COLLECTION,
                {
                    "outcomes_loaded": outcomes_available,
                },
            )

            # Stage 2: Teacher Training
            teacher_result = await self._train_teacher()
            if teacher_result is None:
                self.monitor.complete_training(
                    success=False, error_message="Teacher training failed or insufficient data"
                )
                return {
                    "success": False,
                    "run_id": run_id,
                    "message": "Teacher training failed",
                }

            self._teacher_result = teacher_result
            self.monitor.update_stage(
                TrainingStage.TEACHER_TRAINING,
                {
                    "metrics": {
                        "auc": float(teacher_result.metrics.auc_roc),
                        "accuracy": float(teacher_result.metrics.accuracy),
                    }
                },
            )

            # Stage 3: Student Distillation
            student_result = await self._distill_student(teacher_result)
            if student_result is None:
                self.monitor.complete_training(
                    success=False, error_message="Student distillation failed quality gates"
                )
                return {
                    "success": False,
                    "run_id": run_id,
                    "message": "Student distillation failed quality gates",
                    "teacher_metrics": {
                        "auc": float(teacher_result.metrics.auc_roc),
                    },
                }

            self._student_result = student_result
            self.monitor.update_stage(
                TrainingStage.STUDENT_DISTILLATION,
                {
                    "metrics": student_result.metrics,
                },
            )

            # Stage 4: Quality Validation
            quality_passed = self._validate_quality(student_result)
            self.monitor.update_stage(
                TrainingStage.QUALITY_VALIDATION,
                {
                    "passed": quality_passed,
                    "correlation": student_result.metrics.get("correlation"),
                    "calibration_error": student_result.metrics.get("calibration_error"),
                },
            )

            if not quality_passed:
                self.monitor.complete_training(
                    success=False,
                    error_message=(
                        f"Quality validation failed: "
                        f"correlation={student_result.metrics.get('correlation', 0):.3f}"
                    ),
                )
                return {
                    "success": False,
                    "run_id": run_id,
                    "message": "Quality validation failed",
                    "metrics": student_result.metrics,
                }

            # Stage 5: Model Export
            model_version = self._current_model_version + 1
            model_type = await self._export_model(student_result, model_version)

            self.monitor.update_stage(
                TrainingStage.MODEL_EXPORT,
                {
                    "version": model_version,
                    "model_type": model_type,
                },
            )

            # Stage 6: Push to HotPath
            if self.config.push_to_hotpath:
                push_success = await self._push_to_hotpath(model_type)
                self.monitor.update_stage(
                    TrainingStage.HOTPATH_PUSH,
                    {
                        "success": push_success,
                        "state": self._hotpath_push_state["state"],
                        "error": self._hotpath_push_state["error"],
                    },
                )

                if not push_success:
                    logger.warning("HotPath push failed, but model exported successfully")

            # Complete training
            self._current_model_version = model_version
            self.monitor.complete_training(success=True)

            logger.info(
                f"✅ Training pipeline completed: v{model_version}, "
                f"correlation={student_result.metrics.get('correlation', 0):.3f}"
            )

            return {
                "success": True,
                "run_id": run_id,
                "model_version": model_version,
                "message": "Training completed successfully",
                "teacher_metrics": {
                    "auc": float(teacher_result.metrics.auc_roc),
                    "accuracy": float(teacher_result.metrics.accuracy),
                },
                "student_metrics": student_result.metrics,
            }

        except Exception as e:
            error_msg = f"Training pipeline error: {e}"
            logger.error(error_msg, exc_info=True)
            self.monitor.complete_training(success=False, error_message=error_msg)

            return {
                "success": False,
                "run_id": run_id,
                "message": error_msg,
            }

    async def _train_teacher(self) -> TeacherResult | None:
        """Train the teacher ensemble."""
        try:
            trainer = TeacherTrainer(
                db=self.db,
                config=TeacherConfig(
                    lookback_hours=self.config.teacher_lookback_hours,
                    min_outcomes=self.config.min_outcomes,
                ),
            )

            result = await trainer.train()
            return result

        except Exception as e:
            logger.error(f"Teacher training failed: {e}", exc_info=True)
            return None

    async def _distill_student(self, teacher: TeacherResult) -> StudentResult | None:
        """Distill teacher into student model."""
        try:
            distiller = StudentDistiller(
                config=StudentConfig(
                    min_correlation=self.config.min_correlation,
                    max_calibration_error=self.config.max_calibration_error,
                ),
            )

            result = distiller.distill(teacher)
            return result

        except Exception as e:
            logger.error(f"Student distillation failed: {e}", exc_info=True)
            return None

    def _validate_quality(self, student: StudentResult) -> bool:
        """Validate student model quality."""
        correlation = student.metrics.get("correlation", 0)
        calibration_error = student.metrics.get("calibration_error", 1)

        if correlation < self.config.min_correlation:
            logger.warning(
                f"Quality check failed: correlation {correlation:.3f} < "
                f"{self.config.min_correlation}"
            )
            return False

        if calibration_error > self.config.max_calibration_error:
            logger.warning(
                f"Quality check failed: calibration_error {calibration_error:.4f} > "
                f"{self.config.max_calibration_error}"
            )
            return False

        return True

    async def _export_model(self, student: StudentResult, version: int) -> str:
        """Export student model to artifact.

        Returns the model_type (used as identifier for the artifact).
        """
        try:
            exporter = ArtifactExporter()

            artifact = exporter.export(
                student=student,
                version=version,
                model_type="profitability",
                dataset_id=f"distill_v{version}_{datetime.now().strftime('%Y%m%d')}",
            )

            # Log the export
            logger.info(
                f"Model exported: type=profitability, version={version}, "
                f"checksum={artifact.get('checksum', 'unknown')[:16]}..."
            )

            # Store artifact for HotPath push
            self._last_artifact = artifact

            return "profitability"

        except Exception as e:
            logger.error(f"Model export failed: {e}", exc_info=True)
            raise

    async def _push_to_hotpath(self, model_type: str) -> bool:
        """Push model to HotPath via gRPC."""
        try:
            if not hasattr(self, "_last_artifact") or self._last_artifact is None:
                logger.warning("No artifact to push to HotPath")
                self._set_hotpath_push_state("failed", error="No exported artifact available")
                return False

            artifact = self._last_artifact
            version = artifact.get("version", 0)

            self._set_hotpath_push_state("requested", version=version)
            logger.info(
                "HotPath push requested but no publisher is configured: type=%s version=%s",
                model_type,
                version,
            )
            self._set_hotpath_push_state(
                "failed",
                version=version,
                error="Model publisher not configured; artifact not sent",
            )
            return False

        except Exception as e:
            logger.error(f"HotPath push failed: {e}", exc_info=True)
            self._set_hotpath_push_state("failed", error=str(e))
            return False

    def get_status(self) -> dict[str, Any]:
        """Get current pipeline status."""
        summary = self.monitor.get_summary()
        summary["hotpath_push"] = dict(self._hotpath_push_state)
        return summary

    def _set_hotpath_push_state(
        self,
        state: str,
        *,
        version: int | None = None,
        error: str | None = None,
    ) -> None:
        self._hotpath_push_state = {
            "state": state,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "version": version,
            "error": error,
        }

    def record_outcome(self, count: int = 1) -> None:
        """Record new training outcome (for triggering retraining)."""
        self.monitor.record_outcome(count)


# Background task for scheduled training
async def training_scheduler(
    orchestrator: TrainingPipelineOrchestrator,
    check_interval_seconds: int = 300,  # Check every 5 minutes
) -> None:
    """Background task that checks for training triggers periodically.

    Usage:
        orchestrator = TrainingPipelineOrchestrator(db, monitor)
        asyncio.create_task(training_scheduler(orchestrator))
    """
    logger.info("Training scheduler started")

    while True:
        try:
            await asyncio.sleep(check_interval_seconds)

            # Check if we should train
            if orchestrator.should_train():
                logger.info("Scheduled training triggered")
                result = await orchestrator.run_training()

                if result.get("success"):
                    logger.info(f"Scheduled training completed: v{result.get('model_version')}")
                else:
                    logger.warning(f"Scheduled training failed: {result.get('message')}")

        except asyncio.CancelledError:
            logger.info("Training scheduler stopped")
            break
        except Exception as e:
            logger.error(f"Training scheduler error: {e}", exc_info=True)
            await asyncio.sleep(60)  # Wait before retrying
