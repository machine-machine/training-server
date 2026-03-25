"""
Distillation pipeline for compressing ColdPath teacher models into HotPath-compatible students.

Pipeline: TeacherTrainer -> StudentDistiller -> ArtifactExporter -> ModelPublisher
"""

from .artifact_exporter import ArtifactExporter
from .pipeline_orchestrator import (
    PipelineConfig,
    TrainingPipelineOrchestrator,
    training_scheduler,
)
from .student_distiller import StudentConfig, StudentDistiller, StudentResult
from .teacher_trainer import TeacherConfig, TeacherResult, TeacherTrainer

__all__ = [
    "TeacherTrainer",
    "TeacherConfig",
    "TeacherResult",
    "StudentDistiller",
    "StudentConfig",
    "StudentResult",
    "ArtifactExporter",
    "TrainingPipelineOrchestrator",
    "PipelineConfig",
    "training_scheduler",
]
