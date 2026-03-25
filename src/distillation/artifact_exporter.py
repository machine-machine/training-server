"""
Artifact Exporter - Build proto-compatible model artifact with SHA256 checksum.

Produces a dict matching the coldpath.proto ModelArtifact message,
ready for gRPC push to HotPath.
"""

import hashlib
import logging
import struct
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

from .student_distiller import StudentResult

logger = logging.getLogger(__name__)


class ArtifactExporter:
    """Export student model to HotPath-compatible ModelArtifact format."""

    SCHEMA_VERSION = 1

    def export(
        self,
        student: StudentResult,
        version: int,
        model_type: str = "profitability",
        dataset_id: str = "",
    ) -> Dict[str, Any]:
        """Build ModelArtifact dict matching coldpath.proto.

        Args:
            student: Distilled student model result.
            version: Monotonically increasing version number.
            model_type: Model type identifier.
            dataset_id: Training dataset identifier.

        Returns:
            Complete artifact dict ready for gRPC push.
        """
        git_commit = self._get_git_commit()

        # Build feature transforms list
        feature_transforms = [t.to_dict() for t in student.transforms]

        # Build metrics
        metrics = {
            "accuracy": student.metrics.get("accuracy", 0.0),
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "auc_roc": student.metrics.get("teacher_auc", 0.0),
            "log_loss": 0.0,
            "train_samples": int(student.metrics.get("train_samples", 0)),
            "validation_samples": int(student.metrics.get("val_samples", 0)),
            "calibration_error": student.metrics.get("calibration_error", 0.0),
        }

        # Build calibration params
        calibration = {
            "calibration_method": "platt",
            "temperature": student.calibration.temperature,
            "platt_coeffs": [student.calibration.platt_a, student.calibration.platt_b],
        }

        # Build weights
        weights = {
            "student_type": "logistic",
            "linear_weights": student.weights.tolist(),
            "bias": student.bias,
            "onnx_model": b"",
        }

        # Build artifact
        artifact = {
            "model_type": model_type,
            "version": version,
            "schema_version": self.SCHEMA_VERSION,
            "feature_signature": student.feature_names,
            "feature_transforms": feature_transforms,
            "dataset_id": dataset_id,
            "metrics": metrics,
            "created_at": int(datetime.now().timestamp() * 1000),
            "git_commit": git_commit,
            "calibration": calibration,
            "compatibility_notes": (
                f"Distilled from teacher ensemble, "
                f"correlation={student.metrics.get('correlation', 0):.3f}, "
                f"agreement={student.metrics.get('agreement', 0):.3f}"
            ),
            "weights": weights,
        }

        # Compute checksum matching Rust's compute_checksum()
        artifact["checksum"] = self._compute_checksum(artifact)

        logger.info(
            f"Exported artifact: type={model_type}, version={version}, "
            f"checksum={artifact['checksum'][:16]}..."
        )

        return artifact

    def _compute_checksum(self, artifact: Dict[str, Any]) -> str:
        """Compute SHA256 checksum matching Rust's compute_checksum().

        Hash: model_type + version(u32 LE) + schema_version(u32 LE) +
              feature_names + weights(f64 LE each) + bias(f64 LE) +
              confidence_threshold(f64 LE) + temperature(f64 LE) +
              dataset_id + git_commit
        """
        h = hashlib.sha256()

        # model_type as bytes
        h.update(artifact["model_type"].encode("utf-8"))

        # version as u32 little-endian
        h.update(struct.pack("<I", artifact["version"]))

        # schema_version as u32 little-endian
        h.update(struct.pack("<I", artifact["schema_version"]))

        # feature_names concatenated
        for name in artifact["feature_signature"]:
            h.update(name.encode("utf-8"))

        # weights as f64 little-endian each
        for w in artifact["weights"]["linear_weights"]:
            h.update(struct.pack("<d", w))

        # bias as f64 little-endian
        h.update(struct.pack("<d", artifact["weights"]["bias"]))

        # confidence_threshold as f64 little-endian (default 0.5)
        confidence_threshold = 0.5
        h.update(struct.pack("<d", confidence_threshold))

        # temperature as f64 little-endian
        h.update(struct.pack("<d", artifact["calibration"]["temperature"]))

        # dataset_id as bytes
        h.update(artifact["dataset_id"].encode("utf-8"))

        # git_commit as bytes
        h.update(artifact["git_commit"].encode("utf-8"))

        return h.hexdigest()

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return "unknown"
