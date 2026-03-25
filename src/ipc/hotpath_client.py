"""
HotPath gRPC client for training-server -> HotPath communication.

Pushes trained model artifacts to the HotPath engine for hot-reload.
Uses gRPC over TCP (unlike ColdPath which uses Unix socket).
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import grpc

from . import coldpath_pb2 as pb2
from . import coldpath_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)

# Default target for Docker-to-host communication
DEFAULT_GRPC_TARGET = "host.docker.internal:50051"


@dataclass
class PushResult:
    """Result of pushing an artifact to HotPath."""

    success: bool
    applied_version: Optional[str] = None
    error: Optional[str] = None
    replaced_previous: bool = False
    previous_version: Optional[int] = None


class HotPathClient:
    """gRPC client for pushing model artifacts to HotPath.

    Used by the training-server to push trained models to the HotPath
    engine after training completes. The HotPath engine hot-reloads
    the model without restarting.

    Environment variables:
        HOTPATH_GRPC_TARGET: gRPC target address (e.g., "host.docker.internal:50051")
    """

    def __init__(self, target: Optional[str] = None):
        """Initialize the HotPath client.

        Args:
            target: gRPC target address. If None, uses HOTPATH_GRPC_TARGET env var
                    or falls back to host.docker.internal:50051.
        """
        self.target = target or os.environ.get("HOTPATH_GRPC_TARGET", DEFAULT_GRPC_TARGET)
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[pb2_grpc.ModelArtifactServiceStub] = None
        self._connected = False

    def _get_channel(self) -> grpc.Channel:
        """Get or create a gRPC channel."""
        if self._channel is None:
            # Use insecure channel (no TLS for local/development)
            self._channel = grpc.insecure_channel(self.target)
        return self._channel

    def _get_stub(self) -> pb2_grpc.ModelArtifactServiceStub:
        """Get or create the ModelArtifactService stub."""
        if self._stub is None:
            self._stub = pb2_grpc.ModelArtifactServiceStub(self._get_channel())
        return self._stub

    def _build_artifact(self, artifact_dict: Dict[str, Any]) -> pb2.ModelArtifact:
        """Build a protobuf ModelArtifact from a dictionary.

        Args:
            artifact_dict: Dictionary with artifact fields matching coldpath.proto

        Returns:
            Protobuf ModelArtifact message
        """
        # Build feature transforms
        feature_transforms = []
        for ft in artifact_dict.get("feature_transforms", []):
            feature_transforms.append(
                pb2.FeatureTransform(
                    feature_name=ft.get("feature_name", ""),
                    transform_type=ft.get("transform_type", "none"),
                    param1=ft.get("param1", 0.0),
                    param2=ft.get("param2", 0.0),
                )
            )

        # Build metrics
        metrics_data = artifact_dict.get("metrics", {})
        metrics = pb2.ArtifactMetrics(
            accuracy=metrics_data.get("accuracy", 0.0),
            precision=metrics_data.get("precision", 0.0),
            recall=metrics_data.get("recall", 0.0),
            f1_score=metrics_data.get("f1_score", 0.0),
            auc_roc=metrics_data.get("auc_roc", 0.0),
            log_loss=metrics_data.get("log_loss", 0.0),
            train_samples=metrics_data.get("train_samples", 0),
            validation_samples=metrics_data.get("validation_samples", 0),
            calibration_error=metrics_data.get("calibration_error", 0.0),
        )

        # Build calibration params
        cal_data = artifact_dict.get("calibration", {})
        calibration = pb2.CalibrationParams(
            calibration_method=cal_data.get("calibration_method", "none"),
            temperature=cal_data.get("temperature", 1.0),
            platt_coeffs=cal_data.get("platt_coeffs", []),
        )

        # Build weights
        weights_data = artifact_dict.get("weights", {})
        onnx_model = weights_data.get("onnx_model")
        if onnx_model is None:
            onnx_model = b""
        elif isinstance(onnx_model, str):
            # Handle base64-encoded ONNX model
            import base64

            onnx_model = base64.b64decode(onnx_model)

        weights = pb2.ArtifactWeights(
            student_type=weights_data.get("student_type", "logistic"),
            linear_weights=weights_data.get("linear_weights", []),
            bias=weights_data.get("bias", 0.0),
            onnx_model=onnx_model,
        )

        # Build the complete artifact
        artifact = pb2.ModelArtifact(
            model_type=artifact_dict.get("model_type", "profitability"),
            version=artifact_dict.get("version", 1),
            schema_version=artifact_dict.get("schema_version", 1),
            feature_signature=artifact_dict.get("feature_signature", []),
            feature_transforms=feature_transforms,
            dataset_id=artifact_dict.get("dataset_id", ""),
            metrics=metrics,
            created_at=artifact_dict.get("created_at", 0),
            git_commit=artifact_dict.get("git_commit", ""),
            calibration=calibration,
            compatibility_notes=artifact_dict.get("compatibility_notes", ""),
            weights=weights,
            checksum=artifact_dict.get("checksum", ""),
        )

        return artifact

    def push_artifact(self, artifact_dict: Dict[str, Any], timeout: float = 30.0) -> PushResult:
        """Push a model artifact to HotPath.

        Args:
            artifact_dict: Dictionary with artifact fields matching coldpath.proto ModelArtifact
            timeout: RPC timeout in seconds (default: 30s)

        Returns:
            PushResult with success status and version info
        """
        try:
            stub = self._get_stub()
            artifact = self._build_artifact(artifact_dict)

            logger.info(
                f"Pushing artifact to HotPath at {self.target}: "
                f"type={artifact.model_type}, version={artifact.version}"
            )

            response: pb2.ArtifactPushResponse = stub.PushArtifact(
                artifact, timeout=timeout, wait_for_ready=True
            )

            result = PushResult(
                success=response.success,
                applied_version=response.applied_version if response.applied_version else None,
                error=response.error if response.error else None,
                replaced_previous=response.replaced_previous,
                previous_version=response.previous_version if response.previous_version else None,
            )

            if result.success:
                logger.info(
                    f"Artifact pushed successfully: version={result.applied_version}, "
                    f"replaced={result.replaced_previous}"
                )
            else:
                logger.error(f"Failed to push artifact: {result.error}")

            return result

        except grpc.RpcError as e:
            error_msg = f"gRPC error pushing artifact: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return PushResult(success=False, error=error_msg)

        except Exception as e:
            error_msg = f"Unexpected error pushing artifact: {e}"
            logger.error(error_msg)
            return PushResult(success=False, error=error_msg)

    def get_artifact_status(
        self, model_types: Optional[list[str]] = None, timeout: float = 10.0
    ) -> Optional[pb2.ArtifactStatusResponse]:
        """Get the status of loaded artifacts in HotPath.

        Args:
            model_types: List of model types to query, or None for all
            timeout: RPC timeout in seconds

        Returns:
            ArtifactStatusResponse or None on error
        """
        try:
            stub = self._get_stub()
            request = pb2.ArtifactStatusRequest(model_types=model_types or [])
            response = stub.GetArtifactStatus(request, timeout=timeout)
            return response

        except grpc.RpcError as e:
            logger.error(f"gRPC error getting artifact status: {e.code()} - {e.details()}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error getting artifact status: {e}")
            return None

    def rollback_artifact(
        self, model_type: str, target_version: Optional[int] = None, timeout: float = 10.0
    ) -> PushResult:
        """Rollback an artifact to a previous version.

        Args:
            model_type: Model type to rollback
            target_version: Target version, or None to roll back to previous
            timeout: RPC timeout in seconds

        Returns:
            PushResult with rollback status
        """
        try:
            stub = self._get_stub()
            request = pb2.RollbackRequest(model_type=model_type)
            if target_version is not None:
                request.target_version = target_version

            response: pb2.RollbackResponse = stub.RollbackArtifact(request, timeout=timeout)

            return PushResult(
                success=response.success,
                error=response.error if response.error else None,
                previous_version=response.rolled_back_from,
                applied_version=str(response.rolled_back_to),
            )

        except grpc.RpcError as e:
            error_msg = f"gRPC error rolling back artifact: {e.code()} - {e.details()}"
            logger.error(error_msg)
            return PushResult(success=False, error=error_msg)

        except Exception as e:
            error_msg = f"Unexpected error rolling back artifact: {e}"
            logger.error(error_msg)
            return PushResult(success=False, error=error_msg)

    def close(self):
        """Close the gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
            logger.info("Closed HotPath gRPC connection")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def push_artifact_sync(
    artifact_dict: Dict[str, Any],
    target: Optional[str] = None,
    timeout: float = 30.0,
) -> PushResult:
    """Synchronous helper to push an artifact to HotPath.

    This is a convenience function for use in Celery tasks
    where we don't need to manage the client lifecycle.

    Args:
        artifact_dict: Dictionary with artifact fields
        target: Optional gRPC target (uses env var if not set)
        timeout: RPC timeout in seconds

    Returns:
        PushResult with success status
    """
    with HotPathClient(target=target) as client:
        return client.push_artifact(artifact_dict, timeout=timeout)
