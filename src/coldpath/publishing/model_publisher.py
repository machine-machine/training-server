"""
Model Publisher - gRPC client for pushing model artifacts to HotPath.

Connects to HotPath's ModelArtifactService to push trained model artifacts
for hot-reload, query status, and trigger rollbacks.
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:
    import grpc

    from ..ipc import hotpath_pb2, hotpath_pb2_grpc

    GRPC_AVAILABLE = True
except (ImportError, TypeError):
    GRPC_AVAILABLE = False
    logger.warning(
        "gRPC proto stubs not available - model publishing disabled (regenerate with grpc_tools)"
    )


@dataclass
class PushResult:
    """Result of a model artifact push."""

    success: bool
    version: str = ""
    error: str = ""
    replaced_previous: bool = False
    previous_version: int | None = None


@dataclass
class ArtifactStatus:
    """Status of a model artifact in HotPath."""

    model_type: str
    active_version: int = 0
    schema_version: int = 0
    is_fallback: bool = True
    loaded_at: int = 0
    inference_count: int = 0
    error_count: int = 0
    avg_latency_us: float = 0.0


@dataclass
class RollbackResult:
    """Result of a rollback operation."""

    success: bool
    error: str = ""
    rolled_back_from: int = 0
    rolled_back_to: int = 0


class ModelPublisher:
    """gRPC client for pushing model artifacts to HotPath's ModelArtifactService."""

    def __init__(self, host: str = "localhost", port: int = 50051):
        self.host = host
        self.port = port
        self._channel = None
        self._stub = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Connect to HotPath gRPC server."""
        if not GRPC_AVAILABLE:
            logger.warning("gRPC not available, cannot connect")
            return False

        try:
            target = f"{self.host}:{self.port}"
            self._channel = grpc.aio.insecure_channel(target)

            # Test connectivity with a quick deadline
            try:
                await self._channel.channel_ready()
            except grpc.aio.AioRpcError:
                logger.debug(f"gRPC channel not ready at {target}")
                # Channel created but not yet connected - that's okay for lazy connect
                pass

            self._stub = hotpath_pb2_grpc.ModelArtifactServiceStub(self._channel)
            self._connected = True
            logger.info(f"ModelPublisher connected to {target}")
            return True

        except Exception as e:
            logger.warning(f"Failed to connect to HotPath gRPC: {e}")
            self._connected = False
            return False

    async def close(self):
        """Close the gRPC channel."""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None
            self._connected = False

    async def push_artifact(self, artifact: dict[str, Any]) -> PushResult:
        """Push a model artifact to HotPath for hot-reload.

        Args:
            artifact: Artifact dict from ArtifactExporter.export().

        Returns:
            PushResult with success status and version.
        """
        if not GRPC_AVAILABLE or not self._stub:
            return PushResult(success=False, error="gRPC not available or not connected")

        try:
            # Build proto ModelArtifact
            proto_artifact = self._build_proto_artifact(artifact)

            # Call PushArtifact RPC
            response = await self._stub.PushArtifact(
                proto_artifact,
                timeout=30.0,
            )

            result = PushResult(
                success=response.success,
                version=response.applied_version,
                error=response.error or "",
                replaced_previous=response.replaced_previous,
                previous_version=(
                    response.previous_version if response.HasField("previous_version") else None
                ),
            )

            if result.success:
                logger.info(
                    f"Artifact pushed successfully: {artifact['model_type']} v{result.version}"
                )
            else:
                logger.warning(f"Artifact push failed: {artifact['model_type']} - {result.error}")

            return result

        except grpc.aio.AioRpcError as e:
            error_msg = f"gRPC error: {e.code()} - {e.details()}"
            logger.error(f"Artifact push failed: {error_msg}")
            return PushResult(success=False, error=error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"Artifact push failed: {error_msg}")
            return PushResult(success=False, error=error_msg)

    async def get_status(self, model_types: list[str] | None = None) -> list[ArtifactStatus]:
        """Get current artifact status from HotPath.

        Args:
            model_types: List of model types to query. Empty/None = all.

        Returns:
            List of ArtifactStatus objects.
        """
        if not GRPC_AVAILABLE or not self._stub:
            return []

        try:
            request = hotpath_pb2.ArtifactStatusRequest(
                model_types=model_types or [],
            )

            response = await self._stub.GetArtifactStatus(
                request,
                timeout=10.0,
            )

            statuses = []
            for status in response.statuses:
                statuses.append(
                    ArtifactStatus(
                        model_type=status.model_type,
                        active_version=status.active_version,
                        schema_version=status.schema_version,
                        is_fallback=status.is_fallback,
                        loaded_at=status.loaded_at,
                        inference_count=status.inference_count,
                        error_count=status.error_count,
                        avg_latency_us=status.avg_latency_us,
                    )
                )

            return statuses

        except grpc.aio.AioRpcError as e:
            logger.error(f"Failed to get artifact status: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Failed to get artifact status: {e}")
            return []

    async def rollback(
        self,
        model_type: str,
        target_version: int | None = None,
    ) -> RollbackResult:
        """Trigger rollback to a previous model version.

        Args:
            model_type: Type of model to rollback.
            target_version: Specific version to rollback to. None = previous.

        Returns:
            RollbackResult with success status.
        """
        if not GRPC_AVAILABLE or not self._stub:
            return RollbackResult(success=False, error="gRPC not available or not connected")

        try:
            request = hotpath_pb2.RollbackRequest(
                model_type=model_type,
            )
            if target_version is not None:
                request.target_version = target_version

            response = await self._stub.RollbackArtifact(
                request,
                timeout=10.0,
            )

            result = RollbackResult(
                success=response.success,
                error=response.error or "",
                rolled_back_from=response.rolled_back_from,
                rolled_back_to=response.rolled_back_to,
            )

            if result.success:
                logger.info(
                    f"Rollback successful: {model_type} "
                    f"v{result.rolled_back_from} -> v{result.rolled_back_to}"
                )
            else:
                logger.warning(f"Rollback failed: {model_type} - {result.error}")

            return result

        except grpc.aio.AioRpcError as e:
            error_msg = f"gRPC error: {e.code()} - {e.details()}"
            logger.error(f"Rollback failed: {error_msg}")
            return RollbackResult(success=False, error=error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"Rollback failed: {error_msg}")
            return RollbackResult(success=False, error=error_msg)

    def _build_proto_artifact(self, artifact: dict[str, Any]):
        """Build proto ModelArtifact from artifact dict."""
        # Feature transforms
        transforms = []
        for t in artifact.get("feature_transforms", []):
            transforms.append(
                hotpath_pb2.FeatureTransform(
                    feature_name=t["feature_name"],
                    transform_type=t["transform_type"],
                    param1=t["param1"],
                    param2=t["param2"],
                )
            )

        # Metrics
        m = artifact.get("metrics", {})
        metrics = hotpath_pb2.ArtifactMetrics(
            accuracy=m.get("accuracy", 0.0),
            precision=m.get("precision", 0.0),
            recall=m.get("recall", 0.0),
            f1_score=m.get("f1_score", 0.0),
            auc_roc=m.get("auc_roc", 0.0),
            log_loss=m.get("log_loss", 0.0),
            train_samples=m.get("train_samples", 0),
            validation_samples=m.get("validation_samples", 0),
            calibration_error=m.get("calibration_error", 0.0),
        )

        # Calibration
        c = artifact.get("calibration", {})
        calibration = hotpath_pb2.CalibrationParams(
            calibration_method=c.get("calibration_method", "platt"),
            temperature=c.get("temperature", 1.0),
            platt_coeffs=c.get("platt_coeffs", [0.0, 0.0]),
        )

        # Weights
        w = artifact.get("weights", {})
        weights = hotpath_pb2.ArtifactWeights(
            student_type=w.get("student_type", "logistic"),
            linear_weights=w.get("linear_weights", []),
            bias=w.get("bias", 0.0),
        )

        return hotpath_pb2.ModelArtifact(
            model_type=artifact["model_type"],
            version=artifact["version"],
            schema_version=artifact["schema_version"],
            feature_signature=artifact["feature_signature"],
            feature_transforms=transforms,
            dataset_id=artifact.get("dataset_id", ""),
            metrics=metrics,
            created_at=artifact.get("created_at", 0),
            git_commit=artifact.get("git_commit", ""),
            calibration=calibration,
            compatibility_notes=artifact.get("compatibility_notes", ""),
            weights=weights,
            checksum=artifact.get("checksum", ""),
        )
