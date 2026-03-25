"""
Parameter publisher.

Pushes trained model parameters to Hot Path via Unix socket or gRPC.
"""

import asyncio
import json
import logging
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

hotpath_socket: str = os.getenv(
    "HOTPATH_SOCKET", os.path.join(tempfile.gettempdir(), "dexy-engine.sock")
)


@dataclass
class ModelParams:
    """Complete model parameters to push."""

    version: str
    timestamp: int

    # Bandit
    bandit_arm_weights: dict[str, float]
    recommended_arm: str

    # Slippage
    slippage_base_bps: float
    slippage_volatility_coef: float
    slippage_liquidity_coef: float = 0.5
    mev_penalty_bps: float = 0.0

    # Latency
    latency_mean_ms: float = 100.0
    latency_std_ms: float = 30.0
    latency_p99_ms: float = 500.0

    # Inclusion
    inclusion_prob_base: float = 0.85
    inclusion_congestion_decay: float = 0.3
    inclusion_priority_boost: float = 0.1

    # Fraud model (serialized)
    fraud_model_weights: bytes | None = None
    fraud_threshold: float = 0.5
    fraud_model_version: str = ""

    def to_proto_dict(self) -> dict:
        """Convert to proto-compatible dictionary format."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "bandit_arm_weights": self.bandit_arm_weights,
            "recommended_arm": self.recommended_arm,
            "slippage_base_bps": self.slippage_base_bps,
            "slippage_volatility_coef": self.slippage_volatility_coef,
            "slippage_liquidity_coef": self.slippage_liquidity_coef,
            "mev_penalty_bps": self.mev_penalty_bps,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_std_ms": self.latency_std_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "inclusion_prob_base": self.inclusion_prob_base,
            "inclusion_congestion_decay": self.inclusion_congestion_decay,
            "inclusion_priority_boost": self.inclusion_priority_boost,
            "fraud_model_weights": (
                self.fraud_model_weights.hex() if self.fraud_model_weights else ""
            ),
            "fraud_threshold": self.fraud_threshold,
            "fraud_model_version": self.fraud_model_version,
        }


class ParamPusher:
    """
    Pushes parameters to Hot Path.

    Supports two modes:
    1. Unix socket (default for local macOS deployment)
    2. gRPC (for distributed deployments)
    """

    def __init__(
        self,
        socket_path: str = hotpath_socket,
        grpc_address: str | None = None,
    ):
        self.socket_path = socket_path
        self.grpc_address = grpc_address
        self._socket_connected = False
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self) -> bool:
        """Establish connection to Hot Path."""
        if self.grpc_address:
            return await self._connect_grpc()
        else:
            return await self._connect_socket()

    async def _connect_socket(self) -> bool:
        """Connect via Unix socket."""
        socket_path = Path(self.socket_path)

        if not socket_path.exists():
            logger.warning(f"Socket not found at {self.socket_path}")
            return False

        try:
            self._reader, self._writer = await asyncio.open_unix_connection(str(socket_path))
            self._socket_connected = True
            logger.info(f"Connected to Hot Path via Unix socket: {self.socket_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to socket: {e}")
            return False

    async def _connect_grpc(self) -> bool:
        """Connect via gRPC."""
        try:
            import grpc.aio

            self._channel = grpc.aio.insecure_channel(
                self.grpc_address,
                options=[
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 10000),
                ],
            )

            # Verify connection
            await asyncio.wait_for(
                self._channel.channel_ready(),
                timeout=5.0,
            )

            logger.info(f"Connected to Hot Path via gRPC: {self.grpc_address}")
            return True
        except TimeoutError:
            logger.error(f"Timeout connecting to gRPC at {self.grpc_address}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect via gRPC: {e}")
            return False

    async def push(self, params: ModelParams) -> bool:
        """
        Push parameters to Hot Path.

        Returns True if push was successful.
        """
        if self.grpc_address:
            return await self._push_grpc(params)
        else:
            return await self._push_socket(params)

    async def _push_socket(self, params: ModelParams) -> bool:
        """Push via Unix socket using length-prefixed JSON protocol."""
        if not self._socket_connected or not self._writer:
            # Try to reconnect
            if not await self._connect_socket():
                return False

        try:
            # Build the command message
            message = {
                "type": "push_model_params",
                "request_id": f"push_{params.version}",
                "payload": params.to_proto_dict(),
            }

            # Serialize to JSON
            json_bytes = json.dumps(message).encode("utf-8")

            # Send with length prefix (4-byte big-endian)
            length_prefix = struct.pack(">I", len(json_bytes))
            self._writer.write(length_prefix + json_bytes)
            await self._writer.drain()

            # Read response
            response = await self._read_response()

            if response and response.get("success", False):
                logger.info(f"Successfully pushed params version {params.version}")
                return True
            else:
                error = response.get("error", "Unknown error") if response else "No response"
                logger.error(f"Failed to push params: {error}")
                return False

        except Exception as e:
            logger.error(f"Error pushing params via socket: {e}")
            self._socket_connected = False
            return False

    async def _read_response(self) -> dict | None:
        """Read length-prefixed JSON response."""
        if not self._reader:
            return None

        try:
            # Read length prefix
            length_bytes = await asyncio.wait_for(
                self._reader.readexactly(4),
                timeout=5.0,
            )
            length = struct.unpack(">I", length_bytes)[0]

            # Read payload
            payload_bytes = await asyncio.wait_for(
                self._reader.readexactly(length),
                timeout=5.0,
            )

            return json.loads(payload_bytes.decode("utf-8"))

        except TimeoutError:
            logger.warning("Timeout reading response from socket")
            return None
        except Exception as e:
            logger.error(f"Error reading socket response: {e}")
            return None

    async def _push_grpc(self, params: ModelParams) -> bool:
        """Push via gRPC using ParamPublisher service."""
        try:
            # Import and create stub
            from ..ipc import coldpath_pb2, coldpath_pb2_grpc

            stub = coldpath_pb2_grpc.ParamPublisherStub(self._channel)

            # Build proto message
            model_params = coldpath_pb2.ModelParams(
                version=params.version,
                timestamp=params.timestamp,
                bandit_arm_weights=params.bandit_arm_weights,
                recommended_arm=params.recommended_arm,
                slippage_base_bps=params.slippage_base_bps,
                slippage_volatility_coef=params.slippage_volatility_coef,
                slippage_liquidity_coef=params.slippage_liquidity_coef,
                mev_penalty_bps=params.mev_penalty_bps,
                latency_mean_ms=params.latency_mean_ms,
                latency_std_ms=params.latency_std_ms,
                latency_p99_ms=params.latency_p99_ms,
                inclusion_prob_base=params.inclusion_prob_base,
                inclusion_congestion_decay=params.inclusion_congestion_decay,
                inclusion_priority_boost=params.inclusion_priority_boost,
                fraud_model_weights=params.fraud_model_weights or b"",
                fraud_threshold=params.fraud_threshold,
                fraud_model_version=params.fraud_model_version,
            )

            # Call RPC
            response = await stub.PushParams(model_params)

            if response.accepted:
                logger.info(
                    f"Successfully pushed params version {params.version} "
                    f"(applied as {response.applied_version})"
                )
                return True
            else:
                logger.error(f"Hot Path rejected params: {response.error}")
                return False

        except ImportError:
            logger.warning("gRPC proto stubs not generated, falling back to socket")
            # Fall back to socket if proto stubs not available
            return await self._push_socket(params)
        except Exception as e:
            logger.error(f"gRPC push failed: {e}")
            return False

    async def close(self):
        """Close connection."""
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None
            self._socket_connected = False

        if hasattr(self, "_channel") and self._channel:
            await self._channel.close()


class ParamPushScheduler:
    """
    Schedules periodic parameter pushes.

    Integrates with the training loop to push updated params after training.
    """

    def __init__(
        self,
        pusher: ParamPusher,
        push_interval_seconds: int = 300,  # 5 minutes default
    ):
        self.pusher = pusher
        self.push_interval = push_interval_seconds
        self._running = False
        self._last_params: ModelParams | None = None
        self._last_push_time: float = 0

    async def start(self):
        """Start the push scheduler loop."""
        if self._running:
            return

        self._running = True

        # Connect on start
        await self.pusher.connect()

        logger.info(f"ParamPushScheduler started (interval: {self.push_interval}s)")

        while self._running:
            try:
                if self._last_params and self._should_push():
                    success = await self.pusher.push(self._last_params)
                    if success:
                        import time

                        self._last_push_time = time.time()

                await asyncio.sleep(1)  # Check every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in push scheduler: {e}")
                await asyncio.sleep(5)

    def stop(self):
        """Stop the scheduler."""
        self._running = False

    def _should_push(self) -> bool:
        """Check if we should push now."""
        import time

        return (time.time() - self._last_push_time) >= self.push_interval

    def update_params(self, params: ModelParams):
        """
        Update params to be pushed.

        Call this after training completes with updated parameters.
        """
        self._last_params = params
        logger.debug(f"Updated params for push: version={params.version}")

    async def push_now(self, params: ModelParams) -> bool:
        """Immediately push params (bypass scheduler)."""
        self._last_params = params
        success = await self.pusher.push(params)
        if success:
            import time

            self._last_push_time = time.time()
        return success
