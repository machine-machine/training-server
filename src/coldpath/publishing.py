"""
Parameter publishing to Hot Path engine.

Sends optimized model parameters via Unix socket.
"""

import asyncio
import json
import logging
import struct
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ParameterPublisher:
    """Publishes model parameters to the Hot Path engine via Unix socket.

    Uses length-prefixed JSON protocol for reliable message delivery.
    """

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False
        self._lock = asyncio.Lock()
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0

    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected and self._writer is not None

    async def connect(self, timeout: float = 5.0):
        """Connect to the Hot Path Unix socket."""
        async with self._lock:
            if self._connected:
                return

            socket_path = Path(self.socket_path)
            if not socket_path.exists():
                raise ConnectionError(f"Socket does not exist: {self.socket_path}")

            try:
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_unix_connection(self.socket_path),
                    timeout=timeout,
                )
                self._connected = True
                self._reconnect_delay = 1.0  # Reset delay on successful connect
                logger.info(f"Connected to Hot Path at {self.socket_path}")

            except TimeoutError:
                raise ConnectionError(f"Connection timeout to {self.socket_path}") from None
            except Exception as e:
                raise ConnectionError(f"Failed to connect: {e}") from e

    async def close(self):
        """Close the connection."""
        async with self._lock:
            if self._writer:
                try:
                    self._writer.close()
                    await self._writer.wait_closed()
                except Exception as e:
                    logger.debug(f"Error closing writer: {e}")
                finally:
                    self._writer = None

            if self._reader:
                try:
                    # Consume remaining data with timeout to prevent hanging
                    await asyncio.wait_for(self._reader.read(), timeout=0.1)
                except (TimeoutError, asyncio.IncompleteReadError):
                    pass
                except Exception as e:
                    logger.debug(f"Error consuming reader data: {e}")
                finally:
                    self._reader = None

            self._connected = False
            logger.info("Disconnected from Hot Path")

    async def publish(self, params: dict[str, Any]) -> bool:
        """Publish parameters to Hot Path.

        Args:
            params: Dictionary of model parameters to publish.

        Returns:
            True if successfully published, False otherwise.
        """
        if not self._connected:
            raise ConnectionError("Not connected to Hot Path")

        try:
            # Serialize to JSON
            message = json.dumps(
                {
                    "type": "model_params_update",
                    "payload": params,
                }
            ).encode("utf-8")

            # Length-prefixed message (4 bytes, big-endian)
            length_prefix = struct.pack(">I", len(message))

            async with self._lock:
                self._writer.write(length_prefix + message)
                await self._writer.drain()

            logger.debug(f"Published {len(message)} bytes to Hot Path")
            return True

        except (BrokenPipeError, ConnectionResetError) as e:
            logger.warning(f"Connection lost: {e}")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to publish: {e}")
            return False

    async def publish_with_ack(
        self,
        params: dict[str, Any],
        timeout: float = 5.0,
    ) -> bool:
        """Publish parameters and wait for acknowledgment.

        Args:
            params: Dictionary of model parameters.
            timeout: Maximum time to wait for ack.

        Returns:
            True if acknowledged, False otherwise.
        """
        if not await self.publish(params):
            return False

        try:
            # Read length prefix
            length_data = await asyncio.wait_for(
                self._reader.readexactly(4),
                timeout=timeout,
            )
            length = struct.unpack(">I", length_data)[0]

            # Read response
            response_data = await asyncio.wait_for(
                self._reader.readexactly(length),
                timeout=timeout,
            )
            response = json.loads(response_data.decode("utf-8"))

            if response.get("type") == "ack" and response.get("success"):
                logger.debug("Received acknowledgment from Hot Path")
                return True
            else:
                logger.warning(f"Unexpected response: {response}")
                return False

        except TimeoutError:
            logger.warning("Timeout waiting for acknowledgment")
            return False
        except Exception as e:
            logger.error(f"Error reading acknowledgment: {e}")
            return False

    async def reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff.

        Returns:
            True if reconnection successful, False otherwise.
        """
        await self.close()

        while True:
            try:
                await self.connect()
                return True
            except ConnectionError as e:
                logger.debug(f"Reconnection failed: {e}")

                # Exponential backoff
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay,
                )


class MockParameterPublisher(ParameterPublisher):
    """Mock publisher for testing without Hot Path connection."""

    def __init__(self, socket_path: str = ""):
        super().__init__(socket_path)
        self.published_params: list = []

    async def connect(self, timeout: float = 5.0):
        """Mock connect - always succeeds."""
        self._connected = True
        logger.info("Mock publisher connected")

    async def close(self):
        """Mock close."""
        self._connected = False

    async def publish(self, params: dict[str, Any]) -> bool:
        """Mock publish - stores params locally."""
        self.published_params.append(params)
        logger.debug(f"Mock published params: {list(params.keys())}")
        return True
