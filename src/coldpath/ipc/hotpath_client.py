"""
HotPath client for ColdPath -> HotPath communication.

Uses a hybrid approach:
- Unix socket for commands (JSON-based, reliable)
- gRPC for event streaming (requires protobuf stubs when available)

This approach ensures ColdPath can always communicate with HotPath
regardless of whether proper protobuf stubs are available.

Enhanced with:
- Exponential backoff retry logic for transient failures
- Automatic reconnection on connection errors
"""

import asyncio
import json
import logging
import os
import random
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_SOCKET_PATH = os.getenv("HOTPATH_SOCKET", "/tmp/dexy-engine.sock")
hotpath_socket: str = DEFAULT_SOCKET_PATH
MAX_MESSAGE_SIZE = 5 * 1024 * 1024  # 5 MB - matches Rust/Swift limits

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 0.5  # seconds
DEFAULT_MAX_DELAY = 10.0  # seconds
DEFAULT_BACKOFF_MULTIPLIER = 2.0


def _first_value(data: dict[str, Any], *keys: str) -> Any:
    """Return the first non-None value from a dictionary for the given keys."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


@dataclass
class CandidateEvent:
    """Parsed candidate event from HotPath."""

    event_id: str
    timestamp_ms: int
    slot: int
    signature: str
    mint: str
    pool: str
    source: str
    received_at_ms: int

    # Enrichment data
    liquidity_usd: float | None = None
    fdv_usd: float | None = None
    volume_24h: float | None = None
    holder_count: int | None = None
    top_holder_pct: float | None = None

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_ms / 1000)


@dataclass
class TradingSignal:
    """Trading signal to submit to HotPath."""

    signal_id: str
    token_mint: str
    token_symbol: str
    pool_address: str
    action: str  # "BUY" or "SELL"
    position_size_sol: float
    slippage_bps: int
    expected_price: float
    confidence: float
    expires_at_ms: int
    # Optional ML context for realistic paper fill simulation
    ml_context: dict[str, Any] | None = None


@dataclass
class TradeOutcome:
    """Trade outcome from HotPath."""

    signal_id: str
    plan_id: str
    pnl_sol: float
    pnl_pct: float
    slippage_realized_bps: int
    included: bool
    execution_latency_ms: int
    error: str | None = None
    base_fee_lamports: int | None = None
    priority_fee_lamports: int | None = None
    jito_tip_lamports: int | None = None
    dex_fee_lamports: int | None = None
    tx_signature: str | None = None
    amount_out_lamports: int | None = None


class UnixSocketClient:
    """Async Unix socket client for HotPath communication.

    Features:
    - Automatic reconnection on connection errors
    - Exponential backoff retry for transient failures
    - Bounded message sizes to prevent memory exhaustion
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
    ):
        self.socket_path = socket_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False
        self._lock = asyncio.Lock()
        self._request_id = 0
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._event_handlers: list[Callable] = []
        self._read_task: asyncio.Task | None = None

        # Retry configuration
        self._max_retries = max_retries
        self._initial_delay = initial_delay
        self._max_delay = max_delay

    @property
    def is_connected(self) -> bool:
        return self._connected and self._writer is not None

    async def connect(self) -> bool:
        """Connect to HotPath Unix socket."""
        if self._connected:
            return True

        socket_file = Path(self.socket_path)
        if not socket_file.exists():
            logger.warning(f"Socket file does not exist: {self.socket_path}")
            return False

        try:
            reader, writer = await asyncio.open_unix_connection(self.socket_path)
            self._reader = reader
            self._writer = writer
            self._connected = True
            self._read_task = asyncio.create_task(self._read_loop())
            logger.info(f"Connected to HotPath via Unix socket: {self.socket_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Unix socket: {e}")
            return False
        finally:
            # Ensure clean state on all error paths
            if not self._connected:
                if self._writer is not None:
                    self._writer.close()
                    try:
                        await self._writer.wait_closed()
                    except Exception:
                        pass
                self._reader = None
                self._writer = None

    async def disconnect(self):
        """Disconnect from HotPath."""
        self._connected = False

        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        # Cancel all pending requests
        for _request_id, future in list(self._pending_requests.items()):
            if not future.done():
                future.set_exception(ConnectionError("Disconnected"))
        self._pending_requests.clear()

        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass

        self._reader = None
        self._writer = None
        logger.info("Disconnected from HotPath Unix socket")

    async def _read_loop(self):
        """Read responses and events from the socket."""
        try:
            while self._connected and self._reader:
                # Read length prefix (4 bytes, big-endian)
                length_bytes = await self._reader.readexactly(4)
                length = struct.unpack(">I", length_bytes)[0]

                # Validate message size to prevent memory exhaustion
                if length > MAX_MESSAGE_SIZE:
                    logger.error(f"Message too large: {length} bytes (max: {MAX_MESSAGE_SIZE})")
                    self._connected = False
                    break

                # Read message body
                data = await self._reader.readexactly(length)
                message = json.loads(data.decode("utf-8"))

                # Check if it's a response or event
                if "id" in message:
                    # Response to a request
                    request_id = message.get("id")
                    if request_id in self._pending_requests:
                        future = self._pending_requests.pop(request_id)
                        if not future.done():
                            future.set_result(message)
                elif "event_type" in message:
                    # Event push
                    await self._handle_event(message)

        except asyncio.IncompleteReadError:
            logger.info("Socket connection closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Read loop error: {e}")
        finally:
            self._connected = False
            # Cancel all pending requests when read loop exits
            for _request_id, future in list(self._pending_requests.items()):
                if not future.done():
                    future.set_exception(ConnectionError("Read loop terminated"))
            self._pending_requests.clear()

    async def _handle_event(self, event: dict[str, Any]):
        """Handle an event from HotPath."""
        event_type = event.get("event_type")
        payload = event.get("payload", {})

        if event_type in {"candidate", "candidate_detected"}:
            candidate_payload = payload
            if isinstance(payload, dict):
                candidate_payload = (
                    payload.get("candidate")
                    or payload.get("payload", {}).get("candidate")
                    or payload
                )

            enrichment = {}
            if isinstance(candidate_payload, dict):
                enrichment = candidate_payload.get("enrichment") or {}

            candidate = CandidateEvent(
                event_id=_first_value(payload, "event_id", "eventId") or str(uuid.uuid4()),
                timestamp_ms=_first_value(event, "timestamp_ms", "timestampMs")
                or _first_value(payload, "timestamp_ms", "timestampMs")
                or 0,
                slot=_first_value(candidate_payload, "slot") or 0,
                signature=_first_value(candidate_payload, "signature") or "",
                mint=_first_value(candidate_payload, "mint") or "",
                pool=_first_value(candidate_payload, "pool") or "",
                source=_first_value(candidate_payload, "source") or "",
                received_at_ms=_first_value(candidate_payload, "received_at_ms", "receivedAtMs")
                or 0,
                liquidity_usd=_first_value(enrichment, "liquidity_usd", "liquidityUsd"),
                fdv_usd=_first_value(enrichment, "fdv_usd", "fdvUsd"),
                volume_24h=_first_value(enrichment, "volume_24h", "volume24h"),
                holder_count=_first_value(enrichment, "holder_count", "holderCount"),
                top_holder_pct=_first_value(enrichment, "top_holder_pct", "topHolderPct"),
            )

            for handler in self._event_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(candidate)
                    else:
                        handler(candidate)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")

    async def _send_request(
        self,
        msg_type: str,
        payload: dict[str, Any],
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Send a request and wait for response with retry logic.

        Args:
            msg_type: Message type (e.g., "command")
            payload: Request payload
            max_retries: Override max retries (uses default if None)

        Returns:
            Response dictionary

        Raises:
            RuntimeError: If not connected after all retries
            TimeoutError: If request times out after all retries
        """
        retries = max_retries if max_retries is not None else self._max_retries
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                # Check connection and attempt reconnect if needed
                if not self.is_connected:
                    if attempt > 0:
                        # Try to reconnect on retry
                        await self.connect()
                    if not self.is_connected:
                        raise RuntimeError("Not connected to HotPath")

                async with self._lock:
                    request_id = str(uuid.uuid4())
                    request = {
                        "id": request_id,
                        "type": msg_type,
                        "payload": payload,
                    }

                    # Create future for response
                    future: asyncio.Future = asyncio.get_event_loop().create_future()
                    self._pending_requests[request_id] = future

                    # Send request
                    data = json.dumps(request).encode("utf-8")
                    length = struct.pack(">I", len(data))
                    if self._writer is None:
                        raise RuntimeError("Writer is None")
                    self._writer.write(length + data)
                    await self._writer.drain()

                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(future, timeout=30.0)
                    return response
                except TimeoutError:
                    self._pending_requests.pop(request_id, None)
                    raise TimeoutError("Request timed out") from None

            except (RuntimeError, ConnectionError, TimeoutError, OSError) as e:
                last_error = e

                # Don't retry on the last attempt
                if attempt == retries:
                    break

                # Calculate exponential backoff with jitter
                delay = min(
                    self._initial_delay * (DEFAULT_BACKOFF_MULTIPLIER**attempt), self._max_delay
                )
                # Add jitter (±25%)
                jitter = delay * 0.25 * (random.random() * 2 - 1)
                delay = max(0.1, delay + jitter)

                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{retries + 1}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                # Disconnect and reconnect on connection errors
                if isinstance(e, (ConnectionError, OSError, RuntimeError)):
                    self._connected = False
                    if self._writer:
                        try:
                            self._writer.close()
                            await self._writer.wait_closed()
                        except Exception:
                            pass
                        self._writer = None
                        self._reader = None

                await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(f"Request failed after {retries + 1} attempts: {last_error}")
        raise last_error if last_error else RuntimeError("Unknown error")

    def on_candidate(self, handler: Callable[[CandidateEvent], None]):
        """Register a handler for candidate events."""
        self._event_handlers.append(handler)


class HotPathClient:
    """Async client for HotPath communication.

    Uses Unix socket for reliable JSON-based communication.
    Provides bi-directional communication between ColdPath and HotPath:
    - Subscribes to candidate events for AutoTrader processing
    - Submits trading signals for execution
    - Retrieves trade outcomes for continuous learning
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        socket_path: str = DEFAULT_SOCKET_PATH,
        max_reconnect_attempts: int = 10,
        reconnect_delay_seconds: float = 5.0,
    ):
        self.host = host
        self.port = port
        self.socket_path = socket_path
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds

        # Unix socket client (primary communication)
        self._socket_client = UnixSocketClient(socket_path)

        # State
        self._connected = False
        self._shutdown = False

        # Event handlers
        self._event_handlers: list[Callable[[CandidateEvent], None]] = []

        # Pending signals
        self._pending_signals: dict[str, TradingSignal] = {}
        self._signal_outcomes: dict[str, TradeOutcome] = {}

    @property
    def is_connected(self) -> bool:
        return self._socket_client.is_connected

    async def connect(self) -> bool:
        """Connect to HotPath."""
        if self._connected:
            return True

        # Connect via Unix socket
        connected = await self._socket_client.connect()
        if connected:
            self._connected = True
            # Forward event handlers
            for handler in self._event_handlers:
                self._socket_client.on_candidate(handler)
            logger.info("HotPath client connected via Unix socket")
        return connected

    async def disconnect(self):
        """Disconnect from HotPath."""
        self._shutdown = True
        self._connected = False
        await self._socket_client.disconnect()
        logger.info("Disconnected from HotPath")

    def on_candidate(self, handler: Callable[[CandidateEvent], None]):
        """Register a handler for candidate events."""
        self._event_handlers.append(handler)
        if self._socket_client:
            self._socket_client.on_candidate(handler)

    async def start_event_stream(self, event_types: list[str] | None = None):
        """Start receiving events (events come automatically via Unix socket)."""
        # Events are received automatically via the Unix socket connection
        logger.info("Event stream started (via Unix socket)")

    async def submit_trading_signal(self, signal: TradingSignal) -> dict[str, Any]:
        """Submit a trading signal to HotPath for execution.

        Args:
            signal: The trading signal to execute.

        Returns:
            Dict with 'accepted', 'plan_id', and optional 'rejection_reason'.
        """
        if not self.is_connected:
            # Try to reconnect
            if not await self.connect():
                logger.info(
                    f"HotPath submit rejected (not connected): signal_id={signal.signal_id}"
                )
                return {
                    "accepted": False,
                    "plan_id": None,
                    "rejection_reason": "Not connected to HotPath",
                }

        try:
            # Build payload with optional ML context
            payload = {
                "mint": signal.token_mint,
                "side": signal.action,
                "amount_sol": signal.position_size_sol,
                "slippage_bps": signal.slippage_bps,
                "expected_price": signal.expected_price,
                "signal_id": signal.signal_id,
            }
            # Include ML context if available (for realistic paper fill simulation)
            if signal.ml_context is not None:
                payload["ml_context"] = signal.ml_context

            # Use the execute_trade command via Unix socket
            response = await self._socket_client._send_request(
                "command",
                {
                    "type": "execute_trade",
                    "payload": payload,
                },
            )

            if response.get("success"):
                plan_id = response.get("payload", {}).get("plan_id", f"plan-{signal.signal_id}")
                self._pending_signals[signal.signal_id] = signal
                logger.info(
                    f"HotPath execute_trade accepted: signal_id={signal.signal_id} "
                    f"plan_id={plan_id} expected_price={signal.expected_price}"
                )
                return {
                    "accepted": True,
                    "plan_id": plan_id,
                    "rejection_reason": None,
                }
            else:
                logger.info(
                    f"HotPath execute_trade rejected: signal_id={signal.signal_id} "
                    f"expected_price={signal.expected_price} "
                    f"reason={response.get('error', 'Unknown error')}"
                )
                return {
                    "accepted": False,
                    "plan_id": None,
                    "rejection_reason": response.get("error", "Unknown error"),
                }

        except Exception as e:
            logger.error(f"Failed to submit signal: {e}")
            return {
                "accepted": False,
                "plan_id": None,
                "rejection_reason": str(e),
            }

    async def submit_trading_signals_batch(
        self,
        signals: list[TradingSignal],
        max_parallel: int = 10,
    ) -> dict[str, Any]:
        """Submit multiple trading signals in a single batch (100x reduced overhead).

        Args:
            signals: List of trading signals to execute.
            max_parallel: Maximum signals to process in parallel (default: 10).

        Returns:
            Dict with 'batch_id', 'accepted_count', 'rejected_count', and 'results'.
        """
        if not signals:
            return {
                "batch_id": str(uuid.uuid4()),
                "accepted_count": 0,
                "rejected_count": 0,
                "results": [],
            }

        if not self.is_connected:
            if not await self.connect():
                batch_id = str(uuid.uuid4())
                logger.info(
                    f"HotPath batch rejected (not connected): batch_id={batch_id} "
                    f"signals={len(signals)}"
                )
                return {
                    "batch_id": batch_id,
                    "accepted_count": 0,
                    "rejected_count": len(signals),
                    "results": [
                        {
                            "signal_id": s.signal_id,
                            "accepted": False,
                            "plan_id": None,
                            "rejection_reason": "Not connected to HotPath",
                        }
                        for s in signals
                    ],
                }

        try:
            batch_id = str(uuid.uuid4())

            # Build batch payload
            signal_payloads = []
            for signal in signals:
                payload = {
                    "signal_id": signal.signal_id,
                    "mint": signal.token_mint,
                    "symbol": signal.token_symbol,
                    "pool": signal.pool_address,
                    "side": signal.action,
                    "amount_sol": signal.position_size_sol,
                    "slippage_bps": signal.slippage_bps,
                    "expected_price": signal.expected_price,
                    "confidence": signal.confidence,
                    "expires_at_ms": signal.expires_at_ms,
                }
                if signal.ml_context is not None:
                    payload["ml_context"] = signal.ml_context
                signal_payloads.append(payload)

            expected_prices = [
                s.expected_price for s in signals if getattr(s, "expected_price", None) is not None
            ]
            if expected_prices:
                logger.info(
                    f"Sending HotPath batch: batch_id={batch_id} size={len(signals)} "
                    f"expected_price_min={min(expected_prices)} "
                    f"expected_price_max={max(expected_prices)}"
                )
            else:
                logger.info(
                    f"Sending HotPath batch: batch_id={batch_id} size={len(signals)} "
                    "expected_price=unknown"
                )

            # Send batch request via Unix socket
            response = await self._socket_client._send_request(
                "command",
                {
                    "type": "execute_trade_batch",
                    "payload": {
                        "batch_id": batch_id,
                        "signals": signal_payloads,
                        "max_parallel": max_parallel,
                    },
                },
            )

            if response.get("success"):
                result_payload = response.get("payload", {})
                accepted_count = result_payload.get("accepted_count", 0)
                rejected_count = result_payload.get("rejected_count", 0)
                results = result_payload.get("results", [])

                # Track accepted signals
                for result in results:
                    if result.get("accepted"):
                        signal_id = result.get("signal_id")
                        # Find original signal and store it
                        for s in signals:
                            if s.signal_id == signal_id:
                                self._pending_signals[signal_id] = s
                                break

                logger.info(
                    f"Batch {batch_id}: {accepted_count} accepted, {rejected_count} rejected"
                )

                return {
                    "batch_id": batch_id,
                    "accepted_count": accepted_count,
                    "rejected_count": rejected_count,
                    "results": results,
                }
            else:
                # Entire batch failed
                return {
                    "batch_id": batch_id,
                    "accepted_count": 0,
                    "rejected_count": len(signals),
                    "results": [
                        {
                            "signal_id": s.signal_id,
                            "accepted": False,
                            "plan_id": None,
                            "rejection_reason": response.get("error", "Batch request failed"),
                        }
                        for s in signals
                    ],
                }

        except Exception as e:
            logger.error(f"Failed to submit signal batch: {e}")
            return {
                "batch_id": str(uuid.uuid4()),
                "accepted_count": 0,
                "rejected_count": len(signals),
                "results": [
                    {
                        "signal_id": s.signal_id,
                        "accepted": False,
                        "plan_id": None,
                        "rejection_reason": str(e),
                    }
                    for s in signals
                ],
            }

    async def get_trade_outcome(
        self,
        signal_id: str,
        plan_id: str,
    ) -> TradeOutcome | None:
        """Get the outcome of a previously submitted trade.

        Args:
            signal_id: The original signal ID.
            plan_id: The plan ID returned from signal submission.

        Returns:
            TradeOutcome if found, None otherwise.
        """
        # Check cached outcomes
        if signal_id in self._signal_outcomes:
            return self._signal_outcomes.pop(signal_id)

        if not self.is_connected:
            return None

        try:
            # Query exact trade by plan_id (deterministic under high throughput)
            response = await self._socket_client._send_request(
                "command",
                {"type": "get_trades", "payload": {"plan_id": plan_id, "limit": 1}},
            )

            if response.get("success"):
                trades = response.get("payload", {}).get("trades", [])
                for trade in trades:
                    if trade.get("plan_id") == plan_id:
                        fill = trade.get("fill", {})
                        costs = trade.get("costs", {})
                        # Use net_pnl_sol when available; fall back to gross pnl_sol.
                        pnl_sol = trade.get("net_pnl_sol")
                        if pnl_sol is None:
                            pnl_sol = trade.get("pnl_sol")
                        pnl_pct = trade.get("pnl_pct")

                        # Avoid placeholder outcomes when PnL is not yet available.
                        if pnl_sol is None or pnl_pct is None:
                            return None

                        return TradeOutcome(
                            signal_id=signal_id,
                            plan_id=plan_id,
                            pnl_sol=float(pnl_sol),
                            pnl_pct=float(pnl_pct),
                            slippage_realized_bps=fill.get("slippage_bps_realized", 0),
                            included=fill.get("ok", False),
                            execution_latency_ms=trade.get("total_latency_ms", 0),
                            error=fill.get("failure_reason"),
                            base_fee_lamports=costs.get("base_fee_lamports"),
                            priority_fee_lamports=costs.get("priority_fee_lamports"),
                            jito_tip_lamports=costs.get("jito_tip_lamports"),
                            dex_fee_lamports=costs.get("dex_fee_lamports"),
                            tx_signature=trade.get("tx_signature"),
                            amount_out_lamports=fill.get("amount_out_lamports"),
                        )

            return None

        except Exception as e:
            logger.error(f"Failed to get trade outcome: {e}")
            return None

    async def get_status(self) -> dict[str, Any]:
        """Get HotPath engine status."""
        if not self.is_connected:
            return {}

        try:
            response = await self._socket_client._send_request(
                "command",
                {"type": "get_status"},
            )

            if response.get("success"):
                return response.get("payload", {})
            return {}

        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {}

    async def get_recent_candidates(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recently scored candidates from HotPath's AutoPipeline cache."""
        if not self.is_connected:
            return []

        try:
            response = await self._socket_client._send_request(
                "command",
                {"type": "get_recent_candidates", "payload": {"limit": limit}},
            )

            if response.get("success"):
                payload = response.get("payload", [])
                if isinstance(payload, list):
                    return payload
            return []

        except Exception as e:
            logger.error(f"Failed to get recent candidates: {e}")
            return []

    async def start_autotrader(self) -> dict[str, Any]:
        """Start the AutoTrader in HotPath."""
        if not self.is_connected:
            if not await self.connect():
                return {"success": False, "error": "Not connected"}

        try:
            response = await self._socket_client._send_request(
                "command",
                {"type": "start_autotrader"},
            )
            return (
                response.get("payload", {})
                if response.get("success")
                else {"error": response.get("error")}
            )
        except Exception as e:
            logger.error(f"Failed to start autotrader: {e}")
            return {"success": False, "error": str(e)}

    async def stop_autotrader(self) -> dict[str, Any]:
        """Stop the AutoTrader in HotPath."""
        if not self.is_connected:
            return {"success": False, "error": "Not connected"}

        try:
            response = await self._socket_client._send_request(
                "command",
                {"type": "stop_autotrader"},
            )
            return (
                response.get("payload", {})
                if response.get("success")
                else {"error": response.get("error")}
            )
        except Exception as e:
            logger.error(f"Failed to stop autotrader: {e}")
            return {"success": False, "error": str(e)}

    async def get_market_metrics(self) -> dict[str, Any] | None:
        """Get current market metrics from HotPath for regime detection.

        Returns metrics including:
        - volume_24h: 24-hour trading volume
        - volatility_1h: 1-hour volatility measure
        - network_congestion: Network congestion score (0-1)
        - avg_slippage_bps: Average slippage in basis points
        - mev_frequency: MEV attack frequency score

        Returns None if not connected or metrics unavailable.
        """
        if not self.is_connected:
            return None

        try:
            response = await self._socket_client._send_request(
                "command",
                {"type": "get_market_metrics"},
            )

            if response.get("success"):
                payload = response.get("payload", {})
                return {
                    "timestamp_ms": payload.get("timestamp_ms"),
                    "volume_24h": payload.get("volume_24h", 0.0),
                    "volatility_1h": payload.get("volatility_1h", 0.0),
                    "network_congestion": payload.get("network_congestion", 0.5),
                    "avg_slippage_bps": payload.get("avg_slippage_bps", 50.0),
                    "mev_frequency": payload.get("mev_frequency", 0.0),
                }
            return None

        except Exception as e:
            logger.debug(f"Failed to get market metrics: {e}")
            return None


class MockHotPathClient(HotPathClient):
    """Mock client for testing without HotPath connection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mock_candidates: list[CandidateEvent] = []
        self._submitted_signals: list[TradingSignal] = []
        self._mock_connected = False
        self._mock_stream_task: asyncio.Task | None = None

    @property
    def is_connected(self) -> bool:
        """Override to use mock connection state."""
        return self._mock_connected

    async def connect(self) -> bool:
        self._connected = True
        self._mock_connected = True
        logger.info("Mock HotPath client connected")
        return True

    async def disconnect(self):
        self._connected = False
        self._mock_connected = False
        # Cancel tracked mock stream task
        if self._mock_stream_task and not self._mock_stream_task.done():
            self._mock_stream_task.cancel()
            try:
                await self._mock_stream_task
            except asyncio.CancelledError:
                pass
            self._mock_stream_task = None

    def add_mock_candidate(self, candidate: CandidateEvent):
        """Add a mock candidate for testing."""
        self._mock_candidates.append(candidate)

    async def start_event_stream(self, event_types: list[str] | None = None):
        """Start mock event stream."""
        self._mock_stream_task = asyncio.create_task(self._mock_stream_loop())

    async def _mock_stream_loop(self):
        """Emit mock candidates."""
        while self._connected and not self._shutdown:
            if self._mock_candidates:
                candidate = self._mock_candidates.pop(0)
                for handler in self._event_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(candidate)
                        else:
                            handler(candidate)
                    except Exception as e:
                        logger.error(f"Error in mock handler: {e}")
            await asyncio.sleep(0.1)

    async def submit_trading_signal(self, signal: TradingSignal) -> dict[str, Any]:
        """Mock signal submission."""
        self._submitted_signals.append(signal)
        # Log ML context if present for testing
        if signal.ml_context:
            logger.debug(
                f"Mock signal with ML context: "
                f"fraud_score={signal.ml_context.get('fraud_score', 'N/A')}"
            )
        return {
            "accepted": True,
            "plan_id": f"mock-plan-{signal.signal_id}",
            "rejection_reason": None,
        }

    async def get_trade_outcome(
        self,
        signal_id: str,
        plan_id: str,
    ) -> TradeOutcome | None:
        """Mock trade outcome."""
        import random

        return TradeOutcome(
            signal_id=signal_id,
            plan_id=plan_id,
            pnl_sol=random.uniform(-0.05, 0.1),
            pnl_pct=random.uniform(-5, 10),
            slippage_realized_bps=random.randint(50, 200),
            included=random.random() > 0.1,
            execution_latency_ms=random.randint(100, 500),
            base_fee_lamports=5000,
            priority_fee_lamports=random.randint(5_000, 30_000),
            jito_tip_lamports=random.choice([0, 10_000]),
            dex_fee_lamports=random.randint(10_000, 80_000),
            tx_signature=f"mock-sig-{signal_id}",
            amount_out_lamports=random.randint(100_000_000, 10_000_000_000),
        )
