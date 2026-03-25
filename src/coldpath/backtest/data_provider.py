"""
Market data provider for backtesting.

Fetches historical data from Bitquery and other sources.
Supports multiple data sources with fallback and synthetic data generation.
"""

import asyncio
import json
import logging
import math
import os
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)

BITQUERY_ENDPOINT = "https://streaming.bitquery.io/graphql"


@dataclass
class MarketEvent:
    """A market event for backtesting."""

    timestamp_ms: int
    mint: str
    pool: str
    event_type: str  # "ohlcv", "trade", "pool_created"
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class OHLCVBar:
    """OHLCV bar data with validation."""

    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    liquidity_usd: float | None = None

    def __post_init__(self) -> None:
        """Validate OHLCV data after initialization."""

        # Check for NaN/Inf in price fields
        for name in ["open", "high", "low", "close"]:
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"OHLCVBar.{name} must be finite, got {value}")

        # Check for NaN/Inf in volume
        if not math.isfinite(self.volume):
            raise ValueError(f"OHLCVBar.volume must be finite, got {self.volume}")

        # Check for NaN/Inf in optional liquidity
        if self.liquidity_usd is not None and not math.isfinite(self.liquidity_usd):
            raise ValueError(f"OHLCVBar.liquidity_usd must be finite, got {self.liquidity_usd}")

        # Validate timestamp (must be positive)
        if self.timestamp_ms <= 0:
            raise ValueError(f"OHLCVBar.timestamp_ms must be positive, got {self.timestamp_ms}")

        # Validate high >= low
        if self.high < self.low:
            raise ValueError(f"OHLCVBar.high ({self.high}) must be >= low ({self.low})")

        # Validate high >= open, close and low <= open, close
        if self.high < max(self.open, self.close):
            raise ValueError(
                f"OHLCVBar.high ({self.high}) must be >= "
                f"max(open, close) ({max(self.open, self.close)})"
            )
        if self.low > min(self.open, self.close):
            raise ValueError(
                f"OHLCVBar.low ({self.low}) must be <= "
                f"min(open, close) ({min(self.open, self.close)})"
            )

        # Validate non-negative volume
        if self.volume < 0:
            raise ValueError(f"OHLCVBar.volume must be non-negative, got {self.volume}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "liquidity_usd": self.liquidity_usd,
        }


@dataclass
class DataProviderConfig:
    """Configuration for data providers."""

    bitquery_api_key: str | None = None
    cache_dir: str | None = None
    rate_limit_per_minute: int = 30
    timeout_seconds: float = 30.0
    retry_attempts: int = 3

    @classmethod
    def from_env(cls) -> "DataProviderConfig":
        return cls(
            bitquery_api_key=os.getenv("BITQUERY_API_KEY"),
            cache_dir=os.getenv("DATA_CACHE_DIR"),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT", "30")),
        )


class LocalTelemetryDataProvider:
    """Read persisted HotPath telemetry from local SQLite for replay/backtesting."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.getenv("DEXY_DB_PATH", "sniperdesk.db")

    async def stream_events(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        data_source: str,
        mints: list[str] | None = None,
    ) -> AsyncIterator[MarketEvent]:
        del data_source  # Single-source provider.
        events = await asyncio.to_thread(
            self._load_events,
            start_timestamp_ms,
            end_timestamp_ms,
            mints,
        )
        for event in events:
            yield event

    def _load_events(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        mints: list[str] | None,
    ) -> list[MarketEvent]:
        if not os.path.exists(self.db_path):
            logger.warning("SQLite telemetry DB not found: %s", self.db_path)
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            events: list[MarketEvent] = []
            events.extend(self._load_events_raw(conn, start_timestamp_ms, end_timestamp_ms, mints))
            events.extend(self._load_trade_rows(conn, start_timestamp_ms, end_timestamp_ms, mints))
            events.sort(key=lambda e: e.timestamp_ms)
            return events
        finally:
            conn.close()

    def _load_events_raw(
        self,
        conn: sqlite3.Connection,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        mints: list[str] | None,
    ) -> list[MarketEvent]:
        cursor = conn.cursor()
        columns = {
            row["name"] for row in cursor.execute("PRAGMA table_info(events_raw)").fetchall()
        }
        required = {"event_type", "source", "payload_json"}
        if not required.issubset(columns):
            return []
        timestamp_column = None
        if "event_timestamp_ms" in columns:
            timestamp_column = "event_timestamp_ms"
        elif "receive_time_ms" in columns:
            timestamp_column = "receive_time_ms"
        if timestamp_column is None:
            return []

        symbol_expr = "symbol" if "symbol" in columns else "NULL AS symbol"

        query = f"""
            SELECT event_type, source, {symbol_expr},
                   {timestamp_column} AS timestamp_ms, payload_json
            FROM events_raw
            WHERE {timestamp_column} >= ? AND {timestamp_column} <= ?
              AND event_type IN (
                  'strategy_decision', 'trade_outcome',
                  'order_placed', 'order_cancelled'
              )
            ORDER BY timestamp_ms ASC
        """
        params: list[Any] = [start_timestamp_ms, end_timestamp_ms]
        if mints and "symbol" in columns:
            placeholders = ",".join("?" for _ in mints)
            query = query.replace(
                "ORDER BY timestamp_ms ASC",
                f"AND symbol IN ({placeholders}) ORDER BY timestamp_ms ASC",
            )
            params.extend(mints)

        rows = cursor.execute(query, tuple(params)).fetchall()
        mint_filter = set(mints) if mints else None
        out: list[MarketEvent] = []
        for row in rows:
            mint = row["symbol"] or ""
            payload = {}
            payload_raw = row["payload_json"]
            if payload_raw:
                try:
                    payload = json.loads(payload_raw)
                except json.JSONDecodeError:
                    payload = {}
            if not mint:
                mint = (
                    payload.get("symbol") or payload.get("mint") or payload.get("token_mint") or ""
                )
            if mint_filter and mint not in mint_filter:
                continue
            payload["telemetry_event_type"] = row["event_type"]
            payload["telemetry_source"] = row["source"]
            out.append(
                MarketEvent(
                    timestamp_ms=int(row["timestamp_ms"]),
                    mint=mint,
                    pool="",
                    event_type="trade",
                    data=payload,
                )
            )
        return out

    def _load_trade_rows(
        self,
        conn: sqlite3.Connection,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        mints: list[str] | None,
    ) -> list[MarketEvent]:
        cursor = conn.cursor()
        columns = {row["name"] for row in cursor.execute("PRAGMA table_info(trades)").fetchall()}
        required = {
            "trade_id",
            "mint",
            "mode",
            "quoted_slippage_bps",
            "realized_slippage_bps",
            "pnl_sol",
            "pnl_pct",
            "created_at",
        }
        if not required.issubset(columns):
            return []

        query = """
            SELECT
                trade_id,
                mint,
                mode,
                direction,
                amount_in,
                amount_out,
                entry_price,
                exit_price,
                quoted_slippage_bps,
                realized_slippage_bps,
                total_fees_sol,
                net_pnl_sol,
                pnl_sol,
                pnl_pct,
                tx_signature,
                included,
                CAST(strftime('%s', created_at) AS INTEGER) * 1000 AS timestamp_ms
            FROM trades
            WHERE CAST(strftime('%s', created_at) AS INTEGER) * 1000 >= ?
              AND CAST(strftime('%s', created_at) AS INTEGER) * 1000 <= ?
            ORDER BY timestamp_ms ASC
        """
        params: list[Any] = [start_timestamp_ms, end_timestamp_ms]
        if mints:
            placeholders = ",".join("?" for _ in mints)
            query = query.replace(
                "ORDER BY timestamp_ms ASC",
                f"AND mint IN ({placeholders}) ORDER BY timestamp_ms ASC",
            )
            params.extend(mints)

        rows = cursor.execute(query, tuple(params)).fetchall()
        out: list[MarketEvent] = []
        for row in rows:
            out.append(
                MarketEvent(
                    timestamp_ms=int(row["timestamp_ms"]),
                    mint=row["mint"] or "",
                    pool="",
                    event_type="trade",
                    data={
                        "trade_id": row["trade_id"],
                        "mode": row["mode"],
                        "direction": row["direction"],
                        "amount_in": row["amount_in"],
                        "amount_out": row["amount_out"],
                        "entry_price": row["entry_price"],
                        "exit_price": row["exit_price"],
                        "quoted_slippage_bps": row["quoted_slippage_bps"],
                        "realized_slippage_bps": row["realized_slippage_bps"],
                        "total_fees_sol": row["total_fees_sol"],
                        "net_pnl_sol": row["net_pnl_sol"],
                        "pnl_sol": row["pnl_sol"],
                        "pnl_pct": row["pnl_pct"],
                        "tx_signature": row["tx_signature"],
                        "included": row["included"],
                        "telemetry_event_type": "trade_row",
                    },
                )
            )
        return out


class BitqueryDataProvider:
    """Fetches historical data from Bitquery.

    Supports OHLCV and trade-level data with caching and rate limiting.
    Falls back to synthetic data when API key is not available.
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: DataProviderConfig | None = None,
    ):
        self.config = config or DataProviderConfig(bitquery_api_key=api_key)
        self.api_key = self.config.bitquery_api_key
        self.client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        self._rate_limiter = asyncio.Semaphore(self.config.rate_limit_per_minute)

    async def stream_events(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        data_source: str,
        mints: list[str] | None = None,
    ) -> AsyncIterator[MarketEvent]:
        """Stream market events for backtesting.

        Args:
            start_timestamp_ms: Start time in milliseconds.
            end_timestamp_ms: End time in milliseconds.
            data_source: Data source type ('bitquery_ohlcv', 'bitquery_trades', 'synthetic').
            mints: Optional list of specific mints to fetch.

        Yields:
            MarketEvent objects in chronological order.
        """
        if data_source == "bitquery_ohlcv":
            async for event in self._stream_ohlcv(start_timestamp_ms, end_timestamp_ms, mints):
                yield event
        elif data_source == "bitquery_trades":
            async for event in self._stream_trades(start_timestamp_ms, end_timestamp_ms, mints):
                yield event
        elif data_source == "synthetic":
            async for event in self._stream_synthetic(start_timestamp_ms, end_timestamp_ms):
                yield event
        else:
            raise ValueError(f"Unknown data source: {data_source}")

    async def _stream_ohlcv(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        mints: list[str] | None = None,
    ) -> AsyncIterator[MarketEvent]:
        """Stream OHLCV data from Bitquery."""
        if not self.api_key:
            logger.warning("No Bitquery API key - using synthetic data")
            async for event in self._stream_synthetic(start_timestamp_ms, end_timestamp_ms):
                yield event
            return

        # Convert timestamps
        start_dt = datetime.fromtimestamp(start_timestamp_ms / 1000).isoformat()
        end_dt = datetime.fromtimestamp(end_timestamp_ms / 1000).isoformat()

        query = """
        query ($start: DateTime!, $end: DateTime!, $limit: Int!) {
          Solana {
            DEXTradeByTokens(
              where: {
                Block: { Time: { after: $start, before: $end } }
                Trade: { Side: { Amount: { gt: "0" } } }
              }
              orderBy: { ascending: Block_Time }
              limit: { count: $limit }
            ) {
              Block {
                Time
                Slot
              }
              Trade {
                Currency { MintAddress Symbol Name }
                Side { Currency { MintAddress } }
                PriceInUSD
                Amount
              }
            }
          }
        }
        """

        variables = {
            "start": start_dt,
            "end": end_dt,
            "limit": 10000,
        }

        try:
            async with self._rate_limiter:
                response = await self._make_request(query, variables)

            if not response:
                return

            trades_data = response.get("data", {}).get("Solana", {}).get("DEXTradeByTokens", [])

            # Aggregate into OHLCV bars (1-minute intervals)
            bars_by_mint: dict[str, dict[int, OHLCVBar]] = {}

            for trade in trades_data:
                block = trade.get("Block", {})
                trade_data = trade.get("Trade", {})
                currency = trade_data.get("Currency", {})

                mint = currency.get("MintAddress", "")
                if mints and mint not in mints:
                    continue

                time_str = block.get("Time", "")
                if not time_str:
                    continue

                # Parse timestamp and round to minute
                ts = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                minute_ts = int(ts.timestamp() // 60 * 60 * 1000)

                price = float(trade_data.get("PriceInUSD", 0))
                amount = float(trade_data.get("Amount", 0))

                if mint not in bars_by_mint:
                    bars_by_mint[mint] = {}

                if minute_ts not in bars_by_mint[mint]:
                    bars_by_mint[mint][minute_ts] = OHLCVBar(
                        timestamp_ms=minute_ts,
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=amount,
                    )
                else:
                    bar = bars_by_mint[mint][minute_ts]
                    bar.high = max(bar.high, price)
                    bar.low = min(bar.low, price)
                    bar.close = price
                    bar.volume += amount

            # Yield events in chronological order
            all_events = []
            for mint, bars in bars_by_mint.items():
                for ts, bar in bars.items():
                    event_data = bar.to_dict()
                    event_data["volume_sol"] = None
                    event_data["liquidity_sol"] = None
                    event_data["liquidity_tokens"] = None
                    all_events.append(
                        MarketEvent(
                            timestamp_ms=ts,
                            mint=mint,
                            pool="",
                            event_type="ohlcv",
                            data=event_data,
                        )
                    )

            all_events.sort(key=lambda e: e.timestamp_ms)
            for event in all_events:
                yield event

        except Exception as e:
            logger.error(f"Error streaming OHLCV data: {e}")
            # Fall back to synthetic data
            async for event in self._stream_synthetic(start_timestamp_ms, end_timestamp_ms):
                yield event

    async def _stream_trades(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        mints: list[str] | None = None,
    ) -> AsyncIterator[MarketEvent]:
        """Stream individual trades from Bitquery."""
        if not self.api_key:
            logger.warning("No Bitquery API key - using synthetic data")
            async for event in self._stream_synthetic(start_timestamp_ms, end_timestamp_ms):
                yield event
            return

        # Delegate to OHLCV for now
        async for event in self._stream_ohlcv(start_timestamp_ms, end_timestamp_ms, mints):
            yield event

    async def _stream_synthetic(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
    ) -> AsyncIterator[MarketEvent]:
        """Generate synthetic data for testing.

        Creates realistic-looking market data with:
        - Random price movements with mean reversion
        - Volume patterns
        - Multiple tokens with different characteristics
        """
        # Generate 5 synthetic tokens with different characteristics
        tokens = [
            {
                "mint": "Synth1" + "A" * 38,
                "symbol": "MOON",
                "base_price": 0.0001,
                "volatility": 0.05,
            },
            {
                "mint": "Synth2" + "B" * 38,
                "symbol": "PUMP",
                "base_price": 0.001,
                "volatility": 0.03,
            },
            {
                "mint": "Synth3" + "C" * 38,
                "symbol": "DEGEN",
                "base_price": 0.01,
                "volatility": 0.08,
            },
            {
                "mint": "Synth4" + "D" * 38,
                "symbol": "SAFE",
                "base_price": 0.1,
                "volatility": 0.01,
            },
            {
                "mint": "Synth5" + "E" * 38,
                "symbol": "MEME",
                "base_price": 0.00001,
                "volatility": 0.1,
            },
        ]

        # 1-minute bars
        interval_ms = 60 * 1000
        current_ts = start_timestamp_ms

        prices = {t["mint"]: t["base_price"] for t in tokens}
        event_count = 0

        while current_ts < end_timestamp_ms:
            for token in tokens:
                mint = token["mint"]
                volatility = token["volatility"]

                # Random walk with mean reversion
                drift = random.gauss(0, volatility)
                mean_reversion = (token["base_price"] - prices[mint]) / token["base_price"] * 0.05

                price_change = drift + mean_reversion
                new_price = prices[mint] * (1 + price_change)
                new_price = max(new_price, token["base_price"] * 0.01)

                # Generate OHLCV with intrabar volatility
                open_price = prices[mint]
                close_price = new_price
                intrabar_range = abs(close_price - open_price) * (1 + random.uniform(0.1, 0.5))
                high_price = max(open_price, close_price) + intrabar_range * random.uniform(0, 0.5)
                low_price = min(open_price, close_price) - intrabar_range * random.uniform(0, 0.5)
                low_price = max(low_price, token["base_price"] * 0.001)

                # Volume with some randomness
                base_volume = random.uniform(100, 10000)
                volume_spike = 1.0 + (abs(price_change) * 10)  # Higher volume on big moves
                volume = base_volume * volume_spike

                # Liquidity
                liquidity_sol = random.uniform(1000, 50000)
                liquidity_tokens = liquidity_sol / new_price

                prices[mint] = new_price

                yield MarketEvent(
                    timestamp_ms=current_ts,
                    mint=mint,
                    pool=f"Pool{mint[:8]}",
                    event_type="ohlcv",
                    data={
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                        "volume_sol": None,
                        "liquidity_sol": liquidity_sol,
                        "liquidity_tokens": liquidity_tokens,
                    },
                )
                event_count += 1

            current_ts += interval_ms

            # Yield to event loop periodically
            if event_count % 100 == 0:
                await asyncio.sleep(0)

    async def _make_request(
        self,
        query: str,
        variables: dict[str, Any],
    ) -> dict | None:
        """Make a GraphQL request to Bitquery with retry logic."""
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key,
        }

        for attempt in range(self.config.retry_attempts):
            try:
                response = await self.client.post(
                    BITQUERY_ENDPOINT,
                    json={"query": query, "variables": variables},
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait_time = 2**attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"HTTP error: {e}")
                    raise

            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(1)
                else:
                    raise

        return None

    async def get_ohlcv(
        self,
        mint: str,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        interval_seconds: int = 60,
    ) -> list[OHLCVBar]:
        """Get OHLCV bars for a specific token."""
        bars = []
        async for event in self._stream_ohlcv(start_timestamp_ms, end_timestamp_ms, mints=[mint]):
            if event.mint == mint:
                data = event.data
                bars.append(
                    OHLCVBar(
                        timestamp_ms=data.get("timestamp_ms", event.timestamp_ms),
                        open=data.get("open", 0),
                        high=data.get("high", 0),
                        low=data.get("low", 0),
                        close=data.get("close", 0),
                        volume=data.get("volume", 0),
                    )
                )
        return bars

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
