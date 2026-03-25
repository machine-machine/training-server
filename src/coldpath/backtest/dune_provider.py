"""
Dune Analytics Data Provider for Backtesting.

Fetches historical Solana DEX data from Dune Analytics for:
- Token launches and initial trading data
- Pool liquidity history
- Trade-level data for MEV analysis
- Historical sandwich attack patterns

Uses Dune API v3 with query execution and result fetching.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator

import httpx

from .data_provider import MarketEvent, OHLCVBar

logger = logging.getLogger(__name__)

DUNE_API_BASE = "https://api.dune.com/api/v1"


class DuneQueryStatus(Enum):
    """Status of a Dune query execution."""

    PENDING = "QUERY_STATE_PENDING"
    EXECUTING = "QUERY_STATE_EXECUTING"
    COMPLETED = "QUERY_STATE_COMPLETED"
    FAILED = "QUERY_STATE_FAILED"
    CANCELLED = "QUERY_STATE_CANCELLED"


@dataclass
class DuneQuery:
    """A Dune Analytics saved query."""

    query_id: int
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


# Pre-defined Dune queries for Solana DEX data
DUNE_QUERIES = {
    # Solana DEX trades query
    "solana_dex_trades": DuneQuery(
        query_id=3465987,  # Example query ID - replace with actual
        name="Solana DEX Trades",
        description="All DEX trades on Solana with token details",
    ),
    # Raydium pool creation events
    "raydium_pool_created": DuneQuery(
        query_id=3465988,
        name="Raydium Pool Created Events",
        description="New Raydium AMM pool creations",
    ),
    # PumpFun launches
    "pumpfun_launches": DuneQuery(
        query_id=3465989,
        name="PumpFun Token Launches",
        description="PumpFun bonding curve launches",
    ),
    # MEV sandwich attacks
    "mev_sandwich_attacks": DuneQuery(
        query_id=3465990,
        name="MEV Sandwich Attacks",
        description="Detected sandwich attacks on Solana",
    ),
    # Liquidity pool snapshots
    "pool_liquidity_history": DuneQuery(
        query_id=3465991,
        name="Pool Liquidity History",
        description="Historical liquidity levels for pools",
    ),
}


@dataclass
class DuneProviderConfig:
    """Configuration for Dune data provider."""

    api_key: str | None = None
    rate_limit_per_minute: int = 10  # Dune has stricter rate limits
    timeout_seconds: float = 300.0  # Queries can take longer
    poll_interval_seconds: float = 2.0
    max_poll_attempts: int = 150  # 5 minutes max wait
    cache_results: bool = True
    cache_dir: str | None = None

    @classmethod
    def from_env(cls) -> "DuneProviderConfig":
        return cls(
            api_key=os.getenv("DUNE_API_KEY"),
            cache_dir=os.getenv("DATA_CACHE_DIR"),
        )


class DuneDataProvider:
    """
    Fetches historical data from Dune Analytics.

    Dune provides access to decoded Solana blockchain data including:
    - DEX trades (Raydium, Orca, Jupiter)
    - Token launches (PumpFun, Raydium)
    - MEV activity (sandwich attacks, arbitrage)
    - Pool liquidity changes

    Usage:
        provider = DuneDataProvider.from_env()
        async for event in provider.stream_events(start_ts, end_ts, "dune_ohlcv"):
            process(event)
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: DuneProviderConfig | None = None,
    ):
        self._execution_cache: dict[str, list[dict[str, Any]]]
        self.config = config or DuneProviderConfig(api_key=api_key)
        self.api_key = self.config.api_key or os.getenv("DUNE_API_KEY")
        self.client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        self._rate_limiter = asyncio.Semaphore(self.config.rate_limit_per_minute)
        self._execution_cache: dict[str, list[dict[str, Any]]] = {}

    @classmethod
    def from_env(cls) -> "DuneDataProvider":
        """Create provider from environment variables."""
        return cls(config=DuneProviderConfig.from_env())

    async def stream_events(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        data_source: str,
        mints: list[str] | None = None,
    ) -> AsyncIterator[MarketEvent]:
        """
        Stream market events from Dune Analytics.

        Args:
            start_timestamp_ms: Start time in milliseconds
            end_timestamp_ms: End time in milliseconds
            data_source: One of 'dune_ohlcv', 'dune_trades', 'dune_mev', 'dune_liquidity'
            mints: Optional filter for specific token mints

        Yields:
            MarketEvent objects in chronological order
        """
        if not self.api_key:
            logger.error("Dune API key not configured")
            return

        if data_source in {"dune_ohlcv", "dune_dex_trades"}:
            async for event in self._stream_dex_trades(start_timestamp_ms, end_timestamp_ms, mints):
                yield event

        elif data_source == "dune_trades":
            async for event in self._stream_trades(start_timestamp_ms, end_timestamp_ms, mints):
                yield event

        elif data_source == "dune_launches":
            async for event in self._stream_launches(start_timestamp_ms, end_timestamp_ms):
                yield event

        elif data_source == "dune_mev":
            async for event in self._stream_mev_data(start_timestamp_ms, end_timestamp_ms, mints):
                yield event

        elif data_source == "dune_liquidity":
            async for event in self._stream_liquidity_history(
                start_timestamp_ms, end_timestamp_ms, mints
            ):
                yield event

        else:
            logger.error(f"Unknown Dune data source: {data_source}")

    async def _fetch_dex_trades(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        mints: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch raw DEX trades from Dune."""
        start_dt = datetime.fromtimestamp(start_timestamp_ms / 1000)
        end_dt = datetime.fromtimestamp(end_timestamp_ms / 1000)

        query_sql = f"""
        SELECT
            block_time,
            block_slot,
            tx_id,
            token_mint_address,
            token_symbol,
            side,
            token_amount,
            sol_amount,
            price_usd,
            pool_address,
            program_id,
            trader_address
        FROM solana.dex_trades
        WHERE block_time >= timestamp '{start_dt.strftime("%Y-%m-%d %H:%M:%S")}'
          AND block_time < timestamp '{end_dt.strftime("%Y-%m-%d %H:%M:%S")}'
          {"AND token_mint_address IN ('" + "','".join(mints) + "')" if mints else ""}
        ORDER BY block_time ASC
        LIMIT 100000
        """

        results = await self._execute_query_sql(query_sql)
        return results or []

    async def _stream_dex_trades(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        mints: list[str] | None = None,
    ) -> AsyncIterator[MarketEvent]:
        """Stream DEX trade data from Dune."""
        results = await self._fetch_dex_trades(start_timestamp_ms, end_timestamp_ms, mints)

        if not results:
            return

        # Convert to market events
        trades_by_mint: dict[str, list[dict]] = {}
        for row in results:
            mint = row.get("token_mint_address", "")
            if mint not in trades_by_mint:
                trades_by_mint[mint] = []
            trades_by_mint[mint].append(row)

        # Aggregate into OHLCV (1-minute bars)
        all_events = []
        for mint, trades in trades_by_mint.items():
            bars = self._aggregate_trades_to_ohlcv(trades, interval_ms=60000)
            for bar in bars:
                all_events.append(
                    MarketEvent(
                        timestamp_ms=bar["timestamp_ms"],
                        mint=mint,
                        pool=bar.get("pool", ""),
                        event_type="ohlcv",
                        data=bar,
                    )
                )

        # Sort and yield
        all_events.sort(key=lambda e: e.timestamp_ms)
        for event in all_events:
            yield event

    async def _stream_launches(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
    ) -> AsyncIterator[MarketEvent]:
        """Stream token launch events (PumpFun, Raydium migrations)."""
        start_dt = datetime.fromtimestamp(start_timestamp_ms / 1000)
        end_dt = datetime.fromtimestamp(end_timestamp_ms / 1000)

        query_sql = f"""
        SELECT
            block_time,
            block_slot,
            tx_id,
            token_mint,
            token_name,
            token_symbol,
            creator_address,
            initial_liquidity_sol,
            initial_supply,
            bonding_curve_address,
            event_type
        FROM solana.pumpfun_launches
        WHERE block_time >= timestamp '{start_dt.strftime("%Y-%m-%d %H:%M:%S")}'
          AND block_time < timestamp '{end_dt.strftime("%Y-%m-%d %H:%M:%S")}'
        ORDER BY block_time ASC
        """

        results = await self._execute_query_sql(query_sql)

        if not results:
            return

        for row in results:
            block_time = row.get("block_time")
            if isinstance(block_time, str):
                ts = int(
                    datetime.fromisoformat(block_time.replace("Z", "+00:00")).timestamp() * 1000
                )
            else:
                ts = int(block_time.timestamp() * 1000) if block_time else 0

            yield MarketEvent(
                timestamp_ms=ts,
                mint=row.get("token_mint", ""),
                pool=row.get("bonding_curve_address", ""),
                event_type="pool_created",
                data={
                    "name": row.get("token_name"),
                    "symbol": row.get("token_symbol"),
                    "creator": row.get("creator_address"),
                    "initial_liquidity_sol": row.get("initial_liquidity_sol"),
                    "initial_supply": row.get("initial_supply"),
                    "launch_type": row.get("event_type", "pumpfun"),
                },
            )

    async def _stream_mev_data(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        mints: list[str] | None = None,
    ) -> AsyncIterator[MarketEvent]:
        """Stream MEV/sandwich attack data for realistic backtesting."""
        start_dt = datetime.fromtimestamp(start_timestamp_ms / 1000)
        end_dt = datetime.fromtimestamp(end_timestamp_ms / 1000)

        query_sql = f"""
        SELECT
            block_time,
            block_slot,
            victim_tx,
            frontrun_tx,
            backrun_tx,
            token_mint,
            pool_address,
            victim_amount_sol,
            profit_sol,
            price_impact_bps,
            searcher_address
        FROM solana.mev_sandwich_attacks
        WHERE block_time >= timestamp '{start_dt.strftime("%Y-%m-%d %H:%M:%S")}'
          AND block_time < timestamp '{end_dt.strftime("%Y-%m-%d %H:%M:%S")}'
          {"AND token_mint IN ('" + "','".join(mints) + "')" if mints else ""}
        ORDER BY block_time ASC
        """

        results = await self._execute_query_sql(query_sql)

        if not results:
            return

        for row in results:
            block_time = row.get("block_time")
            if isinstance(block_time, str):
                ts = int(
                    datetime.fromisoformat(block_time.replace("Z", "+00:00")).timestamp() * 1000
                )
            else:
                ts = int(block_time.timestamp() * 1000) if block_time else 0

            yield MarketEvent(
                timestamp_ms=ts,
                mint=row.get("token_mint", ""),
                pool=row.get("pool_address", ""),
                event_type="mev_sandwich",
                data={
                    "victim_tx": row.get("victim_tx"),
                    "frontrun_tx": row.get("frontrun_tx"),
                    "backrun_tx": row.get("backrun_tx"),
                    "victim_amount_sol": row.get("victim_amount_sol"),
                    "profit_sol": row.get("profit_sol"),
                    "price_impact_bps": row.get("price_impact_bps"),
                    "searcher": row.get("searcher_address"),
                },
            )

    async def _stream_liquidity_history(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        mints: list[str] | None = None,
    ) -> AsyncIterator[MarketEvent]:
        """Stream pool liquidity snapshots for accurate slippage modeling."""
        start_dt = datetime.fromtimestamp(start_timestamp_ms / 1000)
        end_dt = datetime.fromtimestamp(end_timestamp_ms / 1000)

        query_sql = f"""
        SELECT
            snapshot_time,
            pool_address,
            token_mint,
            token_reserve,
            sol_reserve,
            total_liquidity_usd,
            volume_24h_usd,
            fee_tier_bps
        FROM solana.pool_liquidity_snapshots
        WHERE snapshot_time >= timestamp '{start_dt.strftime("%Y-%m-%d %H:%M:%S")}'
          AND snapshot_time < timestamp '{end_dt.strftime("%Y-%m-%d %H:%M:%S")}'
          {"AND token_mint IN ('" + "','".join(mints) + "')" if mints else ""}
        ORDER BY snapshot_time ASC
        """

        results = await self._execute_query_sql(query_sql)

        if not results:
            return

        for row in results:
            snapshot_time = row.get("snapshot_time")
            if isinstance(snapshot_time, str):
                ts = int(
                    datetime.fromisoformat(snapshot_time.replace("Z", "+00:00")).timestamp() * 1000
                )
            else:
                ts = int(snapshot_time.timestamp() * 1000) if snapshot_time else 0

            yield MarketEvent(
                timestamp_ms=ts,
                mint=row.get("token_mint", ""),
                pool=row.get("pool_address", ""),
                event_type="liquidity_snapshot",
                data={
                    "token_reserve": row.get("token_reserve"),
                    "sol_reserve": row.get("sol_reserve"),
                    "liquidity_usd": row.get("total_liquidity_usd"),
                    "volume_24h": row.get("volume_24h_usd"),
                    "fee_bps": row.get("fee_tier_bps"),
                },
            )

    def _aggregate_trades_to_ohlcv(
        self,
        trades: list[dict],
        interval_ms: int = 60000,
    ) -> list[dict]:
        """Aggregate individual trades into OHLCV bars."""
        if not trades:
            return []

        bars_by_interval: dict[int, dict] = {}

        for trade in trades:
            block_time = trade.get("block_time")
            if isinstance(block_time, str):
                ts = int(
                    datetime.fromisoformat(block_time.replace("Z", "+00:00")).timestamp() * 1000
                )
            else:
                ts = int(block_time.timestamp() * 1000) if block_time else 0

            interval_ts = (ts // interval_ms) * interval_ms
            price = float(trade.get("price_usd", 0))
            volume_tokens = float(trade.get("token_amount", 0))
            volume_sol = float(trade.get("sol_amount", 0))

            if price <= 0:
                continue

            if interval_ts not in bars_by_interval:
                bars_by_interval[interval_ts] = {
                    "timestamp_ms": interval_ts,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volume_tokens,
                    "volume_sol": volume_sol,
                    "pool": trade.get("pool_address", ""),
                    "liquidity_sol": None,
                    "liquidity_tokens": None,
                    "liquidity_usd": None,
                }
            else:
                bar = bars_by_interval[interval_ts]
                bar["high"] = max(bar["high"], price)
                bar["low"] = min(bar["low"], price)
                bar["close"] = price
                bar["volume"] += volume_tokens
                bar["volume_sol"] += volume_sol

        return list(bars_by_interval.values())

    async def _execute_query_sql(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Execute a SQL query on Dune Analytics.

        Uses the query execution API to run custom SQL.
        """
        if not self.api_key:
            logger.error("Dune API key not configured")
            return None

        # Check cache
        cache_key = f"{sql}_{parameters}"
        if cache_key in self._execution_cache:
            return self._execution_cache[cache_key]

        headers = {
            "X-Dune-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with self._rate_limiter:
                # Execute query
                execute_url = f"{DUNE_API_BASE}/query/execute"
                execute_payload: dict[str, Any] = {
                    "query_sql": sql,
                    "performance": "medium",  # medium or large
                }
                if parameters:
                    execute_payload["query_parameters"] = parameters

                response = await self.client.post(
                    execute_url,
                    json=execute_payload,
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()

                execution_id = result.get("execution_id")
                if not execution_id:
                    logger.error("No execution_id in response")
                    return None

                logger.info(f"Dune query executing: {execution_id}")

                # Poll for results
                results = await self._poll_execution(execution_id, headers)

                # Cache results with FIFO eviction
                if results and self.config.cache_results:
                    if len(self._execution_cache) >= 100:
                        # Evict oldest entry (FIFO)
                        oldest_key = next(iter(self._execution_cache))
                        del self._execution_cache[oldest_key]
                    self._execution_cache[cache_key] = results

                return results

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Dune rate limited, waiting...")
                await asyncio.sleep(60)
            else:
                logger.error(f"Dune API error: {e.response.status_code} - {e.response.text}")
            return None

        except Exception as e:
            logger.error(f"Dune query error: {e}")
            return None

    async def _poll_execution(
        self,
        execution_id: str,
        headers: dict[str, str],
    ) -> list[dict] | None:
        """Poll for query execution results."""
        status_url = f"{DUNE_API_BASE}/execution/{execution_id}/status"
        results_url = f"{DUNE_API_BASE}/execution/{execution_id}/results"

        for attempt in range(self.config.max_poll_attempts):
            try:
                # Check status
                response = await self.client.get(status_url, headers=headers)
                response.raise_for_status()
                status_data = response.json()

                state = status_data.get("state")

                if state == DuneQueryStatus.COMPLETED.value:
                    # Fetch results
                    results_response = await self.client.get(
                        results_url,
                        headers=headers,
                    )
                    results_response.raise_for_status()
                    results_data = results_response.json()

                    rows = results_data.get("result", {}).get("rows", [])
                    logger.info(f"Dune query completed: {len(rows)} rows")
                    return rows

                elif state == DuneQueryStatus.FAILED.value:
                    error = status_data.get("error")
                    logger.error(f"Dune query failed: {error}")
                    return None

                elif state == DuneQueryStatus.CANCELLED.value:
                    logger.warning("Dune query was cancelled")
                    return None

                else:
                    # Still executing
                    logger.debug(f"Dune query state: {state}, attempt {attempt + 1}")
                    await asyncio.sleep(self.config.poll_interval_seconds)

            except Exception as e:
                logger.error(f"Error polling Dune execution: {e}")
                await asyncio.sleep(self.config.poll_interval_seconds)

        logger.error(f"Dune query timed out after {self.config.max_poll_attempts} attempts")
        return None

    async def execute_saved_query(
        self,
        query_name: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict] | None:
        """Execute a pre-defined saved query.

        Args:
            query_name: Name of query from DUNE_QUERIES
            parameters: Query parameters

        Returns:
            List of result rows or None
        """
        if query_name not in DUNE_QUERIES:
            logger.error(f"Unknown query: {query_name}")
            return None

        query = DUNE_QUERIES[query_name]
        return await self._execute_saved_query(query.query_id, parameters)

    async def _execute_saved_query(
        self,
        query_id: int,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict] | None:
        """Execute a saved Dune query by ID."""
        if not self.api_key:
            logger.error("Dune API key not configured")
            return None

        headers = {
            "X-Dune-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with self._rate_limiter:
                execute_url = f"{DUNE_API_BASE}/query/{query_id}/execute"
                payload = {}
                if parameters:
                    payload["query_parameters"] = parameters

                response = await self.client.post(
                    execute_url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()

                execution_id = result.get("execution_id")
                if not execution_id:
                    return None

                return await self._poll_execution(execution_id, headers)

        except Exception as e:
            logger.error(f"Error executing saved query {query_id}: {e}")
            return None

    async def get_token_history(
        self,
        mint: str,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
    ) -> list[OHLCVBar]:
        """Get OHLCV history for a specific token.

        Convenience method that returns OHLCVBar objects.
        """
        bars = []
        async for event in self._stream_dex_trades(
            start_timestamp_ms, end_timestamp_ms, mints=[mint]
        ):
            if event.mint == mint and event.event_type == "ohlcv":
                data = event.data
                bars.append(
                    OHLCVBar(
                        timestamp_ms=data.get("timestamp_ms", event.timestamp_ms),
                        open=data.get("open", 0),
                        high=data.get("high", 0),
                        low=data.get("low", 0),
                        close=data.get("close", 0),
                        volume=data.get("volume", 0),
                        liquidity_usd=data.get("liquidity_usd"),
                    )
                )
        return bars

    async def get_mev_statistics(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        mints: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get MEV statistics for a time period.

        Returns aggregate statistics about sandwich attacks for
        realistic slippage modeling in backtesting.
        """
        events = []
        async for event in self._stream_mev_data(start_timestamp_ms, end_timestamp_ms, mints):
            events.append(event.data)

        if not events:
            return {
                "total_attacks": 0,
                "avg_profit_sol": 0,
                "avg_price_impact_bps": 0,
                "total_victim_volume_sol": 0,
            }

        profits = [e.get("profit_sol", 0) for e in events]
        impacts = [e.get("price_impact_bps", 0) for e in events]
        volumes = [e.get("victim_amount_sol", 0) for e in events]

        return {
            "total_attacks": len(events),
            "avg_profit_sol": sum(profits) / len(profits) if profits else 0,
            "avg_price_impact_bps": sum(impacts) / len(impacts) if impacts else 0,
            "total_victim_volume_sol": sum(volumes),
            "max_profit_sol": max(profits) if profits else 0,
            "percentile_95_impact": sorted(impacts)[int(len(impacts) * 0.95)] if impacts else 0,
        }

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _stream_trades(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        mints: list[str] | None = None,
    ) -> AsyncIterator[MarketEvent]:
        """Stream trade-level data from Dune."""
        results = await self._fetch_dex_trades(start_timestamp_ms, end_timestamp_ms, mints)

        if not results:
            return

        for row in results:
            block_time = row.get("block_time")
            if isinstance(block_time, str):
                ts = int(
                    datetime.fromisoformat(block_time.replace("Z", "+00:00")).timestamp() * 1000
                )
            else:
                ts = int(block_time.timestamp() * 1000) if block_time else 0

            yield MarketEvent(
                timestamp_ms=ts,
                mint=row.get("token_mint_address", ""),
                pool=row.get("pool_address", ""),
                event_type="trade",
                data={
                    "side": row.get("side"),
                    "price": row.get("price_usd"),
                    "amount_tokens": row.get("token_amount"),
                    "amount_sol": row.get("sol_amount"),
                    "volume": row.get("token_amount"),
                    "volume_sol": row.get("sol_amount"),
                    "liquidity_sol": None,
                    "liquidity_tokens": None,
                    "liquidity_usd": None,
                    "tx_id": row.get("tx_id"),
                    "trader": row.get("trader_address"),
                },
            )
