"""
PostgreSQL storage backend implementation.

Uses asyncpg for high-performance async PostgreSQL operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

from .storage_backend import StorageBackend

logger = logging.getLogger(__name__)


class PostgresStorageBackend(StorageBackend):
    """PostgreSQL storage backend using asyncpg."""

    def __init__(self, database_url: str):
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is required for PostgreSQL backend. Install with: pip install asyncpg"
            )
        self.database_url = database_url
        self._pool: asyncpg.Pool | None = None

    async def _init_connection(self, conn):
        """Initialize each connection in the pool."""
        await conn.set_type_codec(
            "jsonb",
            encoder=lambda x: x,
            decoder=lambda x: x,
            schema="pg_catalog",
            format="text",
        )

    async def connect(self) -> None:
        """Initialize the connection pool."""
        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
            init=self._init_connection,
        )
        await self._create_tables()
        logger.info("PostgreSQL backend connected")

    async def _create_tables(self) -> None:
        """Create necessary tables if they don't exist."""
        async with self._pool.acquire() as conn:
            # Trades table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    timestamp_ms BIGINT NOT NULL,
                    mint TEXT NOT NULL,
                    symbol TEXT,
                    side TEXT NOT NULL,
                    amount_sol DOUBLE PRECISION NOT NULL,
                    amount_tokens DOUBLE PRECISION NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    slippage_bps INTEGER,
                    quoted_slippage_bps INTEGER,
                    pnl_sol DOUBLE PRECISION,
                    pnl_pct DOUBLE PRECISION,
                    execution_mode TEXT,
                    included BOOLEAN DEFAULT TRUE,
                    latency_ms INTEGER,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp
                ON trades(timestamp_ms)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_mint
                ON trades(mint)
            """)

            # Keep schema aligned with insert_trade() and SQLite backend migrations.
            # This is idempotent and safe for existing deployments.
            await conn.execute("""
                ALTER TABLE trades
                ADD COLUMN IF NOT EXISTS base_fee_lamports BIGINT DEFAULT 0
            """)
            await conn.execute("""
                ALTER TABLE trades
                ADD COLUMN IF NOT EXISTS priority_fee_lamports BIGINT DEFAULT 0
            """)
            await conn.execute("""
                ALTER TABLE trades
                ADD COLUMN IF NOT EXISTS jito_tip_lamports BIGINT DEFAULT 0
            """)
            await conn.execute("""
                ALTER TABLE trades
                ADD COLUMN IF NOT EXISTS dex_fee_lamports BIGINT DEFAULT 0
            """)
            await conn.execute("""
                ALTER TABLE trades
                ADD COLUMN IF NOT EXISTS total_fees_sol DOUBLE PRECISION
            """)
            await conn.execute("""
                ALTER TABLE trades
                ADD COLUMN IF NOT EXISTS net_pnl_sol DOUBLE PRECISION
            """)
            await conn.execute("""
                ALTER TABLE trades
                ADD COLUMN IF NOT EXISTS tx_signature TEXT
            """)
            await conn.execute("""
                ALTER TABLE trades
                ADD COLUMN IF NOT EXISTS slot BIGINT
            """)
            await conn.execute("""
                ALTER TABLE trades
                ADD COLUMN IF NOT EXISTS execution_time_ms INTEGER
            """)
            await conn.execute("""
                ALTER TABLE trades
                ADD COLUMN IF NOT EXISTS amount_out_lamports BIGINT
            """)
            await conn.execute("""
                UPDATE trades
                SET total_fees_sol = (
                    COALESCE(base_fee_lamports, 0) +
                    COALESCE(priority_fee_lamports, 0) +
                    COALESCE(jito_tip_lamports, 0) +
                    COALESCE(dex_fee_lamports, 0)
                )::DOUBLE PRECISION / 1000000000.0
                WHERE total_fees_sol IS NULL
            """)
            await conn.execute("""
                UPDATE trades
                SET net_pnl_sol = COALESCE(pnl_sol, 0) - COALESCE(total_fees_sol, 0)
                WHERE net_pnl_sol IS NULL
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_net_pnl
                ON trades(net_pnl_sol)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_total_fees
                ON trades(total_fees_sol)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_tx_signature
                ON trades(tx_signature)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_slot
                ON trades(slot)
            """)

            # Fill observations table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS fill_observations (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp_ms BIGINT NOT NULL,
                    mode TEXT NOT NULL,
                    quoted_price DOUBLE PRECISION NOT NULL,
                    fill_price DOUBLE PRECISION,
                    quoted_slippage_bps INTEGER NOT NULL,
                    realized_slippage_bps INTEGER,
                    latency_ms INTEGER,
                    included BOOLEAN NOT NULL,
                    mev_detected BOOLEAN DEFAULT FALSE,
                    pool_address TEXT,
                    liquidity_usd DOUBLE PRECISION,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fill_obs_timestamp
                ON fill_observations(timestamp_ms)
            """)

            # Model artifacts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_artifacts (
                    id BIGSERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    params_json JSONB NOT NULL,
                    metrics_json JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(model_name, version)
                )
            """)

            # Bandit arms table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS bandit_arms (
                    arm_name TEXT PRIMARY KEY,
                    slippage_bps INTEGER NOT NULL,
                    pull_count INTEGER DEFAULT 0,
                    total_reward DOUBLE PRECISION DEFAULT 0,
                    last_updated TIMESTAMPTZ DEFAULT NOW()
                )
            """)

        logger.debug("PostgreSQL tables initialized")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection closed")

    async def health_check(self) -> bool:
        """Check if the connection is healthy."""
        if not self._pool:
            return False
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    @property
    def backend_name(self) -> str:
        return "postgres"

    async def get_trade_count(self, since: datetime | None = None) -> int:
        """Get count of trades, optionally since a given time."""
        async with self._pool.acquire() as conn:
            if since:
                since_ms = int(since.timestamp() * 1000)
                return await conn.fetchval(
                    "SELECT COUNT(*) FROM trades WHERE timestamp_ms >= $1",
                    since_ms,
                )
            else:
                return await conn.fetchval("SELECT COUNT(*) FROM trades")

    async def get_recent_trades(self, hours: int = 24) -> pd.DataFrame:
        """Get recent trades as a DataFrame."""
        since_ms = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM trades
                WHERE timestamp_ms >= $1
                ORDER BY timestamp_ms ASC
                """,
                since_ms,
            )

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([dict(row) for row in rows])

    async def insert_trade(self, trade: dict[str, Any]) -> None:
        """Insert a new trade record with complete fee and execution details."""
        async with self._pool.acquire() as conn:
            # Calculate computed fields
            base_fee = trade.get("base_fee_lamports", 0) or 0
            priority_fee = trade.get("priority_fee_lamports", 0) or 0
            jito_tip = trade.get("jito_tip_lamports", 0) or 0
            dex_fee = trade.get("dex_fee_lamports", 0) or 0
            total_fees_sol = (base_fee + priority_fee + jito_tip + dex_fee) / 1_000_000_000.0
            pnl_sol = trade.get("pnl_sol") or 0.0
            net_pnl_sol = pnl_sol - total_fees_sol

            await conn.execute(
                """
                INSERT INTO trades
                (id, timestamp_ms, mint, symbol, side, amount_sol, amount_tokens,
                 price, slippage_bps, quoted_slippage_bps, pnl_sol, pnl_pct,
                 execution_mode, included, latency_ms,
                 base_fee_lamports, priority_fee_lamports, jito_tip_lamports, dex_fee_lamports,
                 total_fees_sol, net_pnl_sol,
                 tx_signature, slot, execution_time_ms, amount_out_lamports)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                        $16, $17, $18, $19, $20, $21, $22, $23, $24, $25)
                ON CONFLICT (id) DO UPDATE SET
                    pnl_sol = EXCLUDED.pnl_sol,
                    pnl_pct = EXCLUDED.pnl_pct,
                    base_fee_lamports = EXCLUDED.base_fee_lamports,
                    priority_fee_lamports = EXCLUDED.priority_fee_lamports,
                    jito_tip_lamports = EXCLUDED.jito_tip_lamports,
                    dex_fee_lamports = EXCLUDED.dex_fee_lamports,
                    total_fees_sol = EXCLUDED.total_fees_sol,
                    net_pnl_sol = EXCLUDED.net_pnl_sol,
                    tx_signature = EXCLUDED.tx_signature,
                    slot = EXCLUDED.slot,
                    execution_time_ms = EXCLUDED.execution_time_ms,
                    amount_out_lamports = EXCLUDED.amount_out_lamports
                """,
                trade.get("id"),
                trade.get("timestamp_ms"),
                trade.get("mint"),
                trade.get("symbol"),
                trade.get("side"),
                trade.get("amount_sol"),
                trade.get("amount_tokens"),
                trade.get("price"),
                trade.get("slippage_bps"),
                trade.get("quoted_slippage_bps"),
                pnl_sol,
                trade.get("pnl_pct"),
                trade.get("execution_mode"),
                trade.get("included", True),
                trade.get("latency_ms"),
                # Fee fields
                base_fee,
                priority_fee,
                jito_tip,
                dex_fee,
                total_fees_sol,
                net_pnl_sol,
                # Execution detail fields
                trade.get("tx_signature"),
                trade.get("slot"),
                trade.get("execution_time_ms"),
                trade.get("amount_out_lamports"),
            )

    async def get_fill_observation_count(self, since: datetime | None = None) -> int:
        """Get count of fill observations."""
        async with self._pool.acquire() as conn:
            if since:
                since_ms = int(since.timestamp() * 1000)
                return await conn.fetchval(
                    "SELECT COUNT(*) FROM fill_observations WHERE timestamp_ms >= $1",
                    since_ms,
                )
            else:
                return await conn.fetchval("SELECT COUNT(*) FROM fill_observations")

    async def get_fill_observations(self, hours: int = 6) -> list[dict[str, Any]]:
        """Get recent fill observations."""
        since_ms = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM fill_observations
                WHERE timestamp_ms >= $1
                ORDER BY timestamp_ms ASC
                """,
                since_ms,
            )

        return [dict(row) for row in rows]

    async def insert_fill_observation(self, obs: dict[str, Any]) -> None:
        """Insert a fill observation."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO fill_observations
                (timestamp_ms, mode, quoted_price, fill_price, quoted_slippage_bps,
                 realized_slippage_bps, latency_ms, included, mev_detected,
                 pool_address, liquidity_usd)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                obs.get("timestamp_ms"),
                obs.get("mode"),
                obs.get("quoted_price"),
                obs.get("fill_price"),
                obs.get("quoted_slippage_bps"),
                obs.get("realized_slippage_bps"),
                obs.get("latency_ms"),
                obs.get("included", True),
                obs.get("mev_detected", False),
                obs.get("pool_address"),
                obs.get("liquidity_usd"),
            )

    async def save_bandit_arm(
        self, arm_name: str, slippage_bps: int, pull_count: int, total_reward: float
    ) -> None:
        """Save bandit arm statistics."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO bandit_arms
                (arm_name, slippage_bps, pull_count, total_reward, last_updated)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (arm_name) DO UPDATE SET
                    slippage_bps = EXCLUDED.slippage_bps,
                    pull_count = EXCLUDED.pull_count,
                    total_reward = EXCLUDED.total_reward,
                    last_updated = NOW()
                """,
                arm_name,
                slippage_bps,
                pull_count,
                total_reward,
            )

    async def get_bandit_arms(self) -> dict[str, dict[str, Any]]:
        """Get all bandit arm statistics."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM bandit_arms")
        return {row["arm_name"]: dict(row) for row in rows}

    # Additional PostgreSQL-specific methods

    async def vacuum_analyze(self) -> None:
        """Run VACUUM ANALYZE to optimize query performance."""
        async with self._pool.acquire() as conn:
            await conn.execute("VACUUM ANALYZE")
        logger.info("VACUUM ANALYZE completed")

    async def get_table_sizes(self) -> dict[str, int]:
        """Get size of all tables in bytes."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    relname as table_name,
                    pg_total_relation_size(relid) as total_size
                FROM pg_catalog.pg_statio_user_tables
                ORDER BY pg_total_relation_size(relid) DESC
                """
            )
        return {row["table_name"]: row["total_size"] for row in rows}
