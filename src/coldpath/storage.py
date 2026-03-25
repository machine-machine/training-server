"""
Storage layer for Cold Path engine.

Manages database connections and model artifact persistence.
"""

import asyncio
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EVLabel:
    """Canonical first-class EV outcome labels."""

    GOOD_EV = "GOOD_EV"
    BAD_EV = "BAD_EV"
    RUG = "RUG"
    MEV_HIT = "MEV_HIT"
    SLIPPAGE_FAIL = "SLIPPAGE_FAIL"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"

    ALL = {
        GOOD_EV,
        BAD_EV,
        RUG,
        MEV_HIT,
        SLIPPAGE_FAIL,
        EXECUTION_TIMEOUT,
    }


def canonicalize_ev_label(outcome: dict[str, Any]) -> str:
    """Map heterogeneous outcome metadata to canonical EV labels."""
    explicit = outcome.get("ev_label")
    if isinstance(explicit, str) and explicit in EVLabel.ALL:
        return explicit

    token_status = str(outcome.get("token_status", "")).lower()
    if "rug" in token_status or "honeypot" in token_status:
        return EVLabel.RUG
    if "timeout" in token_status:
        return EVLabel.EXECUTION_TIMEOUT
    if "mev" in token_status or "sandwich" in token_status:
        return EVLabel.MEV_HIT
    if "slippage" in token_status:
        return EVLabel.SLIPPAGE_FAIL

    pnl_pct = outcome.get("pnl_pct")
    if isinstance(pnl_pct, (int, float)):
        return EVLabel.GOOD_EV if pnl_pct >= 0 else EVLabel.BAD_EV

    label_binary = outcome.get("label_binary")
    if isinstance(label_binary, int):
        return EVLabel.GOOD_EV if label_binary == 1 else EVLabel.BAD_EV

    return EVLabel.BAD_EV


class DatabaseManager:
    """Async SQLite database manager for Cold Path.

    Handles trade data, fill observations, and model metadata.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is not None:
            try:
                self._connection.execute("SELECT 1")
                return self._connection
            except (sqlite3.OperationalError, sqlite3.ProgrammingError):
                logger.warning("Database connection lost, attempting reconnection...")
                try:
                    self._connection.close()
                except Exception:
                    pass
                self._connection = None

        # Attempt reconnection
        try:
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,
            )
            self._connection.row_factory = sqlite3.Row
            logger.info(f"Reconnected to database: {self.db_path}")
            return self._connection
        except Exception as e:
            raise RuntimeError(f"Database reconnection failed: {e}") from e

    @contextmanager
    def _cursor(self):
        cursor = self._require_connection().cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    async def __aenter__(self):
        """Async context manager entry - connects to database."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes database connection."""
        await self.close()
        return False

    async def connect(self):
        """Initialize database connection and create tables."""
        async with self._lock:
            # Close existing connection to prevent leaks on repeated calls
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception:
                    pass
                self._connection = None

            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,  # Autocommit
            )
            self._connection.row_factory = sqlite3.Row
            await self._create_tables()
            logger.info(f"Connected to database: {self.db_path}")

    async def _create_tables(self):
        """Create necessary tables if they don't exist."""
        with self._cursor() as cursor:
            # Schema version table for migrations
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Trades table for bandit training
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    timestamp_ms INTEGER NOT NULL,
                    mint TEXT NOT NULL,
                    symbol TEXT,
                    side TEXT NOT NULL,
                    amount_sol REAL NOT NULL,
                    amount_tokens REAL NOT NULL,
                    price REAL NOT NULL,
                    slippage_bps INTEGER,
                    quoted_slippage_bps INTEGER,
                    pnl_sol REAL,
                    pnl_pct REAL,
                    execution_mode TEXT,
                    included INTEGER DEFAULT 1,
                    latency_ms INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Create indexes for trades table
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp
                ON trades(timestamp_ms)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trades_mint
                ON trades(mint)
                """
            )

            # Fill observations for paper fill calibration
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS fill_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_ms INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    quoted_price REAL NOT NULL,
                    fill_price REAL,
                    quoted_slippage_bps INTEGER NOT NULL,
                    realized_slippage_bps INTEGER,
                    latency_ms INTEGER,
                    included INTEGER NOT NULL,
                    mev_detected INTEGER DEFAULT 0,
                    pool_address TEXT,
                    liquidity_usd REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Model artifacts metadata
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    params_json TEXT NOT NULL,
                    metrics_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_name, version)
                )
                """
            )

            # Bandit arm statistics
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS bandit_arms (
                    arm_name TEXT PRIMARY KEY,
                    slippage_bps INTEGER NOT NULL,
                    pull_count INTEGER DEFAULT 0,
                    total_reward REAL DEFAULT 0,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Scan outcomes for profitability learning (new table)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS scan_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mint TEXT NOT NULL,
                    pool TEXT,
                    timestamp_ms INTEGER NOT NULL,
                    outcome_type TEXT NOT NULL,
                    skip_reason TEXT,
                    features_json TEXT NOT NULL,
                    profitability_score REAL,
                    confidence REAL,
                    expected_return_pct REAL,
                    model_version INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'scanner',
                    -- For trades
                    entry_price REAL,
                    exit_price REAL,
                    pnl_pct REAL,
                    pnl_sol REAL,
                    hold_duration_ms INTEGER,
                    -- For skipped tokens (counterfactual)
                    price_at_skip REAL,
                    price_1h_later REAL,
                    price_24h_later REAL,
                    was_profitable_counterfactual INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Create indexes for scan_outcomes
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scan_outcomes_timestamp
                ON scan_outcomes(timestamp_ms)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scan_outcomes_mint
                ON scan_outcomes(mint)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scan_outcomes_type
                ON scan_outcomes(outcome_type)
                """
            )

            # ========== ML System Tables (Phase 7) ==========

            # Training outcomes with full decision context
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS training_outcomes (
                    record_id TEXT PRIMARY KEY,
                    mint TEXT NOT NULL,
                    decision_timestamp_ms INTEGER NOT NULL,
                    regime TEXT,
                    features_json TEXT NOT NULL,
                    model_version INTEGER NOT NULL,
                    model_score REAL,
                    execution_mode TEXT,
                    pnl_sol REAL,
                    pnl_pct REAL,
                    token_status TEXT,
                    ev_label TEXT,
                    label_binary INTEGER,
                    label_return REAL,
                    label_quality REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Indexes for training_outcomes
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_training_outcomes_timestamp
                ON training_outcomes(decision_timestamp_ms)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_training_outcomes_regime
                ON training_outcomes(regime)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_training_outcomes_mint
                ON training_outcomes(mint)
                """
            )

            # Counterfactual analysis for skipped trades
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS counterfactuals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    base_record_id TEXT NOT NULL,
                    counterfactual_type TEXT NOT NULL,
                    cf_pnl_sol REAL,
                    cf_pnl_pct REAL,
                    regret REAL,
                    analysis_timestamp_ms INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (base_record_id) REFERENCES training_outcomes(record_id)
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_counterfactuals_base_record
                ON counterfactuals(base_record_id)
                """
            )

            # A/B experiments tracking
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ab_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    control_version INTEGER NOT NULL,
                    treatment_version INTEGER NOT NULL,
                    traffic_split REAL DEFAULT 0.5,
                    status TEXT DEFAULT 'active',
                    winner TEXT,
                    start_timestamp_ms INTEGER,
                    end_timestamp_ms INTEGER,
                    control_trades INTEGER DEFAULT 0,
                    treatment_trades INTEGER DEFAULT 0,
                    control_win_rate REAL,
                    treatment_win_rate REAL,
                    control_sharpe REAL,
                    treatment_sharpe REAL,
                    p_value REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ab_experiments_status
                ON ab_experiments(status)
                """
            )

            # Mutual Information scores history
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS mi_scores_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calculation_date TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    mi_score REAL NOT NULL,
                    normalized_mi REAL,
                    redundancy_score REAL,
                    selection_rank INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(calculation_date, regime, feature_name)
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_mi_scores_regime
                ON mi_scores_history(regime)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_mi_scores_date
                ON mi_scores_history(calculation_date)
                """
            )

            # Regime detection history
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS regime_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_ms INTEGER NOT NULL,
                    detected_regime TEXT NOT NULL,
                    regime_probabilities_json TEXT,
                    features_json TEXT,
                    confidence REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_regime_history_timestamp
                ON regime_history(timestamp_ms)
                """
            )

            # Trading signals from AutoTrader
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trading_signals (
                    signal_id TEXT PRIMARY KEY,
                    timestamp_ms INTEGER NOT NULL,
                    token_mint TEXT NOT NULL,
                    token_symbol TEXT,
                    pool_address TEXT,
                    action TEXT NOT NULL,
                    position_size_sol REAL NOT NULL,
                    slippage_bps INTEGER,
                    expected_price REAL,
                    confidence REAL,
                    fraud_score REAL,
                    status TEXT DEFAULT 'pending',
                    plan_id TEXT,
                    expires_at_ms INTEGER,
                    submitted_at_ms INTEGER,
                    completed_at_ms INTEGER,
                    pnl_sol REAL,
                    pnl_pct REAL,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trading_signals_timestamp
                ON trading_signals(timestamp_ms)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trading_signals_status
                ON trading_signals(status)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trading_signals_mint
                ON trading_signals(token_mint)
                """
            )

            # Monte Carlo simulation results
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS monte_carlo_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    num_paths INTEGER NOT NULL,
                    sharpe_mean REAL,
                    sharpe_std REAL,
                    sharpe_ci_lower REAL,
                    sharpe_ci_upper REAL,
                    max_drawdown_mean REAL,
                    max_drawdown_std REAL,
                    var_95 REAL,
                    cvar_95 REAL,
                    risk_of_ruin REAL,
                    final_capital_mean REAL,
                    final_capital_std REAL,
                    config_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_monte_carlo_strategy
                ON monte_carlo_results(strategy_name)
                """
            )

            # ========== Feature Cache (Performance Optimization) ==========
            # Stores pre-computed feature vectors to avoid recomputation during training
            # Provides 3-5x speedup on training runs
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_cache (
                    outcome_id INTEGER PRIMARY KEY,
                    feature_version INTEGER NOT NULL DEFAULT 1,
                    features_binary BLOB NOT NULL,
                    feature_hash TEXT,
                    cached_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (outcome_id) REFERENCES scan_outcomes(id)
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feature_cache_version
                ON feature_cache(feature_version)
                """
            )

            self._require_connection().commit()

        await self._run_migrations()
        logger.debug("Database tables initialized")

    async def _run_migrations(self):
        """Run database schema migrations."""
        with self._cursor() as cursor:
            # Get current schema version
            cursor.execute("SELECT MAX(version) as v FROM schema_version")
            row = cursor.fetchone()
            current_version = row["v"] if row["v"] is not None else 0

            # Migration 1: Add fee tracking columns (2026-02-07)
            if current_version < 1:
                logger.info("Applying migration 1: Fee tracking enhancement")

                # Check if columns already exist (for idempotency)
                cursor.execute("PRAGMA table_info(trades)")
                columns = {row[1] for row in cursor.fetchall()}

                if "base_fee_lamports" not in columns:
                    cursor.execute(
                        "ALTER TABLE trades ADD COLUMN base_fee_lamports INTEGER DEFAULT 0"
                    )

                if "priority_fee_lamports" not in columns:
                    cursor.execute(
                        "ALTER TABLE trades ADD COLUMN priority_fee_lamports INTEGER DEFAULT 0"
                    )

                if "jito_tip_lamports" not in columns:
                    cursor.execute(
                        "ALTER TABLE trades ADD COLUMN jito_tip_lamports INTEGER DEFAULT 0"
                    )

                if "dex_fee_lamports" not in columns:
                    cursor.execute(
                        "ALTER TABLE trades ADD COLUMN dex_fee_lamports INTEGER DEFAULT 0"
                    )

                if "total_fees_sol" not in columns:
                    cursor.execute("ALTER TABLE trades ADD COLUMN total_fees_sol REAL")

                if "net_pnl_sol" not in columns:
                    cursor.execute("ALTER TABLE trades ADD COLUMN net_pnl_sol REAL")

                # Backfill computed columns for existing records
                cursor.execute(
                    """
                    UPDATE trades
                    SET total_fees_sol = CAST(
                        COALESCE(base_fee_lamports, 0) +
                        COALESCE(priority_fee_lamports, 0) +
                        COALESCE(jito_tip_lamports, 0) +
                        COALESCE(dex_fee_lamports, 0)
                        AS REAL) / 1000000000.0
                    WHERE total_fees_sol IS NULL
                    """
                )

                cursor.execute(
                    """
                    UPDATE trades
                    SET net_pnl_sol = COALESCE(pnl_sol, 0) - COALESCE(total_fees_sol, 0)
                    WHERE net_pnl_sol IS NULL
                    """
                )

                # Create indexes
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_net_pnl ON trades(net_pnl_sol)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_total_fees ON trades(total_fees_sol)"
                )

                # Mark migration as applied
                cursor.execute("INSERT INTO schema_version (version) VALUES (1)")

                self._require_connection().commit()
                logger.info("Migration 1 complete: Fee tracking columns added")

            # Migration 2: Add execution detail columns (2026-02-07)
            if current_version < 2:
                logger.info("Applying migration 2: Execution details enhancement")

                # Check if columns already exist (for idempotency)
                cursor.execute("PRAGMA table_info(trades)")
                columns = {row[1] for row in cursor.fetchall()}

                if "tx_signature" not in columns:
                    cursor.execute("ALTER TABLE trades ADD COLUMN tx_signature TEXT")
                    logger.debug("Added tx_signature column")

                if "slot" not in columns:
                    cursor.execute("ALTER TABLE trades ADD COLUMN slot INTEGER")
                    logger.debug("Added slot column")

                if "execution_time_ms" not in columns:
                    cursor.execute("ALTER TABLE trades ADD COLUMN execution_time_ms INTEGER")
                    logger.debug("Added execution_time_ms column")

                if "amount_out_lamports" not in columns:
                    cursor.execute("ALTER TABLE trades ADD COLUMN amount_out_lamports INTEGER")
                    logger.debug("Added amount_out_lamports column")

                # Create indexes for commonly queried fields
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_tx_signature ON trades(tx_signature)"
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_slot ON trades(slot)")

                # Mark migration as applied
                cursor.execute("INSERT INTO schema_version (version) VALUES (2)")

                self._require_connection().commit()
                logger.info("Migration 2 complete: Execution detail columns added")

            # Migration 3: Add canonical EV label taxonomy column (2026-02-13)
            if current_version < 3:
                logger.info("Applying migration 3: EV label taxonomy")
                cursor.execute("PRAGMA table_info(training_outcomes)")
                columns = {row[1] for row in cursor.fetchall()}
                if "ev_label" not in columns:
                    cursor.execute("ALTER TABLE training_outcomes ADD COLUMN ev_label TEXT")
                    logger.debug("Added ev_label column")

                # Backfill ev_label where missing
                cursor.execute(
                    """
                    UPDATE training_outcomes
                    SET ev_label =
                        CASE
                            WHEN token_status LIKE '%rug%'
                                OR token_status LIKE '%honeypot%' THEN 'RUG'
                            WHEN token_status LIKE '%timeout%' THEN 'EXECUTION_TIMEOUT'
                            WHEN token_status LIKE '%mev%'
                                OR token_status LIKE '%sandwich%' THEN 'MEV_HIT'
                            WHEN token_status LIKE '%slippage%' THEN 'SLIPPAGE_FAIL'
                            WHEN COALESCE(pnl_pct, 0) >= 0 THEN 'GOOD_EV'
                            ELSE 'BAD_EV'
                        END
                    WHERE ev_label IS NULL
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_training_outcomes_ev_label "
                    "ON training_outcomes(ev_label)"
                )
                cursor.execute("INSERT INTO schema_version (version) VALUES (3)")
                self._require_connection().commit()
                logger.info("Migration 3 complete: EV labels added")

    async def close(self):
        """Close database connection."""
        async with self._lock:
            if self._connection:
                self._connection.close()
                self._connection = None
                logger.info("Database connection closed")

    async def get_trade_count(self, since: datetime | None = None) -> int:
        """Get count of trades, optionally since a given time."""
        async with self._lock:
            with self._cursor() as cursor:
                if since:
                    since_ms = int(since.timestamp() * 1000)
                    cursor.execute(
                        "SELECT COUNT(*) FROM trades WHERE timestamp_ms >= ?",
                        (since_ms,),
                    )
                else:
                    cursor.execute("SELECT COUNT(*) FROM trades")
                return cursor.fetchone()[0]

    async def get_recent_trades(self, hours: int = 24) -> pd.DataFrame:
        """Get recent trades as a DataFrame."""
        async with self._lock:
            since_ms = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM trades
                    WHERE timestamp_ms >= ?
                    ORDER BY timestamp_ms ASC
                    """,
                    (since_ms,),
                )
                rows = cursor.fetchall()

                if not rows:
                    return pd.DataFrame()

                columns = [desc[0] for desc in cursor.description]
                return pd.DataFrame([dict(row) for row in rows], columns=columns)

    async def insert_trade(self, trade: dict[str, Any]):
        """Insert a new trade record with complete fee and execution details."""
        async with self._lock:
            with self._cursor() as cursor:
                # Calculate computed fields
                base_fee = trade.get("base_fee_lamports", 0) or 0
                priority_fee = trade.get("priority_fee_lamports", 0) or 0
                jito_tip = trade.get("jito_tip_lamports", 0) or 0
                dex_fee = trade.get("dex_fee_lamports", 0) or 0
                total_fees_sol = (base_fee + priority_fee + jito_tip + dex_fee) / 1_000_000_000.0
                pnl_sol = trade.get("pnl_sol") or 0.0
                net_pnl_sol = pnl_sol - total_fees_sol

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO trades
                    (id, timestamp_ms, mint, symbol, side, amount_sol, amount_tokens,
                     price, slippage_bps, quoted_slippage_bps, pnl_sol, pnl_pct,
                     execution_mode, included, latency_ms,
                     base_fee_lamports, priority_fee_lamports, jito_tip_lamports, dex_fee_lamports,
                     total_fees_sol, net_pnl_sol,
                     tx_signature, slot, execution_time_ms, amount_out_lamports)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
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
                        trade.get("included", 1),
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
                    ),
                )

    async def get_fill_observation_count(self, since: datetime | None = None) -> int:
        """Get count of fill observations."""
        async with self._lock:
            with self._cursor() as cursor:
                if since:
                    since_ms = int(since.timestamp() * 1000)
                    cursor.execute(
                        "SELECT COUNT(*) FROM fill_observations WHERE timestamp_ms >= ?",
                        (since_ms,),
                    )
                else:
                    cursor.execute("SELECT COUNT(*) FROM fill_observations")
                return cursor.fetchone()[0]

    async def get_fill_observations(self, hours: int = 6) -> list[dict[str, Any]]:
        """Get recent fill observations."""
        async with self._lock:
            since_ms = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM fill_observations
                    WHERE timestamp_ms >= ?
                    ORDER BY timestamp_ms ASC
                    """,
                    (since_ms,),
                )
                return [dict(row) for row in cursor.fetchall()]

    async def insert_fill_observation(self, obs: dict[str, Any]):
        """Insert a fill observation."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO fill_observations
                    (timestamp_ms, mode, quoted_price, fill_price, quoted_slippage_bps,
                     realized_slippage_bps, latency_ms, included, mev_detected,
                     pool_address, liquidity_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        obs.get("timestamp_ms"),
                        obs.get("mode"),
                        obs.get("quoted_price"),
                        obs.get("fill_price"),
                        obs.get("quoted_slippage_bps"),
                        obs.get("realized_slippage_bps"),
                        obs.get("latency_ms"),
                        obs.get("included", 1),
                        obs.get("mev_detected", 0),
                        obs.get("pool_address"),
                        obs.get("liquidity_usd"),
                    ),
                )

    async def save_bandit_arm(
        self, arm_name: str, slippage_bps: int, pull_count: int, total_reward: float
    ):
        """Save bandit arm statistics."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO bandit_arms
                    (arm_name, slippage_bps, pull_count, total_reward, last_updated)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (arm_name, slippage_bps, pull_count, total_reward),
                )

    async def get_bandit_arms(self) -> dict[str, dict[str, Any]]:
        """Get all bandit arm statistics."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute("SELECT * FROM bandit_arms")
                return {row["arm_name"]: dict(row) for row in cursor.fetchall()}

    # ========== Scan Outcomes (for profitability learning) ==========

    async def insert_scan_outcome(self, outcome: dict[str, Any]) -> int:
        """Insert a scan outcome record.

        Returns the inserted row ID.
        """
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO scan_outcomes
                    (mint, pool, timestamp_ms, outcome_type, skip_reason,
                     features_json, profitability_score, confidence, expected_return_pct,
                     model_version, source, entry_price, exit_price, pnl_pct, pnl_sol,
                     hold_duration_ms, price_at_skip, price_1h_later, price_24h_later,
                     was_profitable_counterfactual)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        outcome.get("mint"),
                        outcome.get("pool"),
                        outcome.get("timestamp_ms"),
                        outcome.get("outcome_type"),
                        outcome.get("skip_reason"),
                        outcome.get("features_json", "{}"),
                        outcome.get("profitability_score"),
                        outcome.get("confidence"),
                        outcome.get("expected_return_pct"),
                        outcome.get("model_version", 0),
                        outcome.get("source", "scanner"),
                        outcome.get("entry_price"),
                        outcome.get("exit_price"),
                        outcome.get("pnl_pct"),
                        outcome.get("pnl_sol"),
                        outcome.get("hold_duration_ms"),
                        outcome.get("price_at_skip"),
                        outcome.get("price_1h_later"),
                        outcome.get("price_24h_later"),
                        outcome.get("was_profitable_counterfactual"),
                    ),
                )
                lastrowid = cursor.lastrowid
                return 0 if lastrowid is None else lastrowid

    async def get_scan_outcomes(
        self,
        since_hours: int = 168,  # 1 week default
        outcome_type: str | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Get scan outcomes for training.

        Args:
            since_hours: How many hours back to query
            outcome_type: Filter by outcome type (traded, skipped, rejected)
            limit: Maximum number of records to return
        """
        async with self._lock:
            since_ms = int((datetime.now() - timedelta(hours=since_hours)).timestamp() * 1000)
            with self._cursor() as cursor:
                if outcome_type:
                    cursor.execute(
                        """
                        SELECT * FROM scan_outcomes
                        WHERE timestamp_ms >= ? AND outcome_type = ?
                        ORDER BY timestamp_ms DESC
                        LIMIT ?
                        """,
                        (since_ms, outcome_type, limit),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * FROM scan_outcomes
                        WHERE timestamp_ms >= ?
                        ORDER BY timestamp_ms DESC
                        LIMIT ?
                        """,
                        (since_ms, limit),
                    )

                return [dict(row) for row in cursor.fetchall()]

    async def update_scan_outcome_counterfactual(
        self,
        mint: str,
        updates: dict[str, Any],
    ):
        """Update counterfactual data for a skipped token.

        Args:
            mint: Token mint address
            updates: Dictionary with fields to update (price_1h_later, price_24h_later, etc.)
        """
        async with self._lock:
            # Build dynamic update query
            set_clauses = []
            values = []

            for key, value in updates.items():
                if key in ("price_1h_later", "price_24h_later", "was_profitable_counterfactual"):
                    set_clauses.append(f"{key} = ?")
                    values.append(value)

            if not set_clauses:
                return

            values.append(mint)

            with self._cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE scan_outcomes
                    SET {", ".join(set_clauses)}
                    WHERE mint = ?
                    AND outcome_type = 'skipped'
                    AND price_1h_later IS NULL
                    """,
                    tuple(values),
                )

    async def get_scan_outcome_stats(self, hours: int = 24) -> dict[str, Any]:
        """Get statistics on scan outcomes."""
        async with self._lock:
            since_ms = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
            with self._cursor() as cursor:
                # Total counts by type
                cursor.execute(
                    """
                    SELECT outcome_type, COUNT(*) as count
                    FROM scan_outcomes
                    WHERE timestamp_ms >= ?
                    GROUP BY outcome_type
                    """,
                    (since_ms,),
                )
                type_counts = {row["outcome_type"]: row["count"] for row in cursor.fetchall()}

                # Win rate for traded
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins
                    FROM scan_outcomes
                    WHERE timestamp_ms >= ?
                    AND outcome_type = 'traded'
                    AND pnl_pct IS NOT NULL
                    """,
                    (since_ms,),
                )
                row = cursor.fetchone()
                traded_total = row["total"] if row else 0
                traded_wins = row["wins"] if row else 0

                # Counterfactual win rate
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN was_profitable_counterfactual = 1 THEN 1 ELSE 0 END) as wins
                    FROM scan_outcomes
                    WHERE timestamp_ms >= ?
                    AND outcome_type = 'skipped'
                    AND was_profitable_counterfactual IS NOT NULL
                    """,
                    (since_ms,),
                )
                row = cursor.fetchone()
                counterfactual_total = row["total"] if row else 0
                counterfactual_wins = row["wins"] if row else 0

                return {
                    "period_hours": hours,
                    "type_counts": type_counts,
                    "traded_total": traded_total,
                    "traded_wins": traded_wins,
                    "traded_win_rate": traded_wins / traded_total if traded_total > 0 else 0,
                    "counterfactual_total": counterfactual_total,
                    "counterfactual_wins": counterfactual_wins,
                    "counterfactual_win_rate": counterfactual_wins / counterfactual_total
                    if counterfactual_total > 0
                    else 0,
                }

    async def cleanup_old_scan_outcomes(self, days: int = 30):
        """Delete old scan outcomes beyond retention period."""
        async with self._lock:
            cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            with self._cursor() as cursor:
                cursor.execute(
                    "DELETE FROM scan_outcomes WHERE timestamp_ms < ?",
                    (cutoff_ms,),
                )
                deleted = cursor.rowcount
                logger.info(f"Cleaned up {deleted} old scan outcomes")
                return deleted

    # ========== Training Outcomes ==========

    async def insert_training_outcome(self, outcome: dict[str, Any]) -> str:
        """Insert a training outcome record."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO training_outcomes
                    (record_id, mint, decision_timestamp_ms, regime, features_json,
                     model_version, model_score, execution_mode, pnl_sol, pnl_pct,
                     token_status, ev_label, label_binary, label_return, label_quality)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        outcome.get("record_id"),
                        outcome.get("mint"),
                        outcome.get("decision_timestamp_ms"),
                        outcome.get("regime"),
                        outcome.get("features_json", "{}"),
                        outcome.get("model_version", 0),
                        outcome.get("model_score"),
                        outcome.get("execution_mode"),
                        outcome.get("pnl_sol"),
                        outcome.get("pnl_pct"),
                        outcome.get("token_status"),
                        canonicalize_ev_label(outcome),
                        outcome.get("label_binary"),
                        outcome.get("label_return"),
                        outcome.get("label_quality"),
                    ),
                )
                return outcome.get("record_id", "")

    async def get_training_outcomes(
        self,
        since_hours: int = 168,
        regime: str | None = None,
        min_quality: float | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Get training outcomes for model training."""
        async with self._lock:
            since_ms = int((datetime.now() - timedelta(hours=since_hours)).timestamp() * 1000)
            with self._cursor() as cursor:
                query = "SELECT * FROM training_outcomes WHERE decision_timestamp_ms >= ?"
                params: list[Any] = [since_ms]

                if regime:
                    query += " AND regime = ?"
                    params.append(regime)

                if min_quality is not None:
                    query += " AND (label_quality IS NULL OR label_quality >= ?)"
                    params.append(min_quality)

                query += " ORDER BY decision_timestamp_ms DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, tuple(params))
                return [dict(row) for row in cursor.fetchall()]

    async def get_hotpath_trade_outcomes(
        self,
        since_hours: int = 168,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Read persisted HotPath trade telemetry and normalize to training outcomes.

        Returns empty when the connected SQLite does not expose the HotPath trade schema.
        """
        async with self._lock:
            since_ms = int((datetime.now() - timedelta(hours=since_hours)).timestamp() * 1000)
            with self._cursor() as cursor:
                columns = {
                    row["name"] for row in cursor.execute("PRAGMA table_info(trades)").fetchall()
                }
                required = {
                    "trade_id",
                    "mode",
                    "direction",
                    "mint",
                    "quoted_slippage_bps",
                    "realized_slippage_bps",
                    "pnl_sol",
                    "pnl_pct",
                    "created_at",
                }
                if not required.issubset(columns):
                    return []

                cursor.execute(
                    """
                    SELECT
                        trade_id,
                        mint,
                        mode,
                        direction,
                        quoted_slippage_bps,
                        realized_slippage_bps,
                        total_fees_sol,
                        net_pnl_sol,
                        entry_price,
                        exit_price,
                        included,
                        pnl_sol,
                        pnl_pct,
                        CAST(strftime('%s', created_at) AS INTEGER) * 1000 AS decision_timestamp_ms
                    FROM trades
                    WHERE CAST(strftime('%s', created_at) AS INTEGER) * 1000 >= ?
                    ORDER BY decision_timestamp_ms DESC
                    LIMIT ?
                    """,
                    (since_ms, limit),
                )
                rows = cursor.fetchall()

                outcomes: list[dict[str, Any]] = []
                for row in rows:
                    pnl_pct = row["pnl_pct"]
                    pnl_sol = row["pnl_sol"]
                    if pnl_pct is None and isinstance(pnl_sol, (int, float)):
                        pnl_pct = float(pnl_sol)
                    label_binary = 1 if isinstance(pnl_pct, (int, float)) and pnl_pct >= 0 else 0
                    features = {
                        "mode": row["mode"],
                        "direction": row["direction"],
                        "quoted_slippage_bps": row["quoted_slippage_bps"],
                        "realized_slippage_bps": row["realized_slippage_bps"],
                        "total_fees_sol": row["total_fees_sol"],
                        "net_pnl_sol": row["net_pnl_sol"],
                        "entry_price": row["entry_price"],
                        "exit_price": row["exit_price"],
                        "included": row["included"],
                        "source": "hotpath_trades",
                    }
                    outcomes.append(
                        {
                            "record_id": row["trade_id"],
                            "mint": row["mint"],
                            "decision_timestamp_ms": row["decision_timestamp_ms"],
                            "regime": "unknown",
                            "features_json": json.dumps(features),
                            "execution_mode": str(row["mode"]).lower()
                            if row["mode"]
                            else "unknown",
                            "pnl_sol": pnl_sol,
                            "pnl_pct": pnl_pct,
                            "token_status": "filled" if row["included"] else "not_included",
                            "ev_label": EVLabel.GOOD_EV if label_binary == 1 else EVLabel.BAD_EV,
                            "label_binary": label_binary,
                            "label_return": pnl_pct if isinstance(pnl_pct, (int, float)) else 0.0,
                            "label_quality": 1.0,
                        }
                    )
                return outcomes

    # ========== Counterfactuals ==========

    async def insert_counterfactual(self, cf: dict[str, Any]) -> int:
        """Insert a counterfactual analysis record."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO counterfactuals
                    (base_record_id, counterfactual_type, cf_pnl_sol, cf_pnl_pct,
                     regret, analysis_timestamp_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cf.get("base_record_id"),
                        cf.get("counterfactual_type"),
                        cf.get("cf_pnl_sol"),
                        cf.get("cf_pnl_pct"),
                        cf.get("regret"),
                        cf.get("analysis_timestamp_ms"),
                    ),
                )
                lastrowid = cursor.lastrowid
                return 0 if lastrowid is None else lastrowid

    async def get_counterfactuals_for_record(self, record_id: str) -> list[dict[str, Any]]:
        """Get counterfactual analyses for a training outcome."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM counterfactuals WHERE base_record_id = ?",
                    (record_id,),
                )
                return [dict(row) for row in cursor.fetchall()]

    async def get_regret_statistics(self, hours: int = 24) -> dict[str, Any]:
        """Get regret statistics for counterfactual analysis."""
        async with self._lock:
            since_ms = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        counterfactual_type,
                        COUNT(*) as count,
                        AVG(regret) as avg_regret,
                        SUM(regret) as total_regret,
                        MAX(regret) as max_regret
                    FROM counterfactuals
                    WHERE analysis_timestamp_ms >= ?
                    GROUP BY counterfactual_type
                    """,
                    (since_ms,),
                )
                return {row["counterfactual_type"]: dict(row) for row in cursor.fetchall()}

    # ========== A/B Experiments ==========

    async def insert_ab_experiment(self, experiment: dict[str, Any]) -> str:
        """Insert a new A/B experiment."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO ab_experiments
                    (experiment_id, name, description, control_version, treatment_version,
                     traffic_split, status, start_timestamp_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        experiment.get("experiment_id"),
                        experiment.get("name"),
                        experiment.get("description"),
                        experiment.get("control_version"),
                        experiment.get("treatment_version"),
                        experiment.get("traffic_split", 0.5),
                        experiment.get("status", "active"),
                        experiment.get("start_timestamp_ms"),
                    ),
                )
                return experiment.get("experiment_id", "")

    async def update_ab_experiment(self, experiment_id: str, updates: dict[str, Any]):
        """Update an A/B experiment."""
        async with self._lock:
            allowed_fields = [
                "status",
                "winner",
                "end_timestamp_ms",
                "control_trades",
                "treatment_trades",
                "control_win_rate",
                "treatment_win_rate",
                "control_sharpe",
                "treatment_sharpe",
                "p_value",
            ]
            set_clauses = []
            values = []

            for key, value in updates.items():
                if key in allowed_fields:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)

            if not set_clauses:
                return

            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            values.append(experiment_id)

            with self._cursor() as cursor:
                cursor.execute(
                    f"UPDATE ab_experiments SET {', '.join(set_clauses)} WHERE experiment_id = ?",
                    tuple(values),
                )

    async def get_active_experiments(self) -> list[dict[str, Any]]:
        """Get all active A/B experiments."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute("SELECT * FROM ab_experiments WHERE status = 'active'")
                return [dict(row) for row in cursor.fetchall()]

    async def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Get an A/B experiment by ID."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM ab_experiments WHERE experiment_id = ?",
                    (experiment_id,),
                )
                row = cursor.fetchone()
                return dict(row) if row else None

    # ========== MI Scores History ==========

    async def insert_mi_scores(self, scores: list[dict[str, Any]]):
        """Insert mutual information scores for a calculation."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO mi_scores_history
                    (calculation_date, regime, feature_name, mi_score,
                     normalized_mi, redundancy_score, selection_rank)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            s.get("calculation_date"),
                            s.get("regime"),
                            s.get("feature_name"),
                            s.get("mi_score"),
                            s.get("normalized_mi"),
                            s.get("redundancy_score"),
                            s.get("selection_rank"),
                        )
                        for s in scores
                    ],
                )

    async def get_mi_scores(
        self,
        regime: str | None = None,
        calculation_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get MI scores, optionally filtered by regime and date."""
        async with self._lock:
            with self._cursor() as cursor:
                query = "SELECT * FROM mi_scores_history WHERE 1=1"
                params: list[Any] = []

                if regime:
                    query += " AND regime = ?"
                    params.append(regime)

                if calculation_date:
                    query += " AND calculation_date = ?"
                    params.append(calculation_date)

                query += " ORDER BY selection_rank ASC"
                cursor.execute(query, tuple(params))
                return [dict(row) for row in cursor.fetchall()]

    async def get_latest_mi_scores(self, regime: str) -> list[dict[str, Any]]:
        """Get the latest MI scores for a regime."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM mi_scores_history
                    WHERE regime = ?
                    AND calculation_date = (
                        SELECT MAX(calculation_date) FROM mi_scores_history WHERE regime = ?
                    )
                    ORDER BY selection_rank ASC
                    """,
                    (regime, regime),
                )
                return [dict(row) for row in cursor.fetchall()]

    # ========== Regime History ==========

    async def insert_regime_detection(self, detection: dict[str, Any]) -> int:
        """Insert a regime detection record."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO regime_history
                    (timestamp_ms, detected_regime, regime_probabilities_json,
                     features_json, confidence)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        detection.get("timestamp_ms"),
                        detection.get("detected_regime"),
                        detection.get("regime_probabilities_json"),
                        detection.get("features_json"),
                        detection.get("confidence"),
                    ),
                )
                lastrowid = cursor.lastrowid
                return 0 if lastrowid is None else lastrowid

    async def get_regime_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get regime detection history."""
        async with self._lock:
            since_ms = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM regime_history
                    WHERE timestamp_ms >= ?
                    ORDER BY timestamp_ms DESC
                    """,
                    (since_ms,),
                )
                return [dict(row) for row in cursor.fetchall()]

    # ========== Monte Carlo Results ==========

    async def insert_monte_carlo_result(self, result: dict[str, Any]) -> int:
        """Insert a Monte Carlo simulation result."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO monte_carlo_results
                    (simulation_id, strategy_name, num_paths, sharpe_mean, sharpe_std,
                     sharpe_ci_lower, sharpe_ci_upper, max_drawdown_mean, max_drawdown_std,
                     var_95, cvar_95, risk_of_ruin, final_capital_mean, final_capital_std,
                     config_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.get("simulation_id"),
                        result.get("strategy_name"),
                        result.get("num_paths"),
                        result.get("sharpe_mean"),
                        result.get("sharpe_std"),
                        result.get("sharpe_ci_lower"),
                        result.get("sharpe_ci_upper"),
                        result.get("max_drawdown_mean"),
                        result.get("max_drawdown_std"),
                        result.get("var_95"),
                        result.get("cvar_95"),
                        result.get("risk_of_ruin"),
                        result.get("final_capital_mean"),
                        result.get("final_capital_std"),
                        result.get("config_json"),
                    ),
                )
                lastrowid = cursor.lastrowid
                return 0 if lastrowid is None else lastrowid

    async def get_monte_carlo_results(
        self,
        strategy_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get Monte Carlo simulation results."""
        async with self._lock:
            with self._cursor() as cursor:
                if strategy_name:
                    cursor.execute(
                        """
                        SELECT * FROM monte_carlo_results
                        WHERE strategy_name = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (strategy_name, limit),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * FROM monte_carlo_results
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )
                return [dict(row) for row in cursor.fetchall()]

    # ========== Trading Signals ==========

    async def insert_trading_signal(self, signal: dict[str, Any]) -> str:
        """Insert a new trading signal."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO trading_signals
                    (signal_id, timestamp_ms, token_mint, token_symbol, pool_address,
                     action, position_size_sol, slippage_bps, expected_price,
                     confidence, fraud_score, status, expires_at_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal.get("signal_id"),
                        signal.get("timestamp_ms"),
                        signal.get("token_mint"),
                        signal.get("token_symbol"),
                        signal.get("pool_address"),
                        signal.get("action"),
                        signal.get("position_size_sol"),
                        signal.get("slippage_bps"),
                        signal.get("expected_price"),
                        signal.get("confidence"),
                        signal.get("fraud_score"),
                        signal.get("status", "pending"),
                        signal.get("expires_at_ms"),
                    ),
                )
                return signal.get("signal_id", "")

    async def update_trading_signal(self, signal_id: str, updates: dict[str, Any]):
        """Update a trading signal status and outcome."""
        async with self._lock:
            allowed_fields = [
                "status",
                "plan_id",
                "submitted_at_ms",
                "completed_at_ms",
                "pnl_sol",
                "pnl_pct",
                "error_message",
            ]
            set_clauses = []
            values = []

            for key, value in updates.items():
                if key in allowed_fields:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)

            if not set_clauses:
                return

            values.append(signal_id)

            with self._cursor() as cursor:
                cursor.execute(
                    f"UPDATE trading_signals SET {', '.join(set_clauses)} WHERE signal_id = ?",
                    tuple(values),
                )

    async def get_trading_signal(self, signal_id: str) -> dict[str, Any] | None:
        """Get a trading signal by ID."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM trading_signals WHERE signal_id = ?",
                    (signal_id,),
                )
                row = cursor.fetchone()
                return dict(row) if row else None

    async def get_pending_trading_signals(self) -> list[dict[str, Any]]:
        """Get all pending trading signals."""
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM trading_signals
                    WHERE status = 'pending'
                    ORDER BY timestamp_ms ASC
                    """
                )
                return [dict(row) for row in cursor.fetchall()]

    async def get_trading_signals(
        self,
        hours: int = 24,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get recent trading signals."""
        async with self._lock:
            since_ms = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
            with self._cursor() as cursor:
                query = "SELECT * FROM trading_signals WHERE timestamp_ms >= ?"
                params: list[Any] = [since_ms]

                if status:
                    query += " AND status = ?"
                    params.append(status)

                query += " ORDER BY timestamp_ms DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, tuple(params))
                return [dict(row) for row in cursor.fetchall()]

    async def get_trading_signal_stats(self, hours: int = 24) -> dict[str, Any]:
        """Get trading signal statistics."""
        async with self._lock:
            since_ms = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
            with self._cursor() as cursor:
                # Status counts
                cursor.execute(
                    """
                    SELECT status, COUNT(*) as count
                    FROM trading_signals
                    WHERE timestamp_ms >= ?
                    GROUP BY status
                    """,
                    (since_ms,),
                )
                status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

                # PnL statistics for completed signals
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN pnl_sol > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(pnl_sol) as total_pnl_sol,
                        AVG(pnl_sol) as avg_pnl_sol,
                        SUM(position_size_sol) as total_volume_sol
                    FROM trading_signals
                    WHERE timestamp_ms >= ?
                    AND status = 'completed'
                    AND pnl_sol IS NOT NULL
                    """,
                    (since_ms,),
                )
                row = cursor.fetchone()
                total = row["total"] if row else 0
                wins = row["wins"] if row else 0

                return {
                    "period_hours": hours,
                    "status_counts": status_counts,
                    "completed_total": total,
                    "completed_wins": wins,
                    "win_rate": wins / total if total > 0 else 0,
                    "total_pnl_sol": row["total_pnl_sol"] if row else 0,
                    "avg_pnl_sol": row["avg_pnl_sol"] if row else 0,
                    "total_volume_sol": row["total_volume_sol"] if row else 0,
                }

    # ==================== Feature Cache Methods ====================

    async def save_cached_features(
        self,
        outcome_id: int,
        features: np.ndarray,
        version: int = 1,
        feature_hash: str | None = None,
    ) -> bool:
        """Save computed features to cache for faster ML training.

        Args:
            outcome_id: ID of the scan_outcome record
            features: NumPy array of computed features
            version: Feature version (bump when feature engineering changes)
            feature_hash: Optional hash of the feature computation for validation

        Returns:
            True if saved successfully, False otherwise
        """
        async with self._lock:
            try:
                with self._cursor() as cursor:
                    features_binary = features.tobytes()
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO feature_cache
                        (outcome_id, feature_version, features_binary, feature_hash, cached_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        (outcome_id, version, features_binary, feature_hash),
                    )
                return True
            except Exception as e:
                logger.error(f"Failed to cache features for outcome {outcome_id}: {e}")
                return False

    async def save_cached_features_batch(
        self,
        features_batch: list[tuple],  # List of (outcome_id, features_array)
        version: int = 1,
        feature_hash: str | None = None,
    ) -> int:
        """Batch save computed features for faster bulk caching.

        Args:
            features_batch: List of (outcome_id, np.ndarray) tuples
            version: Feature version
            feature_hash: Optional hash for validation

        Returns:
            Number of successfully cached entries
        """
        async with self._lock:
            count = 0
            try:
                with self._cursor() as cursor:
                    cursor.execute("BEGIN TRANSACTION")
                    for outcome_id, features in features_batch:
                        try:
                            features_binary = features.tobytes()
                            cursor.execute(
                                """
                                INSERT OR REPLACE INTO feature_cache
                                (outcome_id, feature_version, features_binary,
                                 feature_hash, cached_at)
                                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                                """,
                                (outcome_id, version, features_binary, feature_hash),
                            )
                            count += 1
                        except Exception as e:
                            logger.warning(f"Failed to cache features for {outcome_id}: {e}")
                    cursor.execute("COMMIT")
            except Exception as e:
                logger.error(f"Batch feature caching failed: {e}")
                try:
                    with self._cursor() as cursor:
                        cursor.execute("ROLLBACK")
                except Exception as rollback_error:
                    # ROLLBACK failure is critical but shouldn't mask the original error
                    logger.warning(f"ROLLBACK also failed: {rollback_error}")
            return count

    async def get_cached_features(
        self,
        outcome_ids: list[int],
        version: int = 1,
        feature_dim: int = 50,
    ) -> dict[int, np.ndarray]:
        """Retrieve cached features for multiple outcome IDs.

        Args:
            outcome_ids: List of scan_outcome IDs to retrieve
            version: Feature version to match
            feature_dim: Expected feature dimension for array reconstruction

        Returns:
            Dict mapping outcome_id to feature array (only includes hits)
        """
        if not outcome_ids:
            return {}

        async with self._lock:
            result = {}
            try:
                with self._cursor() as cursor:
                    # Query in batches to avoid large IN clauses
                    batch_size = 500
                    for i in range(0, len(outcome_ids), batch_size):
                        batch = outcome_ids[i : i + batch_size]
                        placeholders = ",".join("?" * len(batch))
                        cursor.execute(
                            f"""
                            SELECT outcome_id, features_binary
                            FROM feature_cache
                            WHERE outcome_id IN ({placeholders})
                            AND feature_version = ?
                            """,
                            (*batch, version),
                        )
                        for row in cursor.fetchall():
                            try:
                                features = np.frombuffer(
                                    row["features_binary"], dtype=np.float64
                                ).reshape(-1)
                                if len(features) == feature_dim:
                                    result[row["outcome_id"]] = features
                                else:
                                    logger.warning(
                                        f"Feature dimension mismatch for {row['outcome_id']}: "
                                        f"expected {feature_dim}, got {len(features)}"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to deserialize features for {row['outcome_id']}: {e}"
                                )
            except Exception as e:
                logger.error(f"Failed to retrieve cached features: {e}")
            return result

    async def get_feature_cache_stats(self, version: int = 1) -> dict[str, Any]:
        """Get statistics about the feature cache.

        Returns:
            Dict with cache statistics including hit rate potential
        """
        async with self._lock:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as cached_count,
                        feature_version,
                        MIN(cached_at) as oldest_cache,
                        MAX(cached_at) as newest_cache
                    FROM feature_cache
                    WHERE feature_version = ?
                    GROUP BY feature_version
                    """,
                    (version,),
                )
                row = cursor.fetchone()

                # Also get total outcomes count
                cursor.execute("SELECT COUNT(*) as total FROM scan_outcomes")
                total_row = cursor.fetchone()
                total_outcomes = total_row["total"] if total_row else 0

                cached_count = row["cached_count"] if row else 0
                return {
                    "version": version,
                    "cached_count": cached_count,
                    "total_outcomes": total_outcomes,
                    "cache_coverage": cached_count / total_outcomes if total_outcomes > 0 else 0,
                    "oldest_cache": row["oldest_cache"] if row else None,
                    "newest_cache": row["newest_cache"] if row else None,
                }

    async def invalidate_feature_cache(self, version: int | None = None) -> int:
        """Invalidate (delete) cached features.

        Args:
            version: If specified, only invalidate this version.
                    If None, invalidate ALL cached features.

        Returns:
            Number of cache entries deleted
        """
        async with self._lock:
            try:
                with self._cursor() as cursor:
                    if version is not None:
                        cursor.execute(
                            "DELETE FROM feature_cache WHERE feature_version = ?",
                            (version,),
                        )
                    else:
                        cursor.execute("DELETE FROM feature_cache")
                    deleted = cursor.rowcount
                    logger.info(
                        f"Invalidated {deleted} feature cache entries"
                        + (f" for version {version}" if version else "")
                    )
                    return deleted
            except Exception as e:
                logger.error(f"Failed to invalidate feature cache: {e}")
                return 0


class ModelArtifactStore:
    """Manages model artifact persistence to disk.

    Stores versioned model parameters as JSON files.
    """

    def __init__(self, base_dir: str, max_versions: int = 10):
        self.base_dir = Path(base_dir)
        self.max_versions = max_versions
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_dir(self, model_name: str) -> Path:
        """Get directory for a specific model."""
        model_dir = self.base_dir / model_name
        model_dir.mkdir(exist_ok=True)
        return model_dir

    def _get_versions(self, model_name: str) -> list[int]:
        """Get sorted list of available versions."""
        model_dir = self._get_model_dir(model_name)
        versions = []
        for f in model_dir.glob("v*.json"):
            try:
                version = int(f.stem[1:])  # Remove 'v' prefix
                versions.append(version)
            except ValueError:
                continue
        return sorted(versions)

    def save(self, model_name: str, state: dict[str, Any]) -> int:
        """Save model state and return version number."""
        model_dir = self._get_model_dir(model_name)
        versions = self._get_versions(model_name)

        # Determine next version
        next_version = (versions[-1] + 1) if versions else 1

        # Save new version
        filepath = model_dir / f"v{next_version}.json"
        with open(filepath, "w") as f:
            json.dump(
                {
                    "version": next_version,
                    "timestamp": datetime.now().isoformat(),
                    "state": state,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(f"Saved {model_name} model version {next_version}")

        # Cleanup old versions
        self._cleanup_old_versions(model_name)

        return next_version

    def _cleanup_old_versions(self, model_name: str):
        """Remove old versions beyond max_versions."""
        versions = self._get_versions(model_name)
        if len(versions) > self.max_versions:
            model_dir = self._get_model_dir(model_name)
            for version in versions[: -self.max_versions]:
                filepath = model_dir / f"v{version}.json"
                filepath.unlink(missing_ok=True)
                logger.debug(f"Removed old model version: {model_name} v{version}")

    def load_latest(self, model_name: str) -> dict[str, Any] | None:
        """Load the latest version of a model."""
        versions = self._get_versions(model_name)
        if not versions:
            return None

        latest_version = versions[-1]
        return self.load(model_name, latest_version)

    def load(self, model_name: str, version: int) -> dict[str, Any] | None:
        """Load a specific version of a model."""
        model_dir = self._get_model_dir(model_name)
        filepath = model_dir / f"v{version}.json"

        if not filepath.exists():
            return None

        with open(filepath) as f:
            data = json.load(f)
            return data.get("state")

    def list_versions(self, model_name: str) -> list[dict[str, Any]]:
        """List all available versions with metadata."""
        model_dir = self._get_model_dir(model_name)
        versions = self._get_versions(model_name)
        result = []

        for version in versions:
            filepath = model_dir / f"v{version}.json"
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    result.append(
                        {
                            "version": version,
                            "timestamp": data.get("timestamp"),
                            "filepath": str(filepath),
                        }
                    )
            except Exception as e:
                logger.warning(f"Error reading version {version}: {e}")

        return result
