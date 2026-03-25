"""
Abstract storage backend interface.

Defines the contract for storage implementations (SQLite, PostgreSQL, etc.)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .storage import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Trade record data class."""

    id: str
    timestamp_ms: int
    mint: str
    symbol: str | None
    side: str
    amount_sol: float
    amount_tokens: float
    price: float
    slippage_bps: int | None
    quoted_slippage_bps: int | None
    pnl_sol: float | None
    pnl_pct: float | None
    execution_mode: str | None
    included: bool
    latency_ms: int | None


@dataclass
class FillObservation:
    """Fill observation for calibration."""

    timestamp_ms: int
    mode: str
    quoted_price: float
    fill_price: float | None
    quoted_slippage_bps: int
    realized_slippage_bps: int | None
    latency_ms: int | None
    included: bool
    mev_detected: bool
    pool_address: str | None
    liquidity_usd: float | None


@dataclass
class BanditArm:
    """Bandit arm statistics."""

    arm_name: str
    slippage_bps: int
    pull_count: int
    total_reward: float
    last_updated: datetime | None


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def connect(self) -> None:
        """Initialize the connection."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the connection is healthy."""
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend name."""
        pass

    # Trade operations
    @abstractmethod
    async def get_trade_count(self, since: datetime | None = None) -> int:
        """Get count of trades, optionally since a given time."""
        pass

    @abstractmethod
    async def get_recent_trades(self, hours: int = 24) -> pd.DataFrame:
        """Get recent trades as a DataFrame."""
        pass

    @abstractmethod
    async def insert_trade(self, trade: dict[str, Any]) -> None:
        """Insert a new trade record."""
        pass

    # Fill observation operations
    @abstractmethod
    async def get_fill_observation_count(self, since: datetime | None = None) -> int:
        """Get count of fill observations."""
        pass

    @abstractmethod
    async def get_fill_observations(self, hours: int = 6) -> list[dict[str, Any]]:
        """Get recent fill observations."""
        pass

    @abstractmethod
    async def insert_fill_observation(self, obs: dict[str, Any]) -> None:
        """Insert a fill observation."""
        pass

    # Bandit arm operations
    @abstractmethod
    async def save_bandit_arm(
        self, arm_name: str, slippage_bps: int, pull_count: int, total_reward: float
    ) -> None:
        """Save bandit arm statistics."""
        pass

    @abstractmethod
    async def get_bandit_arms(self) -> dict[str, dict[str, Any]]:
        """Get all bandit arm statistics."""
        pass


class StorageBackendFactory:
    """Factory for creating storage backends."""

    @staticmethod
    async def create(backend_type: str, **kwargs) -> StorageBackend:
        """Create a storage backend based on type.

        Args:
            backend_type: Either "sqlite" or "postgres"
            **kwargs: Backend-specific configuration
                For sqlite: db_path (str)
                For postgres: database_url (str)

        Returns:
            Configured storage backend
        """
        if backend_type == "sqlite":
            db_path = kwargs.get("db_path", "coldpath.db")
            backend = SqliteStorageBackend(db_path)
            await backend.connect()
            return backend
        elif backend_type in ("postgres", "postgresql"):
            from .storage_postgres import PostgresStorageBackend

            database_url = kwargs.get("database_url")
            if not database_url:
                raise ValueError("database_url required for PostgreSQL backend")
            backend = PostgresStorageBackend(database_url)
            await backend.connect()
            return backend
        else:
            raise ValueError(f"Unknown storage backend: {backend_type}")


class SqliteStorageBackend(StorageBackend):
    """SQLite storage backend wrapper.

    Wraps the existing DatabaseManager for backwards compatibility.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._manager: DatabaseManager | None = None

    async def connect(self) -> None:
        from .storage import DatabaseManager

        self._manager = DatabaseManager(self.db_path)
        await self._manager.connect()
        logger.info(f"SQLite backend connected: {self.db_path}")

    async def close(self) -> None:
        if self._manager:
            await self._manager.close()
            self._manager = None

    async def health_check(self) -> bool:
        if not self._manager or not self._manager._connection:
            return False
        try:
            cursor = self._manager._connection.cursor()
            cursor.execute("SELECT 1")
            return True
        except Exception:
            return False

    @property
    def backend_name(self) -> str:
        return "sqlite"

    async def get_trade_count(self, since: datetime | None = None) -> int:
        return await self._manager.get_trade_count(since)

    async def get_recent_trades(self, hours: int = 24) -> pd.DataFrame:
        return await self._manager.get_recent_trades(hours)

    async def insert_trade(self, trade: dict[str, Any]) -> None:
        await self._manager.insert_trade(trade)

    async def get_fill_observation_count(self, since: datetime | None = None) -> int:
        return await self._manager.get_fill_observation_count(since)

    async def get_fill_observations(self, hours: int = 6) -> list[dict[str, Any]]:
        return await self._manager.get_fill_observations(hours)

    async def insert_fill_observation(self, obs: dict[str, Any]) -> None:
        await self._manager.insert_fill_observation(obs)

    async def save_bandit_arm(
        self, arm_name: str, slippage_bps: int, pull_count: int, total_reward: float
    ) -> None:
        await self._manager.save_bandit_arm(arm_name, slippage_bps, pull_count, total_reward)

    async def get_bandit_arms(self) -> dict[str, dict[str, Any]]:
        return await self._manager.get_bandit_arms()
