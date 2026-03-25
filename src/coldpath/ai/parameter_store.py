"""
Versioned parameter store for strategy configuration.

Provides atomic version updates, history retention, rollback capability,
and performance baseline tracking. Thread-safe for concurrent access.

Usage:
    store = ParameterStore()
    await store.initialize(initial_params)

    current = await store.get_current()
    await store.deploy(new_params, deployed_by="auto", reason="optimization")
    await store.rollback(reason="performance degradation")
"""

import asyncio
import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default strategy parameters matching UserStrategyParams
DEFAULT_STRATEGY_PARAMS: dict[str, Any] = {
    "min_liquidity_usd": 10000.0,
    "max_fdv_usd": 5000000.0,
    "min_holders": 50,
    "min_confidence": 0.6,
    "max_risk_score": 0.45,
    "take_profit_pct": 25.0,
    "stop_loss_pct": 8.0,
    "trailing_stop_pct": None,
    "trailing_activation_pct": 10.0,
    "max_hold_minutes": 30,
    "sizing_mode": "fixed",
    "max_position_sol": 0.1,
    "pct_of_wallet": 0.05,
    "max_daily_loss_sol": 0.5,
    "max_drawdown_pct": 20.0,
    "max_concurrent_positions": 3,
}


@dataclass
class PerformanceBaseline:
    """Performance baseline captured at deployment time."""

    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    capture_timestamp: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "avg_pnl_pct": self.avg_pnl_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades,
            "capture_timestamp": (
                self.capture_timestamp.isoformat() if self.capture_timestamp else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceBaseline":
        """Deserialize from dictionary."""
        ts = data.get("capture_timestamp")
        return cls(
            sharpe_ratio=data.get("sharpe_ratio", 0.0),
            win_rate=data.get("win_rate", 0.0),
            avg_pnl_pct=data.get("avg_pnl_pct", 0.0),
            max_drawdown_pct=data.get("max_drawdown_pct", 0.0),
            total_trades=data.get("total_trades", 0),
            capture_timestamp=(datetime.fromisoformat(ts) if ts else None),
        )


@dataclass
class VersionedStrategyParams:
    """Parameters with version tracking and full audit trail."""

    version: int
    timestamp: datetime
    params: dict[str, Any]
    deployed_by: str  # "auto", "manual", "rollback"
    deployment_reason: str
    optimization_id: str | None = None
    performance_baseline: PerformanceBaseline | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "params": copy.deepcopy(self.params),
            "deployed_by": self.deployed_by,
            "deployment_reason": self.deployment_reason,
            "optimization_id": self.optimization_id,
            "performance_baseline": (
                self.performance_baseline.to_dict() if self.performance_baseline else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VersionedStrategyParams":
        """Deserialize from dictionary."""
        baseline_data = data.get("performance_baseline")
        return cls(
            version=data["version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            params=data["params"],
            deployed_by=data["deployed_by"],
            deployment_reason=data["deployment_reason"],
            optimization_id=data.get("optimization_id"),
            performance_baseline=(
                PerformanceBaseline.from_dict(baseline_data) if baseline_data else None
            ),
        )


class ParameterStore:
    """
    Thread-safe parameter versioning and deployment.

    Features:
    - Atomic version updates with lock protection
    - History retention (last N versions, configurable)
    - Instant rollback capability
    - Performance baseline tracking per version
    - Persistence to disk (JSON)
    - Diff computation between versions

    Concurrency safety:
    - All mutations are protected by an asyncio.Lock
    - Reads return deep copies to prevent aliasing
    """

    def __init__(
        self,
        max_history: int = 10,
        persist_path: str | None = None,
    ) -> None:
        """
        Initialize the parameter store.

        Args:
            max_history: Maximum number of historical versions to retain.
            persist_path: Optional file path for JSON persistence.
        """
        self.max_history = max_history
        self.persist_path = Path(persist_path) if persist_path else None

        self._current: VersionedStrategyParams | None = None
        self._history: list[VersionedStrategyParams] = []
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(
        self,
        initial_params: dict[str, Any] | None = None,
    ) -> VersionedStrategyParams:
        """
        Initialize the store with parameters.

        If a persist file exists, loads from disk. Otherwise uses the
        provided initial_params or DEFAULT_STRATEGY_PARAMS.

        Args:
            initial_params: Initial parameter values. Uses defaults if None.

        Returns:
            The initial VersionedStrategyParams (version 0).
        """
        async with self._lock:
            # Try loading from disk first
            if self.persist_path and self.persist_path.exists():
                try:
                    await self._load_from_disk()
                    self._initialized = True
                    logger.info(
                        "Parameter store loaded from disk: version %d, %d history entries",
                        self._current.version if self._current else -1,
                        len(self._history),
                    )
                    return self._current  # type: ignore[return-value]
                except Exception as exc:
                    logger.warning(
                        "Failed to load parameter store from %s: %s",
                        self.persist_path,
                        exc,
                    )

            params = copy.deepcopy(initial_params if initial_params else DEFAULT_STRATEGY_PARAMS)
            self._current = VersionedStrategyParams(
                version=0,
                timestamp=datetime.now(),
                params=params,
                deployed_by="manual",
                deployment_reason="initial",
            )
            self._history = []
            self._initialized = True

            await self._persist()
            logger.info("Parameter store initialized with version 0")
            return self._current

    def _check_initialized(self) -> None:
        """Raise if the store has not been initialized."""
        if not self._initialized or self._current is None:
            raise RuntimeError("ParameterStore not initialized. Call initialize() first.")

    async def get_current(self) -> dict[str, Any]:
        """
        Get the current parameter values (deep copy).

        Returns:
            A deep copy of the current parameter dictionary.
        """
        self._check_initialized()
        assert self._current is not None
        return copy.deepcopy(self._current.params)

    async def get_current_version(self) -> VersionedStrategyParams:
        """
        Get the full current versioned parameters.

        Returns:
            The current VersionedStrategyParams (attributes are safe to read,
            but .params is a deep copy).
        """
        self._check_initialized()
        assert self._current is not None
        # Return a copy so callers cannot mutate internal state
        return VersionedStrategyParams(
            version=self._current.version,
            timestamp=self._current.timestamp,
            params=copy.deepcopy(self._current.params),
            deployed_by=self._current.deployed_by,
            deployment_reason=self._current.deployment_reason,
            optimization_id=self._current.optimization_id,
            performance_baseline=self._current.performance_baseline,
        )

    async def get_version(self, version: int) -> VersionedStrategyParams | None:
        """
        Get a specific historical version.

        Args:
            version: The version number to retrieve.

        Returns:
            The VersionedStrategyParams if found, else None.
        """
        self._check_initialized()
        assert self._current is not None

        if self._current.version == version:
            return await self.get_current_version()

        for entry in self._history:
            if entry.version == version:
                return VersionedStrategyParams(
                    version=entry.version,
                    timestamp=entry.timestamp,
                    params=copy.deepcopy(entry.params),
                    deployed_by=entry.deployed_by,
                    deployment_reason=entry.deployment_reason,
                    optimization_id=entry.optimization_id,
                    performance_baseline=entry.performance_baseline,
                )
        return None

    async def deploy(
        self,
        new_params: dict[str, Any],
        deployed_by: str,
        reason: str,
        optimization_id: str | None = None,
        performance_baseline: PerformanceBaseline | None = None,
    ) -> VersionedStrategyParams:
        """
        Deploy a new set of parameters atomically.

        Pushes the current version into history, creates a new version,
        trims history if it exceeds max_history, and persists to disk.

        Args:
            new_params: The new parameter dictionary.
            deployed_by: Who deployed ("auto", "manual", "rollback").
            reason: Human-readable reason for the deployment.
            optimization_id: Associated optimization run ID.
            performance_baseline: Performance snapshot at deployment time.

        Returns:
            The newly created VersionedStrategyParams.
        """
        async with self._lock:
            self._check_initialized()
            assert self._current is not None

            # Push current to history
            self._history.append(self._current)

            # Trim history
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history :]

            # Create new version
            new_version = self._current.version + 1
            self._current = VersionedStrategyParams(
                version=new_version,
                timestamp=datetime.now(),
                params=copy.deepcopy(new_params),
                deployed_by=deployed_by,
                deployment_reason=reason,
                optimization_id=optimization_id,
                performance_baseline=performance_baseline,
            )

            await self._persist()

            logger.info(
                "Deployed parameter version %d by %s: %s",
                new_version,
                deployed_by,
                reason,
            )
            return self._current

    async def rollback(
        self,
        reason: str,
        target_version: int | None = None,
    ) -> VersionedStrategyParams:
        """
        Rollback to a previous parameter version.

        If target_version is None, rolls back to the most recent
        historical version.

        Args:
            reason: Why the rollback is happening.
            target_version: Specific version to rollback to (optional).

        Returns:
            The new current VersionedStrategyParams after rollback.

        Raises:
            RuntimeError: If no history available or target version not found.
        """
        async with self._lock:
            self._check_initialized()
            assert self._current is not None

            if not self._history:
                raise RuntimeError("No history available for rollback")

            if target_version is not None:
                target = None
                for entry in self._history:
                    if entry.version == target_version:
                        target = entry
                        break
                if target is None:
                    raise RuntimeError(f"Version {target_version} not found in history")
            else:
                target = self._history[-1]

            # Push current to history before rollback
            self._history.append(self._current)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history :]

            # Create new version with rolled-back params
            new_version = self._current.version + 1
            self._current = VersionedStrategyParams(
                version=new_version,
                timestamp=datetime.now(),
                params=copy.deepcopy(target.params),
                deployed_by="rollback",
                deployment_reason=f"Rollback to v{target.version}: {reason}",
                optimization_id=None,
                performance_baseline=target.performance_baseline,
            )

            await self._persist()

            logger.info(
                "Rolled back to version %d (now v%d): %s",
                target.version,
                new_version,
                reason,
            )
            return self._current

    async def get_history(self) -> list[VersionedStrategyParams]:
        """
        Get the full version history (most recent last).

        Returns:
            List of historical VersionedStrategyParams (deep copies).
        """
        self._check_initialized()
        result = []
        for entry in self._history:
            result.append(
                VersionedStrategyParams(
                    version=entry.version,
                    timestamp=entry.timestamp,
                    params=copy.deepcopy(entry.params),
                    deployed_by=entry.deployed_by,
                    deployment_reason=entry.deployment_reason,
                    optimization_id=entry.optimization_id,
                    performance_baseline=entry.performance_baseline,
                )
            )
        return result

    def compute_diff(
        self,
        old_params: dict[str, Any],
        new_params: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """
        Compute parameter differences between two versions.

        Returns:
            Dict mapping parameter name to {"old": value, "new": value, "change_pct": float}.
            Only parameters that changed are included.
        """
        diff: dict[str, dict[str, Any]] = {}
        all_keys = set(old_params.keys()) | set(new_params.keys())

        for key in all_keys:
            old_val = old_params.get(key)
            new_val = new_params.get(key)

            if old_val != new_val:
                entry: dict[str, Any] = {"old": old_val, "new": new_val}

                # Compute percentage change for numeric values
                if (
                    isinstance(old_val, (int, float))
                    and isinstance(new_val, (int, float))
                    and old_val != 0
                ):
                    entry["change_pct"] = ((new_val - old_val) / abs(old_val)) * 100

                diff[key] = entry

        return diff

    @property
    def current_version_number(self) -> int:
        """Current version number (-1 if not initialized)."""
        if self._current is None:
            return -1
        return self._current.version

    @property
    def history_count(self) -> int:
        """Number of versions in history."""
        return len(self._history)

    # ---- Persistence ----

    async def _persist(self) -> None:
        """Persist current state to disk (if persist_path configured)."""
        if self.persist_path is None:
            return

        try:
            data = {
                "current": (self._current.to_dict() if self._current else None),
                "history": [entry.to_dict() for entry in self._history],
            }

            # Write atomically via temp file
            tmp_path = self.persist_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data, indent=2, default=str))
            tmp_path.rename(self.persist_path)
        except Exception as exc:
            logger.error("Failed to persist parameter store: %s", exc)

    async def _load_from_disk(self) -> None:
        """Load state from disk."""
        assert self.persist_path is not None
        raw = self.persist_path.read_text()
        data = json.loads(raw)

        if data.get("current"):
            self._current = VersionedStrategyParams.from_dict(data["current"])
        if data.get("history"):
            self._history = [VersionedStrategyParams.from_dict(entry) for entry in data["history"]]
