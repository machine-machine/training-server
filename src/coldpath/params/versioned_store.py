"""
Versioned Parameter Store

Prevents race conditions when parameters change during trade execution.
Ensures atomic deployment and provides rollback capability.

Problem solved:
    - Trade enters Sizing stage with params V1
    - Orchestrator deploys params V2 mid-execution
    - Trade executes with mixed V1/V2 parameters -> INCONSISTENT!

Solution:
    - Trade gets a snapshot of params at version N
    - New deployments create version N+1
    - Trade continues with version N snapshot
    - Outcome recorded with version N for attribution

Usage:
    store = ParameterStore()

    # Deploy new parameters
    version = await store.deploy_new_version(
        params={"stop_loss": 8.0, "take_profit": 25.0},
        deployed_by="auto",
        reason="Drift detected: Sharpe degraded 30%"
    )

    # Get current parameters (snapshot for trade execution)
    current = await store.get_current()

    # Rollback if needed
    await store.rollback(versions_back=1)
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VersionedStrategyParams:
    """Parameters with version tracking and audit trail."""

    version: int
    """Unique version number (monotonically increasing)."""

    timestamp: datetime
    """When this version was deployed."""

    params: dict[str, Any]
    """The actual parameter values."""

    deployed_by: str
    """Who deployed this: "auto", "manual", or "rollback"."""

    reason: str | None = None
    """Why this version was deployed."""

    backtest_confidence: float | None = None
    """Backtest confidence score (if available)."""

    backtest_sharpe_improvement: float | None = None
    """Expected Sharpe improvement (if available)."""

    previous_version: int | None = None
    """Which version this replaced (for rollback context)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "params": self.params,
            "deployed_by": self.deployed_by,
            "reason": self.reason,
            "backtest_confidence": self.backtest_confidence,
            "backtest_sharpe_improvement": self.backtest_sharpe_improvement,
            "previous_version": self.previous_version,
        }


@dataclass
class ParameterStoreStats:
    """Statistics for the parameter store."""

    total_deployments: int = 0
    total_rollbacks: int = 0
    total_auto_deployments: int = 0
    total_manual_deployments: int = 0
    current_version: int = 0
    history_size: int = 0


class ParameterStore:
    """
    Atomic parameter management with versioning.

    Thread-safe parameter storage that:
    - Prevents race conditions during deployment
    - Maintains rollback history (last N versions)
    - Tracks who/why/when for each change
    - Provides parameter snapshots for trade execution

    Attributes:
        current_version: The currently active version number.
        max_history: Maximum number of versions to keep for rollback.

    Example:
        store = ParameterStore(max_history=10)

        # Initial deployment
        v1 = await store.deploy_new_version(
            {"stop_loss": 10.0, "take_profit": 30.0},
            deployed_by="manual",
            reason="Initial setup"
        )

        # Auto-optimization
        v2 = await store.deploy_new_version(
            {"stop_loss": 8.0, "take_profit": 25.0},
            deployed_by="auto",
            reason="Sharpe improvement: +0.3",
            backtest_confidence=0.85
        )

        # Rollback if needed
        await store.rollback(versions_back=1)  # Returns to v1 params
    """

    def __init__(self, max_history: int = 10):
        """
        Initialize the parameter store.

        Args:
            max_history: Maximum number of versions to keep for rollback.
        """
        self.max_history = max_history
        self.current_version: int = 0
        self._params: dict[int, VersionedStrategyParams] = {}
        self._lock = asyncio.Lock()
        self._stats = ParameterStoreStats()

    @property
    def version_count(self) -> int:
        """Number of versions in history."""
        return len(self._params)

    def get_stats(self) -> ParameterStoreStats:
        """Get current statistics."""
        return ParameterStoreStats(
            total_deployments=self._stats.total_deployments,
            total_rollbacks=self._stats.total_rollbacks,
            total_auto_deployments=self._stats.total_auto_deployments,
            total_manual_deployments=self._stats.total_manual_deployments,
            current_version=self.current_version,
            history_size=len(self._params),
        )

    async def deploy_new_version(
        self,
        params: dict[str, Any],
        deployed_by: str,
        reason: str | None = None,
        backtest_confidence: float | None = None,
        backtest_sharpe_improvement: float | None = None,
    ) -> int:
        """
        Deploy new parameter version atomically.

        This is an atomic operation protected by an async lock.
        The new version becomes the current version immediately.

        Args:
            params: The new parameter values.
            deployed_by: Who is deploying ("auto", "manual", "rollback").
            reason: Why this deployment is happening.
            backtest_confidence: Confidence score from backtest validation.
            backtest_sharpe_improvement: Expected Sharpe improvement.

        Returns:
            The new version number.

        Example:
            version = await store.deploy_new_version(
                params={"stop_loss": 8.0},
                deployed_by="auto",
                reason="Optimization triggered by drift",
                backtest_confidence=0.82
            )
        """
        async with self._lock:
            previous_version = self.current_version if self.current_version > 0 else None

            self.current_version += 1

            versioned = VersionedStrategyParams(
                version=self.current_version,
                timestamp=datetime.now(),
                params=params.copy(),  # Snapshot, not reference
                deployed_by=deployed_by,
                reason=reason,
                backtest_confidence=backtest_confidence,
                backtest_sharpe_improvement=backtest_sharpe_improvement,
                previous_version=previous_version,
            )

            self._params[self.current_version] = versioned

            # Prune old versions to maintain max_history
            self._prune_history()

            # Update stats
            self._stats.total_deployments += 1
            self._stats.current_version = self.current_version
            self._stats.history_size = len(self._params)

            if deployed_by == "auto":
                self._stats.total_auto_deployments += 1
            elif deployed_by == "manual":
                self._stats.total_manual_deployments += 1

            logger.info(
                "Deployed parameter version %d (by=%s, reason=%s)",
                self.current_version,
                deployed_by,
                reason or "N/A",
            )

            return self.current_version

    async def get_version(self, version: int) -> VersionedStrategyParams | None:
        """
        Get a specific parameter version.

        Args:
            version: The version number to retrieve.

        Returns:
            The versioned parameters, or None if not found.
        """
        return self._params.get(version)

    async def get_current(self) -> VersionedStrategyParams | None:
        """
        Get the current active version.

        Returns:
            The current versioned parameters, or None if no versions exist.
        """
        if self.current_version == 0:
            return None
        return self._params.get(self.current_version)

    async def get_current_params(self) -> dict[str, Any]:
        """
        Get current parameters as a dictionary.

        This returns a copy, safe for use as a snapshot during trade execution.

        Returns:
            Copy of current parameters, or empty dict if none exist.
        """
        current = await self.get_current()
        return current.params.copy() if current else {}

    async def rollback(self, versions_back: int = 1) -> int | None:
        """
        Rollback to a previous version.

        This creates a NEW version with the old parameters (for audit trail).

        Args:
            versions_back: How many versions to go back (1 = previous).

        Returns:
            The new version number, or None if rollback not possible.

        Example:
            # Current is v5, rollback to v4
            new_version = await store.rollback(1)  # Creates v6 with v4's params
        """
        async with self._lock:
            target_version = self.current_version - versions_back

            if target_version <= 0:
                logger.error(
                    "Cannot rollback: target version %d is invalid",
                    target_version,
                )
                return None

            if target_version not in self._params:
                logger.error(
                    "Cannot rollback: version %d not in history (may have been pruned)",
                    target_version,
                )
                return None

            old_params = self._params[target_version]

            # Create new version with old params (for audit trail)
            self.current_version += 1

            rolled_back = VersionedStrategyParams(
                version=self.current_version,
                timestamp=datetime.now(),
                params=old_params.params.copy(),
                deployed_by="rollback",
                reason=f"Rolled back from v{self.current_version - 1} to v{target_version}",
                previous_version=self.current_version - 1,
            )

            self._params[self.current_version] = rolled_back

            # Update stats
            self._stats.total_rollbacks += 1
            self._stats.total_deployments += 1
            self._stats.current_version = self.current_version

            logger.warning(
                "Rollback: v%d -> v%d (new version: v%d)",
                self.current_version - 1,
                target_version,
                self.current_version,
            )

            return self.current_version

    async def get_history(self, limit: int = 10) -> list[VersionedStrategyParams]:
        """
        Get recent parameter history.

        Args:
            limit: Maximum number of versions to return.

        Returns:
            List of versioned parameters, most recent first.
        """
        versions = sorted(self._params.keys(), reverse=True)[:limit]
        return [self._params[v] for v in versions]

    async def compare_versions(
        self,
        version1: int,
        version2: int,
    ) -> dict[str, dict[str, Any]]:
        """
        Compare two parameter versions.

        Args:
            version1: First version to compare.
            version2: Second version to compare.

        Returns:
            Dictionary with 'changed', 'added', 'removed' keys.
        """
        v1 = self._params.get(version1)
        v2 = self._params.get(version2)

        if not v1 or not v2:
            return {"error": "One or both versions not found"}

        p1, p2 = v1.params, v2.params

        changed = {}
        for key in set(p1.keys()) & set(p2.keys()):
            if p1[key] != p2[key]:
                changed[key] = {"from": p1[key], "to": p2[key]}

        added = {k: p2[k] for k in set(p2.keys()) - set(p1.keys())}
        removed = {k: p1[k] for k in set(p1.keys()) - set(p2.keys())}

        return {
            "version1": version1,
            "version2": version2,
            "changed": changed,
            "added": added,
            "removed": removed,
        }

    def _prune_history(self) -> None:
        """Remove oldest versions if we exceed max_history."""
        while len(self._params) > self.max_history:
            oldest = min(self._params.keys())
            del self._params[oldest]
            logger.debug("Pruned parameter version %d from history", oldest)

    def to_dict(self) -> dict[str, Any]:
        """Serialize store state to dictionary."""
        return {
            "current_version": self.current_version,
            "max_history": self.max_history,
            "versions": {v: p.to_dict() for v, p in self._params.items()},
            "stats": {
                "total_deployments": self._stats.total_deployments,
                "total_rollbacks": self._stats.total_rollbacks,
                "total_auto_deployments": self._stats.total_auto_deployments,
                "total_manual_deployments": self._stats.total_manual_deployments,
            },
        }


# Singleton instance
_param_store: ParameterStore | None = None


async def get_param_store() -> ParameterStore:
    """
    Get or create the global parameter store.

    Returns:
        The singleton ParameterStore instance.
    """
    global _param_store
    if _param_store is None:
        _param_store = ParameterStore()
    return _param_store


def reset_param_store() -> None:
    """Reset the global parameter store (for testing)."""
    global _param_store
    _param_store = None
