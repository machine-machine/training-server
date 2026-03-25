"""
Verified Settings Store - Save and load proven configurations.

Stores configurations that have demonstrated:
- Win rate >= 50%
- Daily P&L >= 0.6%
- Max drawdown <= 25%
- Minimum 50 trades (statistical significance)
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VerifiedSettings:
    """A verified configuration with performance metrics."""

    settings_id: str
    settings: dict[str, Any]

    # Performance metrics (required for verification)
    win_rate_pct: float
    daily_pnl_pct: float
    max_drawdown_pct: float
    total_trades: int
    sharpe_ratio: float
    profit_factor: float

    # Metadata
    verified_at: str
    verified_by: str  # "backtest", "live", "paper"
    backtest_period: str | None = None

    # Additional metrics
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    fill_rate_pct: float = 0.0

    # Deployment status
    is_active: bool = False
    deployed_at: str | None = None
    deployment_count: int = 0

    # Performance notes
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def meets_criteria(
        self,
        min_win_rate: float = 50.0,
        min_daily_pnl: float = 0.6,
        max_drawdown: float = 25.0,
        min_trades: int = 50,
        min_sharpe: float = 1.5,
    ) -> bool:
        """Check if settings meet minimum criteria for verification."""
        return (
            self.win_rate_pct >= min_win_rate
            and self.daily_pnl_pct >= min_daily_pnl
            and self.max_drawdown_pct <= max_drawdown
            and self.total_trades >= min_trades
            and self.sharpe_ratio >= min_sharpe
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerifiedSettings":
        """Create from dictionary."""
        return cls(**data)


class VerifiedSettingsStore:
    """Persistent storage for verified configurations.

    Features:
    - Save configurations that meet performance criteria
    - Load best performing configuration
    - Activate configurations for live trading
    - Track deployment history

    Usage:
        store = VerifiedSettingsStore()

        # Save a verified configuration
        settings_id = store.save_verified(
            settings={"stop_loss": 8.0, "take_profit": 20.0},
            metrics={"win_rate_pct": 55, "daily_pnl_pct": 0.9, ...},
            verified_by="backtest",
        )

        # Get best configuration
        best = store.get_best()

        # Activate for live trading
        store.set_active(best.settings_id)
    """

    DEFAULT_STORE_PATH = Path("configs/verified_settings.json")

    # Minimum criteria for saving settings
    MIN_WIN_RATE = 50.0
    MIN_DAILY_PNL = 0.6
    MAX_DRAWDOWN = 25.0
    MIN_TRADES = 50
    MIN_SHARPE = 1.5

    def __init__(self, store_path: Path | None = None):
        """Initialize the settings store.

        Args:
            store_path: Path to the settings JSON file.
                       Defaults to configs/verified_settings.json
        """
        self.store_path = store_path or self.DEFAULT_STORE_PATH
        self._ensure_store_exists()

    def _ensure_store_exists(self):
        """Create store file if it doesn't exist."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self._save_store(
                {
                    "settings": [],
                    "active_id": None,
                    "metadata": {
                        "created_at": datetime.utcnow().isoformat(),
                        "version": 1,
                    },
                }
            )

    def _load_store(self) -> dict:
        """Load the store from disk."""
        try:
            with open(self.store_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load settings store: {e}, creating new")
            self._ensure_store_exists()
            return {"settings": [], "active_id": None, "metadata": {}}

    def _save_store(self, data: dict):
        """Save the store to disk."""
        with open(self.store_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _generate_id(self, settings: dict) -> str:
        """Generate deterministic ID from settings hash."""
        # Sort keys for consistent hashing
        content = json.dumps(settings, sort_keys=True, default=str)
        hash_digest = hashlib.md5(content.encode()).hexdigest()
        return f"cfg_{hash_digest[:12]}"

    def save_verified(
        self,
        settings: dict[str, Any],
        metrics: dict[str, float],
        verified_by: str = "backtest",
        backtest_period: str | None = None,
        notes: str = "",
        tags: list[str] | None = None,
    ) -> str | None:
        """Save a verified configuration.

        Args:
            settings: The configuration parameters
            metrics: Performance metrics from backtest/live
            verified_by: Source of verification (backtest, live, paper)
            backtest_period: Time period of backtest if applicable
            notes: Optional notes about this configuration
            tags: Optional tags for categorization

        Returns:
            settings_id if saved successfully, None if criteria not met
        """
        settings_id = self._generate_id(settings)

        verified = VerifiedSettings(
            settings_id=settings_id,
            settings=settings.copy(),
            win_rate_pct=float(metrics.get("win_rate_pct", 0)),
            daily_pnl_pct=float(metrics.get("daily_pnl_pct", 0)),
            max_drawdown_pct=float(metrics.get("max_drawdown_pct", 100)),
            total_trades=int(metrics.get("total_trades", 0)),
            sharpe_ratio=float(metrics.get("sharpe_ratio", 0)),
            profit_factor=float(metrics.get("profit_factor", 0)),
            avg_win_pct=float(metrics.get("avg_win_pct", 0)),
            avg_loss_pct=float(metrics.get("avg_loss_pct", 0)),
            fill_rate_pct=float(metrics.get("fill_rate_pct", 0)),
            verified_at=datetime.utcnow().isoformat(),
            verified_by=verified_by,
            backtest_period=backtest_period,
            notes=notes,
            tags=tags or [],
        )

        # Check if meets minimum criteria
        if not verified.meets_criteria(
            min_win_rate=self.MIN_WIN_RATE,
            min_daily_pnl=self.MIN_DAILY_PNL,
            max_drawdown=self.MAX_DRAWDOWN,
            min_trades=self.MIN_TRADES,
            min_sharpe=self.MIN_SHARPE,
        ):
            logger.info(
                f"Settings {settings_id} do not meet criteria: "
                f"win_rate={verified.win_rate_pct:.1f}% "
                f"(min {self.MIN_WIN_RATE}%), "
                f"daily_pnl={verified.daily_pnl_pct:.2f}% "
                f"(min {self.MIN_DAILY_PNL}%), "
                f"max_dd={verified.max_drawdown_pct:.1f}% "
                f"(max {self.MAX_DRAWDOWN}%), "
                f"trades={verified.total_trades} "
                f"(min {self.MIN_TRADES}), "
                f"sharpe={verified.sharpe_ratio:.2f} "
                f"(min {self.MIN_SHARPE})"
            )
            return None

        store = self._load_store()

        # Check if this exact config already exists
        existing_idx = next(
            (i for i, s in enumerate(store["settings"]) if s["settings_id"] == settings_id), None
        )

        if existing_idx is not None:
            # Update existing, preserve deployment info
            existing = store["settings"][existing_idx]
            verified.deployment_count = existing.get("deployment_count", 0)
            verified.deployed_at = existing.get("deployed_at")
            verified.is_active = existing.get("is_active", False)
            store["settings"][existing_idx] = verified.to_dict()
            logger.info(f"Updated existing verified settings: {settings_id}")
        else:
            store["settings"].append(verified.to_dict())
            logger.info(
                f"Saved new verified settings: {settings_id} "
                f"(daily_pnl={verified.daily_pnl_pct:.2f}%, "
                f"win_rate={verified.win_rate_pct:.1f}%)"
            )

        # Sort by daily P&L descending (primary metric)
        store["settings"].sort(key=lambda x: x.get("daily_pnl_pct", 0), reverse=True)

        # Keep top 100 configurations
        if len(store["settings"]) > 100:
            removed = store["settings"][100:]
            store["settings"] = store["settings"][:100]
            logger.info(f"Pruned {len(removed)} older configurations")

        self._save_store(store)
        return settings_id

    def get_best(self) -> VerifiedSettings | None:
        """Get the best performing verified configuration."""
        store = self._load_store()
        if not store.get("settings"):
            return None

        best = store["settings"][0]
        return VerifiedSettings.from_dict(best)

    def get_active(self) -> VerifiedSettings | None:
        """Get the currently active configuration for live trading."""
        store = self._load_store()
        if not store.get("active_id"):
            return None

        for s in store["settings"]:
            if s["settings_id"] == store["active_id"]:
                return VerifiedSettings.from_dict(s)
        return None

    def set_active(self, settings_id: str) -> bool:
        """Set a configuration as active for live trading.

        Args:
            settings_id: The ID of the configuration to activate

        Returns:
            True if successfully activated, False if not found
        """
        store = self._load_store()

        # Find the settings
        found = any(s["settings_id"] == settings_id for s in store["settings"])
        if not found:
            logger.warning(f"Settings not found: {settings_id}")
            return False

        # Update active status
        now = datetime.utcnow().isoformat()
        for s in store["settings"]:
            if s["settings_id"] == settings_id:
                s["is_active"] = True
                s["deployed_at"] = now
                s["deployment_count"] = s.get("deployment_count", 0) + 1
            else:
                s["is_active"] = False

        store["active_id"] = settings_id
        self._save_store(store)

        logger.info(f"Activated settings: {settings_id}")
        return True

    def deactivate(self) -> bool:
        """Deactivate the current active configuration."""
        store = self._load_store()

        if not store.get("active_id"):
            return True

        for s in store["settings"]:
            s["is_active"] = False

        store["active_id"] = None
        self._save_store(store)

        logger.info("Deactivated current settings")
        return True

    def get_by_id(self, settings_id: str) -> VerifiedSettings | None:
        """Get a specific configuration by ID."""
        store = self._load_store()

        for s in store["settings"]:
            if s["settings_id"] == settings_id:
                return VerifiedSettings.from_dict(s)
        return None

    def list_all(
        self,
        limit: int = 20,
        tag: str | None = None,
        min_pnl: float | None = None,
    ) -> list[VerifiedSettings]:
        """List all verified configurations.

        Args:
            limit: Maximum number to return
            tag: Filter by tag
            min_pnl: Filter by minimum daily P&L

        Returns:
            List of VerifiedSettings objects
        """
        store = self._load_store()
        results = []

        for s in store["settings"]:
            # Apply filters
            if tag and tag not in s.get("tags", []):
                continue
            if min_pnl and s.get("daily_pnl_pct", 0) < min_pnl:
                continue

            results.append(VerifiedSettings.from_dict(s))

        return results[:limit]

    def delete(self, settings_id: str) -> bool:
        """Delete a configuration.

        Note: Cannot delete the currently active configuration.
        """
        store = self._load_store()

        # Can't delete active
        if store.get("active_id") == settings_id:
            logger.warning(f"Cannot delete active configuration: {settings_id}")
            return False

        # Find and remove
        original_len = len(store["settings"])
        store["settings"] = [s for s in store["settings"] if s["settings_id"] != settings_id]

        if len(store["settings"]) == original_len:
            return False

        self._save_store(store)
        logger.info(f"Deleted settings: {settings_id}")
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about stored configurations."""
        store = self._load_store()
        settings = store.get("settings", [])

        if not settings:
            return {
                "total_count": 0,
                "active_id": None,
                "best_daily_pnl_pct": None,
                "avg_daily_pnl_pct": None,
            }

        pnls = [s.get("daily_pnl_pct", 0) for s in settings]

        return {
            "total_count": len(settings),
            "active_id": store.get("active_id"),
            "best_daily_pnl_pct": max(pnls) if pnls else None,
            "avg_daily_pnl_pct": sum(pnls) / len(pnls) if pnls else None,
            "min_criteria": {
                "min_win_rate": self.MIN_WIN_RATE,
                "min_daily_pnl": self.MIN_DAILY_PNL,
                "max_drawdown": self.MAX_DRAWDOWN,
                "min_trades": self.MIN_TRADES,
                "min_sharpe": self.MIN_SHARPE,
            },
        }
