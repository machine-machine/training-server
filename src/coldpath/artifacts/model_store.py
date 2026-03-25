"""
Model Artifact Store - Version control for ML model artifacts.

Implements version control and deployment:
model_artifacts/
├── v001/
│   ├── weights.json
│   ├── normalization.json
│   ├── calibration.json
│   ├── regime_weights/
│   └── checksum.sha256
└── current -> v001

Features:
- A/B testing: Traffic routing by mint hash (80/20 split)
- Rollback triggers: Win rate <45%, 5+ losses, >15% drawdown/hour
- Hot-reload: Atomic swap via RwLock, clear score cache
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ArtifactStatus(Enum):
    """Status of a model artifact."""

    DRAFT = "draft"  # Being created
    STAGED = "staged"  # Ready for deployment
    ACTIVE = "active"  # Currently serving traffic
    SHADOW = "shadow"  # Shadow mode (logging only)
    DEPRECATED = "deprecated"  # Old version
    ROLLED_BACK = "rolled_back"


class RollbackTrigger(Enum):
    """Reasons for automatic rollback."""

    LOW_WIN_RATE = "low_win_rate"  # Win rate < 45%
    CONSECUTIVE_LOSSES = "consecutive_losses"  # 5+ losses
    HIGH_DRAWDOWN = "high_drawdown"  # >15% drawdown/hour
    ERROR_RATE = "error_rate"  # High inference errors
    MANUAL = "manual"  # Manual rollback


@dataclass
class RollbackConfig:
    """Configuration for automatic rollback."""

    min_win_rate: float = 0.45
    max_consecutive_losses: int = 5
    max_drawdown_per_hour: float = 0.15
    max_error_rate: float = 0.10
    min_trades_before_eval: int = 10
    cooldown_after_rollback_seconds: int = 3600


@dataclass
class ModelMetrics:
    """Performance metrics for a model version."""

    version: int
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl_total: float = 0.0
    max_drawdown: float = 0.0
    consecutive_losses: int = 0
    errors: int = 0
    last_updated: datetime | None = None

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def error_rate(self) -> float:
        total = self.trades + self.errors
        return self.errors / total if total > 0 else 0.0

    def should_rollback(self, config: RollbackConfig) -> RollbackTrigger | None:
        """Check if model should be rolled back."""
        if self.trades < config.min_trades_before_eval:
            return None

        if self.win_rate < config.min_win_rate:
            return RollbackTrigger.LOW_WIN_RATE

        if self.consecutive_losses >= config.max_consecutive_losses:
            return RollbackTrigger.CONSECUTIVE_LOSSES

        if self.max_drawdown > config.max_drawdown_per_hour:
            return RollbackTrigger.HIGH_DRAWDOWN

        if self.error_rate > config.max_error_rate:
            return RollbackTrigger.ERROR_RATE

        return None


@dataclass
class ModelArtifact:
    """A versioned model artifact."""

    version: int
    status: ArtifactStatus
    created_at: datetime
    weights_path: str
    normalization_path: str
    calibration_path: str | None
    checksum: str
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: ModelMetrics | None = None

    # Loaded data (in memory)
    weights: dict[str, Any] | None = None
    normalization: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "weights_path": self.weights_path,
            "normalization_path": self.normalization_path,
            "calibration_path": self.calibration_path,
            "checksum": self.checksum,
            "metadata": self.metadata,
            "metrics": {
                "trades": self.metrics.trades,
                "win_rate": self.metrics.win_rate,
                "pnl_total": self.metrics.pnl_total,
            }
            if self.metrics
            else None,
        }


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""

    control_version: int
    treatment_version: int
    treatment_traffic_pct: float = 0.20  # 20% treatment
    enabled: bool = True
    sticky_routing: bool = True  # Same mint always gets same version


class ModelStore:
    """Version-controlled model artifact storage.

    Manages model versions with:
    - Atomic deploys using symlink swaps
    - A/B testing with deterministic routing
    - Automatic rollback on poor performance
    - Hot-reload without service restart

    Example:
        store = ModelStore("/path/to/artifacts")
        await store.initialize()
        artifact = await store.create_artifact(weights, normalization)
        await store.activate(artifact.version)
    """

    def __init__(
        self,
        base_path: str,
        rollback_config: RollbackConfig | None = None,
    ):
        """Initialize model store.

        Args:
            base_path: Base directory for artifacts
            rollback_config: Automatic rollback configuration
        """
        self.base_path = Path(base_path)
        self.rollback_config = rollback_config or RollbackConfig()

        # Current state
        self._current_version: int | None = None
        self._artifacts: dict[int, ModelArtifact] = {}
        self._ab_test: ABTestConfig | None = None

        # Concurrency control
        self._lock = asyncio.Lock()
        self._reload_callbacks: list[Callable[[int], None]] = []

        # Performance tracking
        self._version_metrics: dict[int, ModelMetrics] = {}
        self._last_rollback_time: datetime | None = None

    async def initialize(self):
        """Initialize store and load existing artifacts."""
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Load existing versions
        for path in self.base_path.iterdir():
            if path.is_dir() and path.name.startswith("v"):
                try:
                    version = int(path.name[1:])
                    await self._load_artifact(version)
                except (ValueError, Exception) as e:
                    logger.warning(f"Failed to load artifact {path}: {e}")

        # Determine current version from symlink
        current_link = self.base_path / "current"
        if current_link.is_symlink():
            target = current_link.resolve()
            try:
                self._current_version = int(target.name[1:])
            except ValueError:
                pass

        logger.info(
            f"Model store initialized: {len(self._artifacts)} versions, "
            f"current={self._current_version}"
        )

    async def create_artifact(
        self,
        weights: dict[str, Any],
        normalization: dict[str, Any],
        calibration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelArtifact:
        """Create a new model artifact.

        Args:
            weights: Model weights dictionary
            normalization: Normalization parameters
            calibration: Optional calibration data
            metadata: Optional metadata

        Returns:
            Created ModelArtifact
        """
        async with self._lock:
            # Determine next version
            existing = list(self._artifacts.keys())
            next_version = max(existing) + 1 if existing else 1

            # Create version directory
            version_dir = self.base_path / f"v{next_version:03d}"
            version_dir.mkdir(parents=True, exist_ok=True)

            # Save weights
            weights_path = version_dir / "weights.json"
            with open(weights_path, "w") as f:
                json.dump(weights, f, indent=2)

            # Save normalization
            norm_path = version_dir / "normalization.json"
            with open(norm_path, "w") as f:
                json.dump(normalization, f, indent=2)

            # Save calibration
            cal_path = None
            if calibration:
                cal_path = version_dir / "calibration.json"
                with open(cal_path, "w") as f:
                    json.dump(calibration, f, indent=2)

            # Compute checksum
            checksum = self._compute_checksum(version_dir)
            with open(version_dir / "checksum.sha256", "w") as f:
                f.write(checksum)

            # Create artifact
            artifact = ModelArtifact(
                version=next_version,
                status=ArtifactStatus.STAGED,
                created_at=datetime.now(),
                weights_path=str(weights_path),
                normalization_path=str(norm_path),
                calibration_path=str(cal_path) if cal_path else None,
                checksum=checksum,
                metadata=metadata or {},
                weights=weights,
                normalization=normalization,
                calibration=calibration,
            )

            self._artifacts[next_version] = artifact
            self._version_metrics[next_version] = ModelMetrics(version=next_version)

            logger.info(f"Created artifact v{next_version}")
            return artifact

    async def activate(self, version: int, clear_caches: bool = True):
        """Activate a model version.

        Args:
            version: Version to activate
            clear_caches: Whether to clear inference caches
        """
        async with self._lock:
            if version not in self._artifacts:
                raise ValueError(f"Version {version} not found")

            artifact = self._artifacts[version]

            # Load if not loaded
            if artifact.weights is None:
                await self._load_artifact(version)

            # Update symlink atomically
            current_link = self.base_path / "current"
            temp_link = self.base_path / "current.tmp"
            version_dir = self.base_path / f"v{version:03d}"

            # Create temp link and rename (atomic on POSIX)
            if temp_link.exists():
                temp_link.unlink()
            temp_link.symlink_to(version_dir)
            temp_link.rename(current_link)

            # Update status
            if self._current_version and self._current_version in self._artifacts:
                self._artifacts[self._current_version].status = ArtifactStatus.DEPRECATED

            artifact.status = ArtifactStatus.ACTIVE
            self._current_version = version

            # Notify reload callbacks
            for callback in self._reload_callbacks:
                try:
                    callback(version)
                except Exception as e:
                    logger.error(f"Reload callback failed: {e}")

            logger.info(f"Activated version {version}")

    async def rollback(
        self,
        to_version: int | None = None,
        reason: RollbackTrigger = RollbackTrigger.MANUAL,
    ):
        """Rollback to a previous version.

        Args:
            to_version: Version to rollback to (default: previous active)
            reason: Reason for rollback
        """
        async with self._lock:
            if to_version is None:
                # Find previous active version
                candidates = [
                    v
                    for v, a in self._artifacts.items()
                    if v < self._current_version
                    and a.status in [ArtifactStatus.DEPRECATED, ArtifactStatus.STAGED]
                ]
                if not candidates:
                    raise ValueError("No previous version to rollback to")
                to_version = max(candidates)

            if to_version == self._current_version:
                return

            logger.warning(
                f"Rolling back from v{self._current_version} to v{to_version}: {reason.value}"
            )

            # Mark current as rolled back
            if self._current_version in self._artifacts:
                self._artifacts[self._current_version].status = ArtifactStatus.ROLLED_BACK

            # Activate previous
            await self.activate(to_version)

            self._last_rollback_time = datetime.now()

    async def record_trade_result(
        self,
        version: int,
        won: bool,
        pnl: float,
    ):
        """Record a trade result for a version.

        Args:
            version: Model version used
            won: Whether trade was profitable
            pnl: PnL of the trade
        """
        if version not in self._version_metrics:
            self._version_metrics[version] = ModelMetrics(version=version)

        metrics = self._version_metrics[version]
        metrics.trades += 1
        metrics.pnl_total += pnl

        if won:
            metrics.wins += 1
            metrics.consecutive_losses = 0
        else:
            metrics.losses += 1
            metrics.consecutive_losses += 1

        metrics.max_drawdown = min(metrics.max_drawdown, pnl)
        metrics.last_updated = datetime.now()

        # Check rollback triggers
        trigger = metrics.should_rollback(self.rollback_config)
        if trigger and version == self._current_version:
            # Check cooldown
            if self._last_rollback_time:
                elapsed = (datetime.now() - self._last_rollback_time).total_seconds()
                if elapsed < self.rollback_config.cooldown_after_rollback_seconds:
                    logger.warning(f"Rollback triggered but in cooldown ({elapsed:.0f}s)")
                    return

            await self.rollback(reason=trigger)

    def get_version_for_mint(self, mint: str) -> int:
        """Get version for a mint (A/B test routing).

        Args:
            mint: Token mint address

        Returns:
            Version number to use
        """
        if not self._ab_test or not self._ab_test.enabled:
            return self._current_version or 1

        if self._ab_test.sticky_routing:
            # Deterministic routing by mint hash
            mint_hash = int(hashlib.md5(mint.encode()).hexdigest()[:8], 16)
            bucket = mint_hash % 100

            if bucket < self._ab_test.treatment_traffic_pct * 100:
                return self._ab_test.treatment_version
            else:
                return self._ab_test.control_version
        else:
            # Random routing
            import random

            if random.random() < self._ab_test.treatment_traffic_pct:
                return self._ab_test.treatment_version
            else:
                return self._ab_test.control_version

    async def start_ab_test(
        self,
        treatment_version: int,
        traffic_pct: float = 0.20,
    ):
        """Start A/B test with a new version.

        Args:
            treatment_version: Version to test
            traffic_pct: Percentage of traffic for treatment
        """
        if treatment_version not in self._artifacts:
            raise ValueError(f"Version {treatment_version} not found")

        self._ab_test = ABTestConfig(
            control_version=self._current_version or 1,
            treatment_version=treatment_version,
            treatment_traffic_pct=traffic_pct,
            enabled=True,
        )

        self._artifacts[treatment_version].status = ArtifactStatus.SHADOW

        logger.info(
            f"Started A/B test: control=v{self._ab_test.control_version}, "
            f"treatment=v{treatment_version} @ {traffic_pct * 100:.0f}%"
        )

    async def stop_ab_test(self, promote_treatment: bool = False):
        """Stop A/B test.

        Args:
            promote_treatment: Whether to promote treatment to active
        """
        if not self._ab_test:
            return

        treatment = self._ab_test.treatment_version

        if promote_treatment:
            await self.activate(treatment)
        else:
            self._artifacts[treatment].status = ArtifactStatus.STAGED

        self._ab_test = None
        logger.info(f"Stopped A/B test, promoted={promote_treatment}")

    def register_reload_callback(self, callback: Callable[[int], None]):
        """Register callback for version changes."""
        self._reload_callbacks.append(callback)

    def get_current_artifact(self) -> ModelArtifact | None:
        """Get currently active artifact."""
        if self._current_version and self._current_version in self._artifacts:
            return self._artifacts[self._current_version]
        return None

    def get_artifact(self, version: int) -> ModelArtifact | None:
        """Get artifact by version."""
        return self._artifacts.get(version)

    def list_versions(self) -> list[dict[str, Any]]:
        """List all versions with metadata."""
        return [
            a.to_dict()
            for a in sorted(
                self._artifacts.values(),
                key=lambda a: a.version,
                reverse=True,
            )
        ]

    async def _load_artifact(self, version: int):
        """Load artifact data from disk."""
        version_dir = self.base_path / f"v{version:03d}"

        weights_path = version_dir / "weights.json"
        norm_path = version_dir / "normalization.json"
        cal_path = version_dir / "calibration.json"
        checksum_path = version_dir / "checksum.sha256"

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found for v{version}")

        with open(weights_path) as f:
            weights = json.load(f)

        with open(norm_path) as f:
            normalization = json.load(f)

        calibration = None
        if cal_path.exists():
            with open(cal_path) as f:
                calibration = json.load(f)

        checksum = ""
        if checksum_path.exists():
            with open(checksum_path) as f:
                checksum = f.read().strip()

        # Determine status
        current_link = self.base_path / "current"
        if current_link.is_symlink() and current_link.resolve() == version_dir:
            status = ArtifactStatus.ACTIVE
        else:
            status = ArtifactStatus.STAGED

        artifact = ModelArtifact(
            version=version,
            status=status,
            created_at=datetime.fromtimestamp(weights_path.stat().st_mtime),
            weights_path=str(weights_path),
            normalization_path=str(norm_path),
            calibration_path=str(cal_path) if cal_path.exists() else None,
            checksum=checksum,
            weights=weights,
            normalization=normalization,
            calibration=calibration,
        )

        self._artifacts[version] = artifact
        self._version_metrics[version] = ModelMetrics(version=version)

    def _compute_checksum(self, directory: Path) -> str:
        """Compute SHA256 checksum of directory contents."""
        hasher = hashlib.sha256()

        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "checksum.sha256":
                hasher.update(path.name.encode())
                hasher.update(path.read_bytes())

        return hasher.hexdigest()

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        return {
            "total_versions": len(self._artifacts),
            "current_version": self._current_version,
            "ab_test_active": self._ab_test is not None,
            "version_metrics": {
                v: {
                    "trades": m.trades,
                    "win_rate": m.win_rate,
                    "pnl": m.pnl_total,
                }
                for v, m in self._version_metrics.items()
            },
        }
