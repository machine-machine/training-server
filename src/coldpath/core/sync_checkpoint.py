"""
Sync checkpoint management for ColdPath crash recovery.

Provides periodic state checkpointing and recovery for handling
SIGKILL (exit code 9) crashes where graceful shutdown is impossible.

The sync directory (/tmp/dexy-sync) is shared between HotPath and ColdPath
for coordination during restarts.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default sync directory (shared with HotPath)
DEFAULT_SYNC_DIR = "/tmp/dexy-sync"
CHECKPOINT_FILE = "coldpath_checkpoint.json"
RECOVERY_FILE = "coldpath_recovery.json"
SYNC_LOCK_FILE = "coldpath.lock"


@dataclass
class CheckpointState:
    """Represents a ColdPath checkpoint state."""

    timestamp: float
    status: str  # "starting", "healthy", "checkpointing", "shutting_down"
    restart_count: int
    pid: int

    # Model state checksums (for verification)
    bandit_checksum: str | None = None
    calibration_checksum: str | None = None
    bias_checksum: str | None = None
    autotrader_checksum: str | None = None

    # Runtime state
    uptime_seconds: float = 0.0
    last_training_time: float | None = None
    last_calibration_time: float | None = None
    pending_signals: int = 0
    pending_plans: int = 0

    # Health metrics
    memory_mb: float = 0.0
    healthy_loops: int = 0
    failed_loops: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RecoveryInfo:
    """Information about a previous crash for recovery."""

    crashed_at: float
    exit_code: int
    was_healthy: bool
    uptime_before_crash: float
    recovery_attempt: int
    state_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SyncCheckpoint:
    """
    Manages ColdPath sync checkpoints for crash recovery.

    Features:
    - Periodic checkpoint writing (configurable interval)
    - Atomic file writes to prevent corruption
    - Checksum verification for state integrity
    - Recovery detection and state validation
    - Lock file coordination with supervisor
    """

    def __init__(
        self,
        sync_dir: str = DEFAULT_SYNC_DIR,
        checkpoint_interval: float = 30.0,
        recovery_timeout: float = 300.0,
    ):
        self.sync_dir = Path(sync_dir)
        self.checkpoint_interval = checkpoint_interval
        self.recovery_timeout = recovery_timeout

        self._state: CheckpointState | None = None
        self._last_checkpoint: float = 0.0
        self._running = False
        self._task: asyncio.Task | None = None

        # Ensure sync directory exists
        self.sync_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self) -> Path:
        return self.sync_dir / CHECKPOINT_FILE

    def _recovery_path(self) -> Path:
        return self.sync_dir / RECOVERY_FILE

    def _lock_path(self) -> Path:
        return self.sync_dir / SYNC_LOCK_FILE

    def initialize(self, pid: int, restart_count: int = 0) -> CheckpointState:
        """Initialize checkpoint state at startup."""
        # Check for existing recovery info
        recovery = self._load_recovery_info()

        if recovery:
            logger.warning(
                f"Detected previous crash: exit_code={recovery.exit_code}, "
                f"uptime={recovery.uptime_before_crash:.1f}s, "
                f"recovery_attempt={recovery.recovery_attempt}"
            )

            if recovery.recovery_attempt >= 5:
                logger.error(
                    "Too many recovery attempts (%d). Consider manual intervention.",
                    recovery.recovery_attempt,
                )

        self._state = CheckpointState(
            timestamp=time.time(),
            status="starting",
            restart_count=restart_count
            if restart_count > 0
            else (recovery.recovery_attempt if recovery else 0),
            pid=pid,
        )

        self._write_checkpoint()
        return self._state

    def _load_recovery_info(self) -> RecoveryInfo | None:
        """Load recovery info from previous crash."""
        recovery_path = self._recovery_path()
        if not recovery_path.exists():
            return None

        try:
            with open(recovery_path) as f:
                data = json.load(f)
            return RecoveryInfo(**data)
        except Exception as e:
            logger.warning(f"Failed to load recovery info: {e}")
            return None

    def _write_checkpoint(self) -> bool:
        """Write checkpoint atomically."""
        if not self._state:
            return False

        checkpoint_path = self._checkpoint_path()
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            self._state.timestamp = time.time()

            with open(temp_path, "w") as f:
                json.dump(self._state.to_dict(), f, indent=2)

            # Atomic rename
            temp_path.rename(checkpoint_path)
            self._last_checkpoint = time.time()
            return True

        except Exception as e:
            logger.error(f"Failed to write checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False

    def update_status(self, status: str) -> None:
        """Update checkpoint status."""
        if self._state:
            self._state.status = status
            self._write_checkpoint()

    def update_metrics(
        self,
        memory_mb: float = 0.0,
        healthy_loops: int = 0,
        failed_loops: int = 0,
        pending_signals: int = 0,
        pending_plans: int = 0,
    ) -> None:
        """Update runtime metrics in checkpoint."""
        if self._state:
            self._state.memory_mb = memory_mb
            self._state.healthy_loops = healthy_loops
            self._state.failed_loops = failed_loops
            self._state.pending_signals = pending_signals
            self._state.pending_plans = pending_plans

            if self._state.timestamp > 0:
                self._state.uptime_seconds = time.time() - self._state.timestamp

    def update_checksums(
        self,
        bandit: str | None = None,
        calibration: str | None = None,
        bias: str | None = None,
        autotrader: str | None = None,
    ) -> None:
        """Update model state checksums."""
        if self._state:
            if bandit:
                self._state.bandit_checksum = bandit
            if calibration:
                self._state.calibration_checksum = calibration
            if bias:
                self._state.bias_checksum = bias
            if autotrader:
                self._state.autotrader_checksum = autotrader

    def update_timing(
        self,
        last_training: float | None = None,
        last_calibration: float | None = None,
    ) -> None:
        """Update timing information."""
        if self._state:
            if last_training:
                self._state.last_training_time = last_training
            if last_calibration:
                self._state.last_calibration_time = last_calibration

    async def start_checkpoint_loop(self) -> None:
        """Start the periodic checkpoint loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._checkpoint_loop())
        logger.info(f"Started checkpoint loop (interval={self.checkpoint_interval}s)")

    async def _checkpoint_loop(self) -> None:
        """Periodically write checkpoints."""
        while self._running:
            try:
                await asyncio.sleep(self.checkpoint_interval)

                if self._state and self._state.status == "healthy":
                    self._write_checkpoint()
                    logger.debug(f"Checkpoint written: uptime={self._state.uptime_seconds:.1f}s")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Checkpoint loop error: {e}")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop checkpoint loop and write final checkpoint."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Write final checkpoint
        if self._state:
            self._state.status = "shutting_down"
            self._write_checkpoint()

        logger.info("Checkpoint manager stopped")

    def mark_crashed(self, exit_code: int) -> None:
        """
        Mark this instance as crashed (called by supervisor before restart).

        This creates a recovery file that the next instance can use
        to detect and recover from the crash.
        """
        if not self._state:
            return

        recovery = RecoveryInfo(
            crashed_at=time.time(),
            exit_code=exit_code,
            was_healthy=self._state.status == "healthy",
            uptime_before_crash=self._state.uptime_seconds,
            recovery_attempt=self._state.restart_count + 1,
            state_valid=self._verify_state_checksums(),
        )

        recovery_path = self._recovery_path()
        try:
            with open(recovery_path, "w") as f:
                json.dump(recovery.to_dict(), f, indent=2)
            logger.info(f"Recovery info written for exit_code={exit_code}")
        except Exception as e:
            logger.error(f"Failed to write recovery info: {e}")

    def _verify_state_checksums(self) -> bool:
        """Verify saved state checksums match current state.

        Validates model file integrity by computing and comparing checksums
        against those stored in the checkpoint state.

        Returns:
            True if all checksums verify (or no checksums to verify), False otherwise
        """
        if not self._state:
            return True

        import hashlib

        # Map checksum field names to expected model file suffixes
        checksum_mappings = [
            ("bandit_checksum", "bandit_model.pkl"),
            ("calibration_checksum", "calibration_model.pkl"),
            ("bias_checksum", "bias_model.pkl"),
            ("autotrader_checksum", "autotrader_model.pkl"),
        ]

        for checksum_field, model_file in checksum_mappings:
            expected_checksum = getattr(self._state, checksum_field, None)
            if expected_checksum is None:
                continue

            model_path = self.sync_dir / model_file
            if not model_path.exists():
                logger.warning(f"Model file {model_file} missing during checksum verification")
                return False

            try:
                # Compute SHA256 checksum of model file
                sha256 = hashlib.sha256()
                with open(model_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256.update(chunk)
                computed_checksum = sha256.hexdigest()[:16]

                if computed_checksum != expected_checksum:
                    logger.warning(
                        f"Checksum mismatch for {model_file}: "
                        f"expected {expected_checksum}, got {computed_checksum}"
                    )
                    return False

            except Exception as e:
                logger.error(f"Error computing checksum for {model_file}: {e}")
                return False

        # All checksums verified (or no checksums to verify)
        return True

    def clear_recovery(self) -> None:
        """Clear recovery info after successful startup."""
        recovery_path = self._recovery_path()
        if recovery_path.exists():
            recovery_path.unlink()
            logger.info("Recovery info cleared")

    def acquire_lock(self, timeout: float = 10.0) -> bool:
        """Acquire sync lock for exclusive operations."""
        lock_path = self._lock_path()
        start = time.time()

        while time.time() - start < timeout:
            try:
                # Try to create lock file exclusively
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.write(fd, f"{os.getpid()}\n".encode())
                os.close(fd)
                return True
            except FileExistsError:
                # Check if lock is stale (older than 30s)
                try:
                    lock_age = time.time() - lock_path.stat().st_mtime
                    if lock_age > 30:
                        lock_path.unlink()
                        continue
                except FileNotFoundError:
                    continue

                time.sleep(0.1)

        return False

    def release_lock(self) -> None:
        """Release sync lock."""
        lock_path = self._lock_path()
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    @property
    def state(self) -> CheckpointState | None:
        return self._state

    def get_checkpoint_age(self) -> float:
        """Get age of last checkpoint in seconds."""
        return time.time() - self._last_checkpoint


def compute_state_checksum(state_dict: dict[str, Any]) -> str:
    """Compute a checksum for state dictionary."""
    import hashlib

    # Sort keys for consistent ordering
    state_str = json.dumps(state_dict, sort_keys=True, default=str)
    return hashlib.sha256(state_str.encode()).hexdigest()[:16]
