"""
Training Data Sync Module - Bridges JSONL files to SQLite tables.

This module solves the data flow gap between:
- SwiftUI DataCollectionService → JSONL files
- ColdPath training_integration → SQLite tables

The sync reads JSONL training samples and inserts them into the SQLite
training_outcomes table so that the training pipeline can use them.

Usage:
    sync = TrainingDataSync(db)
    synced = await sync.sync_all_sources()
    print(f"Synced {synced} samples to database")
"""

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a training data sync operation."""

    total_samples: int = 0
    valid_samples: int = 0
    inserted_samples: int = 0
    duplicate_samples: int = 0
    invalid_samples: int = 0
    sources_synced: list[str] = None
    errors: list[str] = None

    def __post_init__(self):
        if self.sources_synced is None:
            self.sources_synced = []
        if self.errors is None:
            self.errors = []

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 or self.inserted_samples > 0

    def summary(self) -> str:
        return (
            f"SyncResult: {self.inserted_samples} inserted, "
            f"{self.duplicate_samples} duplicates skipped, "
            f"{self.invalid_samples} invalid, "
            f"from {len(self.sources_synced)} sources"
        )


class TrainingDataSync:
    """Syncs training samples from JSONL files to SQLite database.

    Reads from multiple locations:
    1. Swift app: ~/Library/Application Support/2DEXY/TrainingData/
    2. HotPath IPC: EngineColdPath/data/training/
    3. ColdPath export: EngineColdPath/data/training/

    Inserts into:
    - training_outcomes table (for labeled samples with outcomes)
    - scan_outcomes table (for unlabeled candidate samples)
    """

    # Feature names in canonical order (must match HotPath v3 schema)
    FEATURE_NAMES = [
        "liquidity_usd",
        "liquidity_change_1h_pct",
        "liquidity_change_6h_pct",
        "liquidity_change_24h_pct",
        "fdv_usd",
        "fdv_to_liquidity_ratio",
        "market_cap_usd",
        "volume_24h",
        "volume_6h",
        "volume_1h",
        "volume_to_liquidity_ratio",
        "buy_volume_24h_ratio",
        "holder_count",
        "holder_change_1h_pct",
        "holder_change_6h_pct",
        "holder_change_24h_pct",
        "top_holder_pct",
        "top_5_holder_pct",
        "top_10_holder_pct",
        "unique_wallets_24h",
        "whale_count",
        "retail_ratio",
        "creator_token_pct",
        "price_change_5m_pct",
        "price_change_15m_pct",
        "price_change_1h_pct",
        "price_change_6h_pct",
        "price_change_24h_pct",
        "volatility_1h",
        "volatility_6h",
        "volatility_24h",
        "high_low_range_1h",
        "price_momentum",
        "rug_risk_score",
        "honeypot_probability",
        "mint_authority_enabled",
        "freeze_authority_enabled",
        "lp_locked_pct",
        "token_age_hours",
        "tx_count_24h",
        "failed_tx_ratio",
        "suspicious_patterns_score",
        "creator_history_score",
        "ofi",
        "mlofi",
        "trade_arrival_rate",
        "avg_trade_size_usd",
        "buy_concentration",
        "large_tx_ratio",
        "mev_exposure_score",
    ]

    def __init__(self, db: Any, sync_interval_seconds: float = 60.0):
        """
        Args:
            db: DatabaseManager instance
            sync_interval_seconds: How often to run periodic sync
        """
        self.db = db
        self.sync_interval = sync_interval_seconds
        self._running = False
        self._sync_task: asyncio.Task | None = None
        self._last_sync: datetime | None = None
        self._total_synced = 0

        # Track seen record IDs to avoid duplicates
        self._seen_record_ids: set[str] = set()
        self._max_seen_cache = 10000

    def _get_swift_samples_dir(self) -> Path:
        """Get Swift app's training samples directory."""
        home = Path.home()
        return home / "Library" / "Application Support" / "2DEXY" / "TrainingData"

    def _get_coldpath_samples_dir(self) -> Path:
        """Get ColdPath's training samples directory."""
        project_root = Path(__file__).resolve().parents[3]  # EngineColdPath/
        return project_root / "data" / "training"

    def _get_source_paths(self) -> list[tuple[Path, str]]:
        """Get all source paths to check for training samples.

        Returns:
            List of (path, source_name) tuples
        """
        paths = []

        # Swift app samples (highest priority - real user data)
        swift_dir = self._get_swift_samples_dir()
        paths.append((swift_dir / "training_samples.jsonl", "swift_app"))

        # HotPath IPC sync destination
        coldpath_dir = self._get_coldpath_samples_dir()
        paths.append((coldpath_dir / "training_samples.jsonl", "hotpath_ipc"))

        # ColdPath export files
        if coldpath_dir.exists():
            for export_file in coldpath_dir.glob("training_export_*.json"):
                paths.append((export_file, "coldpath_export"))

        return paths

    async def sync_all_sources(self) -> SyncResult:
        """Sync training samples from all sources to database.

        Returns:
            SyncResult with sync statistics
        """
        result = SyncResult()
        sources = self._get_source_paths()

        for path, source_name in sources:
            if not path.exists():
                logger.debug(f"Source path does not exist: {path}")
                continue

            try:
                source_result = await self._sync_source(path, source_name)
                result.total_samples += source_result.total_samples
                result.valid_samples += source_result.valid_samples
                result.inserted_samples += source_result.inserted_samples
                result.duplicate_samples += source_result.duplicate_samples
                result.invalid_samples += source_result.invalid_samples
                if source_result.inserted_samples > 0:
                    result.sources_synced.append(source_name)
                result.errors.extend(source_result.errors)
                logger.info(
                    f"Synced {source_result.inserted_samples} samples from {source_name} "
                    f"({path.name})"
                )
            except Exception as e:
                error_msg = f"Failed to sync {source_name}: {e}"
                logger.error(error_msg, exc_info=True)
                result.errors.append(error_msg)

        self._last_sync = datetime.now()
        self._total_synced += result.inserted_samples

        # Trim seen cache if too large
        if len(self._seen_record_ids) > self._max_seen_cache:
            # Keep most recent half
            self._seen_record_ids = set(list(self._seen_record_ids)[-self._max_seen_cache // 2 :])

        logger.info(result.summary())
        return result

    async def _sync_source(self, path: Path, source_name: str) -> SyncResult:
        """Sync samples from a single source file.

        Args:
            path: Path to JSONL or JSON file
            source_name: Name of the source for logging

        Returns:
            SyncResult for this source
        """
        result = SyncResult()

        if path.suffix == ".json":
            # Handle JSON export files
            result = await self._sync_json_export(path, source_name)
        else:
            # Handle JSONL files
            result = await self._sync_jsonl_file(path, source_name)

        return result

    async def _sync_jsonl_file(self, path: Path, source_name: str) -> SyncResult:
        """Sync samples from a JSONL file.

        Args:
            path: Path to JSONL file
            source_name: Source name

        Returns:
            SyncResult
        """
        result = SyncResult()

        try:
            content = path.read_text(encoding="utf-8")
            lines = content.strip().split("\n")

            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue

                try:
                    sample = json.loads(line)
                    result.total_samples += 1

                    # Validate and insert
                    validation = self._validate_sample(sample)
                    if not validation["valid"]:
                        result.invalid_samples += 1
                        logger.debug(f"Invalid sample at line {line_num}: {validation['reason']}")
                        continue

                    result.valid_samples += 1

                    # Check for duplicates
                    record_id = self._generate_record_id(sample, source_name)
                    if record_id in self._seen_record_ids:
                        result.duplicate_samples += 1
                        continue

                    # Insert to database
                    inserted = await self._insert_sample(sample, record_id, source_name)
                    if inserted:
                        result.inserted_samples += 1
                        self._seen_record_ids.add(record_id)
                    else:
                        result.duplicate_samples += 1

                except json.JSONDecodeError as e:
                    result.invalid_samples += 1
                    result.errors.append(f"JSON parse error at line {line_num}: {e}")
                except Exception as e:
                    result.invalid_samples += 1
                    result.errors.append(f"Error processing line {line_num}: {e}")

        except Exception as e:
            result.errors.append(f"Failed to read file: {e}")

        return result

    async def _sync_json_export(self, path: Path, source_name: str) -> SyncResult:
        """Sync samples from a JSON export file.

        Args:
            path: Path to JSON file
            source_name: Source name

        Returns:
            SyncResult
        """
        result = SyncResult()

        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)

            # Handle TrainingExport format
            samples = data.get("samples", [])
            if not samples and isinstance(data, list):
                samples = data

            for sample in samples:
                result.total_samples += 1

                validation = self._validate_sample(sample)
                if not validation["valid"]:
                    result.invalid_samples += 1
                    continue

                result.valid_samples += 1

                record_id = self._generate_record_id(sample, source_name)
                if record_id in self._seen_record_ids:
                    result.duplicate_samples += 1
                    continue

                inserted = await self._insert_sample(sample, record_id, source_name)
                if inserted:
                    result.inserted_samples += 1
                    self._seen_record_ids.add(record_id)
                else:
                    result.duplicate_samples += 1

        except json.JSONDecodeError as e:
            result.errors.append(f"JSON parse error: {e}")
        except Exception as e:
            result.errors.append(f"Failed to read export: {e}")

        return result

    def _validate_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Validate a training sample.

        Args:
            sample: Sample dictionary

        Returns:
            Dict with 'valid' bool and optional 'reason' string
        """
        # Must have mint
        if not sample.get("mint"):
            return {"valid": False, "reason": "Missing mint address"}

        # Must have features
        features = sample.get("features", [])
        if not features:
            return {"valid": False, "reason": "Missing features"}

        # Must have 50 features (v3 schema)
        if len(features) != 50:
            return {"valid": False, "reason": f"Expected 50 features, got {len(features)}"}

        # Features must be finite
        for i, f in enumerate(features):
            if not isinstance(f, (int, float)) or not math.isfinite(float(f)):
                return {"valid": False, "reason": f"Invalid feature at index {i}"}

        # Sanity bounds check
        for i, f in enumerate(features):
            if abs(float(f)) > 1_000_000:
                return {"valid": False, "reason": f"Feature {i} exceeds sanity bounds"}

        return {"valid": True}

    def _generate_record_id(self, sample: dict[str, Any], source_name: str) -> str:
        """Generate a unique record ID for deduplication.

        Args:
            sample: Sample dictionary
            source_name: Source name

        Returns:
            Unique record ID string
        """
        mint = sample.get("mint", "unknown")
        timestamp_ms = sample.get("timestamp_ms", sample.get("timestamp", 0))

        # Handle different timestamp formats
        if isinstance(timestamp_ms, str):
            try:
                # ISO format
                dt = datetime.fromisoformat(timestamp_ms.replace("Z", "+00:00"))
                timestamp_ms = int(dt.timestamp() * 1000)
            except Exception:
                timestamp_ms = 0

        return f"{source_name}_{mint}_{timestamp_ms}"

    async def _insert_sample(
        self, sample: dict[str, Any], record_id: str, source_name: str
    ) -> bool:
        """Insert a training sample into the database.

        Args:
            sample: Validated sample dictionary
            record_id: Unique record ID
            source_name: Source name

        Returns:
            True if inserted, False if duplicate or error
        """
        try:
            mint = sample.get("mint", "")
            timestamp_ms = sample.get("timestamp_ms", 0)

            # Handle ISO timestamp format
            if isinstance(timestamp_ms, str):
                try:
                    dt = datetime.fromisoformat(timestamp_ms.replace("Z", "+00:00"))
                    timestamp_ms = int(dt.timestamp() * 1000)
                except Exception:
                    timestamp_ms = int(datetime.now().timestamp() * 1000)

            features = sample.get("features", [])
            features_json = json.dumps(dict(zip(self.FEATURE_NAMES, features, strict=False)))

            outcome = sample.get("outcome")

            # Determine label from outcome
            if outcome:
                was_profitable = outcome.get("was_profitable")
                pnl_pct = outcome.get("pnl_percentage", outcome.get("pnl_pct", 0.0))
                pnl_sol = outcome.get("pnl_sol", 0.0)
                exit_reason = outcome.get("exit_reason", "")

                # Skip unlabeled samples (was_profitable is None)
                if was_profitable is None:
                    # Treat as unlabeled - insert into scan_outcomes instead
                    logger.debug(f"Skipping unlabeled sample: {mint}")
                    await self._insert_scan_outcome(sample, source_name)
                    return True

                # Check for expired samples (time-based exit with no PnL)
                if exit_reason == "time_based" and pnl_pct == 0:
                    # Expired sample - skip entirely
                    logger.debug(f"Skipping expired sample: {mint}")
                    return False

                label_binary = 1 if was_profitable else 0
                label_return = pnl_pct / 100.0 if pnl_pct else 0.0

                # Determine ev_label
                ev_label = "GOOD_EV" if was_profitable else "BAD_EV"
            else:
                # Unlabeled sample - insert into scan_outcomes instead
                await self._insert_scan_outcome(sample, source_name)
                return True

            # Insert into training_outcomes table
            await self.db.insert_training_outcome(
                {
                    "record_id": record_id,
                    "mint": mint,
                    "decision_timestamp_ms": timestamp_ms,
                    "regime": sample.get("collection_mode", "unknown"),
                    "features_json": features_json,
                    "model_version": sample.get("model_version", 3),
                    "model_score": None,
                    "execution_mode": sample.get("collection_mode", "paper"),
                    "pnl_sol": pnl_sol,
                    "pnl_pct": pnl_pct,
                    "token_status": "labeled",
                    "ev_label": ev_label,
                    "label_binary": label_binary,
                    "label_return": label_return,
                    "label_quality": 1.0,  # Real labeled data
                }
            )

            logger.debug(f"Inserted training outcome: {record_id}")
            return True

        except Exception as e:
            # Check for duplicate key error
            if "UNIQUE constraint" in str(e) or "duplicate" in str(e).lower():
                logger.debug(f"Duplicate sample skipped: {record_id}")
                return False
            logger.error(f"Failed to insert sample: {e}")
            return False

    async def _insert_scan_outcome(self, sample: dict[str, Any], source_name: str) -> bool:
        """Insert an unlabeled sample into scan_outcomes table.

        Args:
            sample: Sample dictionary
            source_name: Source name

        Returns:
            True if inserted successfully
        """
        try:
            mint = sample.get("mint", "")
            timestamp_ms = sample.get("timestamp_ms", 0)

            if isinstance(timestamp_ms, str):
                try:
                    dt = datetime.fromisoformat(timestamp_ms.replace("Z", "+00:00"))
                    timestamp_ms = int(dt.timestamp() * 1000)
                except Exception:
                    timestamp_ms = int(datetime.now().timestamp() * 1000)

            features = sample.get("features", [])
            features_json = json.dumps(dict(zip(self.FEATURE_NAMES, features, strict=False)))

            await self.db.insert_scan_outcome(
                {
                    "mint": mint,
                    "pool": None,
                    "timestamp_ms": timestamp_ms,
                    "outcome_type": "pending_label",
                    "skip_reason": None,
                    "features_json": features_json,
                    "profitability_score": None,
                    "confidence": None,
                    "expected_return_pct": None,
                    "model_version": sample.get("model_version", 3),
                    "source": source_name,
                }
            )

            return True

        except Exception as e:
            logger.debug(f"Failed to insert scan outcome: {e}")
            return False

    async def start_periodic_sync(self) -> None:
        """Start periodic background sync task."""
        if self._running:
            logger.warning("Training data sync already running")
            return

        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(f"Training data sync started (interval={self.sync_interval}s)")

    async def stop_periodic_sync(self) -> None:
        """Stop periodic sync task."""
        self._running = False

        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        logger.info("Training data sync stopped")

    async def _sync_loop(self) -> None:
        """Background sync loop."""
        # Initial sync after short delay
        await asyncio.sleep(5)

        while self._running:
            try:
                result = await self.sync_all_sources()
                if result.inserted_samples > 0:
                    logger.info(
                        f"Periodic sync: {result.inserted_samples} new samples "
                        f"(total synced: {self._total_synced})"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}", exc_info=True)

            await asyncio.sleep(self.sync_interval)

    def get_status(self) -> dict[str, Any]:
        """Get sync status."""
        return {
            "running": self._running,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "total_synced": self._total_synced,
            "seen_cache_size": len(self._seen_record_ids),
            "sync_interval_seconds": self.sync_interval,
        }


# Global instance
_training_data_sync: TrainingDataSync | None = None


def get_training_data_sync(db: Any | None = None) -> TrainingDataSync:
    """Get or create the global training data sync instance."""
    global _training_data_sync

    if _training_data_sync is None:
        if db is None:
            raise ValueError("Database required for first initialization")
        _training_data_sync = TrainingDataSync(db)

    return _training_data_sync


async def setup_training_data_sync(db: Any) -> TrainingDataSync:
    """Set up and start the training data sync.

    This should be called during ColdPath startup.
    """
    sync = get_training_data_sync(db)
    await sync.start_periodic_sync()
    return sync
