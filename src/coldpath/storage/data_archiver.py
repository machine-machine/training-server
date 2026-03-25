"""
Data Archiver - Long-running stability and sample persistence.

Solves the problem of data accumulation over weeks/months by:
1. Archiving old samples to compressed files
2. Maintaining a sliding window in SQLite
3. Allowing restoration of archived data for retraining

Usage:
    archiver = DataArchiver(db_path, archive_dir)
    await archiver.archive_old_samples(days=30)  # Archive samples older than 30 days
    await archiver.restore_archive(archive_file)  # Restore for retraining
"""

import asyncio
import gzip
import json
import logging
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import sqlite3

logger = logging.getLogger(__name__)


@dataclass
class ArchiveStats:
    """Statistics about an archive operation."""

    archived_samples: int = 0
    archived_trades: int = 0
    archived_counterfactuals: int = 0
    freed_mb: float = 0.0
    archive_file: str = ""
    archive_size_mb: float = 0.0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class ArchiveMetadata:
    """Metadata for an archive file."""

    created_at: str
    samples_count: int
    trades_count: int
    counterfactuals_count: int
    date_range_start: str
    date_range_end: str
    compressed_size_mb: float
    schema_version: int = 1


class DataArchiver:
    """
    Archives old training samples to compressed files.

    Features:
    - Compresses old samples to .jsonl.gz files
    - Maintains sliding window in SQLite (configurable)
    - Creates metadata file for each archive
    - Supports restoration for retraining
    - Runs VACUUM after archival to reclaim space

    Archive Structure:
        archive_dir/
        ├── archive_2026-03-01_to_2026-03-15.jsonl.gz
        ├── archive_2026-03-01_to_2026-03-15.meta.json
        ├── archive_2026-03-16_to_2026-03-31.jsonl.gz
        └── archive_2026-03-16_to_2026-03-31.meta.json
    """

    def __init__(
        self,
        db_path: str | Path,
        archive_dir: str | Path,
        retention_days: int = 90,  # Keep 90 days in SQLite
        vacuum_after_archive: bool = True,
    ):
        self.db_path = Path(db_path)
        self.archive_dir = Path(archive_dir)
        self.retention_days = retention_days
        self.vacuum_after_archive = vacuum_after_archive

        # Ensure archive directory exists
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    async def get_db_stats(self) -> dict[str, Any]:
        """Get current database statistics."""
        def _get_stats():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Table sizes
                stats = {}

                for table in ["scan_outcomes", "training_outcomes", "counterfactuals", "trades"]:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]

                # Oldest/newest samples
                cursor.execute("SELECT MIN(timestamp_ms), MAX(timestamp_ms) FROM scan_outcomes")
                row = cursor.fetchone()
                if row[0]:
                    stats["oldest_sample"] = datetime.fromtimestamp(row[0] / 1000).isoformat()
                    stats["newest_sample"] = datetime.fromtimestamp(row[1] / 1000).isoformat()

                # Database file size
                stats["db_size_mb"] = self.db_path.stat().st_size / (1024 * 1024)

                return stats

        return await asyncio.get_event_loop().run_in_executor(None, _get_stats)

    async def archive_old_samples(
        self,
        days: int | None = None,
        dry_run: bool = False,
    ) -> ArchiveStats:
        """
        Archive samples older than specified days.

        Args:
            days: Days to keep (default: retention_days)
            dry_run: If True, don't actually archive, just report

        Returns:
            ArchiveStats with operation results
        """
        retention_days = days or self.retention_days
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cutoff_ms = int(cutoff_date.timestamp() * 1000)

        stats = ArchiveStats()

        logger.info(f"Archiving samples older than {cutoff_date.isoformat()} (dry_run={dry_run})")

        def _archive():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Collect samples to archive
                cursor.execute(
                    """
                    SELECT * FROM scan_outcomes
                    WHERE timestamp_ms < ?
                    ORDER BY timestamp_ms
                    """,
                    (cutoff_ms,),
                )
                columns = [desc[0] for desc in cursor.description]
                samples = cursor.fetchall()
                stats.archived_samples = len(samples)

                if not samples or dry_run:
                    return stats

                # Find date range
                timestamps = [s[columns.index("timestamp_ms")] for s in samples]
                start_date = datetime.fromtimestamp(min(timestamps) / 1000)
                end_date = datetime.fromtimestamp(max(timestamps) / 1000)

                # Create archive file
                archive_name = f"archive_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"
                archive_path = self.archive_dir / f"{archive_name}.jsonl.gz"
                meta_path = self.archive_dir / f"{archive_name}.meta.json"

                # Write compressed archive
                with gzip.open(archive_path, "wt", encoding="utf-8") as f:
                    for row in samples:
                        record = dict(zip(columns, row))
                        # Convert datetime fields
                        if "created_at" in record and record["created_at"]:
                            record["created_at"] = str(record["created_at"])
                        f.write(json.dumps(record) + "\n")

                # Count trades
                stats.archived_trades = sum(
                    1 for s in samples
                    if s[columns.index("outcome_type")] == "traded"
                )

                # Delete archived samples from SQLite
                cursor.execute(
                    "DELETE FROM scan_outcomes WHERE timestamp_ms < ?",
                    (cutoff_ms,),
                )

                # Also archive related counterfactuals
                cursor.execute(
                    """
                    DELETE FROM counterfactuals
                    WHERE scan_outcome_id IN (
                        SELECT id FROM scan_outcomes WHERE timestamp_ms < ?
                    )
                    """,
                    (cutoff_ms,),
                )
                stats.archived_counterfactuals = cursor.rowcount

                conn.commit()

                # Get archive size
                stats.archive_file = str(archive_path)
                stats.archive_size_mb = archive_path.stat().st_size / (1024 * 1024)

                # Write metadata
                metadata = ArchiveMetadata(
                    created_at=datetime.now().isoformat(),
                    samples_count=stats.archived_samples,
                    trades_count=stats.archived_trades,
                    counterfactuals_count=stats.archived_counterfactuals,
                    date_range_start=start_date.isoformat(),
                    date_range_end=end_date.isoformat(),
                    compressed_size_mb=stats.archive_size_mb,
                )
                with open(meta_path, "w") as f:
                    json.dump(asdict(metadata), f, indent=2)

                return stats

        stats = await asyncio.get_event_loop().run_in_executor(None, _archive)

        if not dry_run and self.vacuum_after_archive and stats.archived_samples > 0:
            await self._vacuum_database()
            # Recalculate freed space
            current_size = self.db_path.stat().st_size / (1024 * 1024)
            stats.freed_mb = stats.archive_size_mb  # Approximate

        logger.info(f"Archive complete: {stats.archived_samples} samples -> {stats.archive_file}")
        return stats

    async def _vacuum_database(self):
        """Run VACUUM to reclaim space."""
        def _vacuum():
            logger.info("Running VACUUM on database...")
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM")
            logger.info("VACUUM complete")

        await asyncio.get_event_loop().run_in_executor(None, _vacuum)

    async def list_archives(self) -> list[dict[str, Any]]:
        """List all available archives."""
        archives = []

        for meta_file in self.archive_dir.glob("*.meta.json"):
            try:
                with open(meta_file) as f:
                    metadata = json.load(f)
                metadata["file_path"] = str(meta_file).replace(".meta.json", ".jsonl.gz")
                archives.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to read archive metadata {meta_file}: {e}")

        return sorted(archives, key=lambda x: x["created_at"], reverse=True)

    async def restore_archive(
        self,
        archive_path: str | Path,
        dry_run: bool = False,
    ) -> int:
        """
        Restore samples from an archive file.

        Args:
            archive_path: Path to .jsonl.gz archive file
            dry_run: If True, don't actually restore

        Returns:
            Number of samples restored
        """
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        logger.info(f"Restoring archive: {archive_path}")

        def _restore():
            restored = 0

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                with gzip.open(archive_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        if dry_run:
                            restored += 1
                            continue

                        record = json.loads(line)

                        # Insert into scan_outcomes
                        columns = list(record.keys())
                        placeholders = ",".join("?" * len(columns))
                        values = [record.get(c) for c in columns]

                        try:
                            cursor.execute(
                                f"INSERT OR IGNORE INTO scan_outcomes ({','.join(columns)}) VALUES ({placeholders})",
                                values,
                            )
                            if cursor.rowcount > 0:
                                restored += 1
                        except Exception as e:
                            logger.warning(f"Failed to restore record: {e}")

                if not dry_run:
                    conn.commit()

            return restored

        count = await asyncio.get_event_loop().run_in_executor(None, _restore)
        logger.info(f"Restored {count} samples from archive")
        return count

    async def get_total_archive_size(self) -> float:
        """Get total size of all archives in MB."""
        total = 0.0
        for archive_file in self.archive_dir.glob("*.jsonl.gz"):
            total += archive_file.stat().st_size / (1024 * 1024)
        return total

    async def prune_old_archives(self, keep_count: int = 12) -> int:
        """
        Remove old archives, keeping only the most recent N.

        Args:
            keep_count: Number of archives to keep

        Returns:
            Number of archives deleted
        """
        archives = await self.list_archives()
        if len(archives) <= keep_count:
            return 0

        to_delete = archives[keep_count:]
        deleted = 0

        for archive in to_delete:
            try:
                archive_path = Path(archive["file_path"])
                meta_path = Path(archive["file_path"]).with_suffix(".meta.json")

                archive_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                deleted += 1
                logger.info(f"Deleted old archive: {archive_path}")
            except Exception as e:
                logger.warning(f"Failed to delete archive: {e}")

        return deleted


# ─────────────────────────────────────────────────────────────
# Convenience function for startup
# ─────────────────────────────────────────────────────────────

async def run_maintenance(
    db_path: str | Path,
    archive_dir: str | Path,
    retention_days: int = 90,
) -> dict[str, Any]:
    """
    Run routine maintenance: archive old samples, vacuum database.

    Call this periodically (e.g., daily via cron or heartbeat).
    """
    archiver = DataArchiver(db_path, archive_dir, retention_days)

    # Get current stats
    before_stats = await archiver.get_db_stats()

    # Archive old samples
    archive_result = await archiver.archive_old_samples()

    # Get new stats
    after_stats = await archiver.get_db_stats()

    return {
        "before": before_stats,
        "after": after_stats,
        "archive": {
            "samples_archived": archive_result.archived_samples,
            "archive_file": archive_result.archive_file,
            "archive_size_mb": archive_result.archive_size_mb,
        },
        "total_archive_size_mb": await archiver.get_total_archive_size(),
    }
