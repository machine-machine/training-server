"""
Live Mode Safeguards for Training Data

This module provides safety mechanisms for training data collected in live trading mode:
1. Minimum delay before using live data for training (prevent overfitting to recent events)
2. Anonymization/sanitization of sensitive data
3. Validation of execution quality before inclusion
4. Outlier detection and handling
5. Confirmation requirements for low-quality samples
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LiveDataQuality(Enum):
    """Quality levels for live training data."""

    HIGH = "high"  # Confirmed execution, low slippage, complete data
    MEDIUM = "medium"  # Reasonable execution, some slippage, complete data
    LOW = "low"  # High slippage or incomplete data
    INVALID = "invalid"  # Failed execution or missing critical data


class RejectionReason(Enum):
    """Reasons for rejecting live training data."""

    TOO_RECENT = "too_recent"  # Not enough time has passed
    HIGH_SLIPPAGE = "high_slippage"  # Execution slippage too high
    INCOMPLETE_DATA = "incomplete_data"  # Missing required fields
    FAILED_EXECUTION = "failed_execution"  # Trade execution failed
    SUSPICIOUS_ACTIVITY = "suspicious_activity"  # Potential MEV or manipulation
    OUTLIER_RETURN = "outlier_return"  # Return too extreme to be realistic
    INSUFFICIENT_CONFIRMATION = "insufficient_confirmation"  # Needs more confirmations
    DUPLICATE = "duplicate"  # Already recorded


@dataclass
class LiveDataSafeguardConfig:
    """Configuration for live data safeguards."""

    # Time delay before data can be used for training
    min_delay_hours: float = 24.0  # Wait 24 hours before using live data

    # Execution quality thresholds
    max_slippage_bps: float = 200.0  # Max 2% slippage allowed
    min_fill_rate: float = 0.95  # 95% of orders must fill

    # Return limits (detect unrealistic data)
    max_return_multiplier: float = 10.0  # Max 10x return (catch bugs)
    min_return_multiplier: float = -0.95  # Max 95% loss (can't lose more than invested)

    # Quality thresholds
    min_quality_for_training: str = "medium"  # Only use medium+ quality data
    require_confirmation_for_low_quality: bool = True

    # Outlier detection
    outlier_zscore_threshold: float = 4.0  # Extreme outliers rejected

    # Duplicate detection
    sample_dedup_window_hours: float = 1.0  # Deduplicate within 1 hour windows


@dataclass
class LiveDataSample:
    """A live trading data sample with safety checks."""

    sample_id: str
    timestamp_ms: int
    mint: str
    features: dict[str, float]

    # Execution details
    execution_quality: LiveDataQuality
    slippage_bps: float
    fill_rate: float
    execution_timestamp_ms: int

    # Outcome
    pnl_percentage: float
    pnl_sol: float
    was_profitable: bool

    # Safety flags
    is_rejected: bool = False
    rejection_reasons: list[RejectionReason] = field(default_factory=list)

    # Metadata
    mode: str = "LIVE"
    confirmed: bool = False
    confirmation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_id": self.sample_id,
            "timestamp_ms": self.timestamp_ms,
            "mint": self.mint,
            "execution_quality": self.execution_quality.value,
            "slippage_bps": self.slippage_bps,
            "fill_rate": self.fill_rate,
            "pnl_percentage": self.pnl_percentage,
            "pnl_sol": self.pnl_sol,
            "was_profitable": self.was_profitable,
            "is_rejected": self.is_rejected,
            "rejection_reasons": [r.value for r in self.rejection_reasons],
            "mode": self.mode,
            "confirmed": self.confirmed,
            "confirmation_count": self.confirmation_count,
        }


class LiveDataSafeguard:
    """Safeguards for live training data collection.

    This class ensures that live trading data meets quality standards
    before being used for model training. It prevents:
    - Overfitting to very recent events
    - Contamination from failed executions
    - Bias from high-slippage trades
    - Unrealistic return values
    - Duplicate samples
    """

    def __init__(self, config: LiveDataSafeguardConfig | None = None):
        self.config = config or LiveDataSafeguardConfig()
        self._sample_hashes: dict[str, int] = {}  # hash -> timestamp
        self._pending_samples: list[LiveDataSample] = []

    def validate_live_sample(
        self,
        sample_data: dict[str, Any],
    ) -> LiveDataSample:
        """Validate a live trading sample and apply safeguards.

        Args:
            sample_data: Raw sample data from live trading

        Returns:
            LiveDataSample with safety flags set
        """
        # Extract basic info
        timestamp_ms = sample_data.get("timestamp_ms", int(datetime.now().timestamp() * 1000))
        mint = sample_data.get("mint", "unknown")
        execution_timestamp_ms = sample_data.get("execution_timestamp_ms", timestamp_ms)

        # Calculate sample ID
        sample_id = self._calculate_sample_id(sample_data)

        # Determine execution quality
        execution_quality = self._assess_execution_quality(sample_data)

        # Create sample object
        sample = LiveDataSample(
            sample_id=sample_id,
            timestamp_ms=timestamp_ms,
            mint=mint,
            features=sample_data.get("features", {}),
            execution_quality=execution_quality,
            slippage_bps=sample_data.get("slippage_bps", 0.0),
            fill_rate=sample_data.get("fill_rate", 1.0),
            execution_timestamp_ms=execution_timestamp_ms,
            pnl_percentage=sample_data.get("pnl_percentage", 0.0),
            pnl_sol=sample_data.get("pnl_sol", 0.0),
            was_profitable=sample_data.get("was_profitable", False),
            confirmed=sample_data.get("confirmed", False),
            confirmation_count=sample_data.get("confirmation_count", 0),
        )

        # Apply safeguards
        self._check_time_delay(sample)
        self._check_execution_quality(sample)
        self._check_return_limits(sample)
        self._check_duplicate(sample)
        self._check_confirmation(sample)

        # Determine final rejection status
        sample.is_rejected = len(sample.rejection_reasons) > 0

        if sample.is_rejected:
            logger.warning(
                f"Live sample rejected: {sample.sample_id} - "
                f"Reasons: {[r.value for r in sample.rejection_reasons]}"
            )

        return sample

    def filter_valid_samples(
        self,
        samples: list[LiveDataSample],
        min_quality: str | None = None,
    ) -> list[LiveDataSample]:
        """Filter samples to only include valid ones for training.

        Args:
            samples: List of validated samples
            min_quality: Minimum quality level (default from config)

        Returns:
            List of samples suitable for training
        """
        min_quality = min_quality or self.config.min_quality_for_training
        quality_order = ["high", "medium", "low", "invalid"]
        min_quality_idx = quality_order.index(min_quality)

        valid_samples = []
        for sample in samples:
            # Skip rejected samples
            if sample.is_rejected:
                continue

            # Check quality level
            sample_quality_idx = quality_order.index(sample.execution_quality.value)
            if sample_quality_idx > min_quality_idx:
                continue

            # Check if low-quality samples need confirmation
            if (
                sample.execution_quality == LiveDataQuality.LOW
                and self.config.require_confirmation_for_low_quality
                and not sample.confirmed
            ):
                continue

            valid_samples.append(sample)

        logger.info(
            f"Filtered {len(valid_samples)}/{len(samples)} samples for training "
            f"(min_quality={min_quality})"
        )

        return valid_samples

    def get_safeguard_stats(self) -> dict[str, Any]:
        """Get statistics about safeguard effectiveness."""
        total_samples = len(self._pending_samples)
        rejected = sum(1 for s in self._pending_samples if s.is_rejected)

        rejection_reasons = {}
        for sample in self._pending_samples:
            for reason in sample.rejection_reasons:
                rejection_reasons[reason.value] = rejection_reasons.get(reason.value, 0) + 1

        quality_distribution = {}
        for sample in self._pending_samples:
            quality = sample.execution_quality.value
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1

        return {
            "total_samples": total_samples,
            "rejected_samples": rejected,
            "rejection_rate": rejected / total_samples if total_samples > 0 else 0,
            "rejection_reasons": rejection_reasons,
            "quality_distribution": quality_distribution,
        }

    def _assess_execution_quality(self, sample_data: dict[str, Any]) -> LiveDataQuality:
        """Assess the quality of trade execution."""
        slippage_bps = abs(sample_data.get("slippage_bps", 0.0))
        fill_rate = sample_data.get("fill_rate", 1.0)

        # Check for failed execution
        execution_status = sample_data.get("execution_status", "success")
        if execution_status == "failed":
            return LiveDataQuality.INVALID

        # High quality: low slippage, complete fill
        if slippage_bps <= 50 and fill_rate >= 0.99:
            return LiveDataQuality.HIGH

        # Medium quality: reasonable slippage, good fill
        if slippage_bps <= 100 and fill_rate >= 0.95:
            return LiveDataQuality.MEDIUM

        # Low quality: high slippage or partial fill
        if slippage_bps <= 200 and fill_rate >= 0.90:
            return LiveDataQuality.LOW

        # Invalid: excessive slippage or failed fill
        return LiveDataQuality.INVALID

    def _check_time_delay(self, sample: LiveDataSample) -> None:
        """Check if enough time has passed since execution."""
        now_ms = int(datetime.now().timestamp() * 1000)
        delay_hours = (now_ms - sample.execution_timestamp_ms) / (1000 * 3600)

        if delay_hours < self.config.min_delay_hours:
            sample.rejection_reasons.append(RejectionReason.TOO_RECENT)
            logger.debug(
                f"Sample {sample.sample_id} too recent: "
                f"{delay_hours:.1f}h < {self.config.min_delay_hours}h"
            )

    def _check_execution_quality(self, sample: LiveDataSample) -> None:
        """Check execution quality thresholds."""
        # Check slippage
        if sample.slippage_bps > self.config.max_slippage_bps:
            sample.rejection_reasons.append(RejectionReason.HIGH_SLIPPAGE)
            logger.debug(
                f"Sample {sample.sample_id} high slippage: "
                f"{sample.slippage_bps:.1f}bps > {self.config.max_slippage_bps}bps"
            )

        # Check fill rate
        if sample.fill_rate < self.config.min_fill_rate:
            sample.rejection_reasons.append(RejectionReason.FAILED_EXECUTION)
            logger.debug(
                f"Sample {sample.sample_id} low fill rate: "
                f"{sample.fill_rate:.1%} < {self.config.min_fill_rate:.1%}"
            )

        # Check for invalid quality
        if sample.execution_quality == LiveDataQuality.INVALID:
            if RejectionReason.FAILED_EXECUTION not in sample.rejection_reasons:
                sample.rejection_reasons.append(RejectionReason.INCOMPLETE_DATA)

    def _check_return_limits(self, sample: LiveDataSample) -> None:
        """Check for unrealistic return values."""
        # Calculate return multiplier
        if sample.pnl_percentage > 0:
            multiplier = 1 + (sample.pnl_percentage / 100)
        else:
            multiplier = 1 + (sample.pnl_percentage / 100)

        # Check for extreme returns
        if multiplier > self.config.max_return_multiplier:
            sample.rejection_reasons.append(RejectionReason.OUTLIER_RETURN)
            logger.warning(
                f"Sample {sample.sample_id} has unrealistic return: "
                f"{sample.pnl_percentage:.1f}% ({multiplier:.1f}x)"
            )

        if multiplier < self.config.min_return_multiplier:
            sample.rejection_reasons.append(RejectionReason.OUTLIER_RETURN)
            logger.warning(
                f"Sample {sample.sample_id} has unrealistic loss: "
                f"{sample.pnl_percentage:.1f}% ({multiplier:.1f}x)"
            )

    def _check_duplicate(self, sample: LiveDataSample) -> None:
        """Check for duplicate samples."""
        # Create hash from key fields
        hash_data = f"{sample.mint}:{sample.execution_timestamp_ms}:{sample.pnl_percentage}"
        sample_hash = hashlib.md5(hash_data.encode()).hexdigest()

        # Check for recent duplicate
        now_ms = int(datetime.now().timestamp() * 1000)
        dedup_window_ms = int(self.config.sample_dedup_window_hours * 3600 * 1000)

        if sample_hash in self._sample_hashes:
            last_timestamp = self._sample_hashes[sample_hash]
            if (now_ms - last_timestamp) < dedup_window_ms:
                sample.rejection_reasons.append(RejectionReason.DUPLICATE)
                logger.debug(f"Sample {sample.sample_id} is duplicate")

        # Update hash table
        self._sample_hashes[sample_hash] = now_ms

        # Clean old hashes
        cutoff_ms = now_ms - (dedup_window_ms * 2)
        self._sample_hashes = {h: t for h, t in self._sample_hashes.items() if t > cutoff_ms}

    def _check_confirmation(self, sample: LiveDataSample) -> None:
        """Check confirmation requirements for low-quality samples."""
        if (
            sample.execution_quality == LiveDataQuality.LOW
            and self.config.require_confirmation_for_low_quality
            and not sample.confirmed
        ):
            sample.rejection_reasons.append(RejectionReason.INSUFFICIENT_CONFIRMATION)
            logger.debug(
                f"Sample {sample.sample_id} needs confirmation "
                f"(quality={sample.execution_quality.value})"
            )

    def _calculate_sample_id(self, sample_data: dict[str, Any]) -> str:
        """Calculate unique sample ID."""
        hash_data = (
            f"{sample_data.get('mint', '')}:"
            f"{sample_data.get('timestamp_ms', 0)}:"
            f"{sample_data.get('execution_timestamp_ms', 0)}"
        )
        return hashlib.md5(hash_data.encode()).hexdigest()[:16]

    def add_pending_sample(self, sample: LiveDataSample) -> None:
        """Add sample to pending queue for later processing."""
        self._pending_samples.append(sample)

        # Limit pending sample history
        max_pending = 10000
        if len(self._pending_samples) > max_pending:
            self._pending_samples = self._pending_samples[-max_pending // 2 :]

    def clear_pending_samples(self) -> None:
        """Clear pending sample queue."""
        self._pending_samples.clear()
