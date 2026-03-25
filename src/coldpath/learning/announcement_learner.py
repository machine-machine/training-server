"""
Announcement learner for improving signal quality over time.

Responsibilities:
- Track which announcements led to profitable trades
- Update source reputation based on outcomes
- Periodically retrain classifier with new labeled data
- Calibrate Thompson sampling thresholds
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


@dataclass
class LearnerConfig:
    """Configuration for announcement learner."""

    # Database
    database_url: str = "sqlite:///sniperdesk.db"

    # Learning intervals
    reputation_update_interval_secs: int = 300  # 5 minutes
    threshold_update_interval_secs: int = 600  # 10 minutes
    classifier_retrain_interval_secs: int = 3600  # 1 hour

    # Minimum samples for learning
    min_samples_for_reputation: int = 5
    min_samples_for_retrain: int = 100

    # Reputation decay
    reputation_decay_factor: float = 0.99  # Decay old observations
    reputation_smoothing: float = 0.1  # Smoothing for new observations

    @classmethod
    def from_env(cls) -> "LearnerConfig":
        """Load configuration from environment."""
        return cls(
            database_url=os.environ.get(
                "DATABASE_URL", f"sqlite:///{os.environ.get('SQLITE_PATH', 'sniperdesk.db')}"
            ),
            reputation_update_interval_secs=int(
                os.environ.get("REPUTATION_UPDATE_INTERVAL_SECS", "300")
            ),
            threshold_update_interval_secs=int(
                os.environ.get("THRESHOLD_UPDATE_INTERVAL_SECS", "600")
            ),
            classifier_retrain_interval_secs=int(
                os.environ.get("CLASSIFIER_RETRAIN_INTERVAL_SECS", "3600")
            ),
        )


@dataclass
class OutcomeRecord:
    """Record of an announcement outcome."""

    announcement_id: str
    signal_id: str | None
    mint: str
    source_id: str
    source_type: str
    was_traded: bool
    was_profitable: bool | None
    pnl_sol: float | None
    was_scam: bool
    lead_time_ms: int | None
    recorded_at_ms: int


@dataclass
class ReputationUpdate:
    """Update to source reputation."""

    source_id: str
    source_type: str
    new_score: float
    total_announcements: int
    valid_trades: int
    scam_count: int


class AnnouncementLearner:
    """
    Learner for improving announcement signal quality.

    Tracks outcomes and updates:
    - Source reputation scores
    - Thompson sampling thresholds
    - Classifier training data
    """

    def __init__(self, config: LearnerConfig | None = None):
        """Initialize the learner."""
        self.config = config or LearnerConfig.from_env()

        # Database engine (sync for simplicity)
        self.engine = create_engine(self.config.database_url)

        # Outcome buffer for batch processing
        self._outcome_buffer: list[OutcomeRecord] = []
        self._buffer_lock = asyncio.Lock()

        # Statistics
        self.stats = {
            "outcomes_recorded": 0,
            "reputation_updates": 0,
            "threshold_updates": 0,
            "retrains": 0,
        }

        # Running state
        self._running = False
        self._tasks: list[asyncio.Task] = []

        logger.info("AnnouncementLearner initialized")

    async def start(self):
        """Start the learner background tasks."""
        if self._running:
            return

        self._running = True

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._reputation_update_loop()),
            asyncio.create_task(self._threshold_update_loop()),
            asyncio.create_task(self._retrain_loop()),
        ]

        logger.info("AnnouncementLearner started")

    async def stop(self):
        """Stop the learner."""
        self._running = False

        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

        # Dispose database engine to close all pooled connections
        if hasattr(self, "engine") and self.engine is not None:
            try:
                self.engine.dispose()
                logger.debug("Database engine disposed")
            except Exception as e:
                logger.warning(f"Error disposing database engine: {e}")

        logger.info("AnnouncementLearner stopped")

    async def record_outcome(self, outcome: OutcomeRecord):
        """
        Record an outcome for learning.

        This is called when we know the result of an announcement
        (trade outcome, scam detection, etc.)
        """
        async with self._buffer_lock:
            self._outcome_buffer.append(outcome)
            self.stats["outcomes_recorded"] += 1

        # Also persist to database
        await self._persist_outcome(outcome)

    async def _persist_outcome(self, outcome: OutcomeRecord):
        """Persist outcome to database."""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO announcement_outcomes (
                            announcement_id, signal_id, mint,
                            was_traded, was_profitable, pnl_sol,
                            was_scam, lead_time_ms, recorded_at_ms
                        ) VALUES (
                            :announcement_id, :signal_id, :mint,
                            :was_traded, :was_profitable, :pnl_sol,
                            :was_scam, :lead_time_ms, :recorded_at_ms
                        )
                    """),
                    {
                        "announcement_id": outcome.announcement_id,
                        "signal_id": outcome.signal_id,
                        "mint": outcome.mint,
                        "was_traded": int(outcome.was_traded),
                        "was_profitable": (
                            int(outcome.was_profitable)
                            if outcome.was_profitable is not None
                            else None
                        ),
                        "pnl_sol": outcome.pnl_sol,
                        "was_scam": int(outcome.was_scam),
                        "lead_time_ms": outcome.lead_time_ms,
                        "recorded_at_ms": outcome.recorded_at_ms,
                    },
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist outcome: {e}")

    async def _reputation_update_loop(self):
        """Background loop for updating source reputations."""
        while self._running:
            try:
                await asyncio.sleep(self.config.reputation_update_interval_secs)
                await self._update_reputations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reputation update error: {e}")

    async def _update_reputations(self):
        """Update source reputations from recent outcomes."""
        logger.debug("Updating source reputations")

        # Get outcomes from buffer
        async with self._buffer_lock:
            outcomes = self._outcome_buffer.copy()
            self._outcome_buffer.clear()

        if not outcomes:
            return

        # Group by source
        by_source: dict[tuple[str, str], list[OutcomeRecord]] = {}
        for outcome in outcomes:
            key = (outcome.source_type, outcome.source_id)
            if key not in by_source:
                by_source[key] = []
            by_source[key].append(outcome)

        # Update each source
        updates = []
        for (source_type, source_id), source_outcomes in by_source.items():
            update = await self._calculate_reputation_update(
                source_type, source_id, source_outcomes
            )
            if update:
                updates.append(update)

        # Persist updates
        for update in updates:
            await self._persist_reputation(update)

        self.stats["reputation_updates"] += len(updates)
        logger.info(f"Updated {len(updates)} source reputations")

    async def _calculate_reputation_update(
        self,
        source_type: str,
        source_id: str,
        outcomes: list[OutcomeRecord],
    ) -> ReputationUpdate | None:
        """Calculate reputation update for a source."""
        if len(outcomes) < self.config.min_samples_for_reputation:
            return None

        # Get current reputation
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT total_announcements, valid_trades, scam_count, reputation_score
                        FROM source_reputation
                        WHERE source_type = :source_type AND source_id = :source_id
                    """),
                    {"source_type": source_type, "source_id": source_id},
                ).fetchone()

                if result:
                    current_total = result[0]
                    current_valid = result[1]
                    current_scam = result[2]
                    current_score = result[3]
                else:
                    current_total = 0
                    current_valid = 0
                    current_scam = 0
                    current_score = 0.5
        except Exception as e:
            logger.error(f"Failed to get current reputation: {e}")
            return None

        # Apply decay to old observations
        decayed_total = current_total * self.config.reputation_decay_factor
        decayed_valid = current_valid * self.config.reputation_decay_factor
        decayed_scam = current_scam * self.config.reputation_decay_factor

        # Add new observations
        new_total = len(outcomes)
        new_valid = sum(1 for o in outcomes if o.was_profitable)
        new_scam = sum(1 for o in outcomes if o.was_scam)

        final_total = decayed_total + new_total
        final_valid = decayed_valid + new_valid
        final_scam = decayed_scam + new_scam

        # Calculate new score
        if final_total > 0:
            valid_rate = final_valid / final_total
            scam_rate = final_scam / final_total
            raw_score = valid_rate - scam_rate

            # Confidence adjustment
            confidence = 1.0 - np.exp(-final_total / 50)

            # Smooth update
            new_score = (
                current_score * (1 - self.config.reputation_smoothing)
                + (0.5 + raw_score * 0.5) * confidence * self.config.reputation_smoothing
            )
            new_score = max(0.0, min(1.0, new_score))
        else:
            new_score = current_score

        return ReputationUpdate(
            source_id=source_id,
            source_type=source_type,
            new_score=new_score,
            total_announcements=int(final_total),
            valid_trades=int(final_valid),
            scam_count=int(final_scam),
        )

    async def _persist_reputation(self, update: ReputationUpdate):
        """Persist reputation update to database."""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO source_reputation (
                            source_id, source_type, total_announcements,
                            valid_trades, scam_count, reputation_score, updated_at_ms
                        ) VALUES (
                            :source_id, :source_type, :total,
                            :valid, :scam, :score, :updated_at
                        )
                        ON CONFLICT(source_id, source_type) DO UPDATE SET
                            total_announcements = :total,
                            valid_trades = :valid,
                            scam_count = :scam,
                            reputation_score = :score,
                            updated_at_ms = :updated_at
                    """),
                    {
                        "source_id": update.source_id,
                        "source_type": update.source_type,
                        "total": update.total_announcements,
                        "valid": update.valid_trades,
                        "scam": update.scam_count,
                        "score": update.new_score,
                        "updated_at": int(time.time() * 1000),
                    },
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist reputation: {e}")

    async def _threshold_update_loop(self):
        """Background loop for updating Thompson sampling thresholds."""
        while self._running:
            try:
                await asyncio.sleep(self.config.threshold_update_interval_secs)
                await self._update_thresholds()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Threshold update error: {e}")

    async def _update_thresholds(self):
        """Update Thompson sampling thresholds from outcomes."""
        logger.debug("Updating Thompson sampling thresholds")

        try:
            with self.engine.connect() as conn:
                # Get recent outcomes for threshold tuning
                result = conn.execute(
                    text("""
                        SELECT was_profitable, was_scam
                        FROM announcement_outcomes
                        WHERE recorded_at_ms > :since
                        AND was_traded = 1
                    """),
                    {"since": int((time.time() - 3600) * 1000)},  # Last hour
                ).fetchall()

                if not result:
                    return

                # Calculate success metrics
                total = len(result)
                profitable = sum(1 for r in result if r[0])
                not_scam = sum(1 for r in result if not r[1])

                # Update confidence threshold sampler
                conn.execute(
                    text("""
                        INSERT INTO thompson_sampling_state (
                            sampler_name, alpha, beta, updated_at_ms,
                            total_samples, total_successes
                        ) VALUES (
                            'confidence_threshold',
                            :alpha, :beta, :updated_at,
                            :total, :successes
                        )
                        ON CONFLICT(sampler_name) DO UPDATE SET
                            alpha = alpha + :new_alpha,
                            beta = beta + :new_beta,
                            updated_at_ms = :updated_at,
                            total_samples = total_samples + :total,
                            total_successes = total_successes + :successes
                    """),
                    {
                        "alpha": 1.0 + profitable,
                        "beta": 1.0 + (total - profitable),
                        "new_alpha": profitable,
                        "new_beta": total - profitable,
                        "updated_at": int(time.time() * 1000),
                        "total": total,
                        "successes": profitable,
                    },
                )

                # Update scam threshold sampler
                conn.execute(
                    text("""
                        INSERT INTO thompson_sampling_state (
                            sampler_name, alpha, beta, updated_at_ms,
                            total_samples, total_successes
                        ) VALUES (
                            'scam_threshold',
                            :alpha, :beta, :updated_at,
                            :total, :successes
                        )
                        ON CONFLICT(sampler_name) DO UPDATE SET
                            alpha = alpha + :new_alpha,
                            beta = beta + :new_beta,
                            updated_at_ms = :updated_at,
                            total_samples = total_samples + :total,
                            total_successes = total_successes + :successes
                    """),
                    {
                        "alpha": 1.0 + not_scam,
                        "beta": 1.0 + (total - not_scam),
                        "new_alpha": not_scam,
                        "new_beta": total - not_scam,
                        "updated_at": int(time.time() * 1000),
                        "total": total,
                        "successes": not_scam,
                    },
                )

                conn.commit()
                self.stats["threshold_updates"] += 1

                logger.info(
                    f"Updated thresholds: {profitable}/{total} profitable, "
                    f"{not_scam}/{total} not scam"
                )

        except Exception as e:
            logger.error(f"Failed to update thresholds: {e}")

    async def _retrain_loop(self):
        """Background loop for periodic classifier retraining."""
        while self._running:
            try:
                await asyncio.sleep(self.config.classifier_retrain_interval_secs)
                await self._retrain_classifier()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retrain error: {e}")

    async def _retrain_classifier(self):
        """Retrain classifier with new labeled data."""
        logger.debug("Checking for classifier retrain")

        try:
            with self.engine.connect() as conn:
                # Get labeled data count
                result = conn.execute(
                    text("""
                        SELECT COUNT(*)
                        FROM announcement_classifications c
                        JOIN announcement_outcomes o ON c.announcement_id = o.announcement_id
                        WHERE o.recorded_at_ms > :since
                    """),
                    {"since": int((time.time() - 86400) * 1000)},  # Last 24 hours
                ).fetchone()

                labeled_count = result[0] if result else 0

                if labeled_count < self.config.min_samples_for_retrain:
                    logger.debug(
                        f"Not enough labeled samples for retrain: {labeled_count} < "
                        f"{self.config.min_samples_for_retrain}"
                    )
                    return

                # Get training data
                training_data = conn.execute(
                    text("""
                        SELECT
                            ar.content,
                            ar.source_type,
                            c.category,
                            c.scam_probability,
                            o.was_scam,
                            o.was_profitable
                        FROM announcements_raw ar
                        JOIN announcement_classifications c ON ar.id = c.announcement_id
                        JOIN announcement_outcomes o ON ar.id = o.announcement_id
                        WHERE o.recorded_at_ms > :since
                        LIMIT 10000
                    """),
                    {"since": int((time.time() - 86400 * 7) * 1000)},  # Last 7 days
                ).fetchall()

                if len(training_data) < self.config.min_samples_for_retrain:
                    return

                # Log that we would retrain (actual retraining would be more complex)
                scam_labels = sum(1 for r in training_data if r[4])
                safe_labels = len(training_data) - scam_labels

                logger.info(
                    f"Would retrain classifier with {len(training_data)} samples "
                    f"({scam_labels} scam, {safe_labels} safe)"
                )

                self.stats["retrains"] += 1

        except Exception as e:
            logger.error(f"Failed to retrain classifier: {e}")

    def get_source_reputation(
        self,
        source_type: str,
        source_id: str,
    ) -> dict[str, Any] | None:
        """Get reputation for a source."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT source_id, source_type, total_announcements,
                               valid_trades, scam_count, reputation_score, updated_at_ms
                        FROM source_reputation
                        WHERE source_type = :source_type AND source_id = :source_id
                    """),
                    {"source_type": source_type, "source_id": source_id},
                ).fetchone()

                if result:
                    return {
                        "source_id": result[0],
                        "source_type": result[1],
                        "total_announcements": result[2],
                        "valid_trades": result[3],
                        "scam_count": result[4],
                        "reputation_score": result[5],
                        "updated_at_ms": result[6],
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get reputation: {e}")
            return None

    def get_thompson_state(self, sampler_name: str) -> dict[str, Any] | None:
        """Get Thompson sampling state."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT alpha, beta, updated_at_ms, total_samples, total_successes
                        FROM thompson_sampling_state
                        WHERE sampler_name = :name
                    """),
                    {"name": sampler_name},
                ).fetchone()

                if result:
                    alpha, beta = result[0], result[1]
                    return {
                        "alpha": alpha,
                        "beta": beta,
                        "mean": alpha / (alpha + beta),
                        "updated_at_ms": result[2],
                        "total_samples": result[3],
                        "total_successes": result[4],
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get Thompson state: {e}")
            return None

    def get_stats(self) -> dict[str, Any]:
        """Get learner statistics."""
        return self.stats.copy()
