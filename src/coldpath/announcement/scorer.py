"""
Ensemble scorer for announcement signals.

Combines:
- Classification confidence
- Entity extraction confidence
- On-chain verification results
- Source reputation score

Uses Thompson sampling for calibrated threshold tuning.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .classifier import AnnouncementCategory, ClassificationResult, Sentiment, Urgency
from .extractor import ExtractedEntities
from .verifier import VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Result of ensemble scoring."""

    announcement_id: str
    combined_confidence: float  # 0.0 - 1.0
    should_accept: bool
    rejection_reason: str | None = None

    # Component scores
    classification_score: float = 0.0
    entity_score: float = 0.0
    verification_score: float = 0.0
    reputation_score: float = 0.0

    # Risk indicators
    scam_risk: float = 0.0
    copy_paste_risk: float = 0.0
    authority_risk: float = 0.0

    # Metadata
    latency_ms: int = 0


@dataclass
class ScorerConfig:
    """Configuration for ensemble scorer."""

    # Acceptance thresholds
    min_confidence: float = 0.5
    max_scam_probability: float = 0.3
    max_copy_paste_score: float = 0.7

    # Component weights for ensemble
    classification_weight: float = 0.30
    entity_weight: float = 0.20
    verification_weight: float = 0.35
    reputation_weight: float = 0.15

    # Thompson sampling parameters
    thompson_alpha: float = 1.0
    thompson_beta: float = 1.0
    enable_thompson: bool = True

    @classmethod
    def from_env(cls) -> "ScorerConfig":
        """Load configuration from environment."""
        return cls(
            min_confidence=float(os.environ.get("ANNOUNCEMENT_MIN_CONFIDENCE", "0.5")),
            max_scam_probability=float(os.environ.get("MAX_SCAM_PROBABILITY", "0.3")),
            max_copy_paste_score=float(os.environ.get("MAX_COPY_PASTE_SCORE", "0.7")),
        )


class ThompsonSampler:
    """Thompson sampling for threshold calibration."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """Initialize with Beta prior parameters."""
        self.alpha = alpha  # Successes
        self.beta = beta  # Failures

    def sample(self) -> float:
        """Sample a threshold from the posterior distribution."""
        return np.random.beta(self.alpha, self.beta)

    def update(self, was_success: bool):
        """Update posterior based on outcome."""
        if was_success:
            self.alpha += 1
        else:
            self.beta += 1

    def mean(self) -> float:
        """Get the posterior mean (expected success probability)."""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Get the posterior variance."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total**2 * (total + 1))


class EnsembleScorer:
    """
    Ensemble scorer combining multiple signals for final decision.

    Weights and combines:
    - Classification confidence and category
    - Entity extraction confidence
    - On-chain verification results
    - Source reputation

    Uses Thompson sampling to calibrate acceptance thresholds
    based on historical outcomes.
    """

    def __init__(self, config: ScorerConfig | None = None):
        """Initialize the scorer."""
        self.config = config or ScorerConfig.from_env()

        # Thompson samplers for threshold tuning
        self.threshold_sampler = ThompsonSampler(
            alpha=self.config.thompson_alpha,
            beta=self.config.thompson_beta,
        )
        self.scam_threshold_sampler = ThompsonSampler(
            alpha=self.config.thompson_alpha,
            beta=self.config.thompson_beta,
        )

        # Statistics
        self.stats = {
            "total_scored": 0,
            "accepted": 0,
            "rejected": 0,
            "outcomes_recorded": 0,
            "total_latency_ms": 0,
        }

        # Rejection reason counters
        self.rejection_reasons: dict[str, int] = {}

    def score(
        self,
        announcement_id: str,
        classification: ClassificationResult,
        entities: ExtractedEntities,
        verification: VerificationResult | None,
        source_reputation: float,
    ) -> ScoringResult:
        """
        Score an announcement and decide acceptance.

        Args:
            announcement_id: Unique announcement ID
            classification: Classification result from classifier
            entities: Extracted entities from extractor
            verification: On-chain verification result (optional)
            source_reputation: Source reputation score (0.0 - 1.0)

        Returns:
            ScoringResult with combined score and decision
        """
        start_time = time.time()

        # Calculate component scores
        classification_score = self._score_classification(classification)
        entity_score = self._score_entities(entities)
        verification_score = self._score_verification(verification) if verification else 0.5
        reputation_score = source_reputation

        # Calculate risk indicators
        scam_risk = classification.scam_probability
        copy_paste_risk = verification.copy_paste_score if verification else 0.0
        authority_risk = self._calculate_authority_risk(verification)

        # Weighted ensemble score
        combined = (
            classification_score * self.config.classification_weight
            + entity_score * self.config.entity_weight
            + verification_score * self.config.verification_weight
            + reputation_score * self.config.reputation_weight
        )

        # Normalize weights
        total_weight = (
            self.config.classification_weight
            + self.config.entity_weight
            + self.config.verification_weight
            + self.config.reputation_weight
        )
        combined = combined / total_weight

        # Get dynamic threshold using Thompson sampling
        if self.config.enable_thompson:
            confidence_threshold = max(
                self.config.min_confidence,
                self.threshold_sampler.sample() * 0.5 + 0.25,  # Scale to [0.25, 0.75]
            )
            scam_threshold = min(
                self.config.max_scam_probability,
                self.scam_threshold_sampler.sample() * 0.4 + 0.1,  # Scale to [0.1, 0.5]
            )
        else:
            confidence_threshold = self.config.min_confidence
            scam_threshold = self.config.max_scam_probability

        # Decision logic
        should_accept, rejection_reason = self._make_decision(
            classification=classification,
            verification=verification,
            combined_confidence=combined,
            scam_risk=scam_risk,
            copy_paste_risk=copy_paste_risk,
            authority_risk=authority_risk,
            confidence_threshold=confidence_threshold,
            scam_threshold=scam_threshold,
        )

        # Update stats
        self.stats["total_scored"] += 1
        if should_accept:
            self.stats["accepted"] += 1
        else:
            self.stats["rejected"] += 1
            if rejection_reason:
                self.rejection_reasons[rejection_reason] = (
                    self.rejection_reasons.get(rejection_reason, 0) + 1
                )

        latency_ms = int((time.time() - start_time) * 1000)
        self.stats["total_latency_ms"] += latency_ms

        return ScoringResult(
            announcement_id=announcement_id,
            combined_confidence=combined,
            should_accept=should_accept,
            rejection_reason=rejection_reason,
            classification_score=classification_score,
            entity_score=entity_score,
            verification_score=verification_score,
            reputation_score=reputation_score,
            scam_risk=scam_risk,
            copy_paste_risk=copy_paste_risk,
            authority_risk=authority_risk,
            latency_ms=latency_ms,
        )

    def _score_classification(self, classification: ClassificationResult) -> float:
        """Calculate score from classification result."""
        base_score = classification.confidence

        # Category adjustments
        if classification.category == AnnouncementCategory.REAL_LAUNCH:
            category_multiplier = 1.0
        elif classification.category == AnnouncementCategory.RUMOR:
            category_multiplier = 0.7
        elif classification.category == AnnouncementCategory.SCAM:
            category_multiplier = 0.1
        elif classification.category == AnnouncementCategory.FUD:
            category_multiplier = 0.2
        elif classification.category == AnnouncementCategory.SPAM:
            category_multiplier = 0.05
        else:
            category_multiplier = 0.4

        # Sentiment adjustments
        if classification.sentiment == Sentiment.VERY_POSITIVE:
            sentiment_boost = 0.1
        elif classification.sentiment == Sentiment.POSITIVE:
            sentiment_boost = 0.05
        elif classification.sentiment == Sentiment.NEGATIVE:
            sentiment_boost = -0.1
        else:
            sentiment_boost = 0.0

        # Urgency adjustments
        if classification.urgency == Urgency.IMMEDIATE:
            urgency_boost = 0.1
        elif classification.urgency == Urgency.HIGH:
            urgency_boost = 0.05
        else:
            urgency_boost = 0.0

        # Combine
        score = base_score * category_multiplier + sentiment_boost + urgency_boost
        return min(1.0, max(0.0, score))

    def _score_entities(self, entities: ExtractedEntities) -> float:
        """Calculate score from extracted entities."""
        score = entities.confidence

        # Bonus for having contract address
        if entities.contract_address:
            score += 0.2

        # Bonus for having symbol
        if entities.token_symbol:
            score += 0.1

        # Bonus for social links
        if entities.twitter_handle or entities.telegram_channel:
            score += 0.05

        return min(1.0, score)

    def _score_verification(self, verification: VerificationResult) -> float:
        """Calculate score from verification result."""
        if not verification.contract_exists:
            return 0.0

        score = 0.5  # Base for existing contract

        if verification.is_valid_token:
            score += 0.3

        # Authority checks
        if verification.mint_authority_enabled is False:
            score += 0.1
        elif verification.mint_authority_enabled is True:
            score -= 0.2

        if verification.freeze_authority_enabled is False:
            score += 0.05
        elif verification.freeze_authority_enabled is True:
            score -= 0.1

        # Copy-paste penalty
        if verification.copy_paste_score > 0.5:
            score -= verification.copy_paste_score * 0.5

        return min(1.0, max(0.0, score))

    def _calculate_authority_risk(self, verification: VerificationResult | None) -> float:
        """Calculate risk from token authorities."""
        if not verification:
            return 0.5  # Unknown

        risk = 0.0

        if verification.mint_authority_enabled is True:
            risk += 0.5

        if verification.freeze_authority_enabled is True:
            risk += 0.3

        return min(1.0, risk)

    def _make_decision(
        self,
        classification: ClassificationResult,
        verification: VerificationResult | None,
        combined_confidence: float,
        scam_risk: float,
        copy_paste_risk: float,
        authority_risk: float,
        confidence_threshold: float,
        scam_threshold: float,
    ) -> tuple[bool, str | None]:
        """
        Make acceptance decision based on all factors.

        Returns:
            Tuple of (should_accept, rejection_reason)
        """
        # Check category
        if classification.category.value not in ["real_launch", "rumor"]:
            return (False, f"Non-tradeable category: {classification.category.value}")

        # Check confidence
        if combined_confidence < confidence_threshold:
            return (
                False,
                f"Low confidence: {combined_confidence:.2f} < {confidence_threshold:.2f}",
            )

        # Check scam probability
        if scam_risk >= scam_threshold:
            return (False, f"High scam risk: {scam_risk:.2f}")

        # Check copy-paste score
        if copy_paste_risk >= self.config.max_copy_paste_score:
            similar = verification.similar_to if verification else "unknown"
            return (False, f"Copy-paste scam detected (similar to {similar})")

        # Check verification if available
        if verification:
            if not verification.contract_exists:
                return (False, "Contract does not exist on-chain")

            if not verification.is_valid_token:
                return (False, "Invalid token contract")

            if verification.mint_authority_enabled is True:
                return (False, "Mint authority enabled (rug risk)")

            if verification.freeze_authority_enabled is True:
                return (False, "Freeze authority enabled")

        # All checks passed
        return (True, None)

    def record_outcome(
        self,
        announcement_id: str,
        was_profitable: bool,
        was_scam: bool,
    ):
        """
        Record outcome for Thompson sampling updates.

        Args:
            announcement_id: Announcement ID
            was_profitable: Whether the trade was profitable
            was_scam: Whether it turned out to be a scam
        """
        self.stats["outcomes_recorded"] += 1

        # Update confidence threshold sampler
        # Success = profitable trade, Failure = unprofitable
        self.threshold_sampler.update(was_profitable)

        # Update scam threshold sampler
        # Success = correctly identified non-scam, Failure = missed scam
        self.scam_threshold_sampler.update(not was_scam)

        logger.debug(
            f"Updated Thompson samplers: "
            f"confidence_mean={self.threshold_sampler.mean():.3f}, "
            f"scam_threshold_mean={self.scam_threshold_sampler.mean():.3f}"
        )

    def get_current_thresholds(self) -> dict[str, float]:
        """Get current calibrated thresholds."""
        return {
            "confidence_mean": self.threshold_sampler.mean(),
            "confidence_variance": self.threshold_sampler.variance(),
            "scam_threshold_mean": self.scam_threshold_sampler.mean(),
            "scam_threshold_variance": self.scam_threshold_sampler.variance(),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get scorer statistics."""
        stats = self.stats.copy()
        stats["rejection_reasons"] = self.rejection_reasons.copy()
        stats["thresholds"] = self.get_current_thresholds()
        return stats
