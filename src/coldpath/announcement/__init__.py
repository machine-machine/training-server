"""
Announcement processing module for DEXY market data scanner.

This module provides ML-based announcement classification, entity extraction,
and on-chain verification for token launch signals.

Components:
- classifier: Text classification using Claude Sonnet for categorization
- extractor: Entity extraction using Claude Opus for complex cases
- verifier: On-chain verification and copy-paste scam detection
- scorer: Ensemble scoring combining classification + verification + reputation
- service: gRPC service implementation for HotPath communication
"""

from .classifier import (
    AnnouncementCategory,
    AnnouncementClassifier,
    ClassificationResult,
    Sentiment,
    Urgency,
)
from .extractor import (
    EntityExtractor,
    ExtractedEntities,
)
from .scorer import (
    EnsembleScorer,
    ScoringResult,
)
from .service import (
    AnnouncementProcessorService,
    start_announcement_service,
)
from .verifier import (
    OnChainVerifier,
    VerificationResult,
)

__all__ = [
    # Classifier
    "AnnouncementClassifier",
    "ClassificationResult",
    "AnnouncementCategory",
    "Sentiment",
    "Urgency",
    # Extractor
    "EntityExtractor",
    "ExtractedEntities",
    # Verifier
    "OnChainVerifier",
    "VerificationResult",
    # Scorer
    "EnsembleScorer",
    "ScoringResult",
    # Service
    "AnnouncementProcessorService",
    "start_announcement_service",
]
