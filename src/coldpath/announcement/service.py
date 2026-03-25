"""
gRPC service implementation for announcement processing.

Provides the AnnouncementProcessor service that handles:
- ProcessBatch: Full pipeline processing
- ClassifyBatch: Classification only
- ExtractEntities: Entity extraction only
- VerifyOnChain: On-chain verification
- GetSourceReputation: Source reputation lookup
- RecordOutcome: Outcome recording for learning
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from .classifier import AnnouncementClassifier, ClassifierConfig
from .extractor import EntityExtractor, ExtractorConfig
from .scorer import EnsembleScorer, ScorerConfig
from .verifier import OnChainVerifier, VerifierConfig

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for the announcement processing service."""

    # gRPC settings
    grpc_port: int = 50052
    max_workers: int = 10
    max_concurrent_rpcs: int = 100

    # Component configs
    classifier_config: ClassifierConfig | None = None
    extractor_config: ExtractorConfig | None = None
    verifier_config: VerifierConfig | None = None
    scorer_config: ScorerConfig | None = None

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Load configuration from environment."""
        return cls(
            grpc_port=int(os.environ.get("ANNOUNCEMENT_GRPC_PORT", "50052")),
            max_workers=int(os.environ.get("ANNOUNCEMENT_MAX_WORKERS", "10")),
            classifier_config=ClassifierConfig.from_env(),
            extractor_config=ExtractorConfig.from_env(),
            verifier_config=VerifierConfig.from_env(),
            scorer_config=ScorerConfig.from_env(),
        )


class SourceReputationStore:
    """Simple in-memory store for source reputation."""

    def __init__(self):
        self._reputations: dict[str, dict[str, Any]] = {}

    def get(self, source_id: str, source_type: str) -> dict[str, Any]:
        """Get reputation for a source."""
        key = f"{source_type}:{source_id}"
        if key not in self._reputations:
            # Default reputation
            self._reputations[key] = {
                "source_id": source_id,
                "source_type": source_type,
                "total_announcements": 0,
                "valid_trades": 0,
                "scam_count": 0,
                "avg_lead_time_ms": 0.0,
                "reputation_score": 0.5,
                "updated_at_ms": 0,
            }
        return self._reputations[key]

    def update(
        self,
        source_id: str,
        source_type: str,
        was_valid: bool,
        was_scam: bool,
        lead_time_ms: float | None = None,
    ):
        """Update reputation based on outcome."""
        rep = self.get(source_id, source_type)

        rep["total_announcements"] += 1
        if was_valid:
            rep["valid_trades"] += 1
        if was_scam:
            rep["scam_count"] += 1

        # Update average lead time
        if lead_time_ms is not None:
            n = rep["total_announcements"]
            rep["avg_lead_time_ms"] = (rep["avg_lead_time_ms"] * (n - 1) + lead_time_ms) / n

        # Recalculate reputation score
        if rep["total_announcements"] > 0:
            n = rep["total_announcements"]
            valid_rate = rep["valid_trades"] / n
            scam_rate = rep["scam_count"] / n
            base_score = valid_rate - scam_rate
            1.0 - (-n / 50.0).__class__.__module__  # exp approximation
            rep["reputation_score"] = max(0.0, min(1.0, 0.5 + base_score * 0.5))

        rep["updated_at_ms"] = int(time.time() * 1000)


class AnnouncementProcessorService:
    """
    gRPC service implementation for announcement processing.

    This service is called by the Rust HotPath to process announcements
    through the ML pipeline.
    """

    def __init__(self, config: ServiceConfig | None = None):
        """Initialize the service."""
        self.config = config or ServiceConfig.from_env()

        # Initialize components
        self.classifier = AnnouncementClassifier(self.config.classifier_config)
        self.extractor = EntityExtractor(self.config.extractor_config)
        self.verifier: OnChainVerifier | None = None  # Lazy init
        self.scorer = EnsembleScorer(self.config.scorer_config)

        # Reputation store
        self.reputation_store = SourceReputationStore()

        # Statistics
        self.stats = {
            "total_requests": 0,
            "process_batch_calls": 0,
            "classify_batch_calls": 0,
            "extract_entities_calls": 0,
            "verify_onchain_calls": 0,
            "total_latency_ms": 0,
        }

        logger.info("AnnouncementProcessorService initialized")

    async def _ensure_verifier(self):
        """Ensure verifier is initialized."""
        if self.verifier is None:
            self.verifier = OnChainVerifier(self.config.verifier_config)
            await self.verifier.__aenter__()

    async def process_batch(
        self,
        announcements: list[dict[str, Any]],
        batch_id: str,
    ) -> dict[str, Any]:
        """
        Process a batch of announcements through the full pipeline.

        Args:
            announcements: List of announcement dicts with keys:
                - id, source_type, source_id, content, timestamp_ms, priority, metadata_json
            batch_id: Unique batch identifier

        Returns:
            Dict with processed results
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        self.stats["process_batch_calls"] += 1

        await self._ensure_verifier()

        results = []

        for ann in announcements:
            ann_id = ann.get("id", "")
            content = ann.get("content", "")
            source_type = ann.get("source_type", "unknown")
            source_id = ann.get("source_id", "")
            metadata_json = ann.get("metadata_json", "{}")

            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except json.JSONDecodeError:
                metadata = {}

            # 1. Classify
            classification = await self.classifier.classify(ann_id, content, source_type)

            # 2. Extract entities
            entities = await self.extractor.extract(ann_id, content, metadata)

            # 3. Verify on-chain (if we have a contract address)
            verification = None
            if entities.contract_address:
                verification = await self.verifier.verify(
                    contract_address=entities.contract_address,
                    pool_address=entities.pool_address,
                    token_name=entities.token_name,
                    token_symbol=entities.token_symbol,
                )

            # 4. Get source reputation
            rep = self.reputation_store.get(source_id, source_type)
            source_reputation = rep["reputation_score"]

            # 5. Score and decide
            scoring_result = self.scorer.score(
                announcement_id=ann_id,
                classification=classification,
                entities=entities,
                verification=verification,
                source_reputation=source_reputation,
            )

            # Build result
            result = {
                "announcement_id": ann_id,
                "classification": {
                    "category": classification.category.value,
                    "confidence": classification.confidence,
                    "sentiment": classification.sentiment.value,
                    "urgency": classification.urgency.value,
                    "scam_probability": classification.scam_probability,
                    "latency_ms": classification.latency_ms,
                },
                "entities": {
                    "token_name": entities.token_name,
                    "token_symbol": entities.token_symbol,
                    "contract_address": entities.contract_address,
                    "pool_address": entities.pool_address,
                    "launch_time": entities.launch_time,
                    "mentioned_market_cap": entities.mentioned_market_cap,
                    "twitter_handle": entities.twitter_handle,
                    "telegram_channel": entities.telegram_channel,
                    "website_url": entities.website_url,
                    "confidence": entities.confidence,
                    "latency_ms": entities.latency_ms,
                },
                "verification": None,
                "source_reputation": source_reputation,
                "combined_confidence": scoring_result.combined_confidence,
                "should_accept": scoring_result.should_accept,
                "rejection_reason": scoring_result.rejection_reason,
            }

            if verification:
                result["verification"] = {
                    "contract_address": verification.contract_address,
                    "contract_exists": verification.contract_exists,
                    "is_valid_token": verification.is_valid_token,
                    "mint_authority_enabled": verification.mint_authority_enabled,
                    "freeze_authority_enabled": verification.freeze_authority_enabled,
                    "supply": verification.supply,
                    "copy_paste_score": verification.copy_paste_score,
                    "similar_to": verification.similar_to,
                    "verified_at_ms": verification.verified_at_ms,
                    "latency_ms": verification.latency_ms,
                }

            results.append(result)

        total_latency_ms = int((time.time() - start_time) * 1000)
        self.stats["total_latency_ms"] += total_latency_ms

        return {
            "announcements": results,
            "total_latency_ms": total_latency_ms,
            "batch_id": batch_id,
        }

    async def classify_batch(
        self,
        announcements: list[dict[str, Any]],
        batch_id: str,
    ) -> dict[str, Any]:
        """
        Classify a batch of announcements (fast path).

        Args:
            announcements: List of announcement dicts
            batch_id: Unique batch identifier

        Returns:
            Dict with classification results
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        self.stats["classify_batch_calls"] += 1

        # Build batch
        batch = [
            (ann.get("id", ""), ann.get("content", ""), ann.get("source_type", "unknown"))
            for ann in announcements
        ]

        # Classify
        results = await self.classifier.classify_batch(batch)

        # Format results
        formatted = [
            {
                "announcement_id": r.announcement_id,
                "category": r.category.value,
                "confidence": r.confidence,
                "sentiment": r.sentiment.value,
                "urgency": r.urgency.value,
                "scam_probability": r.scam_probability,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ]

        total_latency_ms = int((time.time() - start_time) * 1000)
        self.stats["total_latency_ms"] += total_latency_ms

        return {
            "results": formatted,
            "total_latency_ms": total_latency_ms,
            "batch_id": batch_id,
        }

    async def extract_entities(
        self,
        announcement: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract entities from a single announcement.

        Args:
            announcement: Announcement dict

        Returns:
            Dict with extracted entities
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        self.stats["extract_entities_calls"] += 1

        ann_id = announcement.get("id", "")
        content = announcement.get("content", "")
        metadata_json = announcement.get("metadata_json", "{}")

        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except json.JSONDecodeError:
            metadata = {}

        entities = await self.extractor.extract(ann_id, content, metadata)

        total_latency_ms = int((time.time() - start_time) * 1000)
        self.stats["total_latency_ms"] += total_latency_ms

        return {
            "announcement_id": entities.announcement_id,
            "token_name": entities.token_name,
            "token_symbol": entities.token_symbol,
            "contract_address": entities.contract_address,
            "pool_address": entities.pool_address,
            "launch_time": entities.launch_time,
            "mentioned_market_cap": entities.mentioned_market_cap,
            "twitter_handle": entities.twitter_handle,
            "telegram_channel": entities.telegram_channel,
            "website_url": entities.website_url,
            "confidence": entities.confidence,
            "latency_ms": total_latency_ms,
        }

    async def verify_onchain(
        self,
        contract_address: str,
        pool_address: str | None = None,
        token_name: str | None = None,
        token_symbol: str | None = None,
    ) -> dict[str, Any]:
        """
        Verify token on-chain.

        Args:
            contract_address: Token mint address
            pool_address: Pool/bonding curve address (optional)
            token_name: Token name for copy-paste detection
            token_symbol: Token symbol for copy-paste detection

        Returns:
            Dict with verification results
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        self.stats["verify_onchain_calls"] += 1

        await self._ensure_verifier()

        verification = await self.verifier.verify(
            contract_address=contract_address,
            pool_address=pool_address,
            token_name=token_name,
            token_symbol=token_symbol,
        )

        total_latency_ms = int((time.time() - start_time) * 1000)
        self.stats["total_latency_ms"] += total_latency_ms

        return {
            "contract_address": verification.contract_address,
            "contract_exists": verification.contract_exists,
            "is_valid_token": verification.is_valid_token,
            "mint_authority_enabled": verification.mint_authority_enabled,
            "freeze_authority_enabled": verification.freeze_authority_enabled,
            "supply": verification.supply,
            "copy_paste_score": verification.copy_paste_score,
            "similar_to": verification.similar_to,
            "verified_at_ms": verification.verified_at_ms,
            "latency_ms": total_latency_ms,
        }

    def get_source_reputation(
        self,
        source_id: str,
        source_type: str,
    ) -> dict[str, Any]:
        """Get source reputation."""
        return self.reputation_store.get(source_id, source_type)

    def record_outcome(
        self,
        announcement_id: str,
        signal_id: str,
        mint: str,
        source_id: str,
        source_type: str,
        was_traded: bool,
        was_profitable: bool,
        pnl_sol: float | None,
        was_scam: bool,
        lead_time_ms: int,
    ):
        """Record outcome for learning."""
        # Update reputation store
        self.reputation_store.update(
            source_id=source_id,
            source_type=source_type,
            was_valid=was_profitable,
            was_scam=was_scam,
            lead_time_ms=float(lead_time_ms) if lead_time_ms else None,
        )

        # Update scorer Thompson sampling
        self.scorer.record_outcome(
            announcement_id=announcement_id,
            was_profitable=was_profitable,
            was_scam=was_scam,
        )

        logger.info(
            f"Recorded outcome: announcement={announcement_id}, "
            f"profitable={was_profitable}, scam={was_scam}"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "service": self.stats.copy(),
            "classifier": self.classifier.get_stats(),
            "extractor": self.extractor.get_stats(),
            "verifier": self.verifier.get_stats() if self.verifier else {},
            "scorer": self.scorer.get_stats(),
        }

    async def shutdown(self):
        """Shutdown the service."""
        if self.verifier:
            await self.verifier.__aexit__(None, None, None)


async def start_announcement_service(
    config: ServiceConfig | None = None,
) -> AnnouncementProcessorService:
    """
    Start the announcement processing service.

    Args:
        config: Service configuration

    Returns:
        Running service instance
    """
    service = AnnouncementProcessorService(config)
    logger.info(f"Announcement service started on port {service.config.grpc_port}")
    return service
