"""
Announcement classifier using Claude Sonnet for ML-based categorization.

Classifies announcements into categories:
- real_launch: Genuine token launch announcement
- scam: Scam or malicious token
- rumor: Unverified rumor
- fud: Fear, uncertainty, doubt
- spam: Irrelevant spam

Features:
- Batched inference (up to 10 announcements per API call)
- Feature extraction (urgency, sentiment, scam patterns)
- Local fallback for when API is unavailable
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


class AnnouncementCategory(Enum):
    """Classification categories for announcements."""

    REAL_LAUNCH = "real_launch"
    SCAM = "scam"
    RUMOR = "rumor"
    FUD = "fud"
    SPAM = "spam"
    UNCLASSIFIED = "unclassified"


class Sentiment(Enum):
    """Sentiment levels."""

    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class Urgency(Enum):
    """Urgency levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMMEDIATE = "immediate"


@dataclass
class ClassificationResult:
    """Result of classifying an announcement."""

    announcement_id: str
    category: AnnouncementCategory
    confidence: float  # 0.0 - 1.0
    sentiment: Sentiment
    urgency: Urgency
    scam_probability: float  # 0.0 - 1.0
    latency_ms: int = 0
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassifierConfig:
    """Configuration for the announcement classifier."""

    # API settings
    anthropic_api_key: str | None = None
    model: str = "claude-sonnet-4-20250514"  # Claude Sonnet 4.5
    max_tokens: int = 1024
    temperature: float = 0.0  # Deterministic for classification

    # Batching settings
    batch_size: int = 10
    batch_timeout_ms: int = 100

    # Timeout settings
    api_timeout_secs: int = 30

    # Fallback settings
    enable_fallback: bool = True

    @classmethod
    def from_env(cls) -> "ClassifierConfig":
        """Load configuration from environment variables."""
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model=os.environ.get("CLASSIFIER_MODEL", "claude-sonnet-4-20250514"),
            batch_size=int(os.environ.get("ML_BATCH_SIZE", "10")),
            batch_timeout_ms=int(os.environ.get("ML_BATCH_TIMEOUT_MS", "100")),
            api_timeout_secs=int(os.environ.get("ML_API_TIMEOUT_SECS", "30")),
        )


# Scam indicators for rule-based fallback
SCAM_PATTERNS = [
    r"guaranteed\s+(profit|returns?|gains?)",
    r"\b100x\b|\b1000x\b",
    r"free\s+(money|tokens?|airdrop)",
    r"connect\s+wallet",
    r"send\s+(sol|eth|crypto)",
    r"claim\s+(now|your|free)",
    r"hurry\s+up",
    r"limited\s+time",
    r"don't\s+miss",
    r"last\s+chance",
]

# Legitimate launch indicators
LAUNCH_PATTERNS = [
    r"just\s+launched",
    r"launching\s+now",
    r"live\s+(now|on)",
    r"new\s+token",
    r"bonding\s+curve",
    r"pumpfun|pump\.fun",
    r"raydium",
    r"initial\s+liquidity",
]


class AnnouncementClassifier:
    """
    Classifier for announcement text using Claude Sonnet.

    Uses Claude Sonnet 4.5 for fast, cost-effective classification of
    announcements into categories with confidence scores.
    """

    def __init__(self, config: ClassifierConfig | None = None):
        """Initialize the classifier."""
        self.config = config or ClassifierConfig.from_env()

        # Initialize Anthropic client if API key available
        self.client: anthropic.AsyncAnthropic | None = None
        if self.config.anthropic_api_key:
            self.client = anthropic.AsyncAnthropic(
                api_key=self.config.anthropic_api_key,
                timeout=self.config.api_timeout_secs,
            )
            logger.info(f"Initialized Anthropic client with model {self.config.model}")
        else:
            logger.warning("No Anthropic API key, classifier will use local fallback")

        # Compile regex patterns
        self._scam_patterns = [re.compile(p, re.IGNORECASE) for p in SCAM_PATTERNS]
        self._launch_patterns = [re.compile(p, re.IGNORECASE) for p in LAUNCH_PATTERNS]

        # Statistics
        self.stats = {
            "total_classified": 0,
            "api_calls": 0,
            "api_errors": 0,
            "fallback_used": 0,
            "total_latency_ms": 0,
        }

    async def classify(
        self, announcement_id: str, content: str, source_type: str
    ) -> ClassificationResult:
        """Classify a single announcement."""
        results = await self.classify_batch([(announcement_id, content, source_type)])
        return results[0]

    async def classify_batch(
        self,
        announcements: list[tuple[str, str, str]],  # (id, content, source_type)
    ) -> list[ClassificationResult]:
        """
        Classify a batch of announcements.

        Args:
            announcements: List of (announcement_id, content, source_type) tuples

        Returns:
            List of ClassificationResult objects
        """
        start_time = time.time()

        # Try API classification if client available
        if self.client and len(announcements) > 0:
            try:
                results = await self._classify_with_api(announcements)
                latency_ms = int((time.time() - start_time) * 1000)
                self.stats["api_calls"] += 1
                self.stats["total_latency_ms"] += latency_ms

                # Add latency to results
                for result in results:
                    result.latency_ms = latency_ms

                return results

            except Exception as e:
                logger.warning(f"API classification failed: {e}")
                self.stats["api_errors"] += 1

        # Fallback to local classification
        if self.config.enable_fallback:
            self.stats["fallback_used"] += 1
            results = [
                self._classify_local(ann_id, content, source)
                for ann_id, content, source in announcements
            ]

            latency_ms = int((time.time() - start_time) * 1000)
            for result in results:
                result.latency_ms = latency_ms

            return results

        # Return unclassified if no fallback
        return [
            ClassificationResult(
                announcement_id=ann_id,
                category=AnnouncementCategory.UNCLASSIFIED,
                confidence=0.0,
                sentiment=Sentiment.NEUTRAL,
                urgency=Urgency.MEDIUM,
                scam_probability=0.5,
            )
            for ann_id, _, _ in announcements
        ]

    async def _classify_with_api(
        self,
        announcements: list[tuple[str, str, str]],
    ) -> list[ClassificationResult]:
        """Classify using Claude API."""
        # Build prompt
        prompt = self._build_classification_prompt(announcements)

        # Call API
        response = await self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse response
        response_text = response.content[0].text
        results = self._parse_classification_response(response_text, announcements)

        self.stats["total_classified"] += len(results)

        return results

    def _build_classification_prompt(
        self,
        announcements: list[tuple[str, str, str]],
    ) -> str:
        """Build the classification prompt for Claude."""
        announcements_text = "\n\n".join(
            [
                f"[{i + 1}] Source: {source}\nContent: {content}"
                for i, (_, content, source) in enumerate(announcements)
            ]
        )

        prompt = f"""Classify the following cryptocurrency/token announcements into categories.

For each announcement, provide:
1. Category: one of [real_launch, scam, rumor, fud, spam]
2. Confidence: 0.0-1.0 how confident you are
3. Sentiment: one of [negative, neutral, positive, very_positive]
4. Urgency: one of [low, medium, high, immediate]
5. Scam probability: 0.0-1.0

Categories:
- real_launch: Genuine new token launch announcement with verifiable details
- scam: Clear scam indicators (fake promises, phishing, etc.)
- rumor: Unverified claims about launches or prices
- fud: Fear, uncertainty, doubt messaging
- spam: Irrelevant or promotional spam

Announcements to classify:

{announcements_text}

Respond in this exact format for each announcement:
[1]
category: <category>
confidence: <0.0-1.0>
sentiment: <sentiment>
urgency: <urgency>
scam_probability: <0.0-1.0>

[2]
...
"""
        return prompt

    def _parse_classification_response(
        self,
        response_text: str,
        announcements: list[tuple[str, str, str]],
    ) -> list[ClassificationResult]:
        """Parse Claude's response into ClassificationResult objects."""
        results = []

        # Split by announcement number pattern
        sections = re.split(r"\[(\d+)\]", response_text)

        for i, (ann_id, _content, _source) in enumerate(announcements):
            # Default result
            result = ClassificationResult(
                announcement_id=ann_id,
                category=AnnouncementCategory.UNCLASSIFIED,
                confidence=0.0,
                sentiment=Sentiment.NEUTRAL,
                urgency=Urgency.MEDIUM,
                scam_probability=0.5,
            )

            try:
                # Find the section for this announcement
                section_idx = (i + 1) * 2 + 1 if len(sections) > (i + 1) * 2 else -1
                if section_idx > 0 and section_idx < len(sections):
                    section = sections[section_idx]

                    # Parse category
                    cat_match = re.search(r"category:\s*(\w+)", section, re.IGNORECASE)
                    if cat_match:
                        cat_str = cat_match.group(1).lower()
                        try:
                            result.category = AnnouncementCategory(cat_str)
                        except ValueError:
                            pass

                    # Parse confidence
                    conf_match = re.search(r"confidence:\s*([\d.]+)", section)
                    if conf_match:
                        result.confidence = min(1.0, max(0.0, float(conf_match.group(1))))

                    # Parse sentiment
                    sent_match = re.search(r"sentiment:\s*(\w+)", section, re.IGNORECASE)
                    if sent_match:
                        sent_str = sent_match.group(1).lower()
                        try:
                            result.sentiment = Sentiment(sent_str)
                        except ValueError:
                            pass

                    # Parse urgency
                    urg_match = re.search(r"urgency:\s*(\w+)", section, re.IGNORECASE)
                    if urg_match:
                        urg_str = urg_match.group(1).lower()
                        try:
                            result.urgency = Urgency(urg_str)
                        except ValueError:
                            pass

                    # Parse scam probability
                    scam_match = re.search(r"scam_probability:\s*([\d.]+)", section)
                    if scam_match:
                        result.scam_probability = min(1.0, max(0.0, float(scam_match.group(1))))

            except Exception as e:
                logger.warning(f"Failed to parse classification for {ann_id}: {e}")

            results.append(result)

        return results

    def _classify_local(
        self,
        announcement_id: str,
        content: str,
        source_type: str,
    ) -> ClassificationResult:
        """Local rule-based classification fallback."""
        content_lower = content.lower()

        # Extract features
        features = self._extract_features(content)

        # Count pattern matches
        scam_matches = sum(1 for p in self._scam_patterns if p.search(content_lower))
        launch_matches = sum(1 for p in self._launch_patterns if p.search(content_lower))

        # Determine category
        if scam_matches >= 2:
            category = AnnouncementCategory.SCAM
            confidence = min(0.9, 0.5 + scam_matches * 0.1)
            scam_prob = min(0.95, 0.5 + scam_matches * 0.15)
        elif source_type == "pumpfun_ws":
            # Direct PumpFun events are high confidence
            category = AnnouncementCategory.REAL_LAUNCH
            confidence = 0.95
            scam_prob = 0.1
        elif launch_matches >= 2:
            category = AnnouncementCategory.REAL_LAUNCH
            confidence = min(0.8, 0.4 + launch_matches * 0.15)
            scam_prob = max(0.1, 0.3 - launch_matches * 0.05)
        elif launch_matches >= 1:
            category = AnnouncementCategory.RUMOR
            confidence = 0.5
            scam_prob = 0.4
        else:
            category = AnnouncementCategory.UNCLASSIFIED
            confidence = 0.3
            scam_prob = 0.5

        # Determine urgency
        if any(word in content_lower for word in ["now", "live", "just"]):
            urgency = Urgency.IMMEDIATE
        elif any(word in content_lower for word in ["soon", "launching", "coming"]):
            urgency = Urgency.HIGH
        else:
            urgency = Urgency.MEDIUM

        # Determine sentiment
        sentiment = self._analyze_sentiment(content_lower)

        self.stats["total_classified"] += 1

        return ClassificationResult(
            announcement_id=announcement_id,
            category=category,
            confidence=confidence,
            sentiment=sentiment,
            urgency=urgency,
            scam_probability=scam_prob,
            features=features,
        )

    def _extract_features(self, content: str) -> dict[str, Any]:
        """Extract features from announcement content."""
        content_lower = content.lower()

        return {
            "length": len(content),
            "has_url": "http" in content_lower or "www." in content_lower,
            "has_contract_address": bool(re.search(r"[1-9A-HJ-NP-Za-km-z]{32,44}", content)),
            "has_dollar_symbol": "$" in content,
            "exclamation_count": content.count("!"),
            "caps_ratio": sum(1 for c in content if c.isupper()) / max(len(content), 1),
            "emoji_count": len(re.findall(r"[\U0001F300-\U0001F9FF]", content)),
            "scam_keyword_count": sum(1 for p in self._scam_patterns if p.search(content_lower)),
            "launch_keyword_count": sum(
                1 for p in self._launch_patterns if p.search(content_lower)
            ),
        }

    def _analyze_sentiment(self, content_lower: str) -> Sentiment:
        """Analyze sentiment of content."""
        positive_words = ["moon", "gem", "bullish", "amazing", "huge", "great", "best"]
        negative_words = ["dump", "rug", "scam", "avoid", "warning", "fake", "bad"]

        pos_count = sum(1 for w in positive_words if w in content_lower)
        neg_count = sum(1 for w in negative_words if w in content_lower)

        if neg_count > pos_count:
            return Sentiment.NEGATIVE
        elif pos_count > neg_count + 1:
            return Sentiment.VERY_POSITIVE
        elif pos_count > neg_count:
            return Sentiment.POSITIVE
        else:
            return Sentiment.NEUTRAL

    def get_stats(self) -> dict[str, Any]:
        """Get classifier statistics."""
        return self.stats.copy()
