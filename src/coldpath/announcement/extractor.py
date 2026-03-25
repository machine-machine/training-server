"""
Entity extractor using Claude Opus for complex extraction.

Extracts structured information from announcement text:
- Token name and symbol
- Contract address (mint)
- Pool/bonding curve address
- Launch time
- Social links (Twitter, Telegram, website)

Uses regex for fast high-precision extraction with Claude Opus
fallback for complex/ambiguous cases.
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntities:
    """Extracted entities from an announcement."""

    announcement_id: str
    token_name: str | None = None
    token_symbol: str | None = None
    contract_address: str | None = None  # Mint address
    pool_address: str | None = None  # Bonding curve or pool
    launch_time: int | None = None  # Unix timestamp ms
    mentioned_market_cap: float | None = None
    twitter_handle: str | None = None
    telegram_channel: str | None = None
    website_url: str | None = None
    confidence: float = 0.0
    latency_ms: int = 0
    extraction_method: str = "regex"  # "regex" or "llm"


@dataclass
class ExtractorConfig:
    """Configuration for entity extractor."""

    # API settings
    anthropic_api_key: str | None = None
    model: str = "claude-opus-4-5-20251101"  # Claude Opus 4.5 for complex extraction
    max_tokens: int = 1024
    temperature: float = 0.0

    # Timeout settings
    api_timeout_secs: int = 30

    # Strategy settings
    use_llm_for_complex: bool = True  # Use LLM for ambiguous cases
    llm_complexity_threshold: float = 0.5  # Below this confidence, use LLM

    @classmethod
    def from_env(cls) -> "ExtractorConfig":
        """Load configuration from environment."""
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model=os.environ.get("EXTRACTOR_MODEL", "claude-opus-4-5-20251101"),
            api_timeout_secs=int(os.environ.get("ML_API_TIMEOUT_SECS", "30")),
        )


# Solana address pattern (Base58, 32-44 chars)
SOLANA_ADDRESS_PATTERN = re.compile(r"\b([1-9A-HJ-NP-Za-km-z]{32,44})\b")

# Token symbol pattern ($XXX or XXX)
SYMBOL_PATTERN = re.compile(r"\$([A-Za-z][A-Za-z0-9]{1,9})\b")

# Twitter handle pattern
TWITTER_PATTERN = re.compile(r"@([A-Za-z_][A-Za-z0-9_]{0,14})\b")

# Telegram pattern
TELEGRAM_PATTERN = re.compile(r"t\.me/([A-Za-z0-9_]{5,32})\b|telegram\.me/([A-Za-z0-9_]{5,32})\b")

# URL pattern
URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)

# Market cap pattern
MARKET_CAP_PATTERN = re.compile(
    r"(?:market\s*cap|mcap|mc)[:\s]*\$?([\d,.]+)\s*(k|m|b)?(?:illion)?", re.IGNORECASE
)


class EntityExtractor:
    """
    Entity extractor for announcement text.

    Uses high-precision regex patterns for fast extraction,
    with Claude Opus fallback for complex/ambiguous cases.
    """

    def __init__(self, config: ExtractorConfig | None = None):
        """Initialize the extractor."""
        self.config = config or ExtractorConfig.from_env()

        # Initialize Anthropic client if API key available
        self.client: anthropic.AsyncAnthropic | None = None
        if self.config.anthropic_api_key:
            self.client = anthropic.AsyncAnthropic(
                api_key=self.config.anthropic_api_key,
                timeout=self.config.api_timeout_secs,
            )
            logger.info(f"Initialized Anthropic client with model {self.config.model}")
        else:
            logger.warning("No Anthropic API key, extractor will use regex only")

        # Statistics
        self.stats = {
            "total_extracted": 0,
            "regex_only": 0,
            "llm_used": 0,
            "api_errors": 0,
            "total_latency_ms": 0,
        }

    async def extract(
        self,
        announcement_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ExtractedEntities:
        """
        Extract entities from announcement.

        First tries fast regex extraction. If confidence is low and LLM is
        enabled, falls back to Claude Opus for complex extraction.
        """
        start_time = time.time()

        # First, check metadata (e.g., from PumpFun direct events)
        if metadata:
            result = self._extract_from_metadata(announcement_id, metadata)
            if result.confidence >= 0.9:
                result.latency_ms = int((time.time() - start_time) * 1000)
                self.stats["total_extracted"] += 1
                return result

        # Try regex extraction
        result = self._extract_with_regex(announcement_id, content)

        # If low confidence and LLM available, use LLM
        if (
            self.config.use_llm_for_complex
            and result.confidence < self.config.llm_complexity_threshold
            and self.client
        ):
            try:
                result = await self._extract_with_llm(announcement_id, content, result)
                self.stats["llm_used"] += 1
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")
                self.stats["api_errors"] += 1
        else:
            self.stats["regex_only"] += 1

        result.latency_ms = int((time.time() - start_time) * 1000)
        self.stats["total_extracted"] += 1
        self.stats["total_latency_ms"] += result.latency_ms

        return result

    def _extract_from_metadata(
        self,
        announcement_id: str,
        metadata: dict[str, Any],
    ) -> ExtractedEntities:
        """Extract from structured metadata (e.g., PumpFun events)."""
        return ExtractedEntities(
            announcement_id=announcement_id,
            token_name=metadata.get("name"),
            token_symbol=metadata.get("symbol"),
            contract_address=metadata.get("mint"),
            pool_address=metadata.get("bonding_curve") or metadata.get("pool"),
            launch_time=metadata.get("received_at_ms"),
            mentioned_market_cap=metadata.get("market_cap_usd"),
            twitter_handle=metadata.get("twitter"),
            telegram_channel=metadata.get("telegram"),
            website_url=metadata.get("website") or metadata.get("uri"),
            confidence=1.0,  # Direct from source
            extraction_method="metadata",
        )

    def _extract_with_regex(
        self,
        announcement_id: str,
        content: str,
    ) -> ExtractedEntities:
        """Extract entities using regex patterns."""
        result = ExtractedEntities(
            announcement_id=announcement_id,
            extraction_method="regex",
        )

        confidence_scores = []

        # Extract Solana addresses
        addresses = SOLANA_ADDRESS_PATTERN.findall(content)
        if addresses:
            # First address is usually the mint
            result.contract_address = addresses[0]
            confidence_scores.append(0.8)

            # Second address might be the pool
            if len(addresses) > 1:
                result.pool_address = addresses[1]

        # Extract token symbol
        symbols = SYMBOL_PATTERN.findall(content)
        if symbols:
            result.token_symbol = symbols[0].upper()
            confidence_scores.append(0.7)
        else:
            # Try to find ALL CAPS words that look like symbols
            caps_words = re.findall(r"\b([A-Z]{2,10})\b", content)
            if caps_words:
                # Filter out common words
                common_words = {"THE", "AND", "FOR", "NEW", "NOW", "GET", "BUY", "SOL"}
                symbols = [w for w in caps_words if w not in common_words]
                if symbols:
                    result.token_symbol = symbols[0]
                    confidence_scores.append(0.4)

        # Extract token name (usually before symbol or in quotes)
        name_match = re.search(r'"([^"]{2,50})"', content)
        if name_match:
            result.token_name = name_match.group(1)
            confidence_scores.append(0.6)
        elif result.token_symbol:
            # Try to find name before symbol
            name_match = re.search(
                rf"([A-Za-z][A-Za-z\s]{{1,30}})\s*\${result.token_symbol}", content, re.IGNORECASE
            )
            if name_match:
                result.token_name = name_match.group(1).strip()
                confidence_scores.append(0.5)

        # Extract Twitter handle
        twitter_matches = TWITTER_PATTERN.findall(content)
        if twitter_matches:
            # Filter out common words
            handles = [h for h in twitter_matches if len(h) > 2]
            if handles:
                result.twitter_handle = handles[0]
                confidence_scores.append(0.6)

        # Extract Telegram channel
        telegram_matches = TELEGRAM_PATTERN.findall(content)
        for match in telegram_matches:
            channel = match[0] or match[1]
            if channel:
                result.telegram_channel = channel
                confidence_scores.append(0.6)
                break

        # Extract website URL
        urls = URL_PATTERN.findall(content)
        for url in urls:
            # Skip social media links
            if not any(x in url.lower() for x in ["twitter", "t.co", "telegram", "t.me"]):
                result.website_url = url
                confidence_scores.append(0.5)
                break

        # Extract market cap
        mcap_match = MARKET_CAP_PATTERN.search(content)
        if mcap_match:
            value_str = mcap_match.group(1).replace(",", "")
            try:
                value = float(value_str)
                multiplier = mcap_match.group(2)
                if multiplier:
                    multiplier = multiplier.lower()
                    if multiplier == "k":
                        value *= 1_000
                    elif multiplier == "m":
                        value *= 1_000_000
                    elif multiplier == "b":
                        value *= 1_000_000_000
                result.mentioned_market_cap = value
                confidence_scores.append(0.5)
            except ValueError:
                pass

        # Calculate overall confidence
        if confidence_scores:
            result.confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            result.confidence = 0.1

        return result

    async def _extract_with_llm(
        self,
        announcement_id: str,
        content: str,
        regex_result: ExtractedEntities,
    ) -> ExtractedEntities:
        """Extract entities using Claude Opus for complex cases."""
        prompt = f"""Extract structured information from this cryptocurrency/token announcement.

Announcement:
{content}

Already extracted (may be incomplete or incorrect):
- Token Name: {regex_result.token_name or "Unknown"}
- Token Symbol: {regex_result.token_symbol or "Unknown"}
- Contract Address: {regex_result.contract_address or "Unknown"}
- Pool Address: {regex_result.pool_address or "Unknown"}
- Market Cap: {regex_result.mentioned_market_cap or "Unknown"}
- Twitter: {regex_result.twitter_handle or "Unknown"}
- Telegram: {regex_result.telegram_channel or "Unknown"}
- Website: {regex_result.website_url or "Unknown"}

Please provide the correct values, or "null" if not found:
token_name: <name or null>
token_symbol: <symbol or null>
contract_address: <Solana address or null>
pool_address: <pool/bonding curve address or null>
market_cap: <numeric value in USD or null>
twitter: <handle without @ or null>
telegram: <channel name or null>
website: <URL or null>
confidence: <0.0-1.0 how confident in the extraction>
"""

        response = await self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text
        return self._parse_llm_response(announcement_id, response_text, regex_result)

    def _parse_llm_response(
        self,
        announcement_id: str,
        response_text: str,
        fallback: ExtractedEntities,
    ) -> ExtractedEntities:
        """Parse LLM response into ExtractedEntities."""
        result = ExtractedEntities(
            announcement_id=announcement_id,
            extraction_method="llm",
        )

        def extract_value(pattern: str) -> str | None:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value.lower() not in ["null", "none", "unknown", ""]:
                    return value
            return None

        # Parse each field
        result.token_name = extract_value(r"token_name:\s*(.+?)(?:\n|$)") or fallback.token_name
        result.token_symbol = (
            extract_value(r"token_symbol:\s*(.+?)(?:\n|$)") or fallback.token_symbol
        )
        result.contract_address = (
            extract_value(r"contract_address:\s*(.+?)(?:\n|$)") or fallback.contract_address
        )
        result.pool_address = (
            extract_value(r"pool_address:\s*(.+?)(?:\n|$)") or fallback.pool_address
        )
        result.twitter_handle = (
            extract_value(r"twitter:\s*(.+?)(?:\n|$)") or fallback.twitter_handle
        )
        result.telegram_channel = (
            extract_value(r"telegram:\s*(.+?)(?:\n|$)") or fallback.telegram_channel
        )
        result.website_url = extract_value(r"website:\s*(.+?)(?:\n|$)") or fallback.website_url

        # Parse market cap
        mcap_str = extract_value(r"market_cap:\s*([\d,.]+)")
        if mcap_str:
            try:
                result.mentioned_market_cap = float(mcap_str.replace(",", ""))
            except ValueError:
                result.mentioned_market_cap = fallback.mentioned_market_cap
        else:
            result.mentioned_market_cap = fallback.mentioned_market_cap

        # Parse confidence
        conf_match = re.search(r"confidence:\s*([\d.]+)", response_text)
        if conf_match:
            result.confidence = min(1.0, max(0.0, float(conf_match.group(1))))
        else:
            result.confidence = 0.6

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get extractor statistics."""
        return self.stats.copy()
