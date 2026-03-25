"""
On-chain verifier for token announcements.

Verifies:
- Contract exists on-chain via Solana RPC
- Valid SPL token with proper structure
- Mint/freeze authority status
- Copy-paste scam detection via Levenshtein distance

Uses asynchronous RPC calls for low-latency verification.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
from Levenshtein import distance as levenshtein_distance

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of on-chain verification."""

    contract_address: str
    contract_exists: bool = False
    is_valid_token: bool = False
    mint_authority_enabled: bool | None = None
    freeze_authority_enabled: bool | None = None
    supply: int | None = None
    decimals: int | None = None
    copy_paste_score: float = 0.0  # 0.0 - 1.0, higher = more similar to known tokens
    similar_to: str | None = None  # Name of similar token if detected
    verified_at_ms: int = 0
    latency_ms: int = 0


@dataclass
class VerifierConfig:
    """Configuration for on-chain verifier."""

    # Solana RPC URL
    rpc_url: str = "https://api.mainnet-beta.solana.com"

    # Timeout settings
    rpc_timeout_secs: int = 10

    # Copy-paste detection settings
    copy_paste_threshold: float = 0.7  # Above this similarity is flagged
    known_tokens_file: str | None = None

    @classmethod
    def from_env(cls) -> "VerifierConfig":
        """Load configuration from environment."""
        return cls(
            rpc_url=os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
            rpc_timeout_secs=int(os.environ.get("RPC_TIMEOUT_SECS", "10")),
            copy_paste_threshold=float(os.environ.get("COPY_PASTE_THRESHOLD", "0.7")),
        )


# Well-known Solana tokens for copy-paste detection
KNOWN_TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "WIF": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "PYTH": "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3",
    "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
    "ORCA": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
    "SAMO": "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
    "PEPE": "HRj6j9dMdCwZLSxVJzPnhM9S3gMEKPbRz3WqYnKPsng",
    "SHIB": "SHIB79CRAVCVGhMDRF96h7R8qMFcQ1o9vQEWBYYzWLu",
    "DOGE": "6DPdUG3WgPkKz5XPdFCZR1Bz31X3Q9NBh6P8o6M1mY2W",
}

# Common scam name patterns
SCAM_NAME_PATTERNS = [
    "official",
    "real",
    "original",
    "v2",
    "2.0",
    "airdrop",
    "claim",
    "free",
]


class OnChainVerifier:
    """
    On-chain verifier for token contracts.

    Verifies that contract addresses exist and are valid SPL tokens,
    and detects copy-paste scams using name similarity.
    """

    def __init__(self, config: VerifierConfig | None = None):
        """Initialize the verifier."""
        self.config = config or VerifierConfig.from_env()
        self.session: aiohttp.ClientSession | None = None

        # Known tokens for copy-paste detection
        self.known_tokens = KNOWN_TOKENS.copy()
        self.known_names: set[str] = set(self.known_tokens.keys())

        # Track whether we own the session (created outside context manager)
        self._owns_session = False

        # Statistics
        self.stats = {
            "total_verified": 0,
            "contracts_found": 0,
            "valid_tokens": 0,
            "copy_paste_detected": 0,
            "rpc_errors": 0,
            "total_latency_ms": 0,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.rpc_timeout_secs)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            self.session = None

    def __del__(self):
        """Safety net for sessions not properly closed via context manager."""
        if self.session and not self.session.closed:
            logger.warning("OnChainVerifier session not properly closed")

    async def verify(
        self,
        contract_address: str,
        pool_address: str | None = None,
        token_name: str | None = None,
        token_symbol: str | None = None,
    ) -> VerificationResult:
        """
        Verify a token contract on-chain.

        Args:
            contract_address: Token mint address
            pool_address: Pool/bonding curve address (optional)
            token_name: Token name for copy-paste detection
            token_symbol: Token symbol for copy-paste detection

        Returns:
            VerificationResult with verification status
        """
        start_time = time.time()

        result = VerificationResult(
            contract_address=contract_address,
            verified_at_ms=int(time.time() * 1000),
        )

        # Ensure session is initialized (prefer using context manager)
        if not self.session:
            logger.warning("OnChainVerifier used without context manager; creating session")
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.rpc_timeout_secs)
            )

        try:
            # Check if contract exists and get account info
            account_info = await self._get_account_info(contract_address)

            if account_info:
                result.contract_exists = True
                self.stats["contracts_found"] += 1

                # Check if it's a valid SPL token
                token_info = await self._get_token_info(contract_address)

                if token_info:
                    result.is_valid_token = True
                    result.supply = token_info.get("supply")
                    result.decimals = token_info.get("decimals")
                    result.mint_authority_enabled = token_info.get("mint_authority") is not None
                    result.freeze_authority_enabled = token_info.get("freeze_authority") is not None
                    self.stats["valid_tokens"] += 1

            # Check for copy-paste scam
            if token_name or token_symbol:
                copy_score, similar_name = self._detect_copy_paste(token_name, token_symbol)
                result.copy_paste_score = copy_score
                result.similar_to = similar_name

                if copy_score >= self.config.copy_paste_threshold:
                    self.stats["copy_paste_detected"] += 1
                    logger.warning(
                        f"Copy-paste scam detected: {token_name}/{token_symbol} "
                        f"similar to {similar_name} (score: {copy_score:.2f})"
                    )

        except Exception as e:
            logger.error(f"Verification error for {contract_address}: {e}")
            self.stats["rpc_errors"] += 1

        result.latency_ms = int((time.time() - start_time) * 1000)
        self.stats["total_verified"] += 1
        self.stats["total_latency_ms"] += result.latency_ms

        return result

    async def _get_account_info(self, address: str) -> dict[str, Any] | None:
        """Get account info from Solana RPC."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getAccountInfo",
            "params": [address, {"encoding": "base64"}],
        }

        try:
            async with self.session.post(self.config.rpc_url, json=payload) as response:
                data = await response.json()
                result = data.get("result", {})
                return result.get("value")
        except Exception as e:
            logger.error(f"getAccountInfo error: {e}")
            return None

    async def _get_token_info(self, mint_address: str) -> dict[str, Any] | None:
        """Get token mint info from Solana RPC."""
        # Use getAccountInfo with parsed encoding for SPL tokens
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getAccountInfo",
            "params": [mint_address, {"encoding": "jsonParsed"}],
        }

        try:
            async with self.session.post(self.config.rpc_url, json=payload) as response:
                data = await response.json()
                result = data.get("result", {})
                value = result.get("value")

                if not value:
                    return None

                parsed = value.get("data", {}).get("parsed", {})
                if parsed.get("type") != "mint":
                    return None

                info = parsed.get("info", {})
                return {
                    "supply": int(info.get("supply", 0)),
                    "decimals": info.get("decimals"),
                    "mint_authority": info.get("mintAuthority"),
                    "freeze_authority": info.get("freezeAuthority"),
                    "is_initialized": info.get("isInitialized", False),
                }
        except Exception as e:
            logger.error(f"getTokenInfo error: {e}")
            return None

    def _detect_copy_paste(
        self,
        token_name: str | None,
        token_symbol: str | None,
    ) -> tuple[float, str | None]:
        """
        Detect copy-paste scam by comparing to known tokens.

        Uses Levenshtein distance to find similar names/symbols.

        Returns:
            Tuple of (similarity_score, most_similar_token_name)
        """
        max_score = 0.0
        most_similar: str | None = None

        # Normalize inputs
        name_lower = token_name.lower().strip() if token_name else ""
        symbol_upper = token_symbol.upper().strip() if token_symbol else ""

        # Check symbol against known tokens
        for known_symbol in self.known_names:
            if not symbol_upper:
                continue

            # Exact match with scam prefix/suffix
            for pattern in SCAM_NAME_PATTERNS:
                if pattern in name_lower and known_symbol.lower() in name_lower:
                    return (0.95, known_symbol)

            # Calculate symbol similarity
            if len(symbol_upper) > 0 and len(known_symbol) > 0:
                dist = levenshtein_distance(symbol_upper, known_symbol)
                max_len = max(len(symbol_upper), len(known_symbol))
                similarity = 1.0 - (dist / max_len)

                # Very similar symbol is suspicious
                if similarity > max_score and similarity > 0.5:
                    max_score = similarity
                    most_similar = known_symbol

        # Check name against known tokens
        for known_name in self.known_names:
            if not name_lower:
                continue

            known_lower = known_name.lower()

            # Name contains known token name
            if known_lower in name_lower and known_lower != name_lower:
                # Calculate how much of the name is the known token
                containment_score = len(known_lower) / len(name_lower)
                if containment_score > 0.5:
                    score = 0.8 + containment_score * 0.2
                    if score > max_score:
                        max_score = score
                        most_similar = known_name

            # Levenshtein similarity for names
            dist = levenshtein_distance(name_lower, known_lower)
            max_len = max(len(name_lower), len(known_lower))
            similarity = 1.0 - (dist / max_len)

            if similarity > max_score and similarity > 0.6:
                max_score = similarity
                most_similar = known_name

        return (max_score, most_similar)

    async def verify_batch(
        self,
        tokens: list[dict[str, str]],
    ) -> list[VerificationResult]:
        """
        Verify multiple tokens in parallel.

        Args:
            tokens: List of dicts with keys: contract_address, pool_address,
                   token_name, token_symbol

        Returns:
            List of VerificationResult objects
        """
        tasks = [
            self.verify(
                contract_address=t.get("contract_address", ""),
                pool_address=t.get("pool_address"),
                token_name=t.get("token_name"),
                token_symbol=t.get("token_symbol"),
            )
            for t in tokens
        ]

        return await asyncio.gather(*tasks)

    def add_known_token(self, name: str, address: str):
        """Add a token to the known tokens list."""
        self.known_tokens[name] = address
        self.known_names.add(name)

    def get_stats(self) -> dict[str, Any]:
        """Get verifier statistics."""
        return self.stats.copy()


# Convenience function for standalone verification
async def verify_token(
    contract_address: str,
    token_name: str | None = None,
    token_symbol: str | None = None,
    rpc_url: str | None = None,
) -> VerificationResult:
    """
    Verify a single token contract.

    Args:
        contract_address: Token mint address
        token_name: Token name for copy-paste detection
        token_symbol: Token symbol for copy-paste detection
        rpc_url: Optional Solana RPC URL

    Returns:
        VerificationResult
    """
    config = VerifierConfig.from_env()
    if rpc_url:
        config.rpc_url = rpc_url

    async with OnChainVerifier(config) as verifier:
        return await verifier.verify(
            contract_address=contract_address,
            token_name=token_name,
            token_symbol=token_symbol,
        )
