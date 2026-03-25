"""
Real Training Data Collector

Collects token metrics from Helius API and stores for ML training.
Supports both REST API polling and WebSocket streaming.

Architecture:
1. HeliusApiClient: Low-level API client with rate limiting
2. TokenSnapshot: Data structure for a single observation
3. RealDataCollector: Main orchestrator for data collection
4. OutcomeTracker: Tracks trade outcomes for labeling

Feature Alignment:
Uses the same 50-feature schema as synthetic_data.py for seamless
integration with ProfitabilityLearner.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import hashlib

import aiohttp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CollectorConfig:
    """Configuration for data collector."""
    # API settings
    helius_api_key: Optional[str] = None
    request_timeout_seconds: float = 10.0
    max_retries: int = 3
    
    # Rate limiting (Helius free tier: ~10 req/sec)
    rate_limit_per_second: float = 5.0
    burst_limit: int = 10
    
    # Collection settings
    snapshot_interval_seconds: float = 60.0  # How often to snapshot tracked tokens
    max_tracked_tokens: int = 100
    outcome_horizon_hours: int = 24  # Hours to wait for outcome
    
    # Storage
    output_dir: str = "data/historical"
    batch_size: int = 100  # Snapshots per batch save
    
    # Feature schema
    feature_count: int = 50
    
    @classmethod
    def from_env(cls) -> "CollectorConfig":
        """Create config from environment variables."""
        return cls(
            helius_api_key=os.environ.get("HELIUS_API_KEY"),
            rate_limit_per_second=float(os.environ.get("COLLECTOR_RATE_LIMIT", "5.0")),
            snapshot_interval_seconds=float(os.environ.get("COLLECTOR_INTERVAL", "60.0")),
            output_dir=os.environ.get("COLLECTOR_OUTPUT_DIR", "data/historical"),
        )


# =============================================================================
# Token Snapshot
# =============================================================================

class TokenStatus(Enum):
    """Status of a tracked token."""
    ACTIVE = "active"
    RUGGED = "rugged"
    MIGRATED = "migrated"
    ABANDONED = "abandoned"


@dataclass
class TokenSnapshot:
    """
    Snapshot of token metrics at a point in time.
    
    Aligned with 50-feature schema from feature_engineering.py.
    Used for training ML models on real market data.
    """
    # Identity
    mint: str
    timestamp_ms: int
    
    # === LIQUIDITY & POOL METRICS (0-11) ===
    pool_tvl_sol: float = 0.0          # Index 0: Pool TVL in SOL
    pool_age_seconds: float = 0.0      # Index 1: Pool age
    lp_lock_percentage: float = 0.0    # Index 2: LP lock %
    lp_concentration: float = 0.0      # Index 3: LP concentration
    lp_removal_velocity: float = 0.0   # Index 4: LP removal rate
    lp_addition_velocity: float = 0.0  # Index 5: LP addition rate
    pool_depth_imbalance: float = 0.0  # Index 6: Buy/sell imbalance
    slippage_1pct: float = 0.0         # Index 7: Slippage for 1% swap
    slippage_5pct: float = 0.0         # Index 8: Slippage for 5% swap
    unique_lp_provider_count: float = 0.0  # Index 9: Unique LPs
    deployer_lp_ownership_pct: float = 0.0 # Index 10: Deployer LP %
    emergency_liquidity_flag: float = 0.0  # Index 11: Emergency flag
    
    # === TOKEN SUPPLY & HOLDER (12-22) ===
    total_supply: float = 0.0          # Index 12: Total supply (for log scale)
    deployer_holdings_pct: float = 0.0 # Index 13: Deployer holdings %
    top_10_holder_concentration: float = 0.0  # Index 14: Top 10 holder %
    holder_count_unique: float = 0.0   # Index 15: Unique holders
    holder_growth_velocity: float = 0.0  # Index 16: Holder growth rate
    transfer_concentration: float = 0.0  # Index 17: Transfer concentration
    sniper_bot_count_t0: float = 0.0   # Index 18: Snipers at T0
    bot_to_human_ratio: float = 0.0    # Index 19: Bot/human ratio
    large_holder_churn: float = 0.0    # Index 20: Large holder churn
    mint_authority_revoked: float = 0.0  # Index 21: Mint authority status
    token_freezeable: float = 0.0      # Index 22: Freeze authority
    
    # === PRICE & VOLUME (23-32) ===
    price_momentum_30s: float = 0.0    # Index 23: 30s momentum
    price_momentum_5m: float = 0.0     # Index 24: 5m momentum
    volatility_5m: float = 0.0         # Index 25: 5m volatility
    volume_acceleration: float = 0.0   # Index 26: Volume acceleration
    volume_24h_usd: float = 0.0        # Index 27: 24h volume
    trade_size_variance: float = 0.0   # Index 28: Trade size variance
    vwap_deviation: float = 0.0        # Index 29: VWAP deviation
    price_impact_1pct: float = 0.0     # Index 30: Price impact 1%
    consecutive_buys: float = 0.0      # Index 31: Consecutive buys
    max_buy_in_window: float = 0.0     # Index 32: Max buy in window
    
    # === ON-CHAIN RISK (33-42) ===
    rug_pull_ml_score: float = 0.0     # Index 33: ML rug score
    contract_is_mintable: float = 0.0  # Index 34: Mintable contract
    contract_is_freezeable: float = 0.0  # Index 35: Freezable
    owner_changed_recently: float = 0.0  # Index 36: Owner change
    liquidity_is_locked: float = 0.0   # Index 37: LP locked
    sell_tax_bps: float = 0.0          # Index 38: Sell tax
    buy_tax_bps: float = 0.0           # Index 39: Buy tax
    max_tx_amount: float = 0.0         # Index 40: Max tx limit
    honeypot_probability: float = 0.0  # Index 41: Honeypot score
    mev_sandwich_count_1h: float = 0.0 # Index 42: MEV sandwiches
    
    # === SOCIAL & SENTIMENT (43-49) ===
    # These are populated by social_metrics.rs in HotPath
    # Zero until API keys are configured
    twitter_mention_velocity: float = 0.0  # Index 43
    twitter_follower_count: float = 0.0    # Index 44
    discord_member_count: float = 0.0      # Index 45
    telegram_member_count: float = 0.0     # Index 46
    reddit_subscriber_count: float = 0.0   # Index 47
    social_engagement_score: float = 0.0   # Index 48
    sentiment_score: float = 0.0           # Index 49
    
    # === OUTCOME LABELS (for training) ===
    # Filled after outcome_horizon_hours
    was_profitable: Optional[bool] = None
    pnl_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    exit_reason: Optional[str] = None
    
    # === METADATA ===
    status: TokenStatus = TokenStatus.ACTIVE
    collection_source: str = "helius"
    
    def to_feature_array(self) -> np.ndarray:
        """Convert to 50-feature numpy array."""
        return np.array([
            # Liquidity (0-11)
            self.pool_tvl_sol,
            self.pool_age_seconds,
            self.lp_lock_percentage,
            self.lp_concentration,
            self.lp_removal_velocity,
            self.lp_addition_velocity,
            self.pool_depth_imbalance,
            self.slippage_1pct,
            self.slippage_5pct,
            self.unique_lp_provider_count,
            self.deployer_lp_ownership_pct,
            self.emergency_liquidity_flag,
            # Supply & Holder (12-22)
            self.total_supply,
            self.deployer_holdings_pct,
            self.top_10_holder_concentration,
            self.holder_count_unique,
            self.holder_growth_velocity,
            self.transfer_concentration,
            self.sniper_bot_count_t0,
            self.bot_to_human_ratio,
            self.large_holder_churn,
            self.mint_authority_revoked,
            self.token_freezeable,
            # Price & Volume (23-32)
            self.price_momentum_30s,
            self.price_momentum_5m,
            self.volatility_5m,
            self.volume_acceleration,
            self.volume_24h_usd,
            self.trade_size_variance,
            self.vwap_deviation,
            self.price_impact_1pct,
            self.consecutive_buys,
            self.max_buy_in_window,
            # On-Chain Risk (33-42)
            self.rug_pull_ml_score,
            self.contract_is_mintable,
            self.contract_is_freezeable,
            self.owner_changed_recently,
            self.liquidity_is_locked,
            self.sell_tax_bps,
            self.buy_tax_bps,
            self.max_tx_amount,
            self.honeypot_probability,
            self.mev_sandwich_count_1h,
            # Social (43-49)
            self.twitter_mention_velocity,
            self.twitter_follower_count,
            self.discord_member_count,
            self.telegram_member_count,
            self.reddit_subscriber_count,
            self.social_engagement_score,
            self.sentiment_score,
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["status"] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenSnapshot":
        """Create from dictionary."""
        data["status"] = TokenStatus(data.get("status", "active"))
        return cls(**data)


# =============================================================================
# Helius API Client
# =============================================================================

class HeliusApiClient:
    """
    Low-level Helius API client with rate limiting.
    
    Supports both REST API and DAS (Digital Asset Standard) API.
    """
    
    def __init__(self, api_key: str, config: CollectorConfig):
        self.api_key = api_key
        self.config = config
        self.base_url = f"https://api.helius.xyz/v0"
        self.rpc_url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        
        # Rate limiting state
        self._last_request_time: float = 0.0
        self._request_count: int = 0
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None
    
    async def _rate_limit(self):
        """Apply rate limiting."""
        min_interval = 1.0 / self.config.rate_limit_per_second
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1
    
    async def _request_with_retry(
        self,
        url: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        method: str = "GET"
    ) -> Optional[Dict]:
        """Make HTTP request with retry logic."""
        if not self._session:
            raise RuntimeError("Session not initialized. Use async with.")
        
        for attempt in range(self.config.max_retries):
            await self._rate_limit()
            
            try:
                if method == "GET":
                    async with self._session.get(url, params=params) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        elif resp.status == 429:  # Rate limited
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.warning(f"API error: {resp.status}")
                            return None
                else:
                    async with self._session.post(url, json=json_data) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        elif resp.status == 429:
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            return None
                            
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
            except aiohttp.ClientError as e:
                logger.warning(f"Request error: {e} (attempt {attempt + 1})")
            
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        
        return None
    
    async def get_token_metadata(self, mint: str) -> Optional[Dict]:
        """Fetch token metadata from DAS API."""
        url = f"{self.base_url}/token-metadata"
        params = {"api-key": self.api_key}
        json_data = {"mintAccounts": [mint]}
        
        result = await self._request_with_retry(url, params=params, json_data=json_data, method="POST")
        if result and len(result) > 0:
            return result[0]
        return None
    
    async def get_token_accounts(self, mint: str, limit: int = 100) -> Optional[List[Dict]]:
        """Fetch token holder accounts."""
        url = self.rpc_url
        json_data = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "getTokenAccounts",
            "params": [{
                "mint": mint,
                "limit": limit
            }]
        }
        
        result = await self._request_with_retry(url, json_data=json_data, method="POST")
        if result and "result" in result:
            return result["result"].get("tokenAccounts", [])
        return None
    
    async def get_asset(self, mint: str) -> Optional[Dict]:
        """Fetch asset details from DAS API."""
        url = f"https://api.helius.xyz/v0/asset/{mint}"
        params = {"api-key": self.api_key}
        
        return await self._request_with_retry(url, params=params)
    
    async def get_transactions(
        self,
        mint: str,
        before: Optional[str] = None,
        limit: int = 100
    ) -> Optional[List[Dict]]:
        """Fetch recent transactions for a token."""
        url = f"{self.base_url}/addresses/{mint}/transactions"
        params = {"api-key": self.api_key, "limit": limit}
        if before:
            params["before"] = before
        
        return await self._request_with_retry(url, params=params)


# =============================================================================
# Real Data Collector
# =============================================================================

class RealDataCollector:
    """
    Main orchestrator for real token data collection.
    
    Collects snapshots from Helius API and stores them for training.
    Tracks outcomes over time to generate labeled training data.
    
    Usage:
        async with RealDataCollector(config) as collector:
            # Add token to track
            await collector.track_token("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
            
            # Start collection loop
            await collector.run()
    """
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        self.config = config or CollectorConfig.from_env()
        self._tracked_tokens: Dict[str, Dict[str, Any]] = {}  # mint -> tracking info
        self._snapshots: List[TokenSnapshot] = []
        self._api_client: Optional[HeliusApiClient] = None
        self._running: bool = False
        self._output_path = Path(self.config.output_dir)
        self._output_path.mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self):
        if not self.config.helius_api_key:
            raise ValueError("HELIUS_API_KEY required for data collection")
        
        self._api_client = HeliusApiClient(self.config.helius_api_key, self.config)
        await self._api_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._api_client:
            await self._api_client.__aexit__(exc_type, exc_val, exc_tb)
        await self._save_snapshots()
    
    @property
    def tracked_count(self) -> int:
        """Number of tokens being tracked."""
        return len(self._tracked_tokens)
    
    @property
    def snapshot_count(self) -> int:
        """Number of snapshots collected."""
        return len(self._snapshots)
    
    async def track_token(
        self,
        mint: str,
        entry_price: Optional[float] = None,
        entry_timestamp_ms: Optional[int] = None
    ) -> bool:
        """
        Add a token to the tracking list.
        
        Args:
            mint: Token mint address
            entry_price: Entry price (if known)
            entry_timestamp_ms: Entry timestamp (defaults to now)
        
        Returns:
            True if token added, False if already tracking or limit reached
        """
        if mint in self._tracked_tokens:
            logger.debug(f"Already tracking {mint[:8]}...")
            return False
        
        if len(self._tracked_tokens) >= self.config.max_tracked_tokens:
            logger.warning(f"Max tracked tokens reached ({self.config.max_tracked_tokens})")
            return False
        
        self._tracked_tokens[mint] = {
            "entry_price": entry_price,
            "entry_timestamp_ms": entry_timestamp_ms or int(time.time() * 1000),
            "last_snapshot_ms": 0,
            "snapshot_count": 0,
            "status": TokenStatus.ACTIVE,
        }
        
        logger.info(f"Now tracking {mint[:8]}... ({self.tracked_count} total)")
        return True
    
    async def untrack_token(self, mint: str, reason: str = "manual") -> None:
        """Remove a token from tracking."""
        if mint in self._tracked_tokens:
            del self._tracked_tokens[mint]
            logger.info(f"Stopped tracking {mint[:8]}... (reason: {reason})")
    
    async def collect_snapshot(self, mint: str) -> Optional[TokenSnapshot]:
        """
        Collect a single snapshot for a token.
        
        Fetches data from Helius and creates a TokenSnapshot.
        """
        if not self._api_client:
            raise RuntimeError("Collector not initialized. Use async with.")
        
        tracking = self._tracked_tokens.get(mint)
        if not tracking:
            return None
        
        now_ms = int(time.time() * 1000)

        try:
            logger.info("Collector stage=data_fetch start mint=%s", mint)
            # Fetch token metadata
            metadata = await self._api_client.get_token_metadata(mint)
            if not metadata:
                logger.warning(f"No metadata for {mint[:8]}...")
                return None
            
            # Fetch transactions for activity metrics
            transactions = await self._api_client.get_transactions(mint, limit=100)
            logger.info(
                "Collector stage=data_fetch complete mint=%s transactions=%s",
                mint,
                len(transactions or []),
            )
            
            # Extract features from API responses
            logger.info("Collector stage=feature_engineering start mint=%s", mint)
            snapshot = self._extract_features(
                mint=mint,
                metadata=metadata,
                transactions=transactions or [],
                tracking=tracking,
            )
            logger.info("Collector stage=feature_engineering complete mint=%s", mint)
            
            snapshot.timestamp_ms = now_ms
            
            # Update tracking info
            tracking["last_snapshot_ms"] = now_ms
            tracking["snapshot_count"] += 1
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error collecting snapshot for {mint[:8]}...: {e}", exc_info=True)
            return None
    
    def _extract_features(
        self,
        mint: str,
        metadata: Dict,
        transactions: List[Dict],
        tracking: Dict,
    ) -> TokenSnapshot:
        """
        Extract features from API responses.
        
        Maps Helius API data to the 50-feature schema.
        """
        snapshot = TokenSnapshot(mint=mint, timestamp_ms=0)
        
        # Extract from metadata
        token_info = metadata.get("tokenInfo", {})
        snapshot.total_supply = float(token_info.get("supply", 0))
        
        # Authority info
        authorities = metadata.get("authorities", [])
        for auth in authorities:
            auth_type = auth.get("type", "")
            if auth_type == "mint":
                snapshot.mint_authority_revoked = 0.0 if auth.get("address") else 1.0
            elif auth_type == "freeze":
                snapshot.token_freezeable = 1.0 if auth.get("address") else 0.0
        
        # Calculate from transactions
        if transactions:
            buy_count = 0
            sell_count = 0
            volumes = []
            timestamps = []
            
            for tx in transactions:
                tx_type = tx.get("type", "")
                timestamp = tx.get("timestamp", 0)
                timestamps.append(timestamp)
                
                # Parse swap events
                events = tx.get("events", {})
                swaps = events.get("swap", [])
                for swap in swaps:
                    native = swap.get("nativeInput", {})
                    token_output = swap.get("tokenOutputs", [])
                    if token_output:
                        buy_count += 1
                        volumes.append(float(native.get("amount", 0)) / 1e9)  # Lamports to SOL
                    else:
                        token_input = swap.get("tokenInputs", [])
                        if token_input:
                            sell_count += 1
            
            # Price & Volume features
            snapshot.consecutive_buys = float(buy_count) if buy_count > sell_count else 0.0
            snapshot.volume_24h_usd = sum(volumes) * 100  # Rough USD estimate
            
            if len(timestamps) >= 2:
                time_span = max(timestamps) - min(timestamps)
                if time_span > 0:
                    snapshot.volatility_5m = np.std(volumes) / (np.mean(volumes) + 1e-10)
            
            # Pool imbalance
            total_trades = buy_count + sell_count
            if total_trades > 0:
                snapshot.pool_depth_imbalance = (buy_count - sell_count) / total_trades
        
        # Entry tracking for outcome
        entry_price = tracking.get("entry_price")
        if entry_price:
            snapshot.price_momentum_30s = 0.0  # Would need current price
        
        # Risk indicators (would need more sophisticated analysis)
        snapshot.rug_pull_ml_score = 0.1  # Default low risk
        snapshot.honeypot_probability = 0.05  # Default low
        
        return snapshot
    
    async def run(self, duration_seconds: Optional[float] = None) -> int:
        """
        Run the collection loop.
        
        Args:
            duration_seconds: How long to run (None = forever)
        
        Returns:
            Number of snapshots collected
        """
        if not self._api_client:
            raise RuntimeError("Collector not initialized. Use async with.")
        
        self._running = True
        start_time = time.time()
        snapshots_collected = 0
        next_snapshot_at = time.monotonic()

        logger.info(f"Starting collection loop (interval: {self.config.snapshot_interval_seconds}s)")
        
        while self._running:
            sleep_for = max(0.0, next_snapshot_at - time.monotonic())
            if sleep_for:
                await asyncio.sleep(sleep_for)

            # Check duration
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                logger.info(f"Duration reached ({duration_seconds}s), stopping")
                break
            
            # Collect snapshots for all tracked tokens
            for mint in list(self._tracked_tokens.keys()):
                if not self._running:
                    break
                
                try:
                    snapshot = await asyncio.wait_for(
                        self.collect_snapshot(mint),
                        timeout=max(self.config.request_timeout_seconds * 2, 5.0),
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Collector stage=data_fetch timeout mint=%s timeout_s=%.1f",
                        mint,
                        max(self.config.request_timeout_seconds * 2, 5.0),
                    )
                    continue
                except Exception as exc:
                    logger.error(
                        "Collector stage=data_fetch error mint=%s error=%s",
                        mint,
                        exc,
                        exc_info=True,
                    )
                    continue

                if snapshot:
                    self._snapshots.append(snapshot)
                    snapshots_collected += 1
                    logger.info(
                        "Collector stage=observation_append mint=%s snapshot_count=%s buffered=%s",
                        mint,
                        snapshots_collected,
                        len(self._snapshots),
                    )
                    
                    # Batch save
                    if len(self._snapshots) >= self.config.batch_size:
                        await self._save_snapshots()
            
            next_snapshot_at += self.config.snapshot_interval_seconds
            if next_snapshot_at < time.monotonic():
                logger.warning(
                    "Collector loop lag detected overrun_ms=%s interval_s=%.1f",
                    int((time.monotonic() - next_snapshot_at) * 1000),
                    self.config.snapshot_interval_seconds,
                )
                next_snapshot_at = time.monotonic() + self.config.snapshot_interval_seconds
        
        # Final save
        await self._save_snapshots()
        
        logger.info(f"Collection complete. Total snapshots: {snapshots_collected}")
        return snapshots_collected
    
    def stop(self):
        """Stop the collection loop."""
        self._running = False
    
    async def _save_snapshots(self):
        """Save collected snapshots to disk."""
        if not self._snapshots:
            return
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"snapshots_{timestamp}.parquet"
        filepath = self._output_path / filename
        
        # Convert to DataFrame
        df = pd.DataFrame([s.to_dict() for s in self._snapshots])
        
        # Save as parquet
        df.to_parquet(filepath, index=False)
        
        logger.info(f"Saved {len(self._snapshots)} snapshots to {filepath}")
        self._snapshots = []
    
    def load_snapshots(self) -> pd.DataFrame:
        """Load all collected snapshots from disk."""
        all_dfs = []
        
        for filepath in self._output_path.glob("snapshots_*.parquet"):
            df = pd.read_parquet(filepath)
            all_dfs.append(df)
        
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()


# =============================================================================
# Outcome Tracker
# =============================================================================

class OutcomeTracker:
    """
    Tracks trade outcomes for labeling training data.
    
    After collecting snapshots, this class determines the actual
    outcomes (profitable vs unprofitable) based on price movement.
    """
    
    def __init__(self, outcome_horizon_hours: int = 24):
        self.outcome_horizon_hours = outcome_horizon_hours
        self._pending_outcomes: Dict[str, Dict] = {}  # mint -> entry info
    
    def add_entry(
        self,
        mint: str,
        entry_price: float,
        entry_timestamp_ms: int,
        position_size_sol: float = 0.0
    ):
        """Record an entry for outcome tracking."""
        self._pending_outcomes[mint] = {
            "entry_price": entry_price,
            "entry_timestamp_ms": entry_timestamp_ms,
            "position_size_sol": position_size_sol,
        }
    
    def calculate_outcome(
        self,
        entry_price: float,
        exit_price: float,
        max_price: float = None,
        min_price: float = None
    ) -> Dict[str, Any]:
        """
        Calculate trade outcome metrics.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            max_price: Maximum price during hold (for drawdown)
            min_price: Minimum price during hold (for max gain)
        
        Returns:
            Dict with outcome metrics
        """
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        
        max_drawdown_pct = 0.0
        if max_price and max_price > entry_price:
            if min_price:
                max_drawdown_pct = ((max_price - min_price) / max_price) * 100
        
        # Determine exit reason
        if pnl_pct >= 50:
            exit_reason = "take_profit"
        elif pnl_pct <= -25:
            exit_reason = "stop_loss"
        else:
            exit_reason = "time_exit"
        
        return {
            "was_profitable": pnl_pct > 0,
            "pnl_pct": pnl_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "exit_reason": exit_reason,
        }
    
    def label_snapshots(
        self,
        snapshots: pd.DataFrame,
        price_lookup: Callable[[str, int], Optional[float]]
    ) -> pd.DataFrame:
        """
        Add outcome labels to snapshots.
        
        Args:
            snapshots: DataFrame of snapshots
            price_lookup: Function(mint, timestamp_ms) -> price
        
        Returns:
            DataFrame with outcome labels added
        """
        labeled = snapshots.copy()
        
        for idx, row in labeled.iterrows():
            mint = row["mint"]
            entry_ts = row.get("entry_timestamp_ms", row["timestamp_ms"])
            exit_ts = entry_ts + (self.outcome_horizon_hours * 3600 * 1000)
            
            entry_price = price_lookup(mint, entry_ts)
            exit_price = price_lookup(mint, exit_ts)
            
            if entry_price and exit_price:
                outcome = self.calculate_outcome(entry_price, exit_price)
                labeled.at[idx, "was_profitable"] = outcome["was_profitable"]
                labeled.at[idx, "pnl_pct"] = outcome["pnl_pct"]
                labeled.at[idx, "max_drawdown_pct"] = outcome["max_drawdown_pct"]
                labeled.at[idx, "exit_reason"] = outcome["exit_reason"]
        
        return labeled


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """CLI entry point for data collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect real token data for training")
    parser.add_argument("--duration", type=float, default=60.0, help="Collection duration in seconds")
    parser.add_argument("--interval", type=float, default=10.0, help="Snapshot interval in seconds")
    parser.add_argument("--output", type=str, default="data/historical", help="Output directory")
    parser.add_argument("--tokens", type=str, nargs="*", help="Token mints to track")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    config = CollectorConfig(
        snapshot_interval_seconds=args.interval,
        output_dir=args.output,
    )
    
    async with RealDataCollector(config) as collector:
        # Track specified tokens or use defaults
        tokens = args.tokens or [
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
        ]
        
        for mint in tokens:
            await collector.track_token(mint)
        
        await collector.run(duration_seconds=args.duration)


if __name__ == "__main__":
    asyncio.run(main())
