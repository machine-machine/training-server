"""
Outcome Tracker - Track all scan outcomes for learning.

Records both traded and skipped tokens with counterfactual tracking
for tokens that were skipped (what would have happened).
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage import DatabaseManager

logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    """Type of scan outcome."""
    TRADED = "traded"
    SKIPPED = "skipped"
    REJECTED = "rejected"
    FAILED = "failed"


class SkipReason(Enum):
    """Reason for skipping a token."""
    LOW_CONFIDENCE = "low_confidence"
    LOW_LIQUIDITY = "low_liquidity"
    HIGH_RISK = "high_risk"
    GATEKEEPER_REJECT = "gatekeeper_reject"
    DAILY_LIMIT = "daily_limit"
    POSITION_TOO_SMALL = "position_too_small"
    TIMEOUT = "timeout"
    MANUAL = "manual"
    UNKNOWN = "unknown"


@dataclass
class TradeResult:
    """Result of an executed trade."""
    entry_price: float
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    pnl_sol: Optional[float] = None
    hold_duration_ms: Optional[int] = None
    slippage_bps: Optional[int] = None
    transaction_signature: Optional[str] = None
    execution_mode: str = "paper"


@dataclass
class CounterfactualData:
    """Counterfactual data for skipped tokens."""
    price_at_skip: float
    price_1h_later: Optional[float] = None
    price_24h_later: Optional[float] = None
    max_price_24h: Optional[float] = None
    min_price_24h: Optional[float] = None

    @property
    def return_1h(self) -> Optional[float]:
        """Calculate 1h return if we had bought."""
        if self.price_1h_later and self.price_at_skip > 0:
            return (self.price_1h_later - self.price_at_skip) / self.price_at_skip * 100
        return None

    @property
    def return_24h(self) -> Optional[float]:
        """Calculate 24h return if we had bought."""
        if self.price_24h_later and self.price_at_skip > 0:
            return (self.price_24h_later - self.price_at_skip) / self.price_at_skip * 100
        return None

    @property
    def max_return_24h(self) -> Optional[float]:
        """Calculate max possible return in 24h."""
        if self.max_price_24h and self.price_at_skip > 0:
            return (self.max_price_24h - self.price_at_skip) / self.price_at_skip * 100
        return None

    @property
    def was_profitable(self) -> Optional[bool]:
        """Check if this would have been profitable (>0 return at 1h)."""
        ret = self.return_1h
        return ret > 0 if ret is not None else None


@dataclass
class ScanOutcome:
    """Complete outcome record for a scanned token."""
    # Identification
    mint: str
    pool: str
    timestamp_ms: int

    # Outcome type
    outcome_type: OutcomeType
    skip_reason: Optional[SkipReason] = None

    # Features at scan time (17 features)
    features: Dict[str, float] = field(default_factory=dict)

    # Scoring results
    profitability_score: Optional[float] = None
    confidence: Optional[float] = None
    expected_return_pct: Optional[float] = None

    # Trade result (if traded)
    trade_result: Optional[TradeResult] = None

    # Counterfactual (if skipped)
    counterfactual: Optional[CounterfactualData] = None

    # Metadata
    model_version: int = 0
    source: str = "scanner"

    @property
    def was_profitable(self) -> Optional[bool]:
        """Determine if this outcome was profitable."""
        if self.outcome_type == OutcomeType.TRADED and self.trade_result:
            return self.trade_result.pnl_pct is not None and self.trade_result.pnl_pct > 0
        elif self.outcome_type == OutcomeType.SKIPPED and self.counterfactual:
            return self.counterfactual.was_profitable
        return None

    @property
    def actual_return_pct(self) -> Optional[float]:
        """Get the actual return percentage."""
        if self.outcome_type == OutcomeType.TRADED and self.trade_result:
            return self.trade_result.pnl_pct
        elif self.outcome_type == OutcomeType.SKIPPED and self.counterfactual:
            return self.counterfactual.return_1h
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            "mint": self.mint,
            "pool": self.pool,
            "timestamp_ms": self.timestamp_ms,
            "outcome_type": self.outcome_type.value,
            "skip_reason": self.skip_reason.value if self.skip_reason else None,
            "features_json": json.dumps(self.features),
            "profitability_score": self.profitability_score,
            "confidence": self.confidence,
            "expected_return_pct": self.expected_return_pct,
            "model_version": self.model_version,
            "source": self.source,
        }

        if self.trade_result:
            result.update({
                "entry_price": self.trade_result.entry_price,
                "exit_price": self.trade_result.exit_price,
                "pnl_pct": self.trade_result.pnl_pct,
                "pnl_sol": self.trade_result.pnl_sol,
                "hold_duration_ms": self.trade_result.hold_duration_ms,
            })

        if self.counterfactual:
            result.update({
                "price_at_skip": self.counterfactual.price_at_skip,
                "price_1h_later": self.counterfactual.price_1h_later,
                "price_24h_later": self.counterfactual.price_24h_later,
                "was_profitable_counterfactual": 1 if self.counterfactual.was_profitable else 0,
            })

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScanOutcome":
        """Create from dictionary."""
        features = json.loads(data.get("features_json", "{}"))

        trade_result = None
        if data.get("entry_price") is not None:
            trade_result = TradeResult(
                entry_price=data["entry_price"],
                exit_price=data.get("exit_price"),
                pnl_pct=data.get("pnl_pct"),
                pnl_sol=data.get("pnl_sol"),
                hold_duration_ms=data.get("hold_duration_ms"),
            )

        counterfactual = None
        if data.get("price_at_skip") is not None:
            counterfactual = CounterfactualData(
                price_at_skip=data["price_at_skip"],
                price_1h_later=data.get("price_1h_later"),
                price_24h_later=data.get("price_24h_later"),
            )

        return cls(
            mint=data["mint"],
            pool=data.get("pool", ""),
            timestamp_ms=data["timestamp_ms"],
            outcome_type=OutcomeType(data["outcome_type"]),
            skip_reason=SkipReason(data["skip_reason"]) if data.get("skip_reason") else None,
            features=features,
            profitability_score=data.get("profitability_score"),
            confidence=data.get("confidence"),
            expected_return_pct=data.get("expected_return_pct"),
            trade_result=trade_result,
            counterfactual=counterfactual,
            model_version=data.get("model_version", 0),
            source=data.get("source", "scanner"),
        )


class OutcomeTracker:
    """Track and store scan outcomes for learning."""

    # Default cleanup age
    MAX_PENDING_AGE_HOURS = 48

    def __init__(self, db: "DatabaseManager"):
        self.db = db
        self._pending_counterfactuals: Dict[str, ScanOutcome] = {}
        self._lock = asyncio.Lock()

    async def record_trade(
        self,
        mint: str,
        pool: str,
        features: Dict[str, float],
        profitability_score: float,
        confidence: float,
        expected_return_pct: float,
        trade_result: TradeResult,
        model_version: int = 0,
    ) -> int:
        """Record a completed trade outcome."""
        outcome = ScanOutcome(
            mint=mint,
            pool=pool,
            timestamp_ms=int(datetime.now().timestamp() * 1000),
            outcome_type=OutcomeType.TRADED,
            features=features,
            profitability_score=profitability_score,
            confidence=confidence,
            expected_return_pct=expected_return_pct,
            trade_result=trade_result,
            model_version=model_version,
        )

        return await self._store_outcome(outcome)

    async def record_skip(
        self,
        mint: str,
        pool: str,
        features: Dict[str, float],
        skip_reason: SkipReason,
        profitability_score: Optional[float] = None,
        confidence: Optional[float] = None,
        expected_return_pct: Optional[float] = None,
        current_price: Optional[float] = None,
        model_version: int = 0,
    ) -> int:
        """Record a skipped token and schedule counterfactual tracking."""
        outcome = ScanOutcome(
            mint=mint,
            pool=pool,
            timestamp_ms=int(datetime.now().timestamp() * 1000),
            outcome_type=OutcomeType.SKIPPED,
            skip_reason=skip_reason,
            features=features,
            profitability_score=profitability_score,
            confidence=confidence,
            expected_return_pct=expected_return_pct,
            model_version=model_version,
        )

        if current_price is not None:
            outcome.counterfactual = CounterfactualData(price_at_skip=current_price)

        # Store and track for counterfactual updates
        outcome_id = await self._store_outcome(outcome)

        if current_price is not None:
            async with self._lock:
                self._pending_counterfactuals[mint] = outcome

        return outcome_id

    async def record_rejection(
        self,
        mint: str,
        pool: str,
        features: Dict[str, float],
        reason: str,
        model_version: int = 0,
    ) -> int:
        """Record a rejected token (gatekeeper hard reject)."""
        outcome = ScanOutcome(
            mint=mint,
            pool=pool,
            timestamp_ms=int(datetime.now().timestamp() * 1000),
            outcome_type=OutcomeType.REJECTED,
            skip_reason=SkipReason.GATEKEEPER_REJECT,
            features=features,
            model_version=model_version,
        )

        return await self._store_outcome(outcome)

    async def update_counterfactual(
        self,
        mint: str,
        price_1h: Optional[float] = None,
        price_24h: Optional[float] = None,
    ):
        """Update counterfactual prices for a skipped token."""
        async with self._lock:
            if mint not in self._pending_counterfactuals:
                return

            outcome = self._pending_counterfactuals[mint]
            if outcome.counterfactual is None:
                return

            if price_1h is not None:
                outcome.counterfactual.price_1h_later = price_1h

            if price_24h is not None:
                outcome.counterfactual.price_24h_later = price_24h
                # Remove from pending after 24h update
                del self._pending_counterfactuals[mint]

        # Update in database
        await self._update_counterfactual_db(mint, price_1h, price_24h)

    async def _store_outcome(self, outcome: ScanOutcome) -> int:
        """Store outcome in database."""
        data = outcome.to_dict()
        return await self.db.insert_scan_outcome(data)

    async def _update_counterfactual_db(
        self,
        mint: str,
        price_1h: Optional[float],
        price_24h: Optional[float],
    ):
        """Update counterfactual data in database."""
        updates = {}
        if price_1h is not None:
            updates["price_1h_later"] = price_1h
        if price_24h is not None:
            updates["price_24h_later"] = price_24h
            # Calculate was_profitable
            outcome = self._pending_counterfactuals.get(mint)
            if outcome and outcome.counterfactual:
                was_profitable = outcome.counterfactual.was_profitable
                updates["was_profitable_counterfactual"] = 1 if was_profitable else 0

        if updates:
            await self.db.update_scan_outcome_counterfactual(mint, updates)

    async def get_training_data(
        self,
        since_hours: int = 168,  # 1 week
        min_samples: int = 100,
        include_counterfactuals: bool = True,
    ) -> List[ScanOutcome]:
        """Get outcomes for training the profitability model."""
        outcomes = await self.db.get_scan_outcomes(since_hours=since_hours)

        result = []
        for data in outcomes:
            outcome = ScanOutcome.from_dict(data)

            # Filter to useful outcomes
            if outcome.outcome_type == OutcomeType.TRADED:
                if outcome.trade_result and outcome.trade_result.pnl_pct is not None:
                    result.append(outcome)
            elif outcome.outcome_type == OutcomeType.SKIPPED and include_counterfactuals:
                if outcome.counterfactual and outcome.counterfactual.was_profitable is not None:
                    result.append(outcome)

        return result

    async def get_pending_counterfactual_mints(self) -> List[str]:
        """Get mints that need counterfactual price updates."""
        async with self._lock:
            return list(self._pending_counterfactuals.keys())

    async def cleanup_old_pending(self, max_age_hours: Optional[int] = None):
        """Clean up old pending counterfactuals."""
        if max_age_hours is None:
            max_age_hours = self.MAX_PENDING_AGE_HOURS

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        cutoff_ms = int(cutoff.timestamp() * 1000)

        async with self._lock:
            to_remove = []
            for mint, outcome in self._pending_counterfactuals.items():
                if outcome.timestamp_ms < cutoff_ms:
                    to_remove.append(mint)

            for mint in to_remove:
                del self._pending_counterfactuals[mint]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old pending counterfactuals")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get outcome tracking statistics."""
        outcomes = await self.db.get_scan_outcomes(since_hours=24)

        traded_count = sum(1 for o in outcomes if o.get("outcome_type") == "traded")
        skipped_count = sum(1 for o in outcomes if o.get("outcome_type") == "skipped")
        rejected_count = sum(1 for o in outcomes if o.get("outcome_type") == "rejected")

        # Calculate win rates
        traded_outcomes = [o for o in outcomes if o.get("outcome_type") == "traded"]
        traded_wins = sum(1 for o in traded_outcomes if (o.get("pnl_pct") or 0) > 0)

        counterfactual_outcomes = [
            o for o in outcomes
            if o.get("outcome_type") == "skipped" and o.get("was_profitable_counterfactual") is not None
        ]
        counterfactual_wins = sum(1 for o in counterfactual_outcomes if o.get("was_profitable_counterfactual"))

        return {
            "total_24h": len(outcomes),
            "traded_24h": traded_count,
            "skipped_24h": skipped_count,
            "rejected_24h": rejected_count,
            "traded_win_rate": traded_wins / traded_count if traded_count > 0 else 0,
            "counterfactual_win_rate": counterfactual_wins / len(counterfactual_outcomes) if counterfactual_outcomes else 0,
            "pending_counterfactuals": len(self._pending_counterfactuals),
        }
