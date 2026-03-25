"""
Trade feedback validation for /data/feedback endpoint.

Validates incoming trade data to prevent garbage from entering
the ML training pipeline.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional


@dataclass
class ValidationResult:
    """Result of validating a single trade."""
    is_valid: bool
    trade_id: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class BatchValidationResult:
    """Result of validating a batch of trades."""
    accepted: list[dict]
    rejected: list[ValidationResult]
    
    @property
    def accepted_count(self) -> int:
        return len(self.accepted)
    
    @property
    def rejected_count(self) -> int:
        return len(self.rejected)
    
    @property
    def rejection_reasons(self) -> list[dict]:
        """Get all rejection reasons with trade IDs."""
        return [
            {"trade_id": r.trade_id, "reasons": r.reasons}
            for r in self.rejected
        ]


class TradeFeedbackValidator:
    """Validates trade feedback data before ingestion.
    
    Validation rules:
    - features: no NaN, no Inf, length <= 50
    - confidence: 0.0 <= x <= 1.0
    - pnl_sol: -1000 <= x <= 1000 (sanity bounds)
    - slippage_bps: 0 <= x <= 10000
    - timestamp_ms: within last 30 days
    """
    
    # Validation bounds
    MAX_FEATURES = 50
    MIN_CONFIDENCE = 0.0
    MAX_CONFIDENCE = 1.0
    MIN_PNL_SOL = -1000.0
    MAX_PNL_SOL = 1000.0
    MIN_SLIPPAGE_BPS = 0
    MAX_SLIPPAGE_BPS = 10000
    MAX_TIMESTAMP_AGE_DAYS = 30
    
    def __init__(self, strict: bool = True):
        """Initialize validator.
        
        Args:
            strict: If True, reject invalid trades. If False, accept with warnings.
        """
        self.strict = strict
        self._cutoff_timestamp = self._compute_cutoff()
    
    def _compute_cutoff(self) -> int:
        """Compute the cutoff timestamp for 30 days ago."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.MAX_TIMESTAMP_AGE_DAYS)
        return int(cutoff.timestamp() * 1000)
    
    def _is_finite(self, value: float) -> bool:
        """Check if value is finite (not NaN or Inf)."""
        try:
            return not (math.isnan(value) or math.isinf(value))
        except (TypeError, ValueError):
            return False
    
    def _validate_features(self, features: list[float]) -> list[str]:
        """Validate feature vector."""
        reasons = []
        
        if features is None:
            reasons.append("features is None")
            return reasons
        
        if len(features) == 0:
            reasons.append("features is empty")
            return reasons
        
        if len(features) > self.MAX_FEATURES:
            reasons.append(f"features length {len(features)} exceeds max {self.MAX_FEATURES}")
        
        # Check for NaN/Inf in features
        for i, f in enumerate(features):
            if not self._is_finite(f):
                reasons.append(f"features[{i}] is {'NaN' if math.isnan(f) else 'Inf'}")
                break  # Only report first occurrence
        
        return reasons
    
    def _validate_confidence(self, confidence: float) -> list[str]:
        """Validate confidence value."""
        reasons = []
        
        if not self._is_finite(confidence):
            reasons.append(f"confidence is {'NaN' if math.isnan(confidence) else 'Inf'}")
            return reasons
        
        if confidence < self.MIN_CONFIDENCE or confidence > self.MAX_CONFIDENCE:
            reasons.append(f"confidence {confidence} not in [{self.MIN_CONFIDENCE}, {self.MAX_CONFIDENCE}]")
        
        return reasons
    
    def _validate_pnl(self, pnl_sol: float) -> list[str]:
        """Validate P&L value."""
        reasons = []
        
        if not self._is_finite(pnl_sol):
            reasons.append(f"pnl_sol is {'NaN' if math.isnan(pnl_sol) else 'Inf'}")
            return reasons
        
        if pnl_sol < self.MIN_PNL_SOL or pnl_sol > self.MAX_PNL_SOL:
            reasons.append(f"pnl_sol {pnl_sol} not in sanity bounds [{self.MIN_PNL_SOL}, {self.MAX_PNL_SOL}]")
        
        return reasons
    
    def _validate_slippage(self, slippage_bps: int) -> list[str]:
        """Validate slippage value."""
        reasons = []
        
        if slippage_bps < self.MIN_SLIPPAGE_BPS or slippage_bps > self.MAX_SLIPPAGE_BPS:
            reasons.append(f"slippage_bps {slippage_bps} not in [{self.MIN_SLIPPAGE_BPS}, {self.MAX_SLIPPAGE_BPS}]")
        
        return reasons
    
    def _validate_timestamp(self, timestamp_ms: int) -> list[str]:
        """Validate timestamp is within acceptable range."""
        reasons = []
        
        # Refresh cutoff (in case validator is long-lived)
        cutoff = self._compute_cutoff()
        
        if timestamp_ms < cutoff:
            reasons.append(f"timestamp_ms {timestamp_ms} is older than {self.MAX_TIMESTAMP_AGE_DAYS} days")
        
        # Also check for future timestamps (more than 1 hour in the future)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        max_future = now_ms + (60 * 60 * 1000)  # 1 hour
        
        if timestamp_ms > max_future:
            reasons.append(f"timestamp_ms {timestamp_ms} is in the future")
        
        return reasons
    
    def validate_trade(self, trade: dict) -> ValidationResult:
        """Validate a single trade.
        
        Args:
            trade: Dictionary with trade data (from TradeFeedback.model_dump())
        
        Returns:
            ValidationResult with is_valid, trade_id, and reasons
        """
        trade_id = trade.get("trade_id", "unknown")
        all_reasons = []
        
        # Validate features
        features = trade.get("features", [])
        all_reasons.extend(self._validate_features(features))
        
        # Validate confidence
        confidence = trade.get("confidence", 0.0)
        all_reasons.extend(self._validate_confidence(confidence))
        
        # Validate P&L
        pnl_sol = trade.get("pnl_sol", 0.0)
        all_reasons.extend(self._validate_pnl(pnl_sol))
        
        # Validate slippage
        slippage_bps = trade.get("slippage_bps", 0)
        all_reasons.extend(self._validate_slippage(slippage_bps))
        
        # Validate timestamp
        timestamp_ms = trade.get("timestamp_ms", 0)
        all_reasons.extend(self._validate_timestamp(timestamp_ms))
        
        return ValidationResult(
            is_valid=len(all_reasons) == 0,
            trade_id=trade_id,
            reasons=all_reasons
        )
    
    def validate_batch(self, trades: list[dict]) -> BatchValidationResult:
        """Validate a batch of trades.
        
        Args:
            trades: List of trade dictionaries
        
        Returns:
            BatchValidationResult with accepted and rejected trades
        """
        accepted = []
        rejected = []
        
        for trade in trades:
            result = self.validate_trade(trade)
            
            if result.is_valid:
                accepted.append(trade)
            else:
                rejected.append(result)
        
        return BatchValidationResult(accepted=accepted, rejected=rejected)


# Default validator instance
default_validator = TradeFeedbackValidator(strict=True)


# ============================================================================
# Tests (only run when pytest is available)
# ============================================================================

try:
    import pytest
    _PYTEST_AVAILABLE = True
except ImportError:
    _PYTEST_AVAILABLE = False


if _PYTEST_AVAILABLE:

    class TestTradeFeedbackValidator:
        """Tests for TradeFeedbackValidator."""
        
        @pytest.fixture
        def validator(self):
            return TradeFeedbackValidator(strict=True)
        
        @pytest.fixture
        def valid_trade(self):
            """A valid trade that should pass all validation."""
            return {
                "trade_id": "test-123",
                "timestamp_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
                "mint": "So11111111111111111111111111111111111111112",
                "features": [0.5, 0.3, 0.8, 0.1, 0.9],
                "feature_names": ["f1", "f2", "f3", "f4", "f5"],
                "decision": "buy",
                "confidence": 0.75,
                "amount_in_sol": 0.1,
                "amount_out_tokens": 1000000.0,
                "pnl_sol": 0.015,
                "slippage_bps": 45,
                "execution_latency_ms": 230,
                "model_version": 12,
            }
        
        def test_valid_trade_passes(self, validator, valid_trade):
            """Valid trade should pass validation."""
            result = validator.validate_trade(valid_trade)
            assert result.is_valid
            assert len(result.reasons) == 0
        
        def test_nan_in_features_rejected(self, validator, valid_trade):
            """NaN in features should be rejected."""
            valid_trade["features"] = [0.5, float('nan'), 0.8]
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert any("NaN" in r for r in result.reasons)
        
        def test_inf_in_features_rejected(self, validator, valid_trade):
            """Inf in features should be rejected."""
            valid_trade["features"] = [0.5, float('inf'), 0.8]
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert any("Inf" in r for r in result.reasons)
        
        def test_too_many_features_rejected(self, validator, valid_trade):
            """More than 50 features should be rejected."""
            valid_trade["features"] = [0.1] * 60
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert any("exceeds max" in r for r in result.reasons)
        
        def test_empty_features_rejected(self, validator, valid_trade):
            """Empty features should be rejected."""
            valid_trade["features"] = []
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert any("empty" in r for r in result.reasons)
        
        def test_confidence_out_of_range_high(self, validator, valid_trade):
            """Confidence > 1.0 should be rejected."""
            valid_trade["confidence"] = 1.5
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert any("confidence" in r.lower() for r in result.reasons)
        
        def test_confidence_out_of_range_low(self, validator, valid_trade):
            """Confidence < 0.0 should be rejected."""
            valid_trade["confidence"] = -0.5
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert any("confidence" in r.lower() for r in result.reasons)
        
        def test_confidence_nan_rejected(self, validator, valid_trade):
            """NaN confidence should be rejected."""
            valid_trade["confidence"] = float('nan')
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert any("confidence" in r.lower() for r in result.reasons)
        
        def test_pnl_out_of_bounds_high(self, validator, valid_trade):
            """PnL > 1000 SOL should be rejected."""
            valid_trade["pnl_sol"] = 5000.0
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert any("pnl_sol" in r and "sanity bounds" in r for r in result.reasons)
        
        def test_pnl_out_of_bounds_low(self, validator, valid_trade):
            """PnL < -1000 SOL should be rejected."""
            valid_trade["pnl_sol"] = -5000.0
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert any("pnl_sol" in r and "sanity bounds" in r for r in result.reasons)
        
        def test_pnl_nan_rejected(self, validator, valid_trade):
            """NaN PnL should be rejected."""
            valid_trade["pnl_sol"] = float('nan')
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert any("pnl_sol" in r.lower() for r in result.reasons)
        
        def test_slippage_negative_rejected(self, validator, valid_trade):
            """Negative slippage should be rejected."""
            valid_trade["slippage_bps"] = -10
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert any("slippage_bps" in r for r in result.reasons)
        
        def test_slippage_too_high_rejected(self, validator, valid_trade):
            """Slippage > 10000 bps should be rejected."""
            valid_trade["slippage_bps"] = 50000
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert any("slippage_bps" in r for r in result.reasons)
        
        def test_old_timestamp_rejected(self, validator, valid_trade):
            """Timestamp older than 30 days should be rejected."""
            # 60 days ago
            old_ts = int((datetime.now(timezone.utc) - timedelta(days=60)).timestamp() * 1000)
            valid_trade["timestamp_ms"] = old_ts
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert any("older than" in r for r in result.reasons)
        
        def test_future_timestamp_rejected(self, validator, valid_trade):
            """Future timestamp should be rejected."""
            # 2 hours in the future
            future_ts = int((datetime.now(timezone.utc) + timedelta(hours=2)).timestamp() * 1000)
            valid_trade["timestamp_ms"] = future_ts
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert any("future" in r for r in result.reasons)
        
        def test_batch_validation(self, validator, valid_trade):
            """Batch validation should separate valid and invalid trades."""
            # Create a batch with 2 valid and 2 invalid trades
            invalid_trade_1 = valid_trade.copy()
            invalid_trade_1["trade_id"] = "invalid-1"
            invalid_trade_1["confidence"] = 2.0  # Invalid
            
            invalid_trade_2 = valid_trade.copy()
            invalid_trade_2["trade_id"] = "invalid-2"
            invalid_trade_2["features"] = [float('nan')]  # Invalid
            
            valid_trade_1 = valid_trade.copy()
            valid_trade_1["trade_id"] = "valid-1"
            
            valid_trade_2 = valid_trade.copy()
            valid_trade_2["trade_id"] = "valid-2"
            
            trades = [invalid_trade_1, valid_trade_1, invalid_trade_2, valid_trade_2]
            result = validator.validate_batch(trades)
            
            assert result.accepted_count == 2
            assert result.rejected_count == 2
            assert len(result.rejection_reasons) == 2
            assert result.rejection_reasons[0]["trade_id"] == "invalid-1"
            assert result.rejection_reasons[1]["trade_id"] == "invalid-2"
        
        def test_multiple_validation_errors(self, validator, valid_trade):
            """Trade with multiple errors should report all of them."""
            valid_trade["confidence"] = 2.0  # Invalid
            valid_trade["pnl_sol"] = float('nan')  # Invalid
            valid_trade["slippage_bps"] = -10  # Invalid
            
            result = validator.validate_trade(valid_trade)
            assert not result.is_valid
            assert len(result.reasons) == 3
        
        def test_edge_case_confidence_bounds(self, validator, valid_trade):
            """Confidence at exact bounds should be valid."""
            # Lower bound
            valid_trade["confidence"] = 0.0
            result = validator.validate_trade(valid_trade)
            assert result.is_valid
            
            # Upper bound
            valid_trade["confidence"] = 1.0
            result = validator.validate_trade(valid_trade)
            assert result.is_valid
        
        def test_edge_case_pnl_bounds(self, validator, valid_trade):
            """PnL at exact bounds should be valid."""
            # Lower bound
            valid_trade["pnl_sol"] = -1000.0
            result = validator.validate_trade(valid_trade)
            assert result.is_valid
            
            # Upper bound
            valid_trade["pnl_sol"] = 1000.0
            result = validator.validate_trade(valid_trade)
            assert result.is_valid
        
        def test_edge_case_slippage_bounds(self, validator, valid_trade):
            """Slippage at exact bounds should be valid."""
            # Lower bound
            valid_trade["slippage_bps"] = 0
            result = validator.validate_trade(valid_trade)
            assert result.is_valid
            
            # Upper bound
            valid_trade["slippage_bps"] = 10000
            result = validator.validate_trade(valid_trade)
            assert result.is_valid
        
        def test_edge_case_max_features(self, validator, valid_trade):
            """Exactly 50 features should be valid."""
            valid_trade["features"] = [0.1] * 50
            result = validator.validate_trade(valid_trade)
            assert result.is_valid
