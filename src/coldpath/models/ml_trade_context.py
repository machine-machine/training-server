"""
ML Trade Context - Per-trade ML predictions for paper fill simulation

Python port of EngineHotPath/src/execution/ml_context.rs
Bridges ColdPath ML models (FraudModel, RegimeDetector, ProfitabilityLearner)
with HotPath paper fill simulation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MarketRegime(Enum):
    """Market regime classifications from ColdPath RegimeDetector"""

    BULL = "bull"
    BEAR = "bear"
    CHOP = "chop"
    MEV_HEAVY = "mev_heavy"

    @classmethod
    def from_str(cls, s: str) -> "MarketRegime":
        """Convert from string (matches Python enum values)"""
        mapping = {
            "bull": cls.BULL,
            "bear": cls.BEAR,
            "chop": cls.CHOP,
            "mev_heavy": cls.MEV_HEAVY,
        }
        return mapping.get(s.lower(), cls.CHOP)

    def as_str(self) -> str:
        return self.value


@dataclass
class RegimeAdjustments:
    """
    Regime-specific parameter adjustments.

    These multipliers are applied to base paper fill parameters
    to simulate regime-dependent market behavior.
    """

    slippage_multiplier: float = 1.0
    inclusion_penalty: float = 0.0
    mev_exposure_multiplier: float = 1.0
    confidence_multiplier: float = 1.0
    position_limit_multiplier: float = 1.0

    @classmethod
    def for_regime(cls, regime: MarketRegime) -> "RegimeAdjustments":
        """Create adjustments for a specific regime"""
        if regime == MarketRegime.BULL:
            return cls(
                slippage_multiplier=1.0,
                inclusion_penalty=0.0,
                mev_exposure_multiplier=1.0,
                confidence_multiplier=1.2,
                position_limit_multiplier=1.0,
            )
        elif regime == MarketRegime.BEAR:
            return cls(
                slippage_multiplier=1.2,
                inclusion_penalty=0.05,
                mev_exposure_multiplier=1.0,
                confidence_multiplier=0.8,
                position_limit_multiplier=0.7,
            )
        elif regime == MarketRegime.MEV_HEAVY:
            return cls(
                slippage_multiplier=1.5,
                inclusion_penalty=0.1,
                mev_exposure_multiplier=2.0,
                confidence_multiplier=0.9,
                position_limit_multiplier=0.6,
            )
        else:  # CHOP
            return cls(
                slippage_multiplier=1.0,
                inclusion_penalty=0.0,
                mev_exposure_multiplier=1.0,
                confidence_multiplier=1.0,
                position_limit_multiplier=0.5,
            )

    def with_fraud_adjustment(self, fraud_score: float) -> "RegimeAdjustments":
        """Merge with fraud-based adjustments"""
        # Fraud increases slippage and reduces inclusion
        # fraud_score 0.0 -> no change, 1.0 -> 50% more slippage, 30% inclusion penalty
        self.slippage_multiplier *= 1.0 + (fraud_score * 0.5)
        self.inclusion_penalty += fraud_score * 0.30
        self.inclusion_penalty = min(self.inclusion_penalty, 0.4)  # Cap at 40%
        return self

    def to_dict(self) -> dict[str, float]:
        """Serialize into the JSON-safe wire shape used at the IPC boundary."""
        return {
            "slippage_multiplier": float(self.slippage_multiplier),
            "inclusion_penalty": float(self.inclusion_penalty),
            "mev_exposure_multiplier": float(self.mev_exposure_multiplier),
            "confidence_multiplier": float(self.confidence_multiplier),
            "position_limit_multiplier": float(self.position_limit_multiplier),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RegimeAdjustments":
        """Restore from the JSON-safe wire shape used at the IPC boundary."""
        payload = data or {}
        return cls(
            slippage_multiplier=float(payload.get("slippage_multiplier", 1.0)),
            inclusion_penalty=float(payload.get("inclusion_penalty", 0.0)),
            mev_exposure_multiplier=float(payload.get("mev_exposure_multiplier", 1.0)),
            confidence_multiplier=float(payload.get("confidence_multiplier", 1.0)),
            position_limit_multiplier=float(payload.get("position_limit_multiplier", 1.0)),
        )


@dataclass
class DeployerRisk:
    """Deployer-specific risk factors"""

    previous_tokens: int = 0
    rug_count: int = 0
    rug_rate: float = 0.0
    avg_token_lifespan_hours: float = 0.0

    def risk_score(self) -> float:
        """Calculate deployer risk score (0.0-1.0)"""
        if self.previous_tokens == 0:
            return 0.3  # Unknown deployer = moderate risk

        rug_factor = min(self.rug_rate, 1.0)
        experience_factor = 0.2 if self.previous_tokens < 3 else 0.0
        lifespan_factor = 0.2 if self.avg_token_lifespan_hours < 24.0 else 0.0

        return min(rug_factor + experience_factor + lifespan_factor, 1.0)

    def to_dict(self) -> dict[str, float | int]:
        """Serialize into the JSON-safe wire shape used at the IPC boundary."""
        return {
            "previous_tokens": int(self.previous_tokens),
            "rug_count": int(self.rug_count),
            "rug_rate": float(self.rug_rate),
            "avg_token_lifespan_hours": float(self.avg_token_lifespan_hours),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Optional["DeployerRisk"]:
        """Restore from the JSON-safe wire shape used at the IPC boundary."""
        if not data:
            return None
        return cls(
            previous_tokens=int(data.get("previous_tokens", 0)),
            rug_count=int(data.get("rug_count", 0)),
            rug_rate=float(data.get("rug_rate", 0.0)),
            avg_token_lifespan_hours=float(data.get("avg_token_lifespan_hours", 0.0)),
        )


@dataclass
class MLTradeContext:
    """
    Per-trade ML context from ColdPath models.

    This structure flows from ColdPath scoring to HotPath paper fill simulation,
    enabling fraud-aware slippage, regime-specific parameters, and
    profitability-conditioned fill expectations.
    """

    fraud_score: float = 0.0
    regime: MarketRegime = field(default_factory=lambda: MarketRegime.CHOP)
    regime_confidence: float = 0.5
    profitability_confidence: float = 0.5
    adjustments: RegimeAdjustments = field(default_factory=RegimeAdjustments)
    time_in_regime: int = 0
    transition_probability: float = 0.25
    deployer_risk: DeployerRisk | None = None
    survivorship_discount: float = 0.15

    @classmethod
    def create(
        cls,
        fraud_score: float,
        regime: MarketRegime,
        regime_confidence: float,
        profitability_confidence: float,
    ) -> "MLTradeContext":
        """Create a new MLTradeContext with basic fraud and regime info"""
        adjustments = RegimeAdjustments.for_regime(regime).with_fraud_adjustment(fraud_score)
        survivorship_discount = cls._compute_survivorship_discount(fraud_score, regime)

        return cls(
            fraud_score=fraud_score,
            regime=regime,
            regime_confidence=regime_confidence,
            profitability_confidence=profitability_confidence,
            adjustments=adjustments,
            time_in_regime=0,
            transition_probability=0.25,
            deployer_risk=None,
            survivorship_discount=survivorship_discount,
        )

    @staticmethod
    def _compute_survivorship_discount(fraud_score: float, regime: MarketRegime) -> float:
        """Compute dynamic survivorship discount"""
        base_discount = 0.08  # Base 8%
        fraud_penalty = fraud_score * 0.20  # Up to 20% more for high fraud
        regime_penalty = {
            MarketRegime.BEAR: 0.05,
            MarketRegime.MEV_HEAVY: 0.07,
        }.get(regime, 0.0)

        return min(base_discount + fraud_penalty + regime_penalty, 0.50)  # Cap at 50%

    def effective_slippage_multiplier(self) -> float:
        """Get effective slippage multiplier combining regime and fraud"""
        return self.adjustments.slippage_multiplier

    def effective_inclusion_multiplier(self) -> float:
        """Get effective inclusion multiplier (1.0 - penalty)"""
        return 1.0 - self.adjustments.inclusion_penalty

    def effective_mev_multiplier(self) -> float:
        """Get effective MEV exposure multiplier"""
        return self.adjustments.mev_exposure_multiplier

    def confidence_fill_factor(self) -> float:
        """
        Get confidence-adjusted fill quality factor.
        Low confidence = expect worse fills (more realistic)
        Returns multiplier for amount_out: 0.8 (low confidence) to 1.2 (high confidence)
        """
        return 0.8 + (self.profitability_confidence * 0.4)

    def is_high_risk(self) -> bool:
        """Check if this trade should be more conservatively filled"""
        return self.fraud_score > 0.3 or self.regime == MarketRegime.MEV_HEAVY

    def is_regime_stable(self) -> bool:
        """Check if regime is stable enough for confident trading"""
        return self.regime_confidence > 0.7 and self.transition_probability < 0.3

    def to_dict(self) -> dict[str, Any]:
        """Serialize into the JSON-safe wire shape shared by live and backtest paths."""
        payload: dict[str, Any] = {
            "fraud_score": float(self.fraud_score),
            "regime": self.regime.as_str(),
            "regime_confidence": float(self.regime_confidence),
            "profitability_confidence": float(self.profitability_confidence),
            "adjustments": self.adjustments.to_dict(),
            "time_in_regime": int(self.time_in_regime),
            "transition_probability": float(self.transition_probability),
            "survivorship_discount": float(self.survivorship_discount),
        }
        if self.deployer_risk is not None:
            payload["deployer_risk"] = self.deployer_risk.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Optional["MLTradeContext"]:
        """Restore from the JSON-safe wire shape shared by live and backtest paths."""
        if not data:
            return None

        return cls(
            fraud_score=float(data.get("fraud_score", 0.0)),
            regime=MarketRegime.from_str(str(data.get("regime", MarketRegime.CHOP.value))),
            regime_confidence=float(data.get("regime_confidence", 0.5)),
            profitability_confidence=float(data.get("profitability_confidence", 0.5)),
            adjustments=RegimeAdjustments.from_dict(data.get("adjustments")),
            time_in_regime=int(data.get("time_in_regime", 0)),
            transition_probability=float(data.get("transition_probability", 0.25)),
            deployer_risk=DeployerRisk.from_dict(data.get("deployer_risk")),
            survivorship_discount=float(data.get("survivorship_discount", 0.15)),
        )


@dataclass
class MLPaperTradingConfig:
    """Configuration for ML-aware paper trading"""

    enable_ml_context: bool = True
    enable_monte_carlo: bool = True
    enable_online_learning: bool = True
    fraud_inclusion_penalty_max: float = 0.30
    fraud_slippage_multiplier_max: float = 1.50
    mev_heavy_slippage_multiplier: float = 1.5
    mev_heavy_mev_multiplier: float = 2.0
    bear_inclusion_penalty: float = 0.05
    online_learning_interval_seconds: int = 60
    paper_outcome_base_weight: float = 0.5
    paper_confirmed_weight_bonus: float = 0.3
    drift_recalibration_threshold: float = 0.7
    walk_forward_splits: int = 10
    purge_gap_hours: int = 1
