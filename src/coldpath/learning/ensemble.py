"""
Model Ensemble Coordinator.

Combines 4 models for comprehensive trading decisions:
1. Isolation Forest - Rug detection (anomaly-based)
2. LSTM - Price prediction (T+5/10/15 min)
3. XGBoost - Signal generation (BUY/HOLD/SELL)
4. Kelly Criterion - Position sizing

Veto Logic: High rug risk (>0.7) overrides BUY signals.

LLM Integration: Optional Opus 4.6 / Sonnet 4.5 hybrid orchestration
for enhanced decision explanations and complex analysis.
"""

import asyncio
import logging
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from coldpath.security import secure_load

from ..trading.kelly_criterion import KellyCriterion, PositionSizeResult
from ..training.isolation_forest_detector import (
    IsolationForestRugDetector,
    RugRiskResult,
)
from .lstm_predictor import LSTMPricePredictor, PricePrediction
from .signal_generator import Signal, SignalPrediction, XGBoostSignalGenerator

logger = logging.getLogger(__name__)


# Regime-adaptive veto thresholds
# Different market conditions require different risk tolerance
# UPDATED 2026-02-18: Lowered min_confidence thresholds to allow more trades through
# Analysis showed 60% threshold was rejecting ~50% of profitable signals
REGIME_VETO_THRESHOLDS = {
    "MEME_SEASON": {
        "veto": 0.82,  # High bar: most memecoins are volatile, not rugs
        "exit": 0.92,  # Only exit on near-certain rug detection
        "min_confidence": 0.25,  # Meme season = catch all pumps (was 0.30)
        "description": "High meme activity, optimized for catching pumps",
    },
    "BEAR_VOLATILE": {
        "veto": 0.75,  # Bear volatility != rug, need higher threshold
        "exit": 0.88,
        "min_confidence": 0.40,  # Defensive but still trading (was 0.45)
        "description": "Bear market, defensive but not paralyzed",
    },
    "HIGH_MEV": {
        "veto": 0.78,  # MEV is a cost issue, not a rug indicator
        "exit": 0.90,
        "min_confidence": 0.35,  # MEV is cost, not signal quality issue (was 0.40)
        "description": "High MEV activity, Jito bundles mitigate",
    },
    "NORMAL": {
        "veto": 0.78,  # LOWERED: reduce false positives on volatile tokens
        "exit": 0.92,  # Don't panic-exit good positions
        "min_confidence": 0.45,  # KEY FIX: Allow more trades through (was 0.60 - too restrictive)
        "description": "Normal market conditions, balanced risk/reward",
    },
    "LOW_LIQUIDITY": {
        "veto": 0.73,  # Thin markets need slightly more caution
        "exit": 0.88,  # But still don't force-exit prematurely
        "min_confidence": 0.35,  # Need trades to gather data (was 0.40)
        "description": "Low liquidity environment, careful but active",
    },
    "BULL_STABLE": {
        "veto": 0.82,  # Bull market: lean into opportunities
        "exit": 0.92,  # Hold winners longer
        "min_confidence": 0.25,  # Bull market = ride momentum (was 0.30)
        "description": "Bull market, lean into opportunities",
    },
}


class EnsembleSignal(Enum):
    """Final ensemble trading signal."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    EXIT = "exit"  # Emergency exit due to rug risk


@dataclass
class LLMInsight:
    """LLM-generated insights for ensemble decisions."""

    explanation: str  # Human-readable explanation
    risk_factors: list[str]  # Identified risk factors
    opportunity_factors: list[str]  # Identified opportunities
    recommendation_confidence: float  # LLM's confidence in recommendation
    model_used: str  # "opus" or "sonnet"
    latency_ms: float  # LLM response time

    def to_dict(self) -> dict[str, Any]:
        return {
            "explanation": self.explanation,
            "risk_factors": self.risk_factors,
            "opportunity_factors": self.opportunity_factors,
            "recommendation_confidence": self.recommendation_confidence,
            "model_used": self.model_used,
            "latency_ms": self.latency_ms,
        }


@dataclass
class EnsembleDecision:
    """Complete ensemble decision with all model outputs."""

    # Final decision
    signal: EnsembleSignal
    position_size: float  # Position size in SOL
    position_fraction: float  # Position as fraction of capital

    # Individual model outputs
    rug_risk: RugRiskResult
    price_prediction: PricePrediction
    signal_prediction: SignalPrediction
    kelly_result: PositionSizeResult

    # Metadata
    confidence: float  # Overall confidence (0-1)
    veto_applied: bool  # Whether rug veto was applied
    reasons: list[str]  # Explanation of decision

    # LLM integration (optional)
    llm_insight: LLMInsight | None = None
    hybrid_mode: bool = False  # Whether LLM was used

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "signal": self.signal.value,
            "position_size": self.position_size,
            "position_fraction": self.position_fraction,
            "confidence": self.confidence,
            "veto_applied": self.veto_applied,
            "reasons": self.reasons,
            "rug_risk": self.rug_risk.to_dict(),
            "price_prediction": self.price_prediction.to_dict(),
            "signal_prediction": self.signal_prediction.to_dict(),
            "kelly_result": self.kelly_result.to_dict(),
            "hybrid_mode": self.hybrid_mode,
        }
        if self.llm_insight:
            result["llm_insight"] = self.llm_insight.to_dict()
        return result

    @property
    def should_trade(self) -> bool:
        """Check if we should enter a trade."""
        return (
            self.signal
            in [
                EnsembleSignal.STRONG_BUY,
                EnsembleSignal.BUY,
            ]
            and self.position_size > 0
        )

    @property
    def should_exit(self) -> bool:
        """Check if we should exit existing positions."""
        return self.signal in [
            EnsembleSignal.EXIT,
            EnsembleSignal.STRONG_SELL,
        ]


class ModelEnsemble:
    """Ensemble coordinator for 4-model trading system.

    Combines:
    - Rug detection (safety)
    - Price prediction (direction)
    - Signal classification (timing)
    - Position sizing (risk management)

    Veto rules:
    - Rug risk > 0.85: EXIT (override all)
    - Rug risk > 0.70: REDUCE or HOLD (no new buys)
    - Rug risk > 0.50: Reduce position size

    Signal combination:
    - XGBoost BUY + LSTM upward + Low rug = STRONG_BUY
    - XGBoost BUY + Low rug = BUY
    - XGBoost SELL or High rug = SELL/EXIT
    - Otherwise = HOLD
    """

    def __init__(
        self,
        rug_detector: IsolationForestRugDetector | None = None,
        price_predictor: LSTMPricePredictor | None = None,
        signal_generator: XGBoostSignalGenerator | None = None,
        kelly_calculator: KellyCriterion | None = None,
        rug_veto_threshold: float = 0.80,  # Reduce false positives on volatile tokens
        rug_exit_threshold: float = 0.92,  # Only hard-exit on near-certain rugs
        min_confidence_for_trade: float = 0.35,  # Let more signals through, rely on rug veto
        default_capital: float = 10.0,
    ):
        """Initialize the ensemble.

        Args:
            rug_detector: Isolation Forest rug detector
            price_predictor: LSTM price predictor
            signal_generator: XGBoost signal generator
            kelly_calculator: Kelly Criterion calculator
            rug_veto_threshold: Rug risk above which BUY is vetoed
            rug_exit_threshold: Rug risk above which EXIT is triggered
            min_confidence_for_trade: Minimum confidence to trade
            default_capital: Default capital for sizing (SOL)
        """
        self.rug_detector = rug_detector or IsolationForestRugDetector()
        self.price_predictor = price_predictor or LSTMPricePredictor()
        self.signal_generator = signal_generator or XGBoostSignalGenerator()
        self.kelly = kelly_calculator or KellyCriterion()

        self.rug_veto_threshold = rug_veto_threshold
        self.rug_exit_threshold = rug_exit_threshold
        self.min_confidence_for_trade = min_confidence_for_trade
        self.default_capital = default_capital

        # Track if confidence was already applied in Kelly sizing
        self._confidence_applied_in_kelly = False

        # Regime-aware thresholds
        self.current_regime = "NORMAL"
        self._regime_history: list[tuple[str, float]] = []  # (regime, timestamp)

        self._last_decision: EnsembleDecision | None = None

    def set_regime(self, regime: str) -> None:
        """Dynamically adjust thresholds based on detected market regime.

        Args:
            regime: Market regime name (e.g., "MEME_SEASON", "BEAR_VOLATILE", "NORMAL")
        """
        import time

        thresholds = REGIME_VETO_THRESHOLDS.get(regime, REGIME_VETO_THRESHOLDS["NORMAL"])

        old_veto = self.rug_veto_threshold
        old_exit = self.rug_exit_threshold

        self.rug_veto_threshold = thresholds["veto"]
        self.rug_exit_threshold = thresholds["exit"]
        self.min_confidence_for_trade = thresholds["min_confidence"]
        self.current_regime = regime

        # Track regime history
        self._regime_history.append((regime, time.time()))

        logger.info(
            f"Regime updated: {self.current_regime} -> {regime}\n"
            f"  Thresholds: veto {old_veto:.2f}->{self.rug_veto_threshold:.2f}, "
            f"exit {old_exit:.2f}->{self.rug_exit_threshold:.2f}, "
            f"min_conf={self.min_confidence_for_trade:.2f}"
        )

    def get_regime_thresholds(self, regime: str | None = None) -> dict[str, float]:
        """Get thresholds for a specific regime.

        Args:
            regime: Regime name (uses current if None)

        Returns:
            Dictionary with veto, exit, and min_confidence thresholds
        """
        effective_regime = regime or self.current_regime or "NORMAL"
        thresholds = REGIME_VETO_THRESHOLDS.get(effective_regime, REGIME_VETO_THRESHOLDS["NORMAL"])
        return {
            "veto": thresholds["veto"],
            "exit": thresholds["exit"],
            "min_confidence": thresholds["min_confidence"],
        }

    def evaluate_with_regime(
        self,
        features_50: np.ndarray,
        price_sequence: np.ndarray,
        current_price: float,
        capital: float | None = None,
        regime: str | None = None,
    ) -> EnsembleDecision:
        """Evaluate with optional regime override.

        Args:
            features_50: 50-feature vector for rug detection and signals
            price_sequence: Price sequence for LSTM (60 x 5 features)
            current_price: Current price for predictions
            capital: Available capital for sizing
            regime: Optional regime override (temporarily applies regime thresholds)

        Returns:
            EnsembleDecision with final recommendation
        """
        # Apply temporary regime if provided
        if regime and regime != self.current_regime:
            old_regime = self.current_regime
            self.set_regime(regime)
            try:
                decision = self.evaluate(features_50, price_sequence, current_price, capital)
                decision.reasons.append(f"Regime: {regime}")
                return decision
            finally:
                # Restore original regime
                self.set_regime(old_regime)
        else:
            decision = self.evaluate(features_50, price_sequence, current_price, capital)
            decision.reasons.append(f"Regime: {self.current_regime}")
            return decision

    def evaluate(
        self,
        features_50: np.ndarray,
        price_sequence: np.ndarray,
        current_price: float,
        capital: float | None = None,
    ) -> EnsembleDecision:
        """Evaluate all models and produce ensemble decision.

        Args:
            features_50: 50-feature vector for rug detection and signals
            price_sequence: Price sequence for LSTM (60 x 5 features)
            current_price: Current price for predictions
            capital: Available capital for sizing (defaults to self.default_capital)

        Returns:
            EnsembleDecision with final recommendation
        """
        capital = capital or self.default_capital
        reasons = []

        # 1. Rug risk from Isolation Forest
        # Use first 33 features (liquidity + holder + risk)
        rug_features = self._extract_rug_features(features_50)
        rug_risk = self.rug_detector.predict_risk(rug_features)
        reasons.append(f"Rug risk: {rug_risk.risk_score:.2f} ({rug_risk.risk_level.value})")

        # 2. Price predictions from LSTM
        price_prediction = self.price_predictor.predict(price_sequence, current_price)
        reasons.append(
            f"Price trend: {price_prediction.trend}, T+15: {price_prediction.return_t15:+.1f}%"
        )

        # 3. Signal from XGBoost
        signal_prediction = self.signal_generator.predict(features_50)
        reasons.append(
            f"Signal: {signal_prediction.signal.name} (conf: {signal_prediction.confidence:.2f})"
        )

        # 4. Position sizing from Kelly
        effective_capital = capital if capital is not None else 1.0
        kelly_result = self.kelly.calculate_from_signal(
            signal_confidence=signal_prediction.confidence,
            signal_expected_return=signal_prediction.expected_return,
            capital=effective_capital,
        )
        reasons.append(f"Kelly: {kelly_result.position_fraction * 100:.1f}% of capital")

        # === ENSEMBLE LOGIC ===

        veto_applied = False
        final_signal = EnsembleSignal.HOLD
        position_size = 0.0
        position_fraction = 0.0

        # Check for rug veto (highest priority)
        if rug_risk.risk_score >= self.rug_exit_threshold:
            final_signal = EnsembleSignal.EXIT
            veto_applied = True
            reasons.append("VETO: High rug risk - EXIT")

        elif rug_risk.risk_score >= self.rug_veto_threshold:
            # No new buys, consider selling
            if signal_prediction.signal == Signal.SELL:
                final_signal = EnsembleSignal.SELL
            else:
                final_signal = EnsembleSignal.HOLD
            veto_applied = True
            reasons.append("VETO: Elevated rug risk - no new positions")

        else:
            # Normal signal processing
            if signal_prediction.signal == Signal.BUY:
                # Check LSTM confirmation
                if price_prediction.trend in ["strong_up", "up"]:
                    final_signal = EnsembleSignal.STRONG_BUY
                    reasons.append("LSTM confirms upward trend")
                elif price_prediction.trend == "neutral":
                    final_signal = EnsembleSignal.BUY
                else:
                    # XGBoost says BUY but LSTM says down - be cautious
                    final_signal = EnsembleSignal.HOLD
                    reasons.append("LSTM contradicts - holding")

            elif signal_prediction.signal == Signal.SELL:
                if price_prediction.trend in ["strong_down", "down"]:
                    final_signal = EnsembleSignal.STRONG_SELL
                else:
                    final_signal = EnsembleSignal.SELL

            else:  # HOLD
                final_signal = EnsembleSignal.HOLD

        # Calculate position size (only for BUY signals)
        if final_signal in [EnsembleSignal.STRONG_BUY, EnsembleSignal.BUY]:
            # UPDATED 2026-02-18: Smarter rug adjustment that doesn't crush good positions
            # Only apply rug penalty when rug risk is meaningfully high (>0.3)
            # This prevents double-penalizing: Kelly already has confidence baked in
            if rug_risk.risk_score > 0.3:
                # Gradual penalty: 0.3 risk -> 0.95x, 0.5 risk -> 0.85x, 0.7 risk -> 0.70x
                # Using (1 - (risk - 0.3) * 0.7) for smoother penalty curve
                rug_adjustment = max(0.70, 1.0 - (rug_risk.risk_score - 0.3) * 0.7)
            else:
                # Low rug risk: no penalty
                rug_adjustment = 1.0

            # Base Kelly sizing with smart rug adjustment (confidence already in Kelly)
            base_fraction = kelly_result.position_fraction
            adjusted_fraction = base_fraction * rug_adjustment

            # POSITION SIZE FLOOR: Never go below 50% of Kelly when we have positive edge
            # This ensures we capture edge even on marginal signals
            if kelly_result.is_valid and kelly_result.edge > 0.005:  # >0.5% edge
                min_floor = base_fraction * 0.50  # At least 50% of what Kelly says
                adjusted_fraction = max(adjusted_fraction, min_floor)

            # Apply minimum confidence check
            if signal_prediction.confidence < self.min_confidence_for_trade:
                adjusted_fraction = 0.0
                reasons.append(f"Position zeroed: confidence below {self.min_confidence_for_trade}")

            position_fraction = adjusted_fraction
            position_size = position_fraction * effective_capital

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            rug_risk, price_prediction, signal_prediction
        )

        decision = EnsembleDecision(
            signal=final_signal,
            position_size=position_size,
            position_fraction=position_fraction,
            rug_risk=rug_risk,
            price_prediction=price_prediction,
            signal_prediction=signal_prediction,
            kelly_result=kelly_result,
            confidence=overall_confidence,
            veto_applied=veto_applied,
            reasons=reasons,
        )

        self._last_decision = decision
        return decision

    def evaluate_simple(
        self,
        features_50: np.ndarray,
        capital: float | None = None,
    ) -> EnsembleDecision:
        """Simplified evaluation without LSTM (for tokens without price history).

        Args:
            features_50: 50-feature vector
            capital: Available capital

        Returns:
            EnsembleDecision based on rug and signal models only
        """
        capital = capital or self.default_capital
        reasons = []

        # 1. Rug detection
        rug_features = self._extract_rug_features(features_50)
        rug_risk = self.rug_detector.predict_risk(rug_features)
        reasons.append(f"Rug risk: {rug_risk.risk_score:.2f}")

        # 2. Signal generation
        signal_prediction = self.signal_generator.predict(features_50)
        reasons.append(f"Signal: {signal_prediction.signal.name}")

        # 3. Kelly sizing
        effective_capital_simple = capital if capital is not None else 1.0
        kelly_result = self.kelly.calculate_from_signal(
            signal_confidence=signal_prediction.confidence,
            signal_expected_return=signal_prediction.expected_return,
            capital=effective_capital_simple,
        )

        # Create mock price prediction (neutral)
        price_prediction = PricePrediction(
            price_t5=1.0,
            price_t10=1.0,
            price_t15=1.0,
            return_t5=0.0,
            return_t10=0.0,
            return_t15=0.0,
            confidence=0.0,
            current_price=1.0,
        )

        # Simplified logic
        veto_applied = False
        final_signal = EnsembleSignal.HOLD

        if rug_risk.risk_score >= self.rug_exit_threshold:
            final_signal = EnsembleSignal.EXIT
            veto_applied = True
        elif rug_risk.risk_score >= self.rug_veto_threshold:
            final_signal = EnsembleSignal.HOLD
            veto_applied = True
        elif signal_prediction.signal == Signal.BUY:
            final_signal = EnsembleSignal.BUY
        elif signal_prediction.signal == Signal.SELL:
            final_signal = EnsembleSignal.SELL

        position_fraction = 0.0
        if final_signal == EnsembleSignal.BUY:
            if rug_risk.risk_score > 0.3:
                rug_adjustment = max(0.70, 1.0 - (rug_risk.risk_score - 0.3) * 0.7)
            else:
                rug_adjustment = 1.0
            position_fraction = kelly_result.position_fraction * rug_adjustment

        position_size = position_fraction * effective_capital_simple

        confidence = (1.0 - rug_risk.risk_score) * signal_prediction.confidence

        return EnsembleDecision(
            signal=final_signal,
            position_size=position_size,
            position_fraction=position_fraction,
            rug_risk=rug_risk,
            price_prediction=price_prediction,
            signal_prediction=signal_prediction,
            kelly_result=kelly_result,
            confidence=confidence,
            veto_applied=veto_applied,
            reasons=reasons,
        )

    def _extract_rug_features(self, features_50: np.ndarray) -> np.ndarray:
        """Extract 33 rug detection features from 50-feature vector.

        Uses: Liquidity (0-11) + Holder (12-22) + Risk (33-42)
        """
        if features_50.ndim == 1:
            # Liquidity (12) + Holder (11) + Risk (10) = 33 features
            liquidity = features_50[0:12]
            holder = features_50[12:23]
            risk = features_50[33:43]
            return np.concatenate([liquidity, holder, risk])
        else:
            # Batch processing
            liquidity = features_50[:, 0:12]
            holder = features_50[:, 12:23]
            risk = features_50[:, 33:43]
            return np.hstack([liquidity, holder, risk])

    def _calculate_overall_confidence(
        self,
        rug_risk: RugRiskResult,
        price_prediction: PricePrediction,
        signal_prediction: SignalPrediction,
    ) -> float:
        """Calculate overall ensemble confidence.

        Weights (optimized for P&L):
        - Rug detection: 20% (reduced: rug is a veto, not primary driver)
        - Signal classification: 50% (increased: XGBoost is the strongest predictor)
        - Price prediction: 30% (LSTM trend confirmation)
        """
        # Rug confidence (inverted - lower risk = higher confidence)
        rug_confidence = 1.0 - rug_risk.risk_score

        # Signal confidence
        signal_confidence = signal_prediction.confidence

        # Price prediction confidence
        price_confidence = price_prediction.confidence

        # Weighted average: signal-dominant for better P&L
        overall = 0.20 * rug_confidence + 0.50 * signal_confidence + 0.30 * price_confidence

        return float(np.clip(overall, 0.0, 1.0))

    def get_model_status(self) -> dict[str, Any]:
        """Get status of all models including regime info."""
        return {
            "rug_detector": self.rug_detector.is_fitted,
            "price_predictor": self.price_predictor.is_fitted,
            "signal_generator": self.signal_generator.is_fitted,
            "kelly": True,  # Kelly doesn't need fitting
            "current_regime": self.current_regime,
            "thresholds": {
                "veto": self.rug_veto_threshold,
                "exit": self.rug_exit_threshold,
                "min_confidence": self.min_confidence_for_trade,
            },
        }

    def save(self, directory: str) -> None:
        """Save all models to directory."""
        import os

        os.makedirs(directory, exist_ok=True)

        self.rug_detector.save(os.path.join(directory, "rug_detector.pkl"))
        self.price_predictor.save(os.path.join(directory, "price_predictor.pkl"))
        self.signal_generator.save(os.path.join(directory, "signal_generator.pkl"))

        # Save ensemble config
        config = {
            "rug_veto_threshold": self.rug_veto_threshold,
            "rug_exit_threshold": self.rug_exit_threshold,
            "min_confidence_for_trade": self.min_confidence_for_trade,
            "default_capital": self.default_capital,
            "current_regime": self.current_regime,
        }
        with open(os.path.join(directory, "ensemble_config.pkl"), "wb") as f:
            pickle.dump(config, f)

        logger.info(f"Saved ensemble to {directory} (regime: {self.current_regime})")

    def load(self, directory: str) -> "ModelEnsemble":
        """Load all models from directory."""
        import os

        self.rug_detector.load(os.path.join(directory, "rug_detector.pkl"))
        self.price_predictor.load(os.path.join(directory, "price_predictor.pkl"))
        self.signal_generator.load(os.path.join(directory, "signal_generator.pkl"))

        config_path = os.path.join(directory, "ensemble_config.pkl")
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = secure_load(f)
            self.rug_veto_threshold = config.get("rug_veto_threshold", 0.70)
            self.rug_exit_threshold = config.get("rug_exit_threshold", 0.85)
            self.min_confidence_for_trade = config.get("min_confidence_for_trade", 0.6)
            self.default_capital = config.get("default_capital", 10.0)
            self.current_regime = config.get("current_regime", "NORMAL")

        logger.info(f"Loaded ensemble from {directory} (regime: {self.current_regime})")
        return self


def create_ensemble_from_training(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    price_sequences: np.ndarray,
    price_targets: np.ndarray,
    rug_labels: np.ndarray | None = None,
    *,
    xgboost_n_estimators: int = 500,
    lstm_epochs: int = 50,
) -> ModelEnsemble:
    """Create and train a complete ensemble from training data.

    Args:
        features_train: 50-feature training data
        labels_train: Signal labels (0=SELL, 1=HOLD, 2=BUY)
        price_sequences: LSTM training sequences
        price_targets: LSTM training targets (returns)
        rug_labels: Optional rug labels for Isolation Forest
        xgboost_n_estimators: XGBoost boosting rounds for signal generator
        lstm_epochs: Number of LSTM epochs (0 disables LSTM training)

    Returns:
        Trained ModelEnsemble
    """
    logger.info("Creating ensemble from training data")

    # Extract rug features (33 from 50)
    rug_features = np.hstack(
        [
            features_train[:, 0:12],  # Liquidity
            features_train[:, 12:23],  # Holder
            features_train[:, 33:43],  # Risk
        ]
    )

    # Train Isolation Forest
    rug_detector = IsolationForestRugDetector()
    rug_detector.fit(rug_features, rug_labels)

    # Train LSTM
    price_predictor = LSTMPricePredictor()
    if lstm_epochs > 0:
        # Stabilize synthetic outliers before torch training.
        safe_sequences = np.nan_to_num(price_sequences, nan=0.0, posinf=0.0, neginf=0.0)
        safe_targets = np.nan_to_num(price_targets, nan=0.0, posinf=0.0, neginf=0.0)
        safe_sequences = np.clip(safe_sequences, -50.0, 50.0)
        safe_targets = np.clip(safe_targets, -300.0, 300.0)
        if safe_targets.ndim == 1:
            safe_targets = np.column_stack([safe_targets, safe_targets, safe_targets])
        elif safe_targets.ndim == 2 and safe_targets.shape[1] == 1:
            safe_targets = np.repeat(safe_targets, 3, axis=1)

        try:
            price_predictor.fit(safe_sequences, safe_targets, epochs=lstm_epochs)
        except Exception as exc:
            logger.warning("LSTM training failed; falling back to non-LSTM mode: %s", exc)
    else:
        logger.info("Skipping LSTM training (lstm_epochs=0)")

    # Train XGBoost
    signal_generator = XGBoostSignalGenerator(n_estimators=xgboost_n_estimators)
    signal_generator.fit(features_train, labels_train)

    # Create ensemble
    ensemble = ModelEnsemble(
        rug_detector=rug_detector,
        price_predictor=price_predictor,
        signal_generator=signal_generator,
    )

    logger.info("Ensemble training complete")
    return ensemble


class HybridEnsemble:
    """Hybrid ML + LLM Ensemble for enhanced trading decisions.

    Combines the fast ML ensemble with Opus 4.6 / Sonnet 4.5 analysis
    for complex decisions requiring deeper understanding.

    Architecture:
    1. ML Ensemble runs first (fast, <100ms)
    2. LLM analysis runs in parallel for complex cases
    3. Results are aggregated for final decision

    Use Cases:
    - Complex tokens (unusual patterns)
    - High-value decisions (large positions)
    - Explanation generation
    - Regime analysis
    """

    def __init__(
        self,
        ensemble: ModelEnsemble | None = None,
        llm_orchestrator: Any | None = None,  # LLMOrchestrator
        enable_llm: bool = True,
        llm_threshold: float = 0.5,  # Confidence threshold for LLM
        position_threshold: float = 1.0,  # SOL threshold for LLM
        max_llm_latency_ms: float = 500.0,
    ):
        """Initialize hybrid ensemble.

        Args:
            ensemble: Base ML ensemble
            llm_orchestrator: LLM orchestrator for hybrid decisions
            enable_llm: Whether to enable LLM integration
            llm_threshold: Trigger LLM when ML confidence below this
            position_threshold: Trigger LLM for positions above this (SOL)
            max_llm_latency_ms: Maximum wait time for LLM response
        """
        self.ensemble = ensemble or ModelEnsemble()
        self._llm_orchestrator = llm_orchestrator
        self.enable_llm = enable_llm
        self.llm_threshold = llm_threshold
        self.position_threshold = position_threshold
        self.max_llm_latency_ms = max_llm_latency_ms

        self._llm_call_count = 0
        self._llm_total_latency_ms = 0.0

    @property
    def llm_orchestrator(self):
        """Lazy load LLM orchestrator."""
        if self._llm_orchestrator is None and self.enable_llm:
            try:
                from ..ai.claude_client import ClaudeClient
                from ..ai.llm_orchestrator import LLMOrchestrator

                # Create a ClaudeClient for the orchestrator
                client = ClaudeClient()  # Uses ANTHROPIC_API_KEY env var
                self._llm_orchestrator = LLMOrchestrator(client=client)
            except ImportError as e:
                logger.warning(f"LLM Orchestrator not available, disabling LLM integration: {e}")
                self.enable_llm = False
            except Exception as e:
                logger.warning(f"Failed to initialize LLM Orchestrator: {e}")
                self.enable_llm = False
        return self._llm_orchestrator

    def _should_use_llm(
        self,
        decision: EnsembleDecision,
        force_llm: bool = False,
    ) -> bool:
        """Determine if LLM should be invoked for this decision."""
        if force_llm:
            return True

        if not self.enable_llm:
            return False

        # Low confidence decisions benefit from LLM analysis
        if decision.confidence < self.llm_threshold:
            return True

        # Large position sizes warrant deeper analysis
        if decision.position_size >= self.position_threshold:
            return True

        # Conflicting signals need explanation
        if decision.veto_applied:
            return True

        # Edge cases: near-threshold rug risk
        if 0.5 <= decision.rug_risk.risk_score <= 0.75:
            return True

        return False

    async def evaluate_async(
        self,
        features_50: np.ndarray,
        price_sequence: np.ndarray,
        current_price: float,
        capital: float | None = None,
        token_metadata: dict[str, Any] | None = None,
        force_llm: bool = False,
    ) -> EnsembleDecision:
        """Evaluate with optional LLM enhancement (async).

        Args:
            features_50: 50-feature vector
            price_sequence: Price sequence for LSTM
            current_price: Current price
            capital: Available capital
            token_metadata: Optional metadata (symbol, mint, etc.)
            force_llm: Force LLM analysis regardless of thresholds

        Returns:
            EnsembleDecision with optional LLM insights
        """
        # Step 1: Run ML ensemble (fast path)
        ml_decision = self.ensemble.evaluate(
            features_50=features_50,
            price_sequence=price_sequence,
            current_price=current_price,
            capital=capital,
        )

        # Step 2: Check if LLM analysis is needed
        if not self._should_use_llm(ml_decision, force_llm):
            return ml_decision

        # Step 3: Run LLM analysis in parallel (with timeout)
        llm_insight = await self._get_llm_insight(
            ml_decision=ml_decision,
            features_50=features_50,
            token_metadata=token_metadata,
        )

        # Step 4: Aggregate results
        if llm_insight:
            ml_decision.llm_insight = llm_insight
            ml_decision.hybrid_mode = True

            # Optionally adjust confidence based on LLM
            if llm_insight.recommendation_confidence > 0:
                # Weighted average: 70% ML, 30% LLM
                ml_decision.confidence = (
                    0.7 * ml_decision.confidence + 0.3 * llm_insight.recommendation_confidence
                )

            # Add LLM explanation to reasons
            if llm_insight.explanation:
                ml_decision.reasons.append(f"LLM: {llm_insight.explanation[:100]}")

        return ml_decision

    async def _get_llm_insight(
        self,
        ml_decision: EnsembleDecision,
        features_50: np.ndarray,
        token_metadata: dict[str, Any] | None = None,
    ) -> LLMInsight | None:
        """Get LLM insight with timeout."""
        import time

        if not self.llm_orchestrator:
            return None

        start_time = time.time()

        try:
            # Build context for LLM
            context = self._build_llm_context(ml_decision, features_50, token_metadata)

            # Call LLM with timeout
            result = await asyncio.wait_for(
                self._call_llm_orchestrator(context),
                timeout=self.max_llm_latency_ms / 1000.0,
            )

            latency_ms = (time.time() - start_time) * 1000
            self._llm_call_count += 1
            self._llm_total_latency_ms += latency_ms

            if result:
                return LLMInsight(
                    explanation=result.get("explanation", ""),
                    risk_factors=result.get("risk_factors", []),
                    opportunity_factors=result.get("opportunity_factors", []),
                    recommendation_confidence=result.get("confidence", 0.0),
                    model_used=result.get("model", "sonnet"),
                    latency_ms=latency_ms,
                )

        except TimeoutError:
            logger.warning(f"LLM timeout after {self.max_llm_latency_ms}ms")
        except Exception as e:
            logger.error(f"LLM error: {e}")

        return None

    def _build_llm_context(
        self,
        ml_decision: EnsembleDecision,
        features_50: np.ndarray,
        token_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build context dictionary for LLM analysis."""
        return {
            "signal": ml_decision.signal.value,
            "confidence": ml_decision.confidence,
            "rug_risk": ml_decision.rug_risk.risk_score,
            "rug_level": ml_decision.rug_risk.risk_level.value,
            "price_trend": ml_decision.price_prediction.trend,
            "expected_return_t5": ml_decision.price_prediction.return_t5,
            "expected_return_t10": ml_decision.price_prediction.return_t10,
            "expected_return_t15": ml_decision.price_prediction.return_t15,
            "price_predictions": {
                "t5": ml_decision.price_prediction.return_t5,
                "t10": ml_decision.price_prediction.return_t10,
                "t15": ml_decision.price_prediction.return_t15,
            },
            "xgboost_signal": ml_decision.signal_prediction.signal.name,
            "kelly_fraction": ml_decision.kelly_result.position_fraction,
            "position_size": ml_decision.position_size,
            "veto_applied": ml_decision.veto_applied,
            "reasons": ml_decision.reasons,
            "features": {
                "liquidity_score": float(features_50[0]) if len(features_50) > 0 else 0,
                "holder_concentration": float(features_50[15]) if len(features_50) > 15 else 0,
                "volatility": float(features_50[23]) if len(features_50) > 23 else 0,
            },
            "token": token_metadata or {},
        }

    async def _call_llm_orchestrator(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Call LLM orchestrator for analysis."""
        try:
            from ..ai.llm_orchestrator import LLMOrchestrator, MLContext

            if not isinstance(self._llm_orchestrator, LLMOrchestrator):
                return None

            # FIXED: _build_llm_context returns a FLAT dict, not nested
            # Read keys directly from the flat context
            ml_context = MLContext(
                ensemble_score=context.get("confidence", 0.5),
                confidence=context.get("confidence", 0.5),
                rug_risk=context.get("rug_risk", 0.0),
                signal=context.get("xgboost_signal", context.get("signal", "HOLD")),
                price_predictions=context.get("price_predictions", {}),
                regime=context.get("regime", "NORMAL"),
                feature_importances=context.get("features", {}),
                reasons=context.get("reasons", []),
            )

            # Get hybrid decision using correct method name
            token_symbol = context.get("token", {}).get("symbol", "token")
            result = await self._llm_orchestrator.make_hybrid_decision(
                query=f"Analyze trading decision for {token_symbol}",
                ml_context=ml_context,
            )

            return {
                "action": result.action,
                "confidence": result.confidence,
                "position_size": result.position_size,
                "llm_reasoning": result.llm_reasoning,
                "agreement_score": result.agreement_score,
            }

        except Exception as e:
            logger.error(f"LLM orchestrator call failed: {e}")
            return None

    def evaluate(
        self,
        features_50: np.ndarray,
        price_sequence: np.ndarray,
        current_price: float,
        capital: float | None = None,
    ) -> EnsembleDecision:
        """Synchronous evaluate (ML only, no LLM)."""
        return self.ensemble.evaluate(
            features_50=features_50,
            price_sequence=price_sequence,
            current_price=current_price,
            capital=capital,
        )

    def get_llm_stats(self) -> dict[str, Any]:
        """Get LLM usage statistics."""
        avg_latency = (
            self._llm_total_latency_ms / self._llm_call_count if self._llm_call_count > 0 else 0.0
        )
        return {
            "llm_enabled": self.enable_llm,
            "llm_calls": self._llm_call_count,
            "avg_latency_ms": avg_latency,
            "total_latency_ms": self._llm_total_latency_ms,
        }

    def get_model_status(self) -> dict[str, Any]:
        """Get status of all models including LLM."""
        status = self.ensemble.get_model_status()
        status["llm_enabled"] = self.enable_llm
        status["llm_orchestrator"] = self._llm_orchestrator is not None
        return status

    def save(self, directory: str) -> None:
        """Save ensemble (LLM orchestrator is not persisted)."""
        self.ensemble.save(directory)

        # Save hybrid config
        import os

        config = {
            "enable_llm": self.enable_llm,
            "llm_threshold": self.llm_threshold,
            "position_threshold": self.position_threshold,
            "max_llm_latency_ms": self.max_llm_latency_ms,
        }
        with open(os.path.join(directory, "hybrid_config.pkl"), "wb") as f:
            pickle.dump(config, f)

    def load(self, directory: str) -> "HybridEnsemble":
        """Load ensemble from directory."""
        import os

        self.ensemble.load(directory)

        config_path = os.path.join(directory, "hybrid_config.pkl")
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = secure_load(f)
            self.enable_llm = config.get("enable_llm", True)
            self.llm_threshold = config.get("llm_threshold", 0.5)
            self.position_threshold = config.get("position_threshold", 1.0)
            self.max_llm_latency_ms = config.get("max_llm_latency_ms", 500.0)

        return self


def create_hybrid_ensemble(
    ensemble: ModelEnsemble | None = None,
    enable_llm: bool = True,
) -> HybridEnsemble:
    """Factory function to create a hybrid ensemble.

    Args:
        ensemble: Pre-trained ML ensemble (or create new)
        enable_llm: Whether to enable LLM integration

    Returns:
        HybridEnsemble ready for use
    """
    return HybridEnsemble(
        ensemble=ensemble,
        enable_llm=enable_llm,
    )
