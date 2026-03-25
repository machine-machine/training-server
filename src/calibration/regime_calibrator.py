"""
Regime-Triggered Calibrator - Fast-path parameter recalibration on regime changes.

Provides immediate parameter adjustments when market regime changes:
- Detect regime transitions (volume shock, volatility spike, MEV surge)
- Fast-path EMA updates without full L-BFGS-B optimization
- Trigger immediate parameter push on regime change
- Gradual transition blending for stability

Integrates with RegimeDetector and BiasCalibrator.
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RegimeTransitionType(Enum):
    """Type of regime transition detected."""
    NONE = "none"
    VOLUME_SHOCK = "volume_shock"
    VOLATILITY_SPIKE = "volatility_spike"
    CONGESTION_SURGE = "congestion_surge"
    MEV_SURGE = "mev_surge"
    LIQUIDITY_DRAIN = "liquidity_drain"
    GRADUAL_TRANSITION = "gradual_transition"


@dataclass
class RegimeTransition:
    """Details of a detected regime transition."""
    transition_type: RegimeTransitionType
    from_regime: str
    to_regime: str
    confidence: float
    timestamp_ms: int
    trigger_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class RegimeCalibrationConfig:
    """Configuration for regime-triggered calibration."""
    # Detection thresholds
    volume_shock_z_threshold: float = 2.5  # Z-score for volume shock
    volatility_spike_z_threshold: float = 2.0  # Z-score for volatility spike
    congestion_threshold: float = 0.8  # Congestion level 0-1
    mev_rate_threshold: float = 0.15  # MEV sandwich rate
    liquidity_drain_pct: float = 0.30  # 30% liquidity drop

    # EMA parameters for fast updates
    ema_alpha_fast: float = 0.3  # Fast EMA for immediate response
    ema_alpha_slow: float = 0.1  # Slow EMA for stability

    # Transition blending
    blend_duration_seconds: float = 60.0  # Time to fully transition
    min_blend_factor: float = 0.2  # Minimum old parameter weight

    # Calibration intervals
    check_interval_seconds: float = 5.0  # How often to check for regime change
    min_time_between_calibrations_seconds: float = 30.0

    # Parameter bounds (relative to base)
    slippage_multiplier_bounds: Tuple[float, float] = (0.5, 2.5)
    inclusion_multiplier_bounds: Tuple[float, float] = (0.5, 1.2)
    mev_multiplier_bounds: Tuple[float, float] = (0.5, 3.0)
    latency_multiplier_bounds: Tuple[float, float] = (0.8, 2.0)


@dataclass
class CalibratedParameters:
    """Parameters calibrated for current regime."""
    regime: str
    regime_confidence: float

    # Fill simulation parameters
    slippage_multiplier: float = 1.0
    inclusion_multiplier: float = 1.0
    mev_exposure_multiplier: float = 1.0
    latency_multiplier: float = 1.0

    # Survivorship adjustment
    survivorship_discount_adjustment: float = 0.0

    # Confidence scaling
    confidence_multiplier: float = 1.0

    # Metadata
    calibrated_at_ms: int = 0
    transition_type: Optional[RegimeTransitionType] = None
    samples_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "slippage_multiplier": self.slippage_multiplier,
            "inclusion_multiplier": self.inclusion_multiplier,
            "mev_exposure_multiplier": self.mev_exposure_multiplier,
            "latency_multiplier": self.latency_multiplier,
            "survivorship_discount_adjustment": self.survivorship_discount_adjustment,
            "confidence_multiplier": self.confidence_multiplier,
            "calibrated_at_ms": self.calibrated_at_ms,
            "transition_type": self.transition_type.value if self.transition_type else None,
            "samples_used": self.samples_used,
        }


@dataclass
class MarketMetrics:
    """Current market metrics for regime detection."""
    timestamp_ms: int

    # Volume metrics
    volume_1h_usd: float = 0.0
    volume_24h_usd: float = 0.0
    volume_z_score: float = 0.0

    # Volatility metrics
    volatility_1h: float = 0.0
    volatility_24h: float = 0.0
    volatility_z_score: float = 0.0

    # Network metrics
    congestion_level: float = 0.0
    slot_rate: float = 2.5  # slots/second

    # MEV metrics
    sandwich_rate: float = 0.0  # % of trades sandwiched
    mev_volume_pct: float = 0.0

    # Liquidity metrics
    total_liquidity_usd: float = 0.0
    liquidity_change_1h_pct: float = 0.0


class RegimeTriggeredCalibrator:
    """Fast-path parameter recalibration on regime changes.

    Monitors market conditions and triggers immediate parameter
    updates when regime changes are detected, without waiting
    for full model retraining.
    """

    def __init__(
        self,
        config: Optional[RegimeCalibrationConfig] = None,
        on_calibration: Optional[Callable[[CalibratedParameters], None]] = None,
    ):
        self.config = config or RegimeCalibrationConfig()
        self.on_calibration = on_calibration

        # Current state
        self._current_regime = "chop"
        self._current_params = CalibratedParameters(regime="chop", regime_confidence=0.5)

        # History for z-score calculation
        self._volume_history: List[float] = []
        self._volatility_history: List[float] = []
        self._max_history_size = 1000

        # EMA state
        self._ema_slippage = 1.0
        self._ema_inclusion = 1.0
        self._ema_mev = 1.0
        self._ema_latency = 1.0

        # Transition state
        self._transition_start_ms: Optional[int] = None
        self._transition_from_params: Optional[CalibratedParameters] = None
        self._transition_to_params: Optional[CalibratedParameters] = None
        self._active_transition: Optional[RegimeTransition] = None

        # Tracking
        self._last_calibration_ms = 0
        self._calibration_count = 0
        self._running = False

    async def start_monitoring(self, metrics_source: Callable[[], MarketMetrics]) -> None:
        """Start continuous regime monitoring.

        Args:
            metrics_source: Async callable that returns current MarketMetrics
        """
        self._running = True
        logger.info("Regime calibrator started")

        while self._running:
            try:
                metrics = metrics_source()
                await self._check_and_calibrate(metrics)
            except Exception as e:
                logger.error(f"Calibration check error: {e}")

            await asyncio.sleep(self.config.check_interval_seconds)

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._running = False
        logger.info("Regime calibrator stopped")

    async def _check_and_calibrate(self, metrics: MarketMetrics) -> Optional[CalibratedParameters]:
        """Check for regime change and calibrate if needed."""
        # Update history
        self._update_history(metrics)

        # Detect transition
        transition = self._detect_transition(metrics)

        if transition.transition_type != RegimeTransitionType.NONE:
            # Check minimum time between calibrations
            now_ms = int(datetime.now().timestamp() * 1000)
            if now_ms - self._last_calibration_ms < self.config.min_time_between_calibrations_seconds * 1000:
                return None

            # Perform fast calibration
            new_params = self._fast_calibrate(metrics, transition)

            # Start transition blending
            self._start_transition(new_params)

            # Notify callback
            if self.on_calibration:
                try:
                    self.on_calibration(new_params)
                except Exception as e:
                    logger.error(f"Calibration callback error: {e}")

            self._last_calibration_ms = now_ms
            self._calibration_count += 1

            logger.info(
                f"Regime calibration #{self._calibration_count}: "
                f"{transition.from_regime} -> {transition.to_regime} "
                f"({transition.transition_type.value})"
            )

            return new_params

        return None

    def _update_history(self, metrics: MarketMetrics) -> None:
        """Update metric history for z-score calculation."""
        self._volume_history.append(metrics.volume_1h_usd)
        self._volatility_history.append(metrics.volatility_1h)

        # Limit history size
        if len(self._volume_history) > self._max_history_size:
            self._volume_history = self._volume_history[-self._max_history_size:]
            self._volatility_history = self._volatility_history[-self._max_history_size:]

    def _detect_transition(self, metrics: MarketMetrics) -> RegimeTransition:
        """Detect if a regime transition has occurred."""
        now_ms = int(datetime.now().timestamp() * 1000)

        # Calculate z-scores
        volume_z = self._calculate_z_score(metrics.volume_1h_usd, self._volume_history)
        volatility_z = self._calculate_z_score(metrics.volatility_1h, self._volatility_history)

        # Check for volume shock
        if abs(volume_z) >= self.config.volume_shock_z_threshold:
            new_regime = "bull" if volume_z > 0 else "bear"
            return RegimeTransition(
                transition_type=RegimeTransitionType.VOLUME_SHOCK,
                from_regime=self._current_regime,
                to_regime=new_regime,
                confidence=min(1.0, abs(volume_z) / 4.0),
                timestamp_ms=now_ms,
                trigger_values={"volume_z": volume_z},
            )

        # Check for volatility spike
        if volatility_z >= self.config.volatility_spike_z_threshold:
            return RegimeTransition(
                transition_type=RegimeTransitionType.VOLATILITY_SPIKE,
                from_regime=self._current_regime,
                to_regime="bear" if metrics.liquidity_change_1h_pct < -10 else "chop",
                confidence=min(1.0, volatility_z / 4.0),
                timestamp_ms=now_ms,
                trigger_values={"volatility_z": volatility_z},
            )

        # Check for congestion surge
        if metrics.congestion_level >= self.config.congestion_threshold:
            return RegimeTransition(
                transition_type=RegimeTransitionType.CONGESTION_SURGE,
                from_regime=self._current_regime,
                to_regime="mev_heavy",
                confidence=metrics.congestion_level,
                timestamp_ms=now_ms,
                trigger_values={"congestion": metrics.congestion_level},
            )

        # Check for MEV surge
        if metrics.sandwich_rate >= self.config.mev_rate_threshold:
            return RegimeTransition(
                transition_type=RegimeTransitionType.MEV_SURGE,
                from_regime=self._current_regime,
                to_regime="mev_heavy",
                confidence=min(1.0, metrics.sandwich_rate / 0.3),
                timestamp_ms=now_ms,
                trigger_values={"sandwich_rate": metrics.sandwich_rate},
            )

        # Check for liquidity drain
        if metrics.liquidity_change_1h_pct <= -self.config.liquidity_drain_pct * 100:
            return RegimeTransition(
                transition_type=RegimeTransitionType.LIQUIDITY_DRAIN,
                from_regime=self._current_regime,
                to_regime="bear",
                confidence=min(1.0, abs(metrics.liquidity_change_1h_pct) / 50),
                timestamp_ms=now_ms,
                trigger_values={"liquidity_change": metrics.liquidity_change_1h_pct},
            )

        # No transition detected
        return RegimeTransition(
            transition_type=RegimeTransitionType.NONE,
            from_regime=self._current_regime,
            to_regime=self._current_regime,
            confidence=0.0,
            timestamp_ms=now_ms,
        )

    def _calculate_z_score(self, value: float, history: List[float]) -> float:
        """Calculate z-score of value relative to history."""
        if len(history) < 10:
            return 0.0

        mean = np.mean(history)
        std = np.std(history)

        if std < 1e-10:
            return 0.0

        return (value - mean) / std

    def _fast_calibrate(
        self,
        metrics: MarketMetrics,
        transition: RegimeTransition,
    ) -> CalibratedParameters:
        """Perform fast EMA-based calibration.

        Uses exponential moving averages for immediate response
        without expensive optimization.
        """
        now_ms = int(datetime.now().timestamp() * 1000)

        # Get regime-specific base adjustments
        base_params = self._get_regime_base_params(transition.to_regime)

        # Apply EMA updates based on transition type
        alpha = self.config.ema_alpha_fast if transition.confidence > 0.7 else self.config.ema_alpha_slow

        # Update slippage EMA
        target_slippage = base_params.slippage_multiplier
        if transition.transition_type == RegimeTransitionType.VOLATILITY_SPIKE:
            target_slippage *= 1.0 + transition.trigger_values.get("volatility_z", 0) * 0.1
        self._ema_slippage = (1 - alpha) * self._ema_slippage + alpha * target_slippage
        self._ema_slippage = np.clip(
            self._ema_slippage,
            self.config.slippage_multiplier_bounds[0],
            self.config.slippage_multiplier_bounds[1],
        )

        # Update inclusion EMA
        target_inclusion = base_params.inclusion_multiplier
        if transition.transition_type in [RegimeTransitionType.CONGESTION_SURGE, RegimeTransitionType.MEV_SURGE]:
            target_inclusion *= 0.8  # Reduce inclusion in congested/MEV conditions
        self._ema_inclusion = (1 - alpha) * self._ema_inclusion + alpha * target_inclusion
        self._ema_inclusion = np.clip(
            self._ema_inclusion,
            self.config.inclusion_multiplier_bounds[0],
            self.config.inclusion_multiplier_bounds[1],
        )

        # Update MEV EMA
        target_mev = base_params.mev_exposure_multiplier
        if transition.transition_type == RegimeTransitionType.MEV_SURGE:
            target_mev *= 1.0 + transition.trigger_values.get("sandwich_rate", 0) * 3.0
        self._ema_mev = (1 - alpha) * self._ema_mev + alpha * target_mev
        self._ema_mev = np.clip(
            self._ema_mev,
            self.config.mev_multiplier_bounds[0],
            self.config.mev_multiplier_bounds[1],
        )

        # Update latency EMA
        target_latency = base_params.latency_multiplier
        if transition.transition_type == RegimeTransitionType.CONGESTION_SURGE:
            target_latency *= 1.0 + metrics.congestion_level * 0.5
        self._ema_latency = (1 - alpha) * self._ema_latency + alpha * target_latency
        self._ema_latency = np.clip(
            self._ema_latency,
            self.config.latency_multiplier_bounds[0],
            self.config.latency_multiplier_bounds[1],
        )

        # Update current regime
        self._current_regime = transition.to_regime

        return CalibratedParameters(
            regime=transition.to_regime,
            regime_confidence=transition.confidence,
            slippage_multiplier=float(self._ema_slippage),
            inclusion_multiplier=float(self._ema_inclusion),
            mev_exposure_multiplier=float(self._ema_mev),
            latency_multiplier=float(self._ema_latency),
            survivorship_discount_adjustment=self._get_survivorship_adjustment(transition.to_regime),
            confidence_multiplier=base_params.confidence_multiplier,
            calibrated_at_ms=now_ms,
            transition_type=transition.transition_type,
            samples_used=len(self._volume_history),
        )

    def _get_regime_base_params(self, regime: str) -> CalibratedParameters:
        """Get base parameters for a regime."""
        regime_lower = regime.lower()

        if regime_lower == "bull":
            return CalibratedParameters(
                regime=regime,
                regime_confidence=0.8,
                slippage_multiplier=1.0,
                inclusion_multiplier=1.0,
                mev_exposure_multiplier=1.0,
                latency_multiplier=1.0,
                confidence_multiplier=1.2,
            )
        elif regime_lower == "bear":
            return CalibratedParameters(
                regime=regime,
                regime_confidence=0.8,
                slippage_multiplier=1.2,
                inclusion_multiplier=0.95,
                mev_exposure_multiplier=1.0,
                latency_multiplier=1.1,
                confidence_multiplier=0.8,
            )
        elif regime_lower == "mev_heavy":
            return CalibratedParameters(
                regime=regime,
                regime_confidence=0.8,
                slippage_multiplier=1.5,
                inclusion_multiplier=0.9,
                mev_exposure_multiplier=2.0,
                latency_multiplier=1.3,
                confidence_multiplier=0.9,
            )
        else:  # chop
            return CalibratedParameters(
                regime=regime,
                regime_confidence=0.5,
                slippage_multiplier=1.0,
                inclusion_multiplier=1.0,
                mev_exposure_multiplier=1.0,
                latency_multiplier=1.0,
                confidence_multiplier=1.0,
            )

    def _get_survivorship_adjustment(self, regime: str) -> float:
        """Get survivorship discount adjustment for regime."""
        regime_lower = regime.lower()

        if regime_lower == "bear":
            return 0.05  # +5% discount in bear
        elif regime_lower == "mev_heavy":
            return 0.07  # +7% discount in MEV heavy
        return 0.0

    def _start_transition(self, new_params: CalibratedParameters) -> None:
        """Start a smooth transition to new parameters."""
        self._transition_start_ms = int(datetime.now().timestamp() * 1000)
        self._transition_from_params = self._current_params
        self._transition_to_params = new_params
        self._current_params = new_params

    def observe(self, metrics: MarketMetrics) -> CalibratedParameters:
        """Observe market metrics and return current calibrated parameters.

        This is a synchronous entry point for the regime calibrator.
        It updates history, checks for transitions, and returns the
        current (potentially blended) parameters.
        """
        # Update history
        self._update_history(metrics)

        # Detect transition
        transition = self._detect_transition(metrics)

        if transition.transition_type != RegimeTransitionType.NONE:
            # Check minimum time between calibrations
            now_ms = int(datetime.now().timestamp() * 1000)
            if now_ms - self._last_calibration_ms >= self.config.min_time_between_calibrations_seconds * 1000:
                # Perform fast calibration
                new_params = self._fast_calibrate(metrics, transition)

                # Start transition blending
                self._start_transition(new_params)

                # Store active transition
                self._active_transition = transition

                # Notify callback
                if self.on_calibration:
                    try:
                        self.on_calibration(new_params)
                    except Exception as e:
                        logger.error(f"Calibration callback error: {e}")

                self._last_calibration_ms = now_ms
                self._calibration_count += 1

                logger.info(
                    f"Regime calibration #{self._calibration_count}: "
                    f"{transition.from_regime} -> {transition.to_regime} "
                    f"({transition.transition_type.value})"
                )

        return self.get_current_params()

    def get_active_transition(self) -> Optional[RegimeTransition]:
        """Get the currently active regime transition, if any."""
        if self._transition_start_ms is None:
            return None

        now_ms = int(datetime.now().timestamp() * 1000)
        elapsed_ms = now_ms - self._transition_start_ms
        blend_duration_ms = self.config.blend_duration_seconds * 1000

        if elapsed_ms >= blend_duration_ms:
            # Transition complete
            return None

        return getattr(self, '_active_transition', None)

    def get_current_params(self) -> CalibratedParameters:
        """Get current calibrated parameters with transition blending."""
        if self._transition_start_ms is None:
            return self._current_params

        now_ms = int(datetime.now().timestamp() * 1000)
        elapsed_ms = now_ms - self._transition_start_ms
        blend_duration_ms = self.config.blend_duration_seconds * 1000

        if elapsed_ms >= blend_duration_ms:
            # Transition complete
            self._transition_start_ms = None
            self._transition_from_params = None
            self._transition_to_params = None
            return self._current_params

        # Calculate blend factor (0 = old, 1 = new)
        blend = elapsed_ms / blend_duration_ms
        blend = max(self.config.min_blend_factor, min(1.0, blend))

        # Blend parameters
        from_p = self._transition_from_params
        to_p = self._transition_to_params

        if from_p is None or to_p is None:
            return self._current_params

        return CalibratedParameters(
            regime=to_p.regime,
            regime_confidence=to_p.regime_confidence,
            slippage_multiplier=(1 - blend) * from_p.slippage_multiplier + blend * to_p.slippage_multiplier,
            inclusion_multiplier=(1 - blend) * from_p.inclusion_multiplier + blend * to_p.inclusion_multiplier,
            mev_exposure_multiplier=(1 - blend) * from_p.mev_exposure_multiplier + blend * to_p.mev_exposure_multiplier,
            latency_multiplier=(1 - blend) * from_p.latency_multiplier + blend * to_p.latency_multiplier,
            survivorship_discount_adjustment=to_p.survivorship_discount_adjustment,
            confidence_multiplier=(1 - blend) * from_p.confidence_multiplier + blend * to_p.confidence_multiplier,
            calibrated_at_ms=to_p.calibrated_at_ms,
            transition_type=to_p.transition_type,
            samples_used=to_p.samples_used,
        )

    def force_calibrate(self, regime: str, confidence: float = 0.8) -> CalibratedParameters:
        """Force calibration to a specific regime.

        Useful for testing or manual intervention.
        """
        now_ms = int(datetime.now().timestamp() * 1000)

        transition = RegimeTransition(
            transition_type=RegimeTransitionType.GRADUAL_TRANSITION,
            from_regime=self._current_regime,
            to_regime=regime,
            confidence=confidence,
            timestamp_ms=now_ms,
        )

        metrics = MarketMetrics(timestamp_ms=now_ms)
        new_params = self._fast_calibrate(metrics, transition)
        self._start_transition(new_params)

        return new_params

    def get_stats(self) -> Dict[str, Any]:
        """Get calibrator statistics."""
        return {
            "current_regime": self._current_regime,
            "calibration_count": self._calibration_count,
            "last_calibration_ms": self._last_calibration_ms,
            "history_size": len(self._volume_history),
            "in_transition": self._transition_start_ms is not None,
            "current_params": self._current_params.to_dict(),
        }
