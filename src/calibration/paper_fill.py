"""
Paper fill parameter calibration.

Tunes the paper fill engine parameters based on Live Scout telemetry.
Uses statistical fitting to model realistic fill behavior.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from scipy import stats
from datetime import datetime

if TYPE_CHECKING:
    from ..storage import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class FillObservation:
    """Observation from a real or paper fill."""
    timestamp_ms: int
    quoted_price: float
    fill_price: Optional[float]
    quoted_slippage_bps: int
    realized_slippage_bps: Optional[int]
    latency_ms: Optional[int]
    included: bool
    mode: str  # LIVE, PAPER, SHADOW
    pool_address: Optional[str] = None
    liquidity_usd: Optional[float] = None
    mev_detected: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FillObservation":
        """Create from dictionary."""
        return cls(
            timestamp_ms=data.get("timestamp_ms", 0),
            quoted_price=data.get("quoted_price", 0),
            fill_price=data.get("fill_price"),
            quoted_slippage_bps=data.get("quoted_slippage_bps", 0),
            realized_slippage_bps=data.get("realized_slippage_bps"),
            latency_ms=data.get("latency_ms"),
            included=bool(data.get("included", True)),
            mode=data.get("mode", "LIVE"),
            pool_address=data.get("pool_address"),
            liquidity_usd=data.get("liquidity_usd"),
            mev_detected=bool(data.get("mev_detected", False)),
        )


@dataclass
class LatencyModel:
    """Statistical model for transaction latency."""
    mean_ms: float = 150.0
    std_ms: float = 50.0
    min_ms: float = 50.0
    max_ms: float = 2000.0
    p95_ms: float = 350.0
    p99_ms: float = 500.0

    def sample(self) -> float:
        """Sample a latency value from the model."""
        # Use lognormal for right-skewed distribution
        mu = np.log(self.mean_ms) - 0.5 * (self.std_ms / self.mean_ms) ** 2
        sigma = np.sqrt(np.log(1 + (self.std_ms / self.mean_ms) ** 2))
        sample = np.random.lognormal(mu, sigma)
        return float(np.clip(sample, self.min_ms, self.max_ms))

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
        }


@dataclass
class SlippageModel:
    """Statistical model for slippage behavior."""
    base_bps: float = 50.0
    volatility_coef: float = 0.5
    liquidity_coef: float = 0.3  # Inverse relationship with liquidity
    mev_penalty_mean_bps: float = 100.0
    mev_probability: float = 0.3  # Probability of MEV extraction

    def estimate_slippage(
        self,
        quoted_bps: int,
        liquidity_usd: float = 100000,
        volatility: float = 0.5,
    ) -> float:
        """Estimate expected slippage given market conditions."""
        # Base slippage
        slippage = self.base_bps

        # Add volatility component
        slippage += self.volatility_coef * volatility * 100

        # Add liquidity component (higher slippage for lower liquidity)
        if liquidity_usd > 0:
            liquidity_factor = 100000 / liquidity_usd
            slippage += self.liquidity_coef * min(liquidity_factor, 10) * 100

        # Add MEV penalty probabilistically
        if np.random.random() < self.mev_probability:
            mev = np.random.exponential(self.mev_penalty_mean_bps)
            slippage += min(mev, 500)  # Cap MEV at 5%

        return slippage

    def to_dict(self) -> Dict[str, float]:
        return {
            "base_bps": self.base_bps,
            "volatility_coef": self.volatility_coef,
            "liquidity_coef": self.liquidity_coef,
            "mev_penalty_mean_bps": self.mev_penalty_mean_bps,
            "mev_probability": self.mev_probability,
        }


@dataclass
class InclusionModel:
    """Statistical model for transaction inclusion probability."""
    base_probability: float = 0.85
    congestion_decay: float = 0.1
    priority_fee_boost: float = 0.05  # Per priority fee level
    max_probability: float = 0.98

    def estimate_inclusion(
        self,
        priority_fee_level: int = 1,
        network_congestion: float = 0.5,
    ) -> float:
        """Estimate inclusion probability."""
        prob = self.base_probability

        # Reduce for congestion
        prob -= self.congestion_decay * network_congestion

        # Boost for priority fee
        prob += self.priority_fee_boost * priority_fee_level

        return float(np.clip(prob, 0.1, self.max_probability))

    def to_dict(self) -> Dict[str, float]:
        return {
            "base_probability": self.base_probability,
            "congestion_decay": self.congestion_decay,
            "priority_fee_boost": self.priority_fee_boost,
            "max_probability": self.max_probability,
        }


class PaperFillCalibrator:
    """Calibrates paper fill model parameters from Live Scout data.

    Uses statistical fitting on real trade observations to create
    realistic simulation parameters for paper trading.
    """

    def __init__(self, db: Optional["DatabaseManager"] = None):
        self.db = db

        # Model components
        self.latency_model = LatencyModel()
        self.slippage_model = SlippageModel()
        self.inclusion_model = InclusionModel()

        # Calibration metadata
        self.last_calibration: Optional[datetime] = None
        self.observations_used: int = 0
        self.calibration_history: List[Dict[str, Any]] = []

    async def calibrate(
        self,
        observations: Optional[List[Dict[str, Any]]] = None,
        min_observations: int = 20,
    ):
        """Calibrate parameters from observations.

        Args:
            observations: List of observation dictionaries.
            min_observations: Minimum observations required for calibration.
        """
        if observations is None and self.db is not None:
            observations = await self.db.get_fill_observations(hours=6)

        if observations is None or len(observations) < min_observations:
            logger.info(
                f"Insufficient observations for calibration: "
                f"{len(observations) if observations else 0} < {min_observations}"
            )
            return

        # Convert to FillObservation objects
        obs_list = [FillObservation.from_dict(o) for o in observations]

        # Filter to LIVE mode only for calibration
        live_obs = [o for o in obs_list if o.mode == "LIVE"]
        if len(live_obs) < min_observations:
            logger.info(f"Insufficient LIVE observations: {len(live_obs)}")
            return

        logger.info(f"Calibrating on {len(live_obs)} LIVE observations")

        # Calibrate each model component
        self._calibrate_latency(live_obs)
        self._calibrate_slippage(live_obs)
        self._calibrate_inclusion(live_obs)

        # Update metadata
        self.last_calibration = datetime.now()
        self.observations_used = len(live_obs)

        # Store calibration history
        self.calibration_history.append({
            "timestamp": self.last_calibration.isoformat(),
            "observations": self.observations_used,
            "latency_mean": self.latency_model.mean_ms,
            "slippage_base": self.slippage_model.base_bps,
            "inclusion_base": self.inclusion_model.base_probability,
        })

        # Keep only last 100 calibrations
        if len(self.calibration_history) > 100:
            self.calibration_history = self.calibration_history[-100:]

        logger.info(
            f"Calibration complete: latency={self.latency_model.mean_ms:.0f}ms, "
            f"slippage_base={self.slippage_model.base_bps:.0f}bps, "
            f"inclusion={self.inclusion_model.base_probability:.2%}"
        )

    def _calibrate_latency(self, observations: List[FillObservation]):
        """Fit latency distribution using robust statistics."""
        latencies = [o.latency_ms for o in observations if o.latency_ms is not None]
        if len(latencies) < 10:
            return

        latencies = np.array(latencies)

        # Use trimmed mean for robustness
        trim_pct = 0.1
        sorted_lat = np.sort(latencies)
        trim_n = int(len(sorted_lat) * trim_pct)
        trimmed = sorted_lat[trim_n:-trim_n] if trim_n > 0 else sorted_lat

        self.latency_model.mean_ms = float(np.mean(trimmed))
        self.latency_model.std_ms = float(np.std(trimmed))
        self.latency_model.min_ms = float(np.percentile(latencies, 1))
        self.latency_model.max_ms = float(np.percentile(latencies, 99))
        self.latency_model.p95_ms = float(np.percentile(latencies, 95))
        self.latency_model.p99_ms = float(np.percentile(latencies, 99))

    def _calibrate_slippage(self, observations: List[FillObservation]):
        """Fit slippage model using regression."""
        # Filter to included trades only
        included_obs = [o for o in observations if o.included and o.realized_slippage_bps is not None]
        if len(included_obs) < 10:
            return

        # Calculate adverse slippage (realized - quoted)
        adverse_slippages = [
            o.realized_slippage_bps - o.quoted_slippage_bps
            for o in included_obs
        ]

        # Base slippage is median adverse slippage
        self.slippage_model.base_bps = float(np.median(adverse_slippages))

        # MEV detection: positive adverse slippage above base
        mev_candidates = [a - self.slippage_model.base_bps for a in adverse_slippages if a > self.slippage_model.base_bps + 20]
        if len(mev_candidates) >= 5:
            self.slippage_model.mev_penalty_mean_bps = float(np.mean(mev_candidates))
            self.slippage_model.mev_probability = len(mev_candidates) / len(included_obs)

        # Liquidity coefficient estimation
        obs_with_liquidity = [(o.liquidity_usd, o.realized_slippage_bps - o.quoted_slippage_bps)
                              for o in included_obs if o.liquidity_usd and o.liquidity_usd > 0]
        if len(obs_with_liquidity) >= 10:
            liquidity_vals = np.array([o[0] for o in obs_with_liquidity])
            slippage_vals = np.array([o[1] for o in obs_with_liquidity])

            # Inverse relationship: higher liquidity = lower slippage
            inv_liquidity = 1.0 / liquidity_vals
            if np.std(inv_liquidity) > 0:
                correlation = np.corrcoef(inv_liquidity, slippage_vals)[0, 1]
                if not np.isnan(correlation) and correlation > 0:
                    self.slippage_model.liquidity_coef = float(np.clip(correlation, 0, 1))

    def _calibrate_inclusion(self, observations: List[FillObservation]):
        """Fit inclusion probability model."""
        if not observations:
            return

        included_count = sum(1 for o in observations if o.included)
        total = len(observations)

        self.inclusion_model.base_probability = float(included_count / total)

        # Cap at reasonable bounds
        self.inclusion_model.base_probability = np.clip(
            self.inclusion_model.base_probability, 0.5, 0.98
        )

    def to_params(self) -> Dict[str, Any]:
        """Export calibrated parameters for Hot Path."""
        return {
            "latency": self.latency_model.to_dict(),
            "slippage": self.slippage_model.to_dict(),
            "inclusion": self.inclusion_model.to_dict(),
            "last_calibration": self.last_calibration.isoformat() if self.last_calibration else None,
            "observations_used": self.observations_used,
        }

    def to_state(self) -> Dict[str, Any]:
        """Export full state for persistence."""
        return {
            "latency": self.latency_model.to_dict(),
            "slippage": self.slippage_model.to_dict(),
            "inclusion": self.inclusion_model.to_dict(),
            "last_calibration": self.last_calibration.isoformat() if self.last_calibration else None,
            "observations_used": self.observations_used,
            "calibration_history": self.calibration_history[-10:],  # Last 10
        }

    def load_state(self, state: Dict[str, Any]):
        """Load state from persistence."""
        if "latency" in state:
            lat = state["latency"]
            self.latency_model = LatencyModel(
                mean_ms=lat.get("mean_ms", 150.0),
                std_ms=lat.get("std_ms", 50.0),
                min_ms=lat.get("min_ms", 50.0),
                max_ms=lat.get("max_ms", 2000.0),
                p95_ms=lat.get("p95_ms", 350.0),
                p99_ms=lat.get("p99_ms", 500.0),
            )

        if "slippage" in state:
            slip = state["slippage"]
            self.slippage_model = SlippageModel(
                base_bps=slip.get("base_bps", 50.0),
                volatility_coef=slip.get("volatility_coef", 0.5),
                liquidity_coef=slip.get("liquidity_coef", 0.3),
                mev_penalty_mean_bps=slip.get("mev_penalty_mean_bps", 100.0),
                mev_probability=slip.get("mev_probability", 0.3),
            )

        if "inclusion" in state:
            inc = state["inclusion"]
            self.inclusion_model = InclusionModel(
                base_probability=inc.get("base_probability", 0.85),
                congestion_decay=inc.get("congestion_decay", 0.1),
                priority_fee_boost=inc.get("priority_fee_boost", 0.05),
                max_probability=inc.get("max_probability", 0.98),
            )

        if "last_calibration" in state and state["last_calibration"]:
            self.last_calibration = datetime.fromisoformat(state["last_calibration"])

        self.observations_used = state.get("observations_used", 0)
        self.calibration_history = state.get("calibration_history", [])

        logger.info(f"Loaded calibration state from {self.last_calibration}")
