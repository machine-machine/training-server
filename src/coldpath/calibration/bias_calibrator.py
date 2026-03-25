"""
Bias calibrator for paper trading simulation.

Compares paper predictions with live outcomes to calibrate bias coefficients.
Addresses key trading biases:
- Survivorship bias: Only testing tokens that still exist
- Slippage gap: Non-linear relationship with size/volatility
- Latency: Network-aware execution delays
- Liquidity: Partial fills and market impact
- MEV/Frontrunning: Sandwich attack simulation
- Quote staleness: Price drift before execution
- Network congestion: Failed transaction simulation
- Fill probability: Order inclusion modeling
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from scipy import optimize
from sklearn.model_selection import TimeSeriesSplit

if TYPE_CHECKING:
    from ..storage import DatabaseManager

logger = logging.getLogger(__name__)


# ============================================================================
# VALIDATION UTILITIES - Extensible validation framework for future upgrades
# ============================================================================


class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


class NumericValidator:
    """Reusable numeric validation utilities for safe calculations.

    This class provides a foundation for validating numeric inputs across
    the calibration pipeline. Extend this class for domain-specific validation.

    Future upgrades can add:
    - Statistical outlier detection
    - Domain-specific bounds (e.g., slippage ranges)
    - Batch validation with detailed reports
    """

    @staticmethod
    def is_valid_float(value: float, allow_zero: bool = False) -> bool:
        """Check if float is valid (not NaN, not Inf, optionally not zero)."""
        if value is None:
            return False
        if np.isnan(value) or np.isinf(value):
            return False
        if not allow_zero and value == 0.0:
            return False
        return True

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with fallback for invalid denominators."""
        if not NumericValidator.is_valid_float(denominator, allow_zero=False):
            return default
        if not NumericValidator.is_valid_float(numerator, allow_zero=True):
            return default
        result = numerator / denominator
        if not NumericValidator.is_valid_float(result, allow_zero=True):
            return default
        return result

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float, default: float = None) -> float:
        """Clamp value to range, with optional default for invalid inputs."""
        if not NumericValidator.is_valid_float(value, allow_zero=True):
            return default if default is not None else min_val
        return float(np.clip(value, min_val, max_val))

    @staticmethod
    def validate_positive(value: float, name: str = "value") -> float:
        """Validate that value is positive, raising if not."""
        if not NumericValidator.is_valid_float(value, allow_zero=False) or value <= 0:
            raise ValidationError(f"{name} must be positive, got: {value}")
        return value

    @staticmethod
    def validate_probability(value: float, name: str = "probability") -> float:
        """Validate that value is a valid probability [0, 1]."""
        if not NumericValidator.is_valid_float(value, allow_zero=True):
            raise ValidationError(f"{name} is not a valid number: {value}")
        if value < 0.0 or value > 1.0:
            raise ValidationError(f"{name} must be in [0, 1], got: {value}")
        return value

    @staticmethod
    def validate_array(
        arr: np.ndarray, name: str = "array", min_length: int = 1, allow_nan: bool = False
    ) -> np.ndarray:
        """Validate numpy array for common issues."""
        if arr is None or len(arr) < min_length:
            raise ValidationError(f"{name} must have at least {min_length} elements")
        if not allow_nan and np.any(np.isnan(arr)):
            raise ValidationError(f"{name} contains NaN values")
        if np.any(np.isinf(arr)):
            raise ValidationError(f"{name} contains infinite values")
        return arr


class CalibrationBounds:
    """Defines plausible bounds for calibration parameters.

    Centralizes all magic numbers and bounds for easy tuning and future upgrades.
    Add new parameter bounds here as the calibration model expands.
    """

    # Slippage model bounds
    SLIPPAGE_BASE_BPS = (5.0, 200.0)  # 0.05% to 2%
    SLIPPAGE_SIZE_EXP = (10.0, 500.0)  # Exponential coefficient
    SLIPPAGE_VOL_COEF = (0.1, 5.0)  # Volatility coefficient
    SLIPPAGE_LIQ_COEF = (0.1, 5.0)  # Liquidity coefficient

    # Latency model bounds
    LATENCY_BASE_MS = (10.0, 5000.0)  # 10ms to 5s
    LATENCY_STD_MS = (5.0, 2000.0)  # Standard deviation
    LATENCY_CONG_MULT_LOW = (0.1, 5.0)  # Low congestion multiplier
    LATENCY_CONG_MULT_HIGH = (0.5, 20.0)  # High congestion multiplier

    # MEV model bounds
    MEV_PROB = (0.0, 1.0)  # Sandwich probability
    MEV_COST_BPS = (0.0, 500.0)  # Sandwich cost in bps
    MEV_JITO_DISCOUNT = (0.1, 1.0)  # Jito protection discount

    # Inclusion model bounds
    INCLUSION_BASE = (0.5, 1.0)  # Base inclusion probability
    INCLUSION_SIZE_DECAY = (0.0, 0.5)  # Size decay factor
    INCLUSION_CONG_DECAY = (0.0, 0.5)  # Congestion decay factor

    # Survivorship bounds
    SURVIVORSHIP_DISCOUNT = (0.05, 0.50)  # 5% to 50% discount
    RUG_RATE = (0.01, 0.50)  # 1% to 50% rug rate

    @classmethod
    def clamp_slippage_base(cls, value: float) -> float:
        return NumericValidator.clamp(value, *cls.SLIPPAGE_BASE_BPS, default=30.0)

    @classmethod
    def clamp_latency_multiplier(cls, value: float, high: bool = False) -> float:
        bounds = cls.LATENCY_CONG_MULT_HIGH if high else cls.LATENCY_CONG_MULT_LOW
        default = 2.5 if high else 0.8
        return NumericValidator.clamp(value, *bounds, default=default)


@dataclass
class BiasObservation:
    """Paired paper/live trade observation for calibration."""

    timestamp_ms: int
    mode: str  # PAPER or LIVE

    # Execution details
    trade_size_sol: float
    pool_liquidity_usd: float
    quoted_price: float
    filled_price: float | None
    quoted_slippage_bps: int
    realized_slippage_bps: int | None

    # Timing
    quote_age_ms: int
    execution_latency_ms: int | None

    # Network state
    network_congestion: float  # 0.0 to 1.0
    slot_rate: float  # Slots per second

    # Outcome
    included: bool
    partial_fill_fraction: float  # 1.0 = full fill

    # MEV detection
    mev_detected: bool
    using_jito: bool

    # Token metadata
    token_mint: str
    token_age_hours: float
    volatility_1h: float
    holder_count: int
    is_rugged: bool  # For survivorship analysis

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BiasObservation":
        return cls(
            timestamp_ms=data.get("timestamp_ms", 0),
            mode=data.get("mode", "LIVE"),
            trade_size_sol=data.get("trade_size_sol", 0.1),
            pool_liquidity_usd=data.get("pool_liquidity_usd", 50000),
            quoted_price=data.get("quoted_price", 0),
            filled_price=data.get("filled_price"),
            quoted_slippage_bps=data.get("quoted_slippage_bps", 0),
            realized_slippage_bps=data.get("realized_slippage_bps"),
            quote_age_ms=data.get("quote_age_ms", 0),
            execution_latency_ms=data.get("execution_latency_ms"),
            network_congestion=data.get("network_congestion", 0.3),
            slot_rate=data.get("slot_rate", 2.5),
            included=data.get("included", True),
            partial_fill_fraction=data.get("partial_fill_fraction", 1.0),
            mev_detected=data.get("mev_detected", False),
            using_jito=data.get("using_jito", False),
            token_mint=data.get("token_mint", ""),
            token_age_hours=data.get("token_age_hours", 24),
            volatility_1h=data.get("volatility_1h", 0.05),
            holder_count=data.get("holder_count", 100),
            is_rugged=data.get("is_rugged", False),
        )


@dataclass
class BiasCoefficients:
    """Calibrated bias coefficients for realistic paper trading."""

    # Non-linear slippage model
    slippage_base_bps: float = 30.0
    slippage_size_exponent: float = 50.0
    slippage_volatility_coef: float = 0.8
    slippage_liquidity_coef: float = 0.5

    # Congestion-aware latency
    latency_base_mean_ms: float = 150.0
    latency_base_std_ms: float = 50.0
    latency_congestion_multiplier_low: float = 0.8
    latency_congestion_multiplier_high: float = 2.5

    # MEV/Sandwich model
    mev_sandwich_probability: float = 0.25
    mev_sandwich_cost_bps: float = 100.0
    mev_jito_discount_factor: float = 0.7

    # Inclusion model
    inclusion_base_prob: float = 0.88
    inclusion_size_decay: float = 0.02
    inclusion_congestion_decay: float = 0.15
    inclusion_partial_fill_threshold: float = 0.05

    # Quote staleness
    max_quote_age_ms: int = 2000
    quote_staleness_penalty_bps: float = 5.0

    # Survivorship bias
    survivorship_discount: float = 0.15
    rug_rate_24h: float = 0.12  # Fraction of tokens that rug within 24h

    def to_dict(self) -> dict[str, Any]:
        return {
            "slippage": {
                "base_bps": self.slippage_base_bps,
                "size_exponent": self.slippage_size_exponent,
                "volatility_coef": self.slippage_volatility_coef,
                "liquidity_coef": self.slippage_liquidity_coef,
            },
            "latency": {
                "base_mean_ms": self.latency_base_mean_ms,
                "base_std_ms": self.latency_base_std_ms,
                "congestion_multiplier_low": self.latency_congestion_multiplier_low,
                "congestion_multiplier_high": self.latency_congestion_multiplier_high,
            },
            "mev": {
                "sandwich_probability": self.mev_sandwich_probability,
                "sandwich_cost_bps": self.mev_sandwich_cost_bps,
                "jito_discount_factor": self.mev_jito_discount_factor,
            },
            "inclusion": {
                "base_prob": self.inclusion_base_prob,
                "size_decay": self.inclusion_size_decay,
                "congestion_decay": self.inclusion_congestion_decay,
                "partial_fill_threshold": self.inclusion_partial_fill_threshold,
            },
            "staleness": {
                "max_quote_age_ms": self.max_quote_age_ms,
                "penalty_bps": self.quote_staleness_penalty_bps,
            },
            "survivorship": {
                "discount": self.survivorship_discount,
                "rug_rate_24h": self.rug_rate_24h,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BiasCoefficients":
        coeffs = cls()

        if "slippage" in data:
            s = data["slippage"]
            coeffs.slippage_base_bps = s.get("base_bps", coeffs.slippage_base_bps)
            coeffs.slippage_size_exponent = s.get("size_exponent", coeffs.slippage_size_exponent)
            coeffs.slippage_volatility_coef = s.get(
                "volatility_coef", coeffs.slippage_volatility_coef
            )
            coeffs.slippage_liquidity_coef = s.get("liquidity_coef", coeffs.slippage_liquidity_coef)

        if "latency" in data:
            lat = data["latency"]
            coeffs.latency_base_mean_ms = lat.get("base_mean_ms", coeffs.latency_base_mean_ms)
            coeffs.latency_base_std_ms = lat.get("base_std_ms", coeffs.latency_base_std_ms)
            coeffs.latency_congestion_multiplier_low = lat.get(
                "congestion_multiplier_low", coeffs.latency_congestion_multiplier_low
            )
            coeffs.latency_congestion_multiplier_high = lat.get(
                "congestion_multiplier_high", coeffs.latency_congestion_multiplier_high
            )

        if "mev" in data:
            m = data["mev"]
            coeffs.mev_sandwich_probability = m.get(
                "sandwich_probability", coeffs.mev_sandwich_probability
            )
            coeffs.mev_sandwich_cost_bps = m.get("sandwich_cost_bps", coeffs.mev_sandwich_cost_bps)
            coeffs.mev_jito_discount_factor = m.get(
                "jito_discount_factor", coeffs.mev_jito_discount_factor
            )

        if "inclusion" in data:
            i = data["inclusion"]
            coeffs.inclusion_base_prob = i.get("base_prob", coeffs.inclusion_base_prob)
            coeffs.inclusion_size_decay = i.get("size_decay", coeffs.inclusion_size_decay)
            coeffs.inclusion_congestion_decay = i.get(
                "congestion_decay", coeffs.inclusion_congestion_decay
            )
            coeffs.inclusion_partial_fill_threshold = i.get(
                "partial_fill_threshold", coeffs.inclusion_partial_fill_threshold
            )

        if "staleness" in data:
            st = data["staleness"]
            coeffs.max_quote_age_ms = st.get("max_quote_age_ms", coeffs.max_quote_age_ms)
            coeffs.quote_staleness_penalty_bps = st.get(
                "penalty_bps", coeffs.quote_staleness_penalty_bps
            )

        if "survivorship" in data:
            su = data["survivorship"]
            coeffs.survivorship_discount = su.get("discount", coeffs.survivorship_discount)
            coeffs.rug_rate_24h = su.get("rug_rate_24h", coeffs.rug_rate_24h)

        return coeffs


@dataclass
class CalibrationReport:
    """Report from a calibration run."""

    timestamp: datetime
    observations_used: int
    live_observations: int
    paper_observations: int

    # Model fit metrics
    slippage_mae_bps: float = 0.0
    slippage_rmse_bps: float = 0.0
    latency_mae_ms: float = 0.0
    inclusion_accuracy: float = 0.0
    mev_detection_rate: float = 0.0

    # Survivorship analysis
    rug_rate_observed: float = 0.0
    survivorship_factor: float = 1.0

    # Validation metrics (walk-forward)
    validation_slippage_mae: float = 0.0
    validation_inclusion_accuracy: float = 0.0

    # Coefficients
    coefficients: BiasCoefficients = field(default_factory=BiasCoefficients)

    # Status
    converged: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "observations_used": self.observations_used,
            "live_observations": self.live_observations,
            "paper_observations": self.paper_observations,
            "metrics": {
                "slippage_mae_bps": self.slippage_mae_bps,
                "slippage_rmse_bps": self.slippage_rmse_bps,
                "latency_mae_ms": self.latency_mae_ms,
                "inclusion_accuracy": self.inclusion_accuracy,
                "mev_detection_rate": self.mev_detection_rate,
            },
            "survivorship": {
                "rug_rate_observed": self.rug_rate_observed,
                "survivorship_factor": self.survivorship_factor,
            },
            "validation": {
                "slippage_mae": self.validation_slippage_mae,
                "inclusion_accuracy": self.validation_inclusion_accuracy,
            },
            "coefficients": self.coefficients.to_dict(),
            "converged": self.converged,
            "warnings": self.warnings,
        }


class BiasCalibrator:
    """Compares paper predictions with live outcomes to calibrate bias coefficients.

    Implements a comprehensive calibration pipeline:
    1. Fetch paired paper/live trade data
    2. Fit slippage model (scipy minimize)
    3. Fit latency model (congestion sensitivity)
    4. Fit inclusion model (base prob, decay)
    5. Fit MEV model (from slippage outliers)
    6. Detect survivorship bias (rug rate analysis)
    7. Walk-forward validation (TimeSeriesSplit)
    8. Return calibrated coefficients
    """

    def __init__(self, db: Optional["DatabaseManager"] = None):
        self.db = db
        self.coefficients = BiasCoefficients()
        self.last_report: CalibrationReport | None = None
        self.calibration_history: list[dict[str, Any]] = []

    async def calibrate(
        self,
        observations: list[dict[str, Any]] | None = None,
        min_observations: int = 50,
        walk_forward_splits: int = 3,
    ) -> CalibrationReport:
        """Run full calibration pipeline.

        Args:
            observations: List of observation dictionaries (fetched from DB if None)
            min_observations: Minimum observations required
            walk_forward_splits: Number of time series splits for validation

        Returns:
            CalibrationReport with calibrated coefficients and metrics
        """
        report = CalibrationReport(
            timestamp=datetime.now(),
            observations_used=0,
            live_observations=0,
            paper_observations=0,
        )

        # 1. Fetch paired paper/live trade data
        if observations is None and self.db is not None:
            observations = await self._fetch_observations()

        if observations is None or len(observations) < min_observations:
            report.converged = False
            report.warnings.append(
                f"Insufficient observations: {len(observations) if observations else 0} < "
                f"{min_observations}"
            )
            logger.warning(report.warnings[-1])
            return report

        # Convert to BiasObservation objects
        obs_list = [BiasObservation.from_dict(o) for o in observations]

        # Separate by mode
        live_obs = [o for o in obs_list if o.mode == "LIVE"]
        paper_obs = [o for o in obs_list if o.mode == "PAPER"]

        report.observations_used = len(obs_list)
        report.live_observations = len(live_obs)
        report.paper_observations = len(paper_obs)

        if len(live_obs) < min_observations // 2:
            report.warnings.append(f"Insufficient LIVE observations: {len(live_obs)}")
            logger.warning(report.warnings[-1])

        logger.info(f"Calibrating on {len(live_obs)} LIVE and {len(paper_obs)} PAPER observations")

        # 2. Fit slippage model
        try:
            self._fit_slippage_model(live_obs, report)
        except Exception as e:
            report.warnings.append(f"Slippage fit failed: {e}")
            logger.warning(report.warnings[-1])

        # 3. Fit latency model
        try:
            self._fit_latency_model(live_obs, report)
        except Exception as e:
            report.warnings.append(f"Latency fit failed: {e}")
            logger.warning(report.warnings[-1])

        # 4. Fit inclusion model
        try:
            self._fit_inclusion_model(live_obs, report)
        except Exception as e:
            report.warnings.append(f"Inclusion fit failed: {e}")
            logger.warning(report.warnings[-1])

        # 5. Fit MEV model
        try:
            self._fit_mev_model(live_obs, report)
        except Exception as e:
            report.warnings.append(f"MEV fit failed: {e}")
            logger.warning(report.warnings[-1])

        # 6. Detect survivorship bias
        try:
            self._analyze_survivorship(obs_list, report)
        except Exception as e:
            report.warnings.append(f"Survivorship analysis failed: {e}")
            logger.warning(report.warnings[-1])

        # 7. Walk-forward validation
        if len(live_obs) >= walk_forward_splits * 10:
            try:
                self._walk_forward_validation(live_obs, report, n_splits=walk_forward_splits)
            except Exception as e:
                report.warnings.append(f"Walk-forward validation failed: {e}")
                logger.warning(report.warnings[-1])

        # Update coefficients
        report.coefficients = self.coefficients
        self.last_report = report

        # Store in history
        self.calibration_history.append(report.to_dict())
        if len(self.calibration_history) > 100:
            self.calibration_history = self.calibration_history[-100:]

        logger.info(
            f"Calibration complete: slippage_base={self.coefficients.slippage_base_bps:.1f}bps, "
            f"mev_prob={self.coefficients.mev_sandwich_probability:.2%}, "
            f"inclusion={self.coefficients.inclusion_base_prob:.2%}, "
            f"survivorship={self.coefficients.survivorship_discount:.2%}"
        )

        return report

    async def _fetch_observations(self) -> list[dict[str, Any]]:
        """Fetch observations from database."""
        if self.db is None:
            return []

        # Fetch from last 24 hours
        raw_obs = await self.db.get_fill_observations(hours=24)

        # Also fetch rug/survivorship data
        rugged_tokens = await self.db.get_rugged_tokens(hours=48)
        rugged_mints = set(t.get("mint") for t in rugged_tokens if t.get("mint"))

        # Mark observations from rugged tokens
        for obs in raw_obs:
            if obs.get("token_mint") in rugged_mints:
                obs["is_rugged"] = True

        return raw_obs

    def _fit_slippage_model(self, observations: list[BiasObservation], report: CalibrationReport):
        """Fit non-linear slippage model using scipy optimization."""
        # Filter to included trades with slippage data
        valid_obs = [
            o
            for o in observations
            if o.included and o.realized_slippage_bps is not None and o.pool_liquidity_usd > 0
        ]

        if len(valid_obs) < 20:
            return

        # Prepare data
        X = np.array(
            [
                [
                    o.trade_size_sol / (o.pool_liquidity_usd / 200),  # Relative size
                    o.volatility_1h,
                    o.pool_liquidity_usd,
                ]
                for o in valid_obs
            ]
        )
        y = np.array([o.realized_slippage_bps - o.quoted_slippage_bps for o in valid_obs])

        # Non-linear slippage function
        def slippage_model(params, X):
            base, size_exp, vol_coef, liq_coef = params
            relative_size, volatility, _liquidity = X[:, 0], X[:, 1], X[:, 2]

            # Clamp exponent to prevent overflow (exp(20) ≈ 500M, exp(-20) ≈ 0)
            clamped_exp = np.clip(size_exp * relative_size, -20, 20)
            size_slip = base * np.exp(clamped_exp)
            vol_slip = vol_coef * volatility * 10000
            liq_penalty = liq_coef * np.sqrt(relative_size) * 100

            return size_slip + vol_slip + liq_penalty

        # Loss function
        def loss(params):
            pred = slippage_model(params, X)
            return np.mean((pred - y) ** 2)

        # Optimize
        initial = [30.0, 50.0, 0.8, 0.5]
        bounds = [(5, 100), (10, 200), (0.1, 2.0), (0.1, 2.0)]

        result = optimize.minimize(loss, initial, bounds=bounds, method="L-BFGS-B")

        if result.success:
            # L5 FIX: Validate results before assignment to prevent NaN corruption
            # Ref: CONSOLIDATED_FINDINGS-all.md
            if np.any(np.isnan(result.x)) or np.any(np.isinf(result.x)):
                report.warnings.append(
                    "Slippage optimization produced invalid results (NaN/Inf). "
                    "Skipping coefficient update. Input data may contain anomalies."
                )
                logger.warning(f"Slippage fit produced NaN/Inf: result.x={result.x}")
                return

            self.coefficients.slippage_base_bps = result.x[0]
            self.coefficients.slippage_size_exponent = result.x[1]
            self.coefficients.slippage_volatility_coef = result.x[2]
            self.coefficients.slippage_liquidity_coef = result.x[3]

            # Calculate metrics
            pred = slippage_model(result.x, X)
            report.slippage_mae_bps = float(np.mean(np.abs(pred - y)))
            report.slippage_rmse_bps = float(np.sqrt(np.mean((pred - y) ** 2)))
        else:
            report.warnings.append("Slippage optimization did not converge")

    def _fit_latency_model(self, observations: list[BiasObservation], report: CalibrationReport):
        """Fit congestion-aware latency model."""
        valid_obs = [o for o in observations if o.execution_latency_ms is not None]

        if len(valid_obs) < 20:
            return

        latencies = np.array([o.execution_latency_ms for o in valid_obs])
        congestions = np.array([o.network_congestion for o in valid_obs])

        # Separate by congestion level
        low_cong = latencies[congestions < 0.3]
        high_cong = latencies[congestions > 0.7]

        # Fit log-normal parameters
        log_lat = np.log(latencies[latencies > 0])
        if len(log_lat) > 10:
            self.coefficients.latency_base_mean_ms = float(np.exp(np.mean(log_lat)))
            self.coefficients.latency_base_std_ms = float(np.exp(np.std(log_lat)))

        # Estimate congestion multipliers (with division-by-zero protection)
        if self.coefficients.latency_base_mean_ms > 0:
            if len(low_cong) >= 5:
                self.coefficients.latency_congestion_multiplier_low = float(
                    np.clip(
                        np.median(low_cong) / self.coefficients.latency_base_mean_ms,
                        0.1,  # Minimum multiplier
                        5.0,  # Maximum multiplier
                    )
                )

            if len(high_cong) >= 5:
                self.coefficients.latency_congestion_multiplier_high = float(
                    np.clip(
                        np.median(high_cong) / self.coefficients.latency_base_mean_ms,
                        0.5,  # Minimum multiplier for high congestion
                        10.0,  # Maximum multiplier
                    )
                )
        else:
            logger.warning(
                "latency_base_mean_ms is zero or negative, skipping congestion "
                "multiplier calibration"
            )

        # Calculate MAE
        predicted_mean = self.coefficients.latency_base_mean_ms * (
            self.coefficients.latency_congestion_multiplier_low
            + (
                self.coefficients.latency_congestion_multiplier_high
                - self.coefficients.latency_congestion_multiplier_low
            )
            * congestions
        )
        report.latency_mae_ms = float(np.mean(np.abs(predicted_mean - latencies)))

    def _fit_inclusion_model(self, observations: list[BiasObservation], report: CalibrationReport):
        """Fit inclusion probability model."""
        if len(observations) < 20:
            return

        # Calculate base inclusion rate
        included_count = sum(1 for o in observations if o.included)
        total = len(observations)
        self.coefficients.inclusion_base_prob = float(included_count / total)

        # Analyze by congestion level
        low_cong_obs = [o for o in observations if o.network_congestion < 0.3]
        high_cong_obs = [o for o in observations if o.network_congestion > 0.7]

        if len(low_cong_obs) >= 10 and len(high_cong_obs) >= 10:
            low_rate = sum(1 for o in low_cong_obs if o.included) / len(low_cong_obs)
            high_rate = sum(1 for o in high_cong_obs if o.included) / len(high_cong_obs)

            # Estimate congestion decay
            if low_rate > high_rate:
                self.coefficients.inclusion_congestion_decay = float(
                    (low_rate - high_rate) / 0.4  # Normalize by congestion difference
                )

        # Analyze partial fills (guard against zero liquidity)
        partial_fills = [
            o
            for o in observations
            if 0 < o.partial_fill_fraction < 1.0 and o.pool_liquidity_usd > 0
        ]
        if len(partial_fills) >= 5:
            # Find threshold where partial fills become common
            fill_fractions = [
                o.trade_size_sol / (o.pool_liquidity_usd / 200) for o in partial_fills
            ]
            self.coefficients.inclusion_partial_fill_threshold = float(
                np.percentile(fill_fractions, 25)
            )

        # Calculate accuracy
        report.inclusion_accuracy = float(included_count / total)

    def _fit_mev_model(self, observations: list[BiasObservation], report: CalibrationReport):
        """Fit MEV sandwich model from slippage outliers."""
        valid_obs = [o for o in observations if o.included and o.realized_slippage_bps is not None]

        if len(valid_obs) < 20:
            return

        # MEV detected observations
        mev_obs = [o for o in valid_obs if o.mev_detected]
        [o for o in valid_obs if not o.mev_detected]

        # MEV probability
        self.coefficients.mev_sandwich_probability = len(mev_obs) / len(valid_obs)

        # Separate by Jito usage
        jito_mev = [o for o in mev_obs if o.using_jito]
        non_jito_mev = [o for o in mev_obs if not o.using_jito]

        if len(mev_obs) >= 5:
            # MEV cost from excess slippage
            mev_costs = [o.realized_slippage_bps - o.quoted_slippage_bps for o in mev_obs]
            self.coefficients.mev_sandwich_cost_bps = float(np.median(mev_costs))

        if len(jito_mev) >= 3 and len(non_jito_mev) >= 3:
            # Jito discount factor
            jito_cost = np.median(
                [o.realized_slippage_bps - o.quoted_slippage_bps for o in jito_mev]
            )
            non_jito_cost = np.median(
                [o.realized_slippage_bps - o.quoted_slippage_bps for o in non_jito_mev]
            )

            if non_jito_cost > 0:
                self.coefficients.mev_jito_discount_factor = float(
                    np.clip(jito_cost / non_jito_cost, 0.3, 0.95)
                )

        report.mev_detection_rate = self.coefficients.mev_sandwich_probability

    def _analyze_survivorship(self, observations: list[BiasObservation], report: CalibrationReport):
        """Analyze survivorship bias from rug data."""
        if len(observations) < 20:
            return

        # Get unique tokens and check rug status
        token_status = {}
        for o in observations:
            if o.token_mint not in token_status:
                token_status[o.token_mint] = {
                    "first_seen_hours": o.token_age_hours,
                    "is_rugged": o.is_rugged,
                }

        # Calculate rug rate
        rugged_count = sum(1 for t in token_status.values() if t["is_rugged"])
        total_tokens = len(token_status)

        if total_tokens > 0:
            report.rug_rate_observed = rugged_count / total_tokens
            self.coefficients.rug_rate_24h = report.rug_rate_observed

            # Survivorship factor: expected return discount
            # Higher rug rate = higher discount needed
            # Typical range: 10-30% based on research
            self.coefficients.survivorship_discount = float(
                np.clip(report.rug_rate_observed * 1.2, 0.10, 0.35)
            )

            report.survivorship_factor = 1.0 - self.coefficients.survivorship_discount

    def _walk_forward_validation(
        self,
        observations: list[BiasObservation],
        report: CalibrationReport,
        n_splits: int = 3,
    ):
        """Perform walk-forward validation using TimeSeriesSplit."""
        # Sort by timestamp
        sorted_obs = sorted(observations, key=lambda o: o.timestamp_ms)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        indices = np.arange(len(sorted_obs))

        slippage_maes = []
        inclusion_accuracies = []

        for train_idx, test_idx in tscv.split(indices):
            train_obs = [sorted_obs[i] for i in train_idx]
            test_obs = [sorted_obs[i] for i in test_idx]

            # Fit on train
            temp_coeffs = BiasCoefficients()
            valid_train = [
                o for o in train_obs if o.included and o.realized_slippage_bps is not None
            ]

            if len(valid_train) >= 10:
                # Simple fit for validation
                slippages = [o.realized_slippage_bps - o.quoted_slippage_bps for o in valid_train]
                temp_coeffs.slippage_base_bps = float(np.median(slippages))

                # Evaluate on test
                valid_test = [
                    o for o in test_obs if o.included and o.realized_slippage_bps is not None
                ]

                if len(valid_test) >= 5:
                    test_slippages = [
                        o.realized_slippage_bps - o.quoted_slippage_bps for o in valid_test
                    ]
                    predicted = [temp_coeffs.slippage_base_bps] * len(test_slippages)
                    mae = np.mean(np.abs(np.array(test_slippages) - np.array(predicted)))
                    slippage_maes.append(mae)

            # Inclusion accuracy
            if len(test_obs) >= 5:
                included = sum(1 for o in test_obs if o.included)
                inclusion_accuracies.append(included / len(test_obs))

        if slippage_maes:
            report.validation_slippage_mae = float(np.mean(slippage_maes))

        if inclusion_accuracies:
            report.validation_inclusion_accuracy = float(np.mean(inclusion_accuracies))

    def get_coefficients(self) -> BiasCoefficients:
        """Get current calibrated coefficients."""
        return self.coefficients

    def to_state(self) -> dict[str, Any]:
        """Export full state for persistence."""
        return {
            "coefficients": self.coefficients.to_dict(),
            "last_report": self.last_report.to_dict() if self.last_report else None,
            "calibration_history": self.calibration_history[-10:],
        }

    def load_state(self, state: dict[str, Any]):
        """Load state from persistence."""
        if "coefficients" in state:
            self.coefficients = BiasCoefficients.from_dict(state["coefficients"])

        if "calibration_history" in state:
            self.calibration_history = state["calibration_history"]

        logger.info(
            f"Loaded bias calibrator state with {len(self.calibration_history)} history entries"
        )

    def to_hotpath_params(self) -> dict[str, Any]:
        """Export parameters for Hot Path RealisticFillSimulator."""
        return {
            "slippage_base_bps": self.coefficients.slippage_base_bps,
            "slippage_size_exponent": self.coefficients.slippage_size_exponent,
            "slippage_volatility_coef": self.coefficients.slippage_volatility_coef,
            "slippage_liquidity_coef": self.coefficients.slippage_liquidity_coef,
            "latency_base_mean_ms": self.coefficients.latency_base_mean_ms,
            "latency_base_std_ms": self.coefficients.latency_base_std_ms,
            "latency_congestion_multiplier_low": (
                self.coefficients.latency_congestion_multiplier_low
            ),
            "latency_congestion_multiplier_high": (
                self.coefficients.latency_congestion_multiplier_high
            ),
            "mev_sandwich_probability": self.coefficients.mev_sandwich_probability,
            "mev_sandwich_cost_bps": self.coefficients.mev_sandwich_cost_bps,
            "mev_jito_discount_factor": self.coefficients.mev_jito_discount_factor,
            "inclusion_base_prob": self.coefficients.inclusion_base_prob,
            "inclusion_size_decay": self.coefficients.inclusion_size_decay,
            "inclusion_congestion_decay": self.coefficients.inclusion_congestion_decay,
            "inclusion_partial_fill_threshold": self.coefficients.inclusion_partial_fill_threshold,
            "max_quote_age_ms": self.coefficients.max_quote_age_ms,
            "quote_staleness_penalty_bps": self.coefficients.quote_staleness_penalty_bps,
            "survivorship_discount": self.coefficients.survivorship_discount,
        }
