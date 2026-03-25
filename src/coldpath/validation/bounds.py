"""
Configurable validation bounds for different domains.

Centralizes magic numbers and provides domain-specific bound configurations
for trading, calibration, and ML model validation.
"""

from dataclasses import dataclass

import numpy as np

from .numeric import clamp, is_valid_float


@dataclass
class ValidationBounds:
    """Base class for configurable validation bounds.

    # Extensibility
    Subclass to add domain-specific bounds:

    ```python
    @dataclass
    class TokenBounds(ValidationBounds):
        min_holder_count: int = 10
        max_concentration: float = 0.9
    ```
    """

    @classmethod
    def default(cls) -> "ValidationBounds":
        """Create default bounds."""
        return cls()

    def clamp_value(
        self, value: float, min_val: float, max_val: float, default: float | None = None
    ) -> float:
        """Clamp a value to bounds with optional default."""
        return clamp(value, min_val, max_val, default)


@dataclass
class TradingBounds(ValidationBounds):
    """Bounds for trading-related validation."""

    # Position sizing
    min_position_sol: float = 0.001
    max_position_sol: float = 100.0

    # Liquidity requirements
    min_liquidity_usd: float = 1000.0
    max_liquidity_usd: float = 100_000_000.0

    # Slippage bounds (basis points)
    min_slippage_bps: int = 0
    max_slippage_bps: int = 5000  # 50%

    # Price bounds (for sanity checks)
    min_price_usd: float = 1e-12
    max_price_usd: float = 1e12

    # PnL bounds
    max_pnl_percent: float = 1000.0  # 1000% max gain
    min_pnl_percent: float = -100.0  # -100% max loss

    # Risk limits
    max_drawdown_percent: float = 100.0
    max_daily_loss_sol: float = 1000.0

    @classmethod
    def conservative(cls) -> "TradingBounds":
        """Create conservative bounds for lower risk."""
        return cls(
            min_position_sol=0.01,
            max_position_sol=1.0,
            min_liquidity_usd=10000.0,
            max_slippage_bps=500,
            max_drawdown_percent=20.0,
            max_daily_loss_sol=0.5,
        )

    @classmethod
    def aggressive(cls) -> "TradingBounds":
        """Create aggressive bounds for higher risk tolerance."""
        return cls(
            min_position_sol=0.001,
            max_position_sol=10.0,
            min_liquidity_usd=1000.0,
            max_slippage_bps=2000,
            max_drawdown_percent=50.0,
            max_daily_loss_sol=5.0,
        )

    def clamp_position_size(self, size: float) -> float:
        """Clamp position size to valid range."""
        return self.clamp_value(
            size, self.min_position_sol, self.max_position_sol, self.min_position_sol
        )

    def clamp_slippage_bps(self, slippage: int) -> int:
        """Clamp slippage to valid range."""
        return int(self.clamp_value(slippage, self.min_slippage_bps, self.max_slippage_bps, 0))

    def clamp_pnl_percent(self, pnl: float) -> float:
        """Clamp PnL percentage to valid range."""
        return self.clamp_value(pnl, self.min_pnl_percent, self.max_pnl_percent, 0.0)

    def is_valid_liquidity(self, liquidity_usd: float) -> bool:
        """Check if liquidity meets minimum requirements."""
        if not is_valid_float(liquidity_usd, allow_zero=True):
            return False
        return liquidity_usd >= self.min_liquidity_usd


@dataclass
class CalibrationBounds(ValidationBounds):
    """Bounds for calibration model parameters."""

    # Slippage model
    slippage_base_bps: tuple[float, float] = (5.0, 200.0)
    slippage_size_exp: tuple[float, float] = (10.0, 500.0)
    slippage_vol_coef: tuple[float, float] = (0.1, 5.0)
    slippage_liq_coef: tuple[float, float] = (0.1, 5.0)

    # Latency model
    latency_base_ms: tuple[float, float] = (10.0, 5000.0)
    latency_std_ms: tuple[float, float] = (5.0, 2000.0)
    latency_cong_mult_low: tuple[float, float] = (0.1, 5.0)
    latency_cong_mult_high: tuple[float, float] = (0.5, 20.0)

    # MEV model
    mev_prob: tuple[float, float] = (0.0, 1.0)
    mev_cost_bps: tuple[float, float] = (0.0, 500.0)
    mev_jito_discount: tuple[float, float] = (0.1, 1.0)

    # Inclusion model
    inclusion_base: tuple[float, float] = (0.5, 1.0)
    inclusion_decay: tuple[float, float] = (0.0, 0.5)

    # Survivorship
    survivorship_discount: tuple[float, float] = (0.05, 0.50)
    rug_rate: tuple[float, float] = (0.01, 0.50)

    def clamp_slippage_base(self, value: float) -> float:
        """Clamp slippage base to valid range."""
        return self.clamp_value(value, *self.slippage_base_bps, default=30.0)

    def clamp_latency_multiplier(self, value: float, high: bool = False) -> float:
        """Clamp latency multiplier to valid range."""
        bounds = self.latency_cong_mult_high if high else self.latency_cong_mult_low
        default = 2.5 if high else 0.8
        return self.clamp_value(value, *bounds, default=default)

    def clamp_mev_probability(self, value: float) -> float:
        """Clamp MEV probability to valid range."""
        return self.clamp_value(value, *self.mev_prob, default=0.25)

    def clamp_inclusion_base(self, value: float) -> float:
        """Clamp inclusion base probability to valid range."""
        return self.clamp_value(value, *self.inclusion_base, default=0.88)


@dataclass
class MLModelBounds(ValidationBounds):
    """Bounds for ML model inputs and outputs."""

    # Feature bounds
    min_feature_value: float = -1e6
    max_feature_value: float = 1e6

    # Prediction bounds
    min_prediction: float = 0.0
    max_prediction: float = 1.0

    # Confidence bounds
    min_confidence: float = 0.0
    max_confidence: float = 1.0

    # Score bounds
    min_score: float = 0.0
    max_score: float = 1.0

    # Training bounds
    min_samples_for_training: int = 10
    max_samples_for_training: int = 1_000_000

    # Hyperparameter bounds
    learning_rate: tuple[float, float] = (1e-6, 1.0)
    regularization: tuple[float, float] = (0.0, 10.0)
    dropout: tuple[float, float] = (0.0, 0.9)

    def clamp_prediction(self, value: float) -> float:
        """Clamp prediction to valid range."""
        return self.clamp_value(value, self.min_prediction, self.max_prediction, 0.5)

    def clamp_confidence(self, value: float) -> float:
        """Clamp confidence to valid range."""
        return self.clamp_value(value, self.min_confidence, self.max_confidence, 0.0)

    def clamp_score(self, value: float) -> float:
        """Clamp score to valid range."""
        return self.clamp_value(value, self.min_score, self.max_score, 0.0)

    def clamp_features(self, features: np.ndarray) -> np.ndarray:
        """Clamp feature array to valid range."""
        return np.clip(features, self.min_feature_value, self.max_feature_value)

    def is_valid_sample_count(self, count: int) -> bool:
        """Check if sample count is valid for training."""
        return self.min_samples_for_training <= count <= self.max_samples_for_training

    def clamp_learning_rate(self, value: float) -> float:
        """Clamp learning rate to valid range."""
        return self.clamp_value(value, *self.learning_rate, default=0.01)

    def clamp_regularization(self, value: float) -> float:
        """Clamp regularization to valid range."""
        return self.clamp_value(value, *self.regularization, default=0.01)

    def clamp_dropout(self, value: float) -> float:
        """Clamp dropout to valid range."""
        return self.clamp_value(value, *self.dropout, default=0.1)

    def validate_hyperparameters(
        self,
        learning_rate: float | None = None,
        regularization: float | None = None,
        dropout: float | None = None,
    ) -> dict:
        """Validate and clamp hyperparameters.

        Returns dict with validated (clamped) values.
        Logs warnings when values are clamped.

        Args:
            learning_rate: Learning rate value to validate
            regularization: Regularization value to validate
            dropout: Dropout value to validate

        Returns:
            Dict with validated hyperparameters
        """
        import logging

        logger = logging.getLogger(__name__)

        result = {}

        if learning_rate is not None:
            if not is_valid_float(learning_rate):
                logger.warning(f"Invalid learning_rate: {learning_rate}, using default 0.01")
                result["learning_rate"] = 0.01
            elif learning_rate < self.learning_rate[0] or learning_rate > self.learning_rate[1]:
                clamped = self.clamp_learning_rate(learning_rate)
                logger.warning(
                    f"learning_rate {learning_rate} clamped to {clamped} "
                    f"(valid range: {self.learning_rate})"
                )
                result["learning_rate"] = clamped
            else:
                result["learning_rate"] = learning_rate

        if regularization is not None:
            if not is_valid_float(regularization):
                logger.warning(f"Invalid regularization: {regularization}, using default 0.01")
                result["regularization"] = 0.01
            elif regularization < self.regularization[0] or regularization > self.regularization[1]:
                clamped = self.clamp_regularization(regularization)
                logger.warning(
                    f"regularization {regularization} clamped to {clamped} "
                    f"(valid range: {self.regularization})"
                )
                result["regularization"] = clamped
            else:
                result["regularization"] = regularization

        if dropout is not None:
            if not is_valid_float(dropout):
                logger.warning(f"Invalid dropout: {dropout}, using default 0.1")
                result["dropout"] = 0.1
            elif dropout < self.dropout[0] or dropout > self.dropout[1]:
                clamped = self.clamp_dropout(dropout)
                logger.warning(
                    f"dropout {dropout} clamped to {clamped} (valid range: {self.dropout})"
                )
                result["dropout"] = clamped
            else:
                result["dropout"] = dropout

        return result

    def validate_training_config(self, config: dict) -> dict:
        """Validate a training configuration dict.

        Validates common hyperparameters and returns sanitized config.
        Unknown keys are passed through unchanged.

        Args:
            config: Training configuration dictionary

        Returns:
            Validated configuration dictionary
        """
        result = config.copy()

        # Validate known hyperparameters
        hp_keys = {
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "dropout": self.dropout,
        }

        for key, bounds in hp_keys.items():
            if key in result:
                value = result[key]
                if not is_valid_float(value):
                    result[key] = bounds[0]  # Use min as default
                else:
                    result[key] = self.clamp_value(value, *bounds, default=bounds[0])

        return result
