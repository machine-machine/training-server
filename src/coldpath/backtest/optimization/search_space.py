"""
Search space definitions for hyperparameter optimization.

Defines parameter ranges, types, and sampling strategies for
trading strategy optimization.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ParameterType(Enum):
    """Parameter sampling type for Optuna."""

    UNIFORM = "uniform"  # Uniform distribution
    LOG_UNIFORM = "log"  # Log-uniform (for parameters spanning orders of magnitude)
    INT = "int"  # Integer values
    INT_LOG = "int_log"  # Log-uniform integer
    CATEGORICAL = "categorical"  # Discrete choices


@dataclass
class ParameterDefinition:
    """Definition of a single hyperparameter."""

    name: str
    param_type: ParameterType
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None
    default: Any | None = None
    description: str = ""

    def suggest(self, trial) -> Any:
        """Suggest a value using Optuna trial."""
        if self.param_type == ParameterType.UNIFORM:
            return trial.suggest_float(self.name, self.low, self.high)
        elif self.param_type == ParameterType.LOG_UNIFORM:
            return trial.suggest_float(self.name, self.low, self.high, log=True)
        elif self.param_type == ParameterType.INT:
            return trial.suggest_int(self.name, int(self.low), int(self.high))
        elif self.param_type == ParameterType.INT_LOG:
            return trial.suggest_int(self.name, int(self.low), int(self.high), log=True)
        elif self.param_type == ParameterType.CATEGORICAL:
            return trial.suggest_categorical(self.name, self.choices)
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")

    def to_dict(self) -> dict[str, Any]:
        """Export parameter definition."""
        return {
            "name": self.name,
            "param_type": self.param_type.value,
            "low": self.low,
            "high": self.high,
            "choices": self.choices,
            "default": self.default,
            "description": self.description,
        }


@dataclass
class SearchSpace:
    """Collection of hyperparameters to optimize."""

    name: str
    parameters: dict[str, ParameterDefinition] = field(default_factory=dict)
    description: str = ""

    def add_parameter(
        self,
        name: str,
        param_type: ParameterType,
        low: float | None = None,
        high: float | None = None,
        choices: list[Any] | None = None,
        default: Any | None = None,
        description: str = "",
    ) -> "SearchSpace":
        """Add a parameter to the search space."""
        self.parameters[name] = ParameterDefinition(
            name=name,
            param_type=param_type,
            low=low,
            high=high,
            choices=choices,
            default=default,
            description=description,
        )
        return self

    def suggest_all(self, trial) -> dict[str, Any]:
        """Suggest values for all parameters."""
        return {name: param.suggest(trial) for name, param in self.parameters.items()}

    def get_defaults(self) -> dict[str, Any]:
        """Get default values for all parameters."""
        return {
            name: param.default
            for name, param in self.parameters.items()
            if param.default is not None
        }

    def merge(self, other: "SearchSpace") -> "SearchSpace":
        """Merge with another search space."""
        merged = SearchSpace(
            name=f"{self.name}+{other.name}",
            parameters={**self.parameters, **other.parameters},
        )
        return merged

    def __len__(self) -> int:
        return len(self.parameters)

    def to_dict(self) -> dict[str, Any]:
        """Export search space definition."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {name: param.to_dict() for name, param in self.parameters.items()},
        }


# =============================================================================
# Pre-defined Search Spaces
# =============================================================================


def create_trading_search_space() -> SearchSpace:
    """Create search space for trading parameters."""
    space = SearchSpace(
        name="trading",
        description="Core trading parameters: entry/exit, position sizing",
    )

    # Entry/Exit conditions
    space.add_parameter(
        name="min_liquidity_usd",
        param_type=ParameterType.LOG_UNIFORM,
        low=1_000,
        high=100_000,
        default=10_000,
        description="Minimum pool liquidity in USD",
    )
    space.add_parameter(
        name="take_profit_pct",
        param_type=ParameterType.UNIFORM,
        low=10,
        high=200,
        default=50,
        description="Take profit percentage",
    )
    space.add_parameter(
        name="stop_loss_pct",
        param_type=ParameterType.UNIFORM,
        low=5,
        high=50,
        default=15,
        description="Stop loss percentage",
    )
    space.add_parameter(
        name="max_hold_periods",
        param_type=ParameterType.INT,
        low=10,
        high=180,
        default=60,
        description="Maximum hold time in periods",
    )

    # Position sizing
    space.add_parameter(
        name="kelly_fraction",
        param_type=ParameterType.UNIFORM,
        low=0.1,
        high=0.5,
        default=0.25,
        description="Kelly criterion fraction for position sizing",
    )
    space.add_parameter(
        name="max_position_pct",
        param_type=ParameterType.UNIFORM,
        low=0.05,
        high=0.25,
        default=0.10,
        description="Maximum position size as fraction of capital",
    )
    space.add_parameter(
        name="max_concurrent_positions",
        param_type=ParameterType.INT,
        low=1,
        high=10,
        default=5,
        description="Maximum number of concurrent positions",
    )

    return space


def create_risk_search_space() -> SearchSpace:
    """Create search space for risk management parameters."""
    space = SearchSpace(
        name="risk",
        description="Risk management and slippage parameters",
    )

    # Slippage
    space.add_parameter(
        name="slippage_tolerance_bps",
        param_type=ParameterType.INT,
        low=50,
        high=500,
        default=200,
        description="Slippage tolerance in basis points",
    )
    space.add_parameter(
        name="mev_penalty_bps",
        param_type=ParameterType.INT,
        low=0,
        high=200,
        default=100,
        description="MEV penalty assumption in basis points",
    )

    # Risk limits
    space.add_parameter(
        name="max_daily_loss_pct",
        param_type=ParameterType.UNIFORM,
        low=5,
        high=25,
        default=10,
        description="Maximum daily loss before stopping",
    )
    space.add_parameter(
        name="max_drawdown_pct",
        param_type=ParameterType.UNIFORM,
        low=10,
        high=40,
        default=20,
        description="Maximum drawdown allowed",
    )
    space.add_parameter(
        name="inclusion_prob_pessimistic",
        param_type=ParameterType.UNIFORM,
        low=0.3,
        high=0.9,
        default=0.7,
        description="Pessimistic transaction inclusion probability",
    )

    return space


def create_signal_search_space() -> SearchSpace:
    """Create search space for signal generation parameters."""
    space = SearchSpace(
        name="signal",
        description="Signal generation and filtering parameters",
    )

    # Momentum signals
    space.add_parameter(
        name="min_price_change_pct",
        param_type=ParameterType.UNIFORM,
        low=1,
        high=20,
        default=5,
        description="Minimum price change for entry signal",
    )
    space.add_parameter(
        name="min_volume_spike",
        param_type=ParameterType.UNIFORM,
        low=1.0,
        high=5.0,
        default=2.0,
        description="Minimum volume spike multiplier",
    )
    space.add_parameter(
        name="momentum_lookback",
        param_type=ParameterType.INT,
        low=1,
        high=30,
        default=5,
        description="Lookback periods for momentum calculation",
    )

    # Technical indicators
    space.add_parameter(
        name="rsi_period",
        param_type=ParameterType.INT,
        low=5,
        high=21,
        default=14,
        description="RSI calculation period",
    )
    space.add_parameter(
        name="rsi_oversold",
        param_type=ParameterType.UNIFORM,
        low=20,
        high=40,
        default=30,
        description="RSI oversold threshold",
    )
    space.add_parameter(
        name="rsi_overbought",
        param_type=ParameterType.UNIFORM,
        low=60,
        high=80,
        default=70,
        description="RSI overbought threshold",
    )

    # Filtering
    space.add_parameter(
        name="min_holder_count",
        param_type=ParameterType.INT,
        low=10,
        high=500,
        default=50,
        description="Minimum holder count filter",
    )
    space.add_parameter(
        name="max_top_holder_pct",
        param_type=ParameterType.UNIFORM,
        low=20,
        high=80,
        default=50,
        description="Maximum top holder percentage",
    )
    space.add_parameter(
        name="min_lp_burn_pct",
        param_type=ParameterType.UNIFORM,
        low=0,
        high=100,
        default=50,
        description="Minimum LP burn percentage",
    )

    return space


def create_full_search_space() -> SearchSpace:
    """Create the full 15+ parameter search space."""
    trading = create_trading_search_space()
    risk = create_risk_search_space()
    signal = create_signal_search_space()

    full = trading.merge(risk).merge(signal)
    full.name = "full"
    full.description = "Complete optimization search space (15+ parameters)"

    return full


# Pre-instantiated search spaces for convenience
TRADING_SEARCH_SPACE = create_trading_search_space()
RISK_SEARCH_SPACE = create_risk_search_space()
SIGNAL_SEARCH_SPACE = create_signal_search_space()
FULL_SEARCH_SPACE = create_full_search_space()
