"""
Robustness Validator - Monte Carlo parameter perturbation testing.

Validates strategy robustness by:
1. Perturbing each parameter by +/-N% (default 10%)
2. Running backtest on perturbed params
3. Measuring sensitivity (Sharpe change / param change)
4. Computing overall robustness score

Good strategies should be robust to small parameter changes.
A high robustness score means the strategy generalizes well
and is not overfit to exact parameter values.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParameterSensitivity:
    """Sensitivity analysis for a single parameter.

    Attributes:
        parameter_name: Name of the parameter.
        base_value: Original parameter value.
        sensitivity: Normalized sensitivity (Sharpe std / param perturbation).
        sharpe_mean: Mean Sharpe across perturbations.
        sharpe_std: Standard deviation of Sharpe across perturbations.
        sharpe_min: Minimum Sharpe across perturbations.
        sharpe_max: Maximum Sharpe across perturbations.
        is_sensitive: Whether this parameter is highly sensitive.
    """

    parameter_name: str
    base_value: float
    sensitivity: float
    sharpe_mean: float
    sharpe_std: float
    sharpe_min: float
    sharpe_max: float
    is_sensitive: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "parameter_name": self.parameter_name,
            "base_value": self.base_value,
            "sensitivity": self.sensitivity,
            "sharpe_mean": self.sharpe_mean,
            "sharpe_std": self.sharpe_std,
            "sharpe_min": self.sharpe_min,
            "sharpe_max": self.sharpe_max,
            "is_sensitive": self.is_sensitive,
        }


@dataclass
class RobustnessReport:
    """Complete robustness validation report.

    Attributes:
        robustness_score: Overall robustness (0-1, higher = more robust).
        base_sharpe: Sharpe with original parameters.
        mean_perturbed_sharpe: Mean Sharpe across all perturbations.
        std_perturbed_sharpe: Std of Sharpe across all perturbations.
        num_trials: Number of perturbation trials run.
        perturbation_pct: Perturbation percentage used.
        parameter_sensitivities: Per-parameter sensitivity analysis.
        most_sensitive_param: Name of the most sensitive parameter.
        is_robust: Whether the strategy passes robustness threshold.
    """

    robustness_score: float
    base_sharpe: float
    mean_perturbed_sharpe: float
    std_perturbed_sharpe: float
    num_trials: int
    perturbation_pct: float
    parameter_sensitivities: list[ParameterSensitivity]
    most_sensitive_param: str
    is_robust: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "robustness_score": self.robustness_score,
            "base_sharpe": self.base_sharpe,
            "mean_perturbed_sharpe": self.mean_perturbed_sharpe,
            "std_perturbed_sharpe": self.std_perturbed_sharpe,
            "num_trials": self.num_trials,
            "perturbation_pct": self.perturbation_pct,
            "parameter_sensitivities": [s.to_dict() for s in self.parameter_sensitivities],
            "most_sensitive_param": self.most_sensitive_param,
            "is_robust": self.is_robust,
        }


class RobustnessValidator:
    """
    Validate strategy robustness via parameter perturbation.

    For each trial:
    - Randomly perturb each parameter by +/-N%
    - Run backtest with perturbed parameters
    - Record Sharpe ratio

    Robustness score = 1 - (std(Sharpe) / mean(Sharpe))
    A score close to 1.0 means the strategy is robust.
    A score close to 0.0 means it is sensitive to parameter changes.

    Usage:
        validator = RobustnessValidator()
        report = await validator.test_robustness(
            params={"stop_loss_pct": 8.0, ...},
            historical_data=data,
            backtest_fn=my_backtest,
            perturbation_pct=10.0,
            num_trials=100,
        )
    """

    def __init__(
        self,
        sensitivity_threshold: float = 0.3,
        robustness_threshold: float = 0.6,
        random_seed: int | None = 42,
    ):
        """Initialize the robustness validator.

        Args:
            sensitivity_threshold: Threshold above which a parameter
                is considered "sensitive".
            robustness_threshold: Minimum robustness score to pass.
            random_seed: Random seed for reproducibility.
        """
        self.sensitivity_threshold = sensitivity_threshold
        self.robustness_threshold = robustness_threshold

        if random_seed is not None:
            np.random.seed(random_seed)

    async def test_robustness(
        self,
        params: dict[str, Any],
        historical_data: Any,
        backtest_fn: Callable,
        perturbation_pct: float = 10.0,
        num_trials: int = 100,
        param_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> RobustnessReport:
        """
        Run Monte Carlo parameter perturbation test.

        Args:
            params: Base strategy parameters.
            historical_data: Historical data for backtesting.
            backtest_fn: Backtest function(params, data) -> result dict.
            perturbation_pct: Maximum perturbation percentage.
            num_trials: Number of random perturbation trials.
            param_bounds: Optional bounds for parameters.

        Returns:
            RobustnessReport with robustness analysis.
        """
        logger.info(
            "Starting robustness test: %d trials, %.1f%% perturbation",
            num_trials,
            perturbation_pct,
        )

        # Get base Sharpe
        base_result = await self._evaluate(params, historical_data, backtest_fn)
        base_sharpe = base_result.get("sharpe_ratio", base_result.get("sharpe", 0.0))

        # Identify numeric parameters
        numeric_params = {k: v for k, v in params.items() if isinstance(v, (int, float)) and v != 0}

        # Run perturbation trials
        all_sharpes: list[float] = []
        per_param_sharpes: dict[str, list[float]] = {k: [] for k in numeric_params}

        for _trial in range(num_trials):
            perturbed = self._perturb_params(
                params,
                numeric_params,
                perturbation_pct,
                param_bounds,
            )

            result = await self._evaluate(perturbed, historical_data, backtest_fn)
            sharpe = result.get("sharpe_ratio", result.get("sharpe", 0.0))
            all_sharpes.append(sharpe)

            # Track which param was most perturbed
            for k in numeric_params:
                if perturbed.get(k) != params.get(k):
                    per_param_sharpes[k].append(sharpe)

        # Compute statistics
        all_sharpes_arr = np.array(all_sharpes)
        mean_sharpe = float(np.mean(all_sharpes_arr))
        std_sharpe = float(np.std(all_sharpes_arr))

        # Robustness score
        if mean_sharpe != 0:
            robustness = max(0.0, min(1.0, 1.0 - (std_sharpe / abs(mean_sharpe))))
        else:
            robustness = 0.0

        # Per-parameter sensitivity
        sensitivities: list[ParameterSensitivity] = []
        for param_name, param_sharpes in per_param_sharpes.items():
            if not param_sharpes:
                continue

            ps_arr = np.array(param_sharpes)
            sensitivity = float(np.std(ps_arr)) if len(ps_arr) > 1 else 0.0

            sensitivities.append(
                ParameterSensitivity(
                    parameter_name=param_name,
                    base_value=float(numeric_params[param_name]),
                    sensitivity=sensitivity,
                    sharpe_mean=float(np.mean(ps_arr)),
                    sharpe_std=float(np.std(ps_arr)),
                    sharpe_min=float(np.min(ps_arr)),
                    sharpe_max=float(np.max(ps_arr)),
                    is_sensitive=sensitivity > self.sensitivity_threshold,
                )
            )

        # Sort by sensitivity (most sensitive first)
        sensitivities.sort(key=lambda s: s.sensitivity, reverse=True)

        most_sensitive = sensitivities[0].parameter_name if sensitivities else "none"

        report = RobustnessReport(
            robustness_score=robustness,
            base_sharpe=base_sharpe,
            mean_perturbed_sharpe=mean_sharpe,
            std_perturbed_sharpe=std_sharpe,
            num_trials=num_trials,
            perturbation_pct=perturbation_pct,
            parameter_sensitivities=sensitivities,
            most_sensitive_param=most_sensitive,
            is_robust=robustness >= self.robustness_threshold,
        )

        logger.info(
            "Robustness test complete: score=%.2f, base_sharpe=%.2f, "
            "mean_perturbed=%.2f, std=%.2f, most_sensitive=%s",
            robustness,
            base_sharpe,
            mean_sharpe,
            std_sharpe,
            most_sensitive,
        )

        return report

    def _perturb_params(
        self,
        params: dict[str, Any],
        numeric_params: dict[str, Any],
        perturbation_pct: float,
        param_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        """Create a perturbed copy of parameters.

        Args:
            params: Original parameters.
            numeric_params: Numeric parameters to perturb.
            perturbation_pct: Maximum perturbation percentage.
            param_bounds: Optional bounds.

        Returns:
            Perturbed parameter dictionary.
        """
        perturbed = dict(params)

        for key, value in numeric_params.items():
            # Random perturbation within +/- perturbation_pct
            delta = value * perturbation_pct / 100.0
            new_value = value + np.random.uniform(-delta, delta)

            # Apply bounds if provided
            if param_bounds and key in param_bounds:
                lb, ub = param_bounds[key]
                new_value = max(lb, min(ub, new_value))
            else:
                # Default: keep positive for most trading params
                if value > 0:
                    new_value = max(0.001, new_value)

            perturbed[key] = new_value

        return perturbed

    async def _evaluate(
        self,
        params: dict[str, Any],
        data: Any,
        backtest_fn: Callable,
    ) -> dict[str, Any]:
        """Evaluate parameters using the backtest function.

        Args:
            params: Parameters to evaluate.
            data: Historical data.
            backtest_fn: Backtest function.

        Returns:
            Result dictionary.
        """
        try:
            if asyncio.iscoroutinefunction(backtest_fn):
                result = await backtest_fn(params, data)
            else:
                result = backtest_fn(params, data)

            return result if isinstance(result, dict) else {"sharpe_ratio": 0.0}
        except Exception as exc:
            logger.warning("Backtest evaluation failed: %s", exc)
            return {"sharpe_ratio": 0.0}
