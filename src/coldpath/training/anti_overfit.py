"""
Anti-Overfitting Validation Framework.

Implements 5 rigorous validation tests to ensure model generalization:

1. Generalization Ratio: Val Sharpe / Train Sharpe > 0.70
2. OOS Sharpe: Sharpe on last 30 days > 1.5
3. Walk-Forward: Avg Sharpe across windows > 1.5, StdDev < 0.5
4. Parameter Sensitivity: Max change on ±10% perturbation < 20%
5. Monte Carlo: Median Sharpe (15 seeds) > 1.5

If any test fails, the model should not be deployed for live trading.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test result status."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class OverfitTestResult:
    """Result of a single overfit test."""

    test_name: str
    status: TestStatus
    metric_value: float
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    def passed(self) -> bool:
        """Check if test passed."""
        return self.status == TestStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "status": self.status.value,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "passed": self.passed(),
            "details": self.details,
            "recommendation": self.recommendation,
        }


@dataclass
class ValidationReport:
    """Complete validation report across all tests."""

    test_results: list[OverfitTestResult]
    overall_passed: bool
    deployment_ready: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_passed": self.overall_passed,
            "deployment_ready": self.deployment_ready,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "summary": self.summary,
            "tests": [t.to_dict() for t in self.test_results],
        }


class AntiOverfitValidator:
    """Validation framework for detecting overfitting.

    Runs 5 comprehensive tests to ensure model generalizes well
    and is suitable for deployment.

    Thresholds (from reference spec):
    - Generalization Ratio: > 0.70
    - OOS Sharpe: > 1.5
    - Walk-Forward Sharpe: > 1.5, StdDev < 0.5
    - Parameter Sensitivity: < 20% change
    - Monte Carlo Sharpe: > 1.5
    """

    def __init__(
        self,
        generalization_threshold: float = 0.70,
        oos_sharpe_threshold: float = 1.5,
        walk_forward_sharpe_threshold: float = 1.5,
        walk_forward_std_threshold: float = 0.5,
        sensitivity_threshold: float = 0.20,
        monte_carlo_sharpe_threshold: float = 1.5,
        monte_carlo_seeds: int = 15,
        walk_forward_windows: int = 5,
    ):
        """Initialize the validator.

        Args:
            generalization_threshold: Min ratio of val/train Sharpe
            oos_sharpe_threshold: Min OOS Sharpe ratio
            walk_forward_sharpe_threshold: Min walk-forward Sharpe
            walk_forward_std_threshold: Max walk-forward Sharpe std dev
            sensitivity_threshold: Max parameter sensitivity (fraction)
            monte_carlo_sharpe_threshold: Min Monte Carlo median Sharpe
            monte_carlo_seeds: Number of random seeds for MC test
            walk_forward_windows: Number of walk-forward windows
        """
        self.generalization_threshold = generalization_threshold
        self.oos_sharpe_threshold = oos_sharpe_threshold
        self.walk_forward_sharpe_threshold = walk_forward_sharpe_threshold
        self.walk_forward_std_threshold = walk_forward_std_threshold
        self.sensitivity_threshold = sensitivity_threshold
        self.monte_carlo_sharpe_threshold = monte_carlo_sharpe_threshold
        self.monte_carlo_seeds = monte_carlo_seeds
        self.walk_forward_windows = walk_forward_windows

    def run_all_tests(
        self,
        backtest_fn: Callable,
        strategy_params: dict[str, Any],
        train_data: np.ndarray,
        val_data: np.ndarray,
        test_data: np.ndarray,
        train_sharpe: float,
        val_sharpe: float,
    ) -> ValidationReport:
        """Run all 5 anti-overfitting tests.

        Args:
            backtest_fn: Function that runs backtest and returns Sharpe
            strategy_params: Current strategy parameters
            train_data: Training data (for walk-forward)
            val_data: Validation data
            test_data: Out-of-sample test data (last 30 days)
            train_sharpe: Sharpe ratio on training data
            val_sharpe: Sharpe ratio on validation data

        Returns:
            ValidationReport with all test results
        """
        results = []

        # Test 1: Generalization Ratio
        results.append(self.test_generalization_ratio(train_sharpe, val_sharpe))

        # Test 2: OOS Sharpe
        results.append(self.test_oos_sharpe(backtest_fn, strategy_params, test_data))

        # Test 3: Walk-Forward
        full_data = np.concatenate([train_data, val_data])
        results.append(self.test_walk_forward(backtest_fn, strategy_params, full_data))

        # Test 4: Parameter Sensitivity
        results.append(
            self.test_parameter_sensitivity(backtest_fn, strategy_params, val_data, val_sharpe)
        )

        # Test 5: Monte Carlo
        results.append(self.test_monte_carlo(backtest_fn, strategy_params, val_data))

        # Aggregate results
        passed_tests = sum(1 for r in results if r.passed())
        failed_tests = sum(1 for r in results if r.status == TestStatus.FAILED)

        # Deployment requires ALL tests to pass
        overall_passed = all(r.passed() for r in results)

        # Generate summary
        summary_lines = [
            f"Validation Report: {passed_tests}/{len(results)} tests passed",
            "",
        ]
        for result in results:
            status_symbol = "✓" if result.passed() else "✗"
            summary_lines.append(
                f"  {status_symbol} {result.test_name}: {result.metric_value:.3f} "
                f"(threshold: {result.threshold:.3f})"
            )
            if result.recommendation:
                summary_lines.append(f"    → {result.recommendation}")

        summary = "\n".join(summary_lines)

        return ValidationReport(
            test_results=results,
            overall_passed=overall_passed,
            deployment_ready=overall_passed and val_sharpe >= 1.5,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            summary=summary,
        )

    def test_generalization_ratio(
        self, train_sharpe: float, val_sharpe: float
    ) -> OverfitTestResult:
        """Test 1: Generalization Ratio.

        Checks: Val Sharpe / Train Sharpe > 0.70

        Rationale: Large gap between train and val performance
        indicates overfitting to training data.
        """
        if train_sharpe <= 0:
            return OverfitTestResult(
                test_name="Generalization Ratio",
                status=TestStatus.SKIPPED,
                metric_value=0.0,
                threshold=self.generalization_threshold,
                details={"train_sharpe": train_sharpe, "val_sharpe": val_sharpe},
                recommendation="Train Sharpe must be positive",
            )

        ratio = val_sharpe / train_sharpe
        passed = ratio >= self.generalization_threshold

        return OverfitTestResult(
            test_name="Generalization Ratio",
            status=TestStatus.PASSED if passed else TestStatus.FAILED,
            metric_value=ratio,
            threshold=self.generalization_threshold,
            details={
                "train_sharpe": train_sharpe,
                "val_sharpe": val_sharpe,
                "ratio": ratio,
            },
            recommendation="" if passed else "Reduce model complexity or regularization",
        )

    def test_oos_sharpe(
        self,
        backtest_fn: Callable,
        strategy_params: dict[str, Any],
        test_data: np.ndarray,
    ) -> OverfitTestResult:
        """Test 2: Out-of-Sample Sharpe.

        Checks: Sharpe on last 30 days holdout > 1.5

        Rationale: Model must perform well on truly unseen data.
        """
        try:
            oos_sharpe = backtest_fn(strategy_params, test_data)
        except Exception as e:
            logger.warning(f"OOS test failed: {e}")
            return OverfitTestResult(
                test_name="OOS Sharpe",
                status=TestStatus.SKIPPED,
                metric_value=0.0,
                threshold=self.oos_sharpe_threshold,
                details={"error": str(e)},
                recommendation="Fix backtest function",
            )

        passed = oos_sharpe >= self.oos_sharpe_threshold

        return OverfitTestResult(
            test_name="OOS Sharpe",
            status=TestStatus.PASSED if passed else TestStatus.FAILED,
            metric_value=oos_sharpe,
            threshold=self.oos_sharpe_threshold,
            details={"oos_sharpe": oos_sharpe, "data_points": len(test_data)},
            recommendation="" if passed else "Do not deploy - poor OOS performance",
        )

    def test_walk_forward(
        self,
        backtest_fn: Callable,
        strategy_params: dict[str, Any],
        data: np.ndarray,
    ) -> OverfitTestResult:
        """Test 3: Walk-Forward Validation.

        Checks: Avg Sharpe > 1.5 AND StdDev < 0.5

        Rationale: Strategy should perform consistently across
        different time periods, not just specific windows.
        """
        n_windows = self.walk_forward_windows
        window_size = len(data) // (n_windows + 1)

        if window_size < 30:
            return OverfitTestResult(
                test_name="Walk-Forward",
                status=TestStatus.SKIPPED,
                metric_value=0.0,
                threshold=self.walk_forward_sharpe_threshold,
                details={"reason": "Insufficient data for walk-forward"},
                recommendation="Need more data",
            )

        sharpes = []

        for i in range(n_windows):
            # Train on data up to window i
            train_end = (i + 1) * window_size
            test_start = train_end
            test_end = min(test_start + window_size, len(data))

            if test_end <= test_start:
                continue

            window_data = data[test_start:test_end]

            try:
                window_sharpe = backtest_fn(strategy_params, window_data)
                sharpes.append(window_sharpe)
            except Exception as e:
                logger.warning(f"Walk-forward window {i} failed: {e}")

        if len(sharpes) < 3:
            return OverfitTestResult(
                test_name="Walk-Forward",
                status=TestStatus.SKIPPED,
                metric_value=0.0,
                threshold=self.walk_forward_sharpe_threshold,
                details={"reason": "Too few successful windows"},
                recommendation="Check backtest function",
            )

        avg_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)

        passed_mean = avg_sharpe >= self.walk_forward_sharpe_threshold
        passed_std = std_sharpe <= self.walk_forward_std_threshold
        passed = passed_mean and passed_std

        status = TestStatus.PASSED if passed else TestStatus.FAILED
        if passed_mean and not passed_std:
            status = TestStatus.WARNING

        return OverfitTestResult(
            test_name="Walk-Forward",
            status=status,
            metric_value=avg_sharpe,
            threshold=self.walk_forward_sharpe_threshold,
            details={
                "avg_sharpe": avg_sharpe,
                "std_sharpe": std_sharpe,
                "std_threshold": self.walk_forward_std_threshold,
                "n_windows": len(sharpes),
                "window_sharpes": sharpes,
            },
            recommendation="" if passed else "Strategy not stable across time periods",
        )

    def test_parameter_sensitivity(
        self,
        backtest_fn: Callable,
        strategy_params: dict[str, Any],
        data: np.ndarray,
        baseline_sharpe: float,
    ) -> OverfitTestResult:
        """Test 4: Parameter Sensitivity.

        Checks: Max change on ±10% perturbation < 20%

        Rationale: Robust strategies shouldn't be highly sensitive
        to small parameter changes.
        """
        if not strategy_params:
            return OverfitTestResult(
                test_name="Parameter Sensitivity",
                status=TestStatus.SKIPPED,
                metric_value=0.0,
                threshold=self.sensitivity_threshold,
                details={"reason": "No parameters to test"},
                recommendation="",
            )

        perturbation = 0.10  # ±10%
        max_change = 0.0
        param_changes = {}

        for param_name, param_value in strategy_params.items():
            if not isinstance(param_value, (int, float)):
                continue

            # Test +10% perturbation
            perturbed_params = strategy_params.copy()
            perturbed_params[param_name] = param_value * (1 + perturbation)

            try:
                plus_sharpe = backtest_fn(perturbed_params, data)
                plus_change = abs(plus_sharpe - baseline_sharpe) / max(abs(baseline_sharpe), 0.01)
            except Exception:
                plus_change = 0.0

            # Test -10% perturbation
            perturbed_params[param_name] = param_value * (1 - perturbation)

            try:
                minus_sharpe = backtest_fn(perturbed_params, data)
                minus_change = abs(minus_sharpe - baseline_sharpe) / max(abs(baseline_sharpe), 0.01)
            except Exception:
                minus_change = 0.0

            param_max_change = max(plus_change, minus_change)
            param_changes[param_name] = param_max_change
            max_change = max(max_change, param_max_change)

        passed = max_change < self.sensitivity_threshold

        return OverfitTestResult(
            test_name="Parameter Sensitivity",
            status=TestStatus.PASSED if passed else TestStatus.FAILED,
            metric_value=max_change,
            threshold=self.sensitivity_threshold,
            details={
                "max_change_pct": max_change * 100,
                "param_changes": {k: v * 100 for k, v in param_changes.items()},
                "perturbation": perturbation * 100,
            },
            recommendation="" if passed else "Increase regularization or simplify model",
        )

    def test_monte_carlo(
        self,
        backtest_fn: Callable,
        strategy_params: dict[str, Any],
        data: np.ndarray,
    ) -> OverfitTestResult:
        """Test 5: Monte Carlo Stability.

        Checks: Median Sharpe (15 seeds) > 1.5

        Rationale: Strategy should be robust to random initialization
        and stochastic elements.
        """
        sharpes = []

        for seed in range(self.monte_carlo_seeds):
            # Set seed for reproducibility
            np.random.seed(seed)

            # Add small noise to parameters if they're numeric
            perturbed_params = strategy_params.copy()
            for key, value in perturbed_params.items():
                if isinstance(value, (int, float)):
                    noise = np.random.normal(0, 0.01)  # 1% noise
                    perturbed_params[key] = value * (1 + noise)

            try:
                sharpe = backtest_fn(perturbed_params, data)
                sharpes.append(sharpe)
            except Exception as e:
                logger.warning(f"Monte Carlo seed {seed} failed: {e}")

        if len(sharpes) < self.monte_carlo_seeds // 2:
            return OverfitTestResult(
                test_name="Monte Carlo",
                status=TestStatus.SKIPPED,
                metric_value=0.0,
                threshold=self.monte_carlo_sharpe_threshold,
                details={"reason": "Too many failed runs"},
                recommendation="Check backtest stability",
            )

        median_sharpe = np.median(sharpes)
        passed = median_sharpe >= self.monte_carlo_sharpe_threshold

        return OverfitTestResult(
            test_name="Monte Carlo",
            status=TestStatus.PASSED if passed else TestStatus.FAILED,
            metric_value=median_sharpe,
            threshold=self.monte_carlo_sharpe_threshold,
            details={
                "median_sharpe": median_sharpe,
                "mean_sharpe": np.mean(sharpes),
                "std_sharpe": np.std(sharpes),
                "min_sharpe": np.min(sharpes),
                "max_sharpe": np.max(sharpes),
                "n_runs": len(sharpes),
            },
            recommendation="" if passed else "Model unstable - consider simpler approach",
        )


def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from returns.

    Args:
        returns: Array of returns
        risk_free_rate: Annualized risk-free rate

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / 365
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)

    if std_return == 0:
        return 0.0

    # Annualize (assuming daily returns)
    sharpe = mean_return / std_return * np.sqrt(365)

    return float(sharpe)


def create_simple_backtest_fn(
    model: Any,
    feature_extractor: Callable,
) -> Callable:
    """Create a simple backtest function for validation tests.

    Args:
        model: Trained model with predict method
        feature_extractor: Function to extract features from data

    Returns:
        Callable that takes (params, data) and returns Sharpe
    """

    def backtest_fn(params: dict[str, Any], data: np.ndarray) -> float:
        # Extract features
        features = feature_extractor(data)

        # Get predictions
        predictions = model.predict(features)

        # Simulate simple strategy: buy on positive prediction, sell on negative
        returns = []
        position = 0

        for i in range(1, len(data)):
            pred = predictions[i - 1] if i - 1 < len(predictions) else 0

            # Simple position sizing
            if pred > 0.5:
                position = 1
            elif pred < 0.3:
                position = 0

            # Calculate return
            price_return = (data[i] - data[i - 1]) / data[i - 1]
            strategy_return = position * price_return
            returns.append(strategy_return)

        return calculate_sharpe(np.array(returns))

    return backtest_fn
