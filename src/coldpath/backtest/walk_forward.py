"""
Walk-Forward Validator - Robust out-of-sample validation.

Implements walk-forward analysis to prevent overfitting:
- Train on historical window, test on next period
- Rolling window validation
- Anchored window validation
- Performance degradation analysis
- Parameter stability testing

This provides a more realistic estimate of live trading performance
than single backtest runs.

Usage:
    from coldpath.backtest.walk_forward import WalkForwardValidator

    validator = WalkForwardValidator(
        backtest_runner=my_backtest_fn,
        optimizer=my_optimize_fn,
    )

    result = await validator.validate(
        data=data,
        train_window_days=30,
        test_window_days=7,
        step_days=7,
    )

    print(f"In-sample Sharpe: {result.avg_in_sample_sharpe:.2f}")
    print(f"Out-of-sample Sharpe: {result.avg_out_of_sample_sharpe:.2f}")
    print(f"Degradation: {result.degradation_pct:.1f}%")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Walk-forward validation mode."""

    ROLLING = "rolling"  # Fixed-size rolling window
    ANCHORED = "anchored"  # Expanding window anchored at start


@dataclass
class FoldResult:
    """Result from a single train/test fold."""

    fold_index: int

    # Time periods
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # Parameters (required)
    optimized_params: dict[str, Any]

    # Metrics (required)
    in_sample_metrics: dict[str, float]
    out_of_sample_metrics: dict[str, float]

    # In-sample metrics (training period)
    in_sample_sharpe: float = 0.0
    in_sample_return: float = 0.0

    # Out-of-sample metrics (test period)
    out_of_sample_sharpe: float = 0.0
    out_of_sample_return: float = 0.0

    # Degradation
    sharpe_degradation: float = 0.0  # Negative = worse OOS
    return_degradation: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "train_period": f"{self.train_start} to {self.train_end}",
            "test_period": f"{self.test_start} to {self.test_end}",
            "optimized_params": self.optimized_params,
            "in_sample_sharpe": self.in_sample_sharpe,
            "out_of_sample_sharpe": self.out_of_sample_sharpe,
            "sharpe_degradation": self.sharpe_degradation,
        }


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation result."""

    # Configuration
    mode: ValidationMode
    train_window_days: int
    test_window_days: int
    step_days: int
    total_folds: int

    # Fold results
    folds: list[FoldResult]

    # Aggregated metrics
    avg_in_sample_sharpe: float = 0.0
    avg_out_of_sample_sharpe: float = 0.0
    avg_in_sample_return: float = 0.0
    avg_out_of_sample_return: float = 0.0
    avg_sharpe_degradation: float = 0.0

    # Performance summary
    degradation_pct: float = 0.0  # OOS as % of in-sample
    consistency_score: float = 0.0  # How consistent are OOS results
    stability_score: float = 0.0  # Parameter stability across folds

    # Best and worst folds
    best_fold: int | None = None
    worst_fold: int | None = None

    # Overfitting indicators
    overfitting_risk: str = "low"  # low, medium, high
    recommended_for_live: bool = True

    # Recommended parameters (median or best OOS)
    recommended_params: dict[str, Any] = field(default_factory=dict)

    # Timing
    time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "total_folds": self.total_folds,
            "avg_in_sample_sharpe": self.avg_in_sample_sharpe,
            "avg_out_of_sample_sharpe": self.avg_out_of_sample_sharpe,
            "degradation_pct": self.degradation_pct,
            "consistency_score": self.consistency_score,
            "stability_score": self.stability_score,
            "overfitting_risk": self.overfitting_risk,
            "recommended_for_live": self.recommended_for_live,
            "recommended_params": self.recommended_params,
            "folds": [f.to_dict() for f in self.folds],
        }


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    # Window sizes
    train_window_days: int = 30
    test_window_days: int = 7
    step_days: int = 7
    min_train_days: int = 14  # Minimum training period

    # Validation mode
    mode: ValidationMode = ValidationMode.ROLLING

    # Performance thresholds
    max_degradation_pct: float = 30.0  # Max acceptable degradation
    min_oos_sharpe: float = 0.5  # Minimum OOS Sharpe required
    consistency_threshold: float = 0.6  # Min consistency score

    # Parameter selection
    parameter_selection: str = "median"  # median, best_oos, vote

    # Parallelization
    parallel_folds: int = 1  # Folds to run in parallel

    # Logging
    log_progress: bool = True


class WalkForwardValidator:
    """Walk-forward validation for robust strategy testing.

    Provides rigorous out-of-sample testing by:
    - Training parameters on historical data
    - Testing on unseen future data
    - Rolling through time periods
    - Measuring performance degradation

    This is the gold standard for validating trading strategies
    and detecting overfitting before live deployment.

    Example:
        validator = WalkForwardValidator(
            backtest_runner=my_backtest,
            optimizer=my_optimizer,
            config=WalkForwardConfig(train_window_days=30, test_window_days=7),
        )

        result = await validator.validate(
            start_date="2024-01-01",
            end_date="2024-06-30",
            data=my_historical_data,
        )

        if result.recommended_for_live:
            print(f"Strategy validated! OOS Sharpe: {result.avg_out_of_sample_sharpe:.2f}")
            print(f"Use params: {result.recommended_params}")
        else:
            print(f"Warning: Overfitting detected ({result.overfitting_risk})")
    """

    def __init__(
        self,
        backtest_runner: Callable,
        optimizer: Callable | None = None,
        config: WalkForwardConfig | None = None,
    ):
        """Initialize the walk-forward validator.

        Args:
            backtest_runner: Async function to run backtests
            optimizer: Optional async function to optimize params on training data
            config: Validation configuration
        """
        self.backtest_runner = backtest_runner
        self.optimizer = optimizer
        self.config = config or WalkForwardConfig()

    async def validate(
        self,
        start_date: str,
        end_date: str,
        initial_params: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> WalkForwardResult:
        """Run walk-forward validation.

        Args:
            start_date: Start date for validation (YYYY-MM-DD)
            end_date: End date for validation (YYYY-MM-DD)
            initial_params: Starting parameters (used if no optimizer)
            data: Historical data for backtesting

        Returns:
            WalkForwardResult with validation metrics
        """
        start_time = datetime.utcnow()

        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Generate folds
        folds = self._generate_folds(start_dt, end_dt)

        if not folds:
            logger.error("No valid folds generated")
            return WalkForwardResult(
                mode=self.config.mode,
                train_window_days=self.config.train_window_days,
                test_window_days=self.config.test_window_days,
                step_days=self.config.step_days,
                total_folds=0,
                folds=[],
                time_seconds=0.0,
            )

        logger.info(f"Starting walk-forward validation: {len(folds)} folds")

        # Run each fold
        fold_results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            if self.config.log_progress:
                logger.info(f"Running fold {i + 1}/{len(folds)}")

            result = await self._run_fold(
                fold_index=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                initial_params=initial_params,
                data=data,
            )

            fold_results.append(result)

            if self.config.log_progress:
                logger.info(
                    f"Fold {i + 1}: IS Sharpe={result.in_sample_sharpe:.2f}, "
                    f"OOS Sharpe={result.out_of_sample_sharpe:.2f}, "
                    f"Degradation={result.sharpe_degradation:.1f}%"
                )

        # Calculate aggregated metrics
        avg_is_sharpe = sum(f.in_sample_sharpe for f in fold_results) / len(fold_results)
        avg_oos_sharpe = sum(f.out_of_sample_sharpe for f in fold_results) / len(fold_results)
        avg_is_return = sum(f.in_sample_return for f in fold_results) / len(fold_results)
        avg_oos_return = sum(f.out_of_sample_return for f in fold_results) / len(fold_results)

        # Calculate degradation
        if avg_is_sharpe > 0:
            degradation_pct = (1 - avg_oos_sharpe / avg_is_sharpe) * 100
        else:
            degradation_pct = 100.0

        # Calculate consistency (what % of folds have positive OOS)
        positive_oos = sum(1 for f in fold_results if f.out_of_sample_sharpe > 0)
        consistency = positive_oos / len(fold_results)

        # Calculate parameter stability
        stability = self._calculate_stability(fold_results)

        # Find best and worst folds
        best_fold_idx = max(
            range(len(fold_results)),
            key=lambda i: fold_results[i].out_of_sample_sharpe,
        )
        worst_fold_idx = min(
            range(len(fold_results)),
            key=lambda i: fold_results[i].out_of_sample_sharpe,
        )

        # Assess overfitting risk
        overfitting_risk = self._assess_overfitting(
            degradation_pct=degradation_pct,
            consistency=consistency,
            stability=stability,
            avg_oos_sharpe=avg_oos_sharpe,
        )

        # Determine if recommended for live
        recommended = (
            degradation_pct <= self.config.max_degradation_pct
            and avg_oos_sharpe >= self.config.min_oos_sharpe
            and consistency >= self.config.consistency_threshold
            and overfitting_risk != "high"
        )

        # Get recommended parameters
        recommended_params = self._get_recommended_params(fold_results)

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        return WalkForwardResult(
            mode=self.config.mode,
            train_window_days=self.config.train_window_days,
            test_window_days=self.config.test_window_days,
            step_days=self.config.step_days,
            total_folds=len(fold_results),
            folds=fold_results,
            avg_in_sample_sharpe=avg_is_sharpe,
            avg_out_of_sample_sharpe=avg_oos_sharpe,
            avg_in_sample_return=avg_is_return,
            avg_out_of_sample_return=avg_oos_return,
            avg_sharpe_degradation=sum(f.sharpe_degradation for f in fold_results)
            / len(fold_results),
            degradation_pct=degradation_pct,
            consistency_score=consistency,
            stability_score=stability,
            best_fold=best_fold_idx,
            worst_fold=worst_fold_idx,
            overfitting_risk=overfitting_risk,
            recommended_for_live=recommended,
            recommended_params=recommended_params,
            time_seconds=elapsed,
        )

    def _generate_folds(
        self,
        start: datetime,
        end: datetime,
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """Generate train/test fold boundaries."""
        folds = []

        train_days = timedelta(days=self.config.train_window_days)
        test_days = timedelta(days=self.config.test_window_days)
        step = timedelta(days=self.config.step_days)

        if self.config.mode == ValidationMode.ROLLING:
            # Rolling window: fixed-size training window moves forward
            current_test_start = start + train_days

            while current_test_start + test_days <= end:
                train_start = current_test_start - train_days
                train_end = current_test_start
                test_start = current_test_start
                test_end = current_test_start + test_days

                folds.append((train_start, train_end, test_start, test_end))
                current_test_start += step

        elif self.config.mode == ValidationMode.ANCHORED:
            # Anchored: training window expands from anchor point
            train_start = start
            current_test_start = start + timedelta(days=self.config.min_train_days)

            while current_test_start + test_days <= end:
                train_end = current_test_start
                test_start = current_test_start
                test_end = current_test_start + test_days

                folds.append((train_start, train_end, test_start, test_end))
                current_test_start += step

        return folds

    async def _run_fold(
        self,
        fold_index: int,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
        initial_params: dict[str, Any] | None,
        data: Any | None,
    ) -> FoldResult:
        """Run a single train/test fold."""
        # Optimize parameters on training data
        if self.optimizer:
            optimized_params = await self.optimizer(
                params=initial_params or {},
                start_date=train_start.strftime("%Y-%m-%d"),
                end_date=train_end.strftime("%Y-%m-%d"),
                data=data,
            )
        else:
            optimized_params = initial_params or {}

        # Run in-sample backtest
        is_metrics = await self._run_backtest(
            params=optimized_params,
            start_date=train_start,
            end_date=train_end,
            data=data,
        )

        # Run out-of-sample backtest
        oos_metrics = await self._run_backtest(
            params=optimized_params,
            start_date=test_start,
            end_date=test_end,
            data=data,
        )

        # Extract key metrics
        is_sharpe = is_metrics.get("sharpe_ratio", 0)
        oos_sharpe = oos_metrics.get("sharpe_ratio", 0)
        is_return = is_metrics.get("total_return_pct", 0)
        oos_return = oos_metrics.get("total_return_pct", 0)

        # Calculate degradation
        if is_sharpe > 0:
            sharpe_degradation = (1 - oos_sharpe / is_sharpe) * 100
        else:
            sharpe_degradation = 100.0 if oos_sharpe < 0 else 0.0

        if is_return != 0:
            return_degradation = (1 - oos_return / is_return) * 100
        else:
            return_degradation = 0.0

        return FoldResult(
            fold_index=fold_index,
            train_start=train_start.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
            optimized_params=optimized_params,
            in_sample_metrics=is_metrics,
            in_sample_sharpe=is_sharpe,
            in_sample_return=is_return,
            out_of_sample_metrics=oos_metrics,
            out_of_sample_sharpe=oos_sharpe,
            out_of_sample_return=oos_return,
            sharpe_degradation=sharpe_degradation,
            return_degradation=return_degradation,
        )

    async def _run_backtest(
        self,
        params: dict[str, Any],
        start_date: datetime,
        end_date: datetime,
        data: Any | None,
    ) -> dict[str, float]:
        """Run a single backtest."""
        try:
            result = await self.backtest_runner(
                params=params,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                data=data,
            )
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            return {"sharpe_ratio": 0, "total_return_pct": 0}

    def _calculate_stability(self, folds: list[FoldResult]) -> float:
        """Calculate parameter stability across folds."""
        if len(folds) < 2:
            return 1.0

        # Get all parameters
        all_params = [f.optimized_params for f in folds]

        # Calculate coefficient of variation for numeric params
        stability_scores = []

        # Find common numeric params
        if not all_params or not all_params[0]:
            return 1.0

        param_names = set(all_params[0].keys())
        for params in all_params[1:]:
            param_names &= set(params.keys())

        for param in param_names:
            values = [p.get(param) for p in all_params if isinstance(p.get(param), (int, float))]

            if len(values) < 2:
                continue

            mean_val = sum(values) / len(values)
            if mean_val == 0:
                continue

            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            std_dev = variance**0.5
            cv = std_dev / abs(mean_val)  # Coefficient of variation

            # Lower CV = higher stability (invert to get stability score)
            stability = 1.0 / (1.0 + cv)
            stability_scores.append(stability)

        return sum(stability_scores) / len(stability_scores) if stability_scores else 1.0

    def _assess_overfitting(
        self,
        degradation_pct: float,
        consistency: float,
        stability: float,
        avg_oos_sharpe: float,
    ) -> str:
        """Assess overfitting risk level."""
        risk_score = 0

        # High degradation is bad
        if degradation_pct > 50:
            risk_score += 3
        elif degradation_pct > 30:
            risk_score += 2
        elif degradation_pct > 15:
            risk_score += 1

        # Low consistency is bad
        if consistency < 0.4:
            risk_score += 2
        elif consistency < 0.6:
            risk_score += 1

        # Low stability is bad
        if stability < 0.5:
            risk_score += 2
        elif stability < 0.7:
            risk_score += 1

        # Negative OOS is very bad
        if avg_oos_sharpe < 0:
            risk_score += 3

        if risk_score >= 5:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"

    def _get_recommended_params(self, folds: list[FoldResult]) -> dict[str, Any]:
        """Get recommended parameters based on fold results."""
        if not folds:
            return {}

        if self.config.parameter_selection == "median":
            # Use median of all optimized params
            return self._get_median_params(folds)

        elif self.config.parameter_selection == "best_oos":
            # Use params from best OOS fold
            best_fold = max(folds, key=lambda f: f.out_of_sample_sharpe)
            return best_fold.optimized_params

        elif self.config.parameter_selection == "vote":
            # Use most common value for each param
            return self._get_voted_params(folds)

        else:
            return self._get_median_params(folds)

    def _get_median_params(self, folds: list[FoldResult]) -> dict[str, Any]:
        """Get median parameter values."""
        all_params = [f.optimized_params for f in folds]

        if not all_params or not all_params[0]:
            return {}

        param_names = set(all_params[0].keys())
        result = {}

        for param in param_names:
            values = [p.get(param) for p in all_params if param in p]

            if not values:
                continue

            # Sort and get median
            if isinstance(values[0], (int, float)):
                sorted_vals = sorted(values)
                mid = len(sorted_vals) // 2
                if len(sorted_vals) % 2 == 0:
                    result[param] = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
                else:
                    result[param] = sorted_vals[mid]
            else:
                # Non-numeric: use most common
                result[param] = max(set(values), key=values.count)

        return result

    def _get_voted_params(self, folds: list[FoldResult]) -> dict[str, Any]:
        """Get most common parameter values."""
        all_params = [f.optimized_params for f in folds]

        if not all_params or not all_params[0]:
            return {}

        param_names = set(all_params[0].keys())
        result = {}

        for param in param_names:
            values = [p.get(param) for p in all_params if param in p]

            if values:
                # Use most common value
                result[param] = max(set(values), key=values.count)

        return result


# Convenience function
async def walk_forward_validate(
    backtest_runner: Callable,
    start_date: str,
    end_date: str,
    initial_params: dict[str, Any],
    train_days: int = 30,
    test_days: int = 7,
    **kwargs,
) -> WalkForwardResult:
    """Run walk-forward validation with sensible defaults."""
    config = WalkForwardConfig(train_window_days=train_days, test_window_days=test_days, **kwargs)

    validator = WalkForwardValidator(
        backtest_runner=backtest_runner,
        config=config,
    )

    return await validator.validate(
        start_date=start_date,
        end_date=end_date,
        initial_params=initial_params,
    )
