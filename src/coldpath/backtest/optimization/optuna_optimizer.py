"""
Optuna-based hyperparameter optimization with walk-forward cross-validation.

Implements Bayesian optimization using TPE sampler with Hyperband pruning.
Targets 10,000+ parameter combinations per day for efficient exploration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np

try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .search_space import FULL_SEARCH_SPACE, SearchSpace

logger = logging.getLogger(__name__)


class ObjectiveMetric(Enum):
    """Optimization objective metrics."""

    SHARPE = "sharpe"  # Risk-adjusted return
    TOTAL_RETURN = "total_return"  # Raw return
    WIN_RATE = "win_rate"  # Percentage of winning trades
    PROFIT_FACTOR = "profit_factor"  # Gross profit / Gross loss
    PROFIT_QUALITY = "profit_quality"  # Expectancy + profit factor + drawdown-aware quality
    SORTINO = "sortino"  # Downside risk-adjusted return
    CALMAR = "calmar"  # Return / Max drawdown


@dataclass
class CVFold:
    """Single cross-validation fold with embargo."""

    train_start: int  # Timestamp or index
    train_end: int
    test_start: int
    test_end: int
    embargo_periods: int  # Gap between train and test to prevent leakage


@dataclass
class WalkForwardCV:
    """Walk-forward cross-validation with embargo periods.

    Prevents data leakage by:
    1. Training only on past data
    2. Adding embargo gap between train and test
    3. Never looking forward in time
    """

    n_splits: int = 5
    test_size: float = 0.2  # Fraction of data for testing
    embargo_pct: float = 0.02  # Embargo as fraction of train size (prevents leakage)
    min_train_size: float = 0.3  # Minimum training data fraction

    def split(
        self,
        n_samples: int,
        timestamps: np.ndarray | None = None,
    ) -> list[CVFold]:
        """Generate walk-forward CV folds.

        Args:
            n_samples: Total number of samples
            timestamps: Optional timestamp array for time-based splits

        Returns:
            List of CVFold objects
        """
        folds = []

        # Calculate sizes
        min_train = int(n_samples * self.min_train_size)
        test_size = int(n_samples * self.test_size / self.n_splits)

        # Ensure we have enough data
        if min_train + test_size * self.n_splits > n_samples:
            logger.warning(f"Insufficient data for {self.n_splits} folds, reducing splits")
            self.n_splits = max(2, (n_samples - min_train) // test_size)

        for i in range(self.n_splits):
            # Test window moves forward with each fold
            test_end = n_samples - (self.n_splits - 1 - i) * test_size
            test_start = test_end - test_size

            # Embargo period
            embargo_periods = int(test_size * self.embargo_pct)

            # Training ends before embargo
            train_end = test_start - embargo_periods
            train_start = 0

            # Skip if not enough training data
            if train_end - train_start < min_train:
                continue

            folds.append(
                CVFold(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    embargo_periods=embargo_periods,
                )
            )

        return folds


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""

    # Optimization settings
    n_trials: int = 500  # Number of optimization trials
    timeout_seconds: int | None = None  # Timeout for optimization
    n_jobs: int = 1  # Parallel jobs (-1 for all cores)

    # Objective
    objective_metric: ObjectiveMetric = ObjectiveMetric.PROFIT_QUALITY
    minimize: bool = False  # False = maximize

    # Cross-validation
    n_cv_folds: int = 5
    embargo_pct: float = 0.02  # Embargo as fraction of fold size

    # Early stopping
    enable_pruning: bool = True
    pruning_warmup_trials: int = 20  # Trials before pruning starts
    pruning_warmup_steps: int = 3  # CV folds before pruning

    # Sampler settings
    seed: int = 42
    n_startup_trials: int = 20  # Random trials before TPE

    # Hyperband settings (for multi-fidelity optimization)
    use_hyperband: bool = True
    hyperband_min_resource: int = 1
    hyperband_max_resource: int = 5  # Max CV folds
    hyperband_reduction_factor: int = 3

    # Logging
    log_level: str = "WARNING"
    show_progress_bar: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trials": self.n_trials,
            "timeout_seconds": self.timeout_seconds,
            "n_jobs": self.n_jobs,
            "objective_metric": self.objective_metric.value,
            "n_cv_folds": self.n_cv_folds,
            "embargo_pct": self.embargo_pct,
            "enable_pruning": self.enable_pruning,
            "use_hyperband": self.use_hyperband,
        }


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""

    best_params: dict[str, Any]
    best_value: float
    best_result: dict[str, Any]

    # Study statistics
    n_trials_completed: int = 0
    n_trials_pruned: int = 0
    n_trials_failed: int = 0

    # Timing
    optimization_time_seconds: float = 0.0
    trials_per_second: float = 0.0

    # History
    optimization_history: list[dict[str, Any]] = field(default_factory=list)

    # Parameter importance
    param_importance: dict[str, float] = field(default_factory=dict)

    # CV results for best params
    cv_scores: list[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "best_result": self.best_result,
            "n_trials_completed": self.n_trials_completed,
            "n_trials_pruned": self.n_trials_pruned,
            "optimization_time_seconds": self.optimization_time_seconds,
            "trials_per_second": self.trials_per_second,
            "param_importance": self.param_importance,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class OptunaOptimizer:
    """Bayesian hyperparameter optimization for trading strategies.

    Uses Optuna with TPE sampler and Hyperband pruning for efficient
    exploration of high-dimensional parameter spaces.

    Features:
    - Walk-forward cross-validation with embargo periods
    - Multi-fidelity optimization (Hyperband)
    - Automatic pruning of unpromising trials
    - Parameter importance analysis
    - Resume capability

    Example:
        optimizer = OptunaOptimizer(search_space=FULL_SEARCH_SPACE)
        result = optimizer.optimize(
            objective_fn=run_backtest,
            data=market_data,
            config=OptimizationConfig(n_trials=500),
        )
        print(f"Best params: {result.best_params}")
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        study_name: str | None = None,
        storage: str | None = None,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for optimization. Install with: pip install optuna"
            )

        self.search_space = search_space or FULL_SEARCH_SPACE
        self.study_name = study_name or f"trading_opt_{datetime.now():%Y%m%d_%H%M%S}"
        self.storage = storage  # SQLite, PostgreSQL, etc.

        self._study: optuna.Study | None = None
        self._config: OptimizationConfig | None = None
        self._cv: WalkForwardCV | None = None

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any], Any, CVFold], dict[str, float]],
        data: Any,
        config: OptimizationConfig | None = None,
        cv: WalkForwardCV | None = None,
    ) -> OptimizationResult:
        """Run hyperparameter optimization.

        Args:
            objective_fn: Function that takes (params, data, cv_fold) and returns
                          dict with metric values (e.g., {"sharpe": 1.5, "win_rate": 0.6})
            data: Market data to pass to objective function
            config: Optimization configuration
            cv: Walk-forward cross-validation object

        Returns:
            OptimizationResult with best parameters and statistics
        """
        self._config = config or OptimizationConfig()
        self._cv = cv or WalkForwardCV(
            n_splits=self._config.n_cv_folds,
            embargo_pct=self._config.embargo_pct,
        )

        # Set Optuna logging level
        optuna.logging.set_verbosity(getattr(optuna.logging, self._config.log_level))

        # Create sampler
        sampler = TPESampler(
            seed=self._config.seed,
            n_startup_trials=self._config.n_startup_trials,
            multivariate=True,  # Consider parameter dependencies
        )

        # Create pruner
        if self._config.enable_pruning:
            if self._config.use_hyperband:
                pruner = HyperbandPruner(
                    min_resource=self._config.hyperband_min_resource,
                    max_resource=self._config.hyperband_max_resource,
                    reduction_factor=self._config.hyperband_reduction_factor,
                )
            else:
                pruner = MedianPruner(
                    n_startup_trials=self._config.pruning_warmup_trials,
                    n_warmup_steps=self._config.pruning_warmup_steps,
                )
        else:
            pruner = optuna.pruners.NopPruner()

        # Create or load study
        direction = "minimize" if self._config.minimize else "maximize"
        self._study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=sampler,
            pruner=pruner,
            direction=direction,
            load_if_exists=True,
        )

        # Infer data length for CV
        n_samples = self._infer_data_length(data)
        cv_folds = self._cv.split(n_samples)

        logger.info(
            f"Starting optimization: {self._config.n_trials} trials, "
            f"{len(cv_folds)} CV folds, {len(self.search_space)} parameters"
        )

        start_time = datetime.now()

        # Create objective wrapper
        def objective(trial: optuna.Trial) -> float:
            return self._objective_wrapper(trial, objective_fn, data, cv_folds)

        # Run optimization
        self._study.optimize(
            objective,
            n_trials=self._config.n_trials,
            timeout=self._config.timeout_seconds,
            n_jobs=self._config.n_jobs,
            show_progress_bar=self._config.show_progress_bar,
            catch=(Exception,),  # Continue on errors
        )

        optimization_time = (datetime.now() - start_time).total_seconds()

        # Build result
        result = self._build_result(objective_fn, data, cv_folds, optimization_time)

        logger.info(
            f"Optimization complete: best {self._config.objective_metric.value}="
            f"{result.best_value:.4f}, {result.n_trials_completed} trials in "
            f"{optimization_time:.1f}s ({result.trials_per_second:.1f} trials/s)"
        )

        return result

    def _objective_wrapper(
        self,
        trial: optuna.Trial,
        objective_fn: Callable,
        data: Any,
        cv_folds: list[CVFold],
    ) -> float:
        """Wrapper for objective function with CV and pruning support."""
        # Suggest parameters
        params = self.search_space.suggest_all(trial)

        scores = []

        for fold_idx, fold in enumerate(cv_folds):
            try:
                # Run objective
                metrics = objective_fn(params, data, fold)

                # Extract target metric
                metric_value = metrics.get(
                    self._config.objective_metric.value, metrics.get("score", 0.0)
                )
                scores.append(metric_value)

                # Report for pruning
                trial.report(np.mean(scores), fold_idx)

                # Check if should prune
                if trial.should_prune():
                    raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.debug(f"Trial {trial.number} fold {fold_idx} failed: {e}")
                scores.append(float("-inf") if not self._config.minimize else float("inf"))

        # Return mean score across folds
        return (
            np.mean(scores)
            if scores
            else (float("inf") if self._config.minimize else float("-inf"))
        )

    def _build_result(
        self,
        objective_fn: Callable,
        data: Any,
        cv_folds: list[CVFold],
        optimization_time: float,
    ) -> OptimizationResult:
        """Build optimization result from study."""
        # Trial statistics
        trials = self._study.trials
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
        failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]

        # Best parameters
        best_params = self._study.best_params
        best_value = self._study.best_value

        # Run final evaluation with best params
        final_scores = []
        for fold in cv_folds:
            try:
                metrics = objective_fn(best_params, data, fold)
                final_scores.append(
                    metrics.get(self._config.objective_metric.value, metrics.get("score", 0.0))
                )
            except Exception:
                pass

        # Parameter importance (if enough trials)
        param_importance = {}
        if len(completed) >= 10:
            try:
                importance = optuna.importance.get_param_importances(self._study)
                param_importance = dict(importance)
            except Exception:
                pass

        # Build optimization history
        history = [
            {
                "trial": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            }
            for t in completed[:100]  # Limit history size
        ]

        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            best_result={
                "cv_scores": final_scores,
                "cv_mean": np.mean(final_scores) if final_scores else 0.0,
                "cv_std": np.std(final_scores) if final_scores else 0.0,
            },
            n_trials_completed=len(completed),
            n_trials_pruned=len(pruned),
            n_trials_failed=len(failed),
            optimization_time_seconds=optimization_time,
            trials_per_second=len(completed) / max(0.1, optimization_time),
            optimization_history=history,
            param_importance=param_importance,
            cv_scores=final_scores,
            cv_mean=np.mean(final_scores) if final_scores else 0.0,
            cv_std=np.std(final_scores) if final_scores else 0.0,
        )

    def _infer_data_length(self, data: Any) -> int:
        """Infer the number of samples from data."""
        if hasattr(data, "__len__"):
            return len(data)
        if hasattr(data, "n_timepoints"):
            return data.n_timepoints
        if hasattr(data, "shape"):
            return data.shape[-1] if len(data.shape) > 1 else data.shape[0]
        if hasattr(data, "timestamps"):
            return (
                len(data.timestamps[0])
                if hasattr(data.timestamps, "__getitem__")
                else len(data.timestamps)
            )

        # Default fallback
        logger.warning("Could not infer data length, using default 1000")
        return 1000

    def get_study(self) -> optuna.Study | None:
        """Get the underlying Optuna study."""
        return self._study

    def get_trial_dataframe(self):
        """Get trials as a DataFrame for analysis."""
        if self._study is None:
            return None
        return self._study.trials_dataframe()

    def plot_optimization_history(self):
        """Generate optimization history plot (if matplotlib available)."""
        if self._study is None:
            return None
        try:
            return optuna.visualization.matplotlib.plot_optimization_history(self._study)
        except ImportError:
            logger.warning("matplotlib required for plotting")
            return None

    def plot_param_importances(self):
        """Generate parameter importance plot."""
        if self._study is None:
            return None
        try:
            return optuna.visualization.matplotlib.plot_param_importances(self._study)
        except ImportError:
            logger.warning("matplotlib required for plotting")
            return None


def create_backtest_objective(
    backtester,
    config_class,
) -> Callable[[dict[str, Any], Any, CVFold], dict[str, float]]:
    """Create an objective function for backtesting.

    Factory function that creates an objective compatible with OptunaOptimizer.

    Args:
        backtester: VectorizedBacktester or similar
        config_class: SignalConfig or similar dataclass

    Returns:
        Objective function
    """

    def objective(
        params: dict[str, Any],
        data: Any,
        fold: CVFold,
    ) -> dict[str, float]:
        # Map params to config
        config_params = {
            k: v
            for k, v in params.items()
            if hasattr(config_class, k) or k in config_class.__dataclass_fields__
        }
        config = config_class(**config_params)

        # Slice data for fold
        if hasattr(data, "prices"):
            # VectorizedData
            type(data)(
                timestamps=data.timestamps[:, fold.train_start : fold.train_end],
                prices=data.prices[:, fold.train_start : fold.train_end],
                volumes=data.volumes[:, fold.train_start : fold.train_end],
                liquidities=data.liquidities[:, fold.train_start : fold.train_end],
                token_ids=data.token_ids,
            )
            test_data = type(data)(
                timestamps=data.timestamps[:, fold.test_start : fold.test_end],
                prices=data.prices[:, fold.test_start : fold.test_end],
                volumes=data.volumes[:, fold.test_start : fold.test_end],
                liquidities=data.liquidities[:, fold.test_start : fold.test_end],
                token_ids=data.token_ids,
            )
        else:
            # Generic slicing
            data[fold.train_start : fold.train_end]
            test_data = data[fold.test_start : fold.test_end]

        # Run backtest on test data (train could be used for fitting)
        result = backtester.run(test_data, config)

        # Calculate metrics
        metrics = {
            "total_return": result.total_pnl_pct,
            "win_rate": result.win_rate(),
            "n_trades": result.n_trades,
            "max_drawdown_pct": getattr(result, "max_drawdown_pct", 0.0),
            "profit_quality": getattr(result, "profit_quality_score", 0.0),
            "expectancy_pct": getattr(result, "expectancy_pct", 0.0),
            "compute_backend": getattr(result, "compute_backend", "cpu"),
        }

        # Calculate Sharpe ratio
        if result.n_trades > 0:
            avg_return = result.total_pnl_pct / result.n_trades
            # Estimate std from TP/SL
            estimated_std = (config.take_profit_pct + config.stop_loss_pct) / 2
            metrics["sharpe"] = avg_return / max(estimated_std, 1.0)

            metrics["profit_factor"] = getattr(result, "estimated_profit_factor", 0.0)
        else:
            metrics["sharpe"] = 0.0
            metrics["profit_factor"] = 0.0

        return metrics

    return objective
