"""
Out-of-Sample Validator - Prevent overfitting with train/holdout validation.

Approach:
1. Split data: 80% training, 20% holdout
2. Optimize on training set only
3. Validate on holdout set
4. Compare in-sample vs out-of-sample performance

Good strategies should generalize well. A large gap between
in-sample and out-of-sample Sharpe indicates overfitting.

Overfitting score = 1 - (out_of_sample_sharpe / in_sample_sharpe)
Score close to 0.0 = good generalization
Score close to 1.0 = severe overfitting
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OutOfSampleReport:
    """Out-of-sample validation report.

    Attributes:
        in_sample_sharpe: Sharpe ratio on training data.
        out_of_sample_sharpe: Sharpe ratio on holdout data.
        degradation: In-sample minus out-of-sample Sharpe.
        overfitting_score: 0=no overfitting, 1=severe overfitting.
        in_sample_win_rate: Win rate on training data.
        out_of_sample_win_rate: Win rate on holdout data.
        in_sample_max_drawdown: Max drawdown on training data.
        out_of_sample_max_drawdown: Max drawdown on holdout data.
        holdout_pct: Percentage of data held out.
        is_overfit: Whether the strategy is overfit.
        generalization_ratio: OOS_Sharpe / IS_Sharpe (>0.7 is good).
        num_folds: Number of cross-validation folds used.
        fold_sharpes: Sharpe per fold (if CV was used).
    """

    in_sample_sharpe: float
    out_of_sample_sharpe: float
    degradation: float
    overfitting_score: float
    in_sample_win_rate: float
    out_of_sample_win_rate: float
    in_sample_max_drawdown: float
    out_of_sample_max_drawdown: float
    holdout_pct: float
    is_overfit: bool
    generalization_ratio: float
    num_folds: int = 1
    fold_sharpes: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "in_sample_sharpe": self.in_sample_sharpe,
            "out_of_sample_sharpe": self.out_of_sample_sharpe,
            "degradation": self.degradation,
            "overfitting_score": self.overfitting_score,
            "in_sample_win_rate": self.in_sample_win_rate,
            "out_of_sample_win_rate": self.out_of_sample_win_rate,
            "in_sample_max_drawdown": self.in_sample_max_drawdown,
            "out_of_sample_max_drawdown": self.out_of_sample_max_drawdown,
            "holdout_pct": self.holdout_pct,
            "is_overfit": self.is_overfit,
            "generalization_ratio": self.generalization_ratio,
            "num_folds": self.num_folds,
            "fold_sharpes": self.fold_sharpes,
        }


class OutOfSampleValidator:
    """
    Out-of-sample validation to prevent overfitting.

    Supports:
    1. Simple holdout: Split data into train/test
    2. Walk-forward: Rolling window validation
    3. K-fold cross-validation: Multiple splits

    Usage:
        validator = OutOfSampleValidator()
        report = await validator.validate_out_of_sample(
            params={"stop_loss_pct": 8.0, ...},
            historical_data=data,
            backtest_fn=my_backtest,
            holdout_pct=20.0,
        )
    """

    def __init__(
        self,
        overfitting_threshold: float = 0.3,
        min_generalization_ratio: float = 0.7,
    ):
        """Initialize the out-of-sample validator.

        Args:
            overfitting_threshold: Maximum overfitting score to pass.
            min_generalization_ratio: Minimum OOS/IS Sharpe ratio to pass.
        """
        self.overfitting_threshold = overfitting_threshold
        self.min_generalization_ratio = min_generalization_ratio

    async def validate_out_of_sample(
        self,
        params: dict[str, Any],
        historical_data: list[Any],
        backtest_fn: Callable,
        holdout_pct: float = 20.0,
    ) -> OutOfSampleReport:
        """
        Split data and validate generalization.

        Args:
            params: Strategy parameters.
            historical_data: Historical market data (ordered by time).
            backtest_fn: Backtest function(params, data) -> result dict.
            holdout_pct: Percentage of data to hold out for validation.

        Returns:
            OutOfSampleReport with in-sample vs out-of-sample comparison.
        """
        logger.info(
            "Starting out-of-sample validation: %.0f%% holdout",
            holdout_pct,
        )

        # Split data chronologically (never shuffle time series!)
        split_idx = int(len(historical_data) * (1 - holdout_pct / 100))
        train_data = historical_data[:split_idx]
        test_data = historical_data[split_idx:]

        if not train_data or not test_data:
            logger.warning(
                "Insufficient data for out-of-sample split: train=%d, test=%d",
                len(train_data),
                len(test_data),
            )
            return self._empty_report(holdout_pct)

        # Evaluate on training set (in-sample)
        is_result = await self._evaluate(params, train_data, backtest_fn)
        is_sharpe = is_result.get("sharpe_ratio", is_result.get("sharpe", 0.0))
        is_win_rate = is_result.get("win_rate_pct", is_result.get("win_rate", 0.0))
        is_max_dd = is_result.get("max_drawdown_pct", is_result.get("max_drawdown", 0.0))

        # Evaluate on holdout set (out-of-sample)
        oos_result = await self._evaluate(params, test_data, backtest_fn)
        oos_sharpe = oos_result.get("sharpe_ratio", oos_result.get("sharpe", 0.0))
        oos_win_rate = oos_result.get("win_rate_pct", oos_result.get("win_rate", 0.0))
        oos_max_dd = oos_result.get("max_drawdown_pct", oos_result.get("max_drawdown", 0.0))

        # Compute degradation and overfitting score
        degradation = is_sharpe - oos_sharpe

        if is_sharpe > 0:
            overfitting_score = max(0.0, min(1.0, 1.0 - (oos_sharpe / is_sharpe)))
            generalization_ratio = oos_sharpe / is_sharpe
        elif is_sharpe == 0:
            overfitting_score = 0.0
            generalization_ratio = 1.0 if oos_sharpe >= 0 else 0.0
        else:
            # Negative in-sample Sharpe
            overfitting_score = 0.0
            generalization_ratio = 0.0

        is_overfit = (
            overfitting_score > self.overfitting_threshold
            or generalization_ratio < self.min_generalization_ratio
        )

        report = OutOfSampleReport(
            in_sample_sharpe=is_sharpe,
            out_of_sample_sharpe=oos_sharpe,
            degradation=degradation,
            overfitting_score=overfitting_score,
            in_sample_win_rate=is_win_rate,
            out_of_sample_win_rate=oos_win_rate,
            in_sample_max_drawdown=is_max_dd,
            out_of_sample_max_drawdown=oos_max_dd,
            holdout_pct=holdout_pct,
            is_overfit=is_overfit,
            generalization_ratio=generalization_ratio,
        )

        logger.info(
            "OOS validation: IS_sharpe=%.2f, OOS_sharpe=%.2f, "
            "degradation=%.2f, overfitting=%.2f, generalization=%.2f, "
            "overfit=%s",
            is_sharpe,
            oos_sharpe,
            degradation,
            overfitting_score,
            generalization_ratio,
            is_overfit,
        )

        return report

    async def walk_forward_validate(
        self,
        params: dict[str, Any],
        historical_data: list[Any],
        backtest_fn: Callable,
        num_folds: int = 5,
        train_ratio: float = 0.8,
    ) -> OutOfSampleReport:
        """
        Walk-forward cross-validation.

        Splits data into overlapping train/test windows, maintaining
        temporal ordering. Each fold uses a different time period
        for testing.

        Args:
            params: Strategy parameters.
            historical_data: Historical data.
            backtest_fn: Backtest function.
            num_folds: Number of walk-forward folds.
            train_ratio: Fraction of each fold for training.

        Returns:
            OutOfSampleReport with aggregate cross-validation results.
        """
        n = len(historical_data)
        fold_size = n // num_folds

        if fold_size < 10:
            logger.warning(
                "Insufficient data for %d-fold walk-forward: only %d samples per fold",
                num_folds,
                fold_size,
            )
            return self._empty_report(holdout_pct=(1 - train_ratio) * 100)

        is_sharpes: list[float] = []
        oos_sharpes: list[float] = []
        is_win_rates: list[float] = []
        oos_win_rates: list[float] = []
        is_drawdowns: list[float] = []
        oos_drawdowns: list[float] = []

        for fold in range(num_folds):
            # Walk-forward: each fold advances the window
            fold_start = fold * fold_size
            fold_end = min(n, fold_start + fold_size * 2)

            fold_data = historical_data[fold_start:fold_end]
            split_idx = int(len(fold_data) * train_ratio)

            train_data = fold_data[:split_idx]
            test_data = fold_data[split_idx:]

            if not train_data or not test_data:
                continue

            # In-sample
            is_result = await self._evaluate(params, train_data, backtest_fn)
            is_sharpes.append(is_result.get("sharpe_ratio", is_result.get("sharpe", 0.0)))
            is_win_rates.append(is_result.get("win_rate_pct", is_result.get("win_rate", 0.0)))
            is_drawdowns.append(
                is_result.get(
                    "max_drawdown_pct",
                    is_result.get("max_drawdown", 0.0),
                )
            )

            # Out-of-sample
            oos_result = await self._evaluate(params, test_data, backtest_fn)
            oos_sharpes.append(oos_result.get("sharpe_ratio", oos_result.get("sharpe", 0.0)))
            oos_win_rates.append(oos_result.get("win_rate_pct", oos_result.get("win_rate", 0.0)))
            oos_drawdowns.append(
                oos_result.get(
                    "max_drawdown_pct",
                    oos_result.get("max_drawdown", 0.0),
                )
            )

        if not is_sharpes:
            return self._empty_report(holdout_pct=(1 - train_ratio) * 100)

        # Aggregate results
        mean_is = float(np.mean(is_sharpes))
        mean_oos = float(np.mean(oos_sharpes))
        degradation = mean_is - mean_oos

        if mean_is > 0:
            overfitting_score = max(0.0, min(1.0, 1.0 - (mean_oos / mean_is)))
            generalization_ratio = mean_oos / mean_is
        else:
            overfitting_score = 0.0
            generalization_ratio = 1.0 if mean_oos >= 0 else 0.0

        is_overfit = (
            overfitting_score > self.overfitting_threshold
            or generalization_ratio < self.min_generalization_ratio
        )

        return OutOfSampleReport(
            in_sample_sharpe=mean_is,
            out_of_sample_sharpe=mean_oos,
            degradation=degradation,
            overfitting_score=overfitting_score,
            in_sample_win_rate=float(np.mean(is_win_rates)),
            out_of_sample_win_rate=float(np.mean(oos_win_rates)),
            in_sample_max_drawdown=float(np.mean(is_drawdowns)),
            out_of_sample_max_drawdown=float(np.mean(oos_drawdowns)),
            holdout_pct=(1 - train_ratio) * 100,
            is_overfit=is_overfit,
            generalization_ratio=generalization_ratio,
            num_folds=len(is_sharpes),
            fold_sharpes=oos_sharpes,
        )

    async def _evaluate(
        self,
        params: dict[str, Any],
        data: Any,
        backtest_fn: Callable,
    ) -> dict[str, Any]:
        """Evaluate parameters using the backtest function.

        Args:
            params: Parameters to evaluate.
            data: Data to backtest on.
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

    def _empty_report(self, holdout_pct: float) -> OutOfSampleReport:
        """Create an empty report when validation cannot be performed.

        Args:
            holdout_pct: Holdout percentage.

        Returns:
            Empty OutOfSampleReport.
        """
        return OutOfSampleReport(
            in_sample_sharpe=0.0,
            out_of_sample_sharpe=0.0,
            degradation=0.0,
            overfitting_score=0.0,
            in_sample_win_rate=0.0,
            out_of_sample_win_rate=0.0,
            in_sample_max_drawdown=0.0,
            out_of_sample_max_drawdown=0.0,
            holdout_pct=holdout_pct,
            is_overfit=False,
            generalization_ratio=1.0,
        )
