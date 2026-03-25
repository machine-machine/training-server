"""
Trading-aware loss functions for metric optimization.

These loss functions optimize for actual trading metrics (Sharpe, Sortino, CVaR)
instead of generic ML losses (MSE, cross-entropy).

Usage:
    from coldpath.training.trading_losses import SharpeLoss, SortinoLoss, CVaRLoss

    # In your training loop
    loss_fn = SharpeLoss(risk_free_rate=0.02)
    loss = loss_fn(returns)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    pass


class TradingLossNotAvailableError(ImportError):
    """Raised when trying to use PyTorch loss without PyTorch installed."""

    pass


def _make_unavailable_class(name: str):
    """Create a class that raises an error when instantiated without PyTorch."""

    class UnavailableClass:
        def __init__(self, *args, **kwargs):
            raise TradingLossNotAvailableError(
                f"{name} requires PyTorch. Install with: pip install torch. "
                f"Alternatively, use get_trading_loss('{name.lower()}') for NumPy fallback."
            )

    UnavailableClass.__name__ = name
    return UnavailableClass


@dataclass
class TradingMetrics:
    """Container for computed trading metrics."""

    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    max_drawdown: float
    cvar_05: float  # CVaR at 5% level
    cvar_01: float  # CVaR at 1% level
    win_rate: float
    profit_factor: float
    kelly_fraction: float = 0.0
    tail_ratio: float = 0.0


def compute_trading_metrics(returns: np.ndarray, risk_free_rate: float = 0.02) -> TradingMetrics:
    """Compute all trading metrics from return series.

    Args:
        returns: Array of period returns (as decimals, e.g., 0.05 for 5%)
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        TradingMetrics with all computed values
    """
    if len(returns) == 0:
        return TradingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    rf_period = risk_free_rate / 252  # Daily risk-free rate

    # Sharpe Ratio
    excess_returns = returns - rf_period
    sharpe = excess_returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)

    # Sortino Ratio (downside deviation only)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-8
    sortino = excess_returns.mean() / (downside_std + 1e-8) * np.sqrt(252)

    # Cumulative returns for drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdowns.min())

    # Calmar Ratio
    annual_return = np.prod(1 + returns) ** (252 / len(returns)) - 1
    calmar = annual_return / (max_drawdown + 1e-8)

    # Omega Ratio (gains vs losses at threshold)
    threshold = rf_period
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    omega = (gains.sum() + 1e-8) / (losses.sum() + 1e-8)

    # CVaR (Conditional VaR / Expected Shortfall)
    sorted_returns = np.sort(returns)
    var_05_idx = int(len(returns) * 0.05)
    var_01_idx = int(len(returns) * 0.01)
    cvar_05 = sorted_returns[: max(1, var_05_idx)].mean()
    cvar_01 = sorted_returns[: max(1, var_01_idx)].mean()

    # Win Rate
    win_rate = (returns > 0).sum() / len(returns)

    # Profit Factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = (gross_profit + 1e-8) / (gross_loss + 1e-8)

    # Kelly Fraction: f* = (p * b - q) / b where b = win/loss ratio, p = win rate, q = 1 - p
    avg_win = gross_profit / max(1, (returns > 0).sum())
    avg_loss = gross_loss / max(1, (returns < 0).sum())
    b = avg_win / (avg_loss + 1e-8)
    kelly_fraction = max(0.0, (win_rate * b - (1 - win_rate)) / (b + 1e-8))
    kelly_fraction = min(kelly_fraction, 1.0)  # Cap at 100%

    # Tail Ratio: ratio of 95th percentile gain to 5th percentile loss
    p95_gain = np.percentile(returns, 95)
    p05_loss = abs(np.percentile(returns, 5))
    tail_ratio = (p95_gain + 1e-8) / (p05_loss + 1e-8)

    return TradingMetrics(
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        calmar_ratio=float(calmar),
        omega_ratio=float(omega),
        max_drawdown=float(max_drawdown),
        cvar_05=float(cvar_05),
        cvar_01=float(cvar_01),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        kelly_fraction=float(kelly_fraction),
        tail_ratio=float(tail_ratio),
    )


if TORCH_AVAILABLE:

    class TradingLossBase(nn.Module, ABC):
        """Base class for trading-aware loss functions."""

        def __init__(self, risk_free_rate: float = 0.02, annualization_factor: float = 252.0):
            super().__init__()
            self.rf_rate = risk_free_rate
            self.ann_factor = annualization_factor

        @abstractmethod
        def forward(self, returns: torch.Tensor) -> torch.Tensor:
            pass

    class SharpeLoss(TradingLossBase):
        """Differentiable Sharpe Ratio loss.

        Maximizes risk-adjusted returns. Returns negative Sharpe to minimize.

        Usage:
            loss_fn = SharpeLoss()
            loss = loss_fn(predicted_returns)  # Minimize this
        """

        def forward(self, returns: torch.Tensor) -> torch.Tensor:
            rf_period = self.rf_rate / self.ann_factor
            excess = returns - rf_period

            mean = excess.mean()
            std = returns.std() + 1e-8

            sharpe = mean / std * (self.ann_factor**0.5)
            return -sharpe  # Negative because we minimize loss

    class SortinoLoss(TradingLossBase):
        """Differentiable Sortino Ratio loss.

        Like Sharpe but only penalizes downside volatility.
        Better for asymmetric return distributions.
        """

        def forward(self, returns: torch.Tensor) -> torch.Tensor:
            rf_period = self.rf_rate / self.ann_factor
            excess = returns - rf_period

            mean = excess.mean()

            # Downside deviation
            downside = torch.clamp(returns, max=0)
            downside_std = downside.std() + 1e-8

            sortino = mean / downside_std * (self.ann_factor**0.5)
            return -sortino

    class CVaRLoss(TradingLossBase):
        """Conditional Value-at-Risk (Expected Shortfall) loss.

        Minimizes expected loss in the worst alpha% of cases.
        More sensitive to tail risk than Sharpe/Sortino.

        Args:
            alpha: Tail probability (default 0.05 = 5% worst cases)
        """

        def __init__(self, risk_free_rate: float = 0.02, alpha: float = 0.05):
            super().__init__(risk_free_rate)
            self.alpha = alpha

        def forward(self, returns: torch.Tensor) -> torch.Tensor:
            # Sort returns to find tail
            sorted_returns, _ = torch.sort(returns)
            n = len(returns)
            tail_size = max(1, int(n * self.alpha))

            # CVaR is mean of worst tail_size returns
            cvar = sorted_returns[:tail_size].mean()
            return -cvar  # Minimize loss = maximize returns in tail

    class CalmarLoss(TradingLossBase):
        """Calmar Ratio loss - return divided by maximum drawdown.

        Optimizes for returns while explicitly penalizing large drawdowns.
        Better for long-term capital preservation strategies.
        """

        def forward(self, returns: torch.Tensor) -> torch.Tensor:
            rf_period = self.rf_rate / self.ann_factor
            excess = returns - rf_period

            annual_return = excess.mean() * self.ann_factor

            cumulative = torch.cumprod(1 + returns, dim=0)
            running_max = torch.cummax(cumulative, dim=0)[0]
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = torch.abs(torch.min(drawdowns)) + 1e-8

            calmar = annual_return / max_drawdown
            return -calmar

    class OmegaLoss(TradingLossBase):
        """Omega Ratio loss - probability-weighted gains vs losses.

        Captures the entire return distribution shape, not just mean/variance.
        More sensitive to skewness and kurtosis than Sharpe.
        """

        def __init__(self, risk_free_rate: float = 0.02, threshold_pct: float = 0.0):
            super().__init__(risk_free_rate)
            self.threshold_pct = threshold_pct

        def forward(self, returns: torch.Tensor) -> torch.Tensor:
            rf_period = self.rf_rate / self.ann_factor
            threshold = rf_period + self.threshold_pct / 100.0

            gains = torch.relu(returns - threshold)
            losses = torch.relu(threshold - returns)

            omega = (torch.sum(gains) + 1e-8) / (torch.sum(losses) + 1e-8)
            return -omega

    class TailRiskLoss(nn.Module):
        """Penalizes asymmetric tail risk - worse for negative skew.

        Combines:
        - CVaR for tail loss measurement
        - Skewness penalty for asymmetric distributions
        - Kurtosis penalty for fat tails

        Args:
            cvar_weight: Weight for CVaR component (default 0.5)
            skew_weight: Weight for negative skewness penalty (default 0.3)
            kurt_weight: Weight for excess kurtosis penalty (default 0.2)
            alpha: CVaR tail probability (default 0.05)
        """

        def __init__(
            self,
            cvar_weight: float = 0.5,
            skew_weight: float = 0.3,
            kurt_weight: float = 0.2,
            alpha: float = 0.05,
        ):
            super().__init__()
            self.cvar_weight = cvar_weight
            self.skew_weight = skew_weight
            self.kurt_weight = kurt_weight
            self.alpha = alpha

        def forward(self, returns: torch.Tensor) -> torch.Tensor:
            sorted_returns, _ = torch.sort(returns)
            n = len(returns)
            tail_size = max(1, int(n * self.alpha))
            cvar = sorted_returns[:tail_size].mean()

            mean = returns.mean()
            std = returns.std() + 1e-8
            skew = ((returns - mean) ** 3).mean() / (std**3)
            kurt = ((returns - mean) ** 4).mean() / (std**4) - 3.0

            skew_penalty = torch.relu(-skew) * self.skew_weight
            kurt_penalty = torch.relu(kurt) * self.kurt_weight

            total = -cvar * self.cvar_weight + skew_penalty + kurt_penalty
            return total

    class CombinedTradingLoss(nn.Module):
        """Combines multiple trading losses with configurable weights.

        Args:
            sharpe_weight: Weight for Sharpe ratio (default 0.4)
            sortino_weight: Weight for Sortino ratio (default 0.3)
            cvar_weight: Weight for CVaR (default 0.3)
        """

        def __init__(
            self,
            sharpe_weight: float = 0.4,
            sortino_weight: float = 0.3,
            cvar_weight: float = 0.3,
            risk_free_rate: float = 0.02,
            cvar_alpha: float = 0.05,
        ):
            super().__init__()
            self.sharpe_weight = sharpe_weight
            self.sortino_weight = sortino_weight
            self.cvar_weight = cvar_weight

            self.sharpe_loss = SharpeLoss(risk_free_rate)
            self.sortino_loss = SortinoLoss(risk_free_rate)
            self.cvar_loss = CVaRLoss(risk_free_rate, cvar_alpha)

        def forward(self, returns: torch.Tensor) -> torch.Tensor:
            sharpe = self.sharpe_loss(returns) * self.sharpe_weight
            sortino = self.sortino_loss(returns) * self.sortino_weight
            cvar = self.cvar_loss(returns) * self.cvar_weight

            return sharpe + sortino + cvar

    class ProfitabilityLoss(nn.Module):
        """Loss function for trade profitability prediction.

        Combines:
        - Directional accuracy (did we predict up vs down correctly?)
        - Magnitude accuracy (how close was the predicted return?)
        - Risk penalty (penalize overconfident wrong predictions)
        - Asymmetric penalty (false positives cost more than false negatives)

        For trading: predicting BUY when the trade loses (false positive) is worse
        than missing a profitable trade (false negative), because:
        1. Capital is deployed and at risk
        2. Transaction costs are incurred
        3. Opportunity cost of not being in better trades
        """

        def __init__(
            self,
            direction_weight: float = 0.4,
            magnitude_weight: float = 0.25,
            risk_weight: float = 0.15,
            asymmetric_weight: float = 0.2,
            false_positive_multiplier: float = 2.0,
        ):
            super().__init__()
            self.direction_weight = direction_weight
            self.magnitude_weight = magnitude_weight
            self.risk_weight = risk_weight
            self.asymmetric_weight = asymmetric_weight
            self.fp_multiplier = false_positive_multiplier

        def forward(
            self,
            predicted_returns: torch.Tensor,
            actual_returns: torch.Tensor,
            confidence: torch.Tensor | None = None,
        ) -> torch.Tensor:
            pred_sign = torch.sign(predicted_returns)
            actual_sign = torch.sign(actual_returns)
            direction_correct = (pred_sign == actual_sign).float()
            direction_loss = 1.0 - direction_correct.mean()

            magnitude_loss = nn.functional.mse_loss(predicted_returns, actual_returns)

            if confidence is not None:
                wrong_direction = (pred_sign != actual_sign).float()
                risk_penalty = (wrong_direction * confidence).mean()
            else:
                risk_penalty = torch.tensor(0.0, device=predicted_returns.device)

            false_positive_mask = (pred_sign > 0) & (actual_sign <= 0)
            false_negative_mask = (pred_sign <= 0) & (actual_sign > 0)

            fp_loss = false_positive_mask.float().mean() * self.fp_multiplier
            fn_loss = false_negative_mask.float().mean()
            asymmetric_loss = fp_loss + fn_loss

            total = (
                self.direction_weight * direction_loss
                + self.magnitude_weight * magnitude_loss
                + self.risk_weight * risk_penalty
                + self.asymmetric_weight * asymmetric_loss
            )

            return total


# NumPy fallback implementations for non-PyTorch environments
class NumpyTradingLosses:
    """NumPy implementations of trading losses for environments without PyTorch."""

    @staticmethod
    def sharpe_loss(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Compute negative Sharpe ratio (to minimize)."""
        metrics = compute_trading_metrics(returns, risk_free_rate)
        return -metrics.sharpe_ratio

    @staticmethod
    def sortino_loss(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Compute negative Sortino ratio (to minimize)."""
        metrics = compute_trading_metrics(returns, risk_free_rate)
        return -metrics.sortino_ratio

    @staticmethod
    def cvar_loss(returns: np.ndarray, alpha: float = 0.05) -> float:
        """Compute negative CVaR (to minimize)."""
        sorted_returns = np.sort(returns)
        tail_size = max(1, int(len(returns) * alpha))
        cvar = sorted_returns[:tail_size].mean()
        return -cvar

    @staticmethod
    def calmar_loss(returns: np.ndarray) -> float:
        """Compute negative Calmar ratio (to minimize)."""
        metrics = compute_trading_metrics(returns)
        return -metrics.calmar_ratio

    @staticmethod
    def omega_loss(returns: np.ndarray, threshold_pct: float = 0.0) -> float:
        """Compute negative Omega ratio (to minimize)."""
        metrics = compute_trading_metrics(returns)
        return -metrics.omega_ratio

    @staticmethod
    def tail_risk_loss(
        returns: np.ndarray,
        cvar_weight: float = 0.5,
        skew_weight: float = 0.3,
        kurt_weight: float = 0.2,
        alpha: float = 0.05,
    ) -> float:
        """Compute tail risk loss combining CVaR, skewness, and kurtosis."""
        sorted_returns = np.sort(returns)
        tail_size = max(1, int(len(returns) * alpha))
        cvar = sorted_returns[:tail_size].mean()

        mean = returns.mean()
        std = returns.std() + 1e-8
        skew = ((returns - mean) ** 3).mean() / (std**3)
        kurt = ((returns - mean) ** 4).mean() / (std**4) - 3.0

        skew_penalty = max(0, -skew) * skew_weight
        kurt_penalty = max(0, kurt) * kurt_weight

        return -cvar * cvar_weight + skew_penalty + kurt_penalty

    @staticmethod
    def combined_loss(
        returns: np.ndarray,
        sharpe_weight: float = 0.4,
        sortino_weight: float = 0.3,
        cvar_weight: float = 0.3,
    ) -> float:
        """Combined trading loss."""
        sharpe = NumpyTradingLosses.sharpe_loss(returns)
        sortino = NumpyTradingLosses.sortino_loss(returns)
        cvar = NumpyTradingLosses.cvar_loss(returns)

        return sharpe_weight * sharpe + sortino_weight * sortino + cvar_weight * cvar


def get_trading_loss(loss_type: str = "combined", **kwargs):
    """Factory function to get trading loss.

    Args:
        loss_type: One of "sharpe", "sortino", "cvar", "calmar", "omega",
                   "tail_risk", "combined", "profitability"
        **kwargs: Additional arguments for the loss function

    Returns:
        Loss function instance
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, returning NumPy loss functions")
        return NumpyTradingLosses()

    loss_type = loss_type.lower()

    if loss_type == "sharpe":
        return SharpeLoss(**kwargs)
    elif loss_type == "sortino":
        return SortinoLoss(**kwargs)
    elif loss_type == "cvar":
        return CVaRLoss(**kwargs)
    elif loss_type == "calmar":
        return CalmarLoss(**kwargs)
    elif loss_type == "omega":
        return OmegaLoss(**kwargs)
    elif loss_type == "tail_risk":
        return TailRiskLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedTradingLoss(**kwargs)
    elif loss_type == "profitability":
        return ProfitabilityLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Export stub classes for direct import when PyTorch not available
if TORCH_AVAILABLE:
    pass  # Classes already defined above
else:
    # Create placeholder classes that raise informative errors
    SharpeLoss = _make_unavailable_class("SharpeLoss")
    SortinoLoss = _make_unavailable_class("SortinoLoss")
    CVaRLoss = _make_unavailable_class("CVaRLoss")
    CalmarLoss = _make_unavailable_class("CalmarLoss")
    OmegaLoss = _make_unavailable_class("OmegaLoss")
    TailRiskLoss = _make_unavailable_class("TailRiskLoss")
    CombinedTradingLoss = _make_unavailable_class("CombinedTradingLoss")
    ProfitabilityLoss = _make_unavailable_class("ProfitabilityLoss")
