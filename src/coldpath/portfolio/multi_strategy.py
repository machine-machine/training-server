"""
Multi-Strategy Portfolio Optimizer.

Implements portfolio optimization for multiple trading strategies:
- Correlation analyzer: Ledoit-Wolf shrinkage covariance
- Risk parity allocator: Equal risk contribution
- Hedging optimizer: OLS hedge ratios
- Diversification ratio tracking

Strategies: Momentum, MeanReversion, ML Ensemble, Announcement, Liquidity
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

try:
    from scipy.optimize import Bounds, minimize
    from scipy.stats import pearsonr  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Available strategy types."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ML_ENSEMBLE = "ml_ensemble"
    ANNOUNCEMENT = "announcement"
    LIQUIDITY = "liquidity"


class AllocationMethod(Enum):
    """Portfolio allocation methods."""

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    MAX_DIVERSIFICATION = "max_diversification"


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy."""

    name: str
    strategy_type: StrategyType
    returns: np.ndarray  # Daily returns
    sharpe_ratio: float
    volatility: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_return: float
    correlation_with_market: float = 0.0

    @classmethod
    def from_returns(
        cls,
        name: str,
        strategy_type: StrategyType,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> "StrategyMetrics":
        """Compute metrics from return series."""
        avg_return = np.mean(returns)
        volatility = np.std(returns) + 1e-10
        sharpe = (avg_return - risk_free_rate / 252) / volatility * np.sqrt(252)

        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / running_max - 1
        max_dd = np.min(drawdowns)

        # Win rate
        wins = np.sum(returns > 0)
        total = len(returns)
        win_rate = wins / total if total > 0 else 0

        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 10.0

        return cls(
            name=name,
            strategy_type=strategy_type,
            returns=returns,
            sharpe_ratio=sharpe,
            volatility=volatility,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_return=avg_return,
        )


@dataclass
class StrategyAllocation:
    """Allocation to a single strategy."""

    strategy_name: str
    strategy_type: StrategyType
    weight: float
    risk_contribution: float
    expected_return: float
    expected_risk: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "type": self.strategy_type.value,
            "weight": self.weight,
            "risk_contribution": self.risk_contribution,
            "expected_return": self.expected_return,
            "expected_risk": self.expected_risk,
        }


@dataclass
class AllocationResult:
    """Result of portfolio optimization."""

    allocations: list[StrategyAllocation]
    method: AllocationMethod
    portfolio_return: float
    portfolio_volatility: float
    portfolio_sharpe: float
    diversification_ratio: float
    correlation_matrix: np.ndarray
    covariance_matrix: np.ndarray
    risk_contributions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocations": [a.to_dict() for a in self.allocations],
            "method": self.method.value,
            "portfolio_return": self.portfolio_return,
            "portfolio_volatility": self.portfolio_volatility,
            "portfolio_sharpe": self.portfolio_sharpe,
            "diversification_ratio": self.diversification_ratio,
            "risk_contributions": self.risk_contributions,
        }

    def get_weights(self) -> dict[str, float]:
        """Get weights as dictionary."""
        return {a.strategy_name: a.weight for a in self.allocations}


@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimization."""

    # Constraints
    min_weight: float = 0.05  # Minimum 5% per strategy
    max_weight: float = 0.40  # Maximum 40% per strategy
    max_correlation: float = 0.70  # Max correlation to include

    # Risk parameters
    target_volatility: float = 0.20  # Annual target volatility
    risk_free_rate: float = 0.04  # Annual risk-free rate

    # Optimization settings
    default_method: AllocationMethod = AllocationMethod.RISK_PARITY
    use_shrinkage: bool = True  # Ledoit-Wolf shrinkage
    lookback_days: int = 90  # Days for covariance estimation

    # Rebalancing
    rebalance_threshold: float = 0.10  # Drift threshold for rebalancing


class CorrelationAnalyzer:
    """Analyze correlations between strategies.

    Uses Ledoit-Wolf shrinkage for robust covariance estimation.
    """

    def __init__(self, use_shrinkage: bool = True):
        self.use_shrinkage = use_shrinkage

    def compute_correlation_matrix(
        self,
        returns: np.ndarray,
    ) -> np.ndarray:
        """Compute correlation matrix from returns.

        Args:
            returns: Matrix of returns (n_samples, n_strategies)

        Returns:
            Correlation matrix
        """
        returns.shape[1]
        corr = np.corrcoef(returns.T)

        # Handle NaN
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        return corr

    def compute_covariance_matrix(
        self,
        returns: np.ndarray,
    ) -> np.ndarray:
        """Compute covariance matrix with optional shrinkage.

        Args:
            returns: Matrix of returns (n_samples, n_strategies)

        Returns:
            Covariance matrix
        """
        if self.use_shrinkage:
            return self._ledoit_wolf_shrinkage(returns)
        else:
            return np.cov(returns.T)

    def _ledoit_wolf_shrinkage(self, returns: np.ndarray) -> np.ndarray:
        """Ledoit-Wolf shrinkage covariance estimator.

        Shrinks toward scaled identity matrix for stability.
        """
        n_samples, n_assets = returns.shape

        # Sample covariance
        sample_cov = np.cov(returns.T, ddof=1)

        # Target: scaled identity
        trace = np.trace(sample_cov)
        mu = trace / n_assets
        target = mu * np.eye(n_assets)

        # Compute optimal shrinkage intensity
        delta = sample_cov - target
        delta_sq_sum = np.sum(delta**2)

        # Cross-sectional variance of sample covariance entries
        X2 = returns**2
        sample2 = X2.T @ X2 / n_samples
        pi_mat = sample2 - sample_cov**2

        pi = np.sum(pi_mat)
        gamma = delta_sq_sum

        # Shrinkage intensity
        kappa = (pi - gamma / 2) / n_samples
        shrinkage = max(0, min(1, kappa / gamma)) if gamma > 0 else 1

        # Shrunk covariance
        shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov

        return shrunk_cov

    def identify_redundant_strategies(
        self,
        correlation_matrix: np.ndarray,
        strategy_names: list[str],
        threshold: float = 0.85,
    ) -> list[tuple[str, str, float]]:
        """Identify highly correlated strategy pairs.

        Args:
            correlation_matrix: Correlation matrix
            strategy_names: Names of strategies
            threshold: Correlation threshold

        Returns:
            List of (strategy1, strategy2, correlation) tuples
        """
        redundant = []
        n = len(strategy_names)

        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(correlation_matrix[i, j])
                if corr > threshold:
                    redundant.append(
                        (
                            strategy_names[i],
                            strategy_names[j],
                            corr,
                        )
                    )

        return sorted(redundant, key=lambda x: -x[2])


class RiskParityAllocator:
    """Risk parity allocation algorithm.

    Allocates weights so each strategy contributes equally to portfolio risk.
    """

    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-8):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def allocate(
        self,
        covariance_matrix: np.ndarray,
        expected_returns: np.ndarray | None = None,
        budget: float = 1.0,
        target_risk_contributions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute risk parity weights.

        Args:
            covariance_matrix: Covariance matrix of returns
            expected_returns: Expected returns (optional)
            budget: Total weight budget (default 1.0)
            target_risk_contributions: Target risk contribution per asset

        Returns:
            Optimal weights array
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, using equal weight")
            n = covariance_matrix.shape[0]
            return np.ones(n) / n

        n_assets = covariance_matrix.shape[0]

        # Default: equal risk contribution
        if target_risk_contributions is None:
            target_risk_contributions = np.ones(n_assets) / n_assets

        # Initial weights
        w0 = np.ones(n_assets) / n_assets

        def objective(weights):
            """Minimize sum of squared differences from target risk contribution."""
            weights = np.maximum(weights, 1e-10)
            portfolio_vol = np.sqrt(weights @ covariance_matrix @ weights)

            # Marginal contribution to risk
            marginal_contrib = covariance_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol

            # Normalize to get percentage contribution
            risk_contrib_pct = risk_contrib / np.sum(risk_contrib)

            # Squared error from target
            return np.sum((risk_contrib_pct - target_risk_contributions) ** 2)

        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - budget}]

        # Bounds
        bounds = Bounds(0.001, 1.0)

        # Optimize
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )

        if result.success:
            return result.x / np.sum(result.x) * budget
        else:
            logger.warning("Risk parity optimization failed, using equal weight")
            return np.ones(n_assets) / n_assets * budget

    def compute_risk_contributions(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute risk contribution of each asset.

        Args:
            weights: Portfolio weights
            covariance_matrix: Covariance matrix

        Returns:
            Risk contribution percentages
        """
        portfolio_vol = np.sqrt(weights @ covariance_matrix @ weights)
        marginal_contrib = covariance_matrix @ weights
        risk_contrib = weights * marginal_contrib / (portfolio_vol + 1e-10)
        return risk_contrib / (np.sum(risk_contrib) + 1e-10)


class HedgeOptimizer:
    """Optimize hedge ratios using OLS regression."""

    def compute_hedge_ratio(
        self,
        strategy_returns: np.ndarray,
        hedge_returns: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute optimal hedge ratio using OLS.

        Args:
            strategy_returns: Returns of strategy to hedge
            hedge_returns: Returns of hedging instrument

        Returns:
            Tuple of (hedge_ratio, r_squared, residual_vol)
        """
        # OLS: strategy = alpha + beta * hedge + epsilon
        X = np.column_stack([np.ones(len(hedge_returns)), hedge_returns])
        y = strategy_returns

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            hedge_ratio = -beta[1]  # Negative for hedging

            # R-squared
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # Residual volatility
            residual_vol = np.std(y - y_pred)

            return hedge_ratio, r_squared, residual_vol
        except Exception as e:
            logger.warning(f"Hedge ratio calculation failed: {e}")
            return 0.0, 0.0, np.std(strategy_returns)


class MultiStrategyPortfolio:
    """Multi-strategy portfolio optimizer.

    Combines multiple trading strategies using various allocation methods
    to achieve optimal risk-adjusted returns.

    Example:
        portfolio = MultiStrategyPortfolio()
        portfolio.add_strategy("momentum", StrategyType.MOMENTUM, returns_mom)
        portfolio.add_strategy("ml", StrategyType.ML_ENSEMBLE, returns_ml)
        result = portfolio.optimize(method=AllocationMethod.RISK_PARITY)
    """

    def __init__(self, config: PortfolioConfig | None = None):
        self.config = config or PortfolioConfig()
        self.strategies: dict[str, StrategyMetrics] = {}

        self.correlation_analyzer = CorrelationAnalyzer(use_shrinkage=self.config.use_shrinkage)
        self.risk_parity = RiskParityAllocator()
        self.hedge_optimizer = HedgeOptimizer()

        self._last_allocation: AllocationResult | None = None

    def add_strategy(
        self,
        name: str,
        strategy_type: StrategyType,
        returns: np.ndarray,
    ):
        """Add a strategy to the portfolio.

        Args:
            name: Strategy name
            strategy_type: Type of strategy
            returns: Historical returns array
        """
        metrics = StrategyMetrics.from_returns(
            name=name,
            strategy_type=strategy_type,
            returns=returns,
            risk_free_rate=self.config.risk_free_rate,
        )
        self.strategies[name] = metrics
        logger.info(f"Added strategy {name}: Sharpe={metrics.sharpe_ratio:.2f}")

    def remove_strategy(self, name: str):
        """Remove a strategy from the portfolio."""
        if name in self.strategies:
            del self.strategies[name]

    def optimize(
        self,
        method: AllocationMethod | None = None,
    ) -> AllocationResult:
        """Optimize portfolio allocation.

        Args:
            method: Allocation method (default from config)

        Returns:
            AllocationResult with optimal weights
        """
        method = method or self.config.default_method

        if len(self.strategies) == 0:
            raise ValueError("No strategies added to portfolio")

        # Build returns matrix
        strategy_names = list(self.strategies.keys())
        min_len = min(len(s.returns) for s in self.strategies.values())
        returns_matrix = np.column_stack(
            [self.strategies[name].returns[-min_len:] for name in strategy_names]
        )

        # Compute covariance and correlation
        cov_matrix = self.correlation_analyzer.compute_covariance_matrix(returns_matrix)
        corr_matrix = self.correlation_analyzer.compute_correlation_matrix(returns_matrix)

        # Expected returns
        expected_returns = np.array([self.strategies[name].avg_return for name in strategy_names])

        # Optimize based on method
        if method == AllocationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight_allocation(len(strategy_names))
        elif method == AllocationMethod.RISK_PARITY:
            weights = self._risk_parity_allocation(cov_matrix)
        elif method == AllocationMethod.MIN_VARIANCE:
            weights = self._min_variance_allocation(cov_matrix)
        elif method == AllocationMethod.MAX_SHARPE:
            weights = self._max_sharpe_allocation(expected_returns, cov_matrix)
        elif method == AllocationMethod.MAX_DIVERSIFICATION:
            weights = self._max_diversification_allocation(cov_matrix)
        else:
            weights = self._equal_weight_allocation(len(strategy_names))

        # Apply constraints
        weights = self._apply_constraints(weights)

        # Calculate portfolio metrics
        portfolio_return = weights @ expected_returns * 252  # Annualized
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)
        portfolio_sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

        # Diversification ratio
        individual_vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)
        weighted_vol_sum = weights @ individual_vols
        div_ratio = weighted_vol_sum / portfolio_vol if portfolio_vol > 0 else 1.0

        # Risk contributions
        risk_contrib = self.risk_parity.compute_risk_contributions(weights, cov_matrix)
        risk_contributions = {name: float(risk_contrib[i]) for i, name in enumerate(strategy_names)}

        # Build allocations
        allocations = []
        for i, name in enumerate(strategy_names):
            alloc = StrategyAllocation(
                strategy_name=name,
                strategy_type=self.strategies[name].strategy_type,
                weight=float(weights[i]),
                risk_contribution=float(risk_contrib[i]),
                expected_return=float(expected_returns[i] * 252),
                expected_risk=float(individual_vols[i]),
            )
            allocations.append(alloc)

        result = AllocationResult(
            allocations=allocations,
            method=method,
            portfolio_return=portfolio_return,
            portfolio_volatility=portfolio_vol,
            portfolio_sharpe=portfolio_sharpe,
            diversification_ratio=div_ratio,
            correlation_matrix=corr_matrix,
            covariance_matrix=cov_matrix,
            risk_contributions=risk_contributions,
        )

        self._last_allocation = result
        return result

    def _equal_weight_allocation(self, n: int) -> np.ndarray:
        """Equal weight allocation."""
        return np.ones(n) / n

    def _risk_parity_allocation(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Risk parity allocation."""
        return self.risk_parity.allocate(cov_matrix)

    def _min_variance_allocation(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Minimum variance allocation."""
        if not SCIPY_AVAILABLE:
            return self._equal_weight_allocation(cov_matrix.shape[0])

        n = cov_matrix.shape[0]
        w0 = np.ones(n) / n

        def objective(w):
            return w @ cov_matrix @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = Bounds(0.0, 1.0)

        result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        return result.x if result.success else w0

    def _max_sharpe_allocation(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """Maximum Sharpe ratio allocation."""
        if not SCIPY_AVAILABLE:
            return self._equal_weight_allocation(cov_matrix.shape[0])

        n = cov_matrix.shape[0]
        w0 = np.ones(n) / n
        rf = self.config.risk_free_rate / 252  # Daily

        def neg_sharpe(w):
            ret = w @ expected_returns
            vol = np.sqrt(w @ cov_matrix @ w) + 1e-10
            return -(ret - rf) / vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = Bounds(0.0, 1.0)

        result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        return result.x if result.success else w0

    def _max_diversification_allocation(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Maximum diversification ratio allocation."""
        if not SCIPY_AVAILABLE:
            return self._equal_weight_allocation(cov_matrix.shape[0])

        n = cov_matrix.shape[0]
        w0 = np.ones(n) / n
        vols = np.sqrt(np.diag(cov_matrix))

        def neg_div_ratio(w):
            weighted_vol = w @ vols
            portfolio_vol = np.sqrt(w @ cov_matrix @ w) + 1e-10
            return -weighted_vol / portfolio_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = Bounds(0.0, 1.0)

        result = minimize(neg_div_ratio, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        return result.x if result.success else w0

    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply min/max weight constraints."""
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)
        return weights / np.sum(weights)

    def should_rebalance(self, current_weights: dict[str, float]) -> bool:
        """Check if portfolio should be rebalanced.

        Args:
            current_weights: Current portfolio weights

        Returns:
            True if rebalancing is recommended
        """
        if self._last_allocation is None:
            return True

        target_weights = self._last_allocation.get_weights()

        for name, current in current_weights.items():
            target = target_weights.get(name, 0)
            if abs(current - target) > self.config.rebalance_threshold:
                return True

        return False

    def get_summary(self) -> dict[str, Any]:
        """Get portfolio summary."""
        return {
            "n_strategies": len(self.strategies),
            "strategies": {
                name: {
                    "type": m.strategy_type.value,
                    "sharpe": m.sharpe_ratio,
                    "volatility": m.volatility,
                    "max_drawdown": m.max_drawdown,
                }
                for name, m in self.strategies.items()
            },
            "last_allocation": (self._last_allocation.to_dict() if self._last_allocation else None),
        }
