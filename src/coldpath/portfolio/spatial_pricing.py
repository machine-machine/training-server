"""Spatial arbitrage pricing with factor + network interactions.

This module provides a pragmatic SAPT-style model:
- Latent factors capture common market movement.
- Spatial interaction matrix captures cross-asset dependence.
- Per-asset spatial rho summarizes neighbor sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SpatialPricingConfig:
    """Configuration for spatial pricing estimation."""

    n_factors: int = 3
    ridge_alpha: float = 0.05
    max_spatial_radius: float = 0.95
    min_observations: int = 30
    zero_threshold: float = 1e-4


@dataclass
class SpatialPricingResult:
    """Fitted artifacts and diagnostics for spatial pricing."""

    factor_loadings: np.ndarray
    factor_returns: np.ndarray
    interaction_matrix: np.ndarray
    spatial_rho: np.ndarray
    reconstruction_r2: float
    residual_volatility: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary to a dictionary."""
        return {
            "factor_loadings_shape": list(self.factor_loadings.shape),
            "factor_returns_shape": list(self.factor_returns.shape),
            "interaction_matrix_shape": list(self.interaction_matrix.shape),
            "spatial_rho": self.spatial_rho.tolist(),
            "reconstruction_r2": float(self.reconstruction_r2),
            "residual_volatility": self.residual_volatility.tolist(),
        }


class SpatialArbitragePricingModel:
    """Estimate a high-dimensional factor + spatial interaction model."""

    def __init__(self, config: SpatialPricingConfig | None = None):
        self.config = config or SpatialPricingConfig()

        self._fitted = False
        self._asset_means: np.ndarray | None = None
        self._factor_loadings: np.ndarray | None = None
        self._factor_returns: np.ndarray | None = None
        self._interaction_matrix: np.ndarray | None = None
        self._spatial_rho: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> SpatialPricingResult:
        """Fit model on return matrix shaped (n_samples, n_assets)."""
        if returns.ndim != 2:
            raise ValueError("returns must be a 2D array")

        n_samples, n_assets = returns.shape
        if n_assets == 0:
            raise ValueError("returns must contain at least one asset")
        if n_samples < self.config.min_observations:
            raise ValueError(
                f"insufficient observations: {n_samples} < {self.config.min_observations}"
            )

        asset_means = np.mean(returns, axis=0)
        centered = returns - asset_means

        n_factors = max(1, min(self.config.n_factors, n_assets))
        factor_loadings, factor_returns = self._estimate_factors(centered, n_factors)
        factor_component = factor_returns @ factor_loadings.T

        residuals = centered - factor_component
        interaction_matrix = self._estimate_interactions(residuals)
        spatial_component = residuals @ interaction_matrix.T

        reconstructed = factor_component + spatial_component
        mse = float(np.mean((centered - reconstructed) ** 2))
        total_var = float(np.var(centered) + 1e-12)
        reconstruction_r2 = 1.0 - (mse / total_var)

        spatial_rho = np.sum(np.abs(interaction_matrix), axis=1)
        residual_volatility = np.std(residuals, axis=0)

        self._asset_means = asset_means
        self._factor_loadings = factor_loadings
        self._factor_returns = factor_returns
        self._interaction_matrix = interaction_matrix
        self._spatial_rho = spatial_rho
        self._fitted = True

        return SpatialPricingResult(
            factor_loadings=factor_loadings,
            factor_returns=factor_returns,
            interaction_matrix=interaction_matrix,
            spatial_rho=spatial_rho,
            reconstruction_r2=reconstruction_r2,
            residual_volatility=residual_volatility,
        )

    def predict_next(self, latest_returns: np.ndarray) -> np.ndarray:
        """Predict next-step returns from current returns."""
        if not self._fitted:
            raise RuntimeError("model must be fitted before predict_next")

        assert self._asset_means is not None
        assert self._factor_loadings is not None
        assert self._interaction_matrix is not None

        current = np.asarray(latest_returns, dtype=float).reshape(-1)
        if current.shape[0] != self._asset_means.shape[0]:
            raise ValueError("latest_returns size does not match fitted asset count")

        centered = current - self._asset_means
        factor_component = self._factor_loadings @ (self._factor_loadings.T @ centered)
        spatial_component = self._interaction_matrix @ centered
        return self._asset_means + factor_component + spatial_component

    def score(self, returns: np.ndarray) -> float:
        """Compute one-step-ahead R^2 score using lagged predictions."""
        if not self._fitted:
            raise RuntimeError("model must be fitted before score")
        if returns.ndim != 2:
            raise ValueError("returns must be a 2D array")
        if returns.shape[0] < 2:
            return 0.0

        predictions = []
        actual = []
        for i in range(1, returns.shape[0]):
            predictions.append(self.predict_next(returns[i - 1]))
            actual.append(returns[i])

        pred = np.asarray(predictions)
        obs = np.asarray(actual)
        mse = float(np.mean((obs - pred) ** 2))
        var = float(np.var(obs) + 1e-12)
        return 1.0 - (mse / var)

    @property
    def interaction_matrix(self) -> np.ndarray:
        """Return fitted interaction matrix W."""
        if self._interaction_matrix is None:
            raise RuntimeError("model is not fitted")
        return self._interaction_matrix

    @property
    def spatial_rho(self) -> np.ndarray:
        """Return per-asset spatial interaction strength."""
        if self._spatial_rho is None:
            raise RuntimeError("model is not fitted")
        return self._spatial_rho

    def _estimate_factors(
        self, centered: np.ndarray, n_factors: int
    ) -> tuple[np.ndarray, np.ndarray]:
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        top_vectors = eigvecs[:, order[:n_factors]]
        factor_returns = centered @ top_vectors
        return top_vectors, factor_returns

    def _estimate_interactions(self, residuals: np.ndarray) -> np.ndarray:
        n_samples, n_assets = residuals.shape
        if n_assets < 2:
            return np.zeros((n_assets, n_assets), dtype=float)

        interaction = np.zeros((n_assets, n_assets), dtype=float)
        alpha = self.config.ridge_alpha

        for i in range(n_assets):
            y = residuals[:, i]
            x = np.delete(residuals, i, axis=1)

            xtx = (x.T @ x) / n_samples
            ridge = xtx + alpha * np.eye(xtx.shape[0])
            xty = (x.T @ y) / n_samples

            coeff = np.linalg.solve(ridge, xty)

            row = np.zeros(n_assets, dtype=float)
            row[np.arange(n_assets) != i] = coeff
            interaction[i] = row

        interaction[np.abs(interaction) < self.config.zero_threshold] = 0.0
        np.fill_diagonal(interaction, 0.0)

        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(interaction))))
        max_radius = self.config.max_spatial_radius
        if spectral_radius > max_radius and spectral_radius > 0:
            interaction *= max_radius / spectral_radius

        return interaction
