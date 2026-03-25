"""End-to-end orchestration layer combining SAPT, semantic parity, and mean-field risk.

This module adapts signal confidence/size using:
- Spatial factor-network pricing diagnostics
- Semantic fragmentation and parity opportunities
- Mean-field stablecoin stress regime detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..analysis.semantic_fragmentation import (
    FragmentationConfig,
    MarketContract,
    SemanticFragmentationAnalyzer,
)
from ..portfolio.spatial_pricing import SpatialArbitragePricingModel, SpatialPricingConfig
from ..simulation.mean_field_stablecoin import MeanFieldConfig, StablecoinMeanFieldSimulator


@dataclass
class AdvancedOrchestratorConfig:
    """Configuration for advanced multi-model orchestration."""

    enable_spatial: bool = True
    enable_semantic: bool = True
    enable_mean_field: bool = True
    min_samples_for_spatial: int = 5
    spatial_refit_interval: int = 8
    max_assets_for_spatial: int = 8
    max_confidence_multiplier: float = 1.15
    min_confidence_multiplier: float = 0.55
    stressed_regime_multiplier: float = 0.80
    parity_boost_scale: float = 3.0
    stablecoin_symbols: tuple[str, ...] = (
        "USDC",
        "USDT",
        "DAI",
        "PYUSD",
        "FDUSD",
        "USDE",
    )


@dataclass
class OrchestrationDecision:
    """Per-event orchestration decision details."""

    confidence_multiplier: float
    spatial_predicted_return: float | None = None
    spatial_model_score: float | None = None
    parity_opportunities: int = 0
    best_parity_net_edge: float = 0.0
    stablecoin_regime: str | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_multiplier": self.confidence_multiplier,
            "spatial_predicted_return": self.spatial_predicted_return,
            "spatial_model_score": self.spatial_model_score,
            "parity_opportunities": self.parity_opportunities,
            "best_parity_net_edge": self.best_parity_net_edge,
            "stablecoin_regime": self.stablecoin_regime,
            "reasons": self.reasons,
        }


class AdvancedStrategyOrchestrator:
    """Apply multi-model adjustments to raw strategy signals."""

    def __init__(self, config: AdvancedOrchestratorConfig | None = None):
        self.config = config or AdvancedOrchestratorConfig()

        self._prices: dict[str, list[float]] = {}
        self._returns: dict[str, list[float]] = {}
        self._events_seen = 0

        self._spatial_model: SpatialArbitragePricingModel | None = None
        self._spatial_assets: list[str] = []
        self._spatial_score: float | None = None

        self._semantic_analyzer = SemanticFragmentationAnalyzer(
            FragmentationConfig(min_similarity=0.35, min_executable_edge=0.002)
        )
        self._recent_contracts: list[MarketContract] = []
        self._recent_parity_count = 0
        self._recent_best_parity_edge = 0.0

        self._mean_field = StablecoinMeanFieldSimulator(MeanFieldConfig(steps=24))
        self._recent_regime_by_mint: dict[str, str] = {}

        self._decisions_applied = 0

    def process_event(self, event: Any) -> None:
        """Update internal model state from each market event."""
        self._events_seen += 1
        mint = getattr(event, "mint", "")
        data = getattr(event, "data", {}) or {}

        price = self._extract_price(data)
        if mint and price is not None:
            prices = self._prices.setdefault(mint, [])
            if prices:
                prev = prices[-1]
                if prev > 0:
                    ret = (price - prev) / prev
                    self._returns.setdefault(mint, []).append(float(ret))
            prices.append(float(price))
            if len(prices) > 400:
                del prices[:-400]
            returns = self._returns.get(mint)
            if returns and len(returns) > 400:
                del returns[:-400]

        if self.config.enable_semantic:
            contract = self._contract_from_event(event)
            if contract is not None:
                self._recent_contracts.append(contract)
                if len(self._recent_contracts) > 250:
                    del self._recent_contracts[:-250]

                # Recompute parity opportunities periodically.
                if self._events_seen % 20 == 0 and len(self._recent_contracts) >= 6:
                    alignment = self._semantic_analyzer.align_contracts(self._recent_contracts)
                    opportunities = self._semantic_analyzer.detect_parity_opportunities(alignment)
                    self._recent_parity_count = len(opportunities)
                    self._recent_best_parity_edge = max(
                        (o.net_edge for o in opportunities),
                        default=0.0,
                    )

        if self.config.enable_mean_field and self._is_stablecoin(mint, data):
            deviation_bps = self._stablecoin_deviation_bps(data)
            states = self._mean_field.simulate(initial_deviation_bps=deviation_bps)
            summary = self._mean_field.summarize(states)
            regime = "stressed" if summary.stressed_steps > (len(states) // 3) else "normal"
            self._recent_regime_by_mint[mint] = regime

        if (
            self.config.enable_spatial
            and self._events_seen % self.config.spatial_refit_interval == 0
        ):
            self._fit_spatial_model()

    def adjust_signal(self, signal: Any, event: Any) -> tuple[Any, OrchestrationDecision]:
        """Adjust strategy signal confidence and sizing from model diagnostics."""
        multiplier = 1.0
        reasons: list[str] = []
        spatial_predicted_return: float | None = None

        mint = getattr(signal, "mint", "")
        signal_type = getattr(getattr(signal, "type", None), "value", "")

        if (
            self.config.enable_spatial
            and self._spatial_model is not None
            and mint in self._spatial_assets
        ):
            try:
                latest_vector = self._latest_returns_vector(self._spatial_assets)
                if latest_vector is not None:
                    pred = self._spatial_model.predict_next(latest_vector)
                    idx = self._spatial_assets.index(mint)
                    spatial_predicted_return = float(pred[idx])
                    if signal_type == "buy":
                        multiplier *= 1.0 + np.clip(spatial_predicted_return * 15.0, -0.25, 0.25)
                    elif signal_type == "sell":
                        multiplier *= 1.0 + np.clip(-spatial_predicted_return * 12.0, -0.20, 0.20)
                    reasons.append("spatial")
            except Exception:
                pass

        if self.config.enable_semantic and self._recent_best_parity_edge > 0:
            parity_boost = min(0.12, self._recent_best_parity_edge * self.config.parity_boost_scale)
            multiplier *= 1.0 + parity_boost
            reasons.append("semantic_fragmentation")

        if self.config.enable_mean_field and self._is_stablecoin(
            mint, getattr(event, "data", {}) or {}
        ):
            regime = self._recent_regime_by_mint.get(mint)
            if regime == "stressed":
                multiplier *= self.config.stressed_regime_multiplier
                reasons.append("mean_field_stress")

        multiplier = float(
            np.clip(
                multiplier,
                self.config.min_confidence_multiplier,
                self.config.max_confidence_multiplier,
            )
        )

        if hasattr(signal, "confidence"):
            signal.confidence = float(np.clip(signal.confidence * multiplier, 0.0, 1.0))

        if hasattr(signal, "target_amount_sol") and signal.target_amount_sol is not None:
            signal.target_amount_sol = max(0.0, float(signal.target_amount_sol) * multiplier)

        self._decisions_applied += 1

        decision = OrchestrationDecision(
            confidence_multiplier=multiplier,
            spatial_predicted_return=spatial_predicted_return,
            spatial_model_score=self._spatial_score,
            parity_opportunities=self._recent_parity_count,
            best_parity_net_edge=self._recent_best_parity_edge,
            stablecoin_regime=self._recent_regime_by_mint.get(mint),
            reasons=reasons,
        )
        return signal, decision

    def summary(self) -> dict[str, Any]:
        """Aggregate orchestration diagnostics."""
        return {
            "events_seen": self._events_seen,
            "decisions_applied": self._decisions_applied,
            "spatial_assets": self._spatial_assets,
            "spatial_model_score": self._spatial_score,
            "parity_opportunities": self._recent_parity_count,
            "best_parity_net_edge": self._recent_best_parity_edge,
            "stablecoin_regimes": self._recent_regime_by_mint,
        }

    def _fit_spatial_model(self) -> None:
        candidates = [
            mint
            for mint, series in self._returns.items()
            if len(series) >= self.config.min_samples_for_spatial
        ]
        if not candidates:
            return

        # Prefer most liquid histories by sample count.
        candidates.sort(key=lambda m: len(self._returns[m]), reverse=True)
        assets = candidates[: self.config.max_assets_for_spatial]
        matrix = self._returns_matrix(assets)
        if matrix is None or matrix.shape[0] < self.config.min_samples_for_spatial:
            return

        model = SpatialArbitragePricingModel(
            SpatialPricingConfig(
                n_factors=min(3, max(1, len(assets) - 1)),
                min_observations=self.config.min_samples_for_spatial,
            )
        )
        try:
            model.fit(matrix)
            self._spatial_score = model.score(matrix)
            self._spatial_model = model
            self._spatial_assets = assets
        except Exception:
            return

    def _returns_matrix(self, assets: list[str]) -> np.ndarray | None:
        if not assets:
            return None
        min_len = min(len(self._returns[a]) for a in assets)
        if min_len <= 1:
            return None
        return np.column_stack(
            [np.asarray(self._returns[a][-min_len:], dtype=float) for a in assets]
        )

    def _latest_returns_vector(self, assets: list[str]) -> np.ndarray | None:
        latest: list[float] = []
        for asset in assets:
            series = self._returns.get(asset)
            if not series:
                return None
            latest.append(float(series[-1]))
        return np.asarray(latest, dtype=float)

    def _extract_price(self, data: dict[str, Any]) -> float | None:
        for key in ("close", "price", "open"):
            value = data.get(key)
            if value is not None:
                try:
                    price = float(value)
                    if price > 0:
                        return price
                except (TypeError, ValueError):
                    continue
        return None

    def _contract_from_event(self, event: Any) -> MarketContract | None:
        data = getattr(event, "data", {}) or {}
        description = data.get("contract_description")
        if not description:
            return None

        venue = str(data.get("venue") or getattr(event, "pool", "unknown") or "unknown")
        price = data.get("market_price", data.get("price", data.get("close", 0.5)))

        try:
            market_price = float(price)
        except (TypeError, ValueError):
            market_price = 0.5

        return MarketContract(
            venue=venue,
            symbol=str(data.get("symbol", getattr(event, "mint", ""))),
            description=str(description),
            outcome=str(data.get("market_outcome", "yes")).lower(),
            price=market_price,
            metadata=data.get("metadata"),
            fee_bps=float(data.get("fee_bps", 10.0)),
            settlement_risk_bps=float(data.get("settlement_risk_bps", 5.0)),
            capital_lockup_bps=float(data.get("capital_lockup_bps", 5.0)),
        )

    def _is_stablecoin(self, mint: str, data: dict[str, Any]) -> bool:
        symbol = str(data.get("symbol", "")).upper()
        mint_upper = mint.upper()
        return any(
            token in symbol or token in mint_upper for token in self.config.stablecoin_symbols
        )

    def _stablecoin_deviation_bps(self, data: dict[str, Any]) -> float:
        price = self._extract_price(data)
        if price is None:
            return 0.0
        return float((price - 1.0) * 10000)
