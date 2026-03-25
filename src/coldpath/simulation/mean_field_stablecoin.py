"""Mean-field stablecoin de-peg/re-peg simulator.

Two populations interact through secondary trading and primary redemption
channels, with congestion-dependent costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeanFieldConfig:
    """Simulation configuration for stablecoin mean-field dynamics."""

    steps: int = 120
    dt_minutes: float = 1.0
    retail_sensitivity: float = 0.020
    arb_sensitivity: float = 0.050
    inventory_penalty: float = 0.010
    primary_clearance_rate: float = 0.030
    redemption_delay_factor: float = 0.20
    base_cost_bps: float = 3.0
    congestion_cost_bps: float = 20.0
    backlog_cost_bps: float = 12.0
    peg_reversion_speed: float = 0.080
    stress_deviation_bps: float = 75.0
    stress_backlog: float = 0.35
    stress_congestion: float = 0.75


@dataclass
class MeanFieldState:
    """Single-step state in the stablecoin mean-field trajectory."""

    step: int
    peg_deviation_bps: float
    primary_backlog: float
    secondary_liquidity: float
    retail_inventory: float
    arbitrage_inventory: float
    retail_flow: float
    arbitrage_flow: float
    congestion: float
    execution_cost_bps: float
    regime: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "step": self.step,
            "peg_deviation_bps": self.peg_deviation_bps,
            "primary_backlog": self.primary_backlog,
            "secondary_liquidity": self.secondary_liquidity,
            "retail_inventory": self.retail_inventory,
            "arbitrage_inventory": self.arbitrage_inventory,
            "retail_flow": self.retail_flow,
            "arbitrage_flow": self.arbitrage_flow,
            "congestion": self.congestion,
            "execution_cost_bps": self.execution_cost_bps,
            "regime": self.regime,
        }


@dataclass
class MeanFieldSummary:
    """Simulation summary statistics."""

    max_deviation_bps: float
    min_deviation_bps: float
    avg_congestion: float
    time_to_repeg_steps: int | None
    stressed_steps: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "max_deviation_bps": self.max_deviation_bps,
            "min_deviation_bps": self.min_deviation_bps,
            "avg_congestion": self.avg_congestion,
            "time_to_repeg_steps": self.time_to_repeg_steps,
            "stressed_steps": self.stressed_steps,
        }


class StablecoinMeanFieldSimulator:
    """Run two-population mean-field simulations for stablecoin stress."""

    def __init__(self, config: MeanFieldConfig | None = None):
        self.config = config or MeanFieldConfig()

    def simulate(
        self,
        initial_deviation_bps: float,
        shock_schedule: dict[int, float] | None = None,
    ) -> list[MeanFieldState]:
        """Simulate de-peg and re-peg trajectory under mean-field interactions."""
        cfg = self.config
        shocks = shock_schedule or {}

        deviation = float(initial_deviation_bps)
        backlog = 0.05
        liquidity = 1.0
        retail_inventory = 0.0
        arb_inventory = 0.0

        states: list[MeanFieldState] = []
        for step in range(cfg.steps):
            shock = shocks.get(step, 0.0)

            retail_flow = self._retail_flow(deviation, retail_inventory, liquidity)
            arbitrage_flow = self._arbitrage_flow(deviation, backlog, arb_inventory)

            mean_flow = abs(retail_flow) + abs(arbitrage_flow)
            congestion = float(np.tanh(mean_flow * 2.5))
            execution_cost = (
                cfg.base_cost_bps
                + cfg.congestion_cost_bps * congestion
                + cfg.backlog_cost_bps * backlog
            )

            # Primary market backlog grows when arbitrage pressure is high.
            backlog_delta = cfg.redemption_delay_factor * abs(
                arbitrage_flow
            ) - cfg.primary_clearance_rate * max(backlog, 0.0)
            backlog = max(0.0, backlog + backlog_delta)

            # Secondary liquidity thins during stressed flow and congestion.
            liquidity = max(0.05, min(2.0, liquidity + 0.01 - 0.03 * mean_flow - 0.02 * congestion))

            retail_inventory += retail_flow
            arb_inventory += arbitrage_flow

            # Price deviation dynamics: reversion reduced by frictions.
            friction_factor = 1.0 + (execution_cost / 100.0)
            reversion = cfg.peg_reversion_speed * (retail_flow + arbitrage_flow) / friction_factor
            deviation = deviation + shock - reversion * 100.0 + (congestion * 2.0)

            regime = self._classify_regime(deviation, backlog, congestion)
            states.append(
                MeanFieldState(
                    step=step,
                    peg_deviation_bps=deviation,
                    primary_backlog=backlog,
                    secondary_liquidity=liquidity,
                    retail_inventory=retail_inventory,
                    arbitrage_inventory=arb_inventory,
                    retail_flow=retail_flow,
                    arbitrage_flow=arbitrage_flow,
                    congestion=congestion,
                    execution_cost_bps=execution_cost,
                    regime=regime,
                )
            )

        return states

    def summarize(
        self, states: list[MeanFieldState], repeg_band_bps: float = 10.0
    ) -> MeanFieldSummary:
        """Summarize mean-field trajectory."""
        if not states:
            return MeanFieldSummary(
                max_deviation_bps=0.0,
                min_deviation_bps=0.0,
                avg_congestion=0.0,
                time_to_repeg_steps=None,
                stressed_steps=0,
            )

        deviations = [s.peg_deviation_bps for s in states]
        congestions = [s.congestion for s in states]
        stressed_steps = sum(1 for s in states if s.regime == "stressed")

        time_to_repeg: int | None = None
        for state in states:
            if abs(state.peg_deviation_bps) <= repeg_band_bps:
                time_to_repeg = state.step
                break

        return MeanFieldSummary(
            max_deviation_bps=float(max(deviations)),
            min_deviation_bps=float(min(deviations)),
            avg_congestion=float(np.mean(congestions)),
            time_to_repeg_steps=time_to_repeg,
            stressed_steps=stressed_steps,
        )

    def _retail_flow(self, deviation_bps: float, inventory: float, liquidity: float) -> float:
        cfg = self.config
        signal = -cfg.retail_sensitivity * np.tanh(deviation_bps / 80.0)
        inv_penalty = cfg.inventory_penalty * inventory
        liquidity_drag = (1.0 - min(liquidity, 1.0)) * 0.02
        return float(signal - inv_penalty - liquidity_drag)

    def _arbitrage_flow(self, deviation_bps: float, backlog: float, inventory: float) -> float:
        cfg = self.config
        signal = -cfg.arb_sensitivity * np.tanh(deviation_bps / 60.0)
        backlog_drag = backlog * 0.04
        inv_penalty = cfg.inventory_penalty * 0.5 * inventory
        return float(signal - backlog_drag - inv_penalty)

    def _classify_regime(self, deviation_bps: float, backlog: float, congestion: float) -> str:
        cfg = self.config
        if (
            abs(deviation_bps) >= cfg.stress_deviation_bps
            or backlog >= cfg.stress_backlog
            or congestion >= cfg.stress_congestion
        ):
            return "stressed"
        return "normal"
