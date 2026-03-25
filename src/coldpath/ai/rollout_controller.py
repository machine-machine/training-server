"""
Controlled rollout controller with evidence gates and auto-rollback.

Rollout path:
    off -> shadow -> paper -> canary -> general
"""

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class RolloutStage(StrEnum):
    OFF = "off"
    SHADOW = "shadow"
    PAPER = "paper"
    CANARY = "canary"
    GENERAL = "general"


TRANSITIONS = {
    RolloutStage.OFF: RolloutStage.SHADOW,
    RolloutStage.SHADOW: RolloutStage.PAPER,
    RolloutStage.PAPER: RolloutStage.CANARY,
    RolloutStage.CANARY: RolloutStage.GENERAL,
    RolloutStage.GENERAL: RolloutStage.GENERAL,
}


@dataclass
class EvidenceGates:
    min_fill_rate_pct: float = 85.0
    max_failed_tx_rate_pct: float = 3.0
    max_inclusion_latency_ms_p95: float = 1800.0
    min_net_pnl_after_costs_sol: float = 0.0
    max_rug_unsellable_incident_rate_pct: float = 1.5
    max_drawdown_pct: float = 15.0


@dataclass
class RolloutDecision:
    allowed: bool
    current_stage: str
    next_stage: str
    auto_rollback_triggered: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "current_stage": self.current_stage,
            "next_stage": self.next_stage,
            "auto_rollback_triggered": self.auto_rollback_triggered,
            "reasons": self.reasons,
        }


class RolloutController:
    """Gate rollout progression using evidence metrics and rollback triggers."""

    def __init__(
        self,
        stage: RolloutStage = RolloutStage.OFF,
        gates: EvidenceGates | None = None,
    ) -> None:
        self.stage = stage
        self.gates = gates or EvidenceGates()

    def get_status(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "gates": asdict(self.gates),
        }

    def evaluate(self, metrics: dict[str, Any]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        fill = float(metrics.get("fill_rate_pct", 100.0) or 100.0)
        failed = float(metrics.get("failed_tx_rate_pct", 0.0) or 0.0)
        p95 = float(metrics.get("inclusion_latency_ms_p95", 0.0) or 0.0)
        net_pnl = float(metrics.get("net_pnl_after_costs_sol", 0.0) or 0.0)
        incidents = float(metrics.get("rug_unsellable_incident_rate_pct", 0.0) or 0.0)
        drawdown = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)

        if fill < self.gates.min_fill_rate_pct:
            reasons.append(f"fill_rate_pct {fill:.2f} < {self.gates.min_fill_rate_pct:.2f}")
        if failed > self.gates.max_failed_tx_rate_pct:
            reasons.append(
                f"failed_tx_rate_pct {failed:.2f} > {self.gates.max_failed_tx_rate_pct:.2f}"
            )
        if p95 > self.gates.max_inclusion_latency_ms_p95:
            reasons.append(
                f"inclusion_latency_ms_p95 {p95:.0f} > "
                f"{self.gates.max_inclusion_latency_ms_p95:.0f}"
            )
        if net_pnl < self.gates.min_net_pnl_after_costs_sol:
            reasons.append(
                f"net_pnl_after_costs_sol {net_pnl:.4f} < "
                f"{self.gates.min_net_pnl_after_costs_sol:.4f}"
            )
        if incidents > self.gates.max_rug_unsellable_incident_rate_pct:
            reasons.append(
                f"rug_unsellable_incident_rate_pct {incidents:.2f} > "
                f"{self.gates.max_rug_unsellable_incident_rate_pct:.2f}"
            )
        if drawdown > self.gates.max_drawdown_pct:
            reasons.append(f"max_drawdown_pct {drawdown:.2f} > {self.gates.max_drawdown_pct:.2f}")
        return len(reasons) == 0, reasons

    def advance_or_rollback(self, metrics: dict[str, Any]) -> RolloutDecision:
        passed, reasons = self.evaluate(metrics)
        current = self.stage
        if passed:
            self.stage = TRANSITIONS[self.stage]
            return RolloutDecision(
                allowed=True,
                current_stage=current.value,
                next_stage=self.stage.value,
                auto_rollback_triggered=False,
                reasons=[],
            )

        # Auto-rollback only from riskier rollout stages.
        auto_rollback = self.stage in (RolloutStage.CANARY, RolloutStage.GENERAL)
        if auto_rollback:
            self.stage = RolloutStage.PAPER
        return RolloutDecision(
            allowed=False,
            current_stage=current.value,
            next_stage=self.stage.value,
            auto_rollback_triggered=auto_rollback,
            reasons=reasons,
        )
