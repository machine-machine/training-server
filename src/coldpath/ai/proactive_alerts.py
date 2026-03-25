"""
Proactive Alert Generator for AI-driven trading alerts.

Triggers alerts on:
- Regime changes (market state transitions)
- Drawdown spikes (consecutive losses)
- Unusual token activity (volume/holder anomalies)
- Circuit breaker activation
- RL policy changes

Uses Sonnet 4.5 for fast 2-3 sentence explanations.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertTrigger(Enum):
    """What triggered the alert."""

    REGIME_CHANGE = "regime_change"
    DRAWDOWN_SPIKE = "drawdown_spike"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    CIRCUIT_BREAKER = "circuit_breaker"
    UNUSUAL_VOLUME = "unusual_volume"
    RL_POLICY_CHANGE = "rl_policy_change"
    POSITION_RISK = "position_risk"
    PERFORMANCE_MILESTONE = "performance_milestone"


@dataclass
class ProactiveAlert:
    """A proactive AI-generated alert."""

    title: str
    explanation: str
    severity: AlertSeverity
    trigger: AlertTrigger
    suggested_actions: list[str]
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "explanation": self.explanation,
            "severity": self.severity.value,
            "trigger": self.trigger.value,
            "suggested_actions": self.suggested_actions,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class AlertConfig:
    """Configuration for the alert generator."""

    # Drawdown thresholds
    drawdown_warning_pct: float = 5.0
    drawdown_critical_pct: float = 10.0

    # Consecutive loss threshold
    consecutive_loss_warning: int = 3
    consecutive_loss_critical: int = 5

    # Volume anomaly (multiple of average)
    volume_anomaly_threshold: float = 3.0

    # Cooldown between same-trigger alerts (seconds)
    alert_cooldown_s: float = 300.0  # 5 minutes

    # Maximum alerts per hour
    max_alerts_per_hour: int = 20

    # Whether to use LLM for explanation generation
    use_llm_explanation: bool = True
    llm_timeout_s: float = 5.0


class ProactiveAlertGenerator:
    """Generates proactive AI alerts based on trading conditions.

    Monitors market and portfolio state, triggering alerts when
    significant events occur. Uses Claude Sonnet for natural-language
    explanations.
    """

    def __init__(
        self,
        claude_client=None,  # Optional ClaudeClient for LLM explanations
        config: AlertConfig | None = None,
    ):
        self.client = claude_client
        self.config = config or AlertConfig()

        # Alert history and cooldown tracking
        self._alerts: list[ProactiveAlert] = []
        self._last_trigger_time: dict[str, float] = {}
        self._alerts_this_hour: int = 0
        self._hour_start: float = time.time()

        # Listeners
        self._listeners: list[Callable[[ProactiveAlert], None]] = []

        # State tracking
        self._last_regime: str | None = None
        self._consecutive_losses: int = 0
        self._peak_equity: float = 1.0
        self._current_drawdown: float = 0.0

    def on_alert(self, callback: Callable[[ProactiveAlert], None]):
        """Register a callback for new alerts."""
        self._listeners.append(callback)

    async def check_regime_change(
        self,
        new_regime: str,
        regime_features: dict[str, float],
    ) -> ProactiveAlert | None:
        """Check for regime change and generate alert if needed."""
        if self._last_regime is None:
            self._last_regime = new_regime
            return None

        if new_regime == self._last_regime:
            return None

        old_regime = self._last_regime
        self._last_regime = new_regime

        severity = AlertSeverity.WARNING
        if new_regime in ("Bear", "MevHeavy"):
            severity = AlertSeverity.CRITICAL

        explanation = await self._generate_explanation(
            f"Market regime changed from {old_regime} to {new_regime}. "
            f"Key features: {regime_features}",
            AlertTrigger.REGIME_CHANGE,
        )

        suggested = []
        if new_regime == "Bear":
            suggested = [
                "Reduce position sizes",
                "Tighten stop losses",
                "Consider pausing auto-trading",
            ]
        elif new_regime == "MevHeavy":
            suggested = [
                "Increase slippage tolerance",
                "Reduce trade frequency",
                "Monitor transaction inclusion",
            ]
        elif new_regime == "Bull":
            suggested = [
                "Consider increasing position sizes",
                "Widen trailing stops for bigger moves",
            ]
        else:
            suggested = ["Review current positions", "No immediate action needed"]

        return await self._emit_alert(
            title=f"Regime Change: {old_regime} → {new_regime}",
            explanation=explanation,
            severity=severity,
            trigger=AlertTrigger.REGIME_CHANGE,
            suggested_actions=suggested,
            metadata={
                "old_regime": old_regime,
                "new_regime": new_regime,
                "features": regime_features,
            },
        )

    async def check_drawdown(
        self,
        current_equity: float,
        peak_equity: float | None = None,
    ) -> ProactiveAlert | None:
        """Check for drawdown threshold breach."""
        if peak_equity is not None:
            self._peak_equity = max(self._peak_equity, peak_equity)
        self._peak_equity = max(self._peak_equity, current_equity)

        if self._peak_equity <= 0:
            return None

        drawdown_pct = (self._peak_equity - current_equity) / self._peak_equity * 100
        self._current_drawdown = drawdown_pct

        if drawdown_pct >= self.config.drawdown_critical_pct:
            severity = AlertSeverity.CRITICAL
        elif drawdown_pct >= self.config.drawdown_warning_pct:
            severity = AlertSeverity.WARNING
        else:
            return None

        explanation = await self._generate_explanation(
            f"Portfolio drawdown reached {drawdown_pct:.1f}% from peak. "
            f"Current equity: {current_equity:.4f}, Peak: {self._peak_equity:.4f}",
            AlertTrigger.DRAWDOWN_SPIKE,
        )

        return await self._emit_alert(
            title=f"Drawdown Alert: {drawdown_pct:.1f}% from peak",
            explanation=explanation,
            severity=severity,
            trigger=AlertTrigger.DRAWDOWN_SPIKE,
            suggested_actions=[
                "Review open positions for risk",
                "Consider reducing exposure",
                "Check if stop losses are properly set",
            ],
            metadata={"drawdown_pct": drawdown_pct, "current_equity": current_equity},
        )

    async def check_trade_result(
        self,
        pnl_pct: float,
        trade_info: dict[str, Any],
    ) -> ProactiveAlert | None:
        """Check trade results for consecutive loss patterns."""
        if pnl_pct < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
            return None

        if self._consecutive_losses >= self.config.consecutive_loss_critical:
            severity = AlertSeverity.CRITICAL
        elif self._consecutive_losses >= self.config.consecutive_loss_warning:
            severity = AlertSeverity.WARNING
        else:
            return None

        explanation = await self._generate_explanation(
            f"{self._consecutive_losses} consecutive losing trades. "
            f"Last loss: {pnl_pct:.2f}%. Trade info: {trade_info}",
            AlertTrigger.CONSECUTIVE_LOSSES,
        )

        return await self._emit_alert(
            title=f"Losing Streak: {self._consecutive_losses} consecutive losses",
            explanation=explanation,
            severity=severity,
            trigger=AlertTrigger.CONSECUTIVE_LOSSES,
            suggested_actions=[
                "Review recent trade parameters",
                "Consider pausing auto-trading temporarily",
                "Check if market conditions have changed",
            ],
            metadata={"streak": self._consecutive_losses, "last_pnl": pnl_pct},
        )

    async def check_circuit_breaker(
        self,
        reason: str,
        details: dict[str, Any],
    ) -> ProactiveAlert | None:
        """Alert on circuit breaker activation."""
        explanation = await self._generate_explanation(
            f"Circuit breaker activated: {reason}. Details: {details}",
            AlertTrigger.CIRCUIT_BREAKER,
        )

        return await self._emit_alert(
            title=f"Circuit Breaker: {reason}",
            explanation=explanation,
            severity=AlertSeverity.CRITICAL,
            trigger=AlertTrigger.CIRCUIT_BREAKER,
            suggested_actions=[
                "Trading is paused — review conditions before resuming",
                "Check daily loss limits and adjust if needed",
                "Review market conditions for abnormalities",
            ],
            metadata=details,
        )

    async def _generate_explanation(self, context: str, trigger: AlertTrigger) -> str:
        """Generate natural-language explanation using Claude Sonnet."""
        if not self.config.use_llm_explanation or self.client is None:
            return context

        try:
            from .claude_client import ModelTier

            response = await asyncio.wait_for(
                self.client.chat(
                    message=f"In 2-3 sentences, explain this trading alert to a trader:\n{context}",
                    tier=ModelTier.SONNET,
                    max_tokens=256,
                    temperature=0.5,
                ),
                timeout=self.config.llm_timeout_s,
            )
            return response.content
        except Exception as e:
            logger.debug(f"LLM explanation failed, using raw context: {e}")
            return context

    async def _emit_alert(self, **kwargs) -> ProactiveAlert | None:
        """Create and emit an alert, respecting cooldown and rate limits."""
        trigger = kwargs["trigger"]
        trigger_key = trigger.value

        # Check cooldown
        now = time.time()
        last_time = self._last_trigger_time.get(trigger_key, 0)
        if now - last_time < self.config.alert_cooldown_s:
            return None

        # Check hourly rate limit
        if now - self._hour_start > 3600:
            self._alerts_this_hour = 0
            self._hour_start = now
        if self._alerts_this_hour >= self.config.max_alerts_per_hour:
            return None

        alert = ProactiveAlert(**kwargs)
        self._alerts.append(alert)
        self._last_trigger_time[trigger_key] = now
        self._alerts_this_hour += 1

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(alert)
            except Exception as e:
                logger.warning(f"Alert listener error: {e}")

        logger.info(f"Alert: [{alert.severity.value}] {alert.title}")
        return alert

    def get_recent_alerts(self, count: int = 10) -> list[ProactiveAlert]:
        """Get the most recent alerts."""
        return list(reversed(self._alerts[-count:]))

    def get_stats(self) -> dict[str, Any]:
        """Get alert generator statistics."""
        return {
            "total_alerts": len(self._alerts),
            "alerts_this_hour": self._alerts_this_hour,
            "consecutive_losses": self._consecutive_losses,
            "current_drawdown_pct": self._current_drawdown,
            "last_regime": self._last_regime,
        }
