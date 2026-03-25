"""AutoTrader module for autonomous trading with continuous learning."""

from .coordinator import (
    AutoTraderConfig,
    AutoTraderCoordinator,
    AutoTraderState,
    DailyMetrics,
    LearningMetrics,
    TradeOutcome,
    TradingCandidate,
    TradingSignal,
)

__all__ = [
    "AutoTraderCoordinator",
    "AutoTraderConfig",
    "AutoTraderState",
    "TradingCandidate",
    "TradingSignal",
    "TradeOutcome",
    "DailyMetrics",
    "LearningMetrics",
]
