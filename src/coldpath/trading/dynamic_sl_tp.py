"""
Dynamic Stop-Loss and Take-Profit Calculator.

Calculates optimal stop-loss and take-profit levels based on:
- Market volatility (ATR-based)
- Support/resistance levels
- Risk/reward optimization
- Market regime awareness
- Historical performance patterns

Key Features:
- ATR-based dynamic levels
- Support/resistance awareness
- Kelly-optimal risk/reward ratios
- Market regime adjustments
- Trailing stop calculations
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Current market condition."""

    LOW_VOLATILITY = "low_volatility"
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    EXTREME_VOLATILITY = "extreme_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"


class StopType(Enum):
    """Type of stop-loss."""

    FIXED = "fixed"
    ATR_BASED = "atr_based"
    PERCENTAGE = "percentage"
    SUPPORT_BASED = "support_based"
    TRAILING = "trailing"


@dataclass
class ATRData:
    """ATR calculation data."""

    atr_value: float
    atr_percent: float
    atr_period: int = 14

    def to_dict(self) -> dict[str, Any]:
        return {
            "atr_value": self.atr_value,
            "atr_percent": self.atr_percent,
            "atr_period": self.atr_period,
        }


@dataclass
class StopLossResult:
    """Result of stop-loss calculation."""

    stop_price: float
    stop_percent: float
    stop_type: StopType
    risk_amount: float
    risk_percent: float
    reason: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_price": self.stop_price,
            "stop_percent": self.stop_percent,
            "stop_type": self.stop_type.value,
            "risk_amount": self.risk_amount,
            "risk_percent": self.risk_percent,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class TakeProfitResult:
    """Result of take-profit calculation."""

    take_profit_price: float
    take_profit_percent: float
    reward_amount: float
    reward_percent: float
    risk_reward_ratio: float
    levels: list[tuple[float, float]] = field(default_factory=list)
    reason: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "take_profit_price": self.take_profit_price,
            "take_profit_percent": self.take_profit_percent,
            "reward_amount": self.reward_amount,
            "reward_percent": self.reward_percent,
            "risk_reward_ratio": self.risk_reward_ratio,
            "levels": self.levels,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class DynamicSLTPResult:
    """Complete SL/TP calculation result."""

    entry_price: float
    stop_loss: StopLossResult
    take_profit: TakeProfitResult
    market_condition: MarketCondition
    recommended_position_size_pct: float
    atr_data: ATRData | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss.to_dict(),
            "take_profit": self.take_profit.to_dict(),
            "market_condition": self.market_condition.value,
            "recommended_position_size_pct": self.recommended_position_size_pct,
            "atr_data": self.atr_data.to_dict() if self.atr_data else None,
            "warnings": self.warnings,
        }


@dataclass
class DynamicSLTPConfig:
    """Configuration for dynamic SL/TP calculator."""

    min_risk_reward_ratio: float = 2.0
    max_risk_reward_ratio: float = 6.0
    target_risk_reward_ratio: float = 3.0

    min_stop_loss_pct: float = 2.0
    max_stop_loss_pct: float = 15.0
    default_stop_loss_pct: float = 6.0

    min_take_profit_pct: float = 5.0
    max_take_profit_pct: float = 50.0

    atr_multiplier_sl: float = 1.5
    atr_multiplier_tp: float = 3.0

    max_risk_per_trade_pct: float = 2.0

    trailing_activation_pct: float = 1.0
    trailing_distance_pct: float = 1.5

    use_support_resistance: bool = True
    support_resistance_threshold: float = 0.5


class DynamicSLTPCalculator:
    """Dynamic Stop-Loss and Take-Profit Calculator.

    Calculates optimal SL/TP levels based on market conditions,
    volatility, and risk management principles.

    Usage:
        calculator = DynamicSLTPCalculator()

        result = calculator.calculate(
            entry_price=100.0,
            atr=2.5,
            atr_percent=2.5,
            position_size=0.1,
            market_condition=MarketCondition.NORMAL,
        )

        print(f"Stop Loss: {result.stop_loss.stop_price}")
        print(f"Take Profit: {result.take_profit.take_profit_price}")
    """

    def __init__(self, config: DynamicSLTPConfig | None = None):
        self.config = config or DynamicSLTPConfig()

        self._support_levels: list[float] = []
        self._resistance_levels: list[float] = []
        self._historical_performance: list[dict] = []

    def calculate(
        self,
        entry_price: float,
        atr: float | None = None,
        atr_percent: float | None = None,
        position_size: float = 1.0,
        market_condition: MarketCondition | None = None,
        support_levels: list[float] | None = None,
        resistance_levels: list[float] | None = None,
        is_long: bool = True,
        win_rate: float | None = None,
        avg_win_pct: float | None = None,
        avg_loss_pct: float | None = None,
    ) -> DynamicSLTPResult:
        """Calculate dynamic stop-loss and take-profit levels.

        Args:
            entry_price: Entry price of the position
            atr: Average True Range value (optional)
            atr_percent: ATR as percentage of price (optional)
            position_size: Position size in base currency
            market_condition: Current market condition
            support_levels: List of support price levels
            resistance_levels: List of resistance price levels
            is_long: True for long position, False for short
            win_rate: Historical win rate (0-1)
            avg_win_pct: Average win percentage
            avg_loss_pct: Average loss percentage

        Returns:
            DynamicSLTPResult with calculated levels
        """
        warnings = []

        if support_levels:
            self._support_levels = sorted(support_levels)
        if resistance_levels:
            self._resistance_levels = sorted(resistance_levels)

        if market_condition is None:
            market_condition = self._infer_market_condition(atr_percent)

        atr_data = None
        if atr is not None and atr_percent is not None:
            atr_data = ATRData(atr_value=atr, atr_percent=atr_percent)

        stop_loss = self._calculate_stop_loss(
            entry_price=entry_price,
            atr=atr,
            atr_percent=atr_percent,
            market_condition=market_condition,
            is_long=is_long,
        )

        take_profit = self._calculate_take_profit(
            entry_price=entry_price,
            stop_loss=stop_loss,
            atr=atr,
            atr_percent=atr_percent,
            market_condition=market_condition,
            is_long=is_long,
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
        )

        if take_profit.risk_reward_ratio < self.config.min_risk_reward_ratio:
            warnings.append(
                f"R/R ratio {take_profit.risk_reward_ratio:.2f} below minimum "
                f"{self.config.min_risk_reward_ratio:.2f}"
            )

        recommended_position = self._calculate_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            position_size=position_size,
            market_condition=market_condition,
        )

        return DynamicSLTPResult(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            market_condition=market_condition,
            recommended_position_size_pct=recommended_position,
            atr_data=atr_data,
            warnings=warnings,
        )

    def _calculate_stop_loss(
        self,
        entry_price: float,
        atr: float | None,
        atr_percent: float | None,
        market_condition: MarketCondition,
        is_long: bool,
    ) -> StopLossResult:
        """Calculate stop-loss level."""

        if atr is not None and atr_percent is not None:
            atr_sl_pct = (atr * self.config.atr_multiplier_sl / entry_price) * 100

            adjustment = self._get_volatility_adjustment(market_condition)
            atr_sl_pct *= adjustment

            if is_long:
                stop_price = entry_price * (1 - atr_sl_pct / 100)
            else:
                stop_price = entry_price * (1 + atr_sl_pct / 100)

            sr_stop = self._check_support_resistance_stop(entry_price, stop_price, is_long)
            if sr_stop is not None:
                stop_price = sr_stop
                atr_sl_pct = abs(entry_price - stop_price) / entry_price * 100

            stop_type = StopType.ATR_BASED
            reason = f"ATR-based stop ({self.config.atr_multiplier_sl}x ATR)"

        else:
            base_sl_pct = self.config.default_stop_loss_pct

            adjustment = self._get_volatility_adjustment(market_condition)
            base_sl_pct *= adjustment

            if is_long:
                stop_price = entry_price * (1 - base_sl_pct / 100)
            else:
                stop_price = entry_price * (1 + base_sl_pct / 100)

            atr_sl_pct = base_sl_pct
            stop_type = StopType.PERCENTAGE
            reason = f"Percentage-based stop ({base_sl_pct:.1f}%)"

        atr_sl_pct = np.clip(
            atr_sl_pct,
            self.config.min_stop_loss_pct,
            self.config.max_stop_loss_pct,
        )

        if is_long:
            stop_price = entry_price * (1 - atr_sl_pct / 100)
        else:
            stop_price = entry_price * (1 + atr_sl_pct / 100)

        risk_amount = abs(entry_price - stop_price)
        risk_percent = atr_sl_pct

        return StopLossResult(
            stop_price=round(stop_price, 6),
            stop_percent=round(atr_sl_pct, 4),
            stop_type=stop_type,
            risk_amount=round(risk_amount, 6),
            risk_percent=round(risk_percent, 4),
            reason=reason,
        )

    def _calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: StopLossResult,
        atr: float | None,
        atr_percent: float | None,
        market_condition: MarketCondition,
        is_long: bool,
        win_rate: float | None,
        avg_win_pct: float | None,
        avg_loss_pct: float | None,
    ) -> TakeProfitResult:
        """Calculate take-profit level(s)."""

        optimal_rr = self._calculate_optimal_risk_reward(
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            market_condition=market_condition,
        )

        tp_percent = stop_loss.stop_percent * optimal_rr
        tp_percent = np.clip(
            tp_percent,
            self.config.min_take_profit_pct,
            self.config.max_take_profit_pct,
        )

        if is_long:
            tp_price = entry_price * (1 + tp_percent / 100)
        else:
            tp_price = entry_price * (1 - tp_percent / 100)

        sr_tp = self._check_support_resistance_tp(entry_price, tp_price, is_long)
        if sr_tp is not None:
            tp_price = sr_tp
            tp_percent = abs(tp_price - entry_price) / entry_price * 100

        reward_amount = abs(tp_price - entry_price)
        reward_percent = tp_percent
        risk_reward = tp_percent / stop_loss.stop_percent if stop_loss.stop_percent > 0 else 0

        levels = self._calculate_partial_tp_levels(
            entry_price=entry_price,
            tp_price=tp_price,
            stop_price=stop_loss.stop_price,
            is_long=is_long,
        )

        return TakeProfitResult(
            take_profit_price=round(tp_price, 6),
            take_profit_percent=round(tp_percent, 4),
            reward_amount=round(reward_amount, 6),
            reward_percent=round(reward_percent, 4),
            risk_reward_ratio=round(risk_reward, 2),
            levels=levels,
            reason=f"R/R {optimal_rr:.1f}x target",
        )

    def _calculate_optimal_risk_reward(
        self,
        win_rate: float | None,
        avg_win_pct: float | None,
        avg_loss_pct: float | None,
        market_condition: MarketCondition,
    ) -> float:
        """Calculate optimal risk/reward ratio based on Kelly and market conditions."""

        base_rr = self.config.target_risk_reward_ratio

        if win_rate is not None and avg_win_pct is not None and avg_loss_pct is not None:
            if win_rate > 0.55:
                base_rr = min(base_rr, 2.5)
            elif win_rate < 0.45:
                base_rr = max(base_rr, 4.0)

        if market_condition == MarketCondition.HIGH_VOLATILITY:
            base_rr *= 1.3
        elif market_condition == MarketCondition.EXTREME_VOLATILITY:
            base_rr *= 1.5
        elif market_condition == MarketCondition.LOW_VOLATILITY:
            base_rr *= 0.8

        return np.clip(
            base_rr,
            self.config.min_risk_reward_ratio,
            self.config.max_risk_reward_ratio,
        )

    def _get_volatility_adjustment(self, market_condition: MarketCondition) -> float:
        """Get adjustment factor based on market condition."""
        adjustments = {
            MarketCondition.LOW_VOLATILITY: 0.7,
            MarketCondition.NORMAL: 1.0,
            MarketCondition.HIGH_VOLATILITY: 1.3,
            MarketCondition.EXTREME_VOLATILITY: 1.5,
            MarketCondition.TRENDING_UP: 1.1,
            MarketCondition.TRENDING_DOWN: 1.2,
            MarketCondition.RANGING: 0.9,
        }
        return adjustments.get(market_condition, 1.0)

    def _check_support_resistance_stop(
        self,
        entry_price: float,
        calculated_stop: float,
        is_long: bool,
    ) -> float | None:
        """Check if there's a better stop based on S/R levels."""

        if not self.config.use_support_resistance:
            return None

        threshold = self.config.support_resistance_threshold

        if is_long:
            for support in self._support_levels:
                if support < entry_price:
                    distance = abs(calculated_stop - support) / entry_price * 100
                    if distance < threshold:
                        return support * 0.995
        else:
            for resistance in self._resistance_levels:
                if resistance > entry_price:
                    distance = abs(calculated_stop - resistance) / entry_price * 100
                    if distance < threshold:
                        return resistance * 1.005

        return None

    def _check_support_resistance_tp(
        self,
        entry_price: float,
        calculated_tp: float,
        is_long: bool,
    ) -> float | None:
        """Check if there's a better TP based on S/R levels."""

        if not self.config.use_support_resistance:
            return None

        threshold = self.config.support_resistance_threshold

        if is_long:
            for resistance in self._resistance_levels:
                if resistance > entry_price:
                    distance = abs(calculated_tp - resistance) / entry_price * 100
                    if distance < threshold and resistance > entry_price:
                        return resistance * 0.995
        else:
            for support in self._support_levels:
                if support < entry_price:
                    distance = abs(calculated_tp - support) / entry_price * 100
                    if distance < threshold and support < entry_price:
                        return support * 1.005

        return None

    def _calculate_partial_tp_levels(
        self,
        entry_price: float,
        tp_price: float,
        stop_price: float,
        is_long: bool,
    ) -> list[tuple[float, float]]:
        """Calculate partial take-profit levels for scaling out."""
        levels = []

        tp_distance = abs(tp_price - entry_price)

        level_percents = [0.33, 0.66, 1.0]
        level_amounts = [0.3, 0.3, 0.4]

        for pct, amt in zip(level_percents, level_amounts, strict=False):
            if is_long:
                level_price = entry_price + tp_distance * pct
            else:
                level_price = entry_price - tp_distance * pct

            levels.append((round(level_price, 6), amt))

        return levels

    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss: StopLossResult,
        position_size: float,
        market_condition: MarketCondition,
    ) -> float:
        """Calculate recommended position size based on risk."""

        base_risk = self.config.max_risk_per_trade_pct

        if market_condition == MarketCondition.HIGH_VOLATILITY:
            base_risk *= 0.7
        elif market_condition == MarketCondition.EXTREME_VOLATILITY:
            base_risk *= 0.5
        elif market_condition == MarketCondition.LOW_VOLATILITY:
            base_risk *= 1.2

        if stop_loss.stop_percent > 0:
            position_pct = base_risk / stop_loss.stop_percent
            position_pct = min(position_pct, 1.0)
        else:
            position_pct = 0.1

        return round(position_pct * 100, 2)

    def _infer_market_condition(
        self,
        atr_percent: float | None,
    ) -> MarketCondition:
        """Infer market condition from ATR percentage."""
        if atr_percent is None:
            return MarketCondition.NORMAL

        if atr_percent < 1.5:
            return MarketCondition.LOW_VOLATILITY
        elif atr_percent < 4.0:
            return MarketCondition.NORMAL
        elif atr_percent < 8.0:
            return MarketCondition.HIGH_VOLATILITY
        else:
            return MarketCondition.EXTREME_VOLATILITY

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        atr: float | None = None,
        is_long: bool = True,
    ) -> StopLossResult:
        """Calculate trailing stop level.

        Args:
            entry_price: Original entry price
            current_price: Current market price
            highest_price: Highest price since entry (for longs)
            atr: Current ATR value
            is_long: True for long position

        Returns:
            StopLossResult with trailing stop level
        """

        profit_pct = (
            ((current_price - entry_price) / entry_price * 100)
            if is_long
            else ((entry_price - current_price) / entry_price * 100)
        )

        if profit_pct < self.config.trailing_activation_pct:
            if is_long:
                stop_price = entry_price * (1 - self.config.default_stop_loss_pct / 100)
            else:
                stop_price = entry_price * (1 + self.config.default_stop_loss_pct / 100)

            return StopLossResult(
                stop_price=round(stop_price, 6),
                stop_percent=round(self.config.default_stop_loss_pct, 4),
                stop_type=StopType.FIXED,
                risk_amount=round(abs(entry_price - stop_price), 6),
                risk_percent=round(self.config.default_stop_loss_pct, 4),
                reason="Trailing not yet activated",
            )

        if atr is not None:
            trailing_distance = atr * 1.5
        else:
            trailing_distance = highest_price * self.config.trailing_distance_pct / 100

        if is_long:
            stop_price = highest_price - trailing_distance
            stop_price = max(stop_price, entry_price)
        else:
            stop_price = highest_price + trailing_distance
            stop_price = min(stop_price, entry_price)

        stop_percent = abs(current_price - stop_price) / current_price * 100

        return StopLossResult(
            stop_price=round(stop_price, 6),
            stop_percent=round(stop_percent, 4),
            stop_type=StopType.TRAILING,
            risk_amount=round(abs(current_price - stop_price), 6),
            risk_percent=round(stop_percent, 4),
            reason=f"Trailing stop activated at {profit_pct:.1f}% profit",
        )

    def update_support_resistance(
        self,
        support_levels: list[float],
        resistance_levels: list[float],
    ) -> None:
        """Update support and resistance levels."""
        self._support_levels = sorted(support_levels)
        self._resistance_levels = sorted(resistance_levels)

    def record_trade_outcome(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        exit_price: float,
        pnl_pct: float,
    ) -> None:
        """Record trade outcome for learning."""
        self._historical_performance.append(
            {
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "hit_sl": exit_price <= stop_loss
                if exit_price < entry_price
                else exit_price >= stop_loss,
                "hit_tp": exit_price >= take_profit
                if exit_price > entry_price
                else exit_price <= take_profit,
            }
        )

    def get_performance_stats(self) -> dict[str, Any]:
        """Get statistics on SL/TP performance."""
        if not self._historical_performance:
            return {"trades": 0}

        trades = self._historical_performance
        total = len(trades)

        hit_sl = sum(1 for t in trades if t["hit_sl"])
        hit_tp = sum(1 for t in trades if t["hit_tp"])

        avg_pnl_at_sl = np.mean([t["pnl_pct"] for t in trades if t["hit_sl"]]) if hit_sl > 0 else 0
        avg_pnl_at_tp = np.mean([t["pnl_pct"] for t in trades if t["hit_tp"]]) if hit_tp > 0 else 0

        return {
            "total_trades": total,
            "hit_stop_loss": hit_sl,
            "hit_take_profit": hit_tp,
            "stopped_out_rate": hit_sl / total * 100 if total > 0 else 0,
            "tp_hit_rate": hit_tp / total * 100 if total > 0 else 0,
            "avg_pnl_at_sl": avg_pnl_at_sl,
            "avg_pnl_at_tp": avg_pnl_at_tp,
        }


def calculate_dynamic_sl_tp(
    entry_price: float,
    atr: float | None = None,
    atr_percent: float | None = None,
    risk_reward_ratio: float = 3.0,
    min_sl_pct: float = 3.0,
    max_sl_pct: float = 15.0,
    is_long: bool = True,
) -> tuple[float, float]:
    """Simple function to calculate SL/TP levels.

    Args:
        entry_price: Entry price
        atr: ATR value (optional)
        atr_percent: ATR as percentage (optional)
        risk_reward_ratio: Target R/R ratio
        min_sl_pct: Minimum stop loss percentage
        max_sl_pct: Maximum stop loss percentage
        is_long: True for long position

    Returns:
        Tuple of (stop_loss_price, take_profit_price)
    """
    calculator = DynamicSLTPCalculator()

    config = DynamicSLTPConfig(
        min_stop_loss_pct=min_sl_pct,
        max_stop_loss_pct=max_sl_pct,
        target_risk_reward_ratio=risk_reward_ratio,
    )
    calculator.config = config

    result = calculator.calculate(
        entry_price=entry_price,
        atr=atr,
        atr_percent=atr_percent,
        is_long=is_long,
    )

    return result.stop_loss.stop_price, result.take_profit.take_profit_price
