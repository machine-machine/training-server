"""
Perpetual Futures Backtesting Engine

This module provides backtesting capabilities for perpetual futures strategies,
including realistic simulation of funding payments, liquidations, and mark price dynamics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


class MarginType(Enum):
    CROSS = "cross"
    ISOLATED = "isolated"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    TAKE_PROFIT = "take_profit"


@dataclass
class PerpPosition:
    """Simulated perpetual position."""
    market: str
    side: PositionSide
    size: float
    entry_price: float
    leverage: int
    margin_type: MarginType
    margin: float
    open_time: datetime
    liquidation_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    accumulated_funding: float = 0.0
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None

    def update_pnl(self, mark_price: float) -> float:
        """Update unrealized P&L based on current mark price."""
        price_diff = mark_price - self.entry_price
        if self.side == PositionSide.SHORT:
            price_diff = -price_diff
        self.unrealized_pnl = self.size * price_diff
        return self.unrealized_pnl

    def margin_ratio(self, mark_price: float) -> float:
        """Calculate current margin ratio."""
        notional = self.size * mark_price
        if notional == 0:
            return 1.0
        return self.margin / notional

    def calculate_liquidation_price(self, mmr: float = 0.005) -> float:
        """Calculate liquidation price given maintenance margin rate."""
        lev = self.leverage
        if self.side == PositionSide.LONG:
            return self.entry_price * (1 - (1 / lev) + mmr)
        else:
            return self.entry_price * (1 + (1 / lev) - mmr)


@dataclass
class PerpOrder:
    """Order for backtesting."""
    market: str
    side: PositionSide
    order_type: OrderType
    size: float
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    leverage: int = 1
    reduce_only: bool = False
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None


@dataclass
class PerpTrade:
    """Executed trade record."""
    timestamp: datetime
    market: str
    side: PositionSide
    size: float
    price: float
    fee: float
    pnl: float
    is_liquidation: bool = False


@dataclass
class FundingPayment:
    """Funding payment record."""
    timestamp: datetime
    market: str
    rate: float
    amount: float
    position_size: float


@dataclass
class BacktestConfig:
    """Configuration for perpetual backtesting."""
    # Initial capital
    initial_capital: float = 10000.0

    # Risk parameters
    max_leverage: int = 20
    max_position_size_pct: float = 0.5  # Max % of capital per position
    min_margin_ratio: float = 0.05  # 5%

    # Fees
    taker_fee_bps: float = 5.0  # 0.05%
    maker_fee_bps: float = 2.0  # 0.02%

    # Funding
    funding_interval_hours: int = 8

    # Liquidation
    maintenance_margin_ratio: float = 0.005  # 0.5%
    liquidation_fee_pct: float = 0.5  # 0.5% of position

    # Slippage simulation
    slippage_bps: float = 5.0  # 0.05% slippage for market orders


@dataclass
class BacktestResult:
    """Results from backtesting."""
    total_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    win_rate: float
    total_trades: int
    total_liquidations: int
    total_funding_paid: float
    total_fees_paid: float
    equity_curve: List[Tuple[datetime, float]]
    trades: List[PerpTrade]
    funding_payments: List[FundingPayment]


class PerpBacktestEngine:
    """
    Perpetual futures backtesting engine.

    Simulates perpetual trading with realistic:
    - Funding payments every 8 hours
    - Liquidation mechanics
    - Slippage and fees
    - Mark price dynamics
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        """Reset engine state."""
        self.capital = self.config.initial_capital
        self.positions: Dict[str, PerpPosition] = {}
        self.trades: List[PerpTrade] = []
        self.funding_payments: List[FundingPayment] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.peak_equity = self.config.initial_capital
        self.max_drawdown = 0.0
        self.current_time: Optional[datetime] = None

    def run_backtest(
        self,
        price_data: pd.DataFrame,
        funding_data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            price_data: DataFrame with columns [timestamp, market, open, high, low, close, volume]
            funding_data: DataFrame with columns [timestamp, market, funding_rate]
            signals: DataFrame with columns [timestamp, market, signal, size, leverage, tp, sl]
                     signal: 1 for long, -1 for short, 0 for close

        Returns:
            BacktestResult with performance metrics and trade history
        """
        self.reset()

        # Sort all data by timestamp
        price_data = price_data.sort_values("timestamp")

        # Track last funding time per market
        last_funding: Dict[str, datetime] = {}

        for idx, row in price_data.iterrows():
            self.current_time = row["timestamp"]
            market = row["market"]
            price = row["close"]
            high = row["high"]
            low = row["low"]

            # 1. Check for liquidations
            self._check_liquidations(market, low if market in self.positions and
                                     self.positions[market].side == PositionSide.LONG else high)

            # 2. Process funding (every 8 hours)
            self._process_funding(market, price, funding_data, last_funding)

            # 3. Check stop loss / take profit
            self._check_tp_sl(market, high, low)

            # 4. Process signals
            market_signals = signals[
                (signals["timestamp"] == self.current_time) &
                (signals["market"] == market)
            ]
            for _, signal in market_signals.iterrows():
                self._process_signal(signal, price)

            # 5. Update equity
            self._update_equity(price_data, idx)

        # Close all remaining positions at last price
        self._close_all_positions(price_data)

        return self._compile_results()

    def _process_signal(self, signal: pd.Series, price: float):
        """Process a trading signal."""
        market = signal["market"]
        sig = signal["signal"]
        size = signal.get("size", 0.1)
        leverage = signal.get("leverage", 1)
        tp = signal.get("tp")
        sl = signal.get("sl")

        if sig == 0:
            # Close position
            if market in self.positions:
                self._close_position(market, price)
        elif sig == 1:
            # Long
            self._open_position(
                market, PositionSide.LONG, size, price, leverage, tp, sl
            )
        elif sig == -1:
            # Short
            self._open_position(
                market, PositionSide.SHORT, size, price, leverage, tp, sl
            )

    def _open_position(
        self,
        market: str,
        side: PositionSide,
        size: float,
        price: float,
        leverage: int,
        take_profit: Optional[float],
        stop_loss: Optional[float],
    ):
        """Open a new position."""
        # Apply slippage
        slippage = price * (self.config.slippage_bps / 10000)
        entry_price = price + slippage if side == PositionSide.LONG else price - slippage

        # Calculate required margin
        notional = size * entry_price
        margin = notional / leverage

        # Check capital
        if margin > self.capital * self.config.max_position_size_pct:
            logger.warning(f"Position size exceeds max: {margin} > {self.capital * self.config.max_position_size_pct}")
            return

        # Calculate fee
        fee = notional * (self.config.taker_fee_bps / 10000)

        if margin + fee > self.capital:
            logger.warning(f"Insufficient capital for position: {margin + fee} > {self.capital}")
            return

        # Deduct margin and fee
        self.capital -= (margin + fee)

        # Create position
        position = PerpPosition(
            market=market,
            side=side,
            size=size,
            entry_price=entry_price,
            leverage=leverage,
            margin_type=MarginType.ISOLATED,
            margin=margin,
            open_time=self.current_time,
            take_profit=take_profit,
            stop_loss=stop_loss,
        )
        position.liquidation_price = position.calculate_liquidation_price(
            self.config.maintenance_margin_ratio
        )

        self.positions[market] = position

        # Record trade
        self.trades.append(PerpTrade(
            timestamp=self.current_time,
            market=market,
            side=side,
            size=size,
            price=entry_price,
            fee=fee,
            pnl=0.0,
        ))

        logger.debug(f"Opened {side.value} {size} {market} @ {entry_price}")

    def _close_position(self, market: str, price: float, is_liquidation: bool = False):
        """Close an existing position."""
        if market not in self.positions:
            return

        position = self.positions[market]

        # Apply slippage
        slippage = price * (self.config.slippage_bps / 10000)
        exit_price = price - slippage if position.side == PositionSide.LONG else price + slippage

        # Calculate P&L
        price_diff = exit_price - position.entry_price
        if position.side == PositionSide.SHORT:
            price_diff = -price_diff
        pnl = position.size * price_diff

        # Calculate fee
        notional = position.size * exit_price
        fee = notional * (self.config.taker_fee_bps / 10000)

        # Liquidation fee
        liq_fee = 0.0
        if is_liquidation:
            liq_fee = notional * (self.config.liquidation_fee_pct / 100)
            pnl -= liq_fee

        # Return margin and P&L
        total_return = position.margin + pnl + position.accumulated_funding - fee
        self.capital += max(0, total_return)  # Can't go below 0

        # Record trade
        self.trades.append(PerpTrade(
            timestamp=self.current_time,
            market=market,
            side=position.side,
            size=position.size,
            price=exit_price,
            fee=fee + liq_fee,
            pnl=pnl + position.accumulated_funding,
            is_liquidation=is_liquidation,
        ))

        logger.debug(
            f"Closed {position.side.value} {position.size} {market} @ {exit_price}, "
            f"PnL: {pnl:.2f}, Funding: {position.accumulated_funding:.2f}"
        )

        del self.positions[market]

    def _check_liquidations(self, market: str, extreme_price: float):
        """Check if position should be liquidated."""
        if market not in self.positions:
            return

        position = self.positions[market]

        # Check if liquidation price was breached
        if position.side == PositionSide.LONG:
            if extreme_price <= position.liquidation_price:
                logger.info(f"LIQUIDATION: {market} {position.side.value} @ {position.liquidation_price}")
                self._close_position(market, position.liquidation_price, is_liquidation=True)
        else:
            if extreme_price >= position.liquidation_price:
                logger.info(f"LIQUIDATION: {market} {position.side.value} @ {position.liquidation_price}")
                self._close_position(market, position.liquidation_price, is_liquidation=True)

    def _check_tp_sl(self, market: str, high: float, low: float):
        """Check take profit and stop loss."""
        if market not in self.positions:
            return

        position = self.positions[market]

        if position.side == PositionSide.LONG:
            # TP hit on high
            if position.take_profit and high >= position.take_profit:
                self._close_position(market, position.take_profit)
                return
            # SL hit on low
            if position.stop_loss and low <= position.stop_loss:
                self._close_position(market, position.stop_loss)
        else:
            # TP hit on low
            if position.take_profit and low <= position.take_profit:
                self._close_position(market, position.take_profit)
                return
            # SL hit on high
            if position.stop_loss and high >= position.stop_loss:
                self._close_position(market, position.stop_loss)

    def _process_funding(
        self,
        market: str,
        price: float,
        funding_data: pd.DataFrame,
        last_funding: Dict[str, datetime],
    ):
        """Process funding payments."""
        if market not in self.positions:
            return

        position = self.positions[market]

        # Check if it's time for funding
        if market not in last_funding:
            last_funding[market] = self.current_time
            return

        hours_since_last = (self.current_time - last_funding[market]).total_seconds() / 3600
        if hours_since_last < self.config.funding_interval_hours:
            return

        # Get funding rate
        funding_row = funding_data[
            (funding_data["market"] == market) &
            (funding_data["timestamp"] <= self.current_time)
        ].tail(1)

        if funding_row.empty:
            return

        funding_rate = funding_row.iloc[0]["funding_rate"]

        # Calculate payment
        notional = position.size * price
        # Longs pay positive funding, shorts pay negative
        if position.side == PositionSide.LONG:
            payment = -notional * funding_rate
        else:
            payment = notional * funding_rate

        position.accumulated_funding += payment
        position.margin += payment  # Adjust margin

        # Record payment
        self.funding_payments.append(FundingPayment(
            timestamp=self.current_time,
            market=market,
            rate=funding_rate,
            amount=payment,
            position_size=position.size,
        ))

        last_funding[market] = self.current_time

        # Check if funding depleted margin
        if position.margin <= 0:
            logger.info(f"Margin depleted by funding: {market}")
            self._close_position(market, price, is_liquidation=True)

    def _update_equity(self, price_data: pd.DataFrame, idx: int):
        """Update equity curve and drawdown."""
        # Calculate total equity
        equity = self.capital

        for market, position in self.positions.items():
            # Get current price
            current_price = price_data.iloc[idx]["close"]
            position.update_pnl(current_price)
            equity += position.margin + position.unrealized_pnl + position.accumulated_funding

        self.equity_curve.append((self.current_time, equity))

        # Update drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = (self.peak_equity - equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def _close_all_positions(self, price_data: pd.DataFrame):
        """Close all remaining positions at last available price."""
        for market in list(self.positions.keys()):
            last_price = price_data[price_data["market"] == market].iloc[-1]["close"]
            self._close_position(market, last_price)

    def _compile_results(self) -> BacktestResult:
        """Compile backtest results."""
        total_pnl = self.capital - self.config.initial_capital
        total_return_pct = (total_pnl / self.config.initial_capital) * 100

        # Calculate statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0

        total_liquidations = sum(1 for t in self.trades if t.is_liquidation)
        total_funding = sum(p.amount for p in self.funding_payments)
        total_fees = sum(t.fee for t in self.trades)

        # Calculate Sharpe ratio from equity curve
        sharpe_ratio = self._calculate_sharpe()

        return BacktestResult(
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            max_drawdown_pct=self.max_drawdown * 100,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            total_trades=len(self.trades),
            total_liquidations=total_liquidations,
            total_funding_paid=total_funding,
            total_fees_paid=total_fees,
            equity_curve=self.equity_curve,
            trades=self.trades,
            funding_payments=self.funding_payments,
        )

    def _calculate_sharpe(self, risk_free_rate: float = 0.02) -> Optional[float]:
        """Calculate annualized Sharpe ratio from equity curve."""
        if len(self.equity_curve) < 2:
            return None

        # Calculate returns
        equities = [e[1] for e in self.equity_curve]
        returns = np.diff(equities) / equities[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return None

        # Annualize (assuming daily data)
        mean_return = np.mean(returns) * 365
        std_return = np.std(returns) * np.sqrt(365)

        return (mean_return - risk_free_rate) / std_return
