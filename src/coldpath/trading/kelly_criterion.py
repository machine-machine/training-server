"""
Kelly Criterion Position Sizing.

Optimal position sizing based on edge and win/loss ratios.
Uses half-Kelly (0.40 safety factor) for algo trading.

Formula: f* = (p*b - q) / b
Where:
- f* = optimal fraction of capital to bet
- p = probability of winning
- q = probability of losing (1 - p)
- b = win/loss ratio (avg_win / avg_loss)

Safety features:
- Half-Kelly safety factor (0.40) - industry standard for algo trading
- Maximum 15% position size per trade
- Minimum edge requirement (0.5%)
- Risk-adjusted sizing based on volatility
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""

    raw_kelly: float  # Raw Kelly fraction
    safe_kelly: float  # Kelly with safety factor
    position_fraction: float  # Final position as fraction of capital
    position_size: float  # Position size in absolute terms (SOL)
    edge: float  # Calculated edge
    is_valid: bool  # Whether sizing is valid (positive edge)
    reason: str  # Explanation of sizing decision

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "raw_kelly": self.raw_kelly,
            "safe_kelly": self.safe_kelly,
            "position_fraction": self.position_fraction,
            "position_size": self.position_size,
            "edge": self.edge,
            "is_valid": self.is_valid,
            "reason": self.reason,
        }


class KellyCriterion:
    """Kelly Criterion-based position sizing.

    Calculates optimal position sizes based on:
    - Win probability from signal generator
    - Historical average win/loss percentages
    - Current capital available

    Implements safety features:
    - Half-Kelly (40% of optimal) - algo trading standard
    - Maximum position cap (15%)
    - Minimum edge threshold (0.5%)
    - Volatility adjustment
    """

    def __init__(
        self,
        safety_factor: float = 0.50,  # TUNED 2026-02-23: Half-Kelly for balance (was 0.40)
        max_position_pct: float = 20.0,  # TUNED 2026-02-23: Allow larger positions (was 15%)
        min_edge_pct: float = 0.4,  # TUNED 2026-02-23: Slightly lower threshold (was 0.5%)
        min_win_prob: float = 0.40,
        max_win_prob: float = 0.90,
        default_avg_win_pct: float = (
            15.0
        ),  # TUNED 2026-02-23: Increased for memecoin upside (was 12%)
        default_avg_loss_pct: float = (
            8.0
        ),  # TUNED 2026-02-23: Tightened from 5% (better stop adherence)
    ):
        """Initialize Kelly Criterion calculator.

        Args:
            safety_factor: Fraction of Kelly to use (0.50 = half-Kelly)
            max_position_pct: Maximum position as % of capital
            min_edge_pct: Minimum required edge to take position
            min_win_prob: Floor for win probability
            max_win_prob: Ceiling for win probability
            default_avg_win_pct: Default average win percentage if not provided
            default_avg_loss_pct: Default average loss percentage if not provided
        """
        self.safety_factor = safety_factor
        self.max_position_pct = max_position_pct / 100.0
        self.min_edge_pct = min_edge_pct / 100.0
        self.min_win_prob = min_win_prob
        self.max_win_prob = max_win_prob
        self.default_avg_win_pct = default_avg_win_pct
        self.default_avg_loss_pct = default_avg_loss_pct

        # Historical tracking for adaptive sizing (deque auto-evicts in O(1))
        self._max_history: int = 100
        self._historical_wins: deque = deque(maxlen=self._max_history)
        self._historical_losses: deque = deque(maxlen=self._max_history)

    def calculate(
        self,
        win_prob: float,
        avg_win_pct: float | None = None,
        avg_loss_pct: float | None = None,
        capital: float = 1.0,
        volatility_factor: float | None = None,
    ) -> PositionSizeResult:
        """Calculate optimal position size.

        Args:
            win_prob: Probability of winning (0-1)
            avg_win_pct: Average win as percentage (e.g., 10 for 10%)
            avg_loss_pct: Average loss as percentage (e.g., 5 for 5%)
            capital: Available capital in SOL
            volatility_factor: Optional volatility multiplier (higher = smaller position)

        Returns:
            PositionSizeResult with sizing details
        """
        # Use defaults if not provided
        if avg_win_pct is None:
            avg_win_pct = self._get_historical_avg_win() or self.default_avg_win_pct
        if avg_loss_pct is None:
            avg_loss_pct = self._get_historical_avg_loss() or self.default_avg_loss_pct

        # Clamp win probability
        win_prob = np.clip(win_prob, self.min_win_prob, self.max_win_prob)

        # Convert percentages to decimals
        avg_win = avg_win_pct / 100.0
        avg_loss = avg_loss_pct / 100.0

        # Protect against division by zero
        if avg_loss <= 0:
            return PositionSizeResult(
                raw_kelly=0.0,
                safe_kelly=0.0,
                position_fraction=0.0,
                position_size=0.0,
                edge=0.0,
                is_valid=False,
                reason="Invalid avg_loss (zero or negative)",
            )

        # Calculate win/loss ratio (b)
        b = avg_win / avg_loss

        # Calculate edge: E[return] = p*avg_win - (1-p)*avg_loss
        edge = win_prob * avg_win - (1 - win_prob) * avg_loss

        # Check minimum edge
        if edge < self.min_edge_pct:
            return PositionSizeResult(
                raw_kelly=0.0,
                safe_kelly=0.0,
                position_fraction=0.0,
                position_size=0.0,
                edge=edge,
                is_valid=False,
                reason=f"Insufficient edge: {edge * 100:.2f}% < {self.min_edge_pct * 100:.2f}%",
            )

        # Calculate Kelly fraction: f* = (p*b - q) / b
        # Simplified: f* = (p*b - (1-p)) / b = p - (1-p)/b
        q = 1 - win_prob
        raw_kelly = (win_prob * b - q) / b

        # Apply safety factor (quarter-Kelly)
        safe_kelly = raw_kelly * self.safety_factor

        # Apply volatility adjustment if provided
        if volatility_factor is not None and volatility_factor > 0:
            # Higher volatility = smaller position
            # Fixed: was 1.0 / (1.0 + volatility_factor - 1.0) which simplifies to 1.0
            volatility_adjustment = 1.0 / (1.0 + volatility_factor)
            safe_kelly *= volatility_adjustment

        # Apply maximum position cap
        position_fraction = min(safe_kelly, self.max_position_pct)

        # Ensure non-negative
        position_fraction = max(0.0, position_fraction)

        # Calculate absolute position size
        position_size = position_fraction * capital

        return PositionSizeResult(
            raw_kelly=raw_kelly,
            safe_kelly=safe_kelly,
            position_fraction=position_fraction,
            position_size=position_size,
            edge=edge,
            is_valid=True,
            reason=f"Kelly sizing: {position_fraction * 100:.2f}% of capital",
        )

    def calculate_from_signal(
        self,
        signal_confidence: float,
        signal_expected_return: float,
        capital: float,
        base_win_prob: float = 0.55,
        volatility_factor: float | None = None,
    ) -> PositionSizeResult:
        """Calculate position size from signal generator output.

        Combines signal confidence with expected return to determine sizing.

        Args:
            signal_confidence: Confidence from signal generator (0-1)
            signal_expected_return: Expected return from signal generator (%)
            capital: Available capital
            base_win_prob: Base win probability for the strategy
            volatility_factor: Optional volatility adjustment

        Returns:
            PositionSizeResult with sizing details
        """
        # Adjust win probability based on signal confidence
        # Higher confidence = higher win probability adjustment
        confidence_adjustment = (signal_confidence - 0.5) * 0.3
        adjusted_win_prob = base_win_prob + confidence_adjustment

        # Determine avg_win/loss from expected return
        if signal_expected_return > 0:
            avg_win_pct = signal_expected_return * 1.5  # Optimistic scenario
            avg_loss_pct = abs(signal_expected_return) * 0.5  # Limited downside
        else:
            avg_win_pct = abs(signal_expected_return) * 0.5
            avg_loss_pct = abs(signal_expected_return) * 1.5

        return self.calculate(
            win_prob=adjusted_win_prob,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            capital=capital,
            volatility_factor=volatility_factor,
        )

    def calculate_with_drawdown_limit(
        self,
        win_prob: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        capital: float,
        max_drawdown_pct: float = 25.0,
    ) -> PositionSizeResult:
        """Calculate position size with drawdown constraint.

        Adjusts Kelly sizing to respect maximum drawdown limits.

        Args:
            win_prob: Probability of winning
            avg_win_pct: Average win percentage
            avg_loss_pct: Average loss percentage
            capital: Available capital
            max_drawdown_pct: Maximum acceptable drawdown

        Returns:
            PositionSizeResult with drawdown-constrained sizing
        """
        # First calculate standard Kelly
        result = self.calculate(
            win_prob=win_prob,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            capital=capital,
        )

        if not result.is_valid:
            return result

        # Estimate expected maximum drawdown for this Kelly fraction
        # Approximation: max_dd ~ f * loss_pct * drawdown_multiplier
        drawdown_multiplier = 3.0  # Empirical factor for consecutive losses
        estimated_max_dd = result.position_fraction * (avg_loss_pct / 100.0) * drawdown_multiplier

        # Scale down if exceeds limit
        max_dd_limit = max_drawdown_pct / 100.0
        if estimated_max_dd > max_dd_limit:
            scale_factor = max_dd_limit / estimated_max_dd
            adjusted_fraction = result.position_fraction * scale_factor
            adjusted_size = adjusted_fraction * capital

            return PositionSizeResult(
                raw_kelly=result.raw_kelly,
                safe_kelly=result.safe_kelly * scale_factor,
                position_fraction=adjusted_fraction,
                position_size=adjusted_size,
                edge=result.edge,
                is_valid=True,
                reason=f"Drawdown-limited to {max_drawdown_pct}% max DD",
            )

        return result

    def record_trade_result(self, return_pct: float) -> None:
        """Record a trade result for adaptive sizing.

        Args:
            return_pct: Trade return as percentage
        """
        if return_pct > 0:
            self._historical_wins.append(return_pct)
        else:
            self._historical_losses.append(abs(return_pct))

    def _get_historical_avg_win(self) -> float | None:
        """Get average win from history."""
        if len(self._historical_wins) >= 10:
            return float(np.mean(self._historical_wins))
        return None

    def _get_historical_avg_loss(self) -> float | None:
        """Get average loss from history."""
        if len(self._historical_losses) >= 10:
            return float(np.mean(self._historical_losses))
        return None

    def get_historical_stats(self) -> dict[str, Any]:
        """Get historical trading statistics.

        Returns:
            Dictionary with historical win/loss statistics
        """
        total_trades = len(self._historical_wins) + len(self._historical_losses)

        return {
            "total_trades": total_trades,
            "wins": len(self._historical_wins),
            "losses": len(self._historical_losses),
            "win_rate": len(self._historical_wins) / total_trades if total_trades > 0 else 0,
            "avg_win_pct": np.mean(self._historical_wins) if self._historical_wins else 0,
            "avg_loss_pct": np.mean(self._historical_losses) if self._historical_losses else 0,
            "profit_factor": (
                sum(self._historical_wins) / sum(self._historical_losses)
                if self._historical_losses and sum(self._historical_losses) > 0
                else (999.0 if self._historical_wins else 0.0)
            ),
        }

    def reset_history(self) -> None:
        """Reset historical tracking."""
        self._historical_wins.clear()
        self._historical_losses.clear()


def calculate_optimal_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Simple Kelly calculation (full Kelly).

    Args:
        win_rate: Win rate (0-1)
        avg_win: Average win amount
        avg_loss: Average loss amount

    Returns:
        Optimal Kelly fraction
    """
    if avg_loss <= 0:
        return 0.0

    b = avg_win / avg_loss
    q = 1 - win_rate

    kelly = (win_rate * b - q) / b

    return max(0.0, kelly)


def calculate_geometric_kelly(
    win_rate: float,
    win_return: float,
    loss_return: float,
) -> float:
    """Kelly for geometric growth (multiplicative returns).

    Better for compound growth scenarios.

    Args:
        win_rate: Win rate (0-1)
        win_return: Return on win (e.g., 0.1 for 10% gain)
        loss_return: Return on loss (e.g., -0.05 for 5% loss)

    Returns:
        Optimal Kelly fraction for geometric growth
    """
    # For geometric growth: f = (p*ln(1+R_w) + q*ln(1+R_l)) / ...
    # Simplified approximation for small returns:
    p = win_rate
    q = 1 - win_rate

    if loss_return >= 0:
        return 0.0

    # Use standard Kelly as approximation
    b = abs(win_return / loss_return)
    kelly = (p * b - q) / b

    return max(0.0, kelly)


@dataclass
class KellyConfig:
    """Configuration for Kelly-based position sizing."""

    safety_factor: float = 0.50  # Half-Kelly (was 0.40)
    max_position_pct: float = 20.0  # Allow larger positions (was 15.0)
    min_edge_pct: float = 0.4  # Slightly lower threshold (was 0.5)
    max_drawdown_pct: float = 25.0
    target_win_rate: float = 0.55
    target_sharpe: float = 2.0

    def create_calculator(self) -> KellyCriterion:
        """Create a KellyCriterion calculator with this config."""
        return KellyCriterion(
            safety_factor=self.safety_factor,
            max_position_pct=self.max_position_pct,
            min_edge_pct=self.min_edge_pct,
        )
