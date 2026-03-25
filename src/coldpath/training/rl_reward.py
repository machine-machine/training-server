"""
Sharpe-Based Reward Function for RL Trading Agents.

Provides risk-adjusted reward signals for PPO and hierarchical RL:
- Sharpe-based: (return - risk_free) / rolling_volatility
- Drawdown penalty: -2x for new drawdown highs
- Consistency bonus for streaks of positive risk-adjusted returns
- Fee-aware: deducts estimated transaction costs from returns
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for the Sharpe-based reward function."""

    # Risk-free rate (annualized, e.g., 5% = 0.05)
    risk_free_rate: float = 0.05

    # Rolling window for volatility estimation (in steps)
    volatility_window: int = 100

    # Drawdown penalty multiplier
    drawdown_penalty: float = 2.0

    # Consistency bonus for consecutive positive returns
    consistency_bonus: float = 0.1
    consistency_streak_min: int = 3

    # Fee drag (basis points per trade)
    fee_bps: float = 30.0

    # Reward clipping bounds
    clip_min: float = -5.0
    clip_max: float = 5.0

    # Discount factor for temporal difference
    gamma: float = 0.99

    # GAE lambda for advantage estimation
    gae_lambda: float = 0.95

    # Annualization factor (trading steps per year, ~252 days * ~24 steps/day)
    annualization_factor: float = 6048.0

    # Minimum volatility floor to avoid division by zero
    min_volatility: float = 1e-6


class SharpeBasedReward:
    """Compute Sharpe-based risk-adjusted rewards for RL trading agents.

    The reward signal combines:
    1. Risk-adjusted return (excess return / volatility)
    2. Drawdown penalty (penalizes new equity lows)
    3. Consistency bonus (rewards streaks of profitable trades)
    4. Fee deduction (realistic cost modeling)
    """

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()

        # Rolling return history
        self._returns: deque = deque(maxlen=self.config.volatility_window)

        # Drawdown tracking
        self._peak_equity: float = 1.0
        self._current_equity: float = 1.0

        # Consistency tracking
        self._positive_streak: int = 0

        # Statistics
        self._total_rewards: int = 0
        self._cumulative_reward: float = 0.0

    def compute_reward(
        self,
        pnl_pct: float,
        position_size: float = 1.0,
        traded: bool = True,
        holding_period_steps: int = 1,
    ) -> float:
        """Compute risk-adjusted reward for a single step.

        Args:
            pnl_pct: PnL as percentage (e.g., 2.5 for +2.5%)
            position_size: Position size as fraction of capital [0, 1]
            traded: Whether a trade was executed (fees only apply if True)
            holding_period_steps: How many steps the position was held

        Returns:
            Risk-adjusted reward (clipped)
        """
        # Scale PnL by position size
        scaled_pnl = pnl_pct * position_size / 100.0

        # Deduct fees if traded
        if traded:
            fee_drag = self.config.fee_bps / 10000.0 * 2  # Entry + exit
            scaled_pnl -= fee_drag

        # Update equity curve
        self._current_equity *= 1.0 + scaled_pnl

        # Track returns
        self._returns.append(scaled_pnl)

        # Compute components
        sharpe_component = self._compute_sharpe_component(scaled_pnl)
        drawdown_component = self._compute_drawdown_penalty()
        consistency_component = self._compute_consistency_bonus(scaled_pnl)

        # Combine
        reward = sharpe_component + drawdown_component + consistency_component

        # Clip
        reward = np.clip(reward, self.config.clip_min, self.config.clip_max)

        # Update stats
        self._total_rewards += 1
        self._cumulative_reward += reward

        return float(reward)

    def _compute_sharpe_component(self, step_return: float) -> float:
        """Compute Sharpe-based reward component."""
        if len(self._returns) < 2:
            return step_return * 10.0  # Early phase: use raw return scaled up

        # Per-step risk-free rate
        rf_per_step = self.config.risk_free_rate / self.config.annualization_factor

        # Excess return
        excess_return = step_return - rf_per_step

        # Rolling volatility
        volatility = max(
            np.std(list(self._returns)),
            self.config.min_volatility,
        )

        # Sharpe component (annualized direction, per-step magnitude)
        sharpe = excess_return / volatility

        return float(sharpe)

    def _compute_drawdown_penalty(self) -> float:
        """Compute drawdown penalty for new equity lows."""
        # Update peak
        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity
            return 0.0  # No penalty at new highs

        # Drawdown from peak
        drawdown = (self._peak_equity - self._current_equity) / self._peak_equity

        # Only penalize if drawdown is increasing
        penalty = -self.config.drawdown_penalty * drawdown

        return float(penalty)

    def _compute_consistency_bonus(self, step_return: float) -> float:
        """Compute bonus for consistent positive returns."""
        if step_return > 0:
            self._positive_streak += 1
        else:
            self._positive_streak = 0

        if self._positive_streak >= self.config.consistency_streak_min:
            # Logarithmic bonus to avoid runaway rewards
            streak_bonus = self.config.consistency_bonus * np.log1p(
                self._positive_streak - self.config.consistency_streak_min
            )
            return float(streak_bonus)

        return 0.0

    def compute_episode_reward(
        self,
        returns: list[float],
        positions: list[float] | None = None,
        traded_flags: list[bool] | None = None,
    ) -> dict[str, float]:
        """Compute reward metrics for an entire episode.

        Args:
            returns: List of per-step PnL percentages
            positions: Optional position sizes per step
            traded_flags: Optional flags for whether each step involved a trade

        Returns:
            Dictionary with episode reward metrics
        """
        self.reset()

        positions = positions or [1.0] * len(returns)
        traded_flags = traded_flags or [True] * len(returns)

        rewards = []
        for ret, pos, traded in zip(returns, positions, traded_flags, strict=False):
            r = self.compute_reward(ret, pos, traded)
            rewards.append(r)

        total_return = self._current_equity - 1.0
        max_drawdown = self._compute_max_drawdown()

        return {
            "total_reward": sum(rewards),
            "mean_reward": np.mean(rewards) if rewards else 0.0,
            "std_reward": np.std(rewards) if rewards else 0.0,
            "total_return_pct": total_return * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_ratio": self._compute_sharpe_ratio(),
            "sortino_ratio": self._compute_sortino_ratio(),
            "calmar_ratio": abs(total_return / max_drawdown) if max_drawdown > 0 else 0.0,
            "n_steps": len(rewards),
        }

    def _compute_max_drawdown(self) -> float:
        """Compute max drawdown from return history."""
        if not self._returns:
            return 0.0

        equity = 1.0
        peak = 1.0
        max_dd = 0.0

        for r in self._returns:
            equity *= 1.0 + r
            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

        return max_dd

    def _compute_sharpe_ratio(self) -> float:
        """Compute annualized Sharpe ratio from return history."""
        if len(self._returns) < 2:
            return 0.0

        returns = np.array(list(self._returns))
        rf_per_step = self.config.risk_free_rate / self.config.annualization_factor
        excess = returns - rf_per_step

        if np.std(excess) < self.config.min_volatility:
            return 0.0

        sharpe = np.mean(excess) / np.std(excess)
        return float(sharpe * np.sqrt(self.config.annualization_factor))

    def _compute_sortino_ratio(self) -> float:
        """Compute annualized Sortino ratio (downside deviation only)."""
        if len(self._returns) < 2:
            return 0.0

        returns = np.array(list(self._returns))
        rf_per_step = self.config.risk_free_rate / self.config.annualization_factor
        excess = returns - rf_per_step

        downside = excess[excess < 0]
        if len(downside) < 1:
            return float(np.mean(excess) * np.sqrt(self.config.annualization_factor) * 10)

        downside_std = np.std(downside)
        if downside_std < self.config.min_volatility:
            return 0.0

        sortino = np.mean(excess) / downside_std
        return float(sortino * np.sqrt(self.config.annualization_factor))

    def compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        next_value: float = 0.0,
        dones: list[bool] | None = None,
    ) -> np.ndarray:
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Per-step rewards
            values: Value function estimates per step
            next_value: Bootstrap value for last step
            dones: Episode termination flags

        Returns:
            Array of advantages for each step
        """
        n = len(rewards)
        dones = dones or [False] * n
        advantages = np.zeros(n)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # Mask for episode boundaries
            mask = 0.0 if dones[t] else 1.0

            delta = rewards[t] + self.config.gamma * next_val * mask - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * mask * last_gae
            advantages[t] = last_gae

        return advantages

    def reset(self):
        """Reset reward state for new episode."""
        self._returns.clear()
        self._peak_equity = 1.0
        self._current_equity = 1.0
        self._positive_streak = 0

    def get_stats(self) -> dict[str, Any]:
        """Get reward statistics."""
        return {
            "total_rewards": self._total_rewards,
            "cumulative_reward": self._cumulative_reward,
            "mean_reward": self._cumulative_reward / max(1, self._total_rewards),
            "current_equity": self._current_equity,
            "peak_equity": self._peak_equity,
            "current_drawdown": (
                (self._peak_equity - self._current_equity) / self._peak_equity
                if self._peak_equity > 0
                else 0
            ),
            "positive_streak": self._positive_streak,
            "volatility_window_size": len(self._returns),
        }
