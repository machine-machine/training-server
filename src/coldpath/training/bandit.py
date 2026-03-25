"""
Multi-Armed Bandit trainer for slippage tolerance optimization.

Uses Upper Confidence Bound (UCB) algorithm to select optimal slippage settings.
Supports Thompson Sampling as an alternative exploration strategy.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..storage import DatabaseManager

logger = logging.getLogger(__name__)


class ExplorationStrategy(Enum):
    """Bandit exploration strategy."""

    UCB = "ucb"  # Upper Confidence Bound
    THOMPSON = "thompson"  # Thompson Sampling
    EPSILON_GREEDY = "epsilon_greedy"


@dataclass
class BanditArm:
    """A single arm in the bandit with full statistics tracking."""

    name: str
    slippage_bps: int
    count: int = 0
    total_reward: float = 0.0
    sum_squared_reward: float = 0.0
    rewards: list[float] = field(default_factory=list)
    recent_rewards: list[float] = field(default_factory=list)  # Last N rewards
    max_recent: int = 100

    # Thompson Sampling priors (Beta distribution for bounded rewards)
    alpha: float = 1.0  # Prior successes
    beta: float = 1.0  # Prior failures

    @property
    def mean_reward(self) -> float:
        """Mean reward across all pulls."""
        if self.count == 0:
            return 0.0
        return self.total_reward / self.count

    @property
    def variance(self) -> float:
        """Variance of rewards."""
        if self.count < 2:
            return 0.0
        mean = self.mean_reward
        return (self.sum_squared_reward / self.count) - (mean * mean)

    @property
    def std_dev(self) -> float:
        """Standard deviation of rewards."""
        return np.sqrt(max(0, self.variance))

    @property
    def recent_mean(self) -> float:
        """Mean of recent rewards (more responsive to changes)."""
        if not self.recent_rewards:
            return self.mean_reward
        return np.mean(self.recent_rewards)

    def update(self, reward: float, success: bool = None):
        """Update arm with observed reward.

        Args:
            reward: The observed reward value.
            success: For Thompson Sampling, whether this was a 'success'.
                     If None, determined by reward > 0.
        """
        self.count += 1
        self.total_reward += reward
        self.sum_squared_reward += reward * reward
        self.rewards.append(reward)

        # Maintain recent rewards window
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.max_recent:
            self.recent_rewards.pop(0)

        # Update Thompson Sampling priors
        if success is None:
            success = reward > 0
        if success:
            self.alpha += 1
        else:
            self.beta += 1

    def sample_thompson(self) -> float:
        """Sample from posterior for Thompson Sampling."""
        return np.random.beta(self.alpha, self.beta)

    def to_dict(self) -> dict[str, Any]:
        """Export arm state."""
        return {
            "name": self.name,
            "slippage_bps": self.slippage_bps,
            "count": self.count,
            "total_reward": self.total_reward,
            "mean_reward": self.mean_reward,
            "variance": self.variance,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BanditArm":
        """Create arm from saved state."""
        arm = cls(
            name=data["name"],
            slippage_bps=data["slippage_bps"],
            count=data.get("count", 0),
            total_reward=data.get("total_reward", 0.0),
        )
        arm.alpha = data.get("alpha", 1.0)
        arm.beta = data.get("beta", 1.0)
        # Reconstruct sum_squared_reward from variance if available
        if "variance" in data and arm.count > 0:
            mean = arm.mean_reward
            arm.sum_squared_reward = (data["variance"] + mean * mean) * arm.count
        return arm


class BanditTrainer:
    """UCB-based multi-armed bandit for slippage optimization.

    Learns optimal slippage tolerance by balancing exploration and exploitation.
    Supports multiple exploration strategies and contextual features.
    """

    # Default arm configurations
    DEFAULT_ARMS = {
        "conservative": 50,  # 0.5% - for high liquidity pools
        "low": 100,  # 1%
        "medium": 300,  # 3%
        "high": 500,  # 5%
        "aggressive": 1000,  # 10% - for low liquidity/volatile
    }

    def __init__(
        self,
        db: Optional["DatabaseManager"] = None,
        exploration_param: float = 2.0,
        strategy: ExplorationStrategy = ExplorationStrategy.UCB,
        epsilon: float = 0.1,  # For epsilon-greedy
        reward_decay: float = 0.99,  # For non-stationary environments
    ):
        self.db = db
        self.exploration_param = exploration_param
        self.strategy = strategy
        self.epsilon = epsilon
        self.reward_decay = reward_decay

        # Initialize arms
        self.arms: dict[str, BanditArm] = {
            name: BanditArm(name=name, slippage_bps=bps) for name, bps in self.DEFAULT_ARMS.items()
        }
        self.total_pulls = 0

        # Training metrics
        self.training_history: list[dict[str, Any]] = []

    def select_arm(self, context: dict[str, Any] | None = None) -> BanditArm:
        """Select an arm using the configured strategy.

        Args:
            context: Optional contextual features (liquidity, volatility, etc.)

        Returns:
            Selected bandit arm.
        """
        self.total_pulls += 1

        # If any arm hasn't been pulled, select it (exploration)
        # Mark arm as pulled so next call explores a different arm
        for arm in self.arms.values():
            if arm.count == 0:
                arm.count = 1  # Mark as explored (will be updated with real count on reward)
                return arm

        if self.strategy == ExplorationStrategy.UCB:
            return self._select_ucb()
        elif self.strategy == ExplorationStrategy.THOMPSON:
            return self._select_thompson()
        elif self.strategy == ExplorationStrategy.EPSILON_GREEDY:
            return self._select_epsilon_greedy()
        else:
            return self._select_ucb()

    def _select_ucb(self) -> BanditArm:
        """Select using Upper Confidence Bound."""
        best_arm = None
        best_score = float("-inf")

        for arm in self.arms.values():
            exploitation = arm.mean_reward
            # Guard against division by zero when arm.count == 0
            if arm.count > 0:
                exploration = self.exploration_param * np.sqrt(np.log(self.total_pulls) / arm.count)
            else:
                # Unexplored arm gets maximum exploration bonus
                exploration = float("inf")
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_arm = arm

        return best_arm

    def _select_thompson(self) -> BanditArm:
        """Select using Thompson Sampling."""
        samples = {name: arm.sample_thompson() for name, arm in self.arms.items()}
        best_arm_name = max(samples, key=samples.get)
        return self.arms[best_arm_name]

    def _select_epsilon_greedy(self) -> BanditArm:
        """Select using epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            # Explore: random arm
            return np.random.choice(list(self.arms.values()))
        else:
            # Exploit: best arm
            return max(self.arms.values(), key=lambda a: a.mean_reward)

    def update(self, arm_name: str, reward: float, success: bool = None):
        """Update arm with observed reward."""
        if arm_name in self.arms:
            self.arms[arm_name].update(reward, success)

    def get_recommendation(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Get the current recommended slippage setting.

        Args:
            context: Optional context for contextual recommendations.

        Returns:
            Dictionary with recommendation details.
        """
        # Check if any arm has data (either from select_arm or direct update)
        total_arm_pulls = sum(arm.count for arm in self.arms.values())
        if total_arm_pulls == 0:
            return {
                "arm": "medium",
                "slippage_bps": 300,
                "confidence": 0.0,
                "reason": "No training data yet",
            }

        # Find best arm by mean reward
        best_arm = max(self.arms.values(), key=lambda a: a.mean_reward)

        # Calculate confidence based on pull count and variance
        pull_confidence = min(1.0, best_arm.count / 100)
        variance_penalty = 1.0 / (1.0 + best_arm.std_dev) if best_arm.std_dev > 0 else 1.0
        confidence = pull_confidence * variance_penalty

        # Calculate UCB gap (how much better is best arm)
        if len(self.arms) > 1:
            second_best = sorted(self.arms.values(), key=lambda a: a.mean_reward, reverse=True)[1]
            gap = best_arm.mean_reward - second_best.mean_reward
        else:
            gap = 0

        return {
            "arm": best_arm.name,
            "slippage_bps": best_arm.slippage_bps,
            "confidence": confidence,
            "mean_reward": best_arm.mean_reward,
            "recent_mean": best_arm.recent_mean,
            "pull_count": best_arm.count,
            "std_dev": best_arm.std_dev,
            "gap_to_second": gap,
        }

    def get_arm_weights(self) -> dict[str, float]:
        """Get normalized weights for each arm based on softmax of rewards."""
        rewards = np.array([arm.mean_reward for arm in self.arms.values()])

        # Handle case where all rewards are zero or negative
        if rewards.max() <= 0:
            return {name: 1.0 / len(self.arms) for name in self.arms}

        # Softmax with temperature
        temperature = 1.0
        exp_rewards = np.exp((rewards - rewards.max()) / temperature)
        softmax = exp_rewards / exp_rewards.sum()

        return {
            name: float(weight) for name, weight in zip(self.arms.keys(), softmax, strict=False)
        }

    def calculate_reward(self, trade: dict[str, Any]) -> float:
        """Calculate reward for a trade.

        Reward function balances:
        - PnL (primary objective)
        - Fill rate (did the trade execute?)
        - Slippage efficiency (actual vs quoted)

        Args:
            trade: Trade data dictionary.

        Returns:
            Reward value (higher is better).
        """
        # Extract trade data
        pnl_pct = trade.get("pnl_pct", 0)
        included = trade.get("included", True)
        realized_slippage = trade.get("slippage_bps", 0)
        quoted_slippage = trade.get("quoted_slippage_bps", 0)

        # Base reward from PnL
        # Scale PnL to reasonable range [-1, 1]
        pnl_reward = np.tanh(pnl_pct / 10)  # 10% PnL maps to ~0.76

        # Fill penalty (if not included, severe penalty)
        if not included:
            return -0.5  # Strong negative for failed inclusion

        # Slippage efficiency bonus
        # If realized slippage < quoted, that's good
        if quoted_slippage > 0:
            slippage_efficiency = (quoted_slippage - realized_slippage) / quoted_slippage
            slippage_bonus = slippage_efficiency * 0.2  # Max 0.2 bonus
        else:
            slippage_bonus = 0

        # Combined reward
        reward = pnl_reward + slippage_bonus

        return reward

    def _classify_arm(self, slippage_bps: int) -> str:
        """Classify a slippage value to the nearest arm."""
        closest_arm = "medium"
        min_diff = float("inf")

        for name, arm in self.arms.items():
            diff = abs(arm.slippage_bps - slippage_bps)
            if diff < min_diff:
                min_diff = diff
                closest_arm = name

        return closest_arm

    async def train(self, trades_df: pd.DataFrame | None = None):
        """Train the bandit on trade data.

        Args:
            trades_df: DataFrame of trades. If None, fetches from database.
        """
        if trades_df is None and self.db is not None:
            trades_df = await self.db.get_recent_trades(hours=24)

        if trades_df is None or trades_df.empty:
            logger.info("No trades available for training")
            return

        logger.info(f"Training bandit on {len(trades_df)} trades")

        # Apply reward decay for non-stationary environments
        if self.reward_decay < 1.0:
            for arm in self.arms.values():
                arm.total_reward *= self.reward_decay
                arm.alpha = max(1.0, arm.alpha * self.reward_decay)
                arm.beta = max(1.0, arm.beta * self.reward_decay)

        # Process each trade
        trades_processed = 0
        total_reward = 0

        for _, trade in trades_df.iterrows():
            # Determine which arm this trade corresponds to
            quoted_slippage = trade.get("quoted_slippage_bps", 300)
            arm_name = self._classify_arm(quoted_slippage)

            # Calculate reward
            reward = self.calculate_reward(trade.to_dict())
            success = reward > 0

            # Update arm
            self.update(arm_name, reward, success)

            trades_processed += 1
            total_reward += reward

        # Log training results
        avg_reward = total_reward / trades_processed if trades_processed > 0 else 0
        recommendation = self.get_recommendation()

        self.training_history.append(
            {
                "trades_processed": trades_processed,
                "avg_reward": avg_reward,
                "recommended_arm": recommendation["arm"],
                "confidence": recommendation["confidence"],
            }
        )

        logger.info(
            f"Training complete: {trades_processed} trades, "
            f"avg reward: {avg_reward:.4f}, "
            f"recommended: {recommendation['arm']} ({recommendation['slippage_bps']}bps) "
            f"with {recommendation['confidence']:.1%} confidence"
        )

        # Persist to database if available
        if self.db is not None:
            for name, arm in self.arms.items():
                await self.db.save_bandit_arm(
                    arm_name=name,
                    slippage_bps=arm.slippage_bps,
                    pull_count=arm.count,
                    total_reward=arm.total_reward,
                )

    def to_params(self) -> dict[str, Any]:
        """Export bandit state as model params for Hot Path."""
        recommendation = self.get_recommendation()
        return {
            "arm_weights": self.get_arm_weights(),
            "recommended_arm": recommendation["arm"],
            "recommended_slippage_bps": recommendation["slippage_bps"],
            "confidence": recommendation["confidence"],
            "total_pulls": self.total_pulls,
            "strategy": self.strategy.value,
            "arm_stats": {
                name: {
                    "count": arm.count,
                    "mean_reward": arm.mean_reward,
                    "recent_mean": arm.recent_mean,
                    "std_dev": arm.std_dev,
                    "slippage_bps": arm.slippage_bps,
                }
                for name, arm in self.arms.items()
            },
        }

    def to_state(self) -> dict[str, Any]:
        """Export full state for persistence."""
        return {
            "arms": {name: arm.to_dict() for name, arm in self.arms.items()},
            "total_pulls": self.total_pulls,
            "exploration_param": self.exploration_param,
            "strategy": self.strategy.value,
            "epsilon": self.epsilon,
            "reward_decay": self.reward_decay,
        }

    def load_state(self, state: dict[str, Any]):
        """Load state from persistence."""
        if "arms" in state:
            for name, arm_data in state["arms"].items():
                if name in self.arms:
                    self.arms[name] = BanditArm.from_dict(arm_data)

        self.total_pulls = state.get("total_pulls", 0)
        self.exploration_param = state.get("exploration_param", 2.0)
        self.epsilon = state.get("epsilon", 0.1)
        self.reward_decay = state.get("reward_decay", 0.99)

        strategy_str = state.get("strategy", "ucb")
        try:
            self.strategy = ExplorationStrategy(strategy_str)
        except ValueError:
            self.strategy = ExplorationStrategy.UCB

        logger.info(f"Loaded bandit state: {self.total_pulls} total pulls")
