"""
RL Parameter Tuning Agent - Reinforcement Learning for adaptive optimization.

Uses reinforcement learning to learn optimal parameter adjustments:
- State: Current market conditions, performance metrics
- Action: Parameter adjustments (increase/decrease)
- Reward: Improvement in backtest metrics

The agent learns from experience to make smarter parameter choices
over time, adapting to changing market conditions.

Usage:
    from coldpath.backtest.rl_tuner import RLTunerAgent

    agent = RLTunerAgent(
        backtest_runner=my_backtest_fn,
        params_to_tune=["stop_loss_pct", "take_profit_pct", "max_position_sol"],
    )

    # Train the agent
    await agent.train(episodes=100)

    # Get tuned parameters
    tuned = agent.get_tuned_params(current_params, market_state)
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Action(Enum):
    """Parameter adjustment actions."""

    INCREASE_LARGE = "increase_large"  # +20%
    INCREASE_SMALL = "increase_small"  # +5%
    HOLD = "hold"  # No change
    DECREASE_SMALL = "decrease_small"  # -5%
    DECREASE_LARGE = "decrease_large"  # -20%


@dataclass
class State:
    """State representation for RL agent."""

    # Market conditions (normalized 0-1)
    volatility: float = 0.5
    trend: float = 0.5  # 0 = strong down, 1 = strong up
    liquidity: float = 0.5

    # Current performance (normalized)
    sharpe_ratio: float = 0.5
    win_rate: float = 0.5
    drawdown: float = 0.5  # 0 = high DD, 1 = low DD

    # Current params relative to bounds
    param_positions: dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> list[float]:
        """Convert to feature vector."""
        base = [
            self.volatility,
            self.trend,
            self.liquidity,
            self.sharpe_ratio,
            self.win_rate,
            self.drawdown,
        ]
        param_vals = list(self.param_positions.values()) if self.param_positions else [0.5]
        return base + param_vals

    def to_dict(self) -> dict[str, Any]:
        return {
            "volatility": self.volatility,
            "trend": self.trend,
            "liquidity": self.liquidity,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "drawdown": self.drawdown,
            "param_positions": self.param_positions,
        }


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""

    state: State
    action: dict[str, Action]  # Param -> Action
    reward: float
    next_state: State
    done: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class RLTunerConfig:
    """Configuration for RL tuner agent."""

    # Training
    episodes: int = 100
    max_steps_per_episode: int = 20

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995

    # Learning
    learning_rate: float = 0.001
    discount_factor: float = 0.95  # Gamma
    batch_size: int = 32

    # Replay buffer
    buffer_size: int = 10000
    min_buffer_size: int = 100

    # Reward shaping
    sharpe_weight: float = 0.4
    win_rate_weight: float = 0.2
    drawdown_weight: float = 0.2
    stability_weight: float = 0.2  # Penalize large changes

    # Parameter bounds
    param_bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "stop_loss_pct": (3.0, 20.0),
            "take_profit_pct": (5.0, 60.0),
            "max_position_sol": (0.01, 0.3),
            "max_risk_score": (0.15, 0.9),
            "slippage_bps": (100, 800),
        }
    )

    # Adjustment magnitudes
    small_adjustment_pct: float = 0.05
    large_adjustment_pct: float = 0.20

    random_seed: int = 42


@dataclass
class TrainingResult:
    """Result from RL training."""

    episodes_completed: int
    total_steps: int
    final_epsilon: float
    avg_reward_last_10: float
    best_reward: float
    best_params: dict[str, float]
    reward_history: list[float]
    time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes_completed": self.episodes_completed,
            "total_steps": self.total_steps,
            "final_epsilon": self.final_epsilon,
            "avg_reward_last_10": self.avg_reward_last_10,
            "best_reward": self.best_reward,
            "best_params": self.best_params,
        }


class RLTunerAgent:
    """Reinforcement Learning agent for parameter tuning.

    Uses a simple Q-learning approach with function approximation
    to learn optimal parameter adjustments based on market conditions
    and current performance.

    Features:
    - Epsilon-greedy exploration
    - Experience replay for stable learning
    - Multi-parameter optimization
    - Adaptive to market conditions

    Example:
        agent = RLTunerAgent(
            backtest_runner=my_backtest,
            params_to_tune=["stop_loss_pct", "take_profit_pct"],
        )

        # Train on historical data
        result = await agent.train(episodes=50)

        # Get recommended adjustments
        action = agent.get_action(current_state)
        new_params = agent.apply_action(current_params, action)
    """

    def __init__(
        self,
        backtest_runner: Callable,
        params_to_tune: list[str],
        config: RLTunerConfig | None = None,
    ):
        """Initialize the RL tuner agent.

        Args:
            backtest_runner: Async function to run backtests
            params_to_tune: List of parameter names to optimize
            config: Agent configuration
        """
        self.backtest_runner = backtest_runner
        self.params_to_tune = params_to_tune
        self.config = config or RLTunerConfig()

        # Validate params
        for param in params_to_tune:
            if param not in self.config.param_bounds:
                raise ValueError(f"No bounds defined for parameter: {param}")

        # Q-table (simplified - using dict for state-action values)
        self._q_table: dict[str, dict[str, float]] = {}

        # Experience replay buffer
        self._replay_buffer: list[Experience] = []

        # Training state
        self._epsilon = self.config.epsilon_start
        self._episode = 0
        self._step = 0

        # Best found
        self._best_reward = float("-inf")
        self._best_params: dict[str, float] = {}

        # Random state
        self._rng = random.Random(self.config.random_seed)

        # Reward history
        self._reward_history: list[float] = []

    def _state_key(self, state: State) -> str:
        """Convert state to hashable key."""
        # Discretize continuous state for tabular Q-learning
        vec = state.to_vector()
        discretized = [round(v * 10) / 10 for v in vec]  # Round to 0.1
        return str(discretized)

    def _get_q_value(self, state_key: str, action_key: str) -> float:
        """Get Q-value for state-action pair."""
        if state_key not in self._q_table:
            self._q_table[state_key] = {}
        return self._q_table[state_key].get(action_key, 0.0)

    def _set_q_value(self, state_key: str, action_key: str, value: float):
        """Set Q-value for state-action pair."""
        if state_key not in self._q_table:
            self._q_table[state_key] = {}
        self._q_table[state_key][action_key] = value

    def _action_to_key(self, action: dict[str, Action]) -> str:
        """Convert action dict to key."""
        return str({k: v.value for k, v in sorted(action.items())})

    def _normalize_value(self, value: float, low: float, high: float) -> float:
        """Normalize value to [0, 1] range."""
        if high <= low:
            return 0.5
        return max(0, min(1, (value - low) / (high - low)))

    def _denormalize_value(self, normalized: float, low: float, high: float) -> float:
        """Convert normalized [0, 1] to actual range."""
        return low + normalized * (high - low)

    def _get_param_position(self, param: str, value: float) -> float:
        """Get normalized position of param in its bounds."""
        low, high = self.config.param_bounds[param]
        return self._normalize_value(value, low, high)

    def _create_state(
        self,
        params: dict[str, float],
        metrics: dict[str, float],
        market_conditions: dict[str, float] | None = None,
    ) -> State:
        """Create state from current conditions."""
        market = market_conditions or {}

        # Normalize metrics
        sharpe = self._normalize_value(metrics.get("sharpe_ratio", 1.0), -2.0, 4.0)
        win_rate = self._normalize_value(metrics.get("win_rate_pct", 50) / 100, 0.0, 1.0)
        drawdown = 1 - self._normalize_value(metrics.get("max_drawdown_pct", 20), 0.0, 50.0)

        # Parameter positions
        param_positions = {}
        for param in self.params_to_tune:
            if param in params:
                param_positions[param] = self._get_param_position(param, params[param])

        return State(
            volatility=market.get("volatility", 0.5),
            trend=market.get("trend", 0.5),
            liquidity=market.get("liquidity", 0.5),
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            drawdown=drawdown,
            param_positions=param_positions,
        )

    def get_action(self, state: State, explore: bool = True) -> dict[str, Action]:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state
            explore: Whether to allow exploration

        Returns:
            Dict mapping parameter to action
        """
        action = {}

        for param in self.params_to_tune:
            if explore and self._rng.random() < self._epsilon:
                # Explore: random action
                action[param] = self._rng.choice(list(Action))
            else:
                # Exploit: best known action
                state_key = self._state_key(state)
                best_action = Action.HOLD
                best_value = float("-inf")

                for act in Action:
                    temp_action = {p: Action.HOLD for p in self.params_to_tune}
                    temp_action[param] = act
                    action_key = self._action_to_key(temp_action)
                    q_val = self._get_q_value(state_key, action_key)

                    if q_val > best_value:
                        best_value = q_val
                        best_action = act

                action[param] = best_action

        return action

    def apply_action(
        self,
        params: dict[str, float],
        action: dict[str, Action],
    ) -> dict[str, float]:
        """Apply action to get new parameters.

        Args:
            params: Current parameters
            action: Actions to apply

        Returns:
            New parameters after applying actions
        """
        new_params = params.copy()

        for param in self.params_to_tune:
            if param not in params:
                continue

            current = params[param]
            low, high = self.config.param_bounds[param]
            act = action.get(param, Action.HOLD)

            if act == Action.INCREASE_LARGE:
                delta = current * self.config.large_adjustment_pct
            elif act == Action.INCREASE_SMALL:
                delta = current * self.config.small_adjustment_pct
            elif act == Action.DECREASE_SMALL:
                delta = -current * self.config.small_adjustment_pct
            elif act == Action.DECREASE_LARGE:
                delta = -current * self.config.large_adjustment_pct
            else:
                delta = 0

            new_params[param] = max(low, min(high, current + delta))

        return new_params

    def _calculate_reward(
        self,
        old_metrics: dict[str, float],
        new_metrics: dict[str, float],
        old_params: dict[str, float],
        new_params: dict[str, float],
    ) -> float:
        """Calculate reward for state transition."""
        # Improvement in Sharpe
        sharpe_improvement = new_metrics.get("sharpe_ratio", 0) - old_metrics.get("sharpe_ratio", 0)
        sharpe_reward = sharpe_improvement * self.config.sharpe_weight * 10

        # Improvement in win rate
        wr_improvement = (
            new_metrics.get("win_rate_pct", 0) - old_metrics.get("win_rate_pct", 0)
        ) / 100
        wr_reward = wr_improvement * self.config.win_rate_weight * 10

        # Improvement in drawdown (lower is better)
        dd_improvement = (
            old_metrics.get("max_drawdown_pct", 50) - new_metrics.get("max_drawdown_pct", 50)
        ) / 50
        dd_reward = dd_improvement * self.config.drawdown_weight * 10

        # Stability penalty (penalize large changes)
        total_change = 0
        for param in self.params_to_tune:
            if param in old_params and param in new_params:
                if old_params[param] != 0:
                    change = abs(new_params[param] - old_params[param]) / abs(old_params[param])
                    total_change += change

        stability_penalty = -total_change * self.config.stability_weight

        return sharpe_reward + wr_reward + dd_reward + stability_penalty

    def _update_q_values(
        self,
        state: State,
        action: dict[str, Action],
        reward: float,
        next_state: State,
        done: bool,
    ):
        """Update Q-values using Q-learning update rule."""
        state_key = self._state_key(state)
        action_key = self._action_to_key(action)
        next_state_key = self._state_key(next_state)

        # Current Q-value
        current_q = self._get_q_value(state_key, action_key)

        # Max Q-value for next state
        max_next_q = 0.0
        for act in Action:
            temp_action = {p: act for p in self.params_to_tune}
            temp_key = self._action_to_key(temp_action)
            q_val = self._get_q_value(next_state_key, temp_key)
            max_next_q = max(max_next_q, q_val)

        # Q-learning update
        if done:
            target = reward
        else:
            target = reward + self.config.discount_factor * max_next_q

        new_q = current_q + self.config.learning_rate * (target - current_q)
        self._set_q_value(state_key, action_key, new_q)

    async def train(
        self,
        initial_params: dict[str, float],
        market_conditions: dict[str, float] | None = None,
        episodes: int | None = None,
    ) -> TrainingResult:
        """Train the RL agent.

        Args:
            initial_params: Starting parameters
            market_conditions: Optional market state
            episodes: Number of episodes (overrides config)

        Returns:
            TrainingResult with training statistics
        """
        start_time = datetime.utcnow()
        episodes = episodes or self.config.episodes
        total_steps = 0

        logger.info(f"Starting RL training for {episodes} episodes")

        for episode in range(episodes):
            self._episode = episode

            # Reset to initial params
            current_params = initial_params.copy()

            # Run initial backtest
            current_metrics = await self._run_backtest(current_params)

            # Create initial state
            current_state = self._create_state(current_params, current_metrics, market_conditions)

            episode_reward = 0

            for step in range(self.config.max_steps_per_episode):
                self._step += 1
                total_steps += 1

                # Get action
                action = self.get_action(current_state, explore=True)

                # Apply action
                new_params = self.apply_action(current_params, action)

                # Run backtest with new params
                new_metrics = await self._run_backtest(new_params)

                # Calculate reward
                reward = self._calculate_reward(
                    current_metrics,
                    new_metrics,
                    current_params,
                    new_params,
                )
                episode_reward += reward

                # Create new state
                new_state = self._create_state(new_params, new_metrics, market_conditions)

                # Store experience
                experience = Experience(
                    state=current_state,
                    action=action,
                    reward=reward,
                    next_state=new_state,
                    done=(step == self.config.max_steps_per_episode - 1),
                )
                self._replay_buffer.append(experience)

                # Trim buffer
                if len(self._replay_buffer) > self.config.buffer_size:
                    self._replay_buffer = self._replay_buffer[-self.config.buffer_size :]

                # Update Q-values
                self._update_q_values(
                    current_state,
                    action,
                    reward,
                    new_state,
                    done=(step == self.config.max_steps_per_episode - 1),
                )

                # Track best
                if reward > self._best_reward:
                    self._best_reward = reward
                    self._best_params = new_params.copy()

                # Update for next step
                current_params = new_params
                current_metrics = new_metrics
                current_state = new_state

            self._reward_history.append(episode_reward)

            # Decay epsilon
            self._epsilon = max(self.config.epsilon_end, self._epsilon * self.config.epsilon_decay)

            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = sum(self._reward_history[-10:]) / 10
                logger.info(
                    f"Episode {episode + 1}/{episodes}: "
                    f"avg_reward={avg_reward:.3f}, "
                    f"epsilon={self._epsilon:.3f}"
                )

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        avg_last_10 = sum(self._reward_history[-10:]) / 10 if self._reward_history else 0

        return TrainingResult(
            episodes_completed=episodes,
            total_steps=total_steps,
            final_epsilon=self._epsilon,
            avg_reward_last_10=avg_last_10,
            best_reward=self._best_reward,
            best_params=self._best_params,
            reward_history=self._reward_history,
            time_seconds=elapsed,
        )

    async def _run_backtest(self, params: dict[str, float]) -> dict[str, float]:
        """Run a single backtest."""
        try:
            result = await self.backtest_runner(params)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            return {"sharpe_ratio": 0, "win_rate_pct": 0, "max_drawdown_pct": 50}

    def get_tuned_params(
        self,
        current_params: dict[str, float],
        market_conditions: dict[str, float] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Get tuned parameters without exploration.

        Args:
            current_params: Current parameters
            market_conditions: Current market state
            metrics: Current performance metrics

        Returns:
            Tuned parameters
        """
        if not self._q_table:
            # Not trained yet, return current params
            return current_params.copy()

        # Create state
        state = self._create_state(
            current_params,
            metrics or {},
            market_conditions,
        )

        # Get best action (no exploration)
        action = self.get_action(state, explore=False)

        # Apply action
        return self.apply_action(current_params, action)

    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "episodes_trained": self._episode,
            "total_steps": self._step,
            "current_epsilon": self._epsilon,
            "q_table_size": len(self._q_table),
            "replay_buffer_size": len(self._replay_buffer),
            "best_reward": self._best_reward,
            "best_params": self._best_params,
            "avg_reward_last_10": (
                sum(self._reward_history[-10:]) / 10 if self._reward_history else 0
            ),
        }

    def save(self, path: str):
        """Save agent state to file."""
        data = {
            "q_table": self._q_table,
            "epsilon": self._epsilon,
            "best_reward": self._best_reward,
            "best_params": self._best_params,
            "reward_history": self._reward_history,
            "config": {
                "episodes": self.config.episodes,
                "epsilon_start": self.config.epsilon_start,
                "epsilon_end": self.config.epsilon_end,
                "epsilon_decay": self.config.epsilon_decay,
                "learning_rate": self.config.learning_rate,
                "discount_factor": self.config.discount_factor,
                "param_bounds": self.config.param_bounds,
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved RL agent to {path}")

    def load(self, path: str):
        """Load agent state from file."""
        with open(path) as f:
            data = json.load(f)

        self._q_table = data.get("q_table", {})
        self._epsilon = data.get("epsilon", self.config.epsilon_start)
        self._best_reward = data.get("best_reward", float("-inf"))
        self._best_params = data.get("best_params", {})
        self._reward_history = data.get("reward_history", [])

        logger.info(f"Loaded RL agent from {path} (Q-table size: {len(self._q_table)})")
