"""
RL-Based Parameter Tuning Agent for Backtest Optimization.

Uses reinforcement learning to learn optimal parameter adjustments
based on historical backtest outcomes. The agent learns which
parameter changes lead to better performance over time.

Key Features:
- State representation from market conditions and current params
- Action space for parameter adjustments
- Reward function based on Sharpe, win rate, drawdown
- Experience replay for stable learning
- Policy gradient and Q-learning implementations

Usage:
    agent = RLParameterAgent(param_bounds=BOUNDS)

    # Train from historical results
    await agent.train(historical_results)

    # Get next parameter suggestion
    state = agent.encode_state(current_params, market_metrics)
    action = agent.select_action(state)
    new_params = agent.apply_action(current_params, action)
"""

import asyncio
import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of parameter adjustments."""

    INCREASE = "increase"
    DECREASE = "decrease"
    HOLD = "hold"
    INCREASE_LARGE = "increase_large"
    DECREASE_LARGE = "decrease_large"


@dataclass
class ParameterBounds:
    """Bounds for a single parameter."""

    name: str
    min_value: float
    max_value: float
    step_size: float  # Increment for discrete adjustments
    is_integer: bool = False

    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1]."""
        return (value - self.min_value) / (self.max_value - self.min_value)

    def denormalize(self, normalized: float) -> float:
        """Convert normalized value back to actual value."""
        value = self.min_value + normalized * (self.max_value - self.min_value)
        if self.is_integer:
            return int(round(value))
        return value


@dataclass
class State:
    """State representation for RL agent."""

    # Parameter state (normalized)
    params_normalized: dict[str, float]

    # Market state
    volatility_level: float  # 0-1
    trend_direction: float  # -1 to 1
    risk_sentiment: float  # 0-1

    # Performance state (recent)
    recent_sharpe: float
    recent_win_rate: float
    recent_drawdown: float

    # Time features
    iteration: int
    time_elapsed: float

    def to_vector(self) -> list[float]:
        """Convert state to feature vector."""
        features = []
        features.extend(self.params_normalized.values())
        features.extend(
            [
                self.volatility_level,
                self.trend_direction,
                self.risk_sentiment,
                self.recent_sharpe / 3.0,  # Normalize
                self.recent_win_rate,
                self.recent_drawdown / 50.0,  # Normalize
                self.iteration / 100.0,  # Normalize
                min(1.0, self.time_elapsed / 3600.0),  # Normalize to 1 hour
            ]
        )
        return features


@dataclass
class Action:
    """Action for parameter adjustment."""

    param_name: str
    action_type: ActionType
    magnitude: float  # 0-1 scale

    def to_index(self, param_names: list[str]) -> int:
        """Convert action to discrete index."""
        param_idx = param_names.index(self.param_name) if self.param_name in param_names else 0
        action_idx = list(ActionType).index(self.action_type)
        return param_idx * len(ActionType) + action_idx


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""

    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RLAgentConfig:
    """Configuration for RL agent."""

    # Learning parameters
    learning_rate: float = 0.001
    discount_factor: float = 0.95  # Gamma
    exploration_rate: float = 0.3  # Epsilon
    exploration_decay: float = 0.995
    min_exploration: float = 0.05

    # Replay buffer
    buffer_size: int = 10000
    batch_size: int = 32
    min_buffer_size: int = 100

    # Reward shaping
    sharpe_weight: float = 0.4
    win_rate_weight: float = 0.2
    drawdown_weight: float = 0.2
    return_weight: float = 0.2

    # Action constraints
    max_change_per_step: float = 0.2  # Max 20% change per param per step

    # Training
    training_epochs: int = 10
    target_update_freq: int = 100


@dataclass
class RLTrainingResult:
    """Result from RL training session."""

    episodes_trained: int
    total_reward: float
    avg_reward: float
    best_reward: float
    final_exploration_rate: float
    experiences_collected: int
    training_time_seconds: float
    loss_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes_trained": self.episodes_trained,
            "total_reward": self.total_reward,
            "avg_reward": self.avg_reward,
            "best_reward": self.best_reward,
            "final_exploration_rate": self.final_exploration_rate,
            "experiences_collected": self.experiences_collected,
            "training_time_seconds": self.training_time_seconds,
        }


class ReplayBuffer:
    """Experience replay buffer for stable learning."""

    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample a batch of experiences."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork:
    """Simple Q-network for parameter tuning.

    This is a lightweight implementation without heavy ML dependencies.
    For production, consider using PyTorch or TensorFlow.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = None,
        learning_rate: float = 0.001,
    ):
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate

        # Simple weight initialization
        # In production, use proper neural network library
        self._weights: list[list[list[float]]] = []
        self._biases: list[list[float]] = []

        dims = [state_dim] + hidden_dims + [action_dim]
        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            weights = [[random.gauss(0, scale) for _ in range(dims[i + 1])] for _ in range(dims[i])]
            bias = [0.0 for _ in range(dims[i + 1])]
            self._weights.append(weights)
            self._biases.append(bias)

    def forward(self, state: list[float]) -> list[float]:
        """Forward pass through network."""
        activation = state[:]

        for i, (weights, bias) in enumerate(zip(self._weights, self._biases, strict=False)):
            next_activation = []
            for j in range(len(bias)):
                total = bias[j]
                for k, a in enumerate(activation):
                    total += a * weights[k][j]
                # ReLU for hidden layers, linear for output
                if i < len(self._weights) - 1:
                    total = max(0, total)
                next_activation.append(total)
            activation = next_activation

        return activation

    def update(
        self,
        state: list[float],
        action: int,
        target: float,
    ):
        """Simple gradient descent update."""
        # This is a simplified update - use proper backprop in production
        q_values = self.forward(state)
        current_q = q_values[action]
        error = target - current_q

        # Adjust output layer bias for selected action
        if self._biases:
            self._biases[-1][action] += self.learning_rate * error

    def copy_from(self, other: "QNetwork"):
        """Copy weights from another network."""
        self._weights = [[[w for w in row] for row in layer] for layer in other._weights]
        self._biases = [[b for b in layer] for layer in other._biases]


class RLParameterAgent:
    """RL-based parameter optimization agent.

    Uses Q-learning to learn optimal parameter adjustments
    based on backtest outcomes.

    Usage:
        agent = RLParameterAgent(
            param_bounds=DEFAULT_PARAM_BOUNDS,
            config=RLAgentConfig(),
        )

        # Train from historical results
        result = await agent.train_from_history(
            historical_results,
            backtest_runner=my_runner,
        )

        # Get parameter suggestion
        suggestion = agent.suggest_parameters(
            current_params,
            market_metrics,
        )
    """

    DEFAULT_PARAM_BOUNDS = {
        "stop_loss_pct": (3.0, 20.0, 1.0, False),
        "take_profit_pct": (8.0, 60.0, 2.0, False),
        "max_position_sol": (0.01, 0.20, 0.01, False),
        "min_liquidity_usd": (5000, 50000, 1000, True),
        "max_risk_score": (0.20, 0.70, 0.05, False),
        "slippage_bps": (150, 600, 25, True),
        "max_hold_minutes": (5, 90, 5, True),
        "kelly_safety_factor": (0.20, 0.70, 0.05, False),
    }

    def __init__(
        self,
        param_bounds: dict[str, tuple] | None = None,
        config: RLAgentConfig | None = None,
    ):
        """Initialize RL agent.

        Args:
            param_bounds: Parameter bounds dict {name: (min, max, step, is_int)}
            config: Agent configuration
        """
        self.config = config or RLAgentConfig()

        # Set up parameter bounds
        bounds = param_bounds or self.DEFAULT_PARAM_BOUNDS
        self.param_bounds: dict[str, ParameterBounds] = {}
        for name, (min_v, max_v, step, is_int) in bounds.items():
            self.param_bounds[name] = ParameterBounds(
                name=name,
                min_value=min_v,
                max_value=max_v,
                step_size=step,
                is_integer=is_int,
            )

        self.param_names = list(self.param_bounds.keys())

        # Calculate dimensions
        self.state_dim = len(self.param_names) + 8  # params + market + performance
        self.action_dim = len(self.param_names) * len(ActionType)

        # Initialize networks
        self.q_network = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            learning_rate=self.config.learning_rate,
        )
        self.target_network = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            learning_rate=self.config.learning_rate,
        )
        self.target_network.copy_from(self.q_network)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=self.config.buffer_size)

        # Training state
        self._exploration_rate = self.config.exploration_rate
        self._update_count = 0
        self._training_history: list[dict[str, Any]] = []

    @property
    def exploration_rate(self) -> float:
        """Current exploration rate."""
        return self._exploration_rate

    def encode_state(
        self,
        params: dict[str, Any],
        market_metrics: dict[str, Any] | None = None,
        recent_performance: dict[str, float] | None = None,
        iteration: int = 0,
        time_elapsed: float = 0.0,
    ) -> State:
        """Encode current state for the agent.

        Args:
            params: Current parameters
            market_metrics: Market condition metrics
            recent_performance: Recent backtest performance
            iteration: Current iteration number
            time_elapsed: Time since optimization started

        Returns:
            State object
        """
        # Normalize parameters
        params_normalized = {}
        for name, bounds in self.param_bounds.items():
            value = params.get(name, bounds.min_value)
            params_normalized[name] = bounds.normalize(value)

        # Extract market state
        market = market_metrics or {}
        volatility = market.get("volatility_24h", 15.0) / 50.0  # Normalize
        trend = market.get("price_change_24h", 0.0) / 50.0  # Normalize
        sentiment = market.get("risk_sentiment", 0.5)

        # Extract performance
        perf = recent_performance or {}
        recent_sharpe = perf.get("sharpe_ratio", 0.0)
        recent_win_rate = perf.get("win_rate_pct", 50.0) / 100.0
        recent_dd = perf.get("max_drawdown_pct", 20.0)

        return State(
            params_normalized=params_normalized,
            volatility_level=min(1.0, volatility),
            trend_direction=max(-1.0, min(1.0, trend)),
            risk_sentiment=sentiment,
            recent_sharpe=recent_sharpe,
            recent_win_rate=recent_win_rate,
            recent_drawdown=recent_dd,
            iteration=iteration,
            time_elapsed=time_elapsed,
        )

    def decode_action(self, action_idx: int) -> Action:
        """Decode action index to Action object."""
        action_types = list(ActionType)
        param_idx = action_idx // len(action_types)
        action_type_idx = action_idx % len(action_types)

        param_name = self.param_names[param_idx % len(self.param_names)]
        action_type = action_types[action_type_idx]

        # Magnitude based on action type
        magnitude_map = {
            ActionType.DECREASE_LARGE: 0.8,
            ActionType.DECREASE: 0.4,
            ActionType.HOLD: 0.0,
            ActionType.INCREASE: 0.4,
            ActionType.INCREASE_LARGE: 0.8,
        }

        return Action(
            param_name=param_name,
            action_type=action_type,
            magnitude=magnitude_map[action_type],
        )

    def apply_action(
        self,
        params: dict[str, Any],
        action: Action,
    ) -> dict[str, Any]:
        """Apply action to parameters.

        Args:
            params: Current parameters
            action: Action to apply

        Returns:
            New parameters with action applied
        """
        new_params = params.copy()

        if action.param_name not in self.param_bounds:
            return new_params

        bounds = self.param_bounds[action.param_name]
        current = params.get(action.param_name, bounds.min_value)

        # Calculate change based on action
        if action.action_type == ActionType.HOLD:
            return new_params

        range_size = bounds.max_value - bounds.min_value
        change = range_size * action.magnitude * self.config.max_change_per_step

        if action.action_type in [ActionType.DECREASE, ActionType.DECREASE_LARGE]:
            change = -change

        new_value = current + change
        new_value = max(bounds.min_value, min(bounds.max_value, new_value))

        if bounds.is_integer:
            new_value = int(round(new_value))
        else:
            new_value = round(new_value, 4)

        new_params[action.param_name] = new_value
        return new_params

    def select_action(
        self,
        state: State,
        explore: bool = True,
    ) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state
            explore: Whether to use exploration

        Returns:
            Action index
        """
        if explore and random.random() < self._exploration_rate:
            # Random exploration
            return random.randint(0, self.action_dim - 1)

        # Greedy action selection
        state_vec = state.to_vector()
        q_values = self.q_network.forward(state_vec)
        return q_values.index(max(q_values))

    def calculate_reward(
        self,
        metrics: dict[str, float],
        prev_metrics: dict[str, float] | None = None,
    ) -> float:
        """Calculate reward from backtest metrics.

        Args:
            metrics: Current backtest metrics
            prev_metrics: Previous metrics for improvement reward

        Returns:
            Reward value
        """
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate_pct", 0) / 100.0
        max_dd = metrics.get("max_drawdown_pct", 50)
        total_return = metrics.get("total_return_pct", 0)

        # Normalize components
        sharpe_norm = min(1.0, max(0, sharpe / 3.0))
        wr_norm = win_rate
        dd_norm = max(0, 1 - max_dd / 50.0)
        ret_norm = min(1.0, max(0, (total_return + 50) / 150.0))  # -50% to 100%

        # Weighted reward
        reward = (
            sharpe_norm * self.config.sharpe_weight
            + wr_norm * self.config.win_rate_weight
            + dd_norm * self.config.drawdown_weight
            + ret_norm * self.config.return_weight
        )

        # Bonus for improvement
        if prev_metrics:
            prev_sharpe = prev_metrics.get("sharpe_ratio", 0)
            improvement = sharpe - prev_sharpe
            if improvement > 0:
                reward += improvement * 0.1  # Bonus for improving

        return reward

    async def train_from_history(
        self,
        historical_results: list[dict[str, Any]],
        backtest_runner: Callable | None = None,
        epochs: int | None = None,
    ) -> RLTrainingResult:
        """Train agent from historical backtest results.

        Args:
            historical_results: List of past backtest results
            backtest_runner: Optional runner for additional training
            epochs: Number of training epochs

        Returns:
            Training result
        """
        start_time = datetime.now()
        epochs = epochs or self.config.training_epochs

        total_reward = 0.0
        best_reward = 0.0
        loss_history = []

        # Sort by timestamp
        sorted_results = sorted(
            historical_results,
            key=lambda x: x.get("timestamp", ""),
        )

        for _epoch in range(epochs):
            epoch_reward = 0.0

            for i in range(1, len(sorted_results)):
                prev_result = sorted_results[i - 1]
                curr_result = sorted_results[i]

                prev_params = prev_result.get("params", {})
                curr_params = curr_result.get("params", {})
                curr_metrics = curr_result.get("metrics", {})

                # Create states
                prev_state = self.encode_state(prev_params)
                curr_state = self.encode_state(curr_params)

                # Calculate reward
                reward = self.calculate_reward(
                    curr_metrics,
                    prev_result.get("metrics"),
                )

                # Infer action from parameter change
                action_idx = self._infer_action(prev_params, curr_params)

                # Store experience
                experience = Experience(
                    state=prev_state.to_vector(),
                    action=action_idx,
                    reward=reward,
                    next_state=curr_state.to_vector(),
                    done=(i == len(sorted_results) - 1),
                )
                self.replay_buffer.push(experience)

                epoch_reward += reward
                total_reward += reward
                best_reward = max(best_reward, reward)

            # Train on batch
            if len(self.replay_buffer) >= self.config.min_buffer_size:
                batch = self.replay_buffer.sample(self.config.batch_size)
                loss = self._train_batch(batch)
                loss_history.append(loss)

            # Decay exploration
            self._exploration_rate = max(
                self.config.min_exploration,
                self._exploration_rate * self.config.exploration_decay,
            )

        training_time = (datetime.now() - start_time).total_seconds()

        return RLTrainingResult(
            episodes_trained=epochs,
            total_reward=total_reward,
            avg_reward=total_reward / max(1, len(sorted_results) * epochs),
            best_reward=best_reward,
            final_exploration_rate=self._exploration_rate,
            experiences_collected=len(self.replay_buffer),
            training_time_seconds=training_time,
            loss_history=loss_history,
        )

    async def train_online(
        self,
        initial_params: dict[str, Any],
        backtest_runner: Callable,
        num_episodes: int = 50,
        market_metrics: dict[str, Any] | None = None,
        on_progress: Callable[[int, float], None] | None = None,
    ) -> RLTrainingResult:
        """Train agent online through actual backtests.

        Args:
            initial_params: Starting parameters
            backtest_runner: Async function to run backtests
            num_episodes: Number of training episodes
            market_metrics: Market conditions
            on_progress: Progress callback

        Returns:
            Training result
        """
        start_time = datetime.now()

        current_params = initial_params.copy()
        total_reward = 0.0
        best_reward = 0.0
        best_params = initial_params.copy()
        best_metrics = {}
        loss_history = []

        prev_metrics = None

        for episode in range(num_episodes):
            # Encode state
            state = self.encode_state(
                current_params,
                market_metrics,
                prev_metrics,
                iteration=episode,
            )

            # Select action
            action_idx = self.select_action(state, explore=True)
            action = self.decode_action(action_idx)

            # Apply action
            new_params = self.apply_action(current_params, action)

            # Run backtest
            metrics = await self._run_backtest(backtest_runner, new_params)

            # Calculate reward
            reward = self.calculate_reward(metrics, prev_metrics)

            # Track best
            if reward > best_reward:
                best_reward = reward
                best_params = new_params.copy()
                best_metrics = metrics.copy()

            # Create next state
            next_state = self.encode_state(
                new_params,
                market_metrics,
                metrics,
                iteration=episode + 1,
            )

            # Store experience
            experience = Experience(
                state=state.to_vector(),
                action=action_idx,
                reward=reward,
                next_state=next_state.to_vector(),
                done=(episode == num_episodes - 1),
            )
            self.replay_buffer.push(experience)

            # Train on batch
            if len(self.replay_buffer) >= self.config.min_buffer_size:
                batch = self.replay_buffer.sample(self.config.batch_size)
                loss = self._train_batch(batch)
                loss_history.append(loss)

            # Update current params
            current_params = new_params
            prev_metrics = metrics
            total_reward += reward

            # Progress callback
            if on_progress:
                on_progress(episode + 1, total_reward / (episode + 1))

            # Decay exploration
            self._exploration_rate = max(
                self.config.min_exploration,
                self._exploration_rate * self.config.exploration_decay,
            )

        training_time = (datetime.now() - start_time).total_seconds()

        # Store training result
        self._training_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "episodes": num_episodes,
                "total_reward": total_reward,
                "best_reward": best_reward,
                "best_params": best_params,
                "best_metrics": best_metrics,
            }
        )

        return RLTrainingResult(
            episodes_trained=num_episodes,
            total_reward=total_reward,
            avg_reward=total_reward / num_episodes,
            best_reward=best_reward,
            final_exploration_rate=self._exploration_rate,
            experiences_collected=len(self.replay_buffer),
            training_time_seconds=training_time,
            loss_history=loss_history,
        )

    def suggest_parameters(
        self,
        current_params: dict[str, Any],
        market_metrics: dict[str, Any] | None = None,
        recent_performance: dict[str, float] | None = None,
        deterministic: bool = False,
    ) -> tuple[dict[str, Any], float]:
        """Suggest next parameters based on learned policy.

        Args:
            current_params: Current parameters
            market_metrics: Market conditions
            recent_performance: Recent performance metrics
            deterministic: Use greedy policy (no exploration)

        Returns:
            Tuple of (suggested_params, confidence)
        """
        state = self.encode_state(
            current_params,
            market_metrics,
            recent_performance,
        )

        action_idx = self.select_action(state, explore=not deterministic)
        action = self.decode_action(action_idx)

        new_params = self.apply_action(current_params, action)

        # Calculate confidence from Q-values
        q_values = self.q_network.forward(state.to_vector())
        max_q = max(q_values)
        avg_q = sum(q_values) / len(q_values)
        confidence = (max_q - avg_q) / max(abs(max_q), abs(avg_q), 0.01)
        confidence = min(1.0, max(0.0, confidence))

        return new_params, confidence

    def get_best_action_for_param(
        self,
        param_name: str,
        state: State,
    ) -> ActionType:
        """Get best action for a specific parameter."""
        param_idx = self.param_names.index(param_name)
        state_vec = state.to_vector()
        q_values = self.q_network.forward(state_vec)

        action_types = list(ActionType)
        best_action = ActionType.HOLD
        best_q = float("-inf")

        for i, action_type in enumerate(action_types):
            action_idx = param_idx * len(action_types) + i
            if action_idx < len(q_values) and q_values[action_idx] > best_q:
                best_q = q_values[action_idx]
                best_action = action_type

        return best_action

    def get_policy_summary(self) -> dict[str, Any]:
        """Get summary of learned policy."""
        summary = {
            "exploration_rate": self._exploration_rate,
            "experiences_collected": len(self.replay_buffer),
            "training_sessions": len(self._training_history),
        }

        # Sample policy for each parameter
        for param_name in self.param_names:
            # Default state
            state = State(
                params_normalized={n: 0.5 for n in self.param_names},
                volatility_level=0.5,
                trend_direction=0.0,
                risk_sentiment=0.5,
                recent_sharpe=1.0,
                recent_win_rate=0.5,
                recent_drawdown=20.0,
                iteration=0,
                time_elapsed=0.0,
            )
            best_action = self.get_best_action_for_param(param_name, state)
            summary[f"{param_name}_default_action"] = best_action.value

        return summary

    def _infer_action(
        self,
        prev_params: dict[str, Any],
        curr_params: dict[str, Any],
    ) -> int:
        """Infer action index from parameter change."""
        for param_name in self.param_names:
            if param_name not in prev_params or param_name not in curr_params:
                continue

            prev_val = prev_params[param_name]
            curr_val = curr_params[param_name]

            if abs(curr_val - prev_val) < 0.0001:
                action_type = ActionType.HOLD
            else:
                bounds = self.param_bounds[param_name]
                change_pct = abs(curr_val - prev_val) / (bounds.max_value - bounds.min_value)

                if curr_val > prev_val:
                    action_type = (
                        ActionType.INCREASE_LARGE if change_pct > 0.1 else ActionType.INCREASE
                    )
                else:
                    action_type = (
                        ActionType.DECREASE_LARGE if change_pct > 0.1 else ActionType.DECREASE
                    )

            param_idx = self.param_names.index(param_name)
            action_idx = param_idx * len(ActionType) + list(ActionType).index(action_type)
            return action_idx

        return 0  # Default: hold first parameter

    def _train_batch(self, batch: list[Experience]) -> float:
        """Train on a batch of experiences."""
        total_loss = 0.0

        for exp in batch:
            # Calculate target Q-value
            current_q = self.q_network.forward(exp.state)[exp.action]

            if exp.done:
                target = exp.reward
            else:
                next_q_values = self.target_network.forward(exp.next_state)
                target = exp.reward + self.config.discount_factor * max(next_q_values)

            # Update network
            self.q_network.update(exp.state, exp.action, target)

            # Track loss
            total_loss += (target - current_q) ** 2

            self._update_count += 1

            # Update target network periodically
            if self._update_count % self.config.target_update_freq == 0:
                self.target_network.copy_from(self.q_network)

        return total_loss / len(batch)

    async def _run_backtest(
        self,
        runner: Callable,
        params: dict[str, Any],
    ) -> dict[str, float]:
        """Run backtest with given parameters."""
        if asyncio.iscoroutinefunction(runner):
            return await runner(params)
        else:
            return runner(params)

    def save_state(self) -> dict[str, Any]:
        """Save agent state for persistence."""
        return {
            "config": {
                "learning_rate": self.config.learning_rate,
                "discount_factor": self.config.discount_factor,
                "exploration_rate": self._exploration_rate,
            },
            "training_history": self._training_history[-100:],  # Keep last 100
            "param_names": self.param_names,
        }

    def load_state(self, state: dict[str, Any]):
        """Load agent state from persistence."""
        if "config" in state:
            self._exploration_rate = state["config"].get(
                "exploration_rate", self.config.exploration_rate
            )
        if "training_history" in state:
            self._training_history = state["training_history"]


# Convenience function
def create_rl_agent(
    risk_tolerance: str = "moderate",
    exploration_rate: float = 0.3,
) -> RLParameterAgent:
    """Create an RL agent configured for specific risk profile.

    Args:
        risk_tolerance: Risk level (affects parameter bounds)
        exploration_rate: Initial exploration rate

    Returns:
        Configured RLParameterAgent
    """
    config = RLAgentConfig(
        exploration_rate=exploration_rate,
    )

    # Adjust bounds based on risk tolerance
    bounds = RLParameterAgent.DEFAULT_PARAM_BOUNDS.copy()

    if risk_tolerance == "conservative":
        bounds["stop_loss_pct"] = (3.0, 10.0, 0.5, False)
        bounds["max_risk_score"] = (0.15, 0.40, 0.05, False)
    elif risk_tolerance == "aggressive":
        bounds["stop_loss_pct"] = (8.0, 25.0, 1.0, False)
        bounds["max_risk_score"] = (0.40, 0.70, 0.05, False)
    elif risk_tolerance == "degen":
        bounds["stop_loss_pct"] = (10.0, 35.0, 2.0, False)
        bounds["max_risk_score"] = (0.50, 0.90, 0.05, False)

    return RLParameterAgent(param_bounds=bounds, config=config)
