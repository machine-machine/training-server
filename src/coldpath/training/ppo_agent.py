"""
PPO Position Sizing Agent for RL-based trading.

Implements Proximal Policy Optimization (PPO) for:
- Continuous position size [0, 1]
- Discrete hold duration (5 classes)
- Discrete exit strategy (3 classes)

State space: 50 market features + 10 portfolio features + 5 risk metrics = 65 dims
Initialized from Kelly criterion sizing for warm start.

Architecture: Actor-Critic with separate networks, GAE (lambda=0.95), clip epsilon=0.2.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical, Normal

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .rl_reward import RewardConfig, SharpeBasedReward

logger = logging.getLogger(__name__)

# State dimensions
MARKET_FEATURES = 50
PORTFOLIO_FEATURES = 10
RISK_FEATURES = 5
STATE_DIM = MARKET_FEATURES + PORTFOLIO_FEATURES + RISK_FEATURES  # 65

# Action dimensions
HOLD_DURATION_CLASSES = 5  # 1min, 5min, 15min, 1h, 4h
EXIT_STRATEGY_CLASSES = 3  # fixed_stop, trailing_stop, atr_stop


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""

    # Network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "tanh"

    # PPO hyperparameters
    clip_epsilon: float = 0.2
    value_clip: float = 0.5
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Training
    learning_rate: float = 3e-4
    n_epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 2048

    # Position sizing bounds
    min_position: float = 0.0
    max_position: float = 1.0

    # Kelly criterion warm start
    use_kelly_warmstart: bool = True
    kelly_fraction: float = 0.25  # Quarter Kelly for safety

    # Logging
    log_interval: int = 100


@dataclass
class Experience:
    """Single transition for PPO buffer."""

    state: np.ndarray  # [65]
    action_position: float  # Continuous [0, 1]
    action_duration: int  # Discrete [0, 4]
    action_exit: int  # Discrete [0, 2]
    reward: float
    next_state: np.ndarray  # [65]
    done: bool
    value: float  # V(s)
    log_prob: float  # log pi(a|s)
    advantage: float = 0.0  # GAE advantage
    returns: float = 0.0  # Discounted return


class RolloutBuffer:
    """Buffer for collecting PPO rollout data."""

    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self.experiences: list[Experience] = []

    def add(self, exp: Experience):
        if len(self.experiences) >= self.max_size:
            self.experiences.pop(0)
        self.experiences.append(exp)

    def is_full(self) -> bool:
        return len(self.experiences) >= self.max_size

    def clear(self):
        self.experiences.clear()

    def compute_advantages(self, reward_fn: SharpeBasedReward):
        """Compute GAE advantages for all experiences."""
        n = len(self.experiences)
        if n == 0:
            return

        rewards = [e.reward for e in self.experiences]
        values = [e.value for e in self.experiences]
        dones = [e.done for e in self.experiences]

        # Bootstrap value
        next_value = 0.0 if self.experiences[-1].done else self.experiences[-1].value

        advantages = reward_fn.compute_gae(rewards, values, next_value, dones)

        # Compute returns (advantages + values)
        for i, exp in enumerate(self.experiences):
            exp.advantage = float(advantages[i])
            exp.returns = exp.advantage + exp.value

    def get_batches(self, batch_size: int):
        """Yield mini-batches for PPO training."""
        indices = np.random.permutation(len(self.experiences))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield [self.experiences[i] for i in batch_indices]

    def __len__(self):
        return len(self.experiences)


if TORCH_AVAILABLE:

    class ActorNetwork(nn.Module):
        """Actor network outputting position size, duration, and exit strategy."""

        def __init__(self, config: PPOConfig):
            super().__init__()
            self.config = config

            # Build shared feature extractor
            layers = []
            in_dim = STATE_DIM
            for h_dim in config.hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.Tanh() if config.activation == "tanh" else nn.ReLU())
                in_dim = h_dim

            self.feature_extractor = nn.Sequential(*layers)

            # Position size head (continuous, Beta distribution for [0,1])
            self.position_mean = nn.Linear(in_dim, 1)
            self.position_log_std = nn.Parameter(torch.zeros(1))

            # Hold duration head (discrete, 5 classes)
            self.duration_head = nn.Linear(in_dim, HOLD_DURATION_CLASSES)

            # Exit strategy head (discrete, 3 classes)
            self.exit_head = nn.Linear(in_dim, EXIT_STRATEGY_CLASSES)

        def forward(self, state: torch.Tensor):
            """Forward pass returning action distributions.

            Returns:
                position_dist: Normal distribution for position size
                duration_logits: Logits for duration classification
                exit_logits: Logits for exit strategy classification
            """
            features = self.feature_extractor(state)

            # Position size (sigmoid to bound [0,1])
            pos_mean = torch.sigmoid(self.position_mean(features))
            pos_std = torch.exp(self.position_log_std).expand_as(pos_mean)
            pos_std = torch.clamp(pos_std, min=0.01, max=0.5)

            # Duration and exit logits
            dur_logits = self.duration_head(features)
            exit_logits = self.exit_head(features)

            return pos_mean, pos_std, dur_logits, exit_logits

        def get_action(self, state: torch.Tensor, deterministic: bool = False):
            """Sample action from policy.

            Returns:
                action_position: Position size [0, 1]
                action_duration: Duration class [0, 4]
                action_exit: Exit strategy class [0, 2]
                log_prob: Total log probability
            """
            pos_mean, pos_std, dur_logits, exit_logits = self.forward(state)

            # Position size
            pos_dist = Normal(pos_mean, pos_std)
            if deterministic:
                action_pos = pos_mean
            else:
                action_pos = pos_dist.sample()
            action_pos = torch.clamp(action_pos, 0.0, 1.0)
            log_prob_pos = pos_dist.log_prob(action_pos).sum(dim=-1)

            # Duration
            dur_dist = Categorical(logits=dur_logits)
            if deterministic:
                action_dur = torch.argmax(dur_logits, dim=-1)
            else:
                action_dur = dur_dist.sample()
            log_prob_dur = dur_dist.log_prob(action_dur)

            # Exit strategy
            exit_dist = Categorical(logits=exit_logits)
            if deterministic:
                action_exit = torch.argmax(exit_logits, dim=-1)
            else:
                action_exit = exit_dist.sample()
            log_prob_exit = exit_dist.log_prob(action_exit)

            total_log_prob = log_prob_pos + log_prob_dur + log_prob_exit

            return action_pos.squeeze(-1), action_dur, action_exit, total_log_prob

        def evaluate_actions(
            self,
            states: torch.Tensor,
            positions: torch.Tensor,
            durations: torch.Tensor,
            exits: torch.Tensor,
        ):
            """Evaluate log probabilities and entropy for given actions.

            Returns:
                log_probs: Total log probability
                entropy: Total entropy
            """
            pos_mean, pos_std, dur_logits, exit_logits = self.forward(states)

            # Position
            pos_dist = Normal(pos_mean, pos_std)
            log_prob_pos = pos_dist.log_prob(positions.unsqueeze(-1)).sum(dim=-1)
            entropy_pos = pos_dist.entropy().sum(dim=-1)

            # Duration
            dur_dist = Categorical(logits=dur_logits)
            log_prob_dur = dur_dist.log_prob(durations)
            entropy_dur = dur_dist.entropy()

            # Exit
            exit_dist = Categorical(logits=exit_logits)
            log_prob_exit = exit_dist.log_prob(exits)
            entropy_exit = exit_dist.entropy()

            total_log_prob = log_prob_pos + log_prob_dur + log_prob_exit
            total_entropy = entropy_pos + entropy_dur + entropy_exit

            return total_log_prob, total_entropy

    class CriticNetwork(nn.Module):
        """Critic network estimating state value V(s)."""

        def __init__(self, config: PPOConfig):
            super().__init__()

            layers = []
            in_dim = STATE_DIM
            for h_dim in config.hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.Tanh() if config.activation == "tanh" else nn.ReLU())
                in_dim = h_dim

            layers.append(nn.Linear(in_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.network(state).squeeze(-1)


class PPOAgent:
    """PPO agent for position sizing, hold duration, and exit strategy.

    Uses Actor-Critic architecture with:
    - Clipped surrogate objective for stable policy updates
    - GAE for advantage estimation
    - Kelly criterion warm start for faster convergence
    - Sharpe-based reward function for risk-adjusted training
    """

    def __init__(self, config: PPOConfig | None = None):
        self.config = config or PPOConfig()

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available; PPO agent will use Kelly fallback only")
            self._actor = None
            self._critic = None
            self._optimizer = None
        else:
            self._actor = ActorNetwork(self.config)
            self._critic = CriticNetwork(self.config)
            self._optimizer = optim.Adam(
                list(self._actor.parameters()) + list(self._critic.parameters()),
                lr=self.config.learning_rate,
            )

        self._reward_fn = SharpeBasedReward(
            RewardConfig(
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
        )
        self._buffer = RolloutBuffer(self.config.buffer_size)

        # Training state
        self._total_steps: int = 0
        self._total_updates: int = 0
        self._is_learning: bool = False
        self._episode_rewards: deque = deque(maxlen=100)

        # Kelly fallback state
        self._win_rate: float = 0.5
        self._avg_win: float = 0.02
        self._avg_loss: float = 0.01

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        """Select trading action from current policy.

        Args:
            state: 65-dimensional state vector
            deterministic: If True, use mean action (no exploration)

        Returns:
            Dictionary with position_size, hold_duration, exit_strategy, value, log_prob
        """
        # Validate state
        if len(state) != STATE_DIM:
            logger.warning(f"State dim mismatch: {len(state)} != {STATE_DIM}, padding/truncating")
            padded = np.zeros(STATE_DIM)
            padded[: min(len(state), STATE_DIM)] = state[:STATE_DIM]
            state = padded

        # Replace NaN/Inf
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        if not TORCH_AVAILABLE or self._actor is None:
            return self._kelly_fallback(state)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)

            # Get action from actor
            pos, dur, exit_s, log_prob = self._actor.get_action(state_t, deterministic)

            # Get value from critic
            value = self._critic(state_t)

        # Map duration class to minutes
        duration_map = {0: 1, 1: 5, 2: 15, 3: 60, 4: 240}
        exit_map = {0: "fixed_stop", 1: "trailing_stop", 2: "atr_stop"}

        position_size = float(pos.item())

        # Apply Kelly fraction as upper bound during early learning
        if self.config.use_kelly_warmstart and self._total_steps < 1000:
            kelly_size = self._compute_kelly_size()
            blend = min(self._total_steps / 1000.0, 1.0)
            position_size = (1 - blend) * kelly_size + blend * position_size

        return {
            "position_size": np.clip(
                position_size, self.config.min_position, self.config.max_position
            ),
            "hold_duration_minutes": duration_map.get(int(dur.item()), 15),
            "hold_duration_class": int(dur.item()),
            "exit_strategy": exit_map.get(int(exit_s.item()), "trailing_stop"),
            "exit_strategy_class": int(exit_s.item()),
            "value": float(value.item()),
            "log_prob": float(log_prob.item()),
        }

    def step(
        self,
        state: np.ndarray,
        action: dict[str, Any],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Record a transition and train if buffer is full.

        Args:
            state: Current state [65]
            action: Action dict from select_action()
            reward: Scalar reward
            next_state: Next state [65]
            done: Episode termination flag
        """
        self._total_steps += 1

        # Update Kelly statistics
        if reward > 0:
            self._avg_win = 0.99 * self._avg_win + 0.01 * reward
            self._win_rate = 0.999 * self._win_rate + 0.001
        else:
            self._avg_loss = 0.99 * self._avg_loss + 0.01 * abs(reward)
            self._win_rate = 0.999 * self._win_rate

        if not TORCH_AVAILABLE or self._actor is None:
            return

        exp = Experience(
            state=state,
            action_position=action.get("position_size", 0.0),
            action_duration=action.get("hold_duration_class", 2),
            action_exit=action.get("exit_strategy_class", 1),
            reward=reward,
            next_state=next_state,
            done=done,
            value=action.get("value", 0.0),
            log_prob=action.get("log_prob", 0.0),
        )
        self._buffer.add(exp)

        # Train when buffer is full
        if self._buffer.is_full():
            self._train()

    def _train(self):
        """Run PPO training on collected rollout buffer."""
        if not TORCH_AVAILABLE or len(self._buffer) < self.config.batch_size:
            return

        # Compute advantages
        self._buffer.compute_advantages(self._reward_fn)

        # Normalize advantages
        advantages = np.array([e.advantage for e in self._buffer.experiences])
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        for exp in self._buffer.experiences:
            exp.advantage = (exp.advantage - adv_mean) / adv_std

        # PPO epochs
        for _epoch in range(self.config.n_epochs):
            for batch in self._buffer.get_batches(self.config.batch_size):
                self._update_step(batch)

        self._total_updates += 1
        self._buffer.clear()
        self._reward_fn.reset()

        if self._total_updates % self.config.log_interval == 0:
            logger.info(f"PPO update #{self._total_updates}, steps={self._total_steps}")

    def _update_step(self, batch: list[Experience]):
        """Single PPO update step on a mini-batch."""
        states = torch.FloatTensor(np.array([e.state for e in batch]))
        positions = torch.FloatTensor(np.array([e.action_position for e in batch]))
        durations = torch.LongTensor(np.array([e.action_duration for e in batch]))
        exits = torch.LongTensor(np.array([e.action_exit for e in batch]))
        old_log_probs = torch.FloatTensor(np.array([e.log_prob for e in batch]))
        advantages = torch.FloatTensor(np.array([e.advantage for e in batch]))
        returns = torch.FloatTensor(np.array([e.returns for e in batch]))
        old_values = torch.FloatTensor(np.array([e.value for e in batch]))

        # Evaluate actions under current policy
        new_log_probs, entropy = self._actor.evaluate_actions(states, positions, durations, exits)

        # PPO clipped surrogate loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss with clipping
        values = self._critic(states)
        if self.config.value_clip > 0:
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.config.value_clip,
                self.config.value_clip,
            )
            v_loss1 = (values - returns).pow(2)
            v_loss2 = (values_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
        else:
            value_loss = 0.5 * (values - returns).pow(2).mean()

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Combined loss
        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            + self.config.entropy_coef * entropy_loss
        )

        # Optimize
        self._optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                list(self._actor.parameters()) + list(self._critic.parameters()),
                self.config.max_grad_norm,
            )
        self._optimizer.step()

    def _kelly_fallback(self, state: np.ndarray) -> dict[str, Any]:
        """Kelly criterion fallback when PyTorch unavailable or early learning."""
        kelly_size = self._compute_kelly_size()

        return {
            "position_size": np.clip(
                kelly_size, self.config.min_position, self.config.max_position
            ),
            "hold_duration_minutes": 15,
            "hold_duration_class": 2,
            "exit_strategy": "trailing_stop",
            "exit_strategy_class": 1,
            "value": 0.0,
            "log_prob": 0.0,
        }

    def _compute_kelly_size(self) -> float:
        """Compute Kelly criterion position size."""
        if self._avg_loss < 1e-8:
            return self.config.kelly_fraction

        odds = self._avg_win / self._avg_loss
        kelly = self._win_rate - (1 - self._win_rate) / odds

        # Apply fractional Kelly
        return max(0.0, kelly * self.config.kelly_fraction)

    def save_weights(self, path: str):
        """Save actor and critic weights."""
        if not TORCH_AVAILABLE or self._actor is None:
            logger.warning("Cannot save weights: PyTorch not available")
            return

        torch.save(
            {
                "actor": self._actor.state_dict(),
                "critic": self._critic.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "config": {
                    "total_steps": self._total_steps,
                    "total_updates": self._total_updates,
                    "win_rate": self._win_rate,
                    "avg_win": self._avg_win,
                    "avg_loss": self._avg_loss,
                },
            },
            path,
        )
        logger.info(f"PPO weights saved to {path}")

    def load_weights(self, path: str):
        """Load actor and critic weights."""
        if not TORCH_AVAILABLE or self._actor is None:
            logger.warning("Cannot load weights: PyTorch not available")
            return

        checkpoint = torch.load(path, map_location="cpu")
        self._actor.load_state_dict(checkpoint["actor"])
        self._critic.load_state_dict(checkpoint["critic"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])

        config = checkpoint.get("config", {})
        self._total_steps = config.get("total_steps", 0)
        self._total_updates = config.get("total_updates", 0)
        self._win_rate = config.get("win_rate", 0.5)
        self._avg_win = config.get("avg_win", 0.02)
        self._avg_loss = config.get("avg_loss", 0.01)

        logger.info(f"PPO weights loaded from {path} (steps={self._total_steps})")

    def export_policy_for_hotpath(self) -> dict[str, Any]:
        """Export current policy as lightweight parameters for HotPath inference.

        Returns a dictionary that can be sent via gRPC to the Rust HotPath
        for fast position sizing decisions without full neural network inference.
        """
        if not TORCH_AVAILABLE or self._actor is None:
            return {
                "type": "kelly",
                "kelly_fraction": self.config.kelly_fraction,
                "win_rate": self._win_rate,
                "avg_win": self._avg_win,
                "avg_loss": self._avg_loss,
            }

        # Extract linear approximation from first + last layer
        with torch.no_grad():
            # Get weight matrices for a linear approximation
            first_weight = self._actor.feature_extractor[0].weight.numpy()
            self._actor.feature_extractor[0].bias.numpy()
            self._actor.position_mean.weight.numpy()
            pos_bias = self._actor.position_mean.bias.numpy()

        return {
            "type": "ppo_linear_approx",
            "total_steps": self._total_steps,
            "total_updates": self._total_updates,
            "feature_weights": first_weight.mean(axis=0).tolist(),  # Averaged first layer
            "position_bias": float(pos_bias[0]),
            "kelly_fraction": self.config.kelly_fraction,
            "win_rate": self._win_rate,
            "avg_win": self._avg_win,
            "avg_loss": self._avg_loss,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_steps": self._total_steps,
            "total_updates": self._total_updates,
            "buffer_size": len(self._buffer),
            "is_learning": self._is_learning,
            "win_rate": self._win_rate,
            "avg_win": self._avg_win,
            "avg_loss": self._avg_loss,
            "kelly_size": self._compute_kelly_size(),
            "reward_stats": self._reward_fn.get_stats(),
            "torch_available": TORCH_AVAILABLE,
        }
