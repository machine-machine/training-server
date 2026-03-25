"""
Model-Based RL with Learned Market Dynamics.

Learns P(s'|s,a) transition model for planning:
- Uses a neural network to predict next-state and reward
- Dyna-style integration: real experience + imagined trajectories
- CEM (Cross-Entropy Method) for action selection via rollouts
- Reduces sample complexity for PPO training
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

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Dimensions
STATE_DIM = 65  # Same as PPO state
ACTION_DIM = 3  # position_size(1) + duration_class(1) + exit_class(1)


@dataclass
class WorldModelConfig:
    """Configuration for the world model."""

    # Network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_buffer_size: int = 10000
    train_every_n: int = 50  # Train world model every N real experiences
    min_train_samples: int = 200

    # Planning (CEM)
    n_rollouts: int = 50  # Number of imagined trajectories
    rollout_horizon: int = 5  # Steps to look ahead
    cem_elite_frac: float = 0.2  # Top 20% of rollouts
    cem_iterations: int = 3

    # Dyna integration
    dyna_ratio: int = 5  # Generate N imagined experiences per real one

    # Uncertainty
    ensemble_size: int = 3  # Number of models for uncertainty estimation


if TORCH_AVAILABLE:

    class DynamicsNetwork(nn.Module):
        """Predicts next state and reward given (state, action)."""

        def __init__(self, config: WorldModelConfig):
            super().__init__()
            input_dim = STATE_DIM + ACTION_DIM

            layers = []
            in_dim = input_dim
            for h in config.hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(h))
                in_dim = h

            self.shared = nn.Sequential(*layers)

            # State prediction head (deterministic for now)
            self.state_head = nn.Linear(in_dim, STATE_DIM)

            # Reward prediction head
            self.reward_head = nn.Linear(in_dim, 1)

            # Done prediction head (binary)
            self.done_head = nn.Linear(in_dim, 1)

        def forward(self, state: torch.Tensor, action: torch.Tensor):
            """Predict next state, reward, and done from (s, a).

            Args:
                state: [batch, STATE_DIM]
                action: [batch, ACTION_DIM]

            Returns:
                next_state: [batch, STATE_DIM]
                reward: [batch]
                done_prob: [batch]
            """
            x = torch.cat([state, action], dim=-1)
            features = self.shared(x)

            # Predict state delta (residual prediction is more stable)
            state_delta = self.state_head(features)
            next_state = state + state_delta

            reward = self.reward_head(features).squeeze(-1)
            done_prob = torch.sigmoid(self.done_head(features).squeeze(-1))

            return next_state, reward, done_prob


class WorldModel:
    """Learned dynamics model for model-based RL planning.

    Maintains an ensemble of dynamics networks for uncertainty estimation.
    Generates imagined trajectories for Dyna-style training augmentation.
    Uses CEM for model-based action selection.
    """

    def __init__(self, config: WorldModelConfig | None = None):
        self.config = config or WorldModelConfig()

        # Experience buffer for training the world model
        self._states: deque = deque(maxlen=self.config.max_buffer_size)
        self._actions: deque = deque(maxlen=self.config.max_buffer_size)
        self._rewards: deque = deque(maxlen=self.config.max_buffer_size)
        self._next_states: deque = deque(maxlen=self.config.max_buffer_size)
        self._dones: deque = deque(maxlen=self.config.max_buffer_size)

        self._total_added = 0
        self._total_trained = 0

        if TORCH_AVAILABLE:
            # Ensemble of dynamics models
            self._models = [DynamicsNetwork(self.config) for _ in range(self.config.ensemble_size)]
            self._optimizers = [
                optim.Adam(m.parameters(), lr=self.config.learning_rate) for m in self._models
            ]
        else:
            self._models = []
            self._optimizers = []

    def add_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a real experience for world model training.

        Args:
            state: Current state [65]
            action: Action vector [3] (position_size, duration_class, exit_class)
            reward: Scalar reward
            next_state: Next state [65]
            done: Episode boundary
        """
        self._states.append(state.copy())
        self._actions.append(action.copy())
        self._rewards.append(reward)
        self._next_states.append(next_state.copy())
        self._dones.append(done)
        self._total_added += 1

        # Train periodically
        if self._total_added % self.config.train_every_n == 0:
            self.train()

    def train(self, n_epochs: int = 5) -> dict[str, float]:
        """Train world model ensemble on collected data.

        Returns:
            Training metrics (loss per model)
        """
        if not TORCH_AVAILABLE or len(self._states) < self.config.min_train_samples:
            return {"status": "skipped", "samples": len(self._states)}

        states = torch.FloatTensor(np.array(list(self._states)))
        actions = torch.FloatTensor(np.array(list(self._actions)))
        rewards = torch.FloatTensor(np.array(list(self._rewards)))
        next_states = torch.FloatTensor(np.array(list(self._next_states)))
        dones = torch.FloatTensor(np.array(list(self._dones), dtype=np.float32))

        n = len(states)
        losses = {}

        for model_idx, (model, optimizer) in enumerate(
            zip(self._models, self._optimizers, strict=False)
        ):
            model.train()
            epoch_losses = []

            for _ in range(n_epochs):
                # Random mini-batch (each model sees different data for diversity)
                indices = np.random.choice(n, min(self.config.batch_size, n), replace=False)
                s = states[indices]
                a = actions[indices]
                r = rewards[indices]
                ns = next_states[indices]
                d = dones[indices]

                # Forward pass
                pred_ns, pred_r, pred_done = model(s, a)

                # Loss: state MSE + reward MSE + done BCE
                state_loss = nn.functional.mse_loss(pred_ns, ns)
                reward_loss = nn.functional.mse_loss(pred_r, r)
                done_loss = nn.functional.binary_cross_entropy(pred_done, d)

                loss = state_loss + reward_loss + 0.1 * done_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

            losses[f"model_{model_idx}"] = np.mean(epoch_losses)

        self._total_trained += 1
        return losses

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float]:
        """Predict next state, reward, done using ensemble mean.

        Returns:
            (next_state, reward, done_prob, uncertainty)
        """
        if not TORCH_AVAILABLE or not self._models:
            # Naive fallback: return same state
            return state.copy(), 0.0, 0.0, 1.0

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            a = torch.FloatTensor(action).unsqueeze(0)

            next_states = []
            rewards = []
            dones = []

            for model in self._models:
                model.eval()
                ns, r, d = model(s, a)
                next_states.append(ns.numpy()[0])
                rewards.append(r.item())
                dones.append(d.item())

        # Ensemble mean
        mean_ns = np.mean(next_states, axis=0)
        mean_r = np.mean(rewards)
        mean_d = np.mean(dones)

        # Uncertainty from ensemble disagreement
        uncertainty = np.mean(np.std(next_states, axis=0))

        return mean_ns, mean_r, mean_d, uncertainty

    def generate_imagined_experiences(
        self,
        start_states: list[np.ndarray],
        action_sampler,  # Callable: state -> action
        n_steps: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate imagined trajectories for Dyna-style training.

        Args:
            start_states: Initial states for rollouts
            action_sampler: Function that returns action given state
            n_steps: Rollout length

        Returns:
            List of (state, action, reward, next_state, done) dicts
        """
        experiences = []

        for state in start_states:
            s = state.copy()
            for _ in range(n_steps):
                a = action_sampler(s)
                ns, r, d_prob, uncertainty = self.predict(s, a)

                # Only add if uncertainty is reasonable
                if uncertainty < 0.5:
                    experiences.append(
                        {
                            "state": s,
                            "action": a,
                            "reward": r,
                            "next_state": ns,
                            "done": d_prob > 0.5,
                            "imagined": True,
                            "uncertainty": uncertainty,
                        }
                    )

                if d_prob > 0.5:
                    break
                s = ns

        return experiences

    def cem_plan(
        self,
        state: np.ndarray,
        reward_fn=None,
    ) -> np.ndarray:
        """Plan best action using Cross-Entropy Method (CEM).

        Rolls out imagined trajectories with different action sequences,
        selects the top-performing sequences, and refines.

        Args:
            state: Current state [65]
            reward_fn: Optional reward function for evaluation

        Returns:
            Best action [3]
        """
        if not TORCH_AVAILABLE or not self._models:
            return np.array([0.5, 2, 1], dtype=np.float64)  # Default balanced action

        n_rollouts = self.config.n_rollouts
        horizon = self.config.rollout_horizon
        n_elite = max(1, int(n_rollouts * self.config.cem_elite_frac))

        # Initialize action distribution
        action_mean = np.array([0.5, 2.0, 1.0])  # position, duration, exit
        action_std = np.array([0.3, 1.5, 0.8])

        for _cem_iter in range(self.config.cem_iterations):
            # Sample action sequences
            action_sequences = (
                np.random.randn(n_rollouts, horizon, ACTION_DIM) * action_std + action_mean
            )

            # Clip actions to valid ranges
            action_sequences[:, :, 0] = np.clip(action_sequences[:, :, 0], 0.0, 1.0)
            action_sequences[:, :, 1] = np.clip(np.round(action_sequences[:, :, 1]), 0, 4)
            action_sequences[:, :, 2] = np.clip(np.round(action_sequences[:, :, 2]), 0, 2)

            # Evaluate rollouts
            total_rewards = np.zeros(n_rollouts)
            for i in range(n_rollouts):
                s = state.copy()
                gamma = 0.99
                for t in range(horizon):
                    a = action_sequences[i, t]
                    ns, r, d_prob, _ = self.predict(s, a)
                    total_rewards[i] += (gamma**t) * r
                    if d_prob > 0.5:
                        break
                    s = ns

            # Select elite sequences
            elite_indices = np.argsort(total_rewards)[-n_elite:]
            elite_actions = action_sequences[elite_indices]

            # Update distribution
            action_mean = elite_actions[:, 0, :].mean(axis=0)
            action_std = elite_actions[:, 0, :].std(axis=0) + 0.01

        return action_mean

    def get_stats(self) -> dict[str, Any]:
        """Get world model statistics."""
        return {
            "buffer_size": len(self._states),
            "total_added": self._total_added,
            "total_trained": self._total_trained,
            "n_models": len(self._models),
            "torch_available": TORCH_AVAILABLE,
        }
