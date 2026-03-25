"""
Hierarchical RL: Macro + Micro Policies for Trading.

Two-level policy hierarchy:
- MacroPolicy: Regime/strategy selection, updates every 5 minutes
  - Actions: aggressive, balanced, defensive, pause
  - State: regime features, portfolio state, recent performance
- MicroPolicy: Execution timing, updates per trade
  - Actions: execute_now, wait_5s, wait_30s, cancel
  - State: order book proxy, congestion, spread estimate
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Macro policy dimensions
MACRO_STATE_DIM = 20  # regime(4) + portfolio(10) + perf(6)
MACRO_ACTIONS = 4  # aggressive, balanced, defensive, pause

# Micro policy dimensions
MICRO_STATE_DIM = 15  # orderbook(5) + congestion(3) + spread(2) + macro_ctx(5)
MICRO_ACTIONS = 4  # execute_now, wait_5s, wait_30s, cancel


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical RL."""

    # Macro policy
    macro_hidden: list[int] = field(default_factory=lambda: [128, 64])
    macro_lr: float = 1e-3
    macro_update_interval_s: float = 300.0  # 5 minutes
    macro_gamma: float = 0.99
    macro_entropy_coef: float = 0.05  # Higher exploration at macro level

    # Micro policy
    micro_hidden: list[int] = field(default_factory=lambda: [64, 32])
    micro_lr: float = 3e-4
    micro_gamma: float = 0.95  # Shorter horizon for execution
    micro_entropy_coef: float = 0.02

    # Training
    batch_size: int = 32
    buffer_size: int = 1024
    n_epochs: int = 5
    clip_epsilon: float = 0.2


@dataclass
class MacroState:
    """State for macro policy (strategy selection)."""

    # Regime features (one-hot or probabilities for 4 regimes)
    regime_probs: list[float] = field(default_factory=lambda: [0.25] * 4)

    # Portfolio state
    total_value_usd: float = 0.0
    n_open_positions: int = 0
    portfolio_pnl_pct: float = 0.0
    cash_fraction: float = 1.0
    avg_position_age_min: float = 0.0
    max_drawdown_pct: float = 0.0
    daily_pnl_pct: float = 0.0
    exposure_pct: float = 0.0
    concentration_hhi: float = 0.0
    correlation_avg: float = 0.0

    # Recent performance window
    win_rate_1h: float = 0.5
    profit_factor_1h: float = 1.0
    sharpe_1h: float = 0.0
    trades_count_1h: int = 0
    avg_return_1h: float = 0.0
    vol_1h: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to 20-dim state vector."""
        v = np.array(
            self.regime_probs[:4]  # 4
            + [
                np.log1p(self.total_value_usd) / 15.0,
                self.n_open_positions / 20.0,
                np.tanh(self.portfolio_pnl_pct / 10.0),
                self.cash_fraction,
                min(self.avg_position_age_min / 60.0, 1.0),
                self.max_drawdown_pct / 100.0,
                np.tanh(self.daily_pnl_pct / 5.0),
                self.exposure_pct / 100.0,
                self.concentration_hhi,
                self.correlation_avg,
            ]  # 10
            + [
                self.win_rate_1h,
                np.log1p(max(0, self.profit_factor_1h - 1.0)),
                np.tanh(self.sharpe_1h),
                min(self.trades_count_1h / 50.0, 1.0),
                np.tanh(self.avg_return_1h),
                min(self.vol_1h, 1.0),
            ],  # 6
            dtype=np.float64,
        )
        return np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)


@dataclass
class MicroState:
    """State for micro policy (execution timing)."""

    # Order book proxy (aggregated from DEX pool data)
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    spread_bps: float = 0.0
    imbalance_ratio: float = 0.5
    mid_price_change_bps: float = 0.0

    # Congestion metrics
    tps_current: float = 0.0
    avg_slot_time_ms: float = 400.0
    priority_fee_lamports: float = 0.0

    # Spread and cost estimate
    estimated_slippage_bps: float = 0.0
    implementation_shortfall_bps: float = 0.0

    # Macro context (from macro policy decision)
    macro_action: int = 1  # 0=aggressive, 1=balanced, 2=defensive, 3=pause
    confidence_score: float = 0.5
    urgency: float = 0.5
    signal_strength: float = 0.0
    time_since_signal_s: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to 15-dim state vector."""
        v = np.array(
            [
                np.log1p(self.bid_depth_usd) / 15.0,
                np.log1p(self.ask_depth_usd) / 15.0,
                min(self.spread_bps / 500.0, 1.0),
                self.imbalance_ratio,
                np.tanh(self.mid_price_change_bps / 100.0),
                min(self.tps_current / 5000.0, 1.0),
                min(self.avg_slot_time_ms / 1000.0, 1.0),
                np.log1p(self.priority_fee_lamports) / 20.0,
                min(self.estimated_slippage_bps / 500.0, 1.0),
                np.tanh(self.implementation_shortfall_bps / 200.0),
                self.macro_action / 3.0,
                self.confidence_score,
                self.urgency,
                np.tanh(self.signal_strength),
                min(self.time_since_signal_s / 60.0, 1.0),
            ],
            dtype=np.float64,
        )
        return np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)


class PolicyBuffer:
    """Simple experience buffer for policy gradient training."""

    def __init__(self, max_size: int = 1024):
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []
        self.max_size = max_size

    def add(self, state, action, reward, log_prob, value, done):
        if len(self.states) >= self.max_size:
            self.clear()  # Reset when full
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def is_ready(self, min_size: int = 32) -> bool:
        return len(self.states) >= min_size

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


if TORCH_AVAILABLE:

    class _PolicyNet(nn.Module):
        """Shared Actor-Critic policy network."""

        def __init__(self, state_dim: int, n_actions: int, hidden: list[int]):
            super().__init__()
            layers = []
            in_dim = state_dim
            for h in hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.Tanh())
                in_dim = h
            self.shared = nn.Sequential(*layers)
            self.actor = nn.Linear(in_dim, n_actions)
            self.critic = nn.Linear(in_dim, 1)

        def forward(self, state):
            features = self.shared(state)
            logits = self.actor(features)
            value = self.critic(features).squeeze(-1)
            return logits, value

        def get_action(self, state, deterministic=False):
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            return action, dist.log_prob(action), value

        def evaluate(self, states, actions):
            logits, values = self.forward(states)
            dist = Categorical(logits=logits)
            return dist.log_prob(actions), dist.entropy(), values


class MacroPolicy:
    """High-level strategy selection policy.

    Decides the overall trading posture (aggressive/balanced/defensive/pause)
    based on regime, portfolio state, and recent performance.
    Updates every ~5 minutes.
    """

    ACTION_NAMES = ["aggressive", "balanced", "defensive", "pause"]

    def __init__(self, config: HierarchicalConfig | None = None):
        self.config = config or HierarchicalConfig()
        self._buffer = PolicyBuffer(self.config.buffer_size)
        self._total_steps = 0
        self._current_action = 1  # Default: balanced

        if TORCH_AVAILABLE:
            self._net = _PolicyNet(MACRO_STATE_DIM, MACRO_ACTIONS, self.config.macro_hidden)
            self._optimizer = optim.Adam(self._net.parameters(), lr=self.config.macro_lr)
        else:
            self._net = None
            self._optimizer = None

    def select_action(self, state: MacroState, deterministic: bool = False) -> dict[str, Any]:
        """Select macro action given current state."""
        state_vec = state.to_vector()

        if not TORCH_AVAILABLE or self._net is None:
            return self._heuristic_action(state)

        with torch.no_grad():
            state_t = torch.FloatTensor(state_vec).unsqueeze(0)
            action, log_prob, value = self._net.get_action(state_t, deterministic)

        self._current_action = int(action.item())
        self._total_steps += 1

        return {
            "action": self._current_action,
            "action_name": self.ACTION_NAMES[self._current_action],
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
            "confidence": float(torch.softmax(self._net(state_t)[0], dim=-1).max().item()),
        }

    def step(self, state: MacroState, action_info: dict, reward: float, done: bool = False):
        """Record a transition and train if buffer ready."""
        state_vec = state.to_vector()
        self._buffer.add(
            state_vec,
            action_info["action"],
            reward,
            action_info["log_prob"],
            action_info["value"],
            done,
        )
        if self._buffer.is_ready(self.config.batch_size):
            self._train()

    def _train(self):
        """PPO-style update on macro buffer."""
        if not TORCH_AVAILABLE or self._net is None:
            self._buffer.clear()
            return

        states = torch.FloatTensor(np.array(self._buffer.states))
        actions = torch.LongTensor(self._buffer.actions)
        old_log_probs = torch.FloatTensor(self._buffer.log_probs)

        # Compute returns
        returns = self._compute_returns(self._buffer.rewards, self._buffer.dones)
        returns_t = torch.FloatTensor(returns)

        for _ in range(self.config.n_epochs):
            new_log_probs, entropy, values = self._net.evaluate(states, actions)
            advantages = returns_t - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            clip_low = 1 - self.config.clip_epsilon
            clip_high = 1 + self.config.clip_epsilon
            surr2 = torch.clamp(ratio, clip_low, clip_high) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (values - returns_t).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + 0.5 * value_loss + self.config.macro_entropy_coef * entropy_loss

            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), 0.5)
            self._optimizer.step()

        self._buffer.clear()

    def _compute_returns(self, rewards, dones) -> list[float]:
        """Compute discounted returns."""
        returns = []
        R = 0.0
        for r, d in zip(reversed(rewards), reversed(dones), strict=False):
            R = r + (0.0 if d else self.config.macro_gamma * R)
            returns.insert(0, R)
        return returns

    def _heuristic_action(self, state: MacroState) -> dict[str, Any]:
        """Fallback heuristic when PyTorch unavailable."""
        # Simple rule: defensive if drawdown high, aggressive if winning
        if state.max_drawdown_pct > 15:
            action = 3  # pause
        elif state.max_drawdown_pct > 8:
            action = 2  # defensive
        elif state.win_rate_1h > 0.6 and state.profit_factor_1h > 1.5:
            action = 0  # aggressive
        else:
            action = 1  # balanced

        return {
            "action": action,
            "action_name": self.ACTION_NAMES[action],
            "log_prob": 0.0,
            "value": 0.0,
            "confidence": 0.5,
        }

    @property
    def current_action(self) -> int:
        return self._current_action

    @property
    def current_action_name(self) -> str:
        return self.ACTION_NAMES[self._current_action]


class MicroPolicy:
    """Low-level execution timing policy.

    Decides when to execute a trade (now, wait, cancel) based on
    current market microstructure and congestion conditions.
    Updates per trade.
    """

    ACTION_NAMES = ["execute_now", "wait_5s", "wait_30s", "cancel"]

    def __init__(self, config: HierarchicalConfig | None = None):
        self.config = config or HierarchicalConfig()
        self._buffer = PolicyBuffer(self.config.buffer_size)
        self._total_steps = 0

        if TORCH_AVAILABLE:
            self._net = _PolicyNet(MICRO_STATE_DIM, MICRO_ACTIONS, self.config.micro_hidden)
            self._optimizer = optim.Adam(self._net.parameters(), lr=self.config.micro_lr)
        else:
            self._net = None
            self._optimizer = None

    def select_action(self, state: MicroState, deterministic: bool = False) -> dict[str, Any]:
        """Select micro action given execution context."""
        state_vec = state.to_vector()

        if not TORCH_AVAILABLE or self._net is None:
            return self._heuristic_action(state)

        with torch.no_grad():
            state_t = torch.FloatTensor(state_vec).unsqueeze(0)
            action, log_prob, value = self._net.get_action(state_t, deterministic)

        self._total_steps += 1

        return {
            "action": int(action.item()),
            "action_name": self.ACTION_NAMES[int(action.item())],
            "delay_seconds": [0, 5, 30, -1][int(action.item())],  # -1 = cancel
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
        }

    def step(self, state: MicroState, action_info: dict, reward: float, done: bool = False):
        """Record transition and train if ready."""
        state_vec = state.to_vector()
        self._buffer.add(
            state_vec,
            action_info["action"],
            reward,
            action_info["log_prob"],
            action_info["value"],
            done,
        )
        if self._buffer.is_ready(self.config.batch_size):
            self._train()

    def _train(self):
        """PPO-style update on micro buffer."""
        if not TORCH_AVAILABLE or self._net is None:
            self._buffer.clear()
            return

        states = torch.FloatTensor(np.array(self._buffer.states))
        actions = torch.LongTensor(self._buffer.actions)
        old_log_probs = torch.FloatTensor(self._buffer.log_probs)
        returns = self._compute_returns(self._buffer.rewards, self._buffer.dones)
        returns_t = torch.FloatTensor(returns)

        for _ in range(self.config.n_epochs):
            new_log_probs, entropy, values = self._net.evaluate(states, actions)
            advantages = returns_t - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            clip_low = 1 - self.config.clip_epsilon
            clip_high = 1 + self.config.clip_epsilon
            surr2 = torch.clamp(ratio, clip_low, clip_high) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (values - returns_t).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + 0.5 * value_loss + self.config.micro_entropy_coef * entropy_loss

            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), 0.5)
            self._optimizer.step()

        self._buffer.clear()

    def _compute_returns(self, rewards, dones) -> list[float]:
        returns = []
        R = 0.0
        for r, d in zip(reversed(rewards), reversed(dones), strict=False):
            R = r + (0.0 if d else self.config.micro_gamma * R)
            returns.insert(0, R)
        return returns

    def _heuristic_action(self, state: MicroState) -> dict[str, Any]:
        """Fallback heuristic for execution timing."""
        if state.macro_action == 3:  # pause
            action = 3  # cancel
        elif state.avg_slot_time_ms > 600 or state.estimated_slippage_bps > 300:
            action = 2  # wait_30s
        elif state.tps_current > 3000 or state.spread_bps > 100:
            action = 1  # wait_5s
        else:
            action = 0  # execute_now

        return {
            "action": action,
            "action_name": self.ACTION_NAMES[action],
            "delay_seconds": [0, 5, 30, -1][action],
            "log_prob": 0.0,
            "value": 0.0,
        }


class HierarchicalController:
    """Coordinates macro and micro policies for two-level RL.

    Usage:
        controller = HierarchicalController()
        # Every 5 minutes:
        macro_action = controller.macro_step(macro_state, macro_reward)
        # Per trade:
        micro_action = controller.micro_step(micro_state, macro_action, micro_reward)
    """

    def __init__(self, config: HierarchicalConfig | None = None):
        self.config = config or HierarchicalConfig()
        self.macro = MacroPolicy(self.config)
        self.micro = MicroPolicy(self.config)

        self._last_macro_action: dict[str, Any] | None = None
        self._last_macro_state: MacroState | None = None

    def macro_step(
        self,
        state: MacroState,
        reward: float | None = None,
        done: bool = False,
    ) -> dict[str, Any]:
        """Update macro policy and get new strategy posture.

        Args:
            state: Current macro state
            reward: Reward from previous macro action (None if first step)
            done: Episode boundary

        Returns:
            Macro action dictionary
        """
        # Record previous transition
        has_prev = (
            reward is not None
            and self._last_macro_action is not None
            and self._last_macro_state is not None
        )
        if has_prev:
            self.macro.step(self._last_macro_state, self._last_macro_action, reward, done)

        # Select new action
        action = self.macro.select_action(state)
        self._last_macro_action = action
        self._last_macro_state = state

        return action

    def micro_step(
        self,
        state: MicroState,
        reward: float | None = None,
        done: bool = False,
    ) -> dict[str, Any]:
        """Get execution timing decision.

        Args:
            state: Current micro state (includes macro context)
            reward: Reward from previous micro action
            done: Trade boundary

        Returns:
            Micro action dictionary
        """
        return self.micro.select_action(state)

    def record_micro_reward(
        self, state: MicroState, action: dict, reward: float, done: bool = False
    ):
        """Record micro policy reward after trade execution."""
        self.micro.step(state, action, reward, done)

    def get_stats(self) -> dict[str, Any]:
        """Get controller statistics."""
        return {
            "macro_action": self.macro.current_action_name,
            "macro_steps": self.macro._total_steps,
            "micro_steps": self.micro._total_steps,
            "macro_buffer_size": len(self.macro._buffer),
            "micro_buffer_size": len(self.micro._buffer),
            "torch_available": TORCH_AVAILABLE,
        }
