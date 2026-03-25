"""
Contextual bandit for context-aware slippage selection.

Implements LinUCB (Linear Upper Confidence Bound) algorithm
for adaptive slippage tolerance based on market context.

Context features:
- Liquidity
- Volatility
- Volume
- Holder concentration
- MEV activity

Output: Optimal slippage tolerance (10-1000 bps)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContextFeatures:
    """Context features for bandit decision making."""

    liquidity_usd: float  # Pool liquidity
    volatility: float  # Price volatility (0-1 normalized)
    volume_24h_usd: float  # 24h trading volume
    holder_concentration: float  # Top holder percentage (0-100)
    mev_activity: float  # MEV activity level (0-1)

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for model input."""
        # Normalize features to reasonable ranges
        features = np.array(
            [
                np.log1p(self.liquidity_usd) / 15.0,  # Log-normalized
                self.volatility,  # Already in [0, 1]
                np.log1p(self.volume_24h_usd) / 15.0,  # Log-normalized
                self.holder_concentration / 100.0,  # [0, 1]
                self.mev_activity,  # Already in [0, 1]
                1.0,  # Bias term
            ],
            dtype=np.float64,
        )
        return features

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextFeatures":
        """Create from dictionary."""
        return cls(
            liquidity_usd=data.get("liquidity_usd", 10000),
            volatility=data.get("volatility", 0.1),
            volume_24h_usd=data.get("volume_24h_usd", 5000),
            holder_concentration=data.get("holder_concentration", 50),
            mev_activity=data.get("mev_activity", 0.1),
        )


@dataclass
class LinUCBConfig:
    """Configuration for LinUCB algorithm."""

    # Slippage arms (bps)
    slippage_arms: list[int] = field(default_factory=lambda: [10, 50, 100, 200, 300, 500, 1000])

    # Algorithm parameters
    alpha: float = 1.0  # Exploration parameter
    regularization: float = 1.0  # Ridge regularization

    # Feature dimension (including bias)
    n_features: int = 6

    # Learning rate for online updates
    learning_rate: float = 1.0

    # Minimum exploration count per arm
    min_arm_pulls: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "slippage_arms": self.slippage_arms,
            "alpha": self.alpha,
            "regularization": self.regularization,
            "n_features": self.n_features,
        }


@dataclass
class BanditState:
    """State of a single arm."""

    arm_id: int
    slippage_bps: int
    n_pulls: int = 0
    total_reward: float = 0.0
    A: np.ndarray = None  # d x d matrix
    b: np.ndarray = None  # d x 1 vector

    def __post_init__(self):
        if self.A is None:
            self.A = np.eye(6)  # Default to 6 features
        if self.b is None:
            self.b = np.zeros(6)


class LinUCBBandit:
    """LinUCB contextual bandit for slippage selection.

    Uses linear regression with upper confidence bounds to balance
    exploration and exploitation in slippage tolerance selection.

    Features:
    - Context-aware: considers market conditions
    - Continuous learning: updates from trade outcomes
    - Uncertainty-aware: explores when unsure

    Example:
        bandit = LinUCBBandit()
        context = ContextFeatures(liquidity_usd=50000, volatility=0.2, ...)
        slippage = bandit.select_slippage(context)
        # After trade outcome
        bandit.update(context, slippage, reward)
    """

    def __init__(self, config: LinUCBConfig | None = None):
        self.config = config or LinUCBConfig()

        # Initialize arms
        self.arms: dict[int, BanditState] = {}
        for i, slippage in enumerate(self.config.slippage_arms):
            self.arms[slippage] = BanditState(
                arm_id=i,
                slippage_bps=slippage,
                A=np.eye(self.config.n_features) * self.config.regularization,
                b=np.zeros(self.config.n_features),
            )

        self.total_pulls = 0

    def select_slippage(self, context: ContextFeatures) -> int:
        """Select optimal slippage tolerance using LinUCB.

        Args:
            context: Current market context features

        Returns:
            Optimal slippage in basis points
        """
        x = context.to_vector()

        # Ensure all arms have minimum pulls first
        for slippage, arm in self.arms.items():
            if arm.n_pulls < self.config.min_arm_pulls:
                return slippage

        best_slippage = self.config.slippage_arms[0]
        best_ucb = float("-inf")

        for slippage, arm in self.arms.items():
            ucb = self._compute_ucb(arm, x)
            if ucb > best_ucb:
                best_ucb = ucb
                best_slippage = slippage

        return best_slippage

    def _compute_ucb(self, arm: BanditState, x: np.ndarray) -> float:
        """Compute Upper Confidence Bound for an arm."""
        try:
            A_inv = np.linalg.inv(arm.A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(arm.A)

        # Expected reward
        theta = A_inv @ arm.b
        expected_reward = float(x @ theta)

        # Uncertainty (exploration bonus)
        uncertainty = self.config.alpha * np.sqrt(float(x @ A_inv @ x))

        return expected_reward + uncertainty

    def update(self, context: ContextFeatures, slippage_bps: int, reward: float):
        """Update bandit with observed reward.

        Args:
            context: Context features when decision was made
            slippage_bps: Selected slippage tolerance
            reward: Observed reward (higher is better)
        """
        if slippage_bps not in self.arms:
            logger.warning(f"Unknown slippage arm: {slippage_bps}")
            slippage_bps = self._nearest_arm(slippage_bps)

        arm = self.arms[slippage_bps]
        x = context.to_vector()

        # Update statistics
        arm.n_pulls += 1
        arm.total_reward += reward
        self.total_pulls += 1

        # Update A matrix and b vector (ridge regression update)
        arm.A += np.outer(x, x) * self.config.learning_rate
        arm.b += x * reward * self.config.learning_rate

    def _nearest_arm(self, slippage_bps: int) -> int:
        """Find nearest available arm."""
        available = list(self.arms.keys())
        return min(available, key=lambda s: abs(s - slippage_bps))

    def get_recommendation(self, context: ContextFeatures | None = None) -> dict[str, Any]:
        """Get current recommendation with confidence.

        Args:
            context: Optional context for specific recommendation

        Returns:
            Dictionary with recommendation details
        """
        if context is None:
            # Use default context
            context = ContextFeatures(
                liquidity_usd=10000,
                volatility=0.2,
                volume_24h_usd=5000,
                holder_concentration=50,
                mev_activity=0.1,
            )

        x = context.to_vector()

        best_slippage = self.select_slippage(context)
        best_arm = self.arms[best_slippage]

        # Compute confidence
        try:
            A_inv = np.linalg.inv(best_arm.A)
            uncertainty = self.config.alpha * np.sqrt(float(x @ A_inv @ x))
        except np.linalg.LinAlgError:
            uncertainty = 1.0

        confidence = 1.0 / (1.0 + uncertainty)

        return {
            "slippage_bps": best_slippage,
            "confidence": float(confidence),
            "mean_reward": float(best_arm.total_reward / max(1, best_arm.n_pulls)),
            "n_pulls": best_arm.n_pulls,
            "total_pulls": self.total_pulls,
            "uncertainty": float(uncertainty),
        }

    def calculate_reward(self, trade: dict[str, Any]) -> float:
        """Calculate reward from trade outcome.

        Reward considers:
        - PnL (primary)
        - Transaction inclusion
        - Slippage efficiency

        Args:
            trade: Trade outcome dictionary

        Returns:
            Reward value (higher is better)
        """
        pnl_pct = trade.get("pnl_pct", 0)
        included = trade.get("included", True)
        realized_slippage = trade.get("realized_slippage_bps", 0)
        quoted_slippage = trade.get("quoted_slippage_bps", 300)

        # Non-inclusion is very bad
        if not included:
            return -1.0

        # PnL component (tanh to bound)
        pnl_reward = np.tanh(pnl_pct / 20)

        # Slippage efficiency
        if quoted_slippage > 0:
            slippage_efficiency = (quoted_slippage - realized_slippage) / quoted_slippage
            slippage_bonus = slippage_efficiency * 0.3
        else:
            slippage_bonus = 0

        return float(pnl_reward + slippage_bonus)

    def get_arm_statistics(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all arms."""
        stats = {}
        for slippage, arm in self.arms.items():
            stats[str(slippage)] = {
                "n_pulls": arm.n_pulls,
                "mean_reward": arm.total_reward / max(1, arm.n_pulls),
                "total_reward": arm.total_reward,
            }
        return stats

    def to_params(self) -> dict[str, Any]:
        """Export bandit state for HotPath."""
        return {
            "config": self.config.to_dict(),
            "total_pulls": self.total_pulls,
            "arm_statistics": self.get_arm_statistics(),
            "default_recommendation": self.get_recommendation(),
        }

    def to_state(self) -> dict[str, Any]:
        """Export full state for persistence."""
        return {
            "config": self.config.to_dict(),
            "total_pulls": self.total_pulls,
            "arms": {
                str(slippage): {
                    "arm_id": arm.arm_id,
                    "slippage_bps": arm.slippage_bps,
                    "n_pulls": arm.n_pulls,
                    "total_reward": arm.total_reward,
                    "A": arm.A.tolist(),
                    "b": arm.b.tolist(),
                }
                for slippage, arm in self.arms.items()
            },
        }

    def load_state(self, state: dict[str, Any]):
        """Load state from persistence."""
        if "config" in state:
            config = state["config"]
            self.config = LinUCBConfig(
                slippage_arms=config.get("slippage_arms", self.config.slippage_arms),
                alpha=config.get("alpha", self.config.alpha),
                regularization=config.get("regularization", self.config.regularization),
                n_features=config.get("n_features", self.config.n_features),
            )

        self.total_pulls = state.get("total_pulls", 0)

        if "arms" in state:
            for slippage_str, arm_data in state["arms"].items():
                slippage = int(slippage_str)
                if slippage not in self.arms:
                    continue

                arm = self.arms[slippage]
                arm.arm_id = arm_data.get("arm_id", arm.arm_id)
                arm.n_pulls = arm_data.get("n_pulls", 0)
                arm.total_reward = arm_data.get("total_reward", 0.0)

                if "A" in arm_data:
                    arm.A = np.array(arm_data["A"])
                if "b" in arm_data:
                    arm.b = np.array(arm_data["b"])


class ThompsonSamplingBandit:
    """Thompson Sampling variant of the contextual bandit.

    Uses Bayesian posterior updates for exploration:
    - Maintains Normal-InverseGamma posterior per arm
    - Samples from posterior for action selection
    - More principled exploration than UCB
    """

    def __init__(self, config: LinUCBConfig | None = None):
        self.config = config or LinUCBConfig()

        # Per-arm posteriors: (mu, lambda, alpha, beta) for Normal-InverseGamma
        self.posteriors: dict[int, dict[str, float]] = {}
        for slippage in self.config.slippage_arms:
            self.posteriors[slippage] = {
                "mu": 0.0,  # Prior mean
                "lam": 1.0,  # Prior precision scaling
                "alpha": 2.0,  # Prior shape (InvGamma)
                "beta": 1.0,  # Prior rate (InvGamma)
                "n": 0,  # Number of observations
                "sum_reward": 0.0,
            }

        self.total_pulls = 0

    def select_slippage(self, context: ContextFeatures) -> int:
        """Select arm via Thompson Sampling (sample from posterior)."""
        best_slippage = self.config.slippage_arms[0]
        best_sample = float("-inf")

        for slippage, post in self.posteriors.items():
            # Sample variance from Inverse-Gamma
            if post["alpha"] > 0 and post["beta"] > 0:
                var_sample = 1.0 / np.random.gamma(post["alpha"], 1.0 / post["beta"])
            else:
                var_sample = 1.0

            # Sample mean from Normal(mu, var/lambda)
            std = np.sqrt(var_sample / max(post["lam"], 1e-8))
            mean_sample = np.random.normal(post["mu"], std)

            if mean_sample > best_sample:
                best_sample = mean_sample
                best_slippage = slippage

        return best_slippage

    def update(self, context: ContextFeatures, slippage_bps: int, reward: float):
        """Update posterior for the selected arm with observed reward."""
        if slippage_bps not in self.posteriors:
            nearest = min(self.posteriors.keys(), key=lambda s: abs(s - slippage_bps))
            slippage_bps = nearest

        post = self.posteriors[slippage_bps]
        n = post["n"]
        old_mu = post["mu"]
        old_lam = post["lam"]

        # Normal-InverseGamma conjugate update
        new_lam = old_lam + 1
        new_mu = (old_lam * old_mu + reward) / new_lam
        new_alpha = post["alpha"] + 0.5
        new_beta = post["beta"] + 0.5 * old_lam * (reward - old_mu) ** 2 / new_lam

        post["mu"] = new_mu
        post["lam"] = new_lam
        post["alpha"] = new_alpha
        post["beta"] = new_beta
        post["n"] = n + 1
        post["sum_reward"] += reward

        self.total_pulls += 1

    def get_recommendation(self) -> dict[str, Any]:
        """Get recommendation based on posterior means."""
        best_slippage = max(
            self.posteriors.items(),
            key=lambda x: x[1]["mu"],
        )
        return {
            "slippage_bps": best_slippage[0],
            "expected_reward": best_slippage[1]["mu"],
            "n_samples": best_slippage[1]["n"],
            "total_pulls": self.total_pulls,
        }

    def to_params(self) -> dict[str, Any]:
        """Export for HotPath."""
        return {
            "type": "thompson_sampling",
            "total_pulls": self.total_pulls,
            "posteriors": {str(k): v for k, v in self.posteriors.items()},
        }

    def to_state(self) -> dict[str, Any]:
        """Export full state for persistence."""
        return {
            "config": self.config.to_dict(),
            "total_pulls": self.total_pulls,
            "posteriors": {str(k): dict(v) for k, v in self.posteriors.items()},
        }

    def load_state(self, state: dict[str, Any]):
        """Load state from persistence."""
        self.total_pulls = state.get("total_pulls", 0)
        if "posteriors" in state:
            for slippage_str, post_data in state["posteriors"].items():
                slippage = int(slippage_str)
                if slippage in self.posteriors:
                    self.posteriors[slippage].update(post_data)


def map_ucb_to_slippage(ucb_value: float, min_bps: int = 10, max_bps: int = 1000) -> int:
    """Map continuous UCB value to discrete slippage.

    Args:
        ucb_value: UCB score (unbounded)
        min_bps: Minimum slippage
        max_bps: Maximum slippage

    Returns:
        Slippage in basis points
    """
    # Sigmoid transform to [0, 1]
    sigmoid = 1.0 / (1.0 + np.exp(-ucb_value))

    # Map to slippage range (log scale for better distribution)
    log_min = np.log(min_bps)
    log_max = np.log(max_bps)
    log_slippage = log_min + sigmoid * (log_max - log_min)

    return int(np.exp(log_slippage))
