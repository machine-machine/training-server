"""
ML Proposal Ranker - Rank strategy proposals using advanced algorithms.

Implements full proposal ranking system:
├── Thompson Sampling (exploration/exploitation)
├── Contextual Bandits (regime-aware LinUCB)
├── Bayesian Optimization (Gaussian Process)
└── Opus 4.5 Analysis (explain trade-offs)

Pipeline:
1. Generate 50-100 candidates (bandit + perturbation + LLM)
2. Fast filter to top 10 (vectorized backtest)
3. Deep evaluation (walk-forward + Monte Carlo)
4. Opus analysis (structured recommendations)
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

try:
    from scipy.linalg import (
        cho_factor,  # noqa: F401
        cho_solve,  # noqa: F401
    )
    from scipy.stats import beta as beta_dist  # noqa: F401
    from scipy.stats import norm

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class RankingMethod(Enum):
    """Proposal ranking methods."""

    THOMPSON_SAMPLING = "thompson"
    LINUCB = "linucb"
    BAYESIAN_OPT = "bayesian"
    HYBRID = "hybrid"


@dataclass
class ProposalCandidate:
    """A strategy proposal candidate."""

    proposal_id: str
    parameters: dict[str, float]
    source: str  # "bandit", "perturbation", "llm", "random"
    parent_id: str | None = None
    generation: int = 0

    # Scores (filled during evaluation)
    fast_score: float | None = None
    backtest_sharpe: float | None = None
    monte_carlo_score: float | None = None
    llm_score: float | None = None
    final_score: float | None = None
    rank: int | None = None

    # Uncertainty estimates
    score_std: float | None = None
    confidence_interval: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.proposal_id,
            "parameters": self.parameters,
            "source": self.source,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "scores": {
                "fast": self.fast_score,
                "backtest_sharpe": self.backtest_sharpe,
                "monte_carlo": self.monte_carlo_score,
                "llm": self.llm_score,
                "final": self.final_score,
            },
            "rank": self.rank,
            "uncertainty": {
                "std": self.score_std,
                "ci": self.confidence_interval,
            },
        }


@dataclass
class RankingResult:
    """Result of proposal ranking."""

    ranked_proposals: list[ProposalCandidate]
    method: RankingMethod
    regime: str | None
    n_evaluated: int
    n_filtered: int
    best_proposal: ProposalCandidate
    exploration_ratio: float
    total_evaluation_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranked_proposals": [p.to_dict() for p in self.ranked_proposals[:10]],
            "method": self.method.value,
            "regime": self.regime,
            "n_evaluated": self.n_evaluated,
            "n_filtered": self.n_filtered,
            "best": self.best_proposal.to_dict(),
            "exploration_ratio": self.exploration_ratio,
            "evaluation_time_ms": self.total_evaluation_time_ms,
        }


@dataclass
class BanditArm:
    """Multi-armed bandit arm with statistics."""

    arm_id: str
    parameters: dict[str, float]
    pull_count: int = 0
    total_reward: float = 0.0
    reward_squared: float = 0.0

    # For Thompson Sampling (Beta prior)
    alpha: float = 1.0  # Success count + prior
    beta: float = 1.0  # Failure count + prior

    # For Gaussian Thompson Sampling
    mean: float = 0.0
    variance: float = 1.0

    @property
    def empirical_mean(self) -> float:
        return self.total_reward / self.pull_count if self.pull_count > 0 else 0.0

    @property
    def empirical_std(self) -> float:
        if self.pull_count < 2:
            return 1.0
        mean_sq = self.reward_squared / self.pull_count
        return math.sqrt(max(0, mean_sq - self.empirical_mean**2))

    def update(self, reward: float):
        """Update arm statistics with new reward."""
        self.pull_count += 1
        self.total_reward += reward
        self.reward_squared += reward**2

        # Update Beta parameters (for binary rewards)
        if reward > 0:
            self.alpha += 1
        else:
            self.beta += 1

        # Update Gaussian parameters (online mean/var update)
        delta = reward - self.mean
        self.mean += delta / self.pull_count
        delta2 = reward - self.mean
        self.variance = (self.variance * (self.pull_count - 1) + delta * delta2) / self.pull_count


class ThompsonSampler:
    """Thompson Sampling for multi-armed bandit exploration.

    Balances exploration/exploitation by sampling from posterior
    distributions over arm rewards.
    """

    def __init__(
        self,
        use_gaussian: bool = True,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
    ):
        """Initialize Thompson Sampler.

        Args:
            use_gaussian: If True, use Gaussian posteriors (for continuous rewards)
                         If False, use Beta posteriors (for binary rewards)
            prior_mean: Prior mean for Gaussian
            prior_variance: Prior variance for Gaussian
        """
        self.use_gaussian = use_gaussian
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.arms: dict[str, BanditArm] = {}

    def add_arm(self, arm_id: str, parameters: dict[str, float]):
        """Add a new arm."""
        self.arms[arm_id] = BanditArm(
            arm_id=arm_id,
            parameters=parameters,
            mean=self.prior_mean,
            variance=self.prior_variance,
        )

    def select_arm(self) -> str:
        """Select arm using Thompson Sampling.

        Returns:
            arm_id of selected arm
        """
        best_sample = -np.inf
        best_arm = None

        for arm_id, arm in self.arms.items():
            if self.use_gaussian:
                # Sample from Gaussian posterior
                std = math.sqrt(arm.variance / max(1, arm.pull_count))
                sample = np.random.normal(arm.mean, std)
            else:
                # Sample from Beta posterior
                sample = np.random.beta(arm.alpha, arm.beta)

            if sample > best_sample:
                best_sample = sample
                best_arm = arm_id

        return best_arm or list(self.arms.keys())[0]

    def update(self, arm_id: str, reward: float):
        """Update arm with observed reward."""
        if arm_id in self.arms:
            self.arms[arm_id].update(reward)

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics for all arms."""
        return {
            arm_id: {
                "pulls": arm.pull_count,
                "mean_reward": arm.empirical_mean,
                "std": arm.empirical_std,
                "ucb": arm.mean + 2 * math.sqrt(arm.variance / max(1, arm.pull_count)),
            }
            for arm_id, arm in self.arms.items()
        }


class LinUCBContextualBandit:
    """LinUCB contextual bandit for regime-aware proposal selection.

    Uses linear regression with UCB exploration bonus.
    Context vector includes regime features for regime-aware selection.
    """

    def __init__(
        self,
        n_features: int,
        alpha: float = 1.0,  # Exploration parameter
    ):
        """Initialize LinUCB bandit.

        Args:
            n_features: Dimension of context vector
            alpha: Exploration parameter (higher = more exploration)
        """
        self.n_features = n_features
        self.alpha = alpha

        # Per-arm models
        self.arms: dict[str, dict[str, np.ndarray]] = {}

    def add_arm(self, arm_id: str, parameters: dict[str, float]):
        """Add a new arm with its own linear model."""
        self.arms[arm_id] = {
            "parameters": parameters,
            "A": np.eye(self.n_features),  # Design matrix
            "b": np.zeros(self.n_features),  # Reward vector
            "theta": np.zeros(self.n_features),  # Model weights
        }

    def select_arm(self, context: np.ndarray) -> str:
        """Select arm using LinUCB.

        Args:
            context: Context vector (e.g., regime features)

        Returns:
            arm_id of selected arm
        """
        best_ucb = -np.inf
        best_arm = None

        for arm_id, arm_data in self.arms.items():
            A = arm_data["A"]
            theta = arm_data["theta"]

            # UCB = theta^T * x + alpha * sqrt(x^T * A^-1 * x)
            A_inv = np.linalg.inv(A)
            mean = context @ theta
            exploration = self.alpha * np.sqrt(context @ A_inv @ context)
            ucb = mean + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm_id

        return best_arm or list(self.arms.keys())[0]

    def update(self, arm_id: str, context: np.ndarray, reward: float):
        """Update arm model with observed reward.

        Args:
            arm_id: Arm that was pulled
            context: Context vector used
            reward: Observed reward
        """
        if arm_id not in self.arms:
            return

        arm = self.arms[arm_id]

        # Update design matrix: A = A + x * x^T
        arm["A"] += np.outer(context, context)

        # Update reward vector: b = b + r * x
        arm["b"] += reward * context

        # Update model weights: theta = A^-1 * b
        arm["theta"] = np.linalg.solve(arm["A"], arm["b"])

    def get_regime_context(self, regime: str) -> np.ndarray:
        """Convert regime name to context vector.

        Args:
            regime: Regime name

        Returns:
            Context vector
        """
        # One-hot encoding for regimes + additional features
        regime_encoding = {
            "MEME_SEASON": [1, 0, 0, 0, 0],
            "NORMAL": [0, 1, 0, 0, 0],
            "BEAR_VOLATILE": [0, 0, 1, 0, 0],
            "HIGH_MEV": [0, 0, 0, 1, 0],
            "LOW_LIQUIDITY": [0, 0, 0, 0, 1],
        }

        base = regime_encoding.get(regime, [0, 1, 0, 0, 0])

        # Pad to n_features if needed
        if len(base) < self.n_features:
            base = base + [0] * (self.n_features - len(base))

        return np.array(base[: self.n_features])


class GaussianProcessOptimizer:
    """Bayesian Optimization using Gaussian Process for proposal ranking.

    Uses GP to model the objective function and acquisition function
    (Expected Improvement) to balance exploration/exploitation.
    """

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]],
        length_scale: float = 1.0,
        noise: float = 0.1,
    ):
        """Initialize GP optimizer.

        Args:
            param_bounds: Dictionary of parameter name -> (min, max)
            length_scale: RBF kernel length scale
            noise: Observation noise level
        """
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_params = len(self.param_names)
        self.length_scale = length_scale
        self.noise = noise

        # Observations
        self.X: list[np.ndarray] = []
        self.y: list[float] = []

        self._K_inv: np.ndarray | None = None

    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """RBF (squared exponential) kernel."""
        diff = x1 - x2
        return np.exp(-0.5 * np.sum(diff**2) / self.length_scale**2)

    def _compute_kernel_matrix(self, X: list[np.ndarray]) -> np.ndarray:
        """Compute kernel matrix for observations."""
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._rbf_kernel(X[i], X[j])
        return K + self.noise**2 * np.eye(n)

    def _params_to_vector(self, params: dict[str, float]) -> np.ndarray:
        """Convert parameter dict to normalized vector."""
        vec = []
        for name in self.param_names:
            val = params.get(name, 0)
            low, high = self.param_bounds[name]
            normalized = (val - low) / (high - low + 1e-10)
            vec.append(normalized)
        return np.array(vec)

    def _vector_to_params(self, vec: np.ndarray) -> dict[str, float]:
        """Convert normalized vector to parameter dict."""
        params = {}
        for i, name in enumerate(self.param_names):
            low, high = self.param_bounds[name]
            params[name] = low + vec[i] * (high - low)
        return params

    def add_observation(self, params: dict[str, float], reward: float):
        """Add an observation."""
        x = self._params_to_vector(params)
        self.X.append(x)
        self.y.append(reward)
        self._K_inv = None  # Invalidate cache

    def predict(self, params: dict[str, float]) -> tuple[float, float]:
        """Predict mean and std at a point.

        Args:
            params: Parameter dict

        Returns:
            Tuple of (mean, std)
        """
        if len(self.X) == 0:
            return 0.0, 1.0

        x = self._params_to_vector(params)
        X_arr = np.array(self.X)
        y_arr = np.array(self.y)

        # Kernel vector
        k = np.array([self._rbf_kernel(x, xi) for xi in X_arr])

        # Kernel matrix
        if self._K_inv is None:
            K = self._compute_kernel_matrix(self.X)
            try:
                self._K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                return 0.0, 1.0

        # Posterior mean and variance
        mean = k @ self._K_inv @ y_arr
        var = self._rbf_kernel(x, x) - k @ self._K_inv @ k
        std = np.sqrt(max(0, var))

        return float(mean), float(std)

    def expected_improvement(
        self,
        params: dict[str, float],
        best_y: float,
        xi: float = 0.01,
    ) -> float:
        """Compute Expected Improvement acquisition function.

        Args:
            params: Parameter dict to evaluate
            best_y: Best observed value so far
            xi: Exploration parameter

        Returns:
            Expected improvement value
        """
        if not SCIPY_AVAILABLE:
            mean, std = self.predict(params)
            return max(0, mean - best_y)

        mean, std = self.predict(params)

        if std < 1e-10:
            return 0.0

        z = (mean - best_y - xi) / std
        ei = (mean - best_y - xi) * norm.cdf(z) + std * norm.pdf(z)

        return float(max(0, ei))

    def suggest_next(self, n_candidates: int = 100) -> dict[str, float]:
        """Suggest next point to evaluate.

        Args:
            n_candidates: Number of random candidates to consider

        Returns:
            Suggested parameters
        """
        best_y = max(self.y) if self.y else 0.0

        best_ei = -np.inf
        best_params = None

        for _ in range(n_candidates):
            # Random candidate
            params = {}
            for name, (low, high) in self.param_bounds.items():
                params[name] = np.random.uniform(low, high)

            ei = self.expected_improvement(params, best_y)

            if ei > best_ei:
                best_ei = ei
                best_params = params

        return best_params or self._random_params()

    def _random_params(self) -> dict[str, float]:
        """Generate random parameters."""
        return {
            name: np.random.uniform(low, high) for name, (low, high) in self.param_bounds.items()
        }


class ProposalRanker:
    """Full proposal ranking system combining multiple methods.

    Pipeline:
    1. Generate candidates using bandits, perturbation, and random search
    2. Fast filter using vectorized scoring
    3. Deep evaluation using backtest and Monte Carlo
    4. Optional LLM analysis for top candidates
    """

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]],
        n_regime_features: int = 5,
        thompson_prior_mean: float = 0.5,
        linucb_alpha: float = 0.5,
    ):
        """Initialize proposal ranker.

        Args:
            param_bounds: Parameter bounds for optimization
            n_regime_features: Number of regime context features
            thompson_prior_mean: Prior mean for Thompson Sampling
            linucb_alpha: Exploration parameter for LinUCB
        """
        self.param_bounds = param_bounds

        # Initialize ranking methods
        self.thompson = ThompsonSampler(
            use_gaussian=True,
            prior_mean=thompson_prior_mean,
        )
        self.linucb = LinUCBContextualBandit(
            n_features=n_regime_features,
            alpha=linucb_alpha,
        )
        self.gp_optimizer = GaussianProcessOptimizer(param_bounds)

        # Tracking
        self._evaluation_history: list[ProposalCandidate] = []
        self._best_params: dict[str, float] | None = None
        self._best_score: float = -np.inf

    def generate_candidates(
        self,
        n_candidates: int = 50,
        regime: str | None = None,
        base_params: dict[str, float] | None = None,
    ) -> list[ProposalCandidate]:
        """Generate candidate proposals.

        Args:
            n_candidates: Number of candidates to generate
            regime: Current market regime
            base_params: Base parameters to perturb from

        Returns:
            List of candidate proposals
        """
        candidates = []
        candidate_id = 0

        # 1. Thompson Sampling candidates (20%)
        n_thompson = int(n_candidates * 0.2)
        for _ in range(n_thompson):
            if self.thompson.arms:
                arm_id = self.thompson.select_arm()
                params = self.thompson.arms[arm_id].parameters.copy()
            else:
                params = self._random_params()

            candidates.append(
                ProposalCandidate(
                    proposal_id=f"ts_{candidate_id}",
                    parameters=params,
                    source="thompson",
                )
            )
            candidate_id += 1

        # 2. LinUCB candidates (20%)
        n_linucb = int(n_candidates * 0.2)
        if regime and self.linucb.arms:
            context = self.linucb.get_regime_context(regime)
            for _ in range(n_linucb):
                arm_id = self.linucb.select_arm(context)
                params = self.linucb.arms[arm_id]["parameters"].copy()
                candidates.append(
                    ProposalCandidate(
                        proposal_id=f"linucb_{candidate_id}",
                        parameters=params,
                        source="linucb",
                    )
                )
                candidate_id += 1

        # 3. Bayesian Optimization candidates (20%)
        n_bo = int(n_candidates * 0.2)
        for _ in range(n_bo):
            params = self.gp_optimizer.suggest_next()
            candidates.append(
                ProposalCandidate(
                    proposal_id=f"bo_{candidate_id}",
                    parameters=params,
                    source="bayesian",
                )
            )
            candidate_id += 1

        # 4. Perturbation candidates (20%)
        n_perturb = int(n_candidates * 0.2)
        if base_params:
            for _ in range(n_perturb):
                params = self._perturb_params(base_params)
                candidates.append(
                    ProposalCandidate(
                        proposal_id=f"perturb_{candidate_id}",
                        parameters=params,
                        source="perturbation",
                        parent_id="base",
                    )
                )
                candidate_id += 1

        # 5. Random candidates (remaining)
        n_random = n_candidates - len(candidates)
        for _ in range(n_random):
            params = self._random_params()
            candidates.append(
                ProposalCandidate(
                    proposal_id=f"random_{candidate_id}",
                    parameters=params,
                    source="random",
                )
            )
            candidate_id += 1

        return candidates

    def rank_candidates(
        self,
        candidates: list[ProposalCandidate],
        fast_scorer: Callable[[dict[str, float]], float],
        deep_scorer: Callable[[dict[str, float]], tuple[float, float]] | None = None,
        n_top: int = 10,
        regime: str | None = None,
    ) -> RankingResult:
        """Rank candidates using multi-stage evaluation.

        Args:
            candidates: List of candidates to rank
            fast_scorer: Fast scoring function (params -> score)
            deep_scorer: Deep scoring function (params -> (score, std))
            n_top: Number of top candidates to return
            regime: Current market regime

        Returns:
            RankingResult with ranked proposals
        """
        import time

        start_time = time.time()

        # Stage 1: Fast scoring
        for candidate in candidates:
            candidate.fast_score = fast_scorer(candidate.parameters)

        # Filter to top candidates
        sorted_by_fast = sorted(candidates, key=lambda c: c.fast_score or 0, reverse=True)
        top_candidates = sorted_by_fast[: n_top * 2]  # Keep 2x for deep eval

        # Stage 2: Deep scoring (if available)
        if deep_scorer:
            for candidate in top_candidates:
                score, std = deep_scorer(candidate.parameters)
                candidate.backtest_sharpe = score
                candidate.score_std = std
                candidate.confidence_interval = (score - 2 * std, score + 2 * std)

            # Sort by deep score
            top_candidates = sorted(
                top_candidates,
                key=lambda c: c.backtest_sharpe or 0,
                reverse=True,
            )

        # Stage 3: Final scoring
        for i, candidate in enumerate(top_candidates[:n_top]):
            if candidate.backtest_sharpe is not None:
                candidate.final_score = candidate.backtest_sharpe
            else:
                candidate.final_score = candidate.fast_score
            candidate.rank = i + 1

        # Update bandit models with results
        self._update_models(top_candidates[:n_top], regime)

        # Track best
        if top_candidates and top_candidates[0].final_score:
            if top_candidates[0].final_score > self._best_score:
                self._best_score = top_candidates[0].final_score
                self._best_params = top_candidates[0].parameters.copy()

        # Calculate exploration ratio
        exploration_sources = {"thompson", "linucb", "random", "bayesian"}
        exploration_count = sum(
            1 for c in top_candidates[:n_top] if c.source in exploration_sources
        )
        exploration_ratio = exploration_count / n_top if n_top > 0 else 0

        elapsed_ms = (time.time() - start_time) * 1000

        return RankingResult(
            ranked_proposals=top_candidates[:n_top],
            method=RankingMethod.HYBRID,
            regime=regime,
            n_evaluated=len(candidates),
            n_filtered=len(candidates) - n_top,
            best_proposal=top_candidates[0] if top_candidates else candidates[0],
            exploration_ratio=exploration_ratio,
            total_evaluation_time_ms=elapsed_ms,
        )

    def _update_models(
        self,
        evaluated: list[ProposalCandidate],
        regime: str | None = None,
    ):
        """Update bandit models with evaluation results."""
        for candidate in evaluated:
            reward = candidate.final_score or 0.0

            # Update Thompson Sampling
            arm_key = self._params_to_key(candidate.parameters)
            if arm_key not in self.thompson.arms:
                self.thompson.add_arm(arm_key, candidate.parameters)
            self.thompson.update(arm_key, reward)

            # Update LinUCB
            if regime:
                context = self.linucb.get_regime_context(regime)
                if arm_key not in self.linucb.arms:
                    self.linucb.add_arm(arm_key, candidate.parameters)
                self.linucb.update(arm_key, context, reward)

            # Update GP
            self.gp_optimizer.add_observation(candidate.parameters, reward)

    def _params_to_key(self, params: dict[str, float]) -> str:
        """Convert params dict to string key."""
        items = sorted(params.items())
        return ":".join(f"{k}={v:.3f}" for k, v in items)

    def _random_params(self) -> dict[str, float]:
        """Generate random parameters within bounds."""
        return {
            name: np.random.uniform(low, high) for name, (low, high) in self.param_bounds.items()
        }

    def _perturb_params(
        self,
        base_params: dict[str, float],
        perturbation_scale: float = 0.1,
    ) -> dict[str, float]:
        """Perturb parameters around a base."""
        perturbed = {}
        for name, value in base_params.items():
            if name in self.param_bounds:
                low, high = self.param_bounds[name]
                range_size = high - low
                noise = np.random.normal(0, perturbation_scale * range_size)
                perturbed[name] = np.clip(value + noise, low, high)
            else:
                perturbed[name] = value
        return perturbed

    def get_best_params(self) -> dict[str, float] | None:
        """Get best parameters found so far."""
        return self._best_params

    def get_statistics(self) -> dict[str, Any]:
        """Get ranker statistics."""
        return {
            "best_score": self._best_score,
            "thompson_stats": self.thompson.get_statistics(),
            "n_gp_observations": len(self.gp_optimizer.X),
            "n_linucb_arms": len(self.linucb.arms),
        }
