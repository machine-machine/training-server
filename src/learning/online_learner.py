"""
Online Learner - Incremental model updates from paper trade outcomes.

Provides real-time learning from paper trading outcomes without full retraining:
- SGD-based partial_fit for logistic regression component
- Adaptive weighting: Paper outcomes start at 0.5x, increase when confirmed
- Time decay: 24-hour half-life for outcome relevance
- Integrates with existing ProfitabilityLearner for coefficient updates
"""

import asyncio
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OutcomeSource(Enum):
    """Source of trade outcome."""
    LIVE = "live"
    PAPER = "paper"
    PAPER_CONFIRMED = "paper_confirmed"  # Paper outcome later confirmed by live
    BACKTEST = "backtest"


@dataclass
class TradeOutcome:
    """Single trade outcome for online learning."""
    trade_id: str
    timestamp_ms: int
    source: OutcomeSource

    # Features at trade time
    features: np.ndarray

    # Outcome
    is_profitable: bool
    pnl_pct: float

    # ML context at trade time (for correlation analysis)
    fraud_score: float = 0.0
    regime: str = "chop"
    regime_confidence: float = 0.5

    # Weighting
    base_weight: float = 1.0
    confirmed: bool = False
    confirmed_at_ms: Optional[int] = None

    def age_hours(self) -> float:
        """Calculate age of this outcome in hours."""
        now_ms = int(datetime.now().timestamp() * 1000)
        return (now_ms - self.timestamp_ms) / (1000 * 3600)

    def effective_weight(self, config: "OnlineLearningConfig") -> float:
        """Calculate effective weight with time decay and source adjustment."""
        age = self.age_hours()

        # Time decay with configurable half-life
        decay_factor = math.exp(-math.log(2) * age / config.time_decay_half_life_hours)

        # Source-based weighting
        if self.source == OutcomeSource.LIVE:
            source_weight = 1.0
        elif self.source == OutcomeSource.PAPER_CONFIRMED:
            source_weight = config.paper_confirmed_weight
        elif self.source == OutcomeSource.PAPER:
            source_weight = config.paper_base_weight
        else:  # BACKTEST
            source_weight = config.backtest_weight

        return self.base_weight * decay_factor * source_weight


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning."""
    # Learning rate for SGD
    learning_rate: float = 0.01

    # Regularization
    l2_penalty: float = 0.0001

    # Time decay
    time_decay_half_life_hours: float = 24.0

    # Source weights
    paper_base_weight: float = 0.5
    paper_confirmed_weight: float = 0.8
    backtest_weight: float = 0.3

    # Batch settings
    min_batch_size: int = 10
    max_batch_size: int = 100
    update_interval_seconds: int = 60

    # Stability
    max_coefficient_change: float = 0.1  # Max change per update
    momentum: float = 0.9  # SGD momentum

    # Feature scaling
    use_feature_scaling: bool = True
    feature_scale_ema_alpha: float = 0.01  # EMA for running mean/std


@dataclass
class OnlineLearningState:
    """State of the online learner."""
    n_updates: int = 0
    last_update_ms: int = 0
    cumulative_samples: int = 0

    # Coefficient tracking
    current_coefficients: Optional[np.ndarray] = None
    coefficient_momentum: Optional[np.ndarray] = None

    # Feature scaling
    feature_means: Optional[np.ndarray] = None
    feature_stds: Optional[np.ndarray] = None

    # Performance tracking
    recent_accuracy: float = 0.5
    recent_loss: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Export state to dictionary."""
        return {
            "n_updates": self.n_updates,
            "last_update_ms": self.last_update_ms,
            "cumulative_samples": self.cumulative_samples,
            "current_coefficients": self.current_coefficients.tolist() if self.current_coefficients is not None else None,
            "recent_accuracy": self.recent_accuracy,
            "recent_loss": self.recent_loss,
        }


class OnlineLearner:
    """Incremental learner for real-time model updates.

    Updates the linear component of the profitability model using
    stochastic gradient descent on streaming paper trade outcomes.

    Key features:
    - SGD with momentum for stable updates
    - Adaptive weighting based on outcome source
    - Time decay for outcome relevance
    - Integration with existing ProfitabilityLearner
    """

    def __init__(
        self,
        n_features: int,
        config: Optional[OnlineLearningConfig] = None,
        initial_coefficients: Optional[np.ndarray] = None,
    ):
        self.n_features = n_features
        self.config = config or OnlineLearningConfig()

        # Initialize state
        self.state = OnlineLearningState()

        if initial_coefficients is not None:
            self.state.current_coefficients = initial_coefficients.copy()
        else:
            # Xavier initialization
            scale = np.sqrt(2.0 / n_features)
            self.state.current_coefficients = np.random.randn(n_features) * scale

        # Initialize momentum
        self.state.coefficient_momentum = np.zeros(n_features)

        # Feature scaling
        self.state.feature_means = np.zeros(n_features)
        self.state.feature_stds = np.ones(n_features)

        # Outcome buffer
        self._outcome_buffer: Deque[TradeOutcome] = deque(maxlen=1000)
        self._pending_confirmations: Dict[str, TradeOutcome] = {}

        # Update tracking
        self._last_update_time = datetime.now()
        self._update_lock = asyncio.Lock()

    def add_outcome(self, outcome: TradeOutcome) -> None:
        """Add a new trade outcome to the buffer."""
        self._outcome_buffer.append(outcome)

        # If paper trade, track for potential confirmation
        if outcome.source == OutcomeSource.PAPER:
            self._pending_confirmations[outcome.trade_id] = outcome

    def confirm_paper_outcome(
        self,
        trade_id: str,
        live_is_profitable: bool,
        live_pnl_pct: float,
    ) -> bool:
        """Confirm a paper trade outcome with live results.

        If the paper prediction matches live outcome, upgrade weight.

        Returns:
            True if confirmation found and processed
        """
        if trade_id not in self._pending_confirmations:
            return False

        paper_outcome = self._pending_confirmations.pop(trade_id)

        # Check if prediction was correct
        if paper_outcome.is_profitable == live_is_profitable:
            # Upgrade to confirmed
            paper_outcome.source = OutcomeSource.PAPER_CONFIRMED
            paper_outcome.confirmed = True
            paper_outcome.confirmed_at_ms = int(datetime.now().timestamp() * 1000)

            logger.debug(
                f"Paper outcome {trade_id} confirmed: "
                f"paper={paper_outcome.pnl_pct:.2f}%, live={live_pnl_pct:.2f}%"
            )
        else:
            logger.debug(
                f"Paper outcome {trade_id} did not match live: "
                f"paper_profitable={paper_outcome.is_profitable}, "
                f"live_profitable={live_is_profitable}"
            )

        return True

    async def update_if_ready(self) -> Optional[Dict[str, Any]]:
        """Check if update is needed and perform if so.

        Returns:
            Update result dict if update performed, None otherwise
        """
        now = datetime.now()
        elapsed = (now - self._last_update_time).total_seconds()

        if elapsed < self.config.update_interval_seconds:
            return None

        if len(self._outcome_buffer) < self.config.min_batch_size:
            return None

        return await self.update()

    async def update(self) -> Dict[str, Any]:
        """Perform an online update on buffered outcomes.

        Returns:
            Dict with update statistics
        """
        async with self._update_lock:
            return self._perform_update()

    def _perform_update(self) -> Dict[str, Any]:
        """Internal update logic."""
        if len(self._outcome_buffer) == 0:
            return {"status": "no_data", "samples": 0}

        # Get batch of outcomes
        batch_size = min(len(self._outcome_buffer), self.config.max_batch_size)
        batch = list(self._outcome_buffer)[-batch_size:]

        # Extract features and labels with weights
        X = np.array([o.features for o in batch])
        y = np.array([1.0 if o.is_profitable else 0.0 for o in batch])
        weights = np.array([o.effective_weight(self.config) for o in batch])

        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum() * len(weights)
        else:
            weights = np.ones(len(weights))

        # Update feature scaling (EMA)
        if self.config.use_feature_scaling:
            self._update_feature_scaling(X)
            X_scaled = (X - self.state.feature_means) / (self.state.feature_stds + 1e-8)
        else:
            X_scaled = X

        # Compute gradients using logistic loss
        predictions = self._sigmoid(X_scaled @ self.state.current_coefficients)
        errors = predictions - y

        # Weighted gradient
        gradient = (X_scaled.T @ (errors * weights)) / len(batch)

        # L2 regularization
        gradient += self.config.l2_penalty * self.state.current_coefficients

        # Apply momentum
        self.state.coefficient_momentum = (
            self.config.momentum * self.state.coefficient_momentum
            - self.config.learning_rate * gradient
        )

        # Clip update magnitude for stability
        update = np.clip(
            self.state.coefficient_momentum,
            -self.config.max_coefficient_change,
            self.config.max_coefficient_change,
        )

        # Apply update
        self.state.current_coefficients += update

        # Compute metrics
        loss = self._log_loss(y, predictions, weights)
        accuracy = np.mean((predictions > 0.5) == y)

        # Update state
        self.state.n_updates += 1
        self.state.last_update_ms = int(datetime.now().timestamp() * 1000)
        self.state.cumulative_samples += batch_size
        self.state.recent_accuracy = accuracy
        self.state.recent_loss = loss

        self._last_update_time = datetime.now()

        # Clear processed outcomes
        for _ in range(batch_size):
            if self._outcome_buffer:
                self._outcome_buffer.popleft()

        logger.info(
            f"Online update #{self.state.n_updates}: "
            f"samples={batch_size}, loss={loss:.4f}, accuracy={accuracy:.3f}"
        )

        return {
            "status": "updated",
            "samples": batch_size,
            "loss": loss,
            "accuracy": accuracy,
            "n_updates": self.state.n_updates,
            "gradient_norm": np.linalg.norm(gradient),
            "coefficient_change": np.linalg.norm(update),
        }

    def _update_feature_scaling(self, X: np.ndarray) -> None:
        """Update running mean and std with EMA."""
        batch_mean = X.mean(axis=0)
        batch_std = X.std(axis=0) + 1e-8

        alpha = self.config.feature_scale_ema_alpha
        self.state.feature_means = (1 - alpha) * self.state.feature_means + alpha * batch_mean
        self.state.feature_stds = (1 - alpha) * self.state.feature_stds + alpha * batch_std

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    @staticmethod
    def _log_loss(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
        """Weighted log loss."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(np.mean(loss))

    def get_coefficients(self) -> np.ndarray:
        """Get current coefficients."""
        return self.state.current_coefficients.copy()

    def set_coefficients(self, coefficients: np.ndarray) -> None:
        """Set coefficients (e.g., from full retrain)."""
        if len(coefficients) != self.n_features:
            raise ValueError(f"Expected {self.n_features} coefficients, got {len(coefficients)}")
        self.state.current_coefficients = coefficients.copy()
        # Reset momentum on coefficient change
        self.state.coefficient_momentum = np.zeros(self.n_features)

    def predict(self, features: np.ndarray) -> float:
        """Predict probability using current coefficients."""
        if self.config.use_feature_scaling:
            features_scaled = (features - self.state.feature_means) / (self.state.feature_stds + 1e-8)
        else:
            features_scaled = features

        logit = np.dot(features_scaled, self.state.current_coefficients)
        return float(self._sigmoid(logit))

    def get_stats(self) -> Dict[str, Any]:
        """Get current learner statistics."""
        return {
            "n_updates": self.state.n_updates,
            "cumulative_samples": self.state.cumulative_samples,
            "buffer_size": len(self._outcome_buffer),
            "pending_confirmations": len(self._pending_confirmations),
            "recent_accuracy": self.state.recent_accuracy,
            "recent_loss": self.state.recent_loss,
            "coefficient_norm": float(np.linalg.norm(self.state.current_coefficients)),
        }

    def cleanup_old_confirmations(self, max_age_hours: float = 24.0) -> int:
        """Remove old pending confirmations that won't be matched."""
        now_ms = int(datetime.now().timestamp() * 1000)
        cutoff_ms = now_ms - int(max_age_hours * 3600 * 1000)

        to_remove = [
            tid for tid, outcome in self._pending_confirmations.items()
            if outcome.timestamp_ms < cutoff_ms
        ]

        for tid in to_remove:
            del self._pending_confirmations[tid]

        return len(to_remove)
