"""
Isolation Forest Rug Detector.

Anomaly-based rug detection using Isolation Forest algorithm.
Detects tokens with anomalous feature patterns that indicate potential rugs.

Key advantages over supervised learning:
- Works with limited labeled data
- Detects novel rug patterns not seen in training
- Fast inference suitable for real-time detection

Thresholds:
- >0.85: IMMEDIATE EXIT
- 0.70-0.85: REDUCE position 50%
- 0.50-0.70: MONITOR
- <0.50: GREEN LIGHT
"""

import logging
import pickle
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Check for RAPIDS cuML GPU acceleration
CUMl_GPU_AVAILABLE = False
try:
    from cuml.ensemble import IsolationForest as cuIsolationForest
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        CUMl_GPU_AVAILABLE = True
        logger.info("cuML Isolation Forest GPU available (CUDA detected)")
except (ImportError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
    pass


class RiskLevel(Enum):
    """Risk level classification based on anomaly score."""
    GREEN_LIGHT = "green_light"  # <0.50 - Safe to proceed
    MONITOR = "monitor"  # 0.50-0.70 - Watch closely
    REDUCE = "reduce"  # 0.70-0.85 - Reduce position 50%
    IMMEDIATE_EXIT = "immediate_exit"  # >0.85 - Exit immediately


@dataclass
class RugRiskResult:
    """Result of rug risk assessment."""
    risk_score: float  # 0-1, higher = more anomalous/risky
    risk_level: RiskLevel
    anomaly_label: int  # -1 = outlier, 1 = inlier
    top_risk_features: List[Tuple[str, float]]  # Feature contributions to risk
    confidence: float  # Model confidence in prediction

    def should_exit(self) -> bool:
        """Check if immediate exit is recommended."""
        return self.risk_level == RiskLevel.IMMEDIATE_EXIT

    def should_reduce(self) -> bool:
        """Check if position reduction is recommended."""
        return self.risk_level in [RiskLevel.REDUCE, RiskLevel.IMMEDIATE_EXIT]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "anomaly_label": self.anomaly_label,
            "top_risk_features": self.top_risk_features,
            "confidence": self.confidence,
            "should_exit": self.should_exit(),
            "should_reduce": self.should_reduce(),
        }


# Feature names for the 33 rug-detection features (liquidity + holder + risk)
RUG_FEATURE_NAMES = [
    # Liquidity & Pool (12)
    "pool_tvl_sol",
    "pool_age_seconds",
    "lp_lock_percentage",
    "lp_concentration",
    "lp_removal_velocity",
    "lp_addition_velocity",
    "pool_depth_imbalance",
    "slippage_1pct",
    "slippage_5pct",
    "unique_lp_provider_count",
    "deployer_lp_ownership_pct",
    "emergency_liquidity_flag",
    # Token Supply & Holder (11)
    "total_supply",
    "deployer_holdings_pct",
    "top_10_holder_concentration",
    "holder_count_unique",
    "holder_growth_velocity",
    "transfer_concentration",
    "sniper_bot_count_t0",
    "bot_to_human_ratio",
    "large_holder_churn",
    "mint_authority_revoked",
    "token_freezeable",
    # On-Chain Risk (10)
    "contract_is_mintable",
    "contract_transfer_fee",
    "hidden_fee_detected",
    "circular_trading_score",
    "benford_law_pvalue",
    "address_clustering_risk",
    "proxy_contract_flag",
    "unverified_code_flag",
    "external_transfer_flag",
    "rug_pull_ml_score",
]


class IsolationForestRugDetector:
    """Anomaly-based rug detection using Isolation Forest.

    Uses 33 features from liquidity, holder, and risk categories
    to identify anomalous token patterns indicative of rugs.

    The model is trained on "normal" (non-rug) tokens and learns
    to identify deviations from normal patterns.
    """

    def __init__(
        self,
        contamination: float = 0.03,
        n_estimators: int = 150,
        max_samples: int = 512,
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
    ):
        """Initialize the rug detector.

        Args:
            contamination: Expected fraction of rugs in training data (default 3%)
            n_estimators: Number of isolation trees
            max_samples: Samples per tree (controls model complexity)
            random_state: Random seed for reproducibility
            feature_names: Names of features (default: RUG_FEATURE_NAMES)
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.feature_names = feature_names or RUG_FEATURE_NAMES

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
            warm_start=False,
        )

        self.scaler = StandardScaler()
        self.is_fitted = False

        # Risk thresholds
        self.threshold_exit = 0.85
        self.threshold_reduce = 0.70
        self.threshold_monitor = 0.50

        # Feature importance tracking
        self._feature_depths: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IsolationForestRugDetector":
        """Fit the model on training data.

        For best results, train primarily on known good (non-rug) tokens.
        The model learns normal patterns and flags deviations.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Optional labels (1=rug, 0=safe). If provided, filters to safe tokens.

        Returns:
            Self for chaining
        """
        logger.info(f"Fitting Isolation Forest on {X.shape[0]} samples, {X.shape[1]} features")

        # If labels provided, train primarily on safe tokens
        if y is not None:
            safe_mask = y == 0
            if safe_mask.sum() > 100:
                # Use mostly safe tokens with small rug sample
                safe_samples = X[safe_mask]
                rug_samples = X[~safe_mask]

                # Include some rugs so contamination makes sense
                n_rugs = min(len(rug_samples), int(len(safe_samples) * self.contamination))
                if n_rugs > 0:
                    rug_indices = np.random.choice(len(rug_samples), n_rugs, replace=False)
                    X_train = np.vstack([safe_samples, rug_samples[rug_indices]])
                else:
                    X_train = safe_samples
            else:
                X_train = X
        else:
            X_train = X

        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Fit isolation forest
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Calculate feature importance approximation
        self._compute_feature_importance(X_scaled)

        logger.info(
            f"Isolation Forest fitted: {self.model.n_features_in_} features, "
            f"contamination={self.contamination}"
        )

        return self

    def predict_risk(self, features: np.ndarray) -> RugRiskResult:
        """Predict rug risk for a single sample.

        Args:
            features: Feature vector (33 features for rug detection)

        Returns:
            RugRiskResult with risk score and recommendations
        """
        if not self.is_fitted:
            logger.warning("Model not fitted, returning default risk")
            return RugRiskResult(
                risk_score=0.5,
                risk_level=RiskLevel.MONITOR,
                anomaly_label=1,
                top_risk_features=[],
                confidence=0.0,
            )

        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Scale and predict
        X_scaled = self.scaler.transform(features)

        # Get anomaly score (negative = more anomalous)
        raw_score = self.model.decision_function(X_scaled)[0]
        anomaly_label = self.model.predict(X_scaled)[0]

        # Convert to 0-1 risk score (higher = more risky/anomalous)
        # decision_function returns ~0 at threshold, negative for outliers
        # Typical range is [-0.5, 0.5] but can vary
        risk_score = self._normalize_score(raw_score)

        # Determine risk level
        risk_level = self._get_risk_level(risk_score)

        # Get top contributing features
        top_features = self._get_top_risk_features(X_scaled[0])

        # Calculate confidence based on score distance from threshold
        confidence = min(1.0, abs(raw_score) * 2)

        return RugRiskResult(
            risk_score=risk_score,
            risk_level=risk_level,
            anomaly_label=anomaly_label,
            top_risk_features=top_features,
            confidence=confidence,
        )

    def predict_risk_batch(self, X: np.ndarray) -> List[RugRiskResult]:
        """Predict rug risk for multiple samples.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            List of RugRiskResult for each sample
        """
        if not self.is_fitted:
            return [
                RugRiskResult(
                    risk_score=0.5,
                    risk_level=RiskLevel.MONITOR,
                    anomaly_label=1,
                    top_risk_features=[],
                    confidence=0.0,
                )
                for _ in range(len(X))
            ]

        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        labels = self.model.predict(X_scaled)

        results = []
        for i, (raw_score, label) in enumerate(zip(raw_scores, labels)):
            risk_score = self._normalize_score(raw_score)
            risk_level = self._get_risk_level(risk_score)
            top_features = self._get_top_risk_features(X_scaled[i])
            confidence = min(1.0, abs(raw_score) * 2)

            results.append(RugRiskResult(
                risk_score=risk_score,
                risk_level=risk_level,
                anomaly_label=label,
                top_risk_features=top_features,
                confidence=confidence,
            ))

        return results

    def _normalize_score(self, raw_score: float) -> float:
        """Normalize decision function score to 0-1 range.

        The decision function typically returns values centered around 0,
        with negative values indicating outliers.
        """
        # Sigmoid transformation centered on 0
        # Adjust scale so typical scores map well to 0-1 range
        normalized = 1.0 / (1.0 + np.exp(raw_score * 5))
        return float(np.clip(normalized, 0.0, 1.0))

    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score."""
        if risk_score >= self.threshold_exit:
            return RiskLevel.IMMEDIATE_EXIT
        elif risk_score >= self.threshold_reduce:
            return RiskLevel.REDUCE
        elif risk_score >= self.threshold_monitor:
            return RiskLevel.MONITOR
        else:
            return RiskLevel.GREEN_LIGHT

    def _compute_feature_importance(self, X_scaled: np.ndarray) -> None:
        """Compute approximate feature importance from tree depths.

        Features that frequently appear in early splits (shallow depths)
        are more important for anomaly detection.
        """
        # Use average path length contribution as importance proxy
        n_features = X_scaled.shape[1]
        self._feature_depths = np.zeros(n_features)

        # Sample some points and track which features contribute most
        sample_size = min(100, len(X_scaled))
        sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)

        for idx in sample_indices:
            sample = X_scaled[idx].reshape(1, -1)
            # Get path lengths through all trees
            depths = self.model.decision_function(sample)
            # Approximate feature contribution by feature variance
            for j in range(n_features):
                self._feature_depths[j] += abs(sample[0, j])

        self._feature_depths /= sample_size

    def _get_top_risk_features(self, x_scaled: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top features contributing to anomaly score.

        Uses absolute deviation from mean as proxy for contribution.
        """
        if self._feature_depths is None:
            return []

        # Combine deviation with importance
        contributions = np.abs(x_scaled) * self._feature_depths

        # Get top-k features
        top_indices = np.argsort(contributions)[-top_k:][::-1]

        result = []
        for idx in top_indices:
            if idx < len(self.feature_names):
                result.append((self.feature_names[idx], float(contributions[idx])))

        return result

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self._feature_depths is None:
            return {}

        # Normalize to sum to 1
        total = self._feature_depths.sum()
        if total > 0:
            normalized = self._feature_depths / total
        else:
            normalized = self._feature_depths

        return dict(zip(self.feature_names, normalized))

    def save(self, path: str) -> None:
        """Save model to file."""
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "is_fitted": self.is_fitted,
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "max_samples": self.max_samples,
                "random_state": self.random_state,
                "feature_names": self.feature_names,
                "feature_depths": self._feature_depths,
                "threshold_exit": self.threshold_exit,
                "threshold_reduce": self.threshold_reduce,
                "threshold_monitor": self.threshold_monitor,
            }, f)
        logger.info(f"Saved Isolation Forest model to {path}")

    def load(self, path: str) -> "IsolationForestRugDetector":
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.is_fitted = data["is_fitted"]
            self.contamination = data.get("contamination", 0.03)
            self.n_estimators = data.get("n_estimators", 150)
            self.max_samples = data.get("max_samples", 512)
            self.random_state = data.get("random_state", 42)
            self.feature_names = data.get("feature_names", RUG_FEATURE_NAMES)
            self._feature_depths = data.get("feature_depths")
            self.threshold_exit = data.get("threshold_exit", 0.85)
            self.threshold_reduce = data.get("threshold_reduce", 0.70)
            self.threshold_monitor = data.get("threshold_monitor", 0.50)
        logger.info(f"Loaded Isolation Forest model from {path}")
        return self

    def set_thresholds(
        self,
        exit_threshold: float = 0.85,
        reduce_threshold: float = 0.70,
        monitor_threshold: float = 0.50,
    ) -> None:
        """Set custom risk thresholds.

        Args:
            exit_threshold: Score above which to exit immediately
            reduce_threshold: Score above which to reduce position
            monitor_threshold: Score above which to monitor closely
        """
        self.threshold_exit = exit_threshold
        self.threshold_reduce = reduce_threshold
        self.threshold_monitor = monitor_threshold
        logger.info(
            f"Set thresholds: exit={exit_threshold}, reduce={reduce_threshold}, "
            f"monitor={monitor_threshold}"
        )


def create_detector_from_features(
    features_50: np.ndarray,
    labels: Optional[np.ndarray] = None,
    **kwargs,
) -> IsolationForestRugDetector:
    """Create and fit a detector from 50-feature vectors.

    Extracts the 33 rug-detection features (liquidity + holder + risk)
    and trains the Isolation Forest.

    Args:
        features_50: Feature matrix with all 50 features
        labels: Optional rug labels (1=rug, 0=safe)
        **kwargs: Additional arguments for IsolationForestRugDetector

    Returns:
        Fitted detector
    """
    # Extract features 0-22 (liquidity + holder) and 33-42 (risk)
    rug_features = np.hstack([
        features_50[:, :23],  # Liquidity (12) + Holder (11)
        features_50[:, 33:43],  # Risk (10)
    ])

    detector = IsolationForestRugDetector(**kwargs)
    detector.fit(rug_features, labels)

    return detector
