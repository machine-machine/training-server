"""
Mutual Information Feature Selection for ML trading models.

Implements MI-based feature selection for the 50-feature vector:
1. Pairwise MI: I(X_i; Y) for profitability target
2. Redundancy filter: Remove correlated features (>0.85)
3. Conditional MI: Greedy forward selection (mRMR)
4. Regime-aware: Compute MI per market regime

Output: Optimal 15-25 features per regime
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import KBinsDiscretizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import entropy, spearmanr
    from scipy.special import digamma
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class SelectionMethod(Enum):
    """Feature selection methods."""
    UNIVARIATE_MI = "univariate_mi"       # Simple pairwise MI
    MRMR = "mrmr"                          # Minimum Redundancy Maximum Relevance
    JMI = "jmi"                            # Joint Mutual Information
    CMIM = "cmim"                          # Conditional Mutual Info Maximization


@dataclass
class FeatureScore:
    """Score for a single feature."""
    feature_name: str
    feature_index: int
    mi_score: float              # Mutual information with target
    redundancy_score: float      # Average MI with other selected features
    mrmr_score: float           # MI - redundancy
    vif: Optional[float] = None  # Variance Inflation Factor
    selected: bool = False
    regime: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature_name,
            "index": self.feature_index,
            "mi_score": self.mi_score,
            "redundancy": self.redundancy_score,
            "mrmr_score": self.mrmr_score,
            "vif": self.vif,
            "selected": self.selected,
            "regime": self.regime,
        }


@dataclass
class SelectionResult:
    """Result of feature selection."""
    selected_features: List[str]
    selected_indices: List[int]
    feature_scores: List[FeatureScore]
    method: SelectionMethod
    regime: Optional[str]
    n_original: int
    n_selected: int
    total_mi: float
    redundancy_filtered: List[str]
    low_mi_filtered: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_features": self.selected_features,
            "selected_indices": self.selected_indices,
            "method": self.method.value,
            "regime": self.regime,
            "n_original": self.n_original,
            "n_selected": self.n_selected,
            "total_mi": self.total_mi,
            "redundancy_filtered": self.redundancy_filtered,
            "low_mi_filtered": self.low_mi_filtered,
            "scores": [s.to_dict() for s in self.feature_scores],
        }


@dataclass
class MIConfig:
    """Configuration for MI feature selection."""
    # Selection parameters
    min_features: int = 15
    max_features: int = 25
    mi_threshold: float = 0.02          # Minimum MI to consider
    redundancy_threshold: float = 0.85  # Max correlation before filtering
    vif_threshold: float = 5.0          # Max VIF before filtering

    # MI estimation
    n_neighbors: int = 3                # For k-NN MI estimation
    n_bins: int = 10                    # For discretization
    random_state: int = 42

    # Regime-specific
    per_regime_selection: bool = True
    regime_weight_smoothing: float = 0.3  # Smooth regime-specific weights


class MutualInformationCalculator:
    """Calculate mutual information between features and target."""

    def __init__(self, config: Optional[MIConfig] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for MI calculation. pip install scikit-learn")
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for MI calculation. pip install scipy")

        self.config = config or MIConfig()

    def compute_mi_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        is_classification: bool = True,
    ) -> List[FeatureScore]:
        """Compute MI between each feature and target.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target array
            feature_names: Names of features
            is_classification: If True, use classification MI

        Returns:
            List of FeatureScore for each feature
        """
        n_features = X.shape[1]

        # Compute MI using sklearn
        if is_classification:
            mi_scores = mutual_info_classif(
                X, y,
                n_neighbors=self.config.n_neighbors,
                random_state=self.config.random_state,
            )
        else:
            mi_scores = mutual_info_regression(
                X, y,
                n_neighbors=self.config.n_neighbors,
                random_state=self.config.random_state,
            )

        # Create FeatureScore objects
        scores = []
        for i in range(n_features):
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            scores.append(FeatureScore(
                feature_name=name,
                feature_index=i,
                mi_score=float(mi_scores[i]),
                redundancy_score=0.0,  # Computed later
                mrmr_score=float(mi_scores[i]),  # Initial: just MI
            ))

        return scores

    def compute_feature_correlation(self, X: np.ndarray) -> np.ndarray:
        """Compute feature-feature correlation matrix.

        Args:
            X: Feature matrix

        Returns:
            Correlation matrix
        """
        n_features = X.shape[1]
        corr_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(i, n_features):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr, _ = spearmanr(X[:, i], X[:, j])
                    corr_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0.0
                    corr_matrix[j, i] = corr_matrix[i, j]

        return corr_matrix

    def compute_feature_mi(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise MI between features.

        Args:
            X: Feature matrix

        Returns:
            MI matrix (n_features, n_features)
        """
        n_features = X.shape[1]
        mi_matrix = np.zeros((n_features, n_features))

        # Check for constant features (zero variance) which crash quantile binning
        variances = np.var(X, axis=0)
        constant_mask = variances < 1e-10
        if constant_mask.any():
            n_constant = constant_mask.sum()
            logger.warning(f"Found {n_constant} constant features - using uniform binning")

        # Use 'uniform' strategy which handles constant features better than 'quantile'
        # Also limit n_bins to not exceed unique values
        n_unique_min = min(len(np.unique(X[:, i])) for i in range(n_features))
        effective_bins = min(self.config.n_bins, max(2, n_unique_min))

        discretizer = KBinsDiscretizer(
            n_bins=effective_bins,
            encode='ordinal',
            strategy='uniform',  # More robust than quantile for edge cases
        )

        try:
            X_discrete = discretizer.fit_transform(X)
        except ValueError as e:
            logger.warning(f"Discretization failed: {e}, using raw values")
            # Fallback: use raw values with rounding
            X_discrete = np.round(X * 10).astype(int)

        for i in range(n_features):
            for j in range(i + 1, n_features):
                mi = self._compute_mi_discrete(X_discrete[:, i], X_discrete[:, j])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        return mi_matrix

    def _compute_mi_discrete(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI between two discrete variables."""
        # Remove NaN values from both arrays
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        n = len(x)
        if n == 0:
            return 0.0

        # Joint probability
        xy_counts = {}
        x_counts = {}
        y_counts = {}

        for xi, yi in zip(x, y):
            # Skip any remaining NaN/inf values
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            xy_counts[(xi, yi)] = xy_counts.get((xi, yi), 0) + 1
            x_counts[xi] = x_counts.get(xi, 0) + 1
            y_counts[yi] = y_counts.get(yi, 0) + 1

        mi = 0.0
        for (xi, yi), count in xy_counts.items():
            p_xy = count / n
            p_x = x_counts.get(xi, 0) / n
            p_y = y_counts.get(yi, 0) / n
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log(p_xy / (p_x * p_y))

        return max(0.0, mi)

    def compute_vif(self, X: np.ndarray, feature_idx: int) -> float:
        """Compute Variance Inflation Factor for a feature.

        Args:
            X: Feature matrix
            feature_idx: Index of feature to compute VIF for

        Returns:
            VIF value
        """
        from numpy.linalg import lstsq

        y = X[:, feature_idx]
        X_others = np.delete(X, feature_idx, axis=1)

        # Add intercept
        X_design = np.column_stack([np.ones(len(y)), X_others])

        # Fit regression
        try:
            coeffs, _, _, _ = lstsq(X_design, y, rcond=None)
            y_pred = X_design @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if r_squared >= 1:
                return 100.0  # Cap at high value
            return 1 / (1 - r_squared)
        except Exception:
            return 1.0


class MutualInformationSelector:
    """Feature selection using Mutual Information.

    Implements multiple selection strategies:
    - Univariate MI: Select top-k by MI with target
    - mRMR: Minimum Redundancy Maximum Relevance
    - JMI: Joint Mutual Information
    - CMIM: Conditional MI Maximization

    Example:
        selector = MutualInformationSelector()
        result = selector.select_features(X_train, y_train, feature_names)
        X_selected = X_train[:, result.selected_indices]
    """

    def __init__(self, config: Optional[MIConfig] = None):
        self.config = config or MIConfig()
        self.calculator = MutualInformationCalculator(config)

        # Cache for regime-specific selections
        self._regime_selections: Dict[str, SelectionResult] = {}

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        method: SelectionMethod = SelectionMethod.MRMR,
        regime: Optional[str] = None,
        is_classification: bool = True,
    ) -> SelectionResult:
        """Select optimal features using MI-based method.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target array
            feature_names: Names of features
            method: Selection method to use
            regime: Optional market regime for regime-specific selection
            is_classification: If True, treat target as categorical

        Returns:
            SelectionResult with selected features
        """
        logger.info(f"Selecting features using {method.value} method")

        n_samples, n_features = X.shape

        # Step 1: Compute MI scores with target
        feature_scores = self.calculator.compute_mi_scores(
            X, y, feature_names, is_classification
        )

        # Step 2: Filter by MI threshold
        low_mi_filtered = []
        for score in feature_scores:
            if score.mi_score < self.config.mi_threshold:
                low_mi_filtered.append(score.feature_name)

        # Step 3: Compute feature-feature correlation for redundancy
        corr_matrix = self.calculator.compute_feature_correlation(X)

        # Step 4: Apply selection method
        if method == SelectionMethod.UNIVARIATE_MI:
            selected_indices = self._select_univariate(feature_scores)
        elif method == SelectionMethod.MRMR:
            selected_indices = self._select_mrmr(feature_scores, corr_matrix)
        elif method == SelectionMethod.JMI:
            mi_matrix = self.calculator.compute_feature_mi(X)
            selected_indices = self._select_jmi(feature_scores, mi_matrix)
        elif method == SelectionMethod.CMIM:
            mi_matrix = self.calculator.compute_feature_mi(X)
            selected_indices = self._select_cmim(feature_scores, mi_matrix)
        else:
            selected_indices = self._select_mrmr(feature_scores, corr_matrix)

        # Step 5: Identify redundancy-filtered features
        redundancy_filtered = self._get_redundant_features(
            feature_scores, selected_indices, corr_matrix
        )

        # Mark selected features
        for idx in selected_indices:
            feature_scores[idx].selected = True
            feature_scores[idx].regime = regime

        # Compute VIF for selected features
        if len(selected_indices) > 1:
            X_selected = X[:, selected_indices]
            for i, idx in enumerate(selected_indices):
                feature_scores[idx].vif = self.calculator.compute_vif(X_selected, i)

        # Build result
        selected_names = [feature_scores[i].feature_name for i in selected_indices]
        total_mi = sum(feature_scores[i].mi_score for i in selected_indices)

        result = SelectionResult(
            selected_features=selected_names,
            selected_indices=selected_indices,
            feature_scores=feature_scores,
            method=method,
            regime=regime,
            n_original=n_features,
            n_selected=len(selected_indices),
            total_mi=total_mi,
            redundancy_filtered=redundancy_filtered,
            low_mi_filtered=low_mi_filtered,
        )

        # Cache regime-specific results
        if regime:
            self._regime_selections[regime] = result

        logger.info(
            f"Selected {len(selected_indices)}/{n_features} features, "
            f"total MI: {total_mi:.4f}"
        )

        return result

    def _select_univariate(
        self,
        scores: List[FeatureScore],
    ) -> List[int]:
        """Select top features by univariate MI."""
        # Sort by MI score
        sorted_scores = sorted(scores, key=lambda x: x.mi_score, reverse=True)

        # Filter by threshold and select top-k
        selected = []
        for score in sorted_scores:
            if score.mi_score >= self.config.mi_threshold:
                selected.append(score.feature_index)
                if len(selected) >= self.config.max_features:
                    break

        return selected[:max(self.config.min_features, len(selected))]

    def _select_mrmr(
        self,
        scores: List[FeatureScore],
        corr_matrix: np.ndarray,
    ) -> List[int]:
        """Select features using mRMR (Minimum Redundancy Maximum Relevance).

        mRMR score = MI(f, y) - (1/|S|) * sum(corr(f, s)) for s in S
        """
        n_features = len(scores)
        selected: List[int] = []
        remaining = set(range(n_features))

        # Filter by MI threshold first
        remaining = {
            i for i in remaining
            if scores[i].mi_score >= self.config.mi_threshold
        }

        if not remaining:
            # Fallback: take top by MI
            return self._select_univariate(scores)

        while len(selected) < self.config.max_features and remaining:
            best_idx = -1
            best_score = -np.inf

            for idx in remaining:
                mi = scores[idx].mi_score

                # Compute redundancy with selected features
                if selected:
                    redundancy = np.mean([corr_matrix[idx, s] for s in selected])
                else:
                    redundancy = 0.0

                # mRMR score
                mrmr = mi - redundancy

                # Update score object
                scores[idx].redundancy_score = redundancy
                scores[idx].mrmr_score = mrmr

                if mrmr > best_score:
                    best_score = mrmr
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

                # Skip highly correlated features
                to_remove = set()
                for idx in remaining:
                    if corr_matrix[best_idx, idx] > self.config.redundancy_threshold:
                        to_remove.add(idx)
                remaining -= to_remove
            else:
                break

        return selected

    def _select_jmi(
        self,
        scores: List[FeatureScore],
        mi_matrix: np.ndarray,
    ) -> List[int]:
        """Select features using Joint Mutual Information.

        JMI maximizes relevance while penalizing redundancy:
        JMI score = MI(f, y) - (1/|S|) * sum(MI(f, s)) for s in S

        This is similar to mRMR but uses MI for redundancy instead of correlation.
        """
        n_features = len(scores)
        selected: List[int] = []
        remaining = set(range(n_features))

        # Filter by threshold
        remaining = {
            i for i in remaining
            if scores[i].mi_score >= self.config.mi_threshold
        }

        while len(selected) < self.config.max_features and remaining:
            best_idx = -1
            best_score = -np.inf

            for idx in remaining:
                mi = scores[idx].mi_score

                # JMI: SUBTRACT redundancy (feature-feature MI) to penalize correlated features
                if selected:
                    redundancy = sum(mi_matrix[idx, s] for s in selected) / len(selected)
                else:
                    redundancy = 0

                jmi_score = mi - redundancy  # FIXED: subtract, not add

                if jmi_score > best_score:
                    best_score = jmi_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        return selected

    def _select_cmim(
        self,
        scores: List[FeatureScore],
        mi_matrix: np.ndarray,
    ) -> List[int]:
        """Select features using Conditional Mutual Information Maximization.

        CMIM: Select feature that maximizes min(MI(f,y|s)) over s in S

        Note: This uses an approximation I(X;Y|Z) ≈ I(X;Y) - I(X;Z)
        which is a lower bound when X and Z are positively correlated with Y.
        The true formula is I(X;Y|Z) = H(X|Z) - H(X|Y,Z), but computing this
        requires joint distributions that are expensive to estimate.

        For feature selection, this approximation works well in practice because:
        1. It penalizes redundant features (high I(X;Z))
        2. It preserves relative ranking of features
        3. Features with high unconditional MI are still prioritized
        """
        n_features = len(scores)
        selected: List[int] = []
        remaining = set(range(n_features))

        # Filter by threshold
        remaining = {
            i for i in remaining
            if scores[i].mi_score >= self.config.mi_threshold
        }

        # First feature: highest MI
        if remaining:
            first = max(remaining, key=lambda i: scores[i].mi_score)
            selected.append(first)
            remaining.remove(first)

        while len(selected) < self.config.max_features and remaining:
            best_idx = -1
            best_score = -np.inf

            for idx in remaining:
                mi = scores[idx].mi_score

                # CMIM: minimum conditional MI (using approximation)
                # I(X;Y|Z) ≈ I(X;Y) - I(X;Z) is a lower bound approximation
                # This ensures we penalize features redundant with already selected ones
                min_cmi = mi
                for s in selected:
                    # Approximate: subtract feature-feature MI to penalize redundancy
                    cmi = mi - mi_matrix[idx, s]
                    # Clamp to non-negative (MI is always >= 0 by definition)
                    cmi = max(0.0, cmi)
                    min_cmi = min(min_cmi, cmi)

                if min_cmi > best_score:
                    best_score = min_cmi
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        return selected

    def _get_redundant_features(
        self,
        scores: List[FeatureScore],
        selected: List[int],
        corr_matrix: np.ndarray,
    ) -> List[str]:
        """Identify features filtered due to redundancy."""
        redundant = []
        selected_set = set(selected)

        for i, score in enumerate(scores):
            if i in selected_set:
                continue
            if score.mi_score < self.config.mi_threshold:
                continue

            # Check if filtered due to correlation
            for s in selected:
                if corr_matrix[i, s] > self.config.redundancy_threshold:
                    redundant.append(score.feature_name)
                    break

        return redundant

    def get_regime_features(self, regime: str) -> Optional[List[int]]:
        """Get selected feature indices for a specific regime."""
        if regime in self._regime_selections:
            return self._regime_selections[regime].selected_indices
        return None

    def get_feature_recommendations(
        self,
        scores: List[FeatureScore],
    ) -> Dict[str, List[str]]:
        """Get feature recommendations based on scores.

        Returns:
            Dict with 'keep', 'remove', 'investigate' lists
        """
        keep = []
        remove = []
        investigate = []

        for score in scores:
            if score.mi_score < self.config.mi_threshold:
                remove.append(f"{score.feature_name} (low MI: {score.mi_score:.4f})")
            elif score.vif and score.vif > self.config.vif_threshold:
                investigate.append(
                    f"{score.feature_name} (high VIF: {score.vif:.2f})"
                )
            elif score.redundancy_score > self.config.redundancy_threshold:
                investigate.append(
                    f"{score.feature_name} (redundant: {score.redundancy_score:.2f})"
                )
            elif score.selected:
                keep.append(f"{score.feature_name} (mRMR: {score.mrmr_score:.4f})")

        return {
            "keep": keep,
            "remove": remove,
            "investigate": investigate,
        }


# Regime-specific feature sets (defaults based on domain knowledge)
REGIME_FEATURE_PRIORITIES = {
    "MEME_SEASON": [
        "holder_growth_velocity",
        "price_momentum_30s",
        "volume_acceleration",
        "twitter_mention_velocity",
        "buy_volume_ratio",
        "pool_tvl_sol",
        "social_authenticity_score",
    ],
    "BEAR_VOLATILE": [
        "rug_pull_ml_score",
        "top_10_holder_concentration",
        "mint_authority_revoked",
        "pool_tvl_sol",
        "volatility_5m",
        "hidden_fee_detected",
        "lp_lock_percentage",
    ],
    "HIGH_MEV": [
        "volatility_5m",
        "slippage_1pct",
        "slippage_5pct",
        "price_impact_1pct",
        "pool_depth_imbalance",
        "consecutive_buys",
        "trade_size_variance",
    ],
    "NORMAL": [
        "pool_tvl_sol",
        "holder_count_unique",
        "price_momentum_5m",
        "rug_pull_ml_score",
        "lp_lock_percentage",
        "top_10_holder_concentration",
        "volume_acceleration",
    ],
    "LOW_LIQUIDITY": [
        "pool_tvl_sol",
        "unique_lp_provider_count",
        "lp_removal_velocity",
        "emergency_liquidity_flag",
        "slippage_5pct",
        "deployer_lp_ownership_pct",
        "price_impact_1pct",
    ],
}


def get_regime_feature_mask(
    regime: str,
    all_feature_names: List[str],
) -> np.ndarray:
    """Get boolean mask for regime-priority features.

    Args:
        regime: Market regime
        all_feature_names: List of all feature names

    Returns:
        Boolean mask array
    """
    priority_features = REGIME_FEATURE_PRIORITIES.get(regime, [])
    mask = np.zeros(len(all_feature_names), dtype=bool)

    for i, name in enumerate(all_feature_names):
        if name in priority_features:
            mask[i] = True

    return mask
