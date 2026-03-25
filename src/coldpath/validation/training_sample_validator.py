"""
Training Sample Validator - Specialized validation for ML training samples.

Validates TrainingSample objects and training data dictionaries for:
- Feature vector completeness (50 features)
- Feature value bounds and sanity
- Label consistency and correctness
- Data distribution quality
- Temporal consistency
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from coldpath.learning.feature_engineering import FEATURE_COUNT, FeatureSet

from .data_quality import ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)


# Feature bounds based on normalized feature ranges
# These are the expected ranges after normalization in HotPath
FEATURE_BOUNDS = {
    # Liquidity & Pool Metrics (0-11)
    "pool_tvl_sol": {"min": -1.0, "max": 1.0, "typical_max": 0.8},
    "pool_age_seconds": {"min": -1.0, "max": 1.0, "typical_max": 0.9},
    "lp_lock_percentage": {"min": -1.0, "max": 1.0, "typical_max": 1.0},
    "lp_concentration": {"min": -1.0, "max": 1.0, "typical_max": 0.9},
    "lp_removal_velocity": {"min": -1.0, "max": 1.0, "typical_max": 0.7},
    "lp_addition_velocity": {"min": -1.0, "max": 1.0, "typical_max": 0.7},
    "pool_depth_imbalance": {"min": -1.0, "max": 1.0, "typical_max": 0.9},
    "slippage_1pct": {"min": -1.0, "max": 1.0, "typical_max": 0.8},
    "slippage_5pct": {"min": -1.0, "max": 1.0, "typical_max": 0.8},
    "unique_lp_provider_count": {"min": -1.0, "max": 1.0, "typical_max": 0.8},
    "deployer_lp_ownership_pct": {"min": -1.0, "max": 1.0, "typical_max": 0.9},
    "emergency_liquidity_flag": {"min": -1.0, "max": 1.0, "typical_max": 1.0},
    # Token Supply & Holder (12-22)
    "total_supply": {"min": -1.0, "max": 1.0, "typical_max": 0.8},
    "deployer_holdings_pct": {"min": -1.0, "max": 1.0, "typical_max": 0.9},
    "top_10_holder_concentration": {"min": -1.0, "max": 1.0, "typical_max": 0.9},
    "holder_count_unique": {"min": -1.0, "max": 1.0, "typical_max": 0.8},
    "holder_growth_velocity": {"min": -1.0, "max": 1.0, "typical_max": 0.9},
    "transfer_concentration": {"min": -1.0, "max": 1.0, "typical_max": 0.8},
    "sniper_bot_count_t0": {"min": -1.0, "max": 1.0, "typical_max": 0.7},
    "bot_to_human_ratio": {"min": -1.0, "max": 1.0, "typical_max": 0.9},
    "large_holder_churn": {"min": -1.0, "max": 1.0, "typical_max": 0.7},
    "mint_authority_revoked": {"min": -1.0, "max": 1.0, "typical_max": 1.0},
    "token_freezeable": {"min": -1.0, "max": 1.0, "typical_max": 1.0},
    # Price & Volume (23-32)
    "price_momentum_30s": {"min": -1.0, "max": 1.0, "typical_max": 0.95},
    "price_momentum_5m": {"min": -1.0, "max": 1.0, "typical_max": 0.95},
    "volatility_5m": {"min": -1.0, "max": 1.0, "typical_max": 0.9},
    "volume_acceleration": {"min": -1.0, "max": 1.0, "typical_max": 0.8},
    "buy_volume_ratio": {"min": -1.0, "max": 1.0, "typical_max": 0.9},
    "trade_size_variance": {"min": -1.0, "max": 1.0, "typical_max": 0.8},
    "vwap_deviation": {"min": -1.0, "max": 1.0, "typical_max": 0.9},
    "price_impact_1pct": {"min": -1.0, "max": 1.0, "typical_max": 0.8},
    "consecutive_buys": {"min": -1.0, "max": 1.0, "typical_max": 0.9},
    "max_buy_in_window": {"min": -1.0, "max": 1.0, "typical_max": 0.9},
    # On-Chain Risk (33-42)
    "contract_is_mintable": {"min": -1.0, "max": 1.0, "typical_max": 1.0},
    "contract_transfer_fee": {"min": -1.0, "max": 1.0, "typical_max": 0.8},
    "hidden_fee_detected": {"min": -1.0, "max": 1.0, "typical_max": 1.0},
    "circular_trading_score": {"min": -1.0, "max": 1.0, "typical_max": 0.7},
    "benford_law_pvalue": {"min": -1.0, "max": 1.0, "typical_max": 0.9},
    "address_clustering_risk": {"min": -1.0, "max": 1.0, "typical_max": 0.8},
    "proxy_contract_flag": {"min": -1.0, "max": 1.0, "typical_max": 1.0},
    "unverified_code_flag": {"min": -1.0, "max": 1.0, "typical_max": 1.0},
    "external_transfer_flag": {"min": -1.0, "max": 1.0, "typical_max": 1.0},
    "rug_pull_ml_score": {"min": -1.0, "max": 1.0, "typical_max": 0.8},
    # Social & Sentiment (43-49) - Note: these are proxied, may have limited range
    "twitter_mention_velocity": {"min": -1.0, "max": 1.0, "typical_max": 0.6},
    "twitter_sentiment_score": {"min": -1.0, "max": 1.0, "typical_max": 0.7},
    "telegram_user_growth": {"min": -1.0, "max": 1.0, "typical_max": 0.6},
    "telegram_message_velocity": {"min": -1.0, "max": 1.0, "typical_max": 0.6},
    "discord_invite_activity": {"min": -1.0, "max": 1.0, "typical_max": 0.0},  # Always zero
    "influencer_mention_flag": {"min": -1.0, "max": 1.0, "typical_max": 0.5},
    "social_authenticity_score": {"min": -1.0, "max": 1.0, "typical_max": 0.6},
}


@dataclass
class TrainingSampleValidationResult:
    """Result of training sample validation."""

    valid: bool
    total_samples: int
    valid_samples: int
    invalid_samples: int
    issues: list[ValidationIssue] = field(default_factory=list)

    # Feature statistics
    feature_completeness: float = 0.0  # % samples with all 50 features
    feature_bounds_compliance: float = 0.0  # % features within bounds
    nan_count: int = 0
    inf_count: int = 0

    # Label statistics
    label_consistency: float = 0.0  # % samples with consistent labels
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0

    # Data quality metrics
    uniqueness_score: float = 0.0  # % unique feature vectors
    temporal_consistency: float = 0.0  # % samples in chronological order

    def summary(self) -> str:
        """Human-readable summary."""
        status = "✅ PASSED" if self.valid else "❌ FAILED"
        return (
            f"Training Sample Validation: {status} | "
            f"Samples: {self.valid_samples}/{self.total_samples} valid | "
            f"Feature completeness: {self.feature_completeness:.1%} | "
            f"Label consistency: {self.label_consistency:.1%} | "
            f"Issues: {len(self.issues)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "invalid_samples": self.invalid_samples,
            "issues": [issue.to_dict() for issue in self.issues],
            "feature_completeness": self.feature_completeness,
            "feature_bounds_compliance": self.feature_bounds_compliance,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "label_consistency": self.label_consistency,
            "positive_ratio": self.positive_ratio,
            "negative_ratio": self.negative_ratio,
            "uniqueness_score": self.uniqueness_score,
            "temporal_consistency": self.temporal_consistency,
        }


class TrainingSampleValidator:
    """Validates training samples for ML model training.

    Comprehensive validation including:
    - Feature vector completeness (all 50 features)
    - Feature value sanity (within bounds, no NaN/Inf)
    - Label correctness and consistency
    - Data distribution quality
    - Temporal ordering
    - Duplicate detection

    Usage:
        validator = TrainingSampleValidator()
        result = validator.validate_samples(samples)
        if not result.valid:
            logger.error(f"Validation failed: {result.summary()}")
    """

    def __init__(
        self,
        strict_mode: bool = False,
        min_samples: int = 100,
        max_nan_ratio: float = 0.05,
        max_inf_ratio: float = 0.01,
    ):
        """Initialize validator.

        Args:
            strict_mode: If True, reject samples with any issues
            min_samples: Minimum number of samples required
            max_nan_ratio: Maximum allowed ratio of NaN values
            max_inf_ratio: Maximum allowed ratio of Inf values
        """
        self.strict_mode = strict_mode
        self.min_samples = min_samples
        self.max_nan_ratio = max_nan_ratio
        self.max_inf_ratio = max_inf_ratio
        self.feature_names = FeatureSet().FEATURE_NAMES

    def validate_samples(
        self,
        samples: list[Any],
    ) -> TrainingSampleValidationResult:
        """Validate a list of training samples.

        Args:
            samples: List of TrainingSample objects or training data dicts

        Returns:
            TrainingSampleValidationResult with validation details
        """
        if not samples:
            return TrainingSampleValidationResult(
                valid=False,
                total_samples=0,
                valid_samples=0,
                invalid_samples=0,
                issues=[
                    ValidationIssue(
                        code="NO_SAMPLES",
                        message="No training samples provided",
                        severity=ValidationSeverity.ERROR,
                    )
                ],
            )

        issues = []
        valid_count = 0
        invalid_count = 0

        # Track statistics
        complete_features = 0
        features_in_bounds = 0
        total_feature_checks = 0
        nan_count = 0
        inf_count = 0
        consistent_labels = 0
        positive_count = 0
        negative_count = 0
        unique_feature_vectors = set()
        timestamps = []

        # Aggregated warning counters (avoid creating thousands of individual issues)
        feature_out_of_bounds_counts: dict[str, int] = {}
        feature_unusual_value_counts: dict[str, int] = {}
        nan_samples = 0
        inf_samples = 0

        # Validate each sample
        for i, sample in enumerate(samples):
            is_valid = True

            # Extract data from sample
            features, labels, timestamp = self._extract_sample_data(sample)

            if features is None:
                invalid_count += 1
                issues.append(
                    ValidationIssue(
                        code="EXTRACTION_FAILED",
                        message=f"Sample {i}: Failed to extract features",
                        severity=ValidationSeverity.ERROR,
                        count=1,
                    )
                )
                continue

            # 1. Feature completeness check
            if len(features) != FEATURE_COUNT:
                invalid_count += 1
                issues.append(
                    ValidationIssue(
                        code="FEATURE_COUNT_MISMATCH",
                        message=(
                            f"Sample {i}: Expected {FEATURE_COUNT} features, got {len(features)}"
                        ),
                        severity=ValidationSeverity.ERROR,
                        count=1,
                        details={"expected": FEATURE_COUNT, "actual": len(features)},
                    )
                )
                is_valid = False
            else:
                complete_features += 1

            # 2. Feature value validation (aggregated)
            feature_issues = self._validate_features_aggregated(
                features, feature_out_of_bounds_counts, feature_unusual_value_counts
            )
            if feature_issues:
                # Only extend with ERROR/CRITICAL issues (warnings are aggregated)
                errors = [
                    iss
                    for iss in feature_issues
                    if iss.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
                ]
                if errors:
                    issues.extend(errors)
                    is_valid = False
                features_in_bounds += 1
            else:
                features_in_bounds += 1
            total_feature_checks += 1

            # 3. NaN/Inf check (aggregated)
            nan_in_sample = np.sum(np.isnan(features))
            inf_in_sample = np.sum(np.isinf(features))
            nan_count += nan_in_sample
            inf_count += inf_in_sample

            if nan_in_sample > 0:
                nan_samples += 1

            if inf_in_sample > 0:
                inf_samples += 1

            # 4. Label validation
            if labels is not None:
                label_issues = self._validate_labels(labels, features, i)
                if label_issues:
                    issues.extend(label_issues)
                    if any(
                        issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
                        for issue in label_issues
                    ):
                        is_valid = False
                else:
                    consistent_labels += 1
                    if labels.get("was_profitable", False):
                        positive_count += 1
                    else:
                        negative_count += 1

            # 5. Track uniqueness
            feature_tuple = tuple(features)
            unique_feature_vectors.add(feature_tuple)

            # 6. Track timestamps
            if timestamp:
                timestamps.append(timestamp)

            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1

        # Add aggregated warnings to issues list
        for feature_name, count in feature_out_of_bounds_counts.items():
            issues.append(
                ValidationIssue(
                    code="FEATURE_OUT_OF_BOUNDS",
                    message=(
                        f"Feature '{feature_name}' had {count} values outside bounds [-1.0, 1.0]"
                    ),
                    severity=ValidationSeverity.WARNING,
                    field=feature_name,
                    count=count,
                )
            )

        for feature_name, count in feature_unusual_value_counts.items():
            issues.append(
                ValidationIssue(
                    code="FEATURE_UNUSUAL_VALUE",
                    message=f"Feature '{feature_name}' had {count} values exceeding typical max",
                    severity=ValidationSeverity.INFO,
                    field=feature_name,
                    count=count,
                )
            )

        if nan_samples > 0:
            issues.append(
                ValidationIssue(
                    code="NAN_VALUES",
                    message=f"{nan_samples} samples contain NaN values ({nan_count} total)",
                    severity=ValidationSeverity.WARNING,
                    count=nan_count,
                )
            )

        if inf_samples > 0:
            issues.append(
                ValidationIssue(
                    code="INF_VALUES",
                    message=f"{inf_samples} samples contain Inf values ({inf_count} total)",
                    severity=ValidationSeverity.WARNING,
                    count=inf_count,
                )
            )

        # Aggregate statistics
        total_samples = len(samples)

        # Feature statistics
        feature_completeness = complete_features / total_samples if total_samples > 0 else 0
        feature_bounds_compliance = (
            features_in_bounds / total_feature_checks if total_feature_checks > 0 else 0
        )

        # Label statistics
        label_consistency = consistent_labels / total_samples if total_samples > 0 else 0
        positive_ratio = (
            positive_count / (positive_count + negative_count)
            if (positive_count + negative_count) > 0
            else 0
        )
        negative_ratio = 1.0 - positive_ratio

        # Uniqueness
        uniqueness_score = len(unique_feature_vectors) / total_samples if total_samples > 0 else 0

        # Temporal consistency
        temporal_consistency = self._check_temporal_ordering(timestamps)

        # Global checks
        if total_samples < self.min_samples:
            issues.append(
                ValidationIssue(
                    code="INSUFFICIENT_SAMPLES",
                    message=f"Only {total_samples} samples, need at least {self.min_samples}",
                    severity=ValidationSeverity.ERROR,
                    details={"total": total_samples, "required": self.min_samples},
                )
            )

        if total_samples > 0:
            nan_ratio = nan_count / (total_samples * FEATURE_COUNT)
            if nan_ratio > self.max_nan_ratio:
                issues.append(
                    ValidationIssue(
                        code="HIGH_NAN_RATIO",
                        message=(
                            f"NaN ratio {nan_ratio:.2%} exceeds maximum {self.max_nan_ratio:.2%}"
                        ),
                        severity=ValidationSeverity.ERROR,
                        details={"nan_ratio": nan_ratio, "max_allowed": self.max_nan_ratio},
                    )
                )

            inf_ratio = inf_count / (total_samples * FEATURE_COUNT)
            if inf_ratio > self.max_inf_ratio:
                issues.append(
                    ValidationIssue(
                        code="HIGH_INF_RATIO",
                        message=(
                            f"Inf ratio {inf_ratio:.2%} exceeds maximum {self.max_inf_ratio:.2%}"
                        ),
                        severity=ValidationSeverity.ERROR,
                        details={"inf_ratio": inf_ratio, "max_allowed": self.max_inf_ratio},
                    )
                )

        # Check for duplicate samples
        duplicate_count = total_samples - len(unique_feature_vectors)
        if duplicate_count > 0:
            dup_pct = duplicate_count / total_samples * 100
            severity = ValidationSeverity.WARNING if dup_pct < 5 else ValidationSeverity.ERROR
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_SAMPLES",
                    message=f"Found {duplicate_count} duplicate samples ({dup_pct:.1f}%)",
                    severity=severity,
                    count=duplicate_count,
                )
            )

        # Determine overall validity
        has_critical_errors = any(
            issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for issue in issues
        )

        valid = not has_critical_errors and valid_count > 0

        return TrainingSampleValidationResult(
            valid=valid,
            total_samples=total_samples,
            valid_samples=valid_count,
            invalid_samples=invalid_count,
            issues=issues,
            feature_completeness=feature_completeness,
            feature_bounds_compliance=feature_bounds_compliance,
            nan_count=nan_count,
            inf_count=inf_count,
            label_consistency=label_consistency,
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            uniqueness_score=uniqueness_score,
            temporal_consistency=temporal_consistency,
        )

    def _extract_sample_data(
        self,
        sample: Any,
    ) -> tuple[np.ndarray | None, dict | None, int | None]:
        """Extract features, labels, and timestamp from sample.

        Args:
            sample: TrainingSample object or dict

        Returns:
            Tuple of (features, labels_dict, timestamp_ms)
        """
        try:
            # Handle TrainingSample object
            if hasattr(sample, "features"):
                features = np.array(sample.features)
                labels = {
                    "was_profitable": getattr(sample, "was_profitable", None),
                    "pnl_percentage": getattr(sample, "pnl_percentage", None),
                    "pnl_sol": getattr(sample, "pnl_sol", None),
                }
                timestamp_str = getattr(sample, "timestamp", None)
                timestamp = self._parse_timestamp(timestamp_str) if timestamp_str else None
                return features, labels, timestamp

            # Handle dict format
            if isinstance(sample, dict):
                # Extract features
                if "features" in sample:
                    features = np.array(sample["features"])
                elif "features_json" in sample:
                    import json

                    features_dict = (
                        json.loads(sample["features_json"])
                        if isinstance(sample["features_json"], str)
                        else sample["features_json"]
                    )
                    features = self._dict_to_array(features_dict)
                else:
                    return None, None, None

                # Extract labels with multiple fallbacks so we can accept
                # datasets coming from the data collector (label_profit/label_2x),
                # HotPath decisions (decision=Go/NoGo), or nested outcome objects.
                outcome = sample.get("outcome") or {}

                def _first_present(*values):
                    for v in values:
                        if v is not None:
                            return v
                    return None

                was_profitable = _first_present(
                    sample.get("was_profitable"),
                    sample.get("label_binary"),
                    outcome.get("was_profitable"),
                    outcome.get("wasProfitable"),
                )

                if was_profitable is None and "label_profit" in sample:
                    was_profitable = bool(sample.get("label_profit"))

                if was_profitable is None and "label_2x" in sample:
                    # Treat 2x pumps as profitable by definition
                    was_profitable = bool(sample.get("label_2x"))

                if was_profitable is None and "decision" in sample:
                    decision = str(sample.get("decision")).lower()
                    if decision in {"go", "buy", "trade", "long", "yes"}:
                        was_profitable = True
                    elif decision in {"nogo", "no_go", "reject", "skip", "no"}:
                        was_profitable = False

                # PnL percentage fallbacks
                pnl_percentage = _first_present(
                    sample.get("pnl_percentage"),
                    sample.get("pnl_pct"),
                    outcome.get("pnl_pct"),
                    outcome.get("pnl_percentage"),
                )

                # If raw prices are present, derive pnl_percentage
                if pnl_percentage is None:
                    price_0 = sample.get("price_0") or outcome.get("price_0")
                    price_1h = sample.get("price_1h") or outcome.get("price_1h")
                    try:
                        if price_0 and price_0 > 0 and price_1h is not None:
                            pnl_percentage = (
                                (float(price_1h) - float(price_0)) / float(price_0) * 100.0
                            )
                    except Exception:
                        pnl_percentage = None

                labels = {
                    "was_profitable": was_profitable,
                    "pnl_percentage": pnl_percentage,
                    "pnl_sol": sample.get("pnl_sol") or outcome.get("pnl_sol"),
                }

                # Extract timestamp
                timestamp = (
                    sample.get("timestamp_ms")
                    or sample.get("timestamp")
                    or sample.get("decision_timestamp_ms")
                    or outcome.get("timestamp_ms")
                    or outcome.get("timestamp")
                    or sample.get("collected_at")
                    or sample.get("created_at")
                )

                # If we received a seconds-based timestamp, convert to ms
                if timestamp and isinstance(timestamp, (int, float)) and timestamp < 1e12:
                    timestamp = int(timestamp * 1000)
                if isinstance(timestamp, str):
                    timestamp = self._parse_timestamp(timestamp)

                return features, labels, timestamp

        except Exception as e:
            logger.debug(f"Failed to extract sample data: {e}")
            return None, None, None

        return None, None, None

    def _dict_to_array(self, features_dict: dict[str, float]) -> np.ndarray:
        """Convert feature dict to array using canonical feature order."""
        arr = np.zeros(FEATURE_COUNT)
        for i, name in enumerate(self.feature_names):
            if i >= FEATURE_COUNT:
                break
            arr[i] = features_dict.get(name, 0.0)
        return arr

    def _validate_features_aggregated(
        self,
        features: np.ndarray,
        out_of_bounds_counts: dict[str, int],
        unusual_value_counts: dict[str, int],
    ) -> list[ValidationIssue]:
        """Validate feature values and aggregate warnings by feature name.

        This is an optimized version that counts issues instead of creating
        individual ValidationIssue objects for each occurrence.

        Args:
            features: Feature vector to validate
            out_of_bounds_counts: Dict to accumulate out-of-bounds counts
            unusual_value_counts: Dict to accumulate unusual value counts

        Returns:
            List of critical/error issues only (warnings are aggregated into dicts)
        """
        issues = []

        for i, (value, name) in enumerate(zip(features, self.feature_names, strict=False)):
            if i >= FEATURE_COUNT:
                break

            # Check bounds
            bounds = FEATURE_BOUNDS.get(name, {})
            min_val = bounds.get("min", -1.0)
            max_val = bounds.get("max", 1.0)
            typical_max = bounds.get("typical_max", 1.0)

            if not np.isfinite(value):
                # NaN/Inf handled separately
                continue

            if value < min_val or value > max_val:
                out_of_bounds_counts[name] = out_of_bounds_counts.get(name, 0) + 1
            elif abs(value) > typical_max:
                unusual_value_counts[name] = unusual_value_counts.get(name, 0) + 1

        return issues

    def _validate_features(
        self,
        features: np.ndarray,
        sample_index: int,
    ) -> list[ValidationIssue]:
        """Validate feature values."""
        issues = []

        for i, (value, name) in enumerate(zip(features, self.feature_names, strict=False)):
            if i >= FEATURE_COUNT:
                break

            # Check bounds
            bounds = FEATURE_BOUNDS.get(name, {})
            min_val = bounds.get("min", -1.0)
            max_val = bounds.get("max", 1.0)
            typical_max = bounds.get("typical_max", 1.0)

            if not np.isfinite(value):
                # NaN/Inf handled separately
                continue

            if value < min_val or value > max_val:
                issues.append(
                    ValidationIssue(
                        code="FEATURE_OUT_OF_BOUNDS",
                        message=(
                            f"Sample {sample_index}: Feature '{name}' value {value:.3f} "
                            f"outside bounds [{min_val}, {max_val}]"
                        ),
                        severity=ValidationSeverity.WARNING,
                        field=name,
                        count=1,
                        details={"value": float(value), "min": min_val, "max": max_val},
                    )
                )
            elif abs(value) > typical_max:
                # Value is within hard bounds but unusually high
                issues.append(
                    ValidationIssue(
                        code="FEATURE_UNUSUAL_VALUE",
                        message=(
                            f"Sample {sample_index}: Feature '{name}' value {value:.3f} "
                            f"exceeds typical max {typical_max}"
                        ),
                        severity=ValidationSeverity.INFO,
                        field=name,
                        count=1,
                        details={"value": float(value), "typical_max": typical_max},
                    )
                )

        return issues

    def _validate_labels(
        self,
        labels: dict[str, Any],
        features: np.ndarray,
        sample_index: int,
    ) -> list[ValidationIssue]:
        """Validate label consistency."""
        issues = []

        was_profitable = labels.get("was_profitable")
        pnl_percentage = labels.get("pnl_percentage")
        pnl_sol = labels.get("pnl_sol")

        # Check binary label exists
        if was_profitable is None:
            issues.append(
                ValidationIssue(
                    code="MISSING_BINARY_LABEL",
                    message=f"Sample {sample_index}: Missing binary label (was_profitable)",
                    severity=ValidationSeverity.ERROR,
                    count=1,
                )
            )
            return issues

        # Check consistency between binary label and pnl
        if pnl_percentage is not None:
            expected_profitable = pnl_percentage > 0
            if was_profitable != expected_profitable:
                issues.append(
                    ValidationIssue(
                        code="LABEL_INCONSISTENCY",
                        message=(
                            f"Sample {sample_index}: Binary label {was_profitable} doesn't match "
                            f"pnl_percentage {pnl_percentage:.2f}%"
                        ),
                        severity=ValidationSeverity.ERROR,
                        count=1,
                        details={
                            "was_profitable": was_profitable,
                            "pnl_percentage": pnl_percentage,
                            "expected": expected_profitable,
                        },
                    )
                )

        if pnl_sol is not None:
            expected_profitable = pnl_sol > 0
            if was_profitable != expected_profitable:
                issues.append(
                    ValidationIssue(
                        code="LABEL_INCONSISTENCY",
                        message=(
                            f"Sample {sample_index}: Binary label {was_profitable} "
                            f"doesn't match pnl_sol {pnl_sol:.4f}"
                        ),
                        severity=ValidationSeverity.WARNING,
                        count=1,
                        details={
                            "was_profitable": was_profitable,
                            "pnl_sol": pnl_sol,
                            "expected": expected_profitable,
                        },
                    )
                )

        return issues

    def _parse_timestamp(self, timestamp_str: str) -> int | None:
        """Parse timestamp string to milliseconds."""
        try:
            # Try ISO format
            if "T" in timestamp_str:
                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                return int(dt.timestamp() * 1000)
            # Try numeric string
            return int(timestamp_str)
        except (ValueError, TypeError):
            return None

    def _check_temporal_ordering(self, timestamps: list[int]) -> float:
        """Check if timestamps are in chronological order."""
        if len(timestamps) < 2:
            return 1.0

        sorted_timestamps = sorted(timestamps)
        violations = 0

        for _i, (actual, expected) in enumerate(zip(timestamps, sorted_timestamps, strict=False)):
            if actual != expected:
                violations += 1

        return 1.0 - (violations / len(timestamps))

    def validate_feature_distribution(
        self,
        samples: list[Any],
    ) -> dict[str, dict[str, float]]:
        """Analyze feature distribution across samples.

        Returns statistics for each feature:
        - mean, std, min, max
        - nan_ratio, inf_ratio
        - zero_ratio
        """
        if not samples:
            return {}

        # Extract all features
        all_features = []
        for sample in samples:
            features, _, _ = self._extract_sample_data(sample)
            if features is not None and len(features) == FEATURE_COUNT:
                all_features.append(features)

        if not all_features:
            return {}

        features_array = np.array(all_features)  # Shape: (n_samples, 50)

        stats = {}
        for i, name in enumerate(self.feature_names):
            if i >= FEATURE_COUNT:
                break

            col = features_array[:, i]

            stats[name] = {
                "mean": float(np.nanmean(col)),
                "std": float(np.nanstd(col)),
                "min": float(np.nanmin(col)),
                "max": float(np.nanmax(col)),
                "nan_ratio": float(np.sum(np.isnan(col)) / len(col)),
                "inf_ratio": float(np.sum(np.isinf(col)) / len(col)),
                "zero_ratio": float(np.sum(col == 0) / len(col)),
            }

        return stats
