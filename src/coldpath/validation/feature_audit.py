"""
Feature leakage detection.

Ensures no future information bleeds into features.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class LeakageAuditResult:
    """Result of leakage audit."""

    features_checked: int
    leakage_detected: bool
    leaking_features: list[str]
    warnings: list[str]


class FeatureAuditor:
    """Audits features for potential leakage."""

    def __init__(self):
        # Known safe features (computed at decision time)
        self.safe_features = {
            "liquidity_usd",
            "fdv_usd",
            "holder_count",
            "top_holder_pct",
            "mint_authority_enabled",
            "freeze_authority_enabled",
            "age_seconds",
            "volume_1h",
            "price_change_1h",
        }

        # Features that need careful handling
        self.suspicious_patterns = [
            "future",
            "next",
            "will",
            "outcome",
            "result",
            "pnl",
            "profit",
            "loss",
            "return",
        ]

    def audit(self, df: pd.DataFrame, label_col: str = "label") -> LeakageAuditResult:
        """Audit dataframe for feature leakage."""
        leaking = []
        warnings = []

        for col in df.columns:
            if col == label_col:
                continue

            # Check for suspicious naming
            col_lower = col.lower()
            for pattern in self.suspicious_patterns:
                if pattern in col_lower:
                    leaking.append(f"{col} (suspicious name pattern: {pattern})")
                    break

            # Check correlation with label
            if label_col in df.columns and df[col].dtype in ["int64", "float64"]:
                corr = df[col].corr(df[label_col])
                if abs(corr) > 0.9:
                    leaking.append(f"{col} (high correlation with label: {corr:.3f})")
                elif abs(corr) > 0.7:
                    warnings.append(f"{col} has moderately high correlation: {corr:.3f}")

        return LeakageAuditResult(
            features_checked=len(df.columns) - 1,
            leakage_detected=len(leaking) > 0,
            leaking_features=leaking,
            warnings=warnings,
        )

    def suggest_safe_features(self, available: set[str]) -> set[str]:
        """Suggest which available features are safe to use."""
        return available & self.safe_features
