"""
Training data quality validation.

Ensures train/test boundaries, data integrity, and ML-readiness.
Comprehensive validation for Solana trading data.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    code: str
    message: str
    severity: ValidationSeverity
    field: str | None = None
    count: int = 1
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "field": self.field,
            "count": self.count,
            "details": self.details,
        }


@dataclass
class DataQualityReport:
    """Comprehensive report on data quality."""

    total_records: int
    valid_records: int
    missing_fields: dict[str, int] = field(default_factory=dict)
    outliers: dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    time_boundary_violations: int = 0
    label_leakage_risk: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    # Quality metrics
    completeness_score: float = 0.0  # % of non-null values
    consistency_score: float = 0.0  # % of valid ranges
    timeliness_score: float = 0.0  # Data freshness
    uniqueness_score: float = 0.0  # % unique records

    @property
    def passed(self) -> bool:
        """Check if validation passed (no errors or critical issues)."""
        return not any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for i in self.issues
        )

    @property
    def overall_score(self) -> float:
        """Calculate overall data quality score (0-1)."""
        weights = {
            "completeness": 0.3,
            "consistency": 0.3,
            "timeliness": 0.2,
            "uniqueness": 0.2,
        }
        return (
            weights["completeness"] * self.completeness_score
            + weights["consistency"] * self.consistency_score
            + weights["timeliness"] * self.timeliness_score
            + weights["uniqueness"] * self.uniqueness_score
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "passed": self.passed,
            "overall_score": self.overall_score,
            "completeness_score": self.completeness_score,
            "consistency_score": self.consistency_score,
            "timeliness_score": self.timeliness_score,
            "uniqueness_score": self.uniqueness_score,
            "missing_fields": self.missing_fields,
            "outliers": self.outliers,
            "duplicates": self.duplicates,
            "time_boundary_violations": self.time_boundary_violations,
            "label_leakage_risk": self.label_leakage_risk,
            "issues": [i.to_dict() for i in self.issues],
        }

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        error_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)
        return (
            f"Data Quality: {status} | "
            f"Score: {self.overall_score:.1%} | "
            f"Records: {self.valid_records}/{self.total_records} valid | "
            f"Errors: {error_count}, Warnings: {warning_count}"
        )


class DataQualityValidator:
    """Validates training data quality for ML pipelines.

    Performs comprehensive validation including:
    - Schema validation (required fields, types)
    - Range validation (reasonable bounds)
    - Temporal validation (ordering, gaps)
    - Statistical validation (outliers, distributions)
    - Leakage detection (label bleeding)
    """

    # Required fields for different data types
    TRADE_FIELDS = {
        "timestamp_ms": {"type": "int", "min": 0},
        "mint": {"type": "str", "min_length": 32, "max_length": 64},
        "side": {"type": "str", "values": ["buy", "sell", "BUY", "SELL"]},
        "amount_sol": {"type": "float", "min": 0.0001, "max": 1000000},
        "price": {"type": "float", "min": 0, "max": 1e12},
    }

    OBSERVATION_FIELDS = {
        "timestamp_ms": {"type": "int", "min": 0},
        "quoted_slippage_bps": {"type": "int", "min": 0, "max": 10000},
        "included": {"type": "bool"},
        "mode": {"type": "str", "values": ["BACKTEST", "PAPER", "SHADOW", "LIVE"]},
    }

    # Fields that suggest label leakage
    LEAKAGE_PATTERNS = [
        "future",
        "next_",
        "will_",
        "outcome",
        "result",
        "target",
        "label",
        "y_",
        "profit",
        "pnl_future",
    ]

    def __init__(
        self,
        required_fields: list[str] | None = None,
        field_specs: dict[str, dict] | None = None,
        outlier_threshold: float = 3.0,
        max_time_gap_hours: float = 24.0,
    ):
        """Initialize validator.

        Args:
            required_fields: List of required field names.
            field_specs: Field specifications for validation.
            outlier_threshold: Z-score threshold for outlier detection.
            max_time_gap_hours: Maximum allowed gap in time series.
        """
        self.required_fields = required_fields or list(self.TRADE_FIELDS.keys())
        self.field_specs = field_specs or self.TRADE_FIELDS
        self.outlier_threshold = outlier_threshold
        self.max_time_gap_hours = max_time_gap_hours

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        data_type: str = "trade",
    ) -> DataQualityReport:
        """Validate a DataFrame for training.

        Args:
            df: DataFrame to validate.
            data_type: Type of data ('trade' or 'observation').

        Returns:
            DataQualityReport with validation results.
        """
        if df is None or df.empty:
            return DataQualityReport(
                total_records=0,
                valid_records=0,
                issues=[
                    ValidationIssue(
                        code="EMPTY_DATA",
                        message="DataFrame is empty or None",
                        severity=ValidationSeverity.ERROR,
                    )
                ],
            )

        # Select field specs based on data type
        if data_type == "observation":
            self.field_specs = self.OBSERVATION_FIELDS
            self.required_fields = list(self.OBSERVATION_FIELDS.keys())

        issues = []
        missing_fields = {}
        outliers = {}

        # 1. Schema validation
        schema_issues = self._validate_schema(df)
        issues.extend(schema_issues)
        missing_fields = {
            f: df[f].isnull().sum()
            for f in self.required_fields
            if f in df.columns and df[f].isnull().sum() > 0
        }

        # 2. Range validation
        range_issues, range_valid = self._validate_ranges(df)
        issues.extend(range_issues)

        # 3. Temporal validation
        temporal_issues, time_violations = self._validate_temporal(df)
        issues.extend(temporal_issues)

        # 4. Duplicate detection
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            dup_pct = duplicates / len(df) * 100
            severity = ValidationSeverity.WARNING if dup_pct < 5 else ValidationSeverity.ERROR
            issues.append(
                ValidationIssue(
                    code="DUPLICATES",
                    message=f"Found {duplicates} duplicate rows ({dup_pct:.1f}%)",
                    severity=severity,
                    count=duplicates,
                )
            )

        # 5. Outlier detection
        outlier_issues, outliers = self._detect_outliers(df)
        issues.extend(outlier_issues)

        # 6. Leakage detection
        leakage_risk = self._detect_leakage(df)
        if leakage_risk:
            issues.append(
                ValidationIssue(
                    code="POTENTIAL_LEAKAGE",
                    message=f"Potential label leakage in columns: {', '.join(leakage_risk)}",
                    severity=ValidationSeverity.WARNING,
                    details={"columns": leakage_risk},
                )
            )

        # Calculate quality scores
        completeness = self._calculate_completeness(df)
        consistency = range_valid
        timeliness = self._calculate_timeliness(df)
        uniqueness = 1 - (duplicates / len(df)) if len(df) > 0 else 0

        # Count valid records
        valid_records = len(df) - duplicates - sum(missing_fields.values())

        return DataQualityReport(
            total_records=len(df),
            valid_records=max(0, valid_records),
            missing_fields=missing_fields,
            outliers=outliers,
            duplicates=duplicates,
            time_boundary_violations=time_violations,
            label_leakage_risk=leakage_risk,
            issues=issues,
            completeness_score=completeness,
            consistency_score=consistency,
            timeliness_score=timeliness,
            uniqueness_score=uniqueness,
        )

    def _validate_schema(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Validate schema and required fields."""
        issues = []

        for field_name in self.required_fields:
            if field_name not in df.columns:
                issues.append(
                    ValidationIssue(
                        code="MISSING_FIELD",
                        message=f"Required field '{field_name}' is missing",
                        severity=ValidationSeverity.ERROR,
                        field=field_name,
                    )
                )
            else:
                null_count = df[field_name].isnull().sum()
                if null_count > 0:
                    null_pct = null_count / len(df) * 100
                    severity = (
                        ValidationSeverity.WARNING if null_pct < 10 else ValidationSeverity.ERROR
                    )
                    issues.append(
                        ValidationIssue(
                            code="NULL_VALUES",
                            message=(
                                f"Field '{field_name}' has {null_count} "
                                f"null values ({null_pct:.1f}%)"
                            ),
                            severity=severity,
                            field=field_name,
                            count=null_count,
                        )
                    )

        return issues

    def _validate_ranges(self, df: pd.DataFrame) -> tuple[list[ValidationIssue], float]:
        """Validate value ranges."""
        issues = []
        total_checks = 0
        passed_checks = 0

        for field_name, spec in self.field_specs.items():
            if field_name not in df.columns:
                continue

            total_checks += 1
            col = df[field_name]

            # Numeric range checks
            if spec.get("min") is not None:
                violations = (col < spec["min"]).sum()
                if violations > 0:
                    issues.append(
                        ValidationIssue(
                            code="BELOW_MIN",
                            message=(
                                f"Field '{field_name}' has {violations} values below "
                                f"minimum ({spec['min']})"
                            ),
                            severity=ValidationSeverity.WARNING,
                            field=field_name,
                            count=violations,
                        )
                    )
                else:
                    passed_checks += 0.5

            if spec.get("max") is not None:
                violations = (col > spec["max"]).sum()
                if violations > 0:
                    issues.append(
                        ValidationIssue(
                            code="ABOVE_MAX",
                            message=(
                                f"Field '{field_name}' has {violations} values above "
                                f"maximum ({spec['max']})"
                            ),
                            severity=ValidationSeverity.WARNING,
                            field=field_name,
                            count=violations,
                        )
                    )
                else:
                    passed_checks += 0.5

            # Categorical value checks
            if spec.get("values"):
                invalid = ~col.isin(spec["values"])
                if invalid.sum() > 0:
                    issues.append(
                        ValidationIssue(
                            code="INVALID_VALUE",
                            message=f"Field '{field_name}' has {invalid.sum()} invalid values",
                            severity=ValidationSeverity.WARNING,
                            field=field_name,
                            count=invalid.sum(),
                            details={"valid_values": spec["values"]},
                        )
                    )

        consistency = passed_checks / total_checks if total_checks > 0 else 1.0
        return issues, consistency

    def _validate_temporal(self, df: pd.DataFrame) -> tuple[list[ValidationIssue], int]:
        """Validate temporal ordering and gaps."""
        issues = []
        violations = 0

        if "timestamp_ms" not in df.columns:
            return issues, 0

        timestamps = df["timestamp_ms"].sort_values()

        # Check ordering
        if not timestamps.equals(df["timestamp_ms"]):
            out_of_order = (df["timestamp_ms"].diff() < 0).sum()
            violations += out_of_order
            issues.append(
                ValidationIssue(
                    code="TIME_ORDER_VIOLATION",
                    message=f"Found {out_of_order} time ordering violations",
                    severity=ValidationSeverity.WARNING,
                    count=out_of_order,
                )
            )

        # Check for large gaps
        if len(timestamps) > 1:
            gaps_ms = timestamps.diff().dropna()
            max_gap_ms = self.max_time_gap_hours * 3600 * 1000
            large_gaps = (gaps_ms > max_gap_ms).sum()
            if large_gaps > 0:
                issues.append(
                    ValidationIssue(
                        code="LARGE_TIME_GAP",
                        message=f"Found {large_gaps} gaps larger than {self.max_time_gap_hours}h",
                        severity=ValidationSeverity.INFO,
                        count=large_gaps,
                    )
                )

        return issues, violations

    def _detect_outliers(self, df: pd.DataFrame) -> tuple[list[ValidationIssue], dict[str, int]]:
        """Detect outliers using z-score method."""
        issues = []
        outliers = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col == "timestamp_ms":  # Skip timestamp
                continue

            col_data = df[col].dropna()
            if len(col_data) < 10:  # Need minimum data
                continue

            # Calculate z-scores
            mean = col_data.mean()
            std = col_data.std()
            if std == 0:
                continue

            z_scores = np.abs((col_data - mean) / std)
            outlier_count = (z_scores > self.outlier_threshold).sum()

            if outlier_count > 0:
                outliers[col] = int(outlier_count)
                outlier_pct = outlier_count / len(col_data) * 100
                severity = (
                    ValidationSeverity.INFO if outlier_pct < 1 else ValidationSeverity.WARNING
                )
                issues.append(
                    ValidationIssue(
                        code="OUTLIERS",
                        message=f"Field '{col}' has {outlier_count} outliers ({outlier_pct:.1f}%)",
                        severity=severity,
                        field=col,
                        count=outlier_count,
                    )
                )

        return issues, outliers

    def _detect_leakage(self, df: pd.DataFrame) -> list[str]:
        """Detect potential label leakage in column names."""
        leakage_risk = []
        for col in df.columns:
            col_lower = col.lower()
            for pattern in self.LEAKAGE_PATTERNS:
                if pattern in col_lower:
                    leakage_risk.append(col)
                    break
        return leakage_risk

    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score."""
        if df.empty:
            return 0.0
        total_values = df.size
        non_null = df.count().sum()
        return non_null / total_values

    def _calculate_timeliness(self, df: pd.DataFrame) -> float:
        """Calculate data timeliness score."""
        if "timestamp_ms" not in df.columns or df.empty:
            return 0.5  # Neutral if no timestamp

        latest_ts = df["timestamp_ms"].max()
        now_ms = int(datetime.now().timestamp() * 1000)

        # Age in hours
        age_hours = (now_ms - latest_ts) / (1000 * 3600)

        # Score decays with age
        if age_hours <= 1:
            return 1.0
        elif age_hours <= 6:
            return 0.9
        elif age_hours <= 24:
            return 0.7
        elif age_hours <= 72:
            return 0.5
        else:
            return max(0.1, 1.0 - (age_hours / 168))  # Decay over week

    def validate_train_test_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        timestamp_col: str = "timestamp_ms",
        id_col: str = "mint",
    ) -> tuple[bool, list[ValidationIssue]]:
        """Validate no temporal leakage between train and test sets.

        Args:
            train_df: Training DataFrame.
            test_df: Test DataFrame.
            timestamp_col: Column containing timestamps.
            id_col: Column containing entity IDs (e.g., mint address).

        Returns:
            Tuple of (passed, issues).
        """
        issues = []

        if train_df.empty or test_df.empty:
            issues.append(
                ValidationIssue(
                    code="EMPTY_SPLIT",
                    message="Train or test set is empty",
                    severity=ValidationSeverity.ERROR,
                )
            )
            return False, issues

        # Check temporal boundary
        if timestamp_col in train_df.columns and timestamp_col in test_df.columns:
            train_max = train_df[timestamp_col].max()
            test_min = test_df[timestamp_col].min()

            if train_max >= test_min:
                overlap_count = (train_df[timestamp_col] >= test_min).sum() + (
                    test_df[timestamp_col] <= train_max
                ).sum()
                issues.append(
                    ValidationIssue(
                        code="TEMPORAL_LEAKAGE",
                        message=f"Train max ({train_max}) >= test min ({test_min})",
                        severity=ValidationSeverity.CRITICAL,
                        count=overlap_count,
                        details={
                            "train_max": int(train_max),
                            "test_min": int(test_min),
                            "overlap_records": int(overlap_count),
                        },
                    )
                )

        # Check entity overlap
        if id_col in train_df.columns and id_col in test_df.columns:
            train_ids = set(train_df[id_col].unique())
            test_ids = set(test_df[id_col].unique())
            overlap = train_ids & test_ids
            overlap_pct = len(overlap) / len(test_ids) * 100 if test_ids else 0

            if overlap_pct > 80:
                issues.append(
                    ValidationIssue(
                        code="HIGH_ID_OVERLAP",
                        message=f"High entity overlap: {len(overlap)} IDs ({overlap_pct:.1f}%)",
                        severity=ValidationSeverity.WARNING,
                        details={"overlap_count": len(overlap), "overlap_pct": overlap_pct},
                    )
                )

        # Check size ratio
        train_size = len(train_df)
        test_size = len(test_df)
        if test_size > 0:
            ratio = train_size / test_size
            if ratio < 2 or ratio > 20:
                issues.append(
                    ValidationIssue(
                        code="UNUSUAL_SPLIT_RATIO",
                        message=f"Unusual train/test ratio: {ratio:.1f}:1",
                        severity=ValidationSeverity.INFO,
                        details={"train_size": train_size, "test_size": test_size, "ratio": ratio},
                    )
                )

        passed = not any(i.severity == ValidationSeverity.CRITICAL for i in issues)
        return passed, issues
