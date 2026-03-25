"""
Multi-method statistical drift detection for trading performance.

Detects when trading performance has significantly degraded compared
to a baseline period using multiple statistical tests and practical
metric comparisons.

Methods:
1. Kolmogorov-Smirnov test (distribution shift in P&L)
2. T-test (mean shift in P&L)
3. Practical degradation (Sharpe ratio, win rate thresholds)

Severity classification:
- severe:   KS p<0.01 AND (Sharpe drop >50% OR win rate drop >15%)
- moderate: KS p<0.05 AND (Sharpe drop >30% OR win rate drop >10%)
- minor:    Sharpe drop >15% OR win rate drop >5%
- none:     No significant degradation

Usage:
    detector = DriftDetector()
    report = await detector.detect_drift(outcomes=all_outcomes)
    if report.severity == "severe":
        trigger_optimization()
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Statistical drift detection results."""

    severity: str  # "none", "minor", "moderate", "severe"

    # Statistical tests
    ks_statistic: float
    ks_pvalue: float
    t_statistic: float
    t_pvalue: float

    # Practical metrics
    sharpe_degradation: float  # Percentage drop (e.g., 0.5 = 50% drop)
    win_rate_degradation: float  # Absolute drop (e.g., 0.15 = 15pp drop)
    baseline_sharpe: float
    recent_sharpe: float
    baseline_wr: float
    recent_wr: float

    # Sample sizes
    baseline_n: int
    recent_n: int

    description: str

    # Timestamp
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "severity": self.severity,
            "ks_statistic": round(self.ks_statistic, 6),
            "ks_pvalue": round(self.ks_pvalue, 6),
            "t_statistic": round(self.t_statistic, 6),
            "t_pvalue": round(self.t_pvalue, 6),
            "sharpe_degradation": round(self.sharpe_degradation, 4),
            "win_rate_degradation": round(self.win_rate_degradation, 4),
            "baseline_sharpe": round(self.baseline_sharpe, 4),
            "recent_sharpe": round(self.recent_sharpe, 4),
            "baseline_wr": round(self.baseline_wr, 4),
            "recent_wr": round(self.recent_wr, 4),
            "baseline_n": self.baseline_n,
            "recent_n": self.recent_n,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
        }


def _compute_sharpe(pnl_values: list[float]) -> float:
    """Compute Sharpe ratio from P&L values."""
    if len(pnl_values) < 2:
        return 0.0
    mean = sum(pnl_values) / len(pnl_values)
    variance = sum((p - mean) ** 2 for p in pnl_values) / (len(pnl_values) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.001
    return mean / std


def _compute_win_rate(pnl_values: list[float]) -> float:
    """Compute win rate from P&L values."""
    if not pnl_values:
        return 0.0
    wins = sum(1 for p in pnl_values if p > 0)
    return wins / len(pnl_values)


def _ks_test(
    sample1: list[float],
    sample2: list[float],
) -> tuple[float, float]:
    """
    Kolmogorov-Smirnov two-sample test.

    Tests whether two samples come from the same distribution.
    Pure Python implementation (no scipy dependency required at runtime).

    Returns:
        (ks_statistic, p_value_approximation)
    """
    n1 = len(sample1)
    n2 = len(sample2)

    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Combine and sort
    combined = [(v, 0) for v in sample1] + [(v, 1) for v in sample2]
    combined.sort(key=lambda x: x[0])

    # Compute the empirical CDF step differences
    d_max = 0.0
    cdf1 = 0.0
    cdf2 = 0.0

    for _value, group in combined:
        if group == 0:
            cdf1 += 1.0 / n1
        else:
            cdf2 += 1.0 / n2
        d = abs(cdf1 - cdf2)
        if d > d_max:
            d_max = d

    # Approximate p-value using the asymptotic formula
    n_eff = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (n_eff + 0.12 + 0.11 / n_eff) * d_max

    # Kolmogorov distribution approximation
    if lam <= 0:
        p_value = 1.0
    elif lam >= 3.0:
        p_value = 0.0
    else:
        # Series approximation
        p_value = 0.0
        for k in range(1, 100):
            sign = (-1) ** (k + 1)
            p_value += sign * math.exp(-2.0 * k * k * lam * lam)
        p_value = max(0.0, min(1.0, 2.0 * p_value))

    return d_max, p_value


def _t_test(
    sample1: list[float],
    sample2: list[float],
) -> tuple[float, float]:
    """
    Welch's t-test for independent samples.

    Tests whether two samples have the same mean.
    Pure Python implementation.

    Returns:
        (t_statistic, p_value_approximation)
    """
    n1 = len(sample1)
    n2 = len(sample2)

    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    mean1 = sum(sample1) / n1
    mean2 = sum(sample2) / n2

    var1 = sum((x - mean1) ** 2 for x in sample1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in sample2) / (n2 - 1)

    se1 = var1 / n1
    se2 = var2 / n2
    se_combined = math.sqrt(se1 + se2) if (se1 + se2) > 0 else 0.001

    t_stat = (mean1 - mean2) / se_combined

    # Welch-Satterthwaite degrees of freedom
    if (se1 + se2) > 0:
        df = ((se1 + se2) ** 2) / (
            (se1**2 / (n1 - 1) if se1 > 0 else 0.0)
            + (se2**2 / (n2 - 1) if se2 > 0 else 0.0)
            + 1e-10  # avoid division by zero
        )
    else:
        df = n1 + n2 - 2

    # Approximate p-value using the normal distribution for large df
    # For small df we use a rough Student-t CDF approximation
    abs_t = abs(t_stat)

    if df >= 30:
        # Normal approximation for large df
        # Using the complementary error function approximation
        p_value = _normal_sf(abs_t) * 2.0
    else:
        # Rough approximation for small df
        # Uses the relationship between t and F distributions
        x = df / (df + abs_t**2)
        p_value = _incomplete_beta(df / 2.0, 0.5, x)

    return t_stat, max(0.0, min(1.0, p_value))


def _normal_sf(x: float) -> float:
    """Survival function (1 - CDF) for standard normal. Approximation."""
    # Abramowitz and Stegun approximation
    if x < 0:
        return 1.0 - _normal_sf(-x)

    t = 1.0 / (1.0 + 0.2316419 * x)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * math.exp(-0.5 * x * x)
    poly = t * (
        0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + 1.330274429 * t)))
    )
    return p * poly


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Very rough approximation of regularized incomplete beta function."""
    # For our use case (df/2, 0.5, x) in t-test p-value
    # Use the normal approximation as a fallback
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Simple numerical integration (trapezoidal rule)
    n_steps = 100
    dx = x / n_steps
    total = 0.0
    for i in range(n_steps):
        t = (i + 0.5) * dx
        if 0 < t < 1:
            try:
                val = t ** (a - 1) * (1 - t) ** (b - 1)
                total += val * dx
            except (OverflowError, ValueError):
                pass

    # Normalize (approximate Beta(a, b))
    try:
        beta_func = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
        if beta_func > 0:
            return min(1.0, total / beta_func)
    except (OverflowError, ValueError):
        pass

    return 0.5  # Fallback


class DriftDetector:
    """
    Multi-method statistical drift detection.

    Compares recent trading performance to a baseline period using
    both statistical tests and practical metric thresholds.

    Configuration:
        min_baseline_trades: Minimum trades for reliable baseline (default 50)
        min_recent_trades: Minimum recent trades for comparison (default 10)
        baseline_window_days: How far back the baseline goes (default 30)
        recent_window_hours: How recent is "recent" (default 24)
    """

    # Severity thresholds
    SEVERE_THRESHOLDS = {
        "ks_pvalue": 0.01,
        "sharpe_drop_pct": 0.50,  # 50% drop
        "win_rate_drop_abs": 0.15,  # 15 percentage points
    }

    MODERATE_THRESHOLDS = {
        "ks_pvalue": 0.05,
        "sharpe_drop_pct": 0.30,  # 30% drop
        "win_rate_drop_abs": 0.10,  # 10 percentage points
    }

    MINOR_THRESHOLDS = {
        "sharpe_drop_pct": 0.15,  # 15% drop
        "win_rate_drop_abs": 0.05,  # 5 percentage points
    }

    def __init__(
        self,
        min_baseline_trades: int = 50,
        min_recent_trades: int = 10,
        baseline_window_days: int = 30,
        recent_window_hours: int = 24,
        task_router: Any = None,
        cost_tracker: Any = None,
    ) -> None:
        self.min_baseline_trades = min_baseline_trades
        self.min_recent_trades = min_recent_trades
        self.baseline_window_days = baseline_window_days
        self.recent_window_hours = recent_window_hours
        self.task_router = task_router
        self.cost_tracker = cost_tracker

        self._last_report: DriftReport | None = None

    async def detect_drift(
        self,
        outcomes: list[Any] | None = None,
        baseline_pnls: list[float] | None = None,
        recent_pnls: list[float] | None = None,
        alpha: float = 0.05,
    ) -> DriftReport:
        """
        Compare recent performance to baseline.

        Can accept either raw TradeOutcome objects (from which baseline/recent
        splits are computed), or pre-computed baseline and recent P&L arrays.

        Args:
            outcomes: List of TradeOutcome objects (auto-splits by time).
            baseline_pnls: Pre-computed baseline P&L values.
            recent_pnls: Pre-computed recent P&L values.
            alpha: Significance level for statistical tests.

        Returns:
            DriftReport with severity classification.
        """
        # Split into baseline and recent if outcomes provided
        if outcomes and (baseline_pnls is None or recent_pnls is None):
            baseline_pnls, recent_pnls = self._split_outcomes(outcomes)

        baseline_pnls = baseline_pnls or []
        recent_pnls = recent_pnls or []

        # Check minimum sample sizes
        if len(baseline_pnls) < self.min_baseline_trades:
            report = DriftReport(
                severity="none",
                ks_statistic=0.0,
                ks_pvalue=1.0,
                t_statistic=0.0,
                t_pvalue=1.0,
                sharpe_degradation=0.0,
                win_rate_degradation=0.0,
                baseline_sharpe=0.0,
                recent_sharpe=0.0,
                baseline_wr=0.0,
                recent_wr=0.0,
                baseline_n=len(baseline_pnls),
                recent_n=len(recent_pnls),
                description=(
                    f"Insufficient baseline data: "
                    f"{len(baseline_pnls)}/{self.min_baseline_trades} trades"
                ),
            )
            self._last_report = report
            return report

        if len(recent_pnls) < self.min_recent_trades:
            report = DriftReport(
                severity="none",
                ks_statistic=0.0,
                ks_pvalue=1.0,
                t_statistic=0.0,
                t_pvalue=1.0,
                sharpe_degradation=0.0,
                win_rate_degradation=0.0,
                baseline_sharpe=0.0,
                recent_sharpe=0.0,
                baseline_wr=0.0,
                recent_wr=0.0,
                baseline_n=len(baseline_pnls),
                recent_n=len(recent_pnls),
                description=(
                    f"Insufficient recent data: {len(recent_pnls)}/{self.min_recent_trades} trades"
                ),
            )
            self._last_report = report
            return report

        # Run statistical tests
        ks_stat, ks_pvalue = _ks_test(baseline_pnls, recent_pnls)
        t_stat, t_pvalue = _t_test(baseline_pnls, recent_pnls)

        # Compute practical metrics
        baseline_sharpe = _compute_sharpe(baseline_pnls)
        recent_sharpe = _compute_sharpe(recent_pnls)
        baseline_wr = _compute_win_rate(baseline_pnls)
        recent_wr = _compute_win_rate(recent_pnls)

        # Compute degradation
        if abs(baseline_sharpe) > 0.001:
            sharpe_degradation = (baseline_sharpe - recent_sharpe) / abs(baseline_sharpe)
        else:
            sharpe_degradation = 0.0

        win_rate_degradation = baseline_wr - recent_wr

        # Classify severity
        severity = self._classify_severity(
            ks_pvalue=ks_pvalue,
            sharpe_degradation=sharpe_degradation,
            win_rate_degradation=win_rate_degradation,
        )

        # Build description
        description = self._build_description(
            severity=severity,
            ks_pvalue=ks_pvalue,
            t_pvalue=t_pvalue,
            sharpe_degradation=sharpe_degradation,
            win_rate_degradation=win_rate_degradation,
            baseline_sharpe=baseline_sharpe,
            recent_sharpe=recent_sharpe,
            baseline_wr=baseline_wr,
            recent_wr=recent_wr,
        )

        report = DriftReport(
            severity=severity,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            t_statistic=t_stat,
            t_pvalue=t_pvalue,
            sharpe_degradation=sharpe_degradation,
            win_rate_degradation=win_rate_degradation,
            baseline_sharpe=baseline_sharpe,
            recent_sharpe=recent_sharpe,
            baseline_wr=baseline_wr,
            recent_wr=recent_wr,
            baseline_n=len(baseline_pnls),
            recent_n=len(recent_pnls),
            description=description,
        )

        self._last_report = report

        if severity != "none":
            logger.warning(
                "Drift detected: severity=%s, KS p=%.4f, Sharpe %.2f->%.2f, WR %.1f%%->%.1f%%",
                severity,
                ks_pvalue,
                baseline_sharpe,
                recent_sharpe,
                baseline_wr * 100,
                recent_wr * 100,
            )

        return report

    def _split_outcomes(
        self,
        outcomes: list[Any],
    ) -> tuple[list[float], list[float]]:
        """
        Split outcomes into baseline and recent periods.

        Args:
            outcomes: List of objects with .timestamp and .pnl_pct attributes.

        Returns:
            (baseline_pnls, recent_pnls)
        """
        now = datetime.now()
        recent_cutoff = now - timedelta(hours=self.recent_window_hours)
        baseline_cutoff = now - timedelta(days=self.baseline_window_days)

        baseline_pnls: list[float] = []
        recent_pnls: list[float] = []

        for outcome in outcomes:
            ts = outcome.timestamp
            pnl = outcome.pnl_pct

            if ts >= recent_cutoff:
                recent_pnls.append(pnl)
            elif ts >= baseline_cutoff:
                baseline_pnls.append(pnl)

        return baseline_pnls, recent_pnls

    def _classify_severity(
        self,
        ks_pvalue: float,
        sharpe_degradation: float,
        win_rate_degradation: float,
    ) -> str:
        """
        Classify drift severity based on test results.

        Severity levels (from most to least severe):
        - severe: Statistical AND practical significance
        - moderate: Statistical AND moderate practical impact
        - minor: Practical significance only
        - none: No significant drift
        """
        # Check SEVERE: KS p<0.01 AND (Sharpe drop >50% OR WR drop >15pp)
        if ks_pvalue < self.SEVERE_THRESHOLDS["ks_pvalue"]:
            if (
                sharpe_degradation > self.SEVERE_THRESHOLDS["sharpe_drop_pct"]
                or win_rate_degradation > self.SEVERE_THRESHOLDS["win_rate_drop_abs"]
            ):
                return "severe"

        # Check MODERATE: KS p<0.05 AND (Sharpe drop >30% OR WR drop >10pp)
        if ks_pvalue < self.MODERATE_THRESHOLDS["ks_pvalue"]:
            if (
                sharpe_degradation > self.MODERATE_THRESHOLDS["sharpe_drop_pct"]
                or win_rate_degradation > self.MODERATE_THRESHOLDS["win_rate_drop_abs"]
            ):
                return "moderate"

        # Check MINOR: Sharpe drop >15% OR WR drop >5pp
        if (
            sharpe_degradation > self.MINOR_THRESHOLDS["sharpe_drop_pct"]
            or win_rate_degradation > self.MINOR_THRESHOLDS["win_rate_drop_abs"]
        ):
            return "minor"

        return "none"

    def _build_description(
        self,
        severity: str,
        ks_pvalue: float,
        t_pvalue: float,
        sharpe_degradation: float,
        win_rate_degradation: float,
        baseline_sharpe: float,
        recent_sharpe: float,
        baseline_wr: float,
        recent_wr: float,
    ) -> str:
        """Build human-readable drift description."""
        if severity == "none":
            return "No significant performance drift detected."

        parts = [f"{severity.upper()} drift detected:"]

        if sharpe_degradation > 0:
            parts.append(
                f"Sharpe ratio dropped {sharpe_degradation:.0%} "
                f"({baseline_sharpe:.2f} -> {recent_sharpe:.2f})"
            )

        if win_rate_degradation > 0:
            parts.append(
                f"Win rate dropped {win_rate_degradation:.1%} "
                f"({baseline_wr:.1%} -> {recent_wr:.1%})"
            )

        parts.append(f"KS test p={ks_pvalue:.4f}, t-test p={t_pvalue:.4f}")

        return ". ".join(parts) + "."

    @property
    def last_report(self) -> DriftReport | None:
        """Most recent drift report."""
        return self._last_report

    def get_routing_decision(
        self,
        severity: str = "none",
    ) -> Any | None:
        """
        Get a model routing decision for drift analysis AI calls.

        Routes to Opus 4.6 for severe drift (requires deep analysis),
        Sonnet 4.5 for minor/no drift (routine checks).

        Args:
            severity: Drift severity level.

        Returns:
            ModelRoutingDecision if task_router is available, else None.
        """
        if self.task_router is None:
            return None

        is_high_stakes = severity in ("severe", "moderate")
        task_type = "risk_assessment" if is_high_stakes else "performance_summary"

        decision = self.task_router.route_task(
            task_type=task_type,
            num_parameters=0,
            has_constraints=False,
            requires_explanation=is_high_stakes,
            is_high_stakes=is_high_stakes,
        )

        return decision

    async def record_drift_cost(
        self,
        optimization_id: str,
        severity: str = "none",
        tokens_input: int = 400,
        tokens_output: int = 800,
        latency_ms: int = 150,
    ) -> None:
        """
        Record the cost of a drift detection AI call.

        Args:
            optimization_id: The optimization run ID.
            severity: Drift severity detected.
            tokens_input: Input tokens consumed.
            tokens_output: Output tokens consumed.
            latency_ms: Request latency.
        """
        if self.cost_tracker is None:
            return

        routing = self.get_routing_decision(severity)
        model = routing.model_id if routing else "claude-sonnet-4-5-20250929"

        await self.cost_tracker.record_api_call(
            optimization_id=optimization_id,
            model=model,
            task_type="drift_detection",
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            cache_hit=False,
        )
