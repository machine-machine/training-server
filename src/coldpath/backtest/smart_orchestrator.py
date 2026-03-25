"""
Smart Backtest Orchestrator - Unified intelligent backtesting system.

Integrates all AI-powered backtesting components:
- AI Guided Setup for optimal initial parameters
- Market Regime Detection for dynamic adjustments
- Auto Mode with Optuna integration for optimization
- Feedback Loop for continuous learning
- Resource Guard for system protection
- Robustness Validator for profit quality

This is the main entry point for intelligent backtesting that eliminates
trial-and-error and adapts to market conditions while ensuring robustness.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Strategy for optimization."""

    FAST = "fast"  # Quick exploration, fewer iterations
    BALANCED = "balanced"  # Balance between speed and thoroughness
    THOROUGH = "thorough"  # Comprehensive search
    ADAPTIVE = "adaptive"  # Adjust based on progress


class LearningMode(Enum):
    """Learning mode for feedback integration."""

    DISABLED = "disabled"
    PASSIVE = "passive"  # Learn from results but don't auto-apply
    ACTIVE = "active"  # Learn and suggest improvements
    AGGRESSIVE = "aggressive"  # Learn and auto-apply best parameters


@dataclass
class SmartOrchestratorConfig:
    """Configuration for the smart orchestrator."""

    # Optimization settings
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    max_optimization_iterations: int = 50
    max_optimization_time_minutes: int = 15

    # Regime detection
    enable_regime_detection: bool = True
    regime_check_interval_seconds: int = 300  # 5 minutes

    # Learning
    learning_mode: LearningMode = LearningMode.ACTIVE
    min_samples_for_learning: int = 50

    # Auto-tuning
    enable_auto_tuning: bool = True
    auto_tune_threshold: float = 0.1  # Improvement threshold to trigger retuning

    # Safety
    max_parameter_change_pct: float = 30.0  # Max single parameter change
    require_validation: bool = True

    # Robustness (CRITICAL for profit quality)
    enable_robustness_validation: bool = True
    robustness_threshold: float = 0.6  # Minimum robustness score (0-1)
    robustness_num_trials: int = 50  # Monte Carlo trials
    robustness_perturbation_pct: float = 10.0  # Parameter perturbation %

    # Resource management (CRITICAL for system stability)
    enable_resource_guard: bool = True
    max_concurrent_optimizations: int = 1
    min_free_memory_mb: int = 500
    max_cpu_percent: float = 70.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimization_strategy": self.optimization_strategy.value,
            "max_optimization_iterations": self.max_optimization_iterations,
            "max_optimization_time_minutes": self.max_optimization_time_minutes,
            "enable_regime_detection": self.enable_regime_detection,
            "learning_mode": self.learning_mode.value,
            "enable_auto_tuning": self.enable_auto_tuning,
            "enable_robustness_validation": self.enable_robustness_validation,
            "robustness_threshold": self.robustness_threshold,
            "enable_resource_guard": self.enable_resource_guard,
        }


@dataclass
class SmartOptimizationResult:
    """Result from smart optimization."""

    # Final parameters
    best_params: dict[str, Any]
    best_metrics: dict[str, float]
    best_score: float

    # Regime info
    detected_regime: str | None = None
    regime_adjustments: list[dict[str, Any]] = field(default_factory=list)

    # Optimization info
    optimization_iterations: int = 0
    optimization_time_seconds: float = 0.0
    improvement_over_baseline: float = 0.0

    # Learning info
    learned_from_history: bool = False
    historical_samples_used: int = 0

    # Robustness info (CRITICAL for profit quality)
    robustness_score: float | None = None
    is_robust: bool | None = None
    robustness_trials: int = 0
    most_sensitive_param: str | None = None
    parameter_sensitivities: list[dict[str, Any]] = field(default_factory=list)

    # Resource info
    resources_checked: bool = False
    resource_guard_passed: bool | None = None

    # Profit quality score (combines optimization + robustness)
    profit_quality_score: float | None = None

    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_metrics": self.best_metrics,
            "best_score": self.best_score,
            "detected_regime": self.detected_regime,
            "optimization_iterations": self.optimization_iterations,
            "optimization_time_seconds": self.optimization_time_seconds,
            "improvement_over_baseline": self.improvement_over_baseline,
            "robustness_score": self.robustness_score,
            "is_robust": self.is_robust,
            "profit_quality_score": self.profit_quality_score,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
        }


class SmartBacktestOrchestrator:
    """Unified intelligent backtesting orchestrator.

    Combines all AI-powered components for optimal backtesting:

    1. **Guided Setup**: Get optimal initial parameters based on risk profile
    2. **Regime Detection**: Adjust parameters for current market conditions
    3. **Smart Optimization**: Auto-tune using Bayesian (Optuna) or evolutionary search
    4. **Continuous Learning**: Improve from historical results
    5. **Resource Guard**: Protect system from resource exhaustion
    6. **Robustness Validation**: Ensure parameters are not overfit

    Usage:
        orchestrator = SmartBacktestOrchestrator(
            backtest_runner=my_backtest_fn,
            config=SmartOrchestratorConfig(),
        )

        # Quick setup
        result = await orchestrator.quick_optimize(
            risk_tolerance="moderate",
            capital_sol=1.0,
        )

        # Full optimization with regime detection
        result = await orchestrator.smart_optimize(
            risk_tolerance="moderate",
            primary_goal="balanced",
            market_metrics=current_metrics,
        )
    """

    def __init__(
        self,
        backtest_runner: Callable,
        config: SmartOrchestratorConfig | None = None,
        feedback_store: Any | None = None,
        historical_results: list[dict[str, Any]] | None = None,
        resource_guard: Any | None = None,
        robustness_validator: Any | None = None,
    ):
        """Initialize the smart orchestrator.

        Args:
            backtest_runner: Async function to run backtests
            config: Orchestrator configuration
            feedback_store: Optional store for feedback loop
            historical_results: Optional historical backtest results
            resource_guard: Optional ResourceGuard for system protection
            robustness_validator: Optional RobustnessValidator for profit quality
        """
        self.backtest_runner = backtest_runner
        self.config = config or SmartOrchestratorConfig()
        self.feedback_store = feedback_store
        self.historical_results = historical_results or []

        # Resource management and robustness
        self._resource_guard = resource_guard
        self._robustness_validator = robustness_validator

        # Initialize sub-components
        self._guide = None
        self._regime_detector = None
        self._auto_mode = None

        # State tracking
        self._last_regime_check: datetime | None = None
        self._current_regime: str | None = None
        self._optimization_count = 0

    def _get_resource_guard(self):
        """Lazy load resource guard."""
        if self._resource_guard is None and self.config.enable_resource_guard:
            try:
                from coldpath.ai.resource_guard import ResourceGuard, ResourceLimits

                limits = ResourceLimits(
                    max_concurrent_optimizations=self.config.max_concurrent_optimizations,
                    min_free_memory_mb=self.config.min_free_memory_mb,
                    max_cpu_percent=self.config.max_cpu_percent,
                )
                self._resource_guard = ResourceGuard(limits=limits)
            except ImportError:
                logger.warning("ResourceGuard not available (psutil may not be installed)")
        return self._resource_guard

    def _get_robustness_validator(self):
        """Lazy load robustness validator."""
        if self._robustness_validator is None and self.config.enable_robustness_validation:
            from coldpath.backtest.robustness_validator import RobustnessValidator

            self._robustness_validator = RobustnessValidator(
                robustness_threshold=self.config.robustness_threshold,
            )
        return self._robustness_validator

    def _get_guide(self):
        """Lazy load AI guide."""
        if self._guide is None:
            from coldpath.backtest.ai_backtest_guide import AIBacktestGuide

            self._guide = AIBacktestGuide(
                historical_performance=self._compile_historical_performance(),
            )
        return self._guide

    def _get_regime_detector(self):
        """Lazy load regime detector."""
        if self._regime_detector is None:
            from coldpath.backtest.market_regime import MarketRegimeDetector

            self._regime_detector = MarketRegimeDetector()
        return self._regime_detector

    async def quick_optimize(
        self,
        risk_tolerance: str = "moderate",
        capital_sol: float = 1.0,
        preset: str | None = None,
    ) -> SmartOptimizationResult:
        """Quick optimization with preset parameters.

        Fastest way to get optimized parameters:
        1. Get preset from AI guide
        2. Optionally run brief optimization
        3. Return optimized parameters

        Args:
            risk_tolerance: Risk level (conservative, moderate, aggressive, degen)
            capital_sol: Available capital
            preset: Optional preset name (overrides risk_tolerance)

        Returns:
            SmartOptimizationResult with optimized parameters
        """
        start_time = datetime.utcnow()
        guide = self._get_guide()

        # Get preset parameters
        if preset:
            base_params = guide.quick_setup(preset=preset, capital_sol=capital_sol)
        else:
            result = guide.create_guided_setup(
                risk_tolerance=risk_tolerance,
                primary_goal="balanced",
                capital_sol=capital_sol,
            )
            base_params = result.suggested_config

        # Run baseline backtest
        baseline_metrics = await self._run_backtest(base_params)
        baseline_score = self._calculate_score(baseline_metrics)

        # Quick optimization (fewer iterations)
        if self.config.optimization_strategy != OptimizationStrategy.FAST:
            optimized_params, optimized_metrics, iterations = await self._run_optimization(
                base_params,
                max_iterations=10,
            )
            optimized_score = self._calculate_score(optimized_metrics)
        else:
            optimized_params = base_params
            optimized_metrics = baseline_metrics
            optimized_score = baseline_score
            iterations = 0

        # Calculate improvement
        improvement = ((optimized_score - baseline_score) / max(baseline_score, 0.01)) * 100

        return SmartOptimizationResult(
            best_params=optimized_params,
            best_metrics=optimized_metrics,
            best_score=optimized_score,
            optimization_iterations=iterations,
            optimization_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
            improvement_over_baseline=improvement,
            recommendations=["Parameters optimized for " + (preset or risk_tolerance) + " profile"],
        )

    async def smart_optimize(
        self,
        risk_tolerance: str = "moderate",
        primary_goal: str = "balanced",
        capital_sol: float = 1.0,
        market_metrics: Any | None = None,
        initial_params: dict[str, Any] | None = None,
    ) -> SmartOptimizationResult:
        """Full smart optimization with all AI components.

        Comprehensive optimization process:
        1. Get AI-guided initial parameters
        2. Detect market regime and adjust
        3. Run Bayesian optimization (Optuna)
        4. Learn from historical results
        5. Validate and return

        Args:
            risk_tolerance: Risk level
            primary_goal: Optimization goal
            capital_sol: Available capital
            market_metrics: Current market data for regime detection
            initial_params: Optional override for initial parameters

        Returns:
            SmartOptimizationResult with optimized parameters
        """
        start_time = datetime.utcnow()
        guide = self._get_guide()
        recommendations = []
        warnings = []
        regime_adjustments = []
        detected_regime = None

        # Robustness tracking

        # Resource tracking
        resources_checked = False

        # Step 0: Check system resources (CRITICAL for stability)
        resource_guard = self._get_resource_guard()
        if resource_guard is not None:
            resources_checked = True
            can_start, reason = await resource_guard.can_start_optimization()

            if not can_start:
                warnings.append(f"Resource check failed: {reason}")
                logger.warning(f"Optimization blocked by resource guard: {reason}")
                # Return early with safe defaults
                return SmartOptimizationResult(
                    best_params=initial_params or {},
                    best_metrics={},
                    best_score=0.0,
                    detected_regime=None,
                    optimization_iterations=0,
                    optimization_time_seconds=0.0,
                    improvement_over_baseline=0.0,
                    resources_checked=resources_checked,
                    resource_guard_passed=False,
                    recommendations=["Wait for resources to become available"],
                    warnings=warnings,
                )

        # Step 1: Get guided setup
        if initial_params:
            base_params = initial_params.copy()
        else:
            setup_result = guide.create_guided_setup(
                risk_tolerance=risk_tolerance,
                primary_goal=primary_goal,
                capital_sol=capital_sol,
            )
            base_params = setup_result.suggested_config
            warnings.extend(setup_result.validation_warnings)

        # Step 2: Regime detection and adjustment
        if self.config.enable_regime_detection and market_metrics:
            detector = self._get_regime_detector()
            adjusted_params, analysis = detector.get_regime_adjusted_params(
                base_params, market_metrics
            )
            detected_regime = analysis.primary_regime.value
            regime_adjustments = [a.__dict__ for a in analysis.adjustments]
            base_params = adjusted_params
            recommendations.extend(analysis.recommended_actions[:3])
            self._current_regime = detected_regime
            self._last_regime_check = datetime.utcnow()

        # Step 3: Run baseline
        baseline_metrics = await self._run_backtest(base_params)
        baseline_score = self._calculate_score(baseline_metrics)

        # Step 4: Learn from history
        learned_params = None
        samples_used = 0
        if (
            self.config.learning_mode != LearningMode.DISABLED
            and len(self.historical_results) >= self.config.min_samples_for_learning
        ):
            learned_params, samples_used = self._learn_from_history(
                base_params,
                risk_tolerance,
            )
            if learned_params:
                recommendations.append(f"Learned from {samples_used} historical samples")

        # Use learned params if better
        if learned_params:
            learned_metrics = await self._run_backtest(learned_params)
            learned_score = self._calculate_score(learned_metrics)
            if learned_score > baseline_score:
                base_params = learned_params
                baseline_metrics = learned_metrics
                baseline_score = learned_score

        # Step 5: Run optimization
        max_iterations = self._get_max_iterations()
        optimized_params, optimized_metrics, iterations = await self._run_optimization(
            base_params,
            max_iterations=max_iterations,
        )
        optimized_score = self._calculate_score(optimized_metrics)

        # Step 6: Validate changes
        if self.config.require_validation:
            validation_warnings = self._validate_parameter_changes(base_params, optimized_params)
            warnings.extend(validation_warnings)

        # Calculate improvement
        improvement = ((optimized_score - baseline_score) / max(baseline_score, 0.01)) * 100

        # Store result for future learning
        self._store_result(optimized_params, optimized_metrics, risk_tolerance)

        self._optimization_count += 1

        return SmartOptimizationResult(
            best_params=optimized_params,
            best_metrics=optimized_metrics,
            best_score=optimized_score,
            detected_regime=detected_regime,
            regime_adjustments=regime_adjustments,
            optimization_iterations=iterations,
            optimization_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
            improvement_over_baseline=improvement,
            learned_from_history=samples_used > 0,
            historical_samples_used=samples_used,
            recommendations=recommendations,
            warnings=warnings,
        )

    async def regime_aware_backtest(
        self,
        params: dict[str, Any],
        market_metrics: Any,
    ) -> dict[str, Any]:
        """Run a backtest with automatic regime-based adjustments.

        Args:
            params: Base parameters
            market_metrics: Current market data

        Returns:
            Dict with adjusted params, regime info, and backtest results
        """
        detector = self._get_regime_detector()

        # Get regime-adjusted params
        adjusted_params, analysis = detector.get_regime_adjusted_params(params, market_metrics)

        # Run backtest
        metrics = await self._run_backtest(adjusted_params)

        return {
            "original_params": params,
            "adjusted_params": adjusted_params,
            "regime": analysis.primary_regime.value,
            "regime_adjustments": [a.__dict__ for a in analysis.adjustments],
            "metrics": metrics,
            "risk_level": analysis.risk_level,
            "recommended_actions": analysis.recommended_actions,
        }

    def analyze_my_params(
        self,
        current_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze current parameters and get improvement suggestions.

        Args:
            current_params: Current backtest parameters

        Returns:
            Analysis with issues, suggestions, and optimization potential
        """
        guide = self._get_guide()
        return guide.analyze_current_params(current_params)

    def get_parameter_info(self, param_name: str) -> dict[str, Any]:
        """Get detailed information about a parameter."""
        guide = self._get_guide()
        return guide.get_parameter_explanation(param_name)

    async def _run_backtest(self, params: dict[str, Any]) -> dict[str, float]:
        """Run a single backtest."""
        if asyncio.iscoroutinefunction(self.backtest_runner):
            return await self.backtest_runner(params)
        else:
            import inspect

            if inspect.iscoroutinefunction(self.backtest_runner):
                return await self.backtest_runner(params)
            return self.backtest_runner(params)

    async def _run_optimization(
        self,
        initial_params: dict[str, Any],
        max_iterations: int = 50,
    ) -> tuple:
        """Run optimization using Auto Mode with optional Optuna integration."""
        from coldpath.backtest.auto_backtest import (
            AutoBacktestMode,
            AutoModeConfig,
        )

        config = AutoModeConfig(
            max_iterations=max_iterations,
            max_time_minutes=self.config.max_optimization_time_minutes,
            enable_feedback_loop=self.config.learning_mode != LearningMode.DISABLED,
        )

        auto_mode = AutoBacktestMode(
            backtest_runner=self.backtest_runner,
            config=config,
            feedback_store=self.feedback_store,
        )

        result = await auto_mode.run(initial_params=initial_params)

        return result.best_params, result.best_metrics, result.total_iterations

    def _calculate_score(self, metrics: dict[str, float]) -> float:
        """Calculate optimization score from metrics."""
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate_pct", 0) / 100
        profit_factor = metrics.get("profit_factor", 0)
        max_dd = metrics.get("max_drawdown_pct", 100)
        pq_score = metrics.get("profit_quality_score", 0)

        # Normalized score
        sharpe_norm = min(1.0, sharpe / 3.0)
        win_rate_norm = win_rate
        pf_norm = min(1.0, profit_factor / 3.0)
        dd_norm = max(0, 1 - max_dd / 50)

        score = (
            sharpe_norm * 0.30
            + win_rate_norm * 0.15
            + pf_norm * 0.20
            + dd_norm * 0.20
            + pq_score * 0.15
        )

        return score

    def _get_max_iterations(self) -> int:
        """Get max iterations based on strategy."""
        strategy_iterations = {
            OptimizationStrategy.FAST: 15,
            OptimizationStrategy.BALANCED: 50,
            OptimizationStrategy.THOROUGH: 150,
            OptimizationStrategy.ADAPTIVE: 30,  # Starts lower, can extend
        }
        return min(
            strategy_iterations.get(self.config.optimization_strategy, 50),
            self.config.max_optimization_iterations,
        )

    def _learn_from_history(
        self,
        current_params: dict[str, Any],
        risk_tolerance: str,
    ) -> tuple:
        """Learn optimal parameters from historical results."""
        if len(self.historical_results) < self.config.min_samples_for_learning:
            return None, 0

        # Filter by risk tolerance
        relevant = [r for r in self.historical_results if r.get("risk_tolerance") == risk_tolerance]

        if len(relevant) < 20:
            relevant = self.historical_results  # Fall back to all

        # Find best historical result
        scored = []
        for result in relevant:
            score = self._calculate_score(result.get("metrics", {}))
            scored.append((score, result))

        scored.sort(reverse=True)

        if scored:
            best_score, best_result = scored[0]
            if best_score > 0.5:  # Only use if decent
                return best_result.get("params", current_params), len(relevant)

        return None, len(relevant)

    def _validate_parameter_changes(
        self,
        original: dict[str, Any],
        optimized: dict[str, Any],
    ) -> list[str]:
        """Validate that parameter changes are reasonable."""
        warnings = []

        for key in optimized:
            if key not in original:
                continue

            orig_val = original[key]
            opt_val = optimized[key]

            if isinstance(orig_val, (int, float)) and orig_val != 0:
                change_pct = abs((opt_val - orig_val) / orig_val) * 100

                if change_pct > self.config.max_parameter_change_pct:
                    warnings.append(
                        f"Large change in {key}: {orig_val:.4f} -> "
                        f"{opt_val:.4f} ({change_pct:.1f}%)"
                    )

        return warnings

    def _compile_historical_performance(self) -> dict[str, Any]:
        """Compile historical performance for the AI guide."""
        if not self.historical_results:
            return {}

        # Aggregate by risk tolerance
        by_risk = {}
        for result in self.historical_results:
            risk = result.get("risk_tolerance", "unknown")
            if risk not in by_risk:
                by_risk[risk] = []
            by_risk[risk].append(result)

        # Calculate averages
        compiled = {}
        for risk, results in by_risk.items():
            if not results:
                continue

            avg_sharpe = sum(r.get("metrics", {}).get("sharpe_ratio", 0) for r in results) / len(
                results
            )

            avg_return = sum(
                r.get("metrics", {}).get("total_return_pct", 0) for r in results
            ) / len(results)

            compiled[risk] = {
                "samples": len(results),
                "avg_sharpe": avg_sharpe,
                "avg_return": avg_return,
            }

        return compiled

    def _store_result(
        self,
        params: dict[str, Any],
        metrics: dict[str, float],
        risk_tolerance: str,
    ):
        """Store result for future learning."""
        result = {
            "params": params,
            "metrics": metrics,
            "risk_tolerance": risk_tolerance,
            "score": self._calculate_score(metrics),
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.historical_results.append(result)

        # Limit history size
        if len(self.historical_results) > 1000:
            self.historical_results = self.historical_results[-500:]

        # Store in feedback store if available
        if self.feedback_store and hasattr(self.feedback_store, "append"):
            self.feedback_store.append(result)

    def add_historical_results(self, results: list[dict[str, Any]]):
        """Add historical results for learning."""
        self.historical_results.extend(results)

        # Reset guide to pick up new history
        self._guide = None

    def get_statistics(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "optimization_count": self._optimization_count,
            "historical_results": len(self.historical_results),
            "current_regime": self._current_regime,
            "last_regime_check": (
                self._last_regime_check.isoformat() if self._last_regime_check else None
            ),
            "config": self.config.to_dict(),
        }


# Convenience factory function
def create_smart_orchestrator(
    backtest_runner: Callable,
    risk_tolerance: str = "moderate",
    enable_regime_detection: bool = True,
    enable_learning: bool = True,
) -> SmartBacktestOrchestrator:
    """Create a pre-configured smart orchestrator.

    Args:
        backtest_runner: Function to run backtests
        risk_tolerance: Default risk tolerance
        enable_regime_detection: Enable market regime detection
        enable_learning: Enable learning from history

    Returns:
        Configured SmartBacktestOrchestrator
    """
    strategy = {
        "conservative": OptimizationStrategy.THOROUGH,
        "moderate": OptimizationStrategy.BALANCED,
        "aggressive": OptimizationStrategy.FAST,
        "degen": OptimizationStrategy.ADAPTIVE,
    }.get(risk_tolerance, OptimizationStrategy.BALANCED)

    learning_mode = LearningMode.ACTIVE if enable_learning else LearningMode.DISABLED

    config = SmartOrchestratorConfig(
        optimization_strategy=strategy,
        enable_regime_detection=enable_regime_detection,
        learning_mode=learning_mode,
    )

    return SmartBacktestOrchestrator(
        backtest_runner=backtest_runner,
        config=config,
    )
