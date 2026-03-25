"""
AI Guided Backtest Setup Wizard.

Eliminates trial-and-error by using intelligent analysis to recommend
optimal starting parameters based on user goals, risk profile, and
historical market conditions.

Key Features:
- Goal-driven parameter recommendations
- Risk profile matching
- Market regime detection
- Parameter validation and sanity checks
- One-click optimal configuration generation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RiskTolerance(Enum):
    """User's risk tolerance level."""

    CONSERVATIVE = "conservative"  # Capital preservation first
    MODERATE = "moderate"  # Balanced growth and safety
    AGGRESSIVE = "aggressive"  # Maximize returns
    DEGEN = "degen"  # High-risk memecoin hunting


class OptimizationGoal(Enum):
    """What the user wants to optimize for."""

    MAX_SHARPE = "max_sharpe"  # Risk-adjusted returns
    MAX_RETURN = "max_return"  # Raw total return
    MIN_DRAWDOWN = "min_drawdown"  # Capital preservation
    MAX_WIN_RATE = "max_win_rate"  # Consistency
    PROFIT_QUALITY = "profit_quality"  # Expectancy + profit factor
    BALANCED = "balanced"  # Multi-objective optimization


class MarketRegime(Enum):
    """Current market conditions."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MEMECOON_SEASON = "memecoin_season"
    RISK_OFF = "risk_off"


@dataclass
class UserProfile:
    """User's trading profile and preferences."""

    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    primary_goal: OptimizationGoal = OptimizationGoal.BALANCED
    capital_sol: float = 1.0
    max_daily_trades: int = 20
    preferred_hold_time_minutes: int = 30
    memecoin_focus: bool = True
    auto_compound: bool = False
    emergency_stop_loss_pct: float = 50.0  # Kill switch threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_tolerance": self.risk_tolerance.value,
            "primary_goal": self.primary_goal.value,
            "capital_sol": self.capital_sol,
            "max_daily_trades": self.max_daily_trades,
            "preferred_hold_time_minutes": self.preferred_hold_time_minutes,
            "memecoin_focus": self.memecoin_focus,
            "auto_compound": self.auto_compound,
            "emergency_stop_loss_pct": self.emergency_stop_loss_pct,
        }


@dataclass
class ParameterRecommendation:
    """A single parameter recommendation with rationale."""

    key: str
    label: str
    value: Any
    min_value: Any
    max_value: Any
    unit: str
    rationale: str
    confidence: float = 0.8
    can_auto_tune: bool = True


@dataclass
class GuidedSetupResult:
    """Result from AI-guided setup process."""

    profile: UserProfile
    recommendations: list[ParameterRecommendation]
    suggested_config: dict[str, Any]
    validation_warnings: list[str]
    expected_metrics: dict[str, float]
    next_steps: list[str]
    setup_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "recommendations": [r.__dict__ for r in self.recommendations],
            "suggested_config": self.suggested_config,
            "validation_warnings": self.validation_warnings,
            "expected_metrics": self.expected_metrics,
            "next_steps": self.next_steps,
            "setup_timestamp": self.setup_timestamp,
        }


# Knowledge base of optimal parameters by risk/goal combination
PARAMETER_KNOWLEDGE_BASE = {
    # Conservative profiles prioritize capital preservation
    (RiskTolerance.CONSERVATIVE, OptimizationGoal.MIN_DRAWDOWN): {
        "stop_loss_pct": (5.0, 3.0, 8.0, "Tight stops limit losses quickly"),
        "take_profit_pct": (12.0, 8.0, 20.0, "Modest profit targets are achievable"),
        "max_position_sol": (0.02, 0.01, 0.05, "Small positions limit exposure"),
        "min_liquidity_usd": (50000, 20000, 100000, "High liquidity reduces slippage risk"),
        "max_risk_score": (0.25, 0.15, 0.35, "Strict risk filtering"),
        "slippage_bps": (200, 100, 400, "Lower slippage tolerance for better fills"),
        "max_hold_minutes": (15, 5, 30, "Quick exits reduce exposure time"),
        "kelly_safety_factor": (0.25, 0.1, 0.35, "Quarter-Kelly for maximum safety"),
    },
    # Conservative + Max Sharpe
    (RiskTolerance.CONSERVATIVE, OptimizationGoal.MAX_SHARPE): {
        "stop_loss_pct": (6.0, 4.0, 8.0, "Balanced stops for risk-adjusted returns"),
        "take_profit_pct": (15.0, 10.0, 25.0, "Target 2:1 reward/risk minimum"),
        "max_position_sol": (0.03, 0.01, 0.05, "Conservative position sizing"),
        "min_liquidity_usd": (30000, 15000, 50000, "Quality liquidity for better fills"),
        "max_risk_score": (0.30, 0.20, 0.40, "Moderate risk filtering"),
        "slippage_bps": (250, 150, 400, "Balanced slippage tolerance"),
        "max_hold_minutes": (20, 10, 40, "Moderate hold times"),
        "kelly_safety_factor": (0.30, 0.20, 0.40, "Third-Kelly balance"),
    },
    # Moderate risk - balanced approach
    (RiskTolerance.MODERATE, OptimizationGoal.BALANCED): {
        "stop_loss_pct": (8.0, 5.0, 12.0, "Standard stop loss for memecoins"),
        "take_profit_pct": (20.0, 12.0, 35.0, "Good profit potential with patience"),
        "max_position_sol": (0.05, 0.02, 0.10, "Balanced position sizing"),
        "min_liquidity_usd": (15000, 8000, 30000, "Adequate liquidity"),
        "max_risk_score": (0.40, 0.30, 0.50, "Moderate risk acceptance"),
        "slippage_bps": (300, 200, 500, "Realistic slippage for memecoins"),
        "max_hold_minutes": (30, 15, 60, "Flexible hold times"),
        "kelly_safety_factor": (0.40, 0.30, 0.50, "Half-Kelly standard"),
    },
    # Moderate + Profit Quality
    (RiskTolerance.MODERATE, OptimizationGoal.PROFIT_QUALITY): {
        "stop_loss_pct": (7.0, 5.0, 10.0, "Slightly tighter stops for quality"),
        "take_profit_pct": (18.0, 12.0, 30.0, "Quality profit targets"),
        "max_position_sol": (0.04, 0.02, 0.08, "Quality-focused sizing"),
        "min_liquidity_usd": (20000, 10000, 40000, "Better liquidity for quality"),
        "max_risk_score": (0.35, 0.25, 0.45, "Selective risk filtering"),
        "slippage_bps": (280, 180, 450, "Quality fill tolerance"),
        "max_hold_minutes": (25, 15, 45, "Quality over speed"),
        "kelly_safety_factor": (0.35, 0.25, 0.45, "Quality Kelly factor"),
    },
    # Aggressive profiles
    (RiskTolerance.AGGRESSIVE, OptimizationGoal.MAX_RETURN): {
        "stop_loss_pct": (12.0, 8.0, 18.0, "Wider stops allow more volatility room"),
        "take_profit_pct": (35.0, 20.0, 60.0, "Target significant gains"),
        "max_position_sol": (0.10, 0.05, 0.20, "Larger positions for bigger wins"),
        "min_liquidity_usd": (8000, 5000, 20000, "Lower liquidity for early entry"),
        "max_risk_score": (0.55, 0.40, 0.70, "Accept higher risk for returns"),
        "slippage_bps": (400, 300, 600, "Higher slippage tolerance"),
        "max_hold_minutes": (45, 20, 90, "Longer holds for moonshots"),
        "kelly_safety_factor": (0.50, 0.40, 0.65, "Half to two-third Kelly"),
    },
    # Aggressive + Max Sharpe
    (RiskTolerance.AGGRESSIVE, OptimizationGoal.MAX_SHARPE): {
        "stop_loss_pct": (10.0, 6.0, 15.0, "Balanced stops for Sharpe"),
        "take_profit_pct": (28.0, 15.0, 45.0, "Good risk/reward ratio"),
        "max_position_sol": (0.08, 0.04, 0.15, "Aggressive but calculated"),
        "min_liquidity_usd": (12000, 6000, 25000, "Sufficient liquidity"),
        "max_risk_score": (0.48, 0.35, 0.60, "Risk-adjusted selection"),
        "slippage_bps": (350, 250, 500, "Realistic for aggressive"),
        "max_hold_minutes": (35, 18, 70, "Flexible timing"),
        "kelly_safety_factor": (0.45, 0.35, 0.55, "Near half-Kelly"),
    },
    # Degen mode - maximum risk
    (RiskTolerance.DEGEN, OptimizationGoal.MAX_RETURN): {
        "stop_loss_pct": (18.0, 10.0, 30.0, "Wide stops for volatile moves"),
        "take_profit_pct": (60.0, 30.0, 100.0, "Target massive gains"),
        "max_position_sol": (0.15, 0.08, 0.30, "Maximum position sizing"),
        "min_liquidity_usd": (5000, 2000, 15000, "Early entry on new launches"),
        "max_risk_score": (0.70, 0.50, 0.90, "High risk tolerance"),
        "slippage_bps": (500, 350, 800, "High slippage for inclusion"),
        "max_hold_minutes": (60, 30, 120, "Long holds for 100x potential"),
        "kelly_safety_factor": (0.65, 0.50, 0.80, "Aggressive Kelly"),
    },
    # Degen + Win Rate (somewhat contradictory but common)
    (RiskTolerance.DEGEN, OptimizationGoal.MAX_WIN_RATE): {
        "stop_loss_pct": (15.0, 10.0, 22.0, "Wide but not reckless stops"),
        "take_profit_pct": (25.0, 15.0, 40.0, "Realistic profit targets"),
        "max_position_sol": (0.12, 0.06, 0.20, "Large positions"),
        "min_liquidity_usd": (10000, 5000, 20000, "Some quality filter"),
        "max_risk_score": (0.60, 0.45, 0.75, "Selective degen"),
        "slippage_bps": (450, 300, 650, "Aggressive slippage"),
        "max_hold_minutes": (40, 20, 80, "Flexible timing"),
        "kelly_safety_factor": (0.55, 0.40, 0.70, "Above half-Kelly"),
    },
}

# Market regime adjustments
MARKET_REGIME_ADJUSTMENTS = {
    MarketRegime.TRENDING_UP: {
        "take_profit_pct": 1.2,  # Increase profit targets
        "max_position_sol": 1.1,  # Slightly larger positions
        "max_hold_minutes": 1.3,  # Longer holds
    },
    MarketRegime.TRENDING_DOWN: {
        "stop_loss_pct": 0.8,  # Tighter stops
        "take_profit_pct": 0.7,  # Lower targets
        "max_position_sol": 0.7,  # Smaller positions
        "min_liquidity_usd": 1.5,  # Higher quality
    },
    MarketRegime.HIGH_VOLATILITY: {
        "stop_loss_pct": 1.3,  # Wider stops
        "slippage_bps": 1.3,  # More slippage tolerance
        "max_position_sol": 0.8,  # Smaller positions
    },
    MarketRegime.MEMECOON_SEASON: {
        "take_profit_pct": 1.5,  # Higher targets
        "max_risk_score": 1.2,  # Accept more risk
        "min_liquidity_usd": 0.6,  # Lower liquidity OK
    },
    MarketRegime.RISK_OFF: {
        "stop_loss_pct": 0.6,  # Tight stops
        "max_position_sol": 0.5,  # Much smaller
        "min_liquidity_usd": 2.0,  # Only high quality
        "max_risk_score": 0.5,  # Very selective
    },
}


class AIBacktestGuide:
    """AI-powered guided setup wizard for backtesting.

    Provides intelligent parameter recommendations based on:
    - User risk profile and goals
    - Market regime detection
    - Historical performance patterns
    - Parameter interaction validation

    Usage:
        guide = AIBacktestGuide()
        result = guide.create_guided_setup(
            risk_tolerance="moderate",
            primary_goal="balanced",
            capital_sol=1.0,
        )
        config = result.suggested_config
    """

    def __init__(
        self,
        market_analyzer: Callable | None = None,
        historical_performance: dict[str, Any] | None = None,
    ):
        """Initialize the AI guide.

        Args:
            market_analyzer: Optional function to detect market regime
            historical_performance: Optional dict of past backtest results
        """
        self.market_analyzer = market_analyzer
        self.historical_performance = historical_performance or {}
        self._parameter_history: list[dict[str, Any]] = []

    def create_guided_setup(
        self,
        risk_tolerance: str = "moderate",
        primary_goal: str = "balanced",
        capital_sol: float = 1.0,
        memecoin_focus: bool = True,
        max_daily_trades: int = 20,
        preferred_hold_time_minutes: int = 30,
        market_regime: str | None = None,
        current_positions: int = 0,
    ) -> GuidedSetupResult:
        """Create a guided setup with AI-powered recommendations.

        Args:
            risk_tolerance: One of "conservative", "moderate", "aggressive", "degen"
            primary_goal: One of "max_sharpe", "max_return", "min_drawdown",
                         "max_win_rate", "profit_quality", "balanced"
            capital_sol: Available capital in SOL
            memecoin_focus: Whether focusing on memecoins
            max_daily_trades: Maximum trades per day
            preferred_hold_time_minutes: Target hold time
            market_regime: Optional current market regime
            current_positions: Number of current open positions

        Returns:
            GuidedSetupResult with recommendations and suggested config
        """
        # Parse enums
        risk = RiskTolerance(risk_tolerance)
        goal = OptimizationGoal(primary_goal)
        regime = MarketRegime(market_regime) if market_regime else None

        # Create user profile
        profile = UserProfile(
            risk_tolerance=risk,
            primary_goal=goal,
            capital_sol=capital_sol,
            max_daily_trades=max_daily_trades,
            preferred_hold_time_minutes=preferred_hold_time_minutes,
            memecoin_focus=memecoin_focus,
        )

        # Get base recommendations from knowledge base
        key = (risk, goal)
        base_params = PARAMETER_KNOWLEDGE_BASE.get(
            key, PARAMETER_KNOWLEDGE_BASE[(RiskTolerance.MODERATE, OptimizationGoal.BALANCED)]
        )

        # Build recommendations
        recommendations = []
        suggested_config = {}

        for param_key, (value, min_v, max_v, rationale) in base_params.items():
            adjusted_value = value
            adjusted_min = min_v
            adjusted_max = max_v

            # Apply market regime adjustments
            if regime and regime in MARKET_REGIME_ADJUSTMENTS:
                adjustments = MARKET_REGIME_ADJUSTMENTS[regime]
                if param_key in adjustments:
                    multiplier = adjustments[param_key]
                    adjusted_value = value * multiplier
                    adjusted_min = min_v * multiplier
                    adjusted_max = max_v * multiplier

            # Apply capital-based scaling for position sizing
            if param_key == "max_position_sol":
                # Scale position size based on capital
                adjusted_value = min(adjusted_value, capital_sol * 0.15)
                adjusted_max = min(adjusted_max, capital_sol * 0.25)

            # Determine unit
            unit = self._get_param_unit(param_key)

            recommendations.append(
                ParameterRecommendation(
                    key=param_key,
                    label=self._get_param_label(param_key),
                    value=round(adjusted_value, 4),
                    min_value=round(adjusted_min, 4),
                    max_value=round(adjusted_max, 4),
                    unit=unit,
                    rationale=rationale,
                    confidence=self._calculate_confidence(param_key, risk, goal),
                    can_auto_tune=param_key not in ["min_liquidity_usd", "max_risk_score"],
                )
            )

            suggested_config[param_key] = adjusted_value

        # Add additional config parameters
        suggested_config.update(
            {
                "initial_capital_sol": capital_sol,
                "use_ensemble": True,
                "use_kelly_sizing": True,
                "enable_adversarial_scenarios": True,
                "enable_advanced_orchestration": True,
                "memecoin_focus": memecoin_focus,
            }
        )

        # Validate parameters
        validation_warnings = self._validate_parameters(suggested_config, profile)

        # Estimate expected metrics
        expected_metrics = self._estimate_metrics(suggested_config, profile)

        # Generate next steps
        next_steps = self._generate_next_steps(profile, validation_warnings)

        return GuidedSetupResult(
            profile=profile,
            recommendations=recommendations,
            suggested_config=suggested_config,
            validation_warnings=validation_warnings,
            expected_metrics=expected_metrics,
            next_steps=next_steps,
        )

    def quick_setup(
        self,
        preset: str = "balanced",
        capital_sol: float = 1.0,
    ) -> dict[str, Any]:
        """Quick setup with predefined presets.

        Args:
            preset: One of "conservative", "balanced", "aggressive", "degen"
            capital_sol: Available capital in SOL

        Returns:
            Ready-to-use configuration dict
        """
        preset_mapping = {
            "conservative": ("conservative", "min_drawdown"),
            "balanced": ("moderate", "balanced"),
            "aggressive": ("aggressive", "max_return"),
            "degen": ("degen", "max_return"),
        }

        risk, goal = preset_mapping.get(preset, ("moderate", "balanced"))
        result = self.create_guided_setup(
            risk_tolerance=risk,
            primary_goal=goal,
            capital_sol=capital_sol,
        )

        return result.suggested_config

    def analyze_current_params(
        self,
        current_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze current parameters and suggest improvements.

        Args:
            current_config: Current backtest configuration

        Returns:
            Analysis with suggestions and warnings
        """
        analysis = {
            "current_params": current_config,
            "issues": [],
            "suggestions": [],
            "optimization_potential": "medium",
        }

        # Check stop loss / take profit ratio
        sl = current_config.get("stop_loss_pct", 8.0)
        tp = current_config.get("take_profit_pct", 15.0)
        ratio = tp / sl if sl > 0 else 0

        if ratio < 1.5:
            analysis["issues"].append(
                {
                    "type": "poor_risk_reward",
                    "message": f"Risk/reward ratio is {ratio:.1f}:1. Minimum recommended is 1.5:1",
                    "suggestion": f"Increase take_profit_pct to at least {sl * 1.5:.1f}%",
                }
            )

        # Check position sizing vs capital
        capital = current_config.get("initial_capital_sol", 1.0)
        max_pos = current_config.get("max_position_sol", 0.05)
        pos_pct = (max_pos / capital) * 100 if capital > 0 else 0

        if pos_pct > 15:
            analysis["issues"].append(
                {
                    "type": "oversized_position",
                    "message": f"Max position is {pos_pct:.1f}% of "
                    "capital. High concentration risk.",
                    "suggestion": f"Reduce max_position_sol to {capital * 0.10:.4f} or less",
                }
            )

        # Check liquidity requirements
        min_liq = current_config.get("min_liquidity_usd", 10000)
        if min_liq < 5000:
            analysis["issues"].append(
                {
                    "type": "low_liquidity_filter",
                    "message": f"Min liquidity ${min_liq} is very low. Higher rug risk.",
                    "suggestion": "Increase min_liquidity_usd to at least 10000",
                }
            )

        # Check slippage
        slippage = current_config.get("slippage_bps", 300)
        if slippage > 500:
            analysis["suggestions"].append(
                {
                    "type": "high_slippage",
                    "message": f"Slippage tolerance {slippage / 100:.1f}% is high. "
                    "May accept bad fills.",
                    "suggestion": "Consider reducing to 300-400 bps for better fill quality",
                }
            )

        # Determine optimization potential
        issue_count = len(analysis["issues"])
        if issue_count == 0:
            analysis["optimization_potential"] = "low"
            analysis["suggestions"].append(
                {
                    "type": "ready",
                    "message": "Parameters look reasonable. Focus on fine-tuning via auto mode.",
                    "suggestion": "Run auto optimization to find optimal values",
                }
            )
        elif issue_count <= 2:
            analysis["optimization_potential"] = "medium"
        else:
            analysis["optimization_potential"] = "high"

        return analysis

    def _get_param_unit(self, key: str) -> str:
        """Get the unit for a parameter."""
        units = {
            "stop_loss_pct": "%",
            "take_profit_pct": "%",
            "max_position_sol": "SOL",
            "min_liquidity_usd": "USD",
            "max_fdv_usd": "USD",
            "max_risk_score": "0-1",
            "slippage_bps": "bps",
            "max_hold_minutes": "min",
            "kelly_safety_factor": "0-1",
        }
        return units.get(key, "")

    def _get_param_label(self, key: str) -> str:
        """Get human-readable label for a parameter."""
        labels = {
            "stop_loss_pct": "Stop Loss",
            "take_profit_pct": "Take Profit",
            "max_position_sol": "Max Position",
            "min_liquidity_usd": "Min Liquidity",
            "max_fdv_usd": "Max FDV",
            "max_risk_score": "Max Risk Score",
            "slippage_bps": "Slippage Tolerance",
            "max_hold_minutes": "Max Hold Time",
            "kelly_safety_factor": "Kelly Safety",
        }
        return labels.get(key, key)

    def _calculate_confidence(
        self, param_key: str, risk: RiskTolerance, goal: OptimizationGoal
    ) -> float:
        """Calculate confidence score for a recommendation."""
        base_confidence = 0.8

        # Higher confidence for core parameters
        if param_key in ["stop_loss_pct", "take_profit_pct", "max_position_sol"]:
            base_confidence = 0.9

        # Lower confidence for extreme risk profiles
        if risk == RiskTolerance.DEGEN:
            base_confidence *= 0.85

        return round(base_confidence, 2)

    def _validate_parameters(
        self,
        config: dict[str, Any],
        profile: UserProfile,
    ) -> list[str]:
        """Validate parameter combinations."""
        warnings = []

        sl = config.get("stop_loss_pct", 8.0)
        tp = config.get("take_profit_pct", 15.0)

        # Risk/reward check
        if tp / sl < 1.5:
            warnings.append(
                f"Take profit ({tp}%) is less than 1.5x stop loss ({sl}%). "
                "Consider increasing profit target."
            )

        # Position concentration
        capital = profile.capital_sol
        max_pos = config.get("max_position_sol", 0.05)
        if max_pos > capital * 0.15:
            warnings.append(
                f"Max position ({max_pos} SOL) is more than 15% of capital. "
                "High concentration risk."
            )

        # Slippage sanity
        slippage = config.get("slippage_bps", 300)
        if slippage > 600:
            warnings.append(
                f"Slippage tolerance ({slippage / 100}%) is very high. "
                "May result in poor fill quality."
            )

        # Liquidity filter
        min_liq = config.get("min_liquidity_usd", 10000)
        if min_liq < 5000 and profile.memecoin_focus:
            warnings.append(
                f"Low liquidity filter (${min_liq}) with memecoin focus. Higher rug pull risk."
            )

        return warnings

    def _estimate_metrics(
        self,
        config: dict[str, Any],
        profile: UserProfile,
    ) -> dict[str, float]:
        """Estimate expected metrics based on parameters and profile."""
        sl = config.get("stop_loss_pct", 8.0)
        tp = config.get("take_profit_pct", 15.0)
        risk_score = config.get("max_risk_score", 0.4)

        # Base estimates by risk tolerance
        base_metrics = {
            RiskTolerance.CONSERVATIVE: {
                "win_rate": 0.55,
                "sharpe": 1.8,
                "max_drawdown": 12.0,
                "monthly_return": 8.0,
            },
            RiskTolerance.MODERATE: {
                "win_rate": 0.50,
                "sharpe": 1.5,
                "max_drawdown": 20.0,
                "monthly_return": 15.0,
            },
            RiskTolerance.AGGRESSIVE: {
                "win_rate": 0.45,
                "sharpe": 1.2,
                "max_drawdown": 30.0,
                "monthly_return": 25.0,
            },
            RiskTolerance.DEGEN: {
                "win_rate": 0.40,
                "sharpe": 0.8,
                "max_drawdown": 50.0,
                "monthly_return": 40.0,
            },
        }

        base = base_metrics.get(profile.risk_tolerance, base_metrics[RiskTolerance.MODERATE])
        metrics = base.copy()

        # Adjust based on risk/reward ratio
        rr_ratio = tp / sl if sl > 0 else 1
        if rr_ratio >= 2.0:
            metrics["sharpe"] *= 1.2
            metrics["monthly_return"] *= 1.1
        elif rr_ratio < 1.5:
            metrics["sharpe"] *= 0.8
            metrics["max_drawdown"] *= 1.2

        # Adjust based on risk score
        if risk_score > 0.6:
            metrics["max_drawdown"] *= 1.3
            metrics["monthly_return"] *= 1.2
        elif risk_score < 0.3:
            metrics["max_drawdown"] *= 0.8
            metrics["monthly_return"] *= 0.9

        # Round values
        for key in metrics:
            metrics[key] = round(metrics[key], 2)

        return metrics

    def _generate_next_steps(
        self,
        profile: UserProfile,
        warnings: list[str],
    ) -> list[str]:
        """Generate recommended next steps."""
        steps = []

        if warnings:
            steps.append("Review and address validation warnings before proceeding")

        steps.append("Run a quick backtest with suggested parameters to validate")
        steps.append("Enable Auto Mode to let AI optimize parameters over time")

        if profile.risk_tolerance in [RiskTolerance.AGGRESSIVE, RiskTolerance.DEGEN]:
            steps.append("Consider paper trading first to validate strategy behavior")

        if profile.memecoin_focus:
            steps.append("Enable adversarial scenarios to test against rug conditions")

        steps.append("Set up monitoring alerts for drawdown thresholds")

        return steps

    def get_parameter_explanation(self, param_key: str) -> dict[str, Any]:
        """Get detailed explanation of a parameter."""
        explanations = {
            "stop_loss_pct": {
                "name": "Stop Loss Percentage",
                "description": "Maximum loss tolerance before automatic exit",
                "impact": "Lower values = tighter risk control but more whipsaws",
                "typical_range": "3-15% for memecoins",
                "pro_tip": "Set at 1.5-2x the normal volatility range",
            },
            "take_profit_pct": {
                "name": "Take Profit Percentage",
                "description": "Target gain percentage for automatic exit",
                "impact": "Higher values = bigger wins but harder to achieve",
                "typical_range": "10-50% for memecoins",
                "pro_tip": "Aim for at least 1.5:1 reward/risk ratio vs stop loss",
            },
            "max_position_sol": {
                "name": "Maximum Position Size",
                "description": "Maximum SOL to allocate per trade",
                "impact": "Larger positions = bigger P&L swings",
                "typical_range": "1-10% of total capital",
                "pro_tip": "Use Kelly Criterion for optimal sizing",
            },
            "slippage_bps": {
                "name": "Slippage Tolerance",
                "description": "Maximum price deviation acceptable for fills (in basis points)",
                "impact": "Higher = more inclusions but worse prices",
                "typical_range": "200-500 bps (2-5%) for memecoins",
                "pro_tip": "Balance between fill rate and execution quality",
            },
            "min_liquidity_usd": {
                "name": "Minimum Liquidity",
                "description": "Minimum pool liquidity required to enter",
                "impact": "Higher = safer but fewer opportunities",
                "typical_range": "$5,000-$50,000",
                "pro_tip": "Higher liquidity correlates with lower rug probability",
            },
            "max_risk_score": {
                "name": "Maximum Risk Score",
                "description": "Maximum acceptable risk score from GoPlus/TokenSniffer",
                "impact": "Lower = more filtering, fewer trades",
                "typical_range": "0.25-0.65",
                "pro_tip": "Never go above 0.7 for memecoins",
            },
        }

        return explanations.get(
            param_key,
            {
                "name": param_key,
                "description": "Parameter description not available",
                "impact": "Unknown",
                "typical_range": "Unknown",
                "pro_tip": "Consult documentation",
            },
        )
