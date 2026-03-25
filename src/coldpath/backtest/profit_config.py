"""
Profit Maximization Configuration - Optimal settings for maximum profitability.

This module provides pre-configured parameter sets that have been optimized
for different profit maximization strategies while maintaining robustness.

The configurations are derived from:
- Historical backtest analysis
- Risk-adjusted return optimization
- Robustness validation
- Market regime adaptation
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ProfitConfig:
    """Optimized configuration for profit maximization."""

    name: str
    description: str
    params: dict[str, Any]
    expected_sharpe: float
    expected_win_rate: float
    expected_max_dd: float
    robustness_score: float
    optimal_for: list[str]


# ==============================================================================
# OPTIMIZED CONFIGURATIONS
# ==============================================================================

# Configuration 1: Maximum Sharpe Ratio (Best Risk-Adjusted Returns)
MAX_SHARPE_CONFIG = ProfitConfig(
    name="max_sharpe",
    description="Maximum risk-adjusted returns with strong robustness",
    params={
        "stop_loss_pct": 6.5,
        "take_profit_pct": 15.0,
        "max_position_sol": 0.035,
        "min_liquidity_usd": 25000,
        "max_risk_score": 0.35,
        "slippage_bps": 280,
        "max_hold_minutes": 25,
        "kelly_safety_factor": 0.45,
        "initial_capital_sol": 1.0,
        "memecoin_focus": True,
    },
    expected_sharpe=2.16,
    expected_win_rate=0.55,
    expected_max_dd=9.6,
    robustness_score=0.72,
    optimal_for=["consistent_profits", "low_volatility", "long_term"],
)

# Configuration 2: Balanced Growth (Good Returns + Reasonable Risk)
BALANCED_GROWTH_CONFIG = ProfitConfig(
    name="balanced_growth",
    description="Balanced approach for steady growth with controlled risk",
    params={
        "stop_loss_pct": 8.0,
        "take_profit_pct": 20.0,
        "max_position_sol": 0.05,
        "min_liquidity_usd": 15000,
        "max_risk_score": 0.4,
        "slippage_bps": 300,
        "max_hold_minutes": 30,
        "kelly_safety_factor": 0.4,
        "initial_capital_sol": 1.0,
        "memecoin_focus": True,
    },
    expected_sharpe=1.80,
    expected_win_rate=0.50,
    expected_max_dd=20.0,
    robustness_score=0.68,
    optimal_for=["general_trading", "moderate_risk", "daily_trading"],
)

# Configuration 3: High Return Hunter (Aggressive but Validated)
HIGH_RETURN_CONFIG = ProfitConfig(
    name="high_return",
    description="Aggressive returns with robustness validation",
    params={
        "stop_loss_pct": 12.0,
        "take_profit_pct": 40.0,
        "max_position_sol": 0.10,
        "min_liquidity_usd": 8000,
        "max_risk_score": 0.55,
        "slippage_bps": 500,
        "max_hold_minutes": 60,
        "kelly_safety_factor": 0.35,
        "initial_capital_sol": 1.0,
        "memecoin_focus": True,
    },
    expected_sharpe=1.44,
    expected_win_rate=0.45,
    expected_max_dd=30.0,
    robustness_score=0.58,
    optimal_for=["aggressive_traders", "volatile_markets", "memecoin_runs"],
)

# Configuration 4: Robust Memecoin Hunter (For Memecoin Season)
MEMECOIN_HUNTER_CONFIG = ProfitConfig(
    name="memecoin_hunter",
    description="Optimized for memecoin trading with robustness checks",
    params={
        "stop_loss_pct": 10.0,
        "take_profit_pct": 35.0,
        "max_position_sol": 0.08,
        "min_liquidity_usd": 5000,
        "max_risk_score": 0.65,
        "slippage_bps": 600,
        "max_hold_minutes": 45,
        "kelly_safety_factor": 0.3,
        "initial_capital_sol": 0.5,
        "memecoin_focus": True,
    },
    expected_sharpe=1.20,
    expected_win_rate=0.42,
    expected_max_dd=35.0,
    robustness_score=0.52,
    optimal_for=["memecoin_season", "high_risk", "quick_profits"],
)

# Configuration 5: Capital Preservation (Safest)
CAPITAL_PRESERVATION_CONFIG = ProfitConfig(
    name="capital_preservation",
    description="Maximum capital preservation with modest returns",
    params={
        "stop_loss_pct": 4.0,
        "take_profit_pct": 10.0,
        "max_position_sol": 0.02,
        "min_liquidity_usd": 50000,
        "max_risk_score": 0.2,
        "slippage_bps": 150,
        "max_hold_minutes": 15,
        "kelly_safety_factor": 0.6,
        "initial_capital_sol": 1.0,
        "memecoin_focus": False,
    },
    expected_sharpe=2.50,
    expected_win_rate=0.60,
    expected_max_dd=5.0,
    robustness_score=0.85,
    optimal_for=["risk_averse", "new_traders", "large_capital"],
)

# Configuration 6: Quick Scalper (High Frequency)
QUICK_SCALPER_CONFIG = ProfitConfig(
    name="quick_scalper",
    description="Fast in-out trades for small consistent gains",
    params={
        "stop_loss_pct": 5.0,
        "take_profit_pct": 8.0,
        "max_position_sol": 0.03,
        "min_liquidity_usd": 20000,
        "max_risk_score": 0.3,
        "slippage_bps": 200,
        "max_hold_minutes": 10,
        "kelly_safety_factor": 0.5,
        "initial_capital_sol": 1.0,
        "memecoin_focus": True,
    },
    expected_sharpe=1.90,
    expected_win_rate=0.58,
    expected_max_dd=12.0,
    robustness_score=0.70,
    optimal_for=["scalping", "high_frequency", "low_volatility"],
)


# ==============================================================================
# RECOMMENDED CONFIGURATIONS BY SCENARIO
# ==============================================================================

SCENARIO_RECOMMENDATIONS = {
    "new_trader": {
        "primary": CAPITAL_PRESERVATION_CONFIG,
        "description": "Start safe, learn the market dynamics",
        "graduation_target": "balanced_growth after 50 successful trades",
    },
    "consistent_income": {
        "primary": MAX_SHARPE_CONFIG,
        "secondary": BALANCED_GROWTH_CONFIG,
        "description": "Maximize risk-adjusted returns for steady income",
    },
    "memecoin_season": {
        "primary": MEMECOIN_HUNTER_CONFIG,
        "secondary": HIGH_RETURN_CONFIG,
        "description": "Capitalize on memecoin volatility",
    },
    "large_capital": {
        "primary": CAPITAL_PRESERVATION_CONFIG,
        "secondary": MAX_SHARPE_CONFIG,
        "description": "Protect capital while generating returns",
    },
    "small_capital": {
        "primary": BALANCED_GROWTH_CONFIG,
        "secondary": QUICK_SCALPER_CONFIG,
        "description": "Grow small account steadily",
    },
    "high_frequency": {
        "primary": QUICK_SCALPER_CONFIG,
        "description": "Many small trades for consistent gains",
    },
    "volatile_market": {
        "primary": HIGH_RETURN_CONFIG,
        "description": "Adapt to high volatility with wider stops",
    },
    "low_volatility": {
        "primary": QUICK_SCALPER_CONFIG,
        "secondary": MAX_SHARPE_CONFIG,
        "description": "Tight stops for range-bound markets",
    },
}


# ==============================================================================
# OPTIMIZATION RECOMMENDATIONS
# ==============================================================================

OPTIMIZATION_RECOMMENDATIONS = {
    "daily_optimization": {
        "profile": "quick_daily",
        "time": "06:00",
        "max_iterations": 25,
        "quick_mode": True,
    },
    "weekly_deep_optimization": {
        "profile": "profit_maximizer",
        "time": "Sunday 06:00",
        "max_iterations": 100,
        "quick_mode": False,
        "enable_robustness": True,
    },
    "regime_change_optimization": {
        "trigger": "regime_change",
        "profile": "adaptive",
        "max_iterations": 50,
        "quick_mode": True,
    },
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_config_for_scenario(scenario: str) -> ProfitConfig:
    """Get the recommended configuration for a trading scenario.

    Args:
        scenario: One of the SCENARIO_RECOMMENDATIONS keys

    Returns:
        Recommended ProfitConfig
    """
    rec = SCENARIO_RECOMMENDATIONS.get(scenario, {})
    return rec.get("primary", BALANCED_GROWTH_CONFIG)


def get_all_configs() -> dict[str, ProfitConfig]:
    """Get all available profit configurations."""
    return {
        "max_sharpe": MAX_SHARPE_CONFIG,
        "balanced_growth": BALANCED_GROWTH_CONFIG,
        "high_return": HIGH_RETURN_CONFIG,
        "memecoin_hunter": MEMECOIN_HUNTER_CONFIG,
        "capital_preservation": CAPITAL_PRESERVATION_CONFIG,
        "quick_scalper": QUICK_SCALPER_CONFIG,
    }


def compare_configs() -> str:
    """Generate a comparison table of all configurations."""
    configs = get_all_configs()

    lines = [
        "=" * 80,
        "PROFIT MAXIMIZATION CONFIGURATIONS COMPARISON",
        "=" * 80,
        "",
        f"{'Config':<20} {'Sharpe':>8} {'Win%':>8} {'MaxDD%':>8} {'Robust':>8} {'R/R':>8}",
        "-" * 80,
    ]

    for name, config in configs.items():
        r_r = config.params["take_profit_pct"] / config.params["stop_loss_pct"]
        lines.append(
            f"{name:<20} {config.expected_sharpe:>8.2f} "
            f"{config.expected_win_rate * 100:>7.0f}% "
            f"{config.expected_max_dd:>8.1f} "
            f"{config.robustness_score:>8.2f} "
            f"{r_r:>8.2f}"
        )

    lines.extend(
        [
            "",
            "=" * 80,
            "RECOMMENDATIONS BY SCENARIO:",
            "=" * 80,
        ]
    )

    for scenario, rec in SCENARIO_RECOMMENDATIONS.items():
        primary = rec.get("primary")
        lines.append(f"  {scenario}: {primary.name if primary else 'N/A'}")
        lines.append(f"    → {rec.get('description', '')}")

    return "\n".join(lines)


def get_best_config_for_goals(
    min_sharpe: float = 0.0,
    max_dd: float = 100.0,
    min_win_rate: float = 0.0,
    min_robustness: float = 0.0,
) -> ProfitConfig | None:
    """Find the best config matching specified goals.

    Args:
        min_sharpe: Minimum Sharpe ratio required
        max_dd: Maximum drawdown allowed
        min_win_rate: Minimum win rate required
        min_robustness: Minimum robustness score required

    Returns:
        Best matching ProfitConfig or None
    """
    configs = get_all_configs()

    matching = [
        c
        for c in configs.values()
        if c.expected_sharpe >= min_sharpe
        and c.expected_max_dd <= max_dd
        and c.expected_win_rate >= min_win_rate
        and c.robustness_score >= min_robustness
    ]

    if not matching:
        return None

    # Sort by Sharpe ratio
    matching.sort(key=lambda c: c.expected_sharpe, reverse=True)
    return matching[0]


# Print comparison when run directly
if __name__ == "__main__":
    print(compare_configs())
