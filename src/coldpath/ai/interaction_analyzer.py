"""
Parameter Interaction Analyzer - Detect synergies and conflicts between parameters.

Analyzes how trading strategy parameters interact and affect each other:
1. Synergies: Parameters that work well together
2. Conflicts: Parameters that counteract each other
3. Dominance: One parameter overshadowing another

Examples:
- Synergy: high min_confidence + aggressive sizing -> balanced risk
- Conflict: tight stop_loss + long max_hold -> contradictory
- Dominance: very high min_confidence -> filters dominate sizing

Uses both rule-based detection (known patterns) and statistical analysis
(correlation from historical data).
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ParameterInteraction:
    """Interaction between two parameters.

    Attributes:
        param1: First parameter name.
        param2: Second parameter name.
        interaction_type: "synergy", "conflict", or "neutral".
        strength: Interaction strength (0.0 to 1.0).
        description: Human-readable description.
        recommendation: Optional recommended action.
    """

    param1: str
    param2: str
    interaction_type: str  # "synergy", "conflict", "neutral", "dominance"
    strength: float  # 0-1
    description: str
    recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "param1": self.param1,
            "param2": self.param2,
            "interaction_type": self.interaction_type,
            "strength": self.strength,
            "description": self.description,
            "recommendation": self.recommendation,
        }


@dataclass
class InteractionReport:
    """Report of all detected parameter interactions.

    Attributes:
        total_interactions: Total interactions detected.
        synergies: List of synergistic interactions.
        conflicts: List of conflicting interactions.
        dominances: List of dominance interactions.
        overall_coherence: Overall parameter coherence (0-1, higher = better).
    """

    total_interactions: int
    synergies: list[ParameterInteraction]
    conflicts: list[ParameterInteraction]
    dominances: list[ParameterInteraction]
    overall_coherence: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_interactions": self.total_interactions,
            "synergies": [s.to_dict() for s in self.synergies],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "dominances": [d.to_dict() for d in self.dominances],
            "overall_coherence": self.overall_coherence,
            "num_synergies": len(self.synergies),
            "num_conflicts": len(self.conflicts),
            "num_dominances": len(self.dominances),
        }


# Type for interaction rule check functions
InteractionCheck = Callable[[float, float], bool]


@dataclass
class InteractionRule:
    """Rule-based interaction definition."""

    param1: str
    param2: str
    check: InteractionCheck
    interaction_type: str
    strength: float
    description: str
    recommendation: str


class ParameterInteractionAnalyzer:
    """
    Analyze how parameters interact and affect each other.

    Combines:
    1. Rule-based detection (known patterns from trading domain)
    2. Statistical analysis (from parameter value ranges)
    3. Ratio analysis (risk/reward, time/risk relationships)

    Usage:
        analyzer = ParameterInteractionAnalyzer()
        interactions = await analyzer.analyze_interactions({
            "stop_loss_pct": 5.0,
            "take_profit_pct": 8.0,
            "max_hold_duration_seconds": 7200,
            "min_confidence": 0.8,
            ...
        })
    """

    # Known interaction rules (rule-based detection)
    INTERACTION_RULES: list[InteractionRule] = [
        # Risk/Reward ratio
        InteractionRule(
            param1="stop_loss_pct",
            param2="take_profit_pct",
            check=lambda sl, tp: sl / tp > 0.5 if tp > 0 else False,
            interaction_type="conflict",
            strength=0.8,
            description="Risk/reward ratio < 2:1 (too tight)",
            recommendation=(
                "Widen take profit to at least 2x stop loss for better risk/reward ratio"
            ),
        ),
        # Stop loss vs hold duration
        InteractionRule(
            param1="stop_loss_pct",
            param2="max_hold_duration_seconds",
            check=lambda sl, hold: sl < 8 and hold > 3600,
            interaction_type="conflict",
            strength=0.7,
            description="Tight stop loss with long hold duration (contradictory)",
            recommendation=("Either widen stop loss for longer holds or shorten max hold duration"),
        ),
        # Stop loss too tight for volatile markets
        InteractionRule(
            param1="stop_loss_pct",
            param2="max_position_sol",
            check=lambda sl, pos: sl < 5 and pos > 0.1,
            interaction_type="conflict",
            strength=0.6,
            description="Tight stop with large position (frequent stops)",
            recommendation=("Reduce position size or widen stop loss to avoid excessive stop-outs"),
        ),
        # Confidence and position sizing synergy
        InteractionRule(
            param1="min_confidence",
            param2="max_position_sol",
            check=lambda conf, pos: conf > 0.7 and pos > 0.05,
            interaction_type="synergy",
            strength=0.7,
            description="High confidence filter with moderate sizing (balanced)",
            recommendation=None,
        ),
        # Very high confidence dominance
        InteractionRule(
            param1="min_confidence",
            param2="max_position_sol",
            check=lambda conf, pos: conf > 0.9,
            interaction_type="dominance",
            strength=0.8,
            description=("Very high confidence threshold dominates - few trades pass"),
            recommendation=(
                "Consider lowering min_confidence below 0.9 to increase trade frequency"
            ),
        ),
        # Take profit and hold duration synergy
        InteractionRule(
            param1="take_profit_pct",
            param2="max_hold_duration_seconds",
            check=lambda tp, hold: tp > 30 and hold > 1800,
            interaction_type="synergy",
            strength=0.6,
            description="Generous take profit with adequate hold time",
            recommendation=None,
        ),
        # Slippage and position size conflict
        InteractionRule(
            param1="slippage_bps",
            param2="max_position_sol",
            check=lambda slip, pos: slip > 300 and pos > 0.1,
            interaction_type="conflict",
            strength=0.7,
            description="High slippage tolerance with large positions (cost drag)",
            recommendation=(
                "Reduce position size to minimize slippage impact, or tighten slippage tolerance"
            ),
        ),
        # Max positions and position size
        InteractionRule(
            param1="max_positions",
            param2="max_position_sol",
            check=lambda mp, ps: mp > 5 and ps > 0.1,
            interaction_type="conflict",
            strength=0.5,
            description="Many positions with large individual sizes (over-exposure)",
            recommendation=(
                "Reduce max_positions or max_position_sol to manage total capital exposure"
            ),
        ),
        # Stop loss and take profit synergy (good ratio)
        InteractionRule(
            param1="stop_loss_pct",
            param2="take_profit_pct",
            check=lambda sl, tp: 2.5 <= tp / sl <= 4.0 if sl > 0 else False,
            interaction_type="synergy",
            strength=0.8,
            description="Optimal risk/reward ratio (2.5:1 to 4:1)",
            recommendation=None,
        ),
    ]

    def __init__(
        self,
        custom_rules: list[InteractionRule] | None = None,
    ):
        """Initialize the analyzer.

        Args:
            custom_rules: Additional custom interaction rules.
        """
        self.rules = list(self.INTERACTION_RULES)
        if custom_rules:
            self.rules.extend(custom_rules)

    async def analyze_interactions(
        self,
        params: dict[str, Any],
    ) -> InteractionReport:
        """
        Analyze all parameter interactions.

        Combines:
        1. Rule-based detection (known patterns)
        2. Ratio analysis (risk/reward)
        3. Range analysis (extreme values)

        Args:
            params: Strategy parameter dictionary.

        Returns:
            InteractionReport with all detected interactions.
        """
        interactions: list[ParameterInteraction] = []

        # 1. Rule-based detection
        rule_interactions = self._check_rules(params)
        interactions.extend(rule_interactions)

        # 2. Range analysis (detect extreme values)
        range_interactions = self._check_ranges(params)
        interactions.extend(range_interactions)

        # 3. Ratio analysis
        ratio_interactions = self._check_ratios(params)
        interactions.extend(ratio_interactions)

        # Deduplicate (same param pair, same type)
        interactions = self._deduplicate(interactions)

        # Classify
        synergies = [i for i in interactions if i.interaction_type == "synergy"]
        conflicts = [i for i in interactions if i.interaction_type == "conflict"]
        dominances = [i for i in interactions if i.interaction_type == "dominance"]

        # Overall coherence score
        if not interactions:
            coherence = 1.0
        else:
            conflict_penalty = sum(c.strength for c in conflicts) * 0.3
            dominance_penalty = sum(d.strength for d in dominances) * 0.1
            synergy_bonus = sum(s.strength for s in synergies) * 0.1
            coherence = max(
                0.0,
                min(1.0, 1.0 - conflict_penalty - dominance_penalty + synergy_bonus),
            )

        report = InteractionReport(
            total_interactions=len(interactions),
            synergies=synergies,
            conflicts=conflicts,
            dominances=dominances,
            overall_coherence=coherence,
        )

        logger.info(
            "Parameter interaction analysis: %d synergies, %d conflicts, "
            "%d dominances, coherence=%.2f",
            len(synergies),
            len(conflicts),
            len(dominances),
            coherence,
        )

        return report

    async def detect_conflicts(
        self,
        params: dict[str, Any],
    ) -> list[ParameterInteraction]:
        """Find conflicting parameter interactions.

        Args:
            params: Strategy parameter dictionary.

        Returns:
            List of conflict interactions.
        """
        report = await self.analyze_interactions(params)
        return report.conflicts

    async def detect_synergies(
        self,
        params: dict[str, Any],
    ) -> list[ParameterInteraction]:
        """Find synergistic parameter interactions.

        Args:
            params: Strategy parameter dictionary.

        Returns:
            List of synergy interactions.
        """
        report = await self.analyze_interactions(params)
        return report.synergies

    def _check_rules(
        self,
        params: dict[str, Any],
    ) -> list[ParameterInteraction]:
        """Apply rule-based interaction checks.

        Args:
            params: Strategy parameter dictionary.

        Returns:
            List of detected interactions.
        """
        interactions: list[ParameterInteraction] = []

        for rule in self.rules:
            val1 = params.get(rule.param1)
            val2 = params.get(rule.param2)

            if val1 is None or val2 is None:
                continue

            try:
                if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
                    continue

                if rule.check(float(val1), float(val2)):
                    interactions.append(
                        ParameterInteraction(
                            param1=rule.param1,
                            param2=rule.param2,
                            interaction_type=rule.interaction_type,
                            strength=rule.strength,
                            description=rule.description,
                            recommendation=rule.recommendation,
                        )
                    )
            except (ValueError, ZeroDivisionError, TypeError):
                continue

        return interactions

    def _check_ranges(
        self,
        params: dict[str, Any],
    ) -> list[ParameterInteraction]:
        """Check for extreme parameter values that create dominance.

        Args:
            params: Strategy parameter dictionary.

        Returns:
            List of detected dominance interactions.
        """
        interactions: list[ParameterInteraction] = []

        # Known extreme value ranges
        extreme_checks = {
            "stop_loss_pct": {
                "low": (0, 3),
                "low_desc": "Very tight stop loss (<3%) will trigger frequently",
                "high": (25, 100),
                "high_desc": "Very wide stop loss (>25%) provides minimal protection",
            },
            "take_profit_pct": {
                "low": (0, 5),
                "low_desc": "Very low take profit (<5%) clips gains",
                "high": (100, 1000),
                "high_desc": "Very high take profit (>100%) rarely triggers",
            },
            "min_confidence": {
                "low": (0, 0.3),
                "low_desc": "Very low confidence threshold accepts noise",
                "high": (0.9, 1.0),
                "high_desc": "Very high confidence threshold filters most signals",
            },
            "max_position_sol": {
                "low": (0, 0.005),
                "low_desc": "Very small position size limits returns",
                "high": (0.5, 100),
                "high_desc": "Very large position size increases risk",
            },
        }

        for param_name, checks in extreme_checks.items():
            value = params.get(param_name)
            if value is None or not isinstance(value, (int, float)):
                continue

            low_min, low_max = checks["low"]
            if low_min <= value <= low_max:
                interactions.append(
                    ParameterInteraction(
                        param1=param_name,
                        param2=param_name,
                        interaction_type="dominance",
                        strength=0.6,
                        description=checks["low_desc"],
                        recommendation=(f"Consider increasing {param_name} from {value}"),
                    )
                )

            high_min, high_max = checks["high"]
            if high_min <= value <= high_max:
                interactions.append(
                    ParameterInteraction(
                        param1=param_name,
                        param2=param_name,
                        interaction_type="dominance",
                        strength=0.6,
                        description=checks["high_desc"],
                        recommendation=(f"Consider decreasing {param_name} from {value}"),
                    )
                )

        return interactions

    def _check_ratios(
        self,
        params: dict[str, Any],
    ) -> list[ParameterInteraction]:
        """Analyze key parameter ratios.

        Args:
            params: Strategy parameter dictionary.

        Returns:
            List of detected interactions from ratio analysis.
        """
        interactions: list[ParameterInteraction] = []

        # Risk/Reward ratio
        sl = params.get("stop_loss_pct")
        tp = params.get("take_profit_pct")
        if (
            sl is not None
            and tp is not None
            and isinstance(sl, (int, float))
            and isinstance(tp, (int, float))
            and sl > 0
        ):
            ratio = tp / sl
            if ratio < 1.0:
                interactions.append(
                    ParameterInteraction(
                        param1="stop_loss_pct",
                        param2="take_profit_pct",
                        interaction_type="conflict",
                        strength=0.9,
                        description=(
                            f"Negative risk/reward ratio ({ratio:.1f}:1). "
                            "Stop loss is wider than take profit"
                        ),
                        recommendation=("Take profit should be at least 2x stop loss"),
                    )
                )

        # Capital exposure ratio
        max_pos = params.get("max_position_sol")
        max_positions = params.get("max_positions")
        if (
            max_pos is not None
            and max_positions is not None
            and isinstance(max_pos, (int, float))
            and isinstance(max_positions, (int, float))
        ):
            total_exposure = max_pos * max_positions
            if total_exposure > 1.0:
                interactions.append(
                    ParameterInteraction(
                        param1="max_position_sol",
                        param2="max_positions",
                        interaction_type="conflict",
                        strength=0.7,
                        description=(
                            f"Total capital exposure {total_exposure:.2f} SOL "
                            "may exceed available capital"
                        ),
                        recommendation=(
                            "Reduce max_position_sol or max_positions "
                            "to keep total exposure within capital"
                        ),
                    )
                )

        return interactions

    def _deduplicate(
        self,
        interactions: list[ParameterInteraction],
    ) -> list[ParameterInteraction]:
        """Deduplicate interactions (keep highest strength).

        Args:
            interactions: List of interactions to deduplicate.

        Returns:
            Deduplicated list.
        """
        seen: dict[tuple[str, str, str], ParameterInteraction] = {}

        for inter in interactions:
            # Normalize key (sorted param names + type)
            key = (
                min(inter.param1, inter.param2),
                max(inter.param1, inter.param2),
                inter.interaction_type,
            )

            if key not in seen or inter.strength > seen[key].strength:
                seen[key] = inter

        return list(seen.values())
