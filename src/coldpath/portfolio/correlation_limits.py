#
# correlation_limits.py
# 2DEXY ColdPath
#
# Correlation-Aware Portfolio Limits
#
# 10 correlated memecoin positions = 1 giant position.
# This module limits effective exposure, not just count.
#

import logging
import math
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TokenCategory(Enum):
    """Token categories for correlation estimation."""

    MEMECOIN = "memecoin"  # High correlation with each other
    DEFI = "defi"  # DeFi tokens
    NFT = "nft"  # NFT-related tokens
    INFRASTRUCTURE = "infrastructure"  # SOL, infrastructure
    STABLECOIN = "stablecoin"  # USDC, USDT
    UNKNOWN = "unknown"


@dataclass
class CorrelationConfig:
    """Configuration for correlation-aware limits."""

    # Maximum effective exposure in "position equivalents"
    max_effective_exposure: float = 3.0

    # Correlation threshold (above this = correlated)
    correlation_threshold: float = 0.5

    # Default correlation for memecoins (assume high correlation)
    default_memecoin_correlation: float = 0.7

    # Default correlation for unknown pairs
    default_correlation: float = 0.5

    # Correlation decay per category difference
    category_correlation_modifier: float = 0.2

    # Warning threshold (log warning when approaching limit)
    warning_threshold: float = 0.8  # 80% of max


@dataclass
class Position:
    """A position in the portfolio."""

    mint: str
    size_sol: float
    category: TokenCategory = TokenCategory.UNKNOWN
    entry_price: float | None = None
    current_price: float | None = None

    @property
    def value_sol(self) -> float:
        """Current value in SOL."""
        if self.current_price and self.entry_price:
            return self.size_sol * (self.current_price / self.entry_price)
        return self.size_sol


@dataclass
class ExposureResult:
    """Result of effective exposure calculation."""

    effective_exposure: float  # Effective exposure in position equivalents
    raw_exposure: float  # Sum of position sizes
    diversification_ratio: float  # Effective / Raw (lower = more correlated)
    can_add_position: bool  # Is there room for more?
    headroom_sol: float  # How much more can be added
    correlation_adjustment: float  # How much correlation increased exposure
    warnings: list[str] = field(default_factory=list)


class CorrelationAwareLimiter:
    """
    Limit effective exposure accounting for correlation.

    Problem: 10 positions of 0.1 SOL in correlated assets is NOT
    the same as 1 SOL in a single asset - it's actually MORE risky
    because they all crash together.

    Solution: Calculate "effective exposure" using correlation:
        effective = sqrt(sum_i sum_j w_i * w_j * rho_ij)

    Where:
        - w_i = weight (position size) of asset i
        - rho_ij = correlation between assets i and j

    This gives a correlation-adjusted exposure measure that
    properly accounts for diversification (or lack thereof).
    """

    # Pre-computed correlation matrix for common tokens
    # Format: (mint_a, mint_b) -> correlation
    # This is a placeholder - in production would be computed from price history
    _known_correlations: dict[tuple[str, str], float] = {}

    # Category base correlations
    CATEGORY_CORRELATIONS: dict[tuple[TokenCategory, TokenCategory], float] = {
        (TokenCategory.MEMECOIN, TokenCategory.MEMECOIN): 0.7,
        (TokenCategory.MEMECOIN, TokenCategory.DEFI): 0.4,
        (TokenCategory.MEMECOIN, TokenCategory.NFT): 0.3,
        (TokenCategory.MEMECOIN, TokenCategory.INFRASTRUCTURE): 0.2,
        (TokenCategory.MEMECOIN, TokenCategory.STABLECOIN): 0.0,
        (TokenCategory.DEFI, TokenCategory.DEFI): 0.5,
        (TokenCategory.DEFI, TokenCategory.INFRASTRUCTURE): 0.4,
        (TokenCategory.DEFI, TokenCategory.STABLECOIN): 0.1,
        (TokenCategory.NFT, TokenCategory.NFT): 0.4,
        (TokenCategory.INFRASTRUCTURE, TokenCategory.INFRASTRUCTURE): 0.3,
        (TokenCategory.STABLECOIN, TokenCategory.STABLECOIN): 0.1,
    }

    def __init__(self, config: CorrelationConfig | None = None):
        self.config = config or CorrelationConfig()
        self._positions: dict[str, Position] = {}
        self._position_categories: dict[str, TokenCategory] = {}

    def set_category(self, mint: str, category: TokenCategory) -> None:
        """Set the category for a token."""
        self._position_categories[mint] = category

    def get_category(self, mint: str) -> TokenCategory:
        """Get category for a token."""
        return self._position_categories.get(mint, TokenCategory.UNKNOWN)

    def set_correlation(self, mint_a: str, mint_b: str, correlation: float) -> None:
        """Set correlation between two tokens."""
        key = (min(mint_a, mint_b), max(mint_a, mint_b))
        self._known_correlations[key] = max(-1.0, min(1.0, correlation))

    def get_correlation(self, mint_a: str, mint_b: str) -> float:
        """
        Get correlation between two tokens.

        Uses known correlations first, then category-based estimation,
        then default correlation.
        """
        if mint_a == mint_b:
            return 1.0

        # Check known correlations
        key = (min(mint_a, mint_b), max(mint_a, mint_b))
        if key in self._known_correlations:
            return self._known_correlations[key]

        # Estimate from categories
        cat_a = self.get_category(mint_a)
        cat_b = self.get_category(mint_b)

        # Look up category correlation
        cat_key = (cat_a, cat_b)
        cat_key_rev = (cat_b, cat_a)

        if cat_key in self.CATEGORY_CORRELATIONS:
            return self.CATEGORY_CORRELATIONS[cat_key]
        elif cat_key_rev in self.CATEGORY_CORRELATIONS:
            return self.CATEGORY_CORRELATIONS[cat_key_rev]

        # Default correlation
        if cat_a == TokenCategory.MEMECOIN and cat_b == TokenCategory.MEMECOIN:
            return self.config.default_memecoin_correlation

        return self.config.default_correlation

    def add_position(self, position: Position) -> None:
        """Add a position to the portfolio."""
        self._positions[position.mint] = position
        if position.category != TokenCategory.UNKNOWN:
            self._position_categories[position.mint] = position.category

    def remove_position(self, mint: str) -> None:
        """Remove a position from the portfolio."""
        self._positions.pop(mint, None)

    def update_position_size(self, mint: str, new_size_sol: float) -> None:
        """Update the size of an existing position."""
        if mint in self._positions:
            self._positions[mint].size_sol = new_size_sol

    def get_positions(self) -> list[Position]:
        """Get all positions."""
        return list(self._positions.values())

    def effective_exposure(self, positions: list[Position] | None = None) -> float:
        """
        Calculate effective exposure given correlations.

        Formula: effective = sqrt(sum_i sum_j w_i * w_j * rho_ij)

        This is the square root of the portfolio variance when
        all positions have unit variance.
        """
        if positions is None:
            positions = list(self._positions.values())

        if not positions:
            return 0.0

        n = len(positions)

        # Get weights (position sizes)
        weights = [p.size_sol for p in positions]

        # Build correlation matrix
        # rho[i][j] = correlation between position i and j
        total = 0.0
        for i in range(n):
            for j in range(n):
                rho = self.get_correlation(positions[i].mint, positions[j].mint)
                total += weights[i] * weights[j] * rho

        # Effective exposure is sqrt of total
        effective = math.sqrt(max(0.0, total))

        return effective

    def can_add_position(
        self,
        new_position: Position,
        positions: list[Position] | None = None,
    ) -> tuple[bool, float]:
        """
        Check if adding position would exceed effective exposure limit.

        Returns: (can_add, headroom_after)
        """
        if positions is None:
            positions = list(self._positions.values())

        # Test with new position added
        test_positions = positions + [new_position]
        effective = self.effective_exposure(test_positions)

        can_add = effective <= self.config.max_effective_exposure
        headroom = self.config.max_effective_exposure - effective

        return can_add, headroom

    def calculate_exposure_result(
        self,
        positions: list[Position] | None = None,
    ) -> ExposureResult:
        """
        Calculate comprehensive exposure metrics.
        """
        if positions is None:
            positions = list(self._positions.values())

        if not positions:
            return ExposureResult(
                effective_exposure=0.0,
                raw_exposure=0.0,
                diversification_ratio=1.0,
                can_add_position=True,
                headroom_sol=self.config.max_effective_exposure,
                correlation_adjustment=1.0,
                warnings=[],
            )

        # Calculate raw and effective exposure
        raw = sum(p.size_sol for p in positions)
        effective = self.effective_exposure(positions)

        # Diversification ratio
        diversification = effective / raw if raw > 0 else 1.0

        # Correlation adjustment factor
        correlation_adj = (
            effective / math.sqrt(sum(p.size_sol**2 for p in positions)) if raw > 0 else 1.0
        )

        # Can add more?
        can_add, headroom = self.can_add_position(
            Position(mint="_test_", size_sol=0.1, category=TokenCategory.MEMECOIN),
            positions,
        )

        # Warnings
        warnings = []
        if effective >= self.config.max_effective_exposure * self.config.warning_threshold:
            warnings.append(
                f"Effective exposure ({effective:.2f}) at "
                f"{effective / self.config.max_effective_exposure:.0%} of limit"
            )

        if diversification < 0.5:
            warnings.append(
                f"Low diversification ({diversification:.2f}) - positions are highly correlated"
            )

        return ExposureResult(
            effective_exposure=effective,
            raw_exposure=raw,
            diversification_ratio=diversification,
            can_add_position=can_add,
            headroom_sol=headroom,
            correlation_adjustment=correlation_adj,
            warnings=warnings,
        )

    def get_max_position_size(
        self,
        mint: str,
        category: TokenCategory,
        positions: list[Position] | None = None,
    ) -> float:
        """
        Calculate maximum position size that can be added for a token.

        Uses binary search to find the maximum size that doesn't
        exceed the effective exposure limit.
        """
        if positions is None:
            positions = list(self._positions.values())

        # Binary search for max size
        low, high = 0.0, self.config.max_effective_exposure * 2

        for _ in range(20):  # 20 iterations gives ~0.0001 precision
            mid = (low + high) / 2
            test_position = Position(mint=mint, size_sol=mid, category=category)
            can_add, _ = self.can_add_position(test_position, positions)

            if can_add:
                low = mid
            else:
                high = mid

        return low

    def get_stats(self) -> dict:
        """Get current statistics."""
        result = self.calculate_exposure_result()

        return {
            "positions": len(self._positions),
            "raw_exposure_sol": result.raw_exposure,
            "effective_exposure": result.effective_exposure,
            "diversification_ratio": result.diversification_ratio,
            "max_exposure": self.config.max_effective_exposure,
            "headroom_sol": result.headroom_sol,
            "warnings": result.warnings,
        }

    def reset(self) -> None:
        """Clear all positions."""
        self._positions.clear()


# Singleton instance
_limiter: CorrelationAwareLimiter | None = None


def get_correlation_limiter() -> CorrelationAwareLimiter:
    """Get the global correlation limiter instance."""
    global _limiter
    if _limiter is None:
        _limiter = CorrelationAwareLimiter()
    return _limiter
