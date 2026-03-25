"""
Fee calculation utilities for accurate P&L tracking.

Provides functions to calculate DEX fees and total trading costs.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def calculate_dex_fee(
    amount_out_lamports: int, dex_type: str = "raydium", side: str = "buy"
) -> int:
    """
    Calculate DEX swap fee based on output amount and DEX type.

    Args:
        amount_out_lamports: Actual amount received from swap (in lamports)
        dex_type: DEX protocol used ('raydium', 'orca', 'jupiter')
        side: Trade side ('buy' or 'sell')

    Returns:
        Fee in lamports

    Common DEX Fees:
        - Raydium: 0.25% (25 bps)
        - Orca: 0.30% (30 bps)
        - Jupiter: Variable (typically 0.30%)

    Example:
        >>> calculate_dex_fee(1_000_000_000, 'raydium')  # 1 SOL out
        2_500_000  # 0.0025 SOL = 0.25% fee
    """
    if amount_out_lamports <= 0:
        return 0

    # Fee in basis points (bps)
    fee_bps = {
        "raydium": 25,  # 0.25%
        "orca": 30,  # 0.30%
        "jupiter": 30,  # Conservative estimate
        "meteora": 25,  # 0.25%
        "lifinity": 30,  # Variable, using 0.30% as typical
    }.get(dex_type.lower(), 30)  # Default to 0.30% for unknown DEXes

    # Calculate fee
    # Formula: fee = amount_out * (fee_bps / 10000)
    fee_lamports = int(amount_out_lamports * fee_bps / 10000)

    logger.debug(
        f"Calculated DEX fee: {fee_lamports} lamports "
        f"({fee_lamports / 1_000_000_000.0:.6f} SOL) "
        f"for {amount_out_lamports} lamports on {dex_type}"
    )

    return fee_lamports


def calculate_total_fees(
    base_fee: int = 0, priority_fee: int = 0, jito_tip: int = 0, dex_fee: int = 0
) -> dict[str, Any]:
    """
    Calculate total fees and return breakdown.

    Args:
        base_fee: Base transaction fee in lamports (~5,000)
        priority_fee: Priority fee in lamports
        jito_tip: Jito bundle tip in lamports
        dex_fee: DEX swap fee in lamports

    Returns:
        Dict with:
            - total_lamports: Total fees in lamports
            - total_sol: Total fees in SOL
            - breakdown: Dict with individual fees in SOL
            - percentage_breakdown: Percentage of each fee type
    """
    total_lamports = base_fee + priority_fee + jito_tip + dex_fee
    total_sol = total_lamports / 1_000_000_000.0

    if total_lamports == 0:
        return {"total_lamports": 0, "total_sol": 0.0, "breakdown": {}, "percentage_breakdown": {}}

    breakdown = {
        "base": base_fee / 1_000_000_000.0,
        "priority": priority_fee / 1_000_000_000.0,
        "jito": jito_tip / 1_000_000_000.0,
        "dex": dex_fee / 1_000_000_000.0,
    }

    percentage_breakdown = {
        "base": (base_fee / total_lamports) * 100 if total_lamports > 0 else 0,
        "priority": (priority_fee / total_lamports) * 100 if total_lamports > 0 else 0,
        "jito": (jito_tip / total_lamports) * 100 if total_lamports > 0 else 0,
        "dex": (dex_fee / total_lamports) * 100 if total_lamports > 0 else 0,
    }

    return {
        "total_lamports": total_lamports,
        "total_sol": total_sol,
        "breakdown": breakdown,
        "percentage_breakdown": percentage_breakdown,
    }


def estimate_missing_fees(
    amount_sol: float, dex_type: str = "raydium", use_jito: bool = False
) -> dict[str, int]:
    """
    Estimate fees when actual fee data is not available.

    Useful for:
    - Backfilling historical trades
    - Paper trading simulation
    - Fee preview calculations

    Args:
        amount_sol: Trade amount in SOL
        dex_type: DEX protocol
        use_jito: Whether Jito bundles were used

    Returns:
        Dict with estimated fees in lamports:
            - base_fee_lamports
            - priority_fee_lamports
            - jito_tip_lamports
            - dex_fee_lamports
    """
    amount_lamports = int(amount_sol * 1_000_000_000)

    # Standard fees
    base_fee = 5000  # ~5,000 lamports typical base fee

    # Priority fee (varies, using median)
    priority_fee = 10000 if not use_jito else 0  # ~10k lamports median

    # Jito tip (if used)
    jito_tip = 50000 if use_jito else 0  # ~50k lamports typical tip

    # DEX fee
    dex_fee = calculate_dex_fee(amount_lamports, dex_type)

    logger.debug(
        f"Estimated fees for {amount_sol} SOL: "
        f"base={base_fee}, priority={priority_fee}, jito={jito_tip}, dex={dex_fee}"
    )

    return {
        "base_fee_lamports": base_fee,
        "priority_fee_lamports": priority_fee,
        "jito_tip_lamports": jito_tip,
        "dex_fee_lamports": dex_fee,
    }


def validate_fees(
    base_fee: int | None,
    priority_fee: int | None,
    jito_tip: int | None,
    dex_fee: int | None,
    trade_amount_sol: float,
) -> tuple[bool, str]:
    """
    Validate that fees are reasonable for a given trade.

    Args:
        base_fee: Base fee in lamports
        priority_fee: Priority fee in lamports
        jito_tip: Jito tip in lamports
        dex_fee: DEX fee in lamports
        trade_amount_sol: Trade size in SOL

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Calculate total
    total_fee_lamports = (base_fee or 0) + (priority_fee or 0) + (jito_tip or 0) + (dex_fee or 0)
    total_fee_sol = total_fee_lamports / 1_000_000_000.0

    # Sanity checks
    if base_fee is not None and base_fee < 0:
        return False, "Base fee cannot be negative"

    if base_fee is not None and base_fee > 1_000_000:  # > 0.001 SOL
        return False, f"Base fee unusually high: {base_fee} lamports"

    if priority_fee is not None and priority_fee > 10_000_000:  # > 0.01 SOL
        return False, f"Priority fee unusually high: {priority_fee} lamports"

    if jito_tip is not None and jito_tip > 1_000_000_000:  # > 1 SOL
        return False, f"Jito tip unusually high: {jito_tip} lamports"

    # Total fees should be < 5% of trade value (very conservative)
    if trade_amount_sol > 0:
        fee_percentage = (total_fee_sol / trade_amount_sol) * 100
        if fee_percentage > 5.0:
            return False, f"Total fees ({fee_percentage:.2f}%) exceed 5% of trade value"

    # DEX fee reasonableness (should be 0.1-1% typically)
    if dex_fee is not None and trade_amount_sol > 0:
        trade_lamports = int(trade_amount_sol * 1_000_000_000)
        dex_fee_pct = (dex_fee / trade_lamports) * 100
        if dex_fee_pct > 2.0:
            return False, f"DEX fee ({dex_fee_pct:.2f}%) exceeds reasonable threshold (2%)"

    return True, ""
