"""
Numba-accelerated technical indicators for fast signal generation.

Provides 10-100x speedup over pure Python implementations
through JIT compilation and parallel execution.
"""

from .numba_indicators import (
    BollingerResult,
    CombinedSignals,
    MACDResult,
    # ATR
    compute_atr,
    compute_atr_batch,
    # Bollinger Bands
    compute_bollinger_bands,
    compute_bollinger_batch,
    # Combined signals
    compute_combined_signals,
    # EMA/SMA utilities
    compute_ema,
    # MACD
    compute_macd,
    compute_macd_batch,
    compute_multi_level_ofi,
    compute_ofi_batch,
    # Order Flow
    compute_order_flow_imbalance,
    compute_parkinson_volatility,
    # Volatility
    compute_realized_volatility,
    # RSI
    compute_rsi,
    compute_rsi_batch,
    compute_sma,
)

__all__ = [
    # RSI
    "compute_rsi",
    "compute_rsi_batch",
    # MACD
    "compute_macd",
    "compute_macd_batch",
    "MACDResult",
    # Bollinger Bands
    "compute_bollinger_bands",
    "compute_bollinger_batch",
    "BollingerResult",
    # ATR
    "compute_atr",
    "compute_atr_batch",
    # Order Flow
    "compute_order_flow_imbalance",
    "compute_ofi_batch",
    "compute_multi_level_ofi",
    # EMA/SMA utilities
    "compute_ema",
    "compute_sma",
    # Volatility
    "compute_realized_volatility",
    "compute_parkinson_volatility",
    # Combined signals
    "compute_combined_signals",
    "CombinedSignals",
]
