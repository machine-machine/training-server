"""
Numba JIT-compiled technical indicators for fast signal generation.

All functions are designed for vectorized operation on price matrices
with shape (n_tokens, n_timepoints) for parallel backtesting.

Features:
- @numba.njit(parallel=True) for 10-100x speedup
- Fallback pure Python implementations when Numba unavailable
- Support for both single-series and batch operations
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

try:
    from numba import njit, prange, float64, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MACDResult:
    """MACD calculation result."""
    macd_line: np.ndarray      # MACD line (fast EMA - slow EMA)
    signal_line: np.ndarray    # Signal line (EMA of MACD)
    histogram: np.ndarray      # MACD - Signal


@dataclass
class BollingerResult:
    """Bollinger Bands calculation result."""
    upper: np.ndarray          # Upper band
    middle: np.ndarray         # Middle band (SMA)
    lower: np.ndarray          # Lower band
    percent_b: np.ndarray      # %B indicator


@dataclass
class CombinedSignals:
    """Combined indicator signals."""
    rsi: np.ndarray
    macd_histogram: np.ndarray
    bollinger_pct_b: np.ndarray
    atr: np.ndarray
    ofi: np.ndarray


# =============================================================================
# Core Numba-Accelerated Functions
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _ema_core(data: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA using Wilder's smoothing."""
        n = len(data)
        result = np.empty(n, dtype=np.float64)
        result[:period] = np.nan

        # Initialize with SMA
        sma_sum = 0.0
        for i in range(period):
            sma_sum += data[i]
        result[period - 1] = sma_sum / period

        # EMA calculation
        multiplier = 2.0 / (period + 1)
        for i in range(period, n):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]

        return result

    @njit(cache=True)
    def _sma_core(data: np.ndarray, period: int) -> np.ndarray:
        """Compute Simple Moving Average."""
        n = len(data)
        result = np.empty(n, dtype=np.float64)
        result[:period - 1] = np.nan

        # Rolling sum
        rolling_sum = 0.0
        for i in range(period):
            rolling_sum += data[i]
        result[period - 1] = rolling_sum / period

        for i in range(period, n):
            rolling_sum = rolling_sum - data[i - period] + data[i]
            result[i] = rolling_sum / period

        return result

    @njit(cache=True)
    def _rsi_core(prices: np.ndarray, period: int) -> np.ndarray:
        """Compute RSI for a single price series."""
        n = len(prices)
        result = np.empty(n, dtype=np.float64)
        result[:period] = np.nan

        # Calculate price changes
        changes = np.empty(n - 1, dtype=np.float64)
        for i in range(n - 1):
            changes[i] = prices[i + 1] - prices[i]

        # Initialize average gains/losses
        avg_gain = 0.0
        avg_loss = 0.0
        for i in range(period):
            if changes[i] > 0:
                avg_gain += changes[i]
            else:
                avg_loss -= changes[i]
        avg_gain /= period
        avg_loss /= period

        # First RSI value
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            result[period] = 100.0 - (100.0 / (1.0 + rs))
        else:
            result[period] = 100.0 if avg_gain > 0 else 50.0

        # Subsequent values using Wilder's smoothing
        for i in range(period, n - 1):
            change = changes[i]
            if change > 0:
                avg_gain = (avg_gain * (period - 1) + change) / period
                avg_loss = avg_loss * (period - 1) / period
            else:
                avg_gain = avg_gain * (period - 1) / period
                avg_loss = (avg_loss * (period - 1) - change) / period

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                result[i + 1] = 100.0 - (100.0 / (1.0 + rs))
            else:
                result[i + 1] = 100.0 if avg_gain > 0 else 50.0

        return result

    @njit(parallel=True, cache=True)
    def _rsi_batch(prices: np.ndarray, period: int) -> np.ndarray:
        """Compute RSI for multiple price series in parallel."""
        n_tokens, n_timepoints = prices.shape
        result = np.empty((n_tokens, n_timepoints), dtype=np.float64)

        for i in prange(n_tokens):
            result[i, :] = _rsi_core(prices[i, :], period)

        return result

    @njit(cache=True)
    def _macd_core(
        prices: np.ndarray,
        fast_period: int,
        slow_period: int,
        signal_period: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute MACD for a single price series."""
        fast_ema = _ema_core(prices, fast_period)
        slow_ema = _ema_core(prices, slow_period)

        macd_line = fast_ema - slow_ema
        signal_line = _ema_core(macd_line, signal_period)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @njit(parallel=True, cache=True)
    def _macd_batch(
        prices: np.ndarray,
        fast_period: int,
        slow_period: int,
        signal_period: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute MACD for multiple price series in parallel."""
        n_tokens, n_timepoints = prices.shape
        macd_lines = np.empty((n_tokens, n_timepoints), dtype=np.float64)
        signal_lines = np.empty((n_tokens, n_timepoints), dtype=np.float64)
        histograms = np.empty((n_tokens, n_timepoints), dtype=np.float64)

        for i in prange(n_tokens):
            ml, sl, hist = _macd_core(
                prices[i, :], fast_period, slow_period, signal_period
            )
            macd_lines[i, :] = ml
            signal_lines[i, :] = sl
            histograms[i, :] = hist

        return macd_lines, signal_lines, histograms

    @njit(cache=True)
    def _bollinger_core(
        prices: np.ndarray,
        period: int,
        num_std: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Bollinger Bands for a single price series."""
        n = len(prices)
        middle = _sma_core(prices, period)

        upper = np.empty(n, dtype=np.float64)
        lower = np.empty(n, dtype=np.float64)
        pct_b = np.empty(n, dtype=np.float64)

        upper[:period - 1] = np.nan
        lower[:period - 1] = np.nan
        pct_b[:period - 1] = np.nan

        for i in range(period - 1, n):
            # Calculate rolling std
            window_sum = 0.0
            window_sq_sum = 0.0
            for j in range(i - period + 1, i + 1):
                window_sum += prices[j]
                window_sq_sum += prices[j] * prices[j]

            mean = window_sum / period
            variance = (window_sq_sum / period) - (mean * mean)
            std = np.sqrt(max(0, variance))

            upper[i] = middle[i] + num_std * std
            lower[i] = middle[i] - num_std * std

            # %B = (Price - Lower) / (Upper - Lower)
            band_width = upper[i] - lower[i]
            if band_width > 0:
                pct_b[i] = (prices[i] - lower[i]) / band_width
            else:
                pct_b[i] = 0.5

        return upper, middle, lower, pct_b

    @njit(parallel=True, cache=True)
    def _bollinger_batch(
        prices: np.ndarray,
        period: int,
        num_std: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Bollinger Bands for multiple price series in parallel."""
        n_tokens, n_timepoints = prices.shape
        upper = np.empty((n_tokens, n_timepoints), dtype=np.float64)
        middle = np.empty((n_tokens, n_timepoints), dtype=np.float64)
        lower = np.empty((n_tokens, n_timepoints), dtype=np.float64)
        pct_b = np.empty((n_tokens, n_timepoints), dtype=np.float64)

        for i in prange(n_tokens):
            u, m, l, p = _bollinger_core(prices[i, :], period, num_std)
            upper[i, :] = u
            middle[i, :] = m
            lower[i, :] = l
            pct_b[i, :] = p

        return upper, middle, lower, pct_b

    @njit(cache=True)
    def _atr_core(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """Compute Average True Range for a single series."""
        n = len(close)
        result = np.empty(n, dtype=np.float64)
        result[:period] = np.nan

        # Calculate True Range
        tr = np.empty(n, dtype=np.float64)
        tr[0] = high[0] - low[0]

        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        # Initial ATR (SMA of TR)
        atr_sum = 0.0
        for i in range(period):
            atr_sum += tr[i]
        result[period - 1] = atr_sum / period

        # Subsequent ATR using Wilder's smoothing
        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

        return result

    @njit(parallel=True, cache=True)
    def _atr_batch(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """Compute ATR for multiple series in parallel."""
        n_tokens, n_timepoints = close.shape
        result = np.empty((n_tokens, n_timepoints), dtype=np.float64)

        for i in prange(n_tokens):
            result[i, :] = _atr_core(high[i, :], low[i, :], close[i, :], period)

        return result

    @njit(cache=True)
    def _ofi_core(
        buy_volume: np.ndarray,
        sell_volume: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """Compute Order Flow Imbalance for a single series."""
        n = len(buy_volume)
        result = np.empty(n, dtype=np.float64)
        result[:period - 1] = np.nan

        for i in range(period - 1, n):
            sum_buy = 0.0
            sum_sell = 0.0
            for j in range(i - period + 1, i + 1):
                sum_buy += buy_volume[j]
                sum_sell += sell_volume[j]

            total = sum_buy + sum_sell
            if total > 0:
                result[i] = (sum_buy - sum_sell) / total
            else:
                result[i] = 0.0

        return result

    @njit(parallel=True, cache=True)
    def _ofi_batch(
        buy_volume: np.ndarray,
        sell_volume: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """Compute OFI for multiple series in parallel."""
        n_tokens, n_timepoints = buy_volume.shape
        result = np.empty((n_tokens, n_timepoints), dtype=np.float64)

        for i in prange(n_tokens):
            result[i, :] = _ofi_core(buy_volume[i, :], sell_volume[i, :], period)

        return result

    @njit(cache=True)
    def _multi_level_ofi(
        bid_volumes: np.ndarray,  # (n_levels, n_timepoints)
        ask_volumes: np.ndarray,  # (n_levels, n_timepoints)
        weights: np.ndarray,      # (n_levels,) - level weights
    ) -> np.ndarray:
        """Compute Multi-Level Order Flow Imbalance.

        Considers multiple price levels with distance-weighted contributions.
        """
        n_levels, n_timepoints = bid_volumes.shape
        result = np.empty(n_timepoints, dtype=np.float64)

        for t in range(n_timepoints):
            weighted_bid = 0.0
            weighted_ask = 0.0
            total_weight = 0.0

            for level in range(n_levels):
                w = weights[level]
                weighted_bid += w * bid_volumes[level, t]
                weighted_ask += w * ask_volumes[level, t]
                total_weight += w

            if total_weight > 0:
                weighted_bid /= total_weight
                weighted_ask /= total_weight

            total = weighted_bid + weighted_ask
            if total > 0:
                result[t] = (weighted_bid - weighted_ask) / total
            else:
                result[t] = 0.0

        return result

    @njit(cache=True)
    def _realized_volatility(
        returns: np.ndarray,
        period: int,
        annualization_factor: float,
    ) -> np.ndarray:
        """Compute realized volatility from returns."""
        n = len(returns)
        result = np.empty(n, dtype=np.float64)
        result[:period - 1] = np.nan

        for i in range(period - 1, n):
            sum_sq = 0.0
            for j in range(i - period + 1, i + 1):
                sum_sq += returns[j] * returns[j]
            result[i] = np.sqrt(sum_sq / period) * np.sqrt(annualization_factor)

        return result

    @njit(cache=True)
    def _parkinson_volatility(
        high: np.ndarray,
        low: np.ndarray,
        period: int,
        annualization_factor: float,
    ) -> np.ndarray:
        """Compute Parkinson volatility estimator.

        More efficient than close-to-close volatility as it uses
        high-low range information.
        """
        n = len(high)
        result = np.empty(n, dtype=np.float64)
        result[:period - 1] = np.nan

        factor = 1.0 / (4.0 * np.log(2.0))

        for i in range(period - 1, n):
            sum_sq = 0.0
            for j in range(i - period + 1, i + 1):
                if low[j] > 0:
                    log_hl = np.log(high[j] / low[j])
                    sum_sq += log_hl * log_hl
            result[i] = np.sqrt(factor * sum_sq / period) * np.sqrt(annualization_factor)

        return result

else:
    # Pure Python fallbacks
    def _ema_core(data, period):
        n = len(data)
        result = np.empty(n)
        result[:period] = np.nan
        result[period - 1] = np.mean(data[:period])
        multiplier = 2.0 / (period + 1)
        for i in range(period, n):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    def _sma_core(data, period):
        n = len(data)
        result = np.empty(n)
        result[:period - 1] = np.nan
        for i in range(period - 1, n):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result

    def _rsi_core(prices, period):
        n = len(prices)
        result = np.empty(n)
        result[:period] = np.nan
        changes = np.diff(prices)
        for i in range(period, n):
            window = changes[i - period:i]
            gains = window[window > 0].sum() / period
            losses = -window[window < 0].sum() / period
            if losses > 0:
                rs = gains / losses
                result[i] = 100.0 - (100.0 / (1.0 + rs))
            else:
                result[i] = 100.0 if gains > 0 else 50.0
        return result

    def _rsi_batch(prices, period):
        n_tokens = prices.shape[0]
        return np.array([_rsi_core(prices[i], period) for i in range(n_tokens)])

    def _macd_core(prices, fast_period, slow_period, signal_period):
        fast_ema = _ema_core(prices, fast_period)
        slow_ema = _ema_core(prices, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = _ema_core(macd_line, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _macd_batch(prices, fast_period, slow_period, signal_period):
        n_tokens = prices.shape[0]
        results = [_macd_core(prices[i], fast_period, slow_period, signal_period)
                   for i in range(n_tokens)]
        return (
            np.array([r[0] for r in results]),
            np.array([r[1] for r in results]),
            np.array([r[2] for r in results]),
        )

    def _bollinger_core(prices, period, num_std):
        n = len(prices)
        middle = _sma_core(prices, period)
        upper = np.empty(n)
        lower = np.empty(n)
        pct_b = np.empty(n)
        upper[:period - 1] = np.nan
        lower[:period - 1] = np.nan
        pct_b[:period - 1] = np.nan
        for i in range(period - 1, n):
            std = np.std(prices[i - period + 1:i + 1])
            upper[i] = middle[i] + num_std * std
            lower[i] = middle[i] - num_std * std
            band_width = upper[i] - lower[i]
            pct_b[i] = (prices[i] - lower[i]) / band_width if band_width > 0 else 0.5
        return upper, middle, lower, pct_b

    def _bollinger_batch(prices, period, num_std):
        n_tokens = prices.shape[0]
        results = [_bollinger_core(prices[i], period, num_std) for i in range(n_tokens)]
        return (
            np.array([r[0] for r in results]),
            np.array([r[1] for r in results]),
            np.array([r[2] for r in results]),
            np.array([r[3] for r in results]),
        )

    def _atr_core(high, low, close, period):
        n = len(close)
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        result = np.empty(n)
        result[:period - 1] = np.nan
        result[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
        return result

    def _atr_batch(high, low, close, period):
        n_tokens = high.shape[0]
        return np.array([_atr_core(high[i], low[i], close[i], period) for i in range(n_tokens)])

    def _ofi_core(buy_volume, sell_volume, period):
        n = len(buy_volume)
        result = np.empty(n)
        result[:period - 1] = np.nan
        for i in range(period - 1, n):
            buy_sum = np.sum(buy_volume[i - period + 1:i + 1])
            sell_sum = np.sum(sell_volume[i - period + 1:i + 1])
            total = buy_sum + sell_sum
            result[i] = (buy_sum - sell_sum) / total if total > 0 else 0.0
        return result

    def _ofi_batch(buy_volume, sell_volume, period):
        n_tokens = buy_volume.shape[0]
        return np.array([_ofi_core(buy_volume[i], sell_volume[i], period) for i in range(n_tokens)])

    def _multi_level_ofi(bid_volumes, ask_volumes, weights):
        n_levels, n_timepoints = bid_volumes.shape
        result = np.empty(n_timepoints)
        for t in range(n_timepoints):
            weighted_bid = np.sum(weights * bid_volumes[:, t])
            weighted_ask = np.sum(weights * ask_volumes[:, t])
            total_weight = np.sum(weights)
            if total_weight > 0:
                weighted_bid /= total_weight
                weighted_ask /= total_weight
            total = weighted_bid + weighted_ask
            result[t] = (weighted_bid - weighted_ask) / total if total > 0 else 0.0
        return result

    def _realized_volatility(returns, period, annualization_factor):
        n = len(returns)
        result = np.empty(n)
        result[:period - 1] = np.nan
        for i in range(period - 1, n):
            result[i] = np.sqrt(np.mean(returns[i - period + 1:i + 1]**2)) * np.sqrt(annualization_factor)
        return result

    def _parkinson_volatility(high, low, period, annualization_factor):
        n = len(high)
        result = np.empty(n)
        result[:period - 1] = np.nan
        factor = 1.0 / (4.0 * np.log(2.0))
        for i in range(period - 1, n):
            log_hl = np.log(high[i - period + 1:i + 1] / low[i - period + 1:i + 1])
            result[i] = np.sqrt(factor * np.mean(log_hl**2)) * np.sqrt(annualization_factor)
        return result


# =============================================================================
# Public API Functions
# =============================================================================

def compute_ema(data: np.ndarray, period: int = 20) -> np.ndarray:
    """Compute Exponential Moving Average."""
    return _ema_core(data, period)


def compute_sma(data: np.ndarray, period: int = 20) -> np.ndarray:
    """Compute Simple Moving Average."""
    return _sma_core(data, period)


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Relative Strength Index for a single price series.

    Args:
        prices: 1D array of prices
        period: RSI calculation period (default 14)

    Returns:
        1D array of RSI values [0, 100]
    """
    return _rsi_core(prices, period)


def compute_rsi_batch(
    prices: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Compute RSI for multiple price series in parallel.

    Args:
        prices: 2D array (n_tokens, n_timepoints)
        period: RSI calculation period

    Returns:
        2D array of RSI values
    """
    return _rsi_batch(prices, period)


def compute_macd(
    prices: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> MACDResult:
    """Compute MACD for a single price series.

    Args:
        prices: 1D array of prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period

    Returns:
        MACDResult with macd_line, signal_line, histogram
    """
    macd_line, signal_line, histogram = _macd_core(
        prices, fast_period, slow_period, signal_period
    )
    return MACDResult(
        macd_line=macd_line,
        signal_line=signal_line,
        histogram=histogram,
    )


def compute_macd_batch(
    prices: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> MACDResult:
    """Compute MACD for multiple price series in parallel."""
    macd_lines, signal_lines, histograms = _macd_batch(
        prices, fast_period, slow_period, signal_period
    )
    return MACDResult(
        macd_line=macd_lines,
        signal_line=signal_lines,
        histogram=histograms,
    )


def compute_bollinger_bands(
    prices: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> BollingerResult:
    """Compute Bollinger Bands for a single price series.

    Args:
        prices: 1D array of prices
        period: SMA period for middle band
        num_std: Number of standard deviations for bands

    Returns:
        BollingerResult with upper, middle, lower, percent_b
    """
    upper, middle, lower, pct_b = _bollinger_core(prices, period, num_std)
    return BollingerResult(
        upper=upper,
        middle=middle,
        lower=lower,
        percent_b=pct_b,
    )


def compute_bollinger_batch(
    prices: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> BollingerResult:
    """Compute Bollinger Bands for multiple price series in parallel."""
    upper, middle, lower, pct_b = _bollinger_batch(prices, period, num_std)
    return BollingerResult(
        upper=upper,
        middle=middle,
        lower=lower,
        percent_b=pct_b,
    )


def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Compute Average True Range.

    Args:
        high: 1D array of high prices
        low: 1D array of low prices
        close: 1D array of close prices
        period: ATR period

    Returns:
        1D array of ATR values
    """
    return _atr_core(high, low, close, period)


def compute_atr_batch(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Compute ATR for multiple series in parallel."""
    return _atr_batch(high, low, close, period)


def compute_order_flow_imbalance(
    buy_volume: np.ndarray,
    sell_volume: np.ndarray,
    period: int = 10,
) -> np.ndarray:
    """Compute Order Flow Imbalance.

    OFI = (Buy Volume - Sell Volume) / (Buy Volume + Sell Volume)

    Args:
        buy_volume: 1D array of buy volumes
        sell_volume: 1D array of sell volumes
        period: Rolling window for smoothing

    Returns:
        1D array of OFI values in [-1, 1]
    """
    return _ofi_core(buy_volume, sell_volume, period)


def compute_ofi_batch(
    buy_volume: np.ndarray,
    sell_volume: np.ndarray,
    period: int = 10,
) -> np.ndarray:
    """Compute OFI for multiple series in parallel."""
    return _ofi_batch(buy_volume, sell_volume, period)


def compute_multi_level_ofi(
    bid_volumes: np.ndarray,
    ask_volumes: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute Multi-Level Order Flow Imbalance.

    Considers order book depth with distance-weighted contributions.

    Args:
        bid_volumes: 2D array (n_levels, n_timepoints)
        ask_volumes: 2D array (n_levels, n_timepoints)
        weights: Optional level weights (default: exponential decay)

    Returns:
        1D array of MLOFI values in [-1, 1]
    """
    n_levels = bid_volumes.shape[0]
    if weights is None:
        # Exponential decay weights
        weights = np.exp(-np.arange(n_levels) * 0.5)

    return _multi_level_ofi(bid_volumes, ask_volumes, weights)


def compute_realized_volatility(
    returns: np.ndarray,
    period: int = 20,
    annualization_factor: float = 252.0,
) -> np.ndarray:
    """Compute realized volatility from returns.

    Args:
        returns: 1D array of returns
        period: Rolling window
        annualization_factor: Periods per year (252 for daily, 365*24 for hourly)

    Returns:
        1D array of annualized volatility
    """
    return _realized_volatility(returns, period, annualization_factor)


def compute_parkinson_volatility(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 20,
    annualization_factor: float = 252.0,
) -> np.ndarray:
    """Compute Parkinson volatility estimator.

    More efficient than close-to-close as it uses high-low range.

    Args:
        high: 1D array of high prices
        low: 1D array of low prices
        period: Rolling window
        annualization_factor: Periods per year

    Returns:
        1D array of annualized volatility
    """
    return _parkinson_volatility(high, low, period, annualization_factor)


def compute_combined_signals(
    prices: np.ndarray,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    buy_volume: Optional[np.ndarray] = None,
    sell_volume: Optional[np.ndarray] = None,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    ofi_period: int = 10,
) -> CombinedSignals:
    """Compute all indicators in one pass for efficiency.

    Args:
        prices: 1D or 2D price array
        high: High prices (optional, uses prices if None)
        low: Low prices (optional, uses prices if None)
        buy_volume: Buy volumes for OFI (optional)
        sell_volume: Sell volumes for OFI (optional)
        Various period parameters...

    Returns:
        CombinedSignals with all indicator values
    """
    # Handle 1D vs 2D input
    is_1d = len(prices.shape) == 1
    if is_1d:
        prices = prices.reshape(1, -1)
        if high is not None:
            high = high.reshape(1, -1)
        if low is not None:
            low = low.reshape(1, -1)
        if buy_volume is not None:
            buy_volume = buy_volume.reshape(1, -1)
        if sell_volume is not None:
            sell_volume = sell_volume.reshape(1, -1)

    # Default high/low to prices
    if high is None:
        high = prices
    if low is None:
        low = prices

    # Compute indicators
    rsi = _rsi_batch(prices, rsi_period)
    macd_lines, signal_lines, histograms = _macd_batch(
        prices, macd_fast, macd_slow, macd_signal
    )
    _, _, _, pct_b = _bollinger_batch(prices, bb_period, bb_std)
    atr = _atr_batch(high, low, prices, atr_period)

    # OFI if volumes provided
    if buy_volume is not None and sell_volume is not None:
        ofi = _ofi_batch(buy_volume, sell_volume, ofi_period)
    else:
        ofi = np.zeros_like(prices)

    # Squeeze back to 1D if input was 1D
    if is_1d:
        rsi = rsi[0]
        histograms = histograms[0]
        pct_b = pct_b[0]
        atr = atr[0]
        ofi = ofi[0]

    return CombinedSignals(
        rsi=rsi,
        macd_histogram=histograms,
        bollinger_pct_b=pct_b,
        atr=atr,
        ofi=ofi,
    )
