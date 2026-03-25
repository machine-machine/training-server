"""
Order Flow Features for microstructure dynamics.

Captures trade flow imbalances, trade size analysis, time & sales patterns,
price impact metrics, and toxicity measures.

Implements 15 features across 5 sub-categories:
- Trade Flow Imbalance (3 features): OFI 1min, OFI 5min, OFI acceleration
- Trade Size Analysis (3 features): buy/sell size ratio, large trade ratio, retail vs whale
- Time & Sales Patterns (3 features): trade velocity, arrival irregularity, consecutive buys
- Price Impact (3 features): Kyle's lambda, realized spread, adverse selection
- Toxicity Metrics (3 features): VPIN, PIN estimate, composite toxicity score

These features complement the core 50-feature unified vector from
``coldpath.learning.feature_engineering`` by adding microstructure signals
derived from raw trade-level data.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OrderFlowFeatures:
    """15 order flow features capturing microstructure dynamics."""

    # Trade Flow Imbalance
    ofi_1min: float = 0.0  # Order Flow Imbalance (buys - sells) / total, 1min
    ofi_5min: float = 0.0  # Order Flow Imbalance, 5min window
    ofi_acceleration: float = 0.0  # Rate of change of OFI

    # Trade Size Analysis
    avg_buy_size_ratio: float = 1.0  # Avg buy size / avg sell size
    large_trade_ratio: float = 0.0  # Trades > 2 std / total trades
    retail_vs_whale_ratio: float = 1.0  # Small trades / large trades

    # Time & Sales Patterns
    trades_per_second: float = 0.0  # Trading velocity
    time_between_trades_std: float = 0.0  # Trade arrival irregularity
    consecutive_buy_run: int = 0  # Max consecutive buys

    # Price Impact
    kyle_lambda: float = 0.0  # Price impact coefficient
    realized_spread: float = 0.0  # Effective spread from fills
    adverse_selection_cost: float = 0.0  # Information content of trades

    # Toxicity Metrics
    vpin: float = 0.0  # Volume-synchronized probability of informed trading
    pin_estimate: float = 0.0  # Probability of informed trading
    flow_toxicity_score: float = 0.0  # Composite toxicity measure


class OrderFlowExtractor:
    """Extract order flow features from trade data.

    Processes a DataFrame of individual trades and computes 15 microstructure
    features grouped into trade flow imbalance, size analysis, time patterns,
    price impact, and toxicity metrics.

    Args:
        vpin_bucket_size: Number of volume buckets for VPIN computation.
        pin_window: Lookback window (in trades) for PIN estimation.
    """

    def __init__(self, vpin_bucket_size: int = 50, pin_window: int = 300):
        self.vpin_bucket_size = vpin_bucket_size
        self.pin_window = pin_window

    def extract(self, trades: pd.DataFrame) -> OrderFlowFeatures:
        """Extract all order flow features from a DataFrame of trades.

        Expected columns: timestamp, price, quantity, side ('buy'/'sell').
        If ``side`` is missing, trades are classified using the Lee-Ready
        algorithm.

        Args:
            trades: DataFrame with at least ``price`` and ``quantity`` columns.

        Returns:
            Populated ``OrderFlowFeatures`` dataclass.
        """
        if trades.empty or len(trades) < 5:
            return OrderFlowFeatures()

        features = OrderFlowFeatures()

        # Classify trades if side not provided
        if "side" not in trades.columns:
            trades = self._classify_trades(trades)

        buys = trades[trades["side"] == "buy"]
        sells = trades[trades["side"] == "sell"]

        # --- Trade Flow Imbalance ---
        features.ofi_1min = self._compute_ofi(trades, window_seconds=60)
        features.ofi_5min = self._compute_ofi(trades, window_seconds=300)
        features.ofi_acceleration = self._compute_ofi_acceleration(trades)

        # --- Trade Size Analysis ---
        features.avg_buy_size_ratio = self._avg_buy_size_ratio(buys, sells)
        features.large_trade_ratio = self._large_trade_ratio(trades)
        features.retail_vs_whale_ratio = self._retail_vs_whale_ratio(trades)

        # --- Time & Sales Patterns ---
        features.trades_per_second = self._trades_per_second(trades)
        features.time_between_trades_std = self._trade_arrival_std(trades)
        features.consecutive_buy_run = self._max_consecutive_buys(trades)

        # --- Price Impact ---
        features.kyle_lambda = self._estimate_kyle_lambda(trades)
        features.realized_spread = self._realized_spread(trades)
        features.adverse_selection_cost = self._adverse_selection(trades)

        # --- Toxicity ---
        features.vpin = self._compute_vpin(trades)
        features.pin_estimate = self._estimate_pin(buys, sells)
        features.flow_toxicity_score = self._composite_toxicity(features)

        return features

    # ------------------------------------------------------------------
    # Trade classification
    # ------------------------------------------------------------------

    def _classify_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Lee-Ready algorithm for trade classification.

        Compares each trade price to the midpoint of the previous and current
        price.  Ties are resolved with the tick test (direction of the last
        non-zero price change).
        """
        df = trades.copy()
        mid_price = (df["price"].shift(1) + df["price"]) / 2
        df["side"] = np.where(df["price"] > mid_price, "buy", "sell")
        # Handle ties with tick test
        tick = df["price"].diff()
        tie_mask = df["price"] == mid_price
        df.loc[tie_mask, "side"] = np.where(tick[tie_mask] > 0, "buy", "sell")
        return df

    # ------------------------------------------------------------------
    # Trade Flow Imbalance
    # ------------------------------------------------------------------

    def _compute_ofi(self, trades: pd.DataFrame, window_seconds: int) -> float:
        """Order Flow Imbalance = (buy_volume - sell_volume) / total_volume."""
        if "timestamp" in trades.columns:
            cutoff = trades["timestamp"].max() - pd.Timedelta(seconds=window_seconds)
            window = trades[trades["timestamp"] >= cutoff]
        else:
            # Use last N trades proportional to window
            n = min(len(trades), max(10, len(trades) * window_seconds // 300))
            window = trades.tail(n)

        buy_vol = window.loc[window["side"] == "buy", "quantity"].sum()
        sell_vol = window.loc[window["side"] == "sell", "quantity"].sum()
        total = buy_vol + sell_vol

        if total == 0:
            return 0.0
        return float((buy_vol - sell_vol) / total)

    def _compute_ofi_acceleration(self, trades: pd.DataFrame) -> float:
        """Rate of change of OFI (OFI_1min - OFI_5min)."""
        ofi_1 = self._compute_ofi(trades, 60)
        ofi_5 = self._compute_ofi(trades, 300)
        return ofi_1 - ofi_5

    # ------------------------------------------------------------------
    # Trade Size Analysis
    # ------------------------------------------------------------------

    def _avg_buy_size_ratio(self, buys: pd.DataFrame, sells: pd.DataFrame) -> float:
        """Average buy trade size / average sell trade size."""
        avg_buy = buys["quantity"].mean() if len(buys) > 0 else 0
        avg_sell = sells["quantity"].mean() if len(sells) > 0 else 1e-10
        return float(min(avg_buy / max(avg_sell, 1e-10), 10.0))

    def _large_trade_ratio(self, trades: pd.DataFrame) -> float:
        """Fraction of trades larger than 2 standard deviations above mean."""
        if len(trades) < 5:
            return 0.0
        threshold = trades["quantity"].mean() + 2 * trades["quantity"].std()
        return float((trades["quantity"] > threshold).mean())

    def _retail_vs_whale_ratio(self, trades: pd.DataFrame) -> float:
        """Ratio of small trades (at or below median) to large trades."""
        if len(trades) < 5:
            return 1.0
        median_size = trades["quantity"].median()
        small = (trades["quantity"] <= median_size).sum()
        large = (trades["quantity"] > median_size).sum()
        return float(min(small / max(large, 1), 10.0))

    # ------------------------------------------------------------------
    # Time & Sales Patterns
    # ------------------------------------------------------------------

    def _trades_per_second(self, trades: pd.DataFrame) -> float:
        """Trading velocity (trades per second)."""
        if len(trades) < 2:
            return 0.0
        if "timestamp" in trades.columns:
            duration = (trades["timestamp"].max() - trades["timestamp"].min()).total_seconds()
        else:
            duration = float(len(trades))  # Assume 1 second per trade as fallback
        return float(len(trades) / max(duration, 1.0))

    def _trade_arrival_std(self, trades: pd.DataFrame) -> float:
        """Standard deviation of time between consecutive trades."""
        if len(trades) < 3 or "timestamp" not in trades.columns:
            return 0.0
        intervals = trades["timestamp"].diff().dt.total_seconds().dropna()
        return float(intervals.std()) if len(intervals) > 1 else 0.0

    def _max_consecutive_buys(self, trades: pd.DataFrame) -> int:
        """Maximum length of consecutive buy run."""
        if len(trades) == 0:
            return 0
        is_buy = (trades["side"] == "buy").astype(int)
        # Group consecutive same-side trades
        groups = (is_buy != is_buy.shift()).cumsum()
        buy_runs = is_buy.groupby(groups).sum()
        return int(buy_runs.max()) if len(buy_runs) > 0 else 0

    # ------------------------------------------------------------------
    # Price Impact
    # ------------------------------------------------------------------

    def _estimate_kyle_lambda(self, trades: pd.DataFrame) -> float:
        """Kyle's Lambda: price impact coefficient.

        Estimated via OLS regression:
            delta_price = lambda * sign(trade) * sqrt(volume)
        """
        if len(trades) < 10:
            return 0.0
        try:
            sign = np.where(trades["side"] == "buy", 1, -1)
            signed_volume = sign * np.sqrt(trades["quantity"].values)
            price_changes = trades["price"].diff().fillna(0).values

            # Simple OLS: lambda = cov(dp, sv) / var(sv)
            sv_centered = signed_volume - signed_volume.mean()
            dp_centered = price_changes - price_changes.mean()
            var_sv = np.var(sv_centered)
            if var_sv < 1e-12:
                return 0.0
            return float(np.sum(dp_centered * sv_centered) / (len(trades) * var_sv))
        except Exception:
            logger.debug("kyle_lambda estimation failed", exc_info=True)
            return 0.0

    def _realized_spread(self, trades: pd.DataFrame) -> float:
        """Effective spread from fills relative to subsequent midprice."""
        if len(trades) < 5:
            return 0.0
        try:
            mid = (trades["price"].shift(1) + trades["price"].shift(-1)) / 2
            sign = np.where(trades["side"] == "buy", 1, -1)
            spread = 2 * sign * (trades["price"] - mid) / mid
            result = spread.dropna().mean()
            return float(result) if np.isfinite(result) else 0.0
        except Exception:
            logger.debug("realized_spread computation failed", exc_info=True)
            return 0.0

    def _adverse_selection(self, trades: pd.DataFrame) -> float:
        """Information content of trades (5-trade forward return by side)."""
        if len(trades) < 10:
            return 0.0
        try:
            forward_ret = trades["price"].pct_change(5).shift(-5)
            sign = np.where(trades["side"] == "buy", 1, -1)
            info_content = (sign * forward_ret).dropna()
            result = info_content.mean()
            return float(result) if np.isfinite(result) else 0.0
        except Exception:
            logger.debug("adverse_selection computation failed", exc_info=True)
            return 0.0

    # ------------------------------------------------------------------
    # Toxicity Metrics
    # ------------------------------------------------------------------

    def _compute_vpin(self, trades: pd.DataFrame) -> float:
        """Volume-Synchronized Probability of Informed Trading.

        Uses volume buckets to measure order flow imbalance.  Each bucket
        accumulates trades until its volume reaches ``total_volume / n_buckets``,
        then the absolute buy/sell imbalance is recorded.
        """
        if len(trades) < self.vpin_bucket_size * 3:
            return 0.0

        buy_volumes: list[float] = []
        sell_volumes: list[float] = []
        bucket_vol = 0.0
        bucket_buy = 0.0
        bucket_sell = 0.0
        n_buckets = max(len(trades) // self.vpin_bucket_size, 1)
        total_vol_per_bucket = trades["quantity"].sum() / n_buckets

        for _, trade in trades.iterrows():
            qty = trade["quantity"]
            if trade["side"] == "buy":
                bucket_buy += qty
            else:
                bucket_sell += qty
            bucket_vol += qty

            if bucket_vol >= total_vol_per_bucket:
                buy_volumes.append(bucket_buy)
                sell_volumes.append(bucket_sell)
                bucket_vol = 0.0
                bucket_buy = 0.0
                bucket_sell = 0.0

        if len(buy_volumes) < 3:
            return 0.0

        buy_arr = np.array(buy_volumes)
        sell_arr = np.array(sell_volumes)
        abs_imbalance = np.abs(buy_arr - sell_arr)
        total_arr = buy_arr + sell_arr
        total_arr = np.where(total_arr == 0, 1, total_arr)

        return float(np.mean(abs_imbalance / total_arr))

    def _estimate_pin(self, buys: pd.DataFrame, sells: pd.DataFrame) -> float:
        """Simplified PIN estimate using buy/sell count imbalance.

        PIN ~ |B - S| / (B + S) as a proxy for the full maximum-likelihood
        Easley-Kiefer-O'Hara model.
        """
        b = len(buys)
        s = len(sells)
        total = b + s
        if total == 0:
            return 0.0
        return float(abs(b - s) / total)

    def _composite_toxicity(self, features: OrderFlowFeatures) -> float:
        """Composite toxicity score combining VPIN, PIN, and adverse selection.

        Weighted average clamped to [0, 1]:
            0.4 * VPIN + 0.3 * PIN + 0.3 * |adverse_selection| (scaled)
        """
        score = (
            0.4 * features.vpin
            + 0.3 * features.pin_estimate
            + 0.3 * min(abs(features.adverse_selection_cost) * 100, 1.0)
        )
        return float(min(max(score, 0.0), 1.0))

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_feature_dict(self, features: OrderFlowFeatures) -> dict[str, float]:
        """Convert ``OrderFlowFeatures`` to a flat dictionary for model input."""
        return {
            "ofi_1min": features.ofi_1min,
            "ofi_5min": features.ofi_5min,
            "ofi_acceleration": features.ofi_acceleration,
            "avg_buy_size_ratio": features.avg_buy_size_ratio,
            "large_trade_ratio": features.large_trade_ratio,
            "retail_vs_whale_ratio": features.retail_vs_whale_ratio,
            "trades_per_second": features.trades_per_second,
            "time_between_trades_std": features.time_between_trades_std,
            "consecutive_buy_run": float(features.consecutive_buy_run),
            "kyle_lambda": features.kyle_lambda,
            "realized_spread": features.realized_spread,
            "adverse_selection_cost": features.adverse_selection_cost,
            "vpin": features.vpin,
            "pin_estimate": features.pin_estimate,
            "flow_toxicity_score": features.flow_toxicity_score,
        }

    @staticmethod
    def feature_names() -> list[str]:
        """Return the ordered list of 15 order flow feature names."""
        return [
            "ofi_1min",
            "ofi_5min",
            "ofi_acceleration",
            "avg_buy_size_ratio",
            "large_trade_ratio",
            "retail_vs_whale_ratio",
            "trades_per_second",
            "time_between_trades_std",
            "consecutive_buy_run",
            "kyle_lambda",
            "realized_spread",
            "adverse_selection_cost",
            "vpin",
            "pin_estimate",
            "flow_toxicity_score",
        ]
