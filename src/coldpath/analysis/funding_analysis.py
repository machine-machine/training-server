"""
Funding Rate Analysis Module

Provides tools for analyzing funding rate patterns, optimizing entry timing,
and calculating carry costs for perpetual futures strategies.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FundingCarryAnalysis:
    """Results from funding carry analysis."""

    market: str
    period_start: datetime
    period_end: datetime
    total_funding_pct: float
    annualized_carry: float
    avg_rate: float
    rate_volatility: float
    positive_rate_pct: float
    max_rate: float
    min_rate: float
    favorable_for_long: bool  # True if shorts pay longs on average


@dataclass
class OptimalEntryWindow:
    """Optimal time window for entry based on funding."""

    market: str
    day_of_week: int  # 0=Monday, 6=Sunday
    hour_of_day: int
    avg_rate_at_window: float
    historical_win_rate: float
    sample_size: int


class FundingAnalyzer:
    """
    Analyzes funding rate patterns and costs.

    Provides:
    - Carry cost calculations
    - Optimal entry timing
    - Funding rate forecasting
    - Strategy impact analysis
    """

    def __init__(self, funding_data: pd.DataFrame):
        """
        Initialize with historical funding data.

        Args:
            funding_data: DataFrame with columns [timestamp, market, funding_rate]
        """
        self.data = funding_data.copy()
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self._preprocess()

    def _preprocess(self):
        """Add derived columns."""
        self.data["hour"] = self.data["timestamp"].dt.hour
        self.data["day_of_week"] = self.data["timestamp"].dt.dayofweek
        self.data["date"] = self.data["timestamp"].dt.date
        self.data["annualized_rate"] = self.data["funding_rate"] * 3 * 365 * 100

    def calculate_carry(
        self,
        market: str,
        start_date: datetime,
        end_date: datetime,
    ) -> FundingCarryAnalysis:
        """
        Calculate funding carry cost for a period.

        Args:
            market: Market symbol
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            FundingCarryAnalysis with carry metrics
        """
        mask = (
            (self.data["market"] == market)
            & (self.data["timestamp"] >= start_date)
            & (self.data["timestamp"] <= end_date)
        )
        period_data = self.data[mask]

        if len(period_data) == 0:
            raise ValueError(f"No data found for {market} in specified period")

        rates = period_data["funding_rate"]

        # Total cumulative funding (as percentage)
        total_funding = rates.sum() * 100

        # Annualized (3 funding periods per day)
        days = (end_date - start_date).days
        if days > 0:
            annualized = (total_funding / days) * 365
        else:
            annualized = 0.0

        return FundingCarryAnalysis(
            market=market,
            period_start=start_date,
            period_end=end_date,
            total_funding_pct=total_funding,
            annualized_carry=annualized,
            avg_rate=rates.mean(),
            rate_volatility=rates.std(),
            positive_rate_pct=(rates > 0).mean() * 100,
            max_rate=rates.max(),
            min_rate=rates.min(),
            favorable_for_long=rates.mean() < 0,  # Negative = shorts pay
        )

    def find_optimal_entry_windows(
        self,
        market: str,
        side: str = "long",
        top_n: int = 5,
    ) -> list[OptimalEntryWindow]:
        """
        Find optimal time windows for entry based on funding patterns.

        Args:
            market: Market symbol
            side: "long" or "short"
            top_n: Number of top windows to return

        Returns:
            List of optimal entry windows sorted by favorability
        """
        market_data = self.data[self.data["market"] == market]

        # Group by day of week and hour
        grouped = (
            market_data.groupby(["day_of_week", "hour"])["funding_rate"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        # For longs, prefer negative funding (shorts pay)
        # For shorts, prefer positive funding (longs pay)
        if side == "long":
            grouped["score"] = -grouped["mean"]  # Lower = better for longs
        else:
            grouped["score"] = grouped["mean"]  # Higher = better for shorts

        # Filter for minimum sample size
        grouped = grouped[grouped["count"] >= 10]

        # Sort and get top windows
        grouped = grouped.sort_values("score", ascending=False).head(top_n)

        windows = []
        for _, row in grouped.iterrows():
            windows.append(
                OptimalEntryWindow(
                    market=market,
                    day_of_week=int(row["day_of_week"]),
                    hour_of_day=int(row["hour"]),
                    avg_rate_at_window=row["mean"],
                    historical_win_rate=0.0,  # Would need price data to calculate
                    sample_size=int(row["count"]),
                )
            )

        return windows

    def calculate_position_cost(
        self,
        market: str,
        side: str,
        position_size_usd: float,
        holding_days: int,
    ) -> dict:
        """
        Calculate expected funding cost for a hypothetical position.

        Args:
            market: Market symbol
            side: "long" or "short"
            position_size_usd: Position notional value
            holding_days: Expected holding period

        Returns:
            Dict with cost projections
        """
        market_data = self.data[self.data["market"] == market]

        # Use recent data for projection
        recent_data = market_data.tail(30 * 3)  # Last 30 days

        avg_rate = recent_data["funding_rate"].mean()
        std_rate = recent_data["funding_rate"].std()

        # Number of funding periods
        funding_periods = holding_days * 3

        # For longs: pay when rate > 0, receive when rate < 0
        # For shorts: opposite
        rate_multiplier = -1 if side == "long" else 1
        expected_cost_per_period = position_size_usd * avg_rate * rate_multiplier

        expected_total = expected_cost_per_period * funding_periods

        # Confidence interval (95%)
        std_total = position_size_usd * std_rate * np.sqrt(funding_periods)
        ci_lower = expected_total - 1.96 * std_total
        ci_upper = expected_total + 1.96 * std_total

        return {
            "expected_cost_usd": expected_total,
            "cost_per_day_usd": expected_cost_per_period * 3,
            "annualized_cost_pct": (expected_cost_per_period * 3 * 365 / position_size_usd) * 100,
            "confidence_interval_95": (ci_lower, ci_upper),
            "avg_funding_rate": avg_rate,
            "rate_volatility": std_rate,
            "is_favorable": expected_total > 0,  # Positive = receiving funding
        }

    def forecast_funding_rate(
        self,
        market: str,
        periods_ahead: int = 3,  # Next 24 hours
        method: str = "arima",
    ) -> list[tuple[datetime, float, float]]:
        """
        Forecast future funding rates.

        Args:
            market: Market symbol
            periods_ahead: Number of funding periods to forecast
            method: Forecasting method ("arima", "moving_avg", "regime")

        Returns:
            List of (timestamp, predicted_rate, confidence) tuples
        """
        market_data = self.data[self.data["market"] == market].sort_values("timestamp")
        rates = market_data["funding_rate"].values
        last_timestamp = market_data["timestamp"].max()

        forecasts = []

        if method == "moving_avg":
            # Simple exponential moving average
            window = min(30, len(rates))
            weights = np.exp(np.linspace(0, 1, window))
            weights /= weights.sum()
            recent_rates = rates[-window:]
            predicted = np.average(recent_rates, weights=weights)

            for i in range(periods_ahead):
                forecast_time = last_timestamp + timedelta(hours=8 * (i + 1))
                confidence = max(0.5, 1.0 - (i * 0.1))  # Decreasing confidence
                forecasts.append((forecast_time, predicted, confidence))

        elif method == "regime":
            # Regime-based forecast
            recent_rates = rates[-10:]
            current_regime = "normal"

            if np.mean(recent_rates) > 0.0005:
                current_regime = "high_positive"
            elif np.mean(recent_rates) < -0.0005:
                current_regime = "high_negative"

            regime_forecasts = {
                "high_positive": 0.0003,
                "high_negative": -0.0003,
                "normal": 0.0001,
            }

            predicted = regime_forecasts[current_regime]

            for i in range(periods_ahead):
                forecast_time = last_timestamp + timedelta(hours=8 * (i + 1))
                confidence = max(0.4, 0.8 - (i * 0.1))
                forecasts.append((forecast_time, predicted, confidence))

        else:  # Default to recent average
            predicted = np.mean(rates[-10:])
            for i in range(periods_ahead):
                forecast_time = last_timestamp + timedelta(hours=8 * (i + 1))
                confidence = max(0.3, 0.7 - (i * 0.1))
                forecasts.append((forecast_time, predicted, confidence))

        return forecasts

    def analyze_funding_spikes(
        self,
        market: str,
        threshold_std: float = 2.0,
    ) -> pd.DataFrame:
        """
        Identify and analyze funding rate spikes.

        Args:
            market: Market symbol
            threshold_std: Standard deviations for spike detection

        Returns:
            DataFrame with spike events and analysis
        """
        market_data = self.data[self.data["market"] == market].copy()

        # Calculate rolling statistics
        market_data["rolling_mean"] = market_data["funding_rate"].rolling(30).mean()
        market_data["rolling_std"] = market_data["funding_rate"].rolling(30).std()

        # Identify spikes
        market_data["z_score"] = (
            market_data["funding_rate"] - market_data["rolling_mean"]
        ) / market_data["rolling_std"]
        market_data["is_spike"] = abs(market_data["z_score"]) > threshold_std

        spikes = market_data[market_data["is_spike"]].copy()

        if len(spikes) > 0:
            spikes["spike_direction"] = np.where(spikes["z_score"] > 0, "positive", "negative")
            spikes["spike_magnitude"] = abs(spikes["z_score"])

        return spikes[
            ["timestamp", "funding_rate", "z_score", "spike_direction", "spike_magnitude"]
        ]

    def compare_markets(
        self,
        markets: list[str],
        period_days: int = 30,
    ) -> pd.DataFrame:
        """
        Compare funding rate characteristics across markets.

        Args:
            markets: List of market symbols to compare
            period_days: Lookback period in days

        Returns:
            DataFrame with market comparison
        """
        cutoff = datetime.now() - timedelta(days=period_days)
        recent_data = self.data[self.data["timestamp"] >= cutoff]

        comparisons = []

        for market in markets:
            market_data = recent_data[recent_data["market"] == market]["funding_rate"]

            if len(market_data) == 0:
                continue

            comparisons.append(
                {
                    "market": market,
                    "avg_rate": market_data.mean(),
                    "std_rate": market_data.std(),
                    "annualized_carry_pct": market_data.mean() * 3 * 365 * 100,
                    "positive_pct": (market_data > 0).mean() * 100,
                    "max_spike": market_data.max(),
                    "min_spike": market_data.min(),
                    "data_points": len(market_data),
                }
            )

        df = pd.DataFrame(comparisons)
        df = df.sort_values("annualized_carry_pct", ascending=False)
        return df

    def get_funding_calendar(
        self,
        market: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Get funding rate calendar for a period.

        Args:
            market: Market symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with daily funding summaries
        """
        mask = (
            (self.data["market"] == market)
            & (self.data["timestamp"] >= start_date)
            & (self.data["timestamp"] <= end_date)
        )
        period_data = self.data[mask].copy()

        daily = (
            period_data.groupby("date")
            .agg(
                {
                    "funding_rate": ["sum", "mean", "std", "count"],
                }
            )
            .reset_index()
        )

        daily.columns = ["date", "daily_total", "avg_rate", "volatility", "periods"]
        daily["daily_total_pct"] = daily["daily_total"] * 100
        daily["annualized"] = daily["avg_rate"] * 3 * 365 * 100

        return daily
