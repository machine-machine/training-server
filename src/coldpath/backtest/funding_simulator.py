"""
Funding Rate Simulator

Simulates funding rate dynamics for perpetual futures backtesting.
Provides realistic funding rate patterns based on historical market behavior.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FundingSimConfig:
    """Configuration for funding rate simulation."""

    # Base parameters
    base_rate_mean: float = 0.0001  # 0.01% average
    base_rate_std: float = 0.0002  # Standard deviation

    # Mean reversion
    mean_reversion_speed: float = 0.1  # Speed of reversion to mean

    # Market regime parameters
    bull_market_bias: float = 0.0002  # Additional rate in bull markets
    bear_market_bias: float = -0.0001  # Additional rate in bear markets

    # Extreme event parameters
    extreme_event_prob: float = 0.02  # 2% chance of extreme funding
    extreme_rate_multiplier: float = 5.0  # Multiplier for extreme events

    # Seasonality (time of day effects)
    enable_seasonality: bool = True

    # Funding interval in hours
    funding_interval_hours: int = 8


class FundingRateSimulator:
    """
    Simulates funding rates for perpetual futures.

    Models:
    - Mean reversion to long-term average
    - Market regime effects (bull/bear)
    - Random extreme events
    - Correlation with price movements
    """

    def __init__(self, config: FundingSimConfig = None):
        self.config = config or FundingSimConfig()
        self.current_rate: dict[str, float] = {}

    def simulate_funding_rates(
        self,
        price_data: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
        markets: list[str],
    ) -> pd.DataFrame:
        """
        Simulate funding rates for given markets and time period.

        Args:
            price_data: Price data with columns [timestamp, market, close]
            start_time: Start of simulation period
            end_time: End of simulation period
            markets: List of market symbols

        Returns:
            DataFrame with columns [timestamp, market, funding_rate]
        """
        records = []
        current_time = start_time

        while current_time <= end_time:
            for market in markets:
                rate = self._generate_rate(market, current_time, price_data)
                records.append(
                    {
                        "timestamp": current_time,
                        "market": market,
                        "funding_rate": rate,
                    }
                )

            current_time += timedelta(hours=self.config.funding_interval_hours)

        return pd.DataFrame(records)

    def _generate_rate(
        self,
        market: str,
        timestamp: datetime,
        price_data: pd.DataFrame,
    ) -> float:
        """Generate a single funding rate."""
        # Get previous rate or initialize
        prev_rate = self.current_rate.get(market, self.config.base_rate_mean)

        # Mean reversion component
        mean_reversion = self.config.mean_reversion_speed * (self.config.base_rate_mean - prev_rate)

        # Random shock
        shock = np.random.normal(0, self.config.base_rate_std)

        # Market regime bias
        regime_bias = self._get_regime_bias(market, timestamp, price_data)

        # Extreme event check
        extreme_multiplier = 1.0
        if np.random.random() < self.config.extreme_event_prob:
            extreme_multiplier = self.config.extreme_rate_multiplier
            # Direction based on market pressure
            if regime_bias > 0:
                shock = abs(shock) * extreme_multiplier
            else:
                shock = -abs(shock) * extreme_multiplier

        # Seasonality adjustment
        seasonality = 0.0
        if self.config.enable_seasonality:
            seasonality = self._get_seasonality_adjustment(timestamp)

        # Calculate new rate
        new_rate = prev_rate + mean_reversion + shock + regime_bias + seasonality

        # Clamp to reasonable range
        new_rate = np.clip(new_rate, -0.01, 0.01)  # -1% to 1%

        self.current_rate[market] = new_rate
        return new_rate

    def _get_regime_bias(
        self,
        market: str,
        timestamp: datetime,
        price_data: pd.DataFrame,
    ) -> float:
        """Determine market regime and return appropriate bias."""
        # Look at recent price action
        lookback = timedelta(days=7)
        recent_data = price_data[
            (price_data["market"] == market)
            & (price_data["timestamp"] >= timestamp - lookback)
            & (price_data["timestamp"] <= timestamp)
        ]

        if len(recent_data) < 2:
            return 0.0

        # Calculate return over period
        first_price = recent_data.iloc[0]["close"]
        last_price = recent_data.iloc[-1]["close"]
        price_return = (last_price - first_price) / first_price

        # Determine regime
        if price_return > 0.05:  # Bull market (>5% return)
            return self.config.bull_market_bias
        elif price_return < -0.05:  # Bear market (<-5% return)
            return self.config.bear_market_bias
        else:
            return 0.0

    def _get_seasonality_adjustment(self, timestamp: datetime) -> float:
        """Get time-based seasonality adjustment."""
        # Funding tends to be slightly higher during Asian trading hours
        # and lower during US hours (simplified model)
        hour = timestamp.hour

        if 0 <= hour < 8:  # Asian hours
            return 0.00002
        elif 8 <= hour < 16:  # European hours
            return 0.0
        else:  # US hours
            return -0.00001


class HistoricalFundingAnalyzer:
    """
    Analyzes historical funding rate data.

    Provides statistics and patterns for calibrating the simulator.
    """

    def __init__(self, funding_data: pd.DataFrame):
        """
        Initialize with historical funding data.

        Args:
            funding_data: DataFrame with columns [timestamp, market, funding_rate]
        """
        self.data = funding_data

    def get_statistics(self, market: str = None) -> dict:
        """Get funding rate statistics."""
        data = self.data
        if market:
            data = data[data["market"] == market]

        rates = data["funding_rate"]

        return {
            "mean": rates.mean(),
            "std": rates.std(),
            "min": rates.min(),
            "max": rates.max(),
            "median": rates.median(),
            "skew": stats.skew(rates),
            "kurtosis": stats.kurtosis(rates),
            "positive_pct": (rates > 0).mean() * 100,
            "annualized_mean": rates.mean() * 3 * 365 * 100,  # Percentage
        }

    def get_regime_analysis(self, market: str, price_data: pd.DataFrame) -> dict:
        """Analyze funding rates by market regime."""
        merged = pd.merge_asof(
            self.data[self.data["market"] == market].sort_values("timestamp"),
            price_data[price_data["market"] == market].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
        )

        # Calculate 7-day returns
        merged["return_7d"] = merged["close"].pct_change(periods=7)

        # Classify regimes
        bull = merged[merged["return_7d"] > 0.05]
        bear = merged[merged["return_7d"] < -0.05]
        neutral = merged[abs(merged["return_7d"]) <= 0.05]

        return {
            "bull_market_avg_rate": bull["funding_rate"].mean() if len(bull) > 0 else 0,
            "bear_market_avg_rate": bear["funding_rate"].mean() if len(bear) > 0 else 0,
            "neutral_avg_rate": neutral["funding_rate"].mean() if len(neutral) > 0 else 0,
            "bull_periods": len(bull),
            "bear_periods": len(bear),
            "neutral_periods": len(neutral),
        }

    def estimate_simulator_params(self, market: str = None) -> FundingSimConfig:
        """Estimate simulator parameters from historical data."""
        stats = self.get_statistics(market)

        return FundingSimConfig(
            base_rate_mean=stats["mean"],
            base_rate_std=stats["std"],
            mean_reversion_speed=0.1,  # Default, could be estimated from autocorrelation
            extreme_event_prob=self._estimate_extreme_prob(market),
        )

    def _estimate_extreme_prob(self, market: str = None) -> float:
        """Estimate probability of extreme funding events."""
        data = self.data
        if market:
            data = data[data["market"] == market]

        rates = data["funding_rate"]
        threshold = rates.std() * 3  # 3 sigma events

        extreme_count = (abs(rates) > threshold).sum()
        return extreme_count / len(rates)

    def get_autocorrelation(self, market: str, lags: int = 10) -> list[float]:
        """Calculate autocorrelation of funding rates."""
        data = self.data[self.data["market"] == market]["funding_rate"]
        return [data.autocorr(lag=i) for i in range(1, lags + 1)]


def generate_synthetic_funding(
    markets: list[str],
    start_date: datetime,
    end_date: datetime,
    price_data: pd.DataFrame,
    config: FundingSimConfig = None,
) -> pd.DataFrame:
    """
    Convenience function to generate synthetic funding data.

    Args:
        markets: List of market symbols
        start_date: Start of simulation
        end_date: End of simulation
        price_data: Price data for market regime detection
        config: Simulator configuration

    Returns:
        DataFrame with simulated funding rates
    """
    simulator = FundingRateSimulator(config)
    return simulator.simulate_funding_rates(price_data, start_date, end_date, markets)
