"""
Nightly quant research pipeline.

What it does (safe defaults):
- Pulls OHLCV for a symbol basket from preferred vendors (Polygon → Alpaca → Yahoo → mock).
- Computes lightweight factors (returns, volatility, momentum, drawdown).
- Builds a covariance matrix + risk-parity weights using riskfolio-lib.
- Writes parquet artifacts under `artifacts/nightly/<YYYYMMDD>/`.

Usage:
    python -m coldpath.pipeline.nightly_quant

Environment knobs:
    NIGHTLY_SYMBOLS   comma-separated tickers (default: "AAPL,MSFT,SPY,QQQ,EURUSD")
    NIGHTLY_LOOKBACK_DAYS  lookback window (default: 120)
    NIGHTLY_GRANULARITY    bar size for pulls (default: "1Day")
    POLYGON_API_KEY / ALPACA_API_KEY / ALPACA_API_SECRET / COVALENT_API_KEY

This is intentionally lightweight so it can run on a laptop or CI box.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

import polars as pl

try:
    import riskfolio_lib as rpl  # type: ignore
except Exception:  # noqa: BLE001
    rpl = None

try:
    import vectorbt as vbt  # type: ignore
except Exception:  # noqa: BLE001
    vbt = None

from coldpath.data.equity_provider import MarketDataRouter, ProviderSettings


def _default_symbols() -> list[str]:
    env = os.getenv("NIGHTLY_SYMBOLS")
    if env:
        return [s.strip().upper() for s in env.split(",") if s.strip()]
    # A small, liquid basket spanning equities + FX so the risk model has diversity
    return ["AAPL", "MSFT", "SPY", "QQQ", "EURUSD=X"]


def _compute_factors(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    df = df.sort("timestamp")
    df = df.with_columns(
        pl.col("close").pct_change().alias("ret"),
    )
    df = df.with_columns(
        pl.col("ret").rolling_std(window_size=20, min_periods=5).alias("vol20"),
        pl.col("ret").rolling_mean(window_size=10, min_periods=5).alias("mom10"),
    )
    # Max drawdown using running max of close
    df = df.with_columns(
        (pl.col("close") / pl.col("close").cum_max() - 1).alias("drawdown"),
    )
    return df


def _risk_parity_weights(price_frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return risk-parity weights; falls back to equal weights if lib missing."""
    symbols = list(price_frames.keys())
    if not symbols:
        return pl.DataFrame()

    # Build daily returns matrix
    returns = {}
    for sym, df in price_frames.items():
        if df.is_empty():
            continue
        ret = df.sort("timestamp").select(pl.col("close").pct_change().alias(sym))
        returns[sym] = ret

    if not returns:
        return pl.DataFrame()

    # Align on index
    joined = None
    for sym, series_df in returns.items():
        if joined is None:
            joined = series_df.rename({sym: "ret"}).with_columns(
                pl.Series("symbol", [sym] * series_df.height)
            )
        else:
            joined = joined.with_columns(series_df[sym])

    # Flatten to Pandas for riskfolio which expects pandas objects
    pdf = pl.concat(list(returns.values()), how="horizontal").to_pandas()

    if rpl is None:
        weights = {sym: 1 / len(returns) for sym in returns}
        return pl.DataFrame({"symbol": list(weights.keys()), "weight": list(weights.values())})

    try:
        port = rpl.Portfolio(returns=pdf)
        port.assets_stats(method_mu="hist", method_cov="hist")
        w = port.rp_optimization(model="Classic", b=None)
        weights = w.iloc[:, 0].to_dict()
        return pl.DataFrame({"symbol": list(weights.keys()), "weight": list(weights.values())})
    except Exception as exc:  # noqa: BLE001
        weights = {sym: 1 / len(returns) for sym in returns}
        return pl.DataFrame(
            {
                "symbol": list(weights.keys()),
                "weight": list(weights.values()),
                "error": [str(exc)] * len(weights),
            }
        )


async def _pull_prices(
    symbols: Iterable[str],
    start: datetime,
    end: datetime,
    granularity: str,
) -> dict[str, pl.DataFrame]:
    settings = ProviderSettings()
    router = MarketDataRouter(settings)
    try:
        out: dict[str, pl.DataFrame] = {}
        for sym in symbols:
            df = await router.get_ohlcv(sym, start, end, granularity)
            out[sym] = df
        return out
    finally:
        await router.close()


async def run_nightly_quant() -> Path:
    symbols = _default_symbols()
    lookback_days = int(os.getenv("NIGHTLY_LOOKBACK_DAYS", "120"))
    granularity = os.getenv("NIGHTLY_GRANULARITY", "1Day")

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=lookback_days)

    price_frames = await _pull_prices(symbols, start, end, granularity)

    # Compute factors
    factor_frames: dict[str, pl.DataFrame] = {
        sym: _compute_factors(df) for sym, df in price_frames.items()
    }

    # Risk model
    weights_df = _risk_parity_weights(price_frames)

    # Persist artifacts
    run_date = end.strftime("%Y%m%d")
    out_dir = Path("artifacts") / "nightly" / run_date
    out_dir.mkdir(parents=True, exist_ok=True)

    for sym, df in factor_frames.items():
        df.write_parquet(out_dir / f"{sym.replace('/', '_')}.parquet")

    if not weights_df.is_empty():
        weights_df.write_parquet(out_dir / "portfolio_weights.parquet")

    # Save a small summary for dashboards
    summary_rows = []
    for sym, df in factor_frames.items():
        if df.is_empty():
            continue
        last = df.sort("timestamp").tail(1)
        summary_rows.append(
            {
                "symbol": sym,
                "close": float(last["close"][0]),
                "ret": float(last["ret"][0]),
                "vol20": float(last["vol20"][0]),
                "drawdown": float(last["drawdown"][0]),
            }
        )
    if summary_rows:
        pl.DataFrame(summary_rows).write_parquet(out_dir / "summary.parquet")

    return out_dir


def main():
    out_dir = asyncio.run(run_nightly_quant())
    print(f"Nightly quant artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
