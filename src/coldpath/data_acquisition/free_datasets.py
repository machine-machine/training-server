"""
Free ML Training Datasets Integration for 2DEXY.

Integrates multiple free data sources for crypto trading ML:
- HuggingFace Datasets (crypto OHLCV, sentiment)
- Yahoo Finance (via yfinance)
- CoinGecko API
- CryptoDataDownload
- Kaggle datasets (with API)

All sources are free and publicly available.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetSource:
    """Metadata for a dataset source."""

    name: str
    provider: str
    description: str
    url: str
    license: str
    symbols: list[str] = field(default_factory=list)
    date_range: tuple | None = None
    size_mb: float | None = None
    format: str = "unknown"
    requires_auth: bool = False
    auth_method: str | None = None


FREE_DATASET_SOURCES = {
    "huggingface_crypto_ohlcv": DatasetSource(
        name="Cryptocurrency Futures OHLCV Dataset",
        provider="HuggingFace",
        description="1-minute OHLCV data for crypto futures, 104M rows",
        url="https://huggingface.co/datasets/arthurneuron/cryptocurrency-futures-ohlcv-dataset-1m",
        license="MIT",
        symbols=["BTC", "ETH", "SOL", "BNB", "XRP", "ADA"],
        format="parquet",
        requires_auth=False,
    ),
    "huggingface_crypto_sentiment": DatasetSource(
        name="Cryptocurrency Tweets Sentiment",
        provider="HuggingFace",
        description="Sentiment-labeled crypto tweets dataset",
        url="https://huggingface.co/datasets/aaurelions/cryptocurrency-tweets-sentiment",
        license="CC-BY-4.0",
        format="json",
        requires_auth=False,
    ),
    "huggingface_ethereum_data": DatasetSource(
        name="Ethereum Block/Transaction Data",
        provider="HuggingFace / BlockDB",
        description="Raw blocks, transactions, logs, token transfers for EVM chains",
        url="https://huggingface.co/datasets/BlockDB/Raw-Blocks-Ethereum-And-EVM-Cryptocurrency-Data",
        license="MIT",
        symbols=["ETH"],
        format="parquet",
        requires_auth=False,
    ),
    "crypto_data_download": DatasetSource(
        name="CryptoDataDownload",
        provider="CryptoDataDownload",
        description="Historical crypto exchange data (Binance, Coinbase, etc.)",
        url="https://www.cryptodatadownload.com/",
        license="Free for research",
        symbols=["BTC", "ETH", "SOL", "BNB", "LTC", "XRP"],
        format="csv",
        requires_auth=False,
    ),
    "coingecko_free": DatasetSource(
        name="CoinGecko Free API",
        provider="CoinGecko",
        description="Free tier: 10-50 calls/min for price, market data",
        url="https://www.coingecko.com/api",
        license="Free tier available",
        symbols=["ALL"],
        format="json",
        requires_auth=False,
    ),
    "yahoo_finance": DatasetSource(
        name="Yahoo Finance",
        provider="Yahoo",
        description="Free stock/crypto historical data via yfinance",
        url="https://github.com/ranaroussi/yfinance",
        license="Apache 2.0",
        symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
        format="dataframe",
        requires_auth=False,
    ),
}


class BaseDatasetFetcher(ABC):
    """Base class for dataset fetchers."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or os.environ.get("DATA_CACHE_DIR", "data/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, source: str, symbol: str, timeframe: str = "1h") -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{source}_{symbol}_{timeframe}.parquet"

    @abstractmethod
    def list_available(self) -> list[str]:
        """List available symbols/datasets."""
        ...


class HuggingFaceDatasetFetcher(BaseDatasetFetcher):
    """
    Fetcher for HuggingFace datasets.

    Supports datasets from huggingface.co/datasets
    """

    SOURCE_NAME = "huggingface"

    def __init__(self, cache_dir: str | None = None):
        super().__init__(cache_dir)
        self._datasets_module = None

    def _ensure_datasets(self):
        """Lazy import of datasets library."""
        if self._datasets_module is None:
            try:
                from datasets import load_dataset  # type: ignore

                self._datasets_module = load_dataset
            except ImportError:
                raise ImportError("Install huggingface datasets: pip install datasets") from None
        return self._datasets_module

    async def fetch_dataset(
        self, dataset_name: str, split: str = "train", streaming: bool = True, **kwargs
    ) -> pd.DataFrame | Any:
        """
        Fetch a HuggingFace dataset.

        Args:
            dataset_name: Dataset name
                (e.g., "arthurneuron/cryptocurrency-futures-ohlcv-dataset-1m")
            split: Dataset split ("train", "test", "validation")
            streaming: Use streaming for large datasets

        Returns:
            DataFrame or iterable
        """
        load_dataset = self._ensure_datasets()

        logger.info(f"Loading HuggingFace dataset: {dataset_name}")

        try:
            if streaming:
                ds = load_dataset(dataset_name, split=split, streaming=True)
                return ds
            else:
                ds = load_dataset(dataset_name, split=split)
                return ds.to_pandas()
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            raise

    async def fetch_crypto_ohlcv(
        self,
        symbol: str = "BTC",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch cryptocurrency OHLCV data from HuggingFace.

        Dataset: arthurneuron/cryptocurrency-futures-ohlcv-dataset-1m
        Contains 104M rows of 1-minute OHLCV data.
        """
        load_dataset = self._ensure_datasets()

        try:
            ds = load_dataset(
                "arthurneuron/cryptocurrency-futures-ohlcv-dataset-1m",
                split="train",
                streaming=True,
            )

            records = []
            for row in ds:
                if row.get("symbol", "").upper() == symbol.upper():
                    ts = pd.to_datetime(row["timestamp"], unit="ms")

                    if start_date and ts < pd.to_datetime(start_date):
                        continue
                    if end_date and ts > pd.to_datetime(end_date):
                        continue

                    records.append(
                        {
                            "timestamp_ms": row["timestamp"],
                            "datetime": ts,
                            "open": row["open"],
                            "high": row["high"],
                            "low": row["low"],
                            "close": row["close"],
                            "volume": row["volume"],
                            "symbol": symbol,
                        }
                    )

            df = pd.DataFrame(records)
            if not df.empty:
                df = df.sort_values("timestamp_ms").reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch crypto OHLCV: {e}")
            return pd.DataFrame()

    def list_available(self) -> list[str]:
        """List available HuggingFace crypto datasets."""
        return [
            "arthurneuron/cryptocurrency-futures-ohlcv-dataset-1m",
            "SpectralDoor/cryptocurrency-coins-hi-res",
            "aaurelions/cryptocurrency-tweets-sentiment",
            "BlockDB/Raw-Blocks-Ethereum-And-EVM-Cryptocurrency-Data",
            "BlockDB/ERC20-Tokens-Ethereum-And-EVM-Cryptocurrency-Data",
            "BlockDB/Swap-Prints-Ethereum-And-EVM-Cryptocurrency-Data",
        ]


class YahooFinanceFetcher(BaseDatasetFetcher):
    """
    Fetcher for Yahoo Finance data via yfinance.

    Free historical data for stocks and crypto.
    """

    SOURCE_NAME = "yfinance"

    def __init__(self, cache_dir: str | None = None):
        super().__init__(cache_dir)
        self._yf = None

    def _ensure_yfinance(self):
        """Lazy import of yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf  # type: ignore

                self._yf = yf
            except ImportError:
                raise ImportError("Install yfinance: pip install yfinance") from None
        return self._yf

    async def fetch(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1h",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g., "SOL-USD", "BTC-USD")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ("1m", "5m", "15m", "1h", "1d")

        Returns:
            DataFrame with OHLCV data
        """
        yf = self._ensure_yfinance()

        start = start_date or "2023-01-01"
        end = end_date or datetime.now().strftime("%Y-%m-%d")

        cache_file = self._cache_path("yfinance", symbol, interval)

        if cache_file.exists():
            cached = pd.read_parquet(cache_file)
            cached_end = pd.to_datetime(cached["datetime"].max())
            if cached_end >= pd.to_datetime(end) - timedelta(days=1):
                logger.info(f"Using cached data for {symbol}")
                return cached

        logger.info(f"Fetching {symbol} from Yahoo Finance ({start} to {end})")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return df

            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            if "datetime" not in df.columns and "date" in df.columns:
                df = df.rename(columns={"date": "datetime"})

            df["timestamp_ms"] = df["datetime"].astype(np.int64) // 1_000_000
            df["symbol"] = symbol
            df["source"] = "yfinance"

            df.to_parquet(cache_file, index=False)
            logger.info(f"Cached {len(df)} rows to {cache_file}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            raise

    async def fetch_multiple(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1h",
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        results = {}
        for symbol in symbols:
            try:
                df = await self.fetch(symbol, start_date, end_date, interval)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
        return results

    def list_available(self) -> list[str]:
        """List common crypto tickers on Yahoo Finance."""
        return [
            "BTC-USD",
            "ETH-USD",
            "SOL-USD",
            "BNB-USD",
            "XRP-USD",
            "ADA-USD",
            "DOGE-USD",
            "DOT-USD",
            "AVAX-USD",
            "MATIC-USD",
            "LINK-USD",
            "UNI-USD",
        ]


class CoinGeckoFetcher(BaseDatasetFetcher):
    """
    Fetcher for CoinGecko free API.

    Free tier: 10-50 calls/minute
    """

    SOURCE_NAME = "coingecko"
    BASE_URL = "https://api.coingecko.com/api/v3"

    SYMBOL_MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "binancecoin",
        "XRP": "ripple",
        "ADA": "cardano",
        "DOGE": "dogecoin",
        "DOT": "polkadot",
        "AVAX": "avalanche-2",
        "MATIC": "matic-network",
    }

    def __init__(self, cache_dir: str | None = None, api_key: str | None = None):
        super().__init__(cache_dir)
        self.api_key = api_key or os.environ.get("COINGECKO_API_KEY")
        self._aiohttp = None

    def _ensure_aiohttp(self):
        if self._aiohttp is None:
            try:
                import aiohttp  # type: ignore

                self._aiohttp = aiohttp
            except ImportError:
                raise ImportError("Install aiohttp: pip install aiohttp") from None
        return self._aiohttp

    async def fetch(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        days: int = 365,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from CoinGecko.

        Args:
            symbol: Symbol (e.g., "SOL", "BTC")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            days: Number of days if no dates specified
        """
        aiohttp = self._ensure_aiohttp()

        coin_id = self.SYMBOL_MAP.get(symbol.upper(), symbol.lower())

        if start_date and end_date:
            start_ts = int(pd.to_datetime(start_date).timestamp())
            end_ts = int(pd.to_datetime(end_date).timestamp())
            url = f"{self.BASE_URL}/coins/{coin_id}/market_chart/range"
            params = {"vs_currency": "usd", "from": start_ts, "to": end_ts}
        else:
            url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
            params = {"vs_currency": "usd", "days": days}

        if self.api_key:
            params["x_cg_api_key"] = self.api_key

        logger.info(f"Fetching {coin_id} from CoinGecko")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 429:
                        logger.warning("Rate limited by CoinGecko, waiting...")
                        await asyncio.sleep(60)
                        async with session.get(url, params=params) as retry_resp:
                            data = await retry_resp.json()
                    else:
                        data = await resp.json()

            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])

            df = pd.DataFrame(
                {
                    "timestamp_ms": [p[0] for p in prices],
                    "close": [p[1] for p in prices],
                    "volume": [v[1] for v in volumes] if volumes else [0] * len(prices),
                }
            )

            df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
            df["symbol"] = symbol
            df["source"] = "coingecko"

            return df

        except Exception as e:
            logger.error(f"CoinGecko fetch failed: {e}")
            return pd.DataFrame()

    async def get_market_data(self, limit: int = 100) -> pd.DataFrame:
        """Get top coins by market cap."""
        aiohttp = self._ensure_aiohttp()

        url = f"{self.BASE_URL}/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
        }

        if self.api_key:
            params["x_cg_api_key"] = self.api_key

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    data = await resp.json()

            df = pd.DataFrame(data)
            return df

        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return pd.DataFrame()

    def list_available(self) -> list[str]:
        return list(self.SYMBOL_MAP.keys())


class CryptoDataDownloadFetcher(BaseDatasetFetcher):
    """
    Fetcher for CryptoDataDownload (free CSV datasets).

    Sources: Binance, Coinbase, Bitstamp, etc.
    URL: https://www.cryptodatadownload.com/
    """

    SOURCE_NAME = "cryptodatadownload"
    BASE_URL = "https://www.cryptodatadownload.com/cdd"

    EXCHANGES = {
        "binance": {
            "url": "https://www.cryptodatadownload.com/cdd",
            "pairs": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
        },
        "coinbase": {
            "url": "https://www.cryptodatadownload.com/cdd",
            "pairs": ["BTCUSD", "ETHUSD", "SOLUSD"],
        },
    }

    def __init__(self, cache_dir: str | None = None):
        super().__init__(cache_dir)
        self._aiohttp = None

    def _ensure_aiohttp(self):
        if self._aiohttp is None:
            try:
                import aiohttp  # type: ignore

                self._aiohttp = aiohttp
            except ImportError:
                raise ImportError("Install aiohttp: pip install aiohttp") from None
        return self._aiohttp

    async def fetch(
        self, symbol: str, exchange: str = "binance", timeframe: str = "1h", **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data from CryptoDataDownload.

        Note: This scrapes the website. Use responsibly.
        For production, download files manually.
        """
        self._ensure_aiohttp()

        symbol_upper = symbol.upper().replace("-", "")

        cache_file = self._cache_path("cdd", f"{exchange}_{symbol_upper}", timeframe)
        if cache_file.exists():
            logger.info(f"Using cached CDD data for {symbol}")
            return pd.read_parquet(cache_file)

        logger.warning(
            "CryptoDataDownload requires manual download. "
            f"Visit https://www.cryptodatadownload.com/{exchange}.html"
        )

        return pd.DataFrame()

    def list_available(self) -> list[str]:
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "BTCUSD", "ETHUSD"]


class UnifiedFreeDatasetFetcher:
    """
    Unified interface for all free dataset sources.

    Usage:
        fetcher = UnifiedFreeDatasetFetcher()

        # Fetch from Yahoo Finance
        df = await fetcher.fetch("SOL-USD", source="yfinance")

        # Fetch from HuggingFace
        df = await fetcher.fetch_crypto_ohlcv("BTC")

        # Fetch from CoinGecko
        df = await fetcher.fetch("SOL", source="coingecko")

        # List all available sources
        sources = fetcher.list_sources()
    """

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir
        self._fetchers: dict[str, BaseDatasetFetcher] = {}

    def _get_fetcher(self, source: str) -> BaseDatasetFetcher:
        """Get or create a fetcher instance."""
        if source not in self._fetchers:
            fetchers = {
                "yfinance": YahooFinanceFetcher,
                "huggingface": HuggingFaceDatasetFetcher,
                "coingecko": CoinGeckoFetcher,
                "cryptodatadownload": CryptoDataDownloadFetcher,
            }

            if source not in fetchers:
                raise ValueError(f"Unknown source: {source}. Available: {list(fetchers.keys())}")

            self._fetchers[source] = fetchers[source](self.cache_dir)

        return self._fetchers[source]

    async def fetch(
        self,
        symbol: str,
        source: str = "yfinance",
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch data from a specific source.

        Args:
            symbol: Symbol to fetch
            source: Data source ("yfinance", "coingecko", etc.)
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        fetcher = self._get_fetcher(source)
        return await fetcher.fetch(symbol, start_date, end_date, **kwargs)  # type: ignore

    async def fetch_all_sources(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch from all available sources and return results."""
        results = {}

        for source in ["yfinance", "coingecko"]:
            try:
                df = await self.fetch(symbol, source, start_date, end_date)
                if not df.empty:
                    results[source] = df
            except Exception as e:
                logger.warning(f"{source} failed for {symbol}: {e}")

        return results

    def list_sources(self) -> dict[str, DatasetSource]:
        """List all available free dataset sources."""
        return FREE_DATASET_SOURCES

    def list_symbols(self, source: str) -> list[str]:
        """List available symbols for a source."""
        fetcher = self._get_fetcher(source)
        return fetcher.list_available()


async def demo_free_datasets():
    """Demo script showing free dataset usage."""
    fetcher = UnifiedFreeDatasetFetcher(cache_dir="data/cache")

    print("=== Free Dataset Sources ===")
    for name, source in fetcher.list_sources().items():
        print(f"\n{name}:")
        print(f"  Provider: {source.provider}")
        print(f"  Description: {source.description}")
        print(f"  URL: {source.url}")

    print("\n=== Fetching SOL data from Yahoo Finance ===")
    df = await fetcher.fetch("SOL-USD", source="yfinance", interval="1h")
    if not df.empty:
        print(f"Fetched {len(df)} rows")
        print(df.head())

    print("\n=== Available symbols per source ===")
    for source in ["yfinance", "coingecko"]:
        symbols = fetcher.list_symbols(source)
        print(f"{source}: {symbols[:5]}...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_free_datasets())
