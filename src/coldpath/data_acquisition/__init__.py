"""
Data Acquisition Module for 2DEXY.

Provides data fetching from multiple free sources.
"""

from .free_datasets import (
    FREE_DATASET_SOURCES,
    BaseDatasetFetcher,
    CoinGeckoFetcher,
    CryptoDataDownloadFetcher,
    DatasetSource,
    HuggingFaceDatasetFetcher,
    UnifiedFreeDatasetFetcher,
    YahooFinanceFetcher,
    demo_free_datasets,
)

__all__ = [
    "BaseDatasetFetcher",
    "CoinGeckoFetcher",
    "CryptoDataDownloadFetcher",
    "DatasetSource",
    "FREE_DATASET_SOURCES",
    "HuggingFaceDatasetFetcher",
    "YahooFinanceFetcher",
    "UnifiedFreeDatasetFetcher",
    "demo_free_datasets",
]
