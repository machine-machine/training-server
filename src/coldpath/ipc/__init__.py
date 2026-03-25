"""
IPC module for ColdPath-HotPath communication.

Provides:
- HotPathClient: Unix socket client for reliable JSON-based IPC
- CandidateEvent: Token candidate events from HotPath
- TradingSignal: Signals sent to HotPath for execution
- TradeOutcome: Results of executed trades
- ColdPathJobsServicerImpl: gRPC servicer for HotPath -> ColdPath calls
- AnnouncementProcessorServicerImpl: gRPC servicer for announcement processing
"""

from .coldpath_servicer import (
    AnnouncementProcessorServicerImpl,
    ColdPathJobsServicerImpl,
)
from .hotpath_client import (
    DEFAULT_SOCKET_PATH,
    CandidateEvent,
    HotPathClient,
    MockHotPathClient,
    TradeOutcome,
    TradingSignal,
)

__all__ = [
    "HotPathClient",
    "MockHotPathClient",
    "CandidateEvent",
    "TradingSignal",
    "TradeOutcome",
    "DEFAULT_SOCKET_PATH",
    "ColdPathJobsServicerImpl",
    "AnnouncementProcessorServicerImpl",
]
