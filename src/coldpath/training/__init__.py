"""Training modules for Cold Path."""

from .fraud_model import (
    BuySellAnalyzer,
    DeployerProfile,
    DeployerTracker,
    EnhancedTokenFeatures,
    FraudModel,
    LPMonitor,
    MutualInfoResult,
    SyntheticDataResult,
    SyntheticMethod,
    TokenFeatures,
    TrainingResult,
)
from .regime_detector import (
    MarketRegime,
    MarketSnapshot,
    RegimeClassification,
    RegimeDetector,
    RegimeHistory,
    detect_regime,
)
from .synthetic_temporal import (
    LIFECYCLE_TEMPLATES,
    REGIME_ADJUSTMENTS,
    LifecyclePhase,
    TemporalSyntheticGenerator,
    TokenLifecycleType,
    create_temporal_generator,
)

# Advanced ML modules - DISABLED to prevent circular import errors
# These modules are imported on-demand when needed, not at package level
# Commenting out due to circular import between:
# training/__init__ → unified_advanced_pipeline → ensemble_trainer
# → learning.ensemble → training.isolation_forest_detector
# from .synthetic_temporal_transformer import (
#     MarketTransformer,
#     TransformerConfig,
#     TransformerTrainer,
#     PositionalEncoding,
#     CausalSelfAttention,
#     TransformerBlock,
# )
# from .synthetic_order_book import (
#     OrderBookSimulator,
#     Order,
#     OrderType,
#     OrderSide,
#     Trade,
#     OrderBookEvent,
#     OrderBookSnapshot,
#     Participant,
#     ParticipantType,
# )
# from .synthetic_mev import (
#     MEVEventGenerator,
#     MEVEvent,
#     MEVEventType,
#     MEVBot,
# )
# from .unified_advanced_pipeline import (
#     AdvancedMLPipeline,
#     PipelineConfig,
# )
from .trading_losses import (
    NumpyTradingLosses,
    TradingMetrics,
    compute_trading_metrics,
    get_trading_loss,
)

TORCH_AVAILABLE = False
try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    pass

if TORCH_AVAILABLE:
    from .trading_losses import (  # noqa: F401
        CombinedTradingLoss,
        CVaRLoss,
        ProfitabilityLoss,
        SharpeLoss,
        SortinoLoss,
    )

__all__ = [
    # Fraud model
    "FraudModel",
    "TokenFeatures",
    "EnhancedTokenFeatures",
    "DeployerProfile",
    "DeployerTracker",
    "BuySellAnalyzer",
    "LPMonitor",
    "MutualInfoResult",
    "SyntheticMethod",
    "SyntheticDataResult",
    "TrainingResult",
    # Regime detection
    "RegimeDetector",
    "MarketRegime",
    "MarketSnapshot",
    "RegimeClassification",
    "RegimeHistory",
    "detect_regime",
    # Temporal synthetic generation
    "TemporalSyntheticGenerator",
    "TokenLifecycleType",
    "LifecyclePhase",
    "LIFECYCLE_TEMPLATES",
    "REGIME_ADJUSTMENTS",
    "create_temporal_generator",
    # Trading-aware loss functions
    "TradingMetrics",
    "compute_trading_metrics",
    "get_trading_loss",
    "NumpyTradingLosses",
    # Advanced ML modules are commented out to prevent circular import
    # They can be imported on-demand when needed:
    # from coldpath.training.synthetic_temporal_transformer import MarketTransformer
    # from coldpath.training.synthetic_order_book import OrderBookSimulator
    # from coldpath.training.synthetic_mev import MEVEventGenerator
    # from coldpath.training.unified_advanced_pipeline import AdvancedMLPipeline
]

if TORCH_AVAILABLE:
    __all__.extend(
        [
            "SharpeLoss",
            "SortinoLoss",
            "CVaRLoss",
            "CombinedTradingLoss",
            "ProfitabilityLoss",
        ]
    )
