"""Training modules for Cold Path."""

from .fraud_model import (
    FraudModel,
    TokenFeatures,
    EnhancedTokenFeatures,
    DeployerProfile,
    DeployerTracker,
    BuySellAnalyzer,
    LPMonitor,
    MutualInfoResult,
    SyntheticMethod,
    SyntheticDataResult,
    TrainingResult,
)

from .regime_detector import (
    RegimeDetector,
    MarketRegime,
    MarketSnapshot,
    RegimeClassification,
    RegimeHistory,
    detect_regime,
)

from .synthetic_temporal import (
    TemporalSyntheticGenerator,
    TokenLifecycleType,
    LifecyclePhase,
    LIFECYCLE_TEMPLATES,
    REGIME_ADJUSTMENTS,
    create_temporal_generator,
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
]
