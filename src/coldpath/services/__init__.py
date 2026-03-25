"""
ColdPath services module.

Infrastructure components for the ML pipeline:
- Feature Store: Real-time feature serving
- A/B Testing: Model comparison framework
- Telegram Signals: Social signal extraction
- Social Features Service: HotPath integration for social metrics
- Feature Service Server: gRPC server for HotPath feature access
- Monitoring: Prometheus metrics and health checks
"""

from coldpath.services.ab_testing import (
    ABTestingFramework,
    Experiment,
    ExperimentMetrics,
    ExperimentResult,
    ExperimentStatus,
    ModelVariant,
)
from coldpath.services.feature_service_server import (
    FeatureServiceServer,
)
from coldpath.services.feature_store import (
    FEATURE_DEFINITIONS,
    FeatureStore,
    FeatureStoreConfig,
    TokenFeatures,
)
from coldpath.services.monitoring import (
    ServiceMetrics,
    ServicesHealthReport,
    ServicesMonitor,
    create_metrics_endpoint,
)
from coldpath.services.social_features_service import (
    SocialFeatures,
    SocialFeaturesService,
    TokenSocialMapping,
)
from coldpath.services.telegram_signals import (
    TelegramMetrics,
    TelegramSignalCollector,
)

__all__ = [
    # Feature Store
    "FeatureStore",
    "FeatureStoreConfig",
    "TokenFeatures",
    "FEATURE_DEFINITIONS",
    # A/B Testing
    "ABTestingFramework",
    "Experiment",
    "ExperimentMetrics",
    "ExperimentResult",
    "ExperimentStatus",
    "ModelVariant",
    # Telegram
    "TelegramSignalCollector",
    "TelegramMetrics",
    # Social Features
    "SocialFeatures",
    "SocialFeaturesService",
    "TokenSocialMapping",
    # Feature Service Server
    "FeatureServiceServer",
    # Monitoring
    "ServiceMetrics",
    "ServicesHealthReport",
    "ServicesMonitor",
    "create_metrics_endpoint",
]
