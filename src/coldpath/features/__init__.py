"""
Features module - Specialized feature extractors for ML pipeline.

This module provides domain-specific feature extraction beyond the core
50-feature unified vector:
- order_flow_features: 15 microstructure dynamics features (OFI, VPIN, toxicity)
- graph_features: 8 network and MEV features (centrality, clustering, wash trading)

# Usage
```python
from coldpath.features import OrderFlowFeatures, OrderFlowExtractor
from coldpath.features import GraphFeatures, GraphFeatureExtractor

# Order flow features
of_extractor = OrderFlowExtractor()
of_features = of_extractor.extract(trades_df)
of_dict = of_extractor.to_feature_dict(of_features)

# Graph-based features
gf_extractor = GraphFeatureExtractor()
gf_features = gf_extractor.extract(wallet_graph, holder_data, token_data, mev_data)
gf_dict = gf_extractor.to_feature_dict(gf_features)
```
"""

from .graph_features import (
    GraphFeatureExtractor,
    GraphFeatures,
)
from .order_flow_features import (
    OrderFlowExtractor,
    OrderFlowFeatures,
)

__all__ = [
    # Order Flow
    "OrderFlowFeatures",
    "OrderFlowExtractor",
    # Graph Features
    "GraphFeatures",
    "GraphFeatureExtractor",
]
