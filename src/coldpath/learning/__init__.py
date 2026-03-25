"""
Learning module - Self-learning profitability optimization.

This module implements continuous learning from trading outcomes:
- outcome_tracker: Record all scan outcomes (trades and skips)
- profitability_learner: Train profitability prediction models
- feature_engineering: Extract features for ML training
- model_updater: Export models to Hot Path
- ensemble: Model ensemble with LLM integration
- mutual_information: MI-based feature selection
- proposal_ranker: Thompson Sampling + LinUCB proposal ranking
- advanced_regime: 5-state HMM regime detection
- stacking_ensemble: Two-level stacking meta-learner with regime-aware weighting
"""

from .advanced_regime import (
    AdvancedRegime,
    AdvancedRegimeDetector,
    AdvancedRegimeState,
    RegimeFeatures8D,
    RegimeModelWeights,
)
from .ensemble import (
    EnsembleDecision,
    EnsembleSignal,
    HybridEnsemble,
    LLMInsight,
    ModelEnsemble,
    create_ensemble_from_training,
    create_hybrid_ensemble,
)
from .feature_engineering import FeatureEngineer, FeatureSet
from .model_updater import LearningScheduler, ModelExport, ModelUpdater, OnlineLearningIntegrator
from .mutual_information import (
    MutualInformationCalculator,
    MutualInformationSelector,
    SelectionMethod,
    SelectionResult,
)
from .online_learner import (
    OnlineLearner,
    OnlineLearningConfig,
    OnlineLearningState,
    OutcomeSource,
    TradeOutcome,
)
from .outcome_tracker import OutcomeTracker, ScanOutcome, TradeResult
from .profitability_learner import ProfitabilityLearner, TrainingConfig, TrainingMetrics
from .proposal_ranker import (
    GaussianProcessOptimizer,
    LinUCBContextualBandit,
    ProposalCandidate,
    ProposalRanker,
    RankingResult,
    ThompsonSampler,
)
from .stacking_ensemble import (
    REGIME_WEIGHTS,
    ModelContribution,
    RegimeAwareEnsemble,
    RegimeEnsembleConfig,
    StackingConfig,
    StackingMetaLearner,
    StackingSignal,
    create_fitted_stacking_ensemble,
    create_stacking_ensemble,
)
from .stacking_ensemble import EnsembleDecision as StackingEnsembleDecision

MIMethod = SelectionMethod
MIResult = SelectionResult

__all__ = [
    # Outcome tracking
    "OutcomeTracker",
    "ScanOutcome",
    "TradeResult",
    # Profitability learning
    "ProfitabilityLearner",
    "TrainingConfig",
    "TrainingMetrics",
    # Feature engineering
    "FeatureEngineer",
    "FeatureSet",
    # Model updates
    "ModelUpdater",
    "ModelExport",
    "LearningScheduler",
    "OnlineLearningIntegrator",
    # Ensemble
    "ModelEnsemble",
    "HybridEnsemble",
    "EnsembleDecision",
    "EnsembleSignal",
    "LLMInsight",
    "create_ensemble_from_training",
    "create_hybrid_ensemble",
    # Mutual Information
    "MutualInformationSelector",
    "MutualInformationCalculator",
    "MIMethod",
    "MIResult",
    # Proposal Ranking
    "ProposalRanker",
    "ThompsonSampler",
    "LinUCBContextualBandit",
    "GaussianProcessOptimizer",
    "RankingResult",
    "ProposalCandidate",
    # Advanced Regime Detection
    "AdvancedRegimeDetector",
    "AdvancedRegime",
    "RegimeFeatures8D",
    "AdvancedRegimeState",
    "RegimeModelWeights",
    # Online Learning
    "OnlineLearner",
    "OnlineLearningConfig",
    "OnlineLearningState",
    "TradeOutcome",
    "OutcomeSource",
    # Stacking Ensemble
    "StackingMetaLearner",
    "RegimeAwareEnsemble",
    "StackingConfig",
    "RegimeEnsembleConfig",
    "StackingSignal",
    "StackingEnsembleDecision",
    "ModelContribution",
    "REGIME_WEIGHTS",
    "create_stacking_ensemble",
    "create_fitted_stacking_ensemble",
]
