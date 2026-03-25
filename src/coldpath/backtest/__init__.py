"""
Backtesting modules for Cold Path.

Includes:
- Standard backtesting engine for spot trading
- Perpetual futures backtesting with funding simulation
- Fill simulation and market replay
- Optuna hyperparameter optimization
- Numba-accelerated technical indicators
- Monte Carlo robustness testing
- AI Guided Setup and Auto Mode (Phase 5)
"""

# AI Guided Setup and Auto Mode (Phase 5)
from .ai_backtest_guide import (
    AIBacktestGuide,
    GuidedSetupResult,
    MarketRegime,
    OptimizationGoal,
    ParameterRecommendation,
    RiskTolerance,
    UserProfile,
)
from .auto_backtest import (
    AutoBacktestMode,
    AutoModeConfig,
    AutoModeResult,
    AutoModeSessionManager,
    AutoModeState,
    IterationResult,
    OptimizationTarget,
    StoppingReason,
)
from .data_provider import (
    BitqueryDataProvider,
    DataProviderConfig,
    MarketEvent,
    OHLCVBar,
)
from .dune_provider import (
    DUNE_QUERIES,
    DuneDataProvider,
    DuneProviderConfig,
    DuneQuery,
)
from .funding_simulator import (
    FundingRateSimulator,
    FundingSimConfig,
    HistoricalFundingAnalyzer,
    generate_synthetic_funding,
)

# Technical indicators (Phase 2)
from .indicators import (
    BollingerResult,
    CombinedSignals,
    MACDResult,
    compute_atr,
    compute_atr_batch,
    compute_bollinger_bands,
    compute_bollinger_batch,
    compute_combined_signals,
    compute_ema,
    compute_macd,
    compute_macd_batch,
    compute_multi_level_ofi,
    compute_ofi_batch,
    compute_order_flow_imbalance,
    compute_parkinson_volatility,
    compute_realized_volatility,
    compute_rsi,
    compute_rsi_batch,
    compute_sma,
)

# Market Regime Detection (Phase 5)
from .market_regime import (
    MarketMetrics,
    MarketRegimeDetector,
    RegimeAdjustment,
    RegimeAnalysis,
    RegimeType,
    TrendStrength,
    VolatilityLevel,
    create_metrics_from_market_data,
)
from .mev_replay import (
    MEVAwareBacktester,
    MEVAwareBacktestResult,
    MEVEvent,
    MEVProfile,
    MEVReplayConfig,
    MEVReplayEngine,
    MEVType,
    run_mev_aware_backtest,
)

# Monte Carlo (Phase 3)
from .monte_carlo import (
    MonteCarloConfig,
    MonteCarloEngine,
    MonteCarloResult,
    PerturbationConfig,
    PerturbationResult,
    Perturbator,
    RiskMetrics,
    compute_probability_of_ruin,
    compute_sharpe_confidence_interval,
    compute_var_cvar,
    run_monte_carlo_simulation,
)

# Optimization module (Phase 1)
from .optimization import (
    FULL_SEARCH_SPACE,
    RISK_SEARCH_SPACE,
    SIGNAL_SEARCH_SPACE,
    TRADING_SEARCH_SPACE,
    OptimizationConfig,
    OptimizationResult,
    OptunaOptimizer,
    ParameterDefinition,
    ParameterType,
    SearchSpace,
    WalkForwardCV,
)

# Out-of-Sample Validation (Phase 2 AI Integration)
from .out_of_sample_validator import (
    OutOfSampleReport,
    OutOfSampleValidator,
)
from .perp_engine import (
    BacktestConfig as PerpBacktestConfig,  # Alias for compatibility
)
from .perp_engine import (
    BacktestResult as PerpBacktestResult,  # Alias for compatibility
)
from .perp_engine import (
    PerpBacktestEngine,
    PerpPosition,
)
from .perp_engine import (
    PerpTrade as PerpBacktestTrade,  # Alias for compatibility
)

# Robustness Validation (Phase 2 AI Integration)
from .robustness_validator import (
    ParameterSensitivity,
    RobustnessReport,
    RobustnessValidator,
)

# Smart Orchestrator (Phase 5)
from .smart_orchestrator import (
    LearningMode,
    OptimizationStrategy,
    SmartBacktestOrchestrator,
    SmartOptimizationResult,
    SmartOrchestratorConfig,
    create_smart_orchestrator,
)

# Stress Testing (Phase 2 AI Integration)
from .stress_test_engine import (
    StressScenario,
    StressTestEngine,
    StressTestReport,
    StressTestResult,
)

# Synthetic Scenarios (Phase 4)
from .synthetic_scenarios import (
    DelayedHoneypotScenario,
    FlashCrashScenario,
    GraduatedHoneypotScenario,
    LPRugScenario,
    SandwichAttackScenario,
    ScenarioConfig,
    ScenarioEngine,
    ScenarioInjector,
    ScenarioStats,
    ScenarioType,
    WhaleDumpScenario,
    create_default_scenario_engine,
)
from .vectorized import (
    EnhancedVectorizedBacktester,
    MonteCarloResults,
    SignalConfig,
    VectorizedBacktester,
    VectorizedData,
    VectorizedResults,
    prepare_data_from_ohlcv,
)

__all__ = [
    # Perp Engine
    "PerpBacktestEngine",
    "PerpBacktestConfig",
    "PerpBacktestResult",
    "PerpBacktestTrade",
    "PerpPosition",
    # Funding Simulator
    "FundingRateSimulator",
    "FundingSimConfig",
    "HistoricalFundingAnalyzer",
    "generate_synthetic_funding",
    # Vectorized Backtesting (Numba-accelerated)
    "VectorizedBacktester",
    "EnhancedVectorizedBacktester",
    "VectorizedData",
    "VectorizedResults",
    "SignalConfig",
    "MonteCarloResults",
    "prepare_data_from_ohlcv",
    # Dune Analytics Provider
    "DuneDataProvider",
    "DuneProviderConfig",
    "DuneQuery",
    "DUNE_QUERIES",
    # Bitquery Provider
    "BitqueryDataProvider",
    "DataProviderConfig",
    "MarketEvent",
    "OHLCVBar",
    # MEV Replay Engine
    "MEVReplayEngine",
    "MEVReplayConfig",
    "MEVProfile",
    "MEVEvent",
    "MEVType",
    "MEVAwareBacktester",
    "MEVAwareBacktestResult",
    "run_mev_aware_backtest",
    # Optimization (Phase 1)
    "OptunaOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "WalkForwardCV",
    "SearchSpace",
    "ParameterDefinition",
    "ParameterType",
    "TRADING_SEARCH_SPACE",
    "RISK_SEARCH_SPACE",
    "SIGNAL_SEARCH_SPACE",
    "FULL_SEARCH_SPACE",
    # Technical Indicators (Phase 2)
    "compute_rsi",
    "compute_rsi_batch",
    "compute_macd",
    "compute_macd_batch",
    "compute_bollinger_bands",
    "compute_bollinger_batch",
    "compute_atr",
    "compute_atr_batch",
    "compute_order_flow_imbalance",
    "compute_ofi_batch",
    "compute_multi_level_ofi",
    "compute_ema",
    "compute_sma",
    "compute_realized_volatility",
    "compute_parkinson_volatility",
    "compute_combined_signals",
    "MACDResult",
    "BollingerResult",
    "CombinedSignals",
    # Monte Carlo (Phase 3)
    "MonteCarloEngine",
    "MonteCarloConfig",
    "MonteCarloResult",
    "RiskMetrics",
    "run_monte_carlo_simulation",
    "compute_var_cvar",
    "compute_probability_of_ruin",
    "compute_sharpe_confidence_interval",
    "Perturbator",
    "PerturbationConfig",
    "PerturbationResult",
    # Synthetic Scenarios (Phase 4)
    "ScenarioType",
    "ScenarioConfig",
    "ScenarioStats",
    "ScenarioInjector",
    "DelayedHoneypotScenario",
    "GraduatedHoneypotScenario",
    "SandwichAttackScenario",
    "LPRugScenario",
    "FlashCrashScenario",
    "WhaleDumpScenario",
    "ScenarioEngine",
    "create_default_scenario_engine",
    # Stress Testing (Phase 2 AI Integration)
    "StressTestEngine",
    "StressScenario",
    "StressTestResult",
    "StressTestReport",
    # Robustness Validation (Phase 2 AI Integration)
    "RobustnessValidator",
    "RobustnessReport",
    "ParameterSensitivity",
    # Out-of-Sample Validation (Phase 2 AI Integration)
    "OutOfSampleValidator",
    "OutOfSampleReport",
    # AI Guided Setup (Phase 5)
    "AIBacktestGuide",
    "UserProfile",
    "RiskTolerance",
    "OptimizationGoal",
    "MarketRegime",
    "ParameterRecommendation",
    "GuidedSetupResult",
    # Auto Mode (Phase 5)
    "AutoBacktestMode",
    "AutoModeConfig",
    "AutoModeResult",
    "AutoModeState",
    "StoppingReason",
    "OptimizationTarget",
    "IterationResult",
    "AutoModeSessionManager",
    # Market Regime Detection (Phase 5)
    "MarketRegimeDetector",
    "MarketMetrics",
    "RegimeType",
    "VolatilityLevel",
    "TrendStrength",
    "RegimeAnalysis",
    "RegimeAdjustment",
    "create_metrics_from_market_data",
    # Smart Orchestrator (Phase 5)
    "SmartBacktestOrchestrator",
    "SmartOrchestratorConfig",
    "SmartOptimizationResult",
    "OptimizationStrategy",
    "LearningMode",
    "create_smart_orchestrator",
]
