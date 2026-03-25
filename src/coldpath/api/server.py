"""
FastAPI server for ColdPath ML Engine.

Provides REST API for:
- AI chat and recommendations (Claude Opus/Sonnet)
- ML proposals (multi-armed bandit)
- Backtesting
- Server administration and self-tuning

Run with: uvicorn coldpath.api.server:app --host 127.0.0.1 --port 8080
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if TYPE_CHECKING:
    from coldpath.ai.approval_workflow import ApprovalWorkflow
    from coldpath.ai.claude_client import ClaudeClient
    from coldpath.ai.claude_router import ClaudeRouter
    from coldpath.ai.drift_detector import DriftDetector
    from coldpath.ai.llm_orchestrator import LLMOrchestrator
    from coldpath.ai.parameter_store import ParameterStore
    from coldpath.ai.parameter_tracker import ParameterEffectivenessTracker
    from coldpath.ai.performance_engine import PerformanceFeedbackEngine
    from coldpath.ai.rollback_manager import RollbackManager
    from coldpath.ai.safety_validator import SafetyValidator
    from coldpath.ai.server_tuner import ServerTuner
    from coldpath.ai.strategy_orchestrator import AIStrategyOrchestrator
    from coldpath.autotrader.coordinator import AutoTraderCoordinator
    from coldpath.backtest.metrics import MetricsEngine
    from coldpath.main import ColdPathEngine
    from coldpath.training.bandit import BanditTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Application state container."""

    claude_client: Optional["ClaudeClient"] = None
    claude_router: Optional["ClaudeRouter"] = None
    llm_orchestrator: Optional["LLMOrchestrator"] = None
    server_tuner: Optional["ServerTuner"] = None
    bandit_trainer: Optional["BanditTrainer"] = None
    metrics_engine: Optional["MetricsEngine"] = None
    autotrader: Optional["AutoTraderCoordinator"] = None
    autotrader_enabled: bool = False
    shutting_down: bool = False
    # Real engine wiring
    engine: Optional["ColdPathEngine"] = None
    engine_task: asyncio.Task | None = None
    # AI optimization components (Phase 1 wiring)
    strategy_orchestrator: Optional["AIStrategyOrchestrator"] = None
    performance_engine_ai: Optional["PerformanceFeedbackEngine"] = None
    drift_detector: Optional["DriftDetector"] = None
    parameter_store: Optional["ParameterStore"] = None
    parameter_tracker: Optional["ParameterEffectivenessTracker"] = None
    safety_validator: Optional["SafetyValidator"] = None
    approval_workflow: Optional["ApprovalWorkflow"] = None
    rollback_manager: Optional["RollbackManager"] = None
    # Feedback loop for continuous model improvement
    feedback_loop: Any | None = None  # FeedbackLoopPipeline
    feedback_scheduler: Any | None = None  # FeedbackLoopScheduler
    # Smart backtest orchestrator (Phase 5)
    smart_backtest_orchestrator: Any | None = None  # SmartBacktestOrchestrator
    smart_backtest_orchestrator_config: Any | None = None  # SmartOrchestratorConfig
    backtest_history: Any | None = None  # List for learning from backtests


# Global state
_app_state: AppState | None = None


def get_app_state() -> AppState:
    """Get global application state."""
    global _app_state
    if _app_state is None:
        _app_state = AppState()
    return _app_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    NOTE: Inlined ColdPathEngine so the REST API reflects real engine state
    (DB, HotPath client, AutoTrader, bandit). This keeps the API-backed UI in
    sync with actual trading activity instead of the previous stubbed/coordinator
    instance.
    """
    logger.info("Starting ColdPath API server...")

    state = get_app_state()
    state.shutting_down = False

    # === Start real ColdPath engine (paper/sim defaults) ===
    try:
        from coldpath.main import ColdPathConfig, ColdPathEngine

        engine_config = ColdPathConfig.from_env()
        # Respect AUTOTRADER_ENABLED if explicitly set; otherwise keep defaults
        autotrader_env = os.getenv("AUTOTRADER_ENABLED")
        if autotrader_env is not None:
            engine_config.autotrader_enabled = autotrader_env.lower() == "true"

        state.engine = ColdPathEngine(engine_config)
        state.engine_task = asyncio.create_task(state.engine.run(), name="coldpath-engine")

        # Expose live components to API routes
        state.autotrader = state.engine.autotrader
        state.bandit_trainer = state.engine.bandit_trainer
        state.autotrader_enabled = engine_config.autotrader_enabled

        logger.info(
            "ColdPathEngine started (autotrader_enabled=%s, db=%s)",
            engine_config.autotrader_enabled,
            engine_config.database_path,
        )
    except Exception as e:
        logger.error(f"Failed to start ColdPathEngine: {e}")

    # Initialize server tuner first (manages configuration)
    try:
        from coldpath.ai.server_tuner import (
            ServerTuner,
            TuningMode,
            set_server_tuner,
        )

        config_path = Path(os.environ.get("TUNER_CONFIG_PATH", "/tmp/coldpath_tuner_config.json"))

        tuning_mode = TuningMode(os.environ.get("TUNING_MODE", "moderate"))

        state.server_tuner = ServerTuner(
            mode=tuning_mode,
            config_path=config_path,
        )

        # Load saved config if exists
        await state.server_tuner.load_config()

        # Set as global instance
        set_server_tuner(state.server_tuner)

        # Start auto-tuning
        await state.server_tuner.start()

        logger.info(f"Server tuner initialized in {tuning_mode.value} mode")
    except Exception as e:
        logger.warning(f"Failed to initialize server tuner: {e}")

    # Initialize Claude client if API key available
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            from coldpath.ai.claude_client import ClaudeClient
            from coldpath.ai.claude_router import ClaudeRouter
            from coldpath.ai.llm_orchestrator import LLMOrchestrator, OrchestratorConfig

            # Get timeout from tuner config if available
            timeout = 60.0
            max_retries = 3
            if state.server_tuner:
                config = state.server_tuner.get_config()
                timeout = config.request_timeout.current_value
                max_retries = int(config.max_retries.current_value)

            state.claude_client = ClaudeClient(
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
            state.claude_router = ClaudeRouter(client=state.claude_client)

            # Initialize orchestrator with tuner integration
            orchestrator_config = OrchestratorConfig()
            state.llm_orchestrator = LLMOrchestrator(
                client=state.claude_client,
                router=state.claude_router,
                config=orchestrator_config,
            )
            if state.server_tuner:

                def on_config_change(new_config):
                    if state.llm_orchestrator:
                        state.llm_orchestrator.config.scoring_budget_ms = int(
                            new_config.scoring_budget_ms.current_value
                        )
                        state.llm_orchestrator.config.chat_budget_ms = int(
                            new_config.chat_budget_ms.current_value
                        )
                        state.llm_orchestrator.config.analysis_budget_ms = int(
                            new_config.analysis_budget_ms.current_value
                        )
                        state.llm_orchestrator.config.ml_weight = new_config.ml_weight.current_value
                        state.llm_orchestrator.config.llm_weight = (
                            new_config.llm_weight.current_value
                        )
                        state.llm_orchestrator.config.consensus_threshold = (
                            new_config.consensus_threshold.current_value
                        )
                        logger.info("Orchestrator config updated from tuner")

                state.server_tuner.on_config_change(on_config_change)

            logger.info("Claude AI client and orchestrator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Claude client: {e}")
    else:
        logger.warning("ANTHROPIC_API_KEY not set. AI features will be unavailable.")

    # Initialize bandit trainer
    try:
        from coldpath.training.bandit import BanditTrainer

        state.bandit_trainer = BanditTrainer()
        logger.info("Bandit trainer initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize bandit trainer: {e}")

    # Initialize metrics engine
    try:
        from coldpath.backtest.metrics import MetricsEngine

        state.metrics_engine = MetricsEngine()
        logger.info("Metrics engine initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize metrics engine: {e}")

    # Initialize AI optimization components (dependency order matters)
    try:
        from coldpath.ai.approval_workflow import ApprovalWorkflow
        from coldpath.ai.drift_detector import DriftDetector
        from coldpath.ai.parameter_store import ParameterStore
        from coldpath.ai.parameter_tracker import ParameterEffectivenessTracker
        from coldpath.ai.performance_engine import PerformanceFeedbackEngine
        from coldpath.ai.rollback_manager import RollbackManager
        from coldpath.ai.safety_validator import SafetyValidator
        from coldpath.ai.strategy_orchestrator import AIStrategyOrchestrator

        # 1. ParameterStore (foundation — stores versioned parameters)
        state.parameter_store = ParameterStore(
            persist_path="/tmp/coldpath_params.json",
        )

        # 2. DriftDetector (detects performance degradation)
        state.drift_detector = DriftDetector()

        # 3. PerformanceFeedbackEngine (records trade outcomes)
        state.parameter_tracker = ParameterEffectivenessTracker()
        state.performance_engine_ai = PerformanceFeedbackEngine(
            parameter_store=state.parameter_store,
            drift_detector=state.drift_detector,
            parameter_tracker=state.parameter_tracker,
        )
        await state.performance_engine_ai.initialize()

        # 4. SafetyValidator (validates proposed parameter changes)
        state.safety_validator = SafetyValidator()

        # 5. ApprovalWorkflow (auto-apply vs manual approval)
        trading_mode = os.environ.get("TRADING_MODE", "paper")
        state.approval_workflow = ApprovalWorkflow(
            trading_mode=trading_mode,
        )

        # 6. RollbackManager (monitors post-deployment, auto-rolls back)
        state.rollback_manager = RollbackManager(
            parameter_store=state.parameter_store,
            performance_engine=state.performance_engine_ai,
        )

        # 7. AIStrategyOrchestrator (central state machine — last, takes all deps)
        state.strategy_orchestrator = AIStrategyOrchestrator(
            performance_engine=state.performance_engine_ai,
            drift_detector=state.drift_detector,
            safety_validator=state.safety_validator,
            approval_workflow=state.approval_workflow,
            rollback_manager=state.rollback_manager,
            claude_client=state.claude_client,
            backtest_engine=state.metrics_engine,
            parameter_store=state.parameter_store,
            parameter_tracker=state.parameter_tracker,
        )

        # Wire feedback loop: drift detected → trigger optimization
        if not hasattr(state, "_background_tasks"):
            state._background_tasks = []

        def on_optimization_needed(reason: str, details: dict) -> None:
            """Bridge sync callback to async optimization trigger."""
            task = asyncio.create_task(
                state.strategy_orchestrator.trigger_optimization(
                    reason=reason,
                    goal="maximize_sharpe",
                    constraints={"max_daily_loss_sol": 1.0},
                    urgency="high" if details.get("drift_severity") == "severe" else "normal",
                ),
                name="auto-optimization-trigger",
            )
            # Store reference to prevent GC and allow cleanup
            state._background_tasks.append(task)
            # Prune completed tasks
            state._background_tasks = [t for t in state._background_tasks if not t.done()]

        state.performance_engine_ai.on_optimization_needed(on_optimization_needed)

        logger.info(
            "AI optimization components initialized: "
            "orchestrator=%s, performance_engine=%s, drift_detector=%s, "
            "parameter_store=%s, safety_validator=%s, approval_workflow=%s, "
            "rollback_manager=%s, parameter_tracker=%s",
            state.strategy_orchestrator is not None,
            state.performance_engine_ai is not None,
            state.drift_detector is not None,
            state.parameter_store is not None,
            state.safety_validator is not None,
            state.approval_workflow is not None,
            state.rollback_manager is not None,
            state.parameter_tracker is not None,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize AI optimization components: {e}")
        logger.exception("AI optimization initialization exception")

    # === Initialize Feedback Loop for Continuous Model Improvement ===
    try:
        from coldpath.api.routes import feedback_loop as feedback_loop_routes
        from coldpath.learning.feedback_loop import (
            EvidenceGates,
            FeedbackLoopPipeline,
            FeedbackLoopScheduler,
        )
        from coldpath.learning.model_updater import ModelUpdater
        from coldpath.training.ensemble_trainer import EnsembleTrainer

        # Get required components from engine if available
        db = state.engine.db if state.engine else None
        trainer = None
        updater = None
        learner = None

        if state.engine is not None:
            trainer = getattr(state.engine, "feedback_trainer", None)
            updater = getattr(state.engine, "feedback_model_updater", None)
            learner = getattr(state.engine, "profitability_learner", None)

            if trainer is None:
                try:
                    trainer = EnsembleTrainer()
                except Exception as trainer_exc:
                    logger.warning(f"Failed to initialize feedback loop trainer: {trainer_exc}")

            if updater is None:
                artifact_store = getattr(state.engine, "artifact_store", None)
                hotpath_client = getattr(state.engine, "hotpath_client", None)
                if artifact_store is not None:
                    try:
                        updater = ModelUpdater(
                            artifact_store=artifact_store,
                            grpc_client=hotpath_client,
                        )
                    except Exception as updater_exc:
                        logger.warning(f"Failed to initialize feedback loop updater: {updater_exc}")

        if db is not None:
            # Create evidence gates with reasonable defaults
            gates = EvidenceGates(
                min_total_samples=200,
                min_traded_samples=50,
                min_win_rate=0.35,
                min_profit_factor=1.2,
                min_sharpe_ratio=0.5,
            )

            # Initialize feedback loop pipeline
            # Note: trainer and updater need to be passed from engine when available
            state.feedback_loop = FeedbackLoopPipeline(
                db=db,
                trainer=trainer,
                updater=updater,
                learner=learner,
                gates=gates,
            )

            # Set global reference for API routes
            feedback_loop_routes.set_feedback_loop(state.feedback_loop)

            # Start the feedback loop scheduler
            state.feedback_scheduler = FeedbackLoopScheduler(
                pipeline=state.feedback_loop,
                min_interval_hours=6.0,
                min_new_outcomes=50,
            )

            # Create and store the task reference
            scheduler_task = asyncio.create_task(
                state.feedback_scheduler.start(), name="feedback-loop-scheduler"
            )
            state.feedback_scheduler.set_task(scheduler_task)

            # Set global reference for restart endpoint
            feedback_loop_routes.set_feedback_scheduler(state.feedback_scheduler)

            logger.info(
                "Feedback loop pipeline initialized and scheduler started "
                "(trainer=%s, updater=%s, learner=%s)",
                trainer is not None,
                updater is not None,
                learner is not None,
            )
        else:
            logger.warning("Database not available, skipping feedback loop initialization")

    except Exception as e:
        logger.warning(f"Failed to initialize feedback loop: {e}")
        logger.exception("Feedback loop initialization exception")

    # Initialize Smart Backtest Orchestrator (Phase 5)
    try:
        from coldpath.backtest.smart_orchestrator import (
            LearningMode,
            SmartOrchestratorConfig,
        )

        # Shared history for learning across backtests
        state.backtest_history = []

        config = SmartOrchestratorConfig(
            learning_mode=LearningMode.ACTIVE,
            enable_regime_detection=True,
        )

        # Will be connected to actual backtest runner when needed
        state.smart_backtest_orchestrator_config = config

        logger.info("Smart backtest orchestrator configured")

    except Exception as e:
        logger.warning(f"Failed to initialize smart backtest orchestrator: {e}")

    # Initialize standalone AutoTrader only if the engine failed to start
    if state.engine is None:
        autotrader_enabled = os.environ.get("AUTOTRADER_ENABLED", "true").lower() == "true"
        state.autotrader_enabled = autotrader_enabled

        if autotrader_enabled:
            try:
                from coldpath.autotrader.coordinator import AutoTraderConfig, AutoTraderCoordinator

                # Initialize AutoTrader with bandit trainer (minimal setup for API server)
                autotrader_config = AutoTraderConfig()

                state.autotrader = AutoTraderCoordinator(
                    config=autotrader_config,
                    bandit=state.bandit_trainer,
                    fraud_model=None,  # Will be loaded if available
                    bias_calibrator=None,
                    db=None,  # No database for API server mode
                    outcome_tracker=None,  # No outcome tracking in API server mode
                )

                # Start the AutoTrader
                await state.autotrader.start()
                logger.info(
                    f"AutoTrader initialized and started (state: {state.autotrader.state.value})"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize AutoTrader: {e}")
                logger.exception("AutoTrader initialization exception")
        else:
            logger.info("AutoTrader disabled (set AUTOTRADER_ENABLED=true to enable)")

    logger.info("ColdPath API server started")
    try:
        yield
    finally:
        state.shutting_down = True
        logger.info("Shutting down ColdPath API server...")

        # 1. Stop tuner first (prevents config updates during shutdown)
        if state.server_tuner:
            await state.server_tuner.stop()

        # 2. Stop engine to flush state and DB
        if state.engine:
            try:
                await asyncio.wait_for(state.engine.shutdown(), timeout=60.0)
            except TimeoutError:
                logger.warning("ColdPathEngine shutdown timed out after 60s")
            except Exception as e:
                logger.warning(f"Error stopping ColdPathEngine: {e}")
            finally:
                if state.engine_task and not state.engine_task.done():
                    state.engine_task.cancel()
                    try:
                        await asyncio.wait_for(state.engine_task, timeout=5.0)
                    except (TimeoutError, asyncio.CancelledError):
                        logger.warning("Engine task did not complete after cancellation")
                logger.info("ColdPathEngine stopped")

        # 3. Stop AutoTrader
        if state.autotrader:
            try:
                await state.autotrader.stop()
                logger.info("AutoTrader stopped")
            except Exception as e:
                logger.warning(f"Error stopping AutoTrader: {e}")

        # 4. Cancel active optimization if running
        if state.strategy_orchestrator and state.strategy_orchestrator._optimization_task:
            task = state.strategy_orchestrator._optimization_task
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (TimeoutError, asyncio.CancelledError):
                    pass
                logger.info("Active optimization cancelled")

        # 5. Persist parameter store
        if state.parameter_store and hasattr(state.parameter_store, "persist"):
            try:
                await state.parameter_store.persist()
                logger.info("Parameter store persisted")
            except Exception as e:
                logger.warning(f"Error persisting parameter store: {e}")

        # 6. Close Claude client connection pool
        if state.claude_client and hasattr(state.claude_client, "close"):
            try:
                await state.claude_client.close()
                logger.info("Claude client closed")
            except Exception as e:
                logger.warning(f"Error closing Claude client: {e}")


# Create FastAPI app
app = FastAPI(
    title="ColdPath ML Engine API",
    description="ML and AI API for trading strategy optimization",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Import and include routers
def _include_routers():
    """Include API routers."""
    from coldpath.api.routes import admin, ai, autotrader, backtest, learning, ml, telemetry

    app.include_router(ai.router)
    app.include_router(ml.router)
    app.include_router(backtest.router)
    app.include_router(autotrader.router)
    app.include_router(telemetry.router)
    app.include_router(learning.router)
    app.include_router(admin.router)

    # AI Strategy Optimization routes
    try:
        from coldpath.api.routes import optimization

        app.include_router(optimization.router)
    except Exception as e:
        logger.warning(f"Failed to include optimization router: {e}")

    # Feedback Loop routes for continuous model improvement
    try:
        from coldpath.api.routes import feedback_loop

        app.include_router(feedback_loop.router)
        logger.info("Feedback loop router included")
    except Exception as e:
        logger.warning(f"Failed to include feedback_loop router: {e}")

    # AI Guided Backtest Setup and Auto Mode
    try:
        from coldpath.api.routes import backtest_guided

        app.include_router(backtest_guided.router)
        logger.info("AI Guided backtest router included")
    except Exception as e:
        logger.warning(f"Failed to include backtest_guided router: {e}")

    # Verified Settings routes for configuration persistence
    try:
        from coldpath.api.routes import settings

        app.include_router(settings.router)
        logger.info("Settings router included")
    except Exception as e:
        logger.warning(f"Failed to include settings router: {e}")

    # Daily Optimizer routes for automated optimization
    try:
        from coldpath.api.routes import daily_optimizer

        app.include_router(daily_optimizer.router)
        logger.info("Daily optimizer router included")
    except Exception as e:
        logger.warning(f"Failed to include daily_optimizer router: {e}")

    # Advanced Optimizers routes (Genetic, Pareto, RL, Ensemble)
    try:
        from coldpath.api.routes import advanced_optimizers

        app.include_router(advanced_optimizers.router)
        logger.info("Advanced optimizers router included")
    except Exception as e:
        logger.warning(f"Failed to include advanced_optimizers router: {e}")

    # ML Enhancement Agent routes (FinGPT-powered ML improvements)
    try:
        from coldpath.api.routes import ml_enhancement

        app.include_router(ml_enhancement.router)
        logger.info("ML Enhancement Agent router included")
    except Exception as e:
        logger.warning(f"Failed to include ml_enhancement router: {e}")

    # Automated Allocation Agent routes (FinGPT-powered portfolio allocation)
    try:
        from coldpath.api.routes import allocation

        app.include_router(allocation.router)
        logger.info("Automated Allocation Agent router included")
    except Exception as e:
        logger.warning(f"Failed to include allocation router: {e}")


_include_routers()


# Health check endpoint


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns status of all components including tuner.
    """
    state = get_app_state()

    components = {
        "claude_client": state.claude_client is not None,
        "claude_router": state.claude_router is not None,
        "llm_orchestrator": state.llm_orchestrator is not None,
        "server_tuner": state.server_tuner is not None,
        "bandit_trainer": state.bandit_trainer is not None,
        "metrics_engine": state.metrics_engine is not None,
        "autotrader": state.autotrader is not None,
        "strategy_orchestrator": state.strategy_orchestrator is not None,
        "performance_engine_ai": state.performance_engine_ai is not None,
        "drift_detector": state.drift_detector is not None,
        "parameter_store": state.parameter_store is not None,
        "feedback_loop": state.feedback_loop is not None,
        "smart_backtest_orchestrator": hasattr(state, "smart_backtest_orchestrator_config"),
    }

    # Backtest learning stats
    backtest_info = {}
    if state.backtest_history is not None:
        backtest_info = {
            "history_count": len(state.backtest_history),
            "feedback_loop_connected": state.feedback_loop is not None,
        }

    # Check if Claude client is actually functional
    claude_available = False
    if state.claude_client:
        try:
            claude_available = state.claude_client.is_available
        except Exception:
            pass

    # Get tuner stats
    tuner_info = {}
    if state.server_tuner:
        tuner_info = {
            "mode": state.server_tuner.mode.value,
            "running": state.server_tuner._running,
            "stats": state.server_tuner.get_stats(),
        }

    # Determine overall status
    all_healthy = all(components.values())
    ai_available = claude_available

    status = "healthy" if all_healthy else "degraded"
    if not any(components.values()):
        status = "unhealthy"

    # AutoTrader state
    autotrader_state = None
    if state.autotrader is not None:
        autotrader_state = state.autotrader.state.value

    # AI optimization status
    optimization_info = {}
    if state.strategy_orchestrator:
        active_opt = state.strategy_orchestrator.active_optimization
        optimization_info = {
            "active": (
                active_opt is not None
                and active_opt.current_stage.value not in ("completed", "failed", "rolled_back")
            ),
            "stage": active_opt.current_stage.value if active_opt else "idle",
        }

    return {
        "status": status,
        "ready": status in ("healthy", "degraded"),
        "autotrader_enabled": state.autotrader_enabled,
        "autotrader_state": autotrader_state,
        "components": components,
        "ai_available": ai_available,
        "tuner": tuner_info,
        "optimization": optimization_info,
        "backtest_learning": backtest_info,
        "version": "0.1.0",
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "ColdPath ML Engine API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


# Entry point for direct execution
if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8080"))

    uvicorn.run(
        "coldpath.api.server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
