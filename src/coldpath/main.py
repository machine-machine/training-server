"""
Cold Path main entry point.

Runs the gRPC client, scheduler, and training jobs.
Handles ML model training, parameter calibration, and backtesting.
"""

# ─────────────────────────────────────────────────────────────
# CRITICAL: Set OMP thread limits BEFORE any numpy/sklearn imports
# This prevents "OMP: Error #179: pthread_mutex_init failed"
# Must be at top of file, before any other imports
# ─────────────────────────────────────────────────────────────
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import asyncio
import hashlib
import json
import logging
import signal
import struct
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .ai.kv_cache_manager import KVCacheManager
from .autotrader.coordinator import AutoTraderConfig, AutoTraderCoordinator, TradingCandidate
from .autotrader.training_integration import get_training_integration
from .backtest.engine import BacktestEngine
from .calibration.bias_calibrator import BiasCalibrator
from .calibration.paper_fill import PaperFillCalibrator
from .calibration.regime_calibrator import (
    MarketMetrics,
    RegimeCalibrationConfig,
    RegimeTriggeredCalibrator,
)
from .calibration.survivorship_model import (
    DynamicSurvivorshipModel,
    SurvivorshipConfig,
)
from .core.connection_pool import ConnectionPoolManager
from .core.health_monitor import HealthMonitor
from .core.memory_monitor import MemoryMonitor

# Resource management imports
from .core.resource_registry import ResourceHealth, ResourceRegistry
from .core.sync_checkpoint import DEFAULT_SYNC_DIR, SyncCheckpoint
from .core.task_queue import TaskQueueManager
from .distillation import (
    ArtifactExporter,
    StudentDistiller,
    TeacherConfig,
    TeacherTrainer,
)
from .ipc.hotpath_client import (
    CandidateEvent,
    HotPathClient,
)
from .ipc.hotpath_client import (
    TradingSignal as IPCTradingSignal,
)
from .learning.feature_engineering import FEATURE_COUNT, FeatureIndex, NormalizationParams

# ML context and online learning imports
from .learning.online_learner import (
    OnlineLearner,
    OnlineLearningConfig,
    OutcomeSource,
    TradeOutcome,
)
from .learning.outcome_tracker import OutcomeTracker
from .learning.training_data_sync import TrainingDataSync
from .models.ml_trade_context import MarketRegime, MLTradeContext
from .publishing import ParameterPublisher
from .publishing.model_publisher import ModelPublisher
from .storage import DatabaseManager, ModelArtifactStore
from .supervisor import TaskSupervisor
from .training.bandit import BanditTrainer
from .training.fraud_model import FraudModel
from .validation.data_quality import DataQualityValidator
from .validation.drift_detector import DriftDetector, DriftDetectorConfig, DriftSeverity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Log rotation for 14-day stability ──
# Prevents coldpath.log from growing to GBs
def _setup_log_rotation():
    """Add rotating file handler to prevent log overflow in long runs."""
    import os
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    log_dir = Path(os.getenv("LOG_DIR", "~/Library/Application Support/2DEXY/logs")).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "coldpath.log"

    max_bytes = int(os.getenv("LOG_MAX_BYTES", str(50 * 1024 * 1024)))  # 50MB
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))  # 250MB max

    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    logging.getLogger().addHandler(handler)

    # Suppress verbose loggers for long runs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    logger.info(
        f"Log rotation configured: {log_path}, "
        f"max={max_bytes // 1024 // 1024}MB × {backup_count} files"
    )


_setup_log_rotation()

FEATURE_NORMALIZATION_DEFAULTS = NormalizationParams()

# Canonical HotPath Unix socket path.
# Use env override when provided, otherwise default to /tmp/dexy-engine.sock
# Note: HotPath uses "/tmp/dexy-engine.sock" explicitly (see config.rs).
# Swift uses FileManager.default.temporaryDirectory which resolves to /tmp on macOS.
DEFAULT_HOTPATH_SOCKET = "/tmp/dexy-engine.sock"
hotpath_socket: str = os.getenv("HOTPATH_SOCKET", DEFAULT_HOTPATH_SOCKET)

# Online-learner feature order (v1-compatible, 15 features)
ONLINE_FEATURE_SIGNATURE = [
    "liquidity_usd",
    "volume_24h",
    "volume_to_liquidity_ratio",
    "holder_count",
    "top_holder_pct",
    "holder_growth",
    "momentum",
    "price_change_24h",
    "volatility",
    "token_age_hours",
    "fdv_usd",
    "lp_burn_pct",
    "mint_authority_enabled",
    "freeze_authority_enabled",
    "fraud_score",
]


@dataclass
class ColdPathConfig:
    """Configuration for Cold Path engine."""

    database_path: str = "sniperdesk.db"
    bitquery_api_key: str | None = None
    hotpath_socket: str = hotpath_socket

    # HotPath gRPC connection
    hotpath_grpc_host: str = "localhost"
    hotpath_grpc_port: int = 50051

    # Training intervals (seconds)
    bandit_training_interval: int = 3600  # 1 hour
    calibration_interval: int = 1800  # 30 minutes
    model_sync_interval: int = 300  # 5 minutes
    cleanup_interval: int = 300  # 5 minutes

    # Task queue sizing
    task_worker_count: int = 4
    task_queue_size: int = 500
    task_max_tracked: int = 5000
    task_auto_cleanup_age_seconds: float = 1800.0

    # Startup throttling
    startup_stagger_seconds: float = 2.0  # delay between background task launches

    # Minimum data requirements
    min_trades_for_training: int = 50
    min_observations_for_training: int = 100
    min_observations_for_calibration: int = 20

    # Model persistence
    model_artifact_dir: str = "models"
    max_model_versions: int = 10

    # AutoTrader - enabled by default for production
    # skip_learning=True for paper mode to avoid chicken-and-egg blocking
    # (can't trade without history, can't build history without trading)
    autotrader_enabled: bool = True
    autotrader_paper_mode: bool = True
    autotrader_skip_learning: bool = True

    # Online learning settings (Phase 4)
    online_learning_enabled: bool = True
    online_learning_interval: int = 60  # seconds
    regime_detection_interval: int = 30  # seconds
    drift_recalibration_threshold: float = 0.7

    # Distillation settings (Phase 5)
    distillation_enabled: bool = True
    distillation_interval: int = 3600  # 1 hour
    distillation_min_outcomes: int = 200

    @classmethod
    def from_env(cls) -> "ColdPathConfig":
        """Load configuration from environment variables."""
        return cls(
            database_path=os.getenv("DEXY_DB_PATH", "sniperdesk.db"),
            bitquery_api_key=os.getenv("BITQUERY_API_KEY"),
            hotpath_socket=hotpath_socket,
            hotpath_grpc_host=os.getenv("HOTPATH_GRPC_HOST", "localhost"),
            hotpath_grpc_port=int(os.getenv("HOTPATH_GRPC_PORT", "50051")),
            bandit_training_interval=int(os.getenv("BANDIT_INTERVAL", "3600")),
            calibration_interval=int(os.getenv("CALIBRATION_INTERVAL", "1800")),
            model_sync_interval=int(os.getenv("MODEL_SYNC_INTERVAL", "300")),
            cleanup_interval=int(os.getenv("CLEANUP_INTERVAL", "300")),
            task_worker_count=int(os.getenv("TASK_WORKERS", "4")),
            task_queue_size=int(os.getenv("TASK_QUEUE_SIZE", "500")),
            task_max_tracked=int(os.getenv("TASK_MAX_TRACKED", "5000")),
            task_auto_cleanup_age_seconds=float(os.getenv("TASK_AUTO_CLEANUP_AGE_SECONDS", "1800")),
            startup_stagger_seconds=float(os.getenv("STARTUP_STAGGER", "2")),
            min_trades_for_training=int(os.getenv("MIN_TRADES_TRAINING", "50")),
            min_observations_for_training=int(os.getenv("MIN_OBSERVATIONS_TRAINING", "100")),
            min_observations_for_calibration=int(os.getenv("MIN_OBS_CALIBRATION", "20")),
            model_artifact_dir=os.getenv("MODEL_DIR", "models"),
            max_model_versions=int(os.getenv("MAX_MODEL_VERSIONS", "10")),
            autotrader_enabled=os.getenv("AUTOTRADER_ENABLED", "true").lower() == "true",
            autotrader_paper_mode=os.getenv("AUTOTRADER_PAPER_MODE", "true").lower() == "true",
            autotrader_skip_learning=os.getenv("AUTOTRADER_SKIP_LEARNING", "true").lower()
            == "true",
            online_learning_enabled=os.getenv("ONLINE_LEARNING_ENABLED", "true").lower() == "true",
            online_learning_interval=int(os.getenv("ONLINE_LEARNING_INTERVAL", "60")),
            regime_detection_interval=int(os.getenv("REGIME_DETECTION_INTERVAL", "30")),
            drift_recalibration_threshold=float(os.getenv("DRIFT_RECALIBRATION_THRESHOLD", "0.7")),
            distillation_enabled=os.getenv("DISTILLATION_ENABLED", "true").lower() == "true",
            distillation_interval=int(os.getenv("DISTILLATION_INTERVAL", "3600")),
            distillation_min_outcomes=int(os.getenv("DISTILLATION_MIN_OUTCOMES", "200")),
        )


@dataclass
class EngineMetrics:
    """Runtime metrics for the Cold Path engine."""

    start_time: datetime = field(default_factory=datetime.now)
    training_runs: int = 0
    calibration_runs: int = 0
    model_syncs: int = 0
    backtests_run: int = 0
    last_training: datetime | None = None
    last_calibration: datetime | None = None
    last_model_sync: datetime | None = None
    errors: int = 0
    # Online learning metrics (Phase 4)
    online_learning_updates: int = 0
    regime_detections: int = 0
    drift_alerts: int = 0
    last_online_update: datetime | None = None
    last_regime_detection: datetime | None = None
    current_regime: str = "chop"
    # Distillation metrics (Phase 5)
    distillation_runs: int = 0
    distillation_pushes: int = 0
    last_distillation: datetime | None = None
    last_distillation_version: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary."""
        uptime = datetime.now() - self.start_time
        return {
            "uptime_seconds": uptime.total_seconds(),
            "training_runs": self.training_runs,
            "calibration_runs": self.calibration_runs,
            "model_syncs": self.model_syncs,
            "backtests_run": self.backtests_run,
            "last_training": self.last_training.isoformat() if self.last_training else None,
            "last_calibration": self.last_calibration.isoformat()
            if self.last_calibration
            else None,
            "last_model_sync": self.last_model_sync.isoformat() if self.last_model_sync else None,
            "errors": self.errors,
            # Online learning metrics
            "online_learning_updates": self.online_learning_updates,
            "regime_detections": self.regime_detections,
            "drift_alerts": self.drift_alerts,
            "last_online_update": self.last_online_update.isoformat()
            if self.last_online_update
            else None,
            "last_regime_detection": self.last_regime_detection.isoformat()
            if self.last_regime_detection
            else None,
            "current_regime": self.current_regime,
            # Distillation metrics
            "distillation_runs": self.distillation_runs,
            "distillation_pushes": self.distillation_pushes,
            "last_distillation": self.last_distillation.isoformat()
            if self.last_distillation
            else None,
            "last_distillation_version": self.last_distillation_version,
        }


class ColdPathEngine:
    """Main Cold Path engine orchestrator.

    Coordinates ML training, parameter calibration, and model publishing
    to the Hot Path engine. Implements graceful shutdown and automatic
    recovery from errors.
    """

    def __init__(self, config: ColdPathConfig):
        self.config = config
        self.metrics = EngineMetrics()
        self._shutdown_event = asyncio.Event()
        self._shutdown_complete = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._supervisor = TaskSupervisor(self._shutdown_event)

        # Initialize resource management infrastructure
        self.resource_registry = ResourceRegistry()
        self.connection_pool = ConnectionPoolManager()
        self.memory_monitor = MemoryMonitor(check_interval_seconds=5.0)
        self.task_queue = TaskQueueManager(
            max_queue_size=config.task_queue_size,
            worker_count=config.task_worker_count,
            max_tracked_tasks=config.task_max_tracked,
            auto_cleanup_age_seconds=config.task_auto_cleanup_age_seconds,
        )

        # Initialize components
        self.db = DatabaseManager(config.database_path)
        self.artifact_store = ModelArtifactStore(
            config.model_artifact_dir,
            max_versions=config.max_model_versions,
        )
        self.bandit_trainer = BanditTrainer(db=self.db)
        self.calibrator = PaperFillCalibrator(db=self.db)
        self.bias_calibrator = BiasCalibrator(db=self.db)

        # Initialize fraud model (will use heuristics until trained)
        self.fraud_model = FraudModel()
        self._load_fraud_model()
        self.outcome_tracker = OutcomeTracker(self.db)

        self.autotrader = AutoTraderCoordinator(
            config=AutoTraderConfig(
                paper_mode=self.config.autotrader_paper_mode,
                skip_learning=self.config.autotrader_skip_learning,
            ),
            bandit=self.bandit_trainer,
            fraud_model=self.fraud_model,  # Now connected!
            bias_calibrator=self.bias_calibrator,
            db=self.db,
            outcome_tracker=self.outcome_tracker,
        )
        self.autotrader.register_outcome_callback(self._ingest_online_outcome_callback)
        self.training_integration = get_training_integration(self.db)
        self.training_integration.config.min_outcomes = self.config.min_observations_for_training
        self.training_integration.config.training_interval_hours = max(
            self.config.bandit_training_interval / 3600.0,
            1.0,
        )
        self.training_integration.monitor.min_outcomes_for_training = (
            self.config.min_observations_for_training
        )
        self.training_integration.monitor.training_interval_hours = (
            self.training_integration.config.training_interval_hours
        )
        self.backtest_engine = BacktestEngine(config.bitquery_api_key)
        self.publisher = ParameterPublisher(config.hotpath_socket)
        self.data_validator = DataQualityValidator()
        self.kv_cache = KVCacheManager(max_size_mb=100)

        # HotPath client for bi-directional communication (Unix socket)
        self.hotpath_client = HotPathClient(
            host=config.hotpath_grpc_host,
            port=config.hotpath_grpc_port,
            socket_path=config.hotpath_socket,
        )
        self.hotpath_client.on_candidate(self._handle_candidate_event)

        # Signal tracking for outcome collection
        self._pending_signal_plan_map: dict[str, str] = {}  # signal_id -> plan_id
        self._pending_signal_timestamps: dict[str, float] = {}  # signal_id -> timestamp
        self._signal_batch_queue: list[IPCTradingSignal] = []
        self._signal_batch_lock = asyncio.Lock()
        self._signal_plan_ttl = 3600.0  # 1 hour max wait for outcome
        self._recent_hotpath_candidate_keys: dict[str, float] = {}

        # Health monitor (initialized after resource registry setup)
        self.health_monitor: HealthMonitor | None = None

        # Online learning components (Phase 4)
        # 15 features: price_usd, liquidity_usd, fdv_usd, volume_24h_usd, holder_count,
        # age_seconds, mint/freeze_authority, lp_burn_pct, top_holder_pct, volatility, etc.
        self.online_learner = OnlineLearner(n_features=15, config=OnlineLearningConfig())
        self.survivorship_model = DynamicSurvivorshipModel(SurvivorshipConfig())
        self.regime_calibrator = RegimeTriggeredCalibrator(RegimeCalibrationConfig())
        self.drift_detector = DriftDetector(DriftDetectorConfig())

        # Track current regime state
        self._current_regime = MarketRegime.CHOP
        self._regime_confidence = 0.5

        # Load persisted model state
        self._load_model_state()

        # Register cache manager with memory monitor for automatic eviction
        self.memory_monitor.register_cache_manager(self.kv_cache)

        # Distillation pipeline (Phase 5)
        self.model_publisher = ModelPublisher(
            host=config.hotpath_grpc_host,
            port=config.hotpath_grpc_port,
        )
        self._model_publish_lock = asyncio.Lock()
        self._distillation_version = 0
        self._online_artifact_version = 2_000_000_000
        self._git_commit = self._get_git_commit()

        # Periodic persistence
        self._last_autotrader_persist: datetime | None = None

        # Sync checkpoint for crash recovery and restart coordination
        self.sync_checkpoint = SyncCheckpoint(
            sync_dir=DEFAULT_SYNC_DIR,
            checkpoint_interval=30.0,  # Checkpoint every 30s
        )
        self._restart_count = 0  # Incremented by supervisor on restart

        # Training data sync (JSONL files → SQLite tables)
        self.training_data_sync = TrainingDataSync(
            db=self.db,
            sync_interval_seconds=60.0,  # Sync every minute
        )

    def _verify_ipc_boundary(self) -> None:
        """Log a read-only IPC sanity check without altering the wire protocol."""
        try:
            sample_context = MLTradeContext.create(
                fraud_score=0.12,
                regime=MarketRegime.CHOP,
                regime_confidence=0.55,
                profitability_confidence=0.58,
            )
            payload = sample_context.to_dict()
            json.dumps(payload)
            restored = MLTradeContext.from_dict(payload)
            status = "PASS" if restored is not None else "FAIL"
            logger.info(
                "IPC verify %s: socket_path=%s ml_context_json=%s",
                status,
                self.config.hotpath_socket,
                "PASS" if restored is not None else "FAIL",
            )
        except Exception as exc:
            logger.error("IPC verify FAIL: %s", exc, exc_info=True)

    @staticmethod
    def _get_git_commit() -> str:
        """Get short git commit hash for artifact metadata."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        """Best-effort conversion to float with a safe default."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_binary_flag(value: Any) -> float:
        """Convert truthy/falsy values to 1.0/0.0."""
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return 1.0
            if normalized in {"0", "false", "no", "n", "off"}:
                return 0.0
        return 1.0 if ColdPathEngine._to_float(value, 0.0) > 0 else 0.0

    def _build_online_feature_vector(
        self,
        features: dict[str, Any],
        fallback_risk_score: float,
    ) -> np.ndarray:
        """Build a 15-feature vector in v1 order for the online learner."""
        liquidity = max(0.0, self._to_float(features.get("liquidity_usd")))
        volume_24h = max(
            0.0,
            self._to_float(features.get("volume_24h_usd", features.get("volume_24h"))),
        )
        vol_liq_ratio = self._to_float(
            features.get("vol_liq_ratio", features.get("volume_to_liquidity_ratio")),
            default=(volume_24h / liquidity if liquidity > 0 else 0.0),
        )
        holder_count = max(0.0, self._to_float(features.get("holder_count")))
        top_holder_pct = self._to_float(features.get("top_holder_pct"))
        holder_growth = self._to_float(
            features.get("holder_growth", features.get("holder_growth_rate"))
        )
        price_change_1h = self._to_float(features.get("price_change_1h"))
        price_change_24h = self._to_float(features.get("price_change_24h"), price_change_1h)
        volatility_1h = self._to_float(features.get("volatility_1h", features.get("volatility")))
        age_seconds = max(0.0, self._to_float(features.get("age_seconds")))
        token_age_hours = self._to_float(
            features.get("token_age_hours"),
            age_seconds / 3600.0,
        )
        fdv_usd = max(0.0, self._to_float(features.get("fdv_usd")))
        lp_burn_pct = self._to_float(features.get("lp_burn_pct"))
        mint_authority = self._to_binary_flag(
            features.get("mint_authority", features.get("mint_authority_enabled"))
        )
        freeze_authority = self._to_binary_flag(
            features.get("freeze_authority", features.get("freeze_authority_enabled"))
        )
        risk_score = self._to_float(
            features.get(
                "risk_score",
                features.get("fraud_score", features.get("goplus_risk_score")),
            ),
            fallback_risk_score,
        )

        return np.array(
            [
                liquidity,
                volume_24h,
                vol_liq_ratio,
                holder_count,
                top_holder_pct,
                holder_growth,
                price_change_1h,
                price_change_24h,
                volatility_1h,
                token_age_hours,
                fdv_usd,
                lp_burn_pct,
                mint_authority,
                freeze_authority,
                risk_score,
            ],
            dtype=np.float64,
        )

    def _ingest_online_outcome_callback(self, event: dict[str, Any]) -> None:
        """Ingest finalized trade outcomes from AutoTrader into OnlineLearner."""
        if not self.config.online_learning_enabled:
            return
        if not event.get("included", True):
            return

        features = event.get("features", {})
        if not isinstance(features, dict):
            features = {}

        fallback_risk = self._to_float(event.get("fraud_score"))
        vector = self._build_online_feature_vector(features, fallback_risk)

        source = (
            OutcomeSource.LIVE
            if str(event.get("source", "")).lower() == "live"
            else OutcomeSource.PAPER
        )

        timestamp_ms = int(
            self._to_float(event.get("timestamp_ms"), datetime.now().timestamp() * 1000)
        )
        outcome = TradeOutcome(
            trade_id=str(event.get("signal_id", f"auto-{timestamp_ms}")),
            timestamp_ms=timestamp_ms,
            source=source,
            features=vector,
            is_profitable=bool(event.get("is_profitable", False)),
            pnl_pct=self._to_float(event.get("pnl_pct")),
            fraud_score=fallback_risk,
            regime=str(event.get("regime", self._current_regime.value)),
            regime_confidence=self._to_float(event.get("regime_confidence"), 0.5),
        )
        self.online_learner.add_outcome(outcome)

    def _compute_artifact_checksum(self, artifact: dict[str, Any]) -> str:
        """Compute artifact checksum compatible with HotPath verification."""
        hasher = hashlib.sha256()
        hasher.update(artifact["model_type"].encode("utf-8"))
        hasher.update(struct.pack("<I", int(artifact["version"])))
        hasher.update(struct.pack("<I", int(artifact["schema_version"])))

        for name in artifact["feature_signature"]:
            hasher.update(name.encode("utf-8"))

        for weight in artifact["weights"]["linear_weights"]:
            hasher.update(struct.pack("<d", float(weight)))
        hasher.update(struct.pack("<d", float(artifact["weights"]["bias"])))

        confidence_threshold = 0.5
        hasher.update(struct.pack("<d", confidence_threshold))
        hasher.update(struct.pack("<d", float(artifact["calibration"]["temperature"])))

        hasher.update(str(artifact.get("dataset_id", "")).encode("utf-8"))
        hasher.update(str(artifact.get("git_commit", "")).encode("utf-8"))
        return hasher.hexdigest()

    def _build_online_artifact(
        self,
        coefficients: np.ndarray,
        update_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a pushable model artifact from online learner coefficients."""
        self._online_artifact_version += 1
        now_ms = int(datetime.now().timestamp() * 1000)
        state = self.online_learner.state
        use_zscore = (
            self.online_learner.config.use_feature_scaling
            and state.feature_means is not None
            and state.feature_stds is not None
            and len(state.feature_means) == len(ONLINE_FEATURE_SIGNATURE)
            and len(state.feature_stds) == len(ONLINE_FEATURE_SIGNATURE)
        )

        feature_transforms = []
        if use_zscore:
            for name, mean, std in zip(
                ONLINE_FEATURE_SIGNATURE,
                state.feature_means,
                state.feature_stds,
                strict=False,
            ):
                safe_std = max(float(std), 1e-6)
                feature_transforms.append(
                    {
                        "feature_name": name,
                        "transform_type": "zscore",
                        "param1": float(mean),
                        "param2": safe_std,
                    }
                )
        else:
            for name in ONLINE_FEATURE_SIGNATURE:
                feature_transforms.append(
                    {
                        "feature_name": name,
                        "transform_type": "none",
                        "param1": 0.0,
                        "param2": 0.0,
                    }
                )

        artifact = {
            "model_type": "profitability",
            "version": self._online_artifact_version,
            "schema_version": 1,
            "feature_signature": list(ONLINE_FEATURE_SIGNATURE),
            "feature_transforms": feature_transforms,
            "dataset_id": "online-learning",
            "metrics": {
                "accuracy": self._to_float(update_result.get("accuracy")),
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc_roc": 0.0,
                "log_loss": self._to_float(update_result.get("loss")),
                "train_samples": int(self._to_float(update_result.get("samples"))),
                "validation_samples": 0,
                "calibration_error": 0.0,
            },
            "created_at": now_ms,
            "git_commit": self._git_commit,
            "calibration": {
                "calibration_method": "platt",
                "temperature": 1.0,
                "platt_coeffs": [0.0, 0.0],
            },
            "compatibility_notes": (
                f"Online SGD update n={int(self._to_float(update_result.get('n_updates')))} "
                f"delta={self._to_float(update_result.get('coefficient_change')):.6f}"
            ),
            "weights": {
                "student_type": "logistic",
                "linear_weights": coefficients.tolist(),
                "bias": 0.0,
                "onnx_model": b"",
            },
        }
        artifact["checksum"] = self._compute_artifact_checksum(artifact)
        return artifact

    async def _publish_online_update(self, update_result: dict[str, Any]) -> None:
        """Publish online-learning coefficient updates to HotPath."""
        if self._to_float(update_result.get("coefficient_change")) <= 0:
            return

        coefficients = self.online_learner.get_coefficients()
        if len(coefficients) != len(ONLINE_FEATURE_SIGNATURE):
            logger.warning(
                f"Online update not published: expected {len(ONLINE_FEATURE_SIGNATURE)} "
                f"coefficients, got {len(coefficients)}"
            )
            return

        async with self._model_publish_lock:
            if not self.model_publisher.is_connected:
                await self.model_publisher.connect()

            if not self.model_publisher.is_connected:
                logger.debug("Online update not published: model publisher unavailable")
                return

            artifact = self._build_online_artifact(coefficients, update_result)
            push_result = await self.model_publisher.push_artifact(artifact)
            if push_result.success:
                logger.info(
                    f"Online model push succeeded: v{push_result.version} "
                    f"(samples={int(self._to_float(update_result.get('samples')))})"
                )
            else:
                logger.warning(f"Online model push failed: {push_result.error}")

    def _load_fraud_model(self):
        """Load persisted fraud model from disk."""
        fraud_model_path = Path(self.config.model_artifact_dir) / "fraud_model.pkl"
        try:
            if fraud_model_path.exists():
                self.fraud_model.load(str(fraud_model_path))
                logger.info(f"Loaded trained fraud model from {fraud_model_path}")
            else:
                logger.info("No trained fraud model found - using heuristic scoring")
        except Exception as e:
            logger.warning(f"Failed to load fraud model: {e} - using heuristic scoring")

    def _load_model_state(self):
        """Load persisted model state from disk."""
        try:
            bandit_state = self.artifact_store.load_latest("bandit")
            if bandit_state:
                self.bandit_trainer.load_state(bandit_state)
                logger.info("Loaded bandit model state from disk")

            calibration_state = self.artifact_store.load_latest("calibration")
            if calibration_state:
                self.calibrator.load_state(calibration_state)
                logger.info("Loaded calibration state from disk")

            bias_calibration_state = self.artifact_store.load_latest("bias_calibration")
            if bias_calibration_state:
                self.bias_calibrator.load_state(bias_calibration_state)
                logger.info("Loaded bias calibration state from disk")

            autotrader_state = self.artifact_store.load_latest("autotrader")
            if autotrader_state:
                self.autotrader.load_state(autotrader_state)
                logger.info("Loaded autotrader state from disk")
        except Exception as e:
            logger.warning(f"Failed to load model state: {e}")

    async def _start_background_tasks(self):
        """Launch background loops under supervision with stagger to reduce startup spikes.

        Each task is registered with the TaskSupervisor which provides:
        - Automatic restart with exponential backoff on failure
        - Configurable max restart count (10 for critical, 5 for non-critical)
        - Restart counter reset after 30 minutes of healthy uptime
        """
        self._tasks = []

        # (coroutine_fn, name, is_critical)
        # Critical tasks are essential for engine operation and get more restart attempts
        task_specs = [
            (self._training_loop, "training", True),
            (self._calibration_loop, "calibration", True),
            (self._model_sync_loop, "model_sync", False),
            (self._health_check_loop, "health_check", True),
            (self._outcome_collection_loop, "outcome_collection", True),
            (self._signal_batch_loop, "signal_batch", False),
            (self._hotpath_candidate_poll_loop, "hotpath_candidate_poll", False),
            (self._hotpath_reconnect_loop, "hotpath_reconnect", False),
            (self._cleanup_loop, "cleanup", False),
            (self._resource_metrics_loop, "resource_metrics", False),
            (self._persist_autotrader_loop, "autotrader_persist", False),
        ]

        if self.config.online_learning_enabled:
            task_specs.extend(
                [
                    (self._online_learning_loop, "online_learning", False),
                    (self._regime_detection_loop, "regime_detection", False),
                ]
            )
            logger.info("Online learning loops enabled; starting with stagger")

        if self.config.distillation_enabled:
            task_specs.append((self._distillation_loop, "distillation", False))
            logger.info("Distillation loop enabled")

        for coro_fn, name, is_critical in task_specs:
            task = self._supervisor.register(
                name=name,
                coro_fn=coro_fn,
                is_critical=is_critical,
                max_restarts=10 if is_critical else 5,
            )
            self._tasks.append(task)
            await asyncio.sleep(self.config.startup_stagger_seconds)

        logger.info(
            f"Started {len(task_specs)} supervised background tasks "
            f"(stagger={self.config.startup_stagger_seconds}s)"
        )

    async def run(self):
        """Run the cold path engine with graceful shutdown support."""
        logger.info("Starting Cold Path Engine")
        logger.info(f"Database: {self.config.database_path}")
        logger.info(f"Hot Path socket: {self.config.hotpath_socket}")
        logger.info(
            f"Hot Path gRPC: {self.config.hotpath_grpc_host}:{self.config.hotpath_grpc_port}"
        )
        logger.info(f"AutoTrader enabled: {self.config.autotrader_enabled}")

        # Initialize sync checkpoint for crash recovery coordination
        import os

        checkpoint_state = self.sync_checkpoint.initialize(
            pid=os.getpid(),
            restart_count=self._restart_count,
        )
        logger.info(f"Sync checkpoint initialized (restart_count={checkpoint_state.restart_count})")

        # Clear any stale recovery info after successful init
        self.sync_checkpoint.clear_recovery()

        # Start resource management infrastructure
        await self.memory_monitor.start()
        await self.task_queue.start()
        logger.info("Resource management infrastructure started")

        # Initialize database connection
        await self.db.connect()

        # Connect to Hot Path for parameter publishing (Unix socket)
        try:
            await self.publisher.connect()
            logger.info("Connected to Hot Path for parameter sync")
        except Exception as e:
            logger.warning(f"Could not connect to Hot Path socket: {e}")
            logger.info("Will retry connection during model sync")

        # Connect to Hot Path for model publishing (gRPC)
        if self.config.distillation_enabled or self.config.online_learning_enabled:
            try:
                if await self.model_publisher.connect():
                    logger.info("Connected to Hot Path for model artifact publishing")
                else:
                    logger.warning("Could not connect to Hot Path gRPC for model publishing")
            except Exception as e:
                logger.warning(f"Could not connect to Hot Path gRPC: {e}")

        # Connect to Hot Path for bi-directional communication (Unix socket)
        try:
            if await self.hotpath_client.connect():
                logger.info("Connected to Hot Path for event streaming")

                # Start event stream
                await self.hotpath_client.start_event_stream(["candidate", "result"])

                # If AutoTrader is enabled, also start HotPath's AutoPipeline
                if self.config.autotrader_enabled:
                    result = await self.hotpath_client.start_autotrader()
                    if result.get("autotrader_running"):
                        logger.info("Started HotPath AutoPipeline")
                    else:
                        logger.warning(f"Could not start HotPath AutoPipeline: {result}")
            else:
                logger.warning("Could not connect to Hot Path - will retry later")
        except Exception as e:
            logger.warning(f"Could not connect to Hot Path: {e}")

        # Fail fast if live trading is requested but HotPath isn't reachable
        if (
            self.config.autotrader_enabled
            and not self.config.autotrader_paper_mode
            and not self.hotpath_client.is_connected
        ):
            logger.critical(
                "Live trading requested (paper_mode disabled) but HotPath is not connected; "
                "aborting startup"
            )
            await self.shutdown()
            return

        # Initialize health monitor with resource registry
        self.health_monitor = HealthMonitor(
            registry=self.resource_registry,
            check_interval_seconds=30.0,
        )

        # Register health monitor callbacks
        def on_degraded(resource_id: str, health: ResourceHealth):
            logger.warning(f"Resource degraded: {resource_id} - {health.message}")

        def on_recovered(resource_id: str):
            logger.info(f"Resource recovered: {resource_id}")

        def on_failure(resource_id: str, reason: str):
            logger.error(f"Resource recovery failed: {resource_id} - {reason}")

        self.health_monitor.on_degraded(on_degraded)
        self.health_monitor.on_recovered(on_recovered)
        self.health_monitor.on_failure(on_failure)

        await self.health_monitor.start()
        logger.info("Health monitoring started")
        self._verify_ipc_boundary()

        # Start background tasks
        await self._start_background_tasks()
        await self.training_integration.start()
        logger.info("Training integration started")

        # Start training data sync (JSONL → SQLite)
        await self.training_data_sync.start_periodic_sync()
        # Initial sync immediately
        initial_sync_result = await self.training_data_sync.sync_all_sources()
        logger.info(
            f"Training data sync started: {initial_sync_result.inserted_samples} initial samples"
        )

        # Start AutoTrader if enabled
        if self.config.autotrader_enabled:
            self.autotrader.set_hotpath_connected(self.hotpath_client.is_connected)
            await self.autotrader.start()
            logger.info("AutoTrader started")

        # Mark checkpoint as healthy and start periodic checkpointing
        self.sync_checkpoint.update_status("healthy")
        await self.sync_checkpoint.start_checkpoint_loop()
        logger.info("Sync checkpoint running - engine is healthy")

        # Keep process alive until a shutdown is requested and completed
        await self._shutdown_complete.wait()

    async def _persist_autotrader_loop(self):
        """Periodically persist autotrader state for status freshness."""
        await asyncio.sleep(10)  # small delay after startup
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)
                state = self.autotrader.to_state()
                self.artifact_store.save("autotrader", state)
                self._last_autotrader_persist = datetime.now()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Autotrader persist error: {e}")
                await asyncio.sleep(10)

    async def _cleanup(self):
        """Cleanup resources on shutdown."""
        self._shutdown_event.set()

        # Stop checkpoint loop first and mark as shutting down
        await self.sync_checkpoint.stop()
        logger.info("Sync checkpoint stopped")

        await asyncio.sleep(1)  # Grace period for background loops to exit

        logger.info("Cold Path Engine shutting down...")

        # Stop health monitoring first
        if self.health_monitor:
            await self.health_monitor.stop()
            logger.info("Health monitor stopped")

        # Stop AutoTrader
        if self.autotrader.state.value != "stopped":
            await self.autotrader.stop()

        # Collect tasks to cancel BEFORE cancellation to prevent race conditions
        # where new tasks spawn between shutdown event and cancellation
        tasks_to_cancel = [t for t in self._tasks if t and not t.done()]

        # Cancel all tasks
        for task in tasks_to_cancel:
            task.cancel()

        # Wait for tasks to complete with hard deadline
        if tasks_to_cancel:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=30,
                )
            except TimeoutError:
                still_running = [t for t in tasks_to_cancel if not t.done()]
                logger.warning(
                    f"Shutdown deadline exceeded, {len(still_running)} tasks still running: "
                    f"{[t.get_name() for t in still_running]}"
                )
                # Force cancel remaining tasks
                for task in still_running:
                    task.cancel()
                    # Let cancellation propagate
                    await asyncio.sleep(0)
                # Brief grace period for final cleanup
                await asyncio.sleep(0.5)

        # Verify no new tasks were created during shutdown
        new_tasks = [t for t in self._tasks if t not in tasks_to_cancel and not t.done()]
        if new_tasks:
            logger.error(f"New tasks created during shutdown: {[t.get_name() for t in new_tasks]}")
            for task in new_tasks:
                task.cancel()

        # Ensure no DB access in flight before saving state
        # Brief delay to let any in-flight DB operations complete
        # TODO: Implement proper DB access barrier (track active operations)
        await asyncio.sleep(0.5)

        # Save model state (needs DB, so do this before closing DB)
        try:
            self.artifact_store.save("bandit", self.bandit_trainer.to_state())
            self.artifact_store.save("calibration", self.calibrator.to_state())
            self.artifact_store.save("bias_calibration", self.bias_calibrator.to_state())
            self.artifact_store.save("autotrader", self.autotrader.to_state())

            # Save fraud model if it has been trained
            if self.fraud_model and self.fraud_model.is_trained:
                fraud_model_path = Path(self.config.model_artifact_dir) / "fraud_model.pkl"
                self.fraud_model.save(str(fraud_model_path))
                logger.info(f"Saved fraud model to {fraud_model_path}")

            logger.info("Saved model state to disk")
        except Exception as e:
            logger.error(f"Failed to save model state: {e}")

        # Shutdown task queue and memory monitor
        await self.task_queue.stop()
        await self.memory_monitor.stop()
        await self.training_data_sync.stop_periodic_sync()
        await self.training_integration.stop()
        logger.info("Task queue, memory monitor, and data sync stopped")

        # Close database (no more writes after this)
        await self.db.close()
        logger.info("Database closed")

        # Close IPC connections
        await self.publisher.close()
        await self.model_publisher.close()
        await self.hotpath_client.disconnect()

        # Close connection pool last
        await self.connection_pool.close()
        logger.info("Resource management infrastructure stopped")

        logger.info("Cold Path Engine shutdown complete")
        self._shutdown_complete.set()

    async def _handle_candidate_event(self, event: CandidateEvent):
        """Handle a candidate event from HotPath and process through AutoTrader."""
        try:
            # Convert CandidateEvent to TradingCandidate
            # HotPath candidate events do not include price; use a safe, non-zero
            # fallback so validation and execution payloads don't drop the candidate.
            fallback_price = 0.0001
            logger.info(
                f"HotPath candidate received: mint={event.mint} pool={event.pool} "
                f"fallback_price={fallback_price}"
            )

            candidate = TradingCandidate(
                token_mint=event.mint,
                token_symbol=event.mint[:8],  # Placeholder, should be enriched
                pool_address=event.pool,
                price_usd=fallback_price,
                liquidity_usd=event.liquidity_usd or 0.0,
                fdv_usd=event.fdv_usd or 0.0,
                volume_24h_usd=event.volume_24h or 0.0,
                holder_count=event.holder_count or 0,
                age_seconds=int((datetime.now().timestamp() * 1000 - event.received_at_ms) / 1000),
                top_holder_pct=event.top_holder_pct or 0.0,
                discovery_source=event.source,
            )

            await self._process_trading_candidate(candidate)

        except Exception as e:
            logger.error(f"Error handling candidate event: {e}", exc_info=True)

    async def _process_trading_candidate(self, candidate: TradingCandidate):
        """Process a TradingCandidate and route any resulting signal."""
        self.autotrader.set_hotpath_connected(self.hotpath_client.is_connected)
        signal = await self.autotrader.process_candidate(candidate)

        if signal is None:
            return

        if self.hotpath_client.is_connected:
            ipc_signal = IPCTradingSignal(
                signal_id=signal.signal_id,
                token_mint=signal.token_mint,
                token_symbol=signal.token_symbol,
                pool_address=signal.pool_address,
                action=signal.action,
                position_size_sol=signal.position_size_sol,
                slippage_bps=signal.slippage_bps,
                expected_price=signal.expected_price,
                confidence=signal.confidence,
                expires_at_ms=int(signal.expires_at.timestamp() * 1000),
                ml_context=self._serialize_ml_context(signal.ml_context),
            )
            async with self._signal_batch_lock:
                if len(self._signal_batch_queue) >= 1000:
                    logger.warning("Signal batch queue full (1000), dropping oldest signal")
                    self._signal_batch_queue.pop(0)
                self._signal_batch_queue.append(ipc_signal)
            logger.info(
                f"Enqueued signal for HotPath batch: signal_id={signal.signal_id} "
                f"expected_price={signal.expected_price}"
            )
        else:
            logger.info(
                f"HotPath disconnected; simulating paper trade for signal {signal.signal_id}"
            )
            await self._simulate_paper_trade(signal, candidate)

    def _serialize_ml_context(self, ml_context: Any | None) -> dict[str, Any] | None:
        """Convert MLTradeContext objects into the Unix-socket JSON payload shape."""
        if ml_context is None:
            return None

        if isinstance(ml_context, dict):
            restored = MLTradeContext.from_dict(ml_context)
            return restored.to_dict() if restored is not None else None

        if hasattr(ml_context, "to_dict"):
            try:
                payload = ml_context.to_dict()
                restored = MLTradeContext.from_dict(payload)
                return restored.to_dict() if restored is not None else payload
            except Exception as exc:
                logger.warning(f"Failed to serialize ML context via to_dict: {exc}", exc_info=True)

        adjustments = getattr(ml_context, "adjustments", None)
        deployer_risk = getattr(ml_context, "deployer_risk", None)
        regime = getattr(ml_context, "regime", None)

        serialized: dict[str, Any] = {
            "fraud_score": float(getattr(ml_context, "fraud_score", 0.0)),
            "regime": regime.value if isinstance(regime, MarketRegime) else str(regime or "chop"),
            "regime_confidence": float(getattr(ml_context, "regime_confidence", 0.5)),
            "profitability_confidence": float(getattr(ml_context, "profitability_confidence", 0.5)),
            "time_in_regime": int(getattr(ml_context, "time_in_regime", 0)),
            "transition_probability": float(getattr(ml_context, "transition_probability", 0.0)),
            "survivorship_discount": float(getattr(ml_context, "survivorship_discount", 0.0)),
        }

        if adjustments is not None:
            serialized["adjustments"] = {
                "slippage_multiplier": float(getattr(adjustments, "slippage_multiplier", 1.0)),
                "inclusion_penalty": float(getattr(adjustments, "inclusion_penalty", 0.0)),
                "mev_exposure_multiplier": float(
                    getattr(adjustments, "mev_exposure_multiplier", 1.0)
                ),
                "confidence_multiplier": float(getattr(adjustments, "confidence_multiplier", 1.0)),
                "position_limit_multiplier": float(
                    getattr(adjustments, "position_limit_multiplier", 1.0)
                ),
            }

        if deployer_risk is not None:
            serialized["deployer_risk"] = {
                "previous_tokens": int(getattr(deployer_risk, "previous_tokens", 0)),
                "rug_count": int(getattr(deployer_risk, "rug_count", 0)),
                "rug_rate": float(getattr(deployer_risk, "rug_rate", 0.0)),
                "avg_token_lifespan_hours": float(
                    getattr(deployer_risk, "avg_token_lifespan_hours", 0.0)
                ),
            }

        return serialized

    @staticmethod
    def _denormalize_log_feature(value: Any, mean: float, std: float) -> float:
        """Best-effort inverse of FeatureEngineer._log_normalize()."""
        normalized = max(-1.0, min(1.0, ColdPathEngine._to_float(value)))
        return max(0.0, float(np.expm1(normalized * 3.0 * std + mean)))

    @staticmethod
    def _denormalize_sigmoid_feature(value: Any, scale: float) -> float:
        """Best-effort inverse of FeatureEngineer._sigmoid_normalize()."""
        normalized = max(-0.999999, min(0.999999, ColdPathEngine._to_float(value)))
        return float(scale * (2.0 * np.arctanh(normalized)))

    @staticmethod
    def _denormalize_tanh_feature(value: Any, scale: float) -> float:
        """Best-effort inverse of tanh(value / scale)."""
        normalized = max(-0.999999, min(0.999999, ColdPathEngine._to_float(value)))
        return float(np.arctanh(normalized) * scale)

    def _build_polled_candidate(
        self,
        item: dict[str, Any],
        timestamp_ms: int,
    ) -> TradingCandidate:
        """Reconstruct a TradingCandidate from HotPath cache output."""
        mint = str(item.get("mint") or "")
        symbol = str(item.get("symbol") or mint[:8])
        source = str(item.get("source") or "hotpath_poll")
        confidence = self._to_float(item.get("confidence"), 0.0)
        features = item.get("features") if isinstance(item.get("features"), list) else []

        liquidity_usd = 100000.0
        volume_24h_usd = 100000.0
        holder_count = 100
        age_seconds = max(self.autotrader.config.min_token_age_seconds, 600)
        top_holder_pct = 5.0
        lp_burn_pct = 90.0
        volatility_1h = 10.0
        price_change_1h = confidence * 20.0
        mint_authority_enabled = False
        freeze_authority_enabled = False
        liquidity_floored = False
        holder_count_floored = False

        if len(features) >= FEATURE_COUNT:
            liquidity_usd = self._denormalize_log_feature(
                features[FeatureIndex.POOL_TVL_SOL],
                FEATURE_NORMALIZATION_DEFAULTS.tvl_log_mean,
                FEATURE_NORMALIZATION_DEFAULTS.tvl_log_std,
            )
            holder_count = max(
                1,
                int(
                    round(
                        self._denormalize_log_feature(
                            features[FeatureIndex.HOLDER_COUNT_UNIQUE],
                            FEATURE_NORMALIZATION_DEFAULTS.holder_log_mean,
                            FEATURE_NORMALIZATION_DEFAULTS.holder_log_std,
                        )
                    )
                ),
            )
            age_seconds = max(
                self.autotrader.config.min_token_age_seconds,
                int(
                    round(
                        self._denormalize_log_feature(
                            features[FeatureIndex.POOL_AGE_SECONDS],
                            FEATURE_NORMALIZATION_DEFAULTS.age_log_mean,
                            FEATURE_NORMALIZATION_DEFAULTS.age_log_std,
                        )
                    )
                ),
            )
            top_holder_pct = max(
                0.0,
                min(
                    100.0,
                    (self._to_float(features[FeatureIndex.TOP_10_HOLDER_CONCENTRATION]) + 1.0)
                    * 50.0,
                ),
            )
            lp_burn_pct = max(
                0.0,
                min(100.0, self._to_float(features[FeatureIndex.LP_LOCK_PERCENTAGE]) * 100.0),
            )
            volatility_1h = max(
                0.0,
                self._denormalize_sigmoid_feature(
                    features[FeatureIndex.VOLATILITY_5M],
                    20.0,
                ),
            )
            price_change_1h = self._denormalize_tanh_feature(
                features[FeatureIndex.PRICE_MOMENTUM_30S],
                50.0,
            )
            mint_authority_enabled = (
                self._to_float(features[FeatureIndex.MINT_AUTHORITY_REVOKED]) < 0.0
            )
            freeze_authority_enabled = self._to_float(features[FeatureIndex.TOKEN_FREEZEABLE]) < 0.0
            volume_24h_usd = max(
                liquidity_usd * 0.5,
                liquidity_usd
                * max(
                    0.5,
                    (self._to_float(features[FeatureIndex.BUY_VOLUME_RATIO]) + 1.0) / 2.0,
                ),
            )

        min_liquidity_usd = max(self.autotrader.config.min_liquidity_usd, 1.0)
        if liquidity_usd < min_liquidity_usd:
            liquidity_usd = min_liquidity_usd
            liquidity_floored = True

        min_holder_count = max(self.autotrader.config.min_holder_count, 1)
        if holder_count < min_holder_count:
            holder_count = min_holder_count
            holder_count_floored = True

        volume_24h_usd = max(
            volume_24h_usd,
            float(self.autotrader.config.min_volume_24h_usd),
            liquidity_usd * 0.5,
        )

        logger.info(
            "Candidate poll stage=feature_engineering mint=%s source=%s confidence=%.4f "
            "features=%s liquidity_usd=%.2f holder_count=%s age_seconds=%s "
            "liquidity_floored=%s holder_count_floored=%s",
            mint,
            source,
            confidence,
            len(features),
            liquidity_usd,
            holder_count,
            age_seconds,
            liquidity_floored,
            holder_count_floored,
        )

        return TradingCandidate(
            token_mint=mint,
            token_symbol=symbol,
            pool_address=f"polled:{mint}",
            price_usd=max(0.0001, self._to_float(item.get("price_usd"), 0.0001)),
            liquidity_usd=liquidity_usd,
            fdv_usd=max(liquidity_usd * 10.0, self._to_float(item.get("fdv_usd"), 1000000.0)),
            volume_24h_usd=volume_24h_usd,
            holder_count=holder_count,
            age_seconds=age_seconds,
            mint_authority_enabled=mint_authority_enabled,
            freeze_authority_enabled=freeze_authority_enabled,
            lp_burn_pct=lp_burn_pct,
            top_holder_pct=top_holder_pct,
            volatility_1h=volatility_1h,
            price_change_1h=price_change_1h,
            discovery_timestamp=datetime.fromtimestamp(timestamp_ms / 1000.0),
            discovery_source=source,
        )

    async def _hotpath_candidate_poll_loop(self):
        """Fallback candidate ingestion when the IPC event stream is quiet."""
        poll_interval_seconds = 5.0
        fetch_timeout_seconds = 4.0
        next_poll_at = time.monotonic() + poll_interval_seconds
        while not self._shutdown_event.is_set():
            try:
                sleep_for = max(0.0, next_poll_at - time.monotonic())
                if sleep_for:
                    await asyncio.sleep(sleep_for)
                cycle_started = time.monotonic()

                if not self.hotpath_client.is_connected:
                    continue

                if self.autotrader.state.value not in {"learning", "trading"}:
                    continue

                now = time.time()
                self._recent_hotpath_candidate_keys = {
                    key: seen_at
                    for key, seen_at in self._recent_hotpath_candidate_keys.items()
                    if now - seen_at < 600
                }

                logger.info(
                    "Candidate poll stage=data_fetch start limit=%s state=%s",
                    25,
                    self.autotrader.state.value,
                )
                recent_candidates = await asyncio.wait_for(
                    self.hotpath_client.get_recent_candidates(limit=25),
                    timeout=fetch_timeout_seconds,
                )
                logger.info(
                    "Candidate poll stage=data_fetch complete received=%s latency_ms=%s",
                    len(recent_candidates),
                    int((time.monotonic() - cycle_started) * 1000),
                )

                duplicates = 0
                stale = 0
                appended = 0
                for item in recent_candidates:
                    mint = item.get("mint")
                    if not mint:
                        continue

                    timestamp_ms = int(item.get("timestamp_ms") or now * 1000)
                    age_ms = max(0, int(now * 1000) - timestamp_ms)
                    if age_ms > 120000:
                        stale += 1
                    dedupe_key = f"{mint}:{timestamp_ms}"
                    if dedupe_key in self._recent_hotpath_candidate_keys:
                        duplicates += 1
                        continue

                    self._recent_hotpath_candidate_keys[dedupe_key] = now
                    candidate = self._build_polled_candidate(item, timestamp_ms)
                    logger.info(
                        "HotPath candidate polled: mint=%s source=%s timestamp_ms=%s age_ms=%s",
                        candidate.token_mint,
                        candidate.discovery_source,
                        timestamp_ms,
                        age_ms,
                    )
                    before_observations = self.autotrader.learning_metrics.observations
                    await self._process_trading_candidate(candidate)
                    after_observations = self.autotrader.learning_metrics.observations
                    if after_observations > before_observations:
                        appended += after_observations - before_observations
                        logger.info(
                            "Candidate poll stage=observation_append mint=%s observations=%s",
                            candidate.token_mint,
                            after_observations,
                        )

                logger.info(
                    "Candidate poll cycle complete received=%s duplicates=%s stale=%s appended=%s",
                    len(recent_candidates),
                    duplicates,
                    stale,
                    appended,
                )

            except asyncio.CancelledError:
                raise
            except TimeoutError:
                logger.error(
                    "HotPath candidate poll timed out after %.1fs",
                    fetch_timeout_seconds,
                )
            except Exception as e:
                logger.error(f"HotPath candidate poll error: {e}", exc_info=True)
            finally:
                next_poll_at += poll_interval_seconds
                now_monotonic = time.monotonic()
                if next_poll_at < now_monotonic:
                    logger.warning(
                        "Candidate poll lag detected: overrun_ms=%s interval_s=%.1f",
                        int((now_monotonic - next_poll_at) * 1000),
                        poll_interval_seconds,
                    )
                    next_poll_at = now_monotonic + poll_interval_seconds

    async def _simulate_paper_trade(
        self, signal: "IPCTradingSignal", candidate: "TradingCandidate"
    ):
        """Simulate a paper trade when HotPath is not available.

        Uses MLTradeContext if available for regime-aware, fraud-adjusted simulation.
        Falls back to quality-based simulation otherwise.
        """
        import random

        from .models.ml_trade_context import MarketRegime

        # Get ML context from signal if available
        ml_context = getattr(signal, "ml_context", None)

        logger.info(
            f"[PAPER] Simulating trade: {signal.token_symbol} "
            f"size={signal.position_size_sol:.4f} SOL, confidence={signal.confidence:.1%}"
            + (f", regime={ml_context.regime.value}" if ml_context else "")
        )

        # Simulate latency
        latency_model = self.calibrator.latency_model
        latency_ms = int(latency_model.sample())

        # Calculate inclusion probability (ML-aware)
        base_inclusion = 0.95
        if ml_context:
            # Apply ML context inclusion penalty
            inclusion_prob = base_inclusion * ml_context.effective_inclusion_multiplier()
        else:
            # Simple fraud-based penalty
            inclusion_prob = base_inclusion * (1 - signal.fraud_score * 0.2)

        if random.random() > inclusion_prob:
            logger.info(f"[PAPER] Trade {signal.signal_id} NOT INCLUDED")
            await self.autotrader.record_trade_outcome(
                signal_id=signal.signal_id,
                pnl_sol=0.0,
                pnl_pct=0.0,
                slippage_realized_bps=0,
                included=False,
                execution_latency_ms=latency_ms,
            )
            return

        # Calculate slippage (ML-aware)
        base_slippage = 50  # 0.5% base
        if candidate.liquidity_usd > 0:
            liquidity_factor = min(2.0, 50000 / candidate.liquidity_usd)
            realized_slippage = base_slippage * liquidity_factor
        else:
            realized_slippage = base_slippage * 2

        # Apply ML context slippage multiplier
        if ml_context:
            realized_slippage *= ml_context.effective_slippage_multiplier()
            # Apply MEV penalty in MEV_HEAVY regime
            if ml_context.regime == MarketRegime.MEV_HEAVY and random.random() < 0.3:
                realized_slippage += random.uniform(50, 150)  # MEV sandwich

        # Calculate quality score
        normalized_confidence = min(1.0, signal.confidence * 2)
        quality_score = normalized_confidence * (1 - signal.fraud_score)

        # Apply ML context adjustments to quality
        if ml_context:
            quality_score *= ml_context.adjustments.confidence_multiplier
            # Stable regimes boost quality
            if ml_context.is_regime_stable():
                quality_score *= 1.1
            # Apply survivorship discount
            quality_score *= 1 - ml_context.survivorship_discount

        quality_score = min(1.0, max(0.0, quality_score))

        # Base win rate depends on quality (45% to 65%)
        win_rate = 0.45 + quality_score * 0.20

        # Regime adjustments to win rate
        if ml_context:
            if ml_context.regime == MarketRegime.BULL:
                win_rate += 0.05
            elif ml_context.regime == MarketRegime.BEAR:
                win_rate -= 0.05

        # Determine if this trade wins
        is_win = random.random() < win_rate

        if is_win:
            # Winning trades: 8% to 25% gain
            base_gain = 0.08 + quality_score * 0.17
            pnl_pct = base_gain * random.uniform(0.7, 1.3)
        else:
            # Losing trades: -5% to -12% loss
            base_loss = 0.05 + (1 - quality_score) * 0.07
            pnl_pct = -base_loss * random.uniform(0.7, 1.3)

        # Subtract slippage cost
        pnl_pct -= realized_slippage / 10000

        pnl_sol = signal.position_size_sol * pnl_pct

        status = "WIN" if pnl_pct > 0 else "LOSS"
        regime_info = f", regime={ml_context.regime.value}" if ml_context else ""
        logger.info(
            f"[PAPER] {status}: {signal.token_symbol} "
            f"PnL={pnl_pct:+.1%} ({pnl_sol:+.4f} SOL), "
            f"quality={quality_score:.2f}{regime_info}"
        )

        # Record outcome for learning
        await self.autotrader.record_trade_outcome(
            signal_id=signal.signal_id,
            pnl_sol=pnl_sol,
            pnl_pct=pnl_pct,  # Keep as decimal, coordinator formats as %
            slippage_realized_bps=int(realized_slippage),
            included=True,
            execution_latency_ms=latency_ms,
        )

        # Update database
        if self.db:
            await self.db.update_trading_signal(
                signal.signal_id,
                {
                    "status": "completed",
                    "completed_at_ms": int(datetime.now().timestamp() * 1000),
                    "pnl_sol": pnl_sol,
                    "pnl_pct": pnl_pct * 100,
                },
            )

    async def _outcome_collection_loop(self):
        """Periodically collect trade outcomes for continuous learning."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                if not self.hotpath_client.is_connected:
                    continue

                # Collect outcomes for pending signals
                completed = []
                for signal_id, plan_id in list(self._pending_signal_plan_map.items()):
                    try:
                        outcome = await self.hotpath_client.get_trade_outcome(signal_id, plan_id)

                        if outcome is not None:
                            # Extract execution details from outcome
                            # Note: Outcome object should have these fields from HotPath
                            # ExecutionResult
                            execution_details = {
                                "base_fee_lamports": getattr(outcome, "base_fee_lamports", 0),
                                "priority_fee_lamports": getattr(
                                    outcome, "priority_fee_lamports", 0
                                ),
                                "jito_tip_lamports": getattr(outcome, "jito_tip_lamports", 0),
                                "dex_fee_lamports": getattr(outcome, "dex_fee_lamports", 0),
                                "tx_signature": getattr(outcome, "tx_signature", None),
                                "slot": getattr(outcome, "slot", None),
                                "execution_time_ms": outcome.execution_latency_ms,
                                "amount_out_lamports": getattr(outcome, "amount_out_lamports", 0),
                            }

                            # Calculate DEX fee if missing and amount_out is available
                            if (
                                execution_details["dex_fee_lamports"] == 0
                                and execution_details["amount_out_lamports"] > 0
                            ):
                                from coldpath.utils.fee_calculator import calculate_dex_fee

                                execution_details["dex_fee_lamports"] = calculate_dex_fee(
                                    execution_details["amount_out_lamports"],
                                    dex_type="raydium",  # TODO: Get from signal metadata
                                )

                            # Report to AutoTrader for learning
                            await self.autotrader.record_trade_outcome(
                                signal_id=outcome.signal_id,
                                pnl_sol=outcome.pnl_sol,
                                pnl_pct=outcome.pnl_pct,
                                slippage_realized_bps=outcome.slippage_realized_bps,
                                included=outcome.included,
                                execution_latency_ms=outcome.execution_latency_ms,
                                execution_details=execution_details,
                            )
                            # Persist fresh metrics/state so REST status stays current
                            try:
                                self.artifact_store.save("autotrader", self.autotrader.to_state())
                            except Exception as e:
                                logger.warning(f"Failed to persist autotrader state: {e}")
                            completed.append(signal_id)
                            logger.debug(f"Recorded outcome for signal {signal_id}")

                    except Exception as e:
                        logger.error(f"Failed to collect outcome for {signal_id}: {e}")

                # Remove completed signals
                for signal_id in completed:
                    self._pending_signal_plan_map.pop(signal_id, None)
                    self._pending_signal_timestamps.pop(signal_id, None)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Outcome collection error: {e}")
                await asyncio.sleep(10)

    async def _signal_batch_loop(self):
        """Batch submit signals to HotPath to reduce IPC overhead."""
        await asyncio.sleep(0.2)
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(0.2)

                if not self.hotpath_client.is_connected:
                    continue

                expired_ids: list[str] = []
                batch: list[IPCTradingSignal] = []
                async with self._signal_batch_lock:
                    # Drop expired signals to avoid stale execution
                    now_ms = int(time.time() * 1000)
                    filtered: list[IPCTradingSignal] = []
                    for s in self._signal_batch_queue:
                        expires = getattr(s, "expires_at_ms", None)
                        if expires and expires < now_ms:
                            expired_ids.append(s.signal_id)
                        else:
                            filtered.append(s)
                    if expired_ids:
                        self._signal_batch_queue = filtered
                        logger.warning(
                            f"Dropped {len(expired_ids)} expired signals before batch submit"
                        )

                    if self._signal_batch_queue:
                        batch = self._signal_batch_queue[:50]
                        self._signal_batch_queue = self._signal_batch_queue[50:]

                # Best-effort mark expired signals as such in the DB outside the lock
                if expired_ids:
                    for sid in expired_ids:
                        try:
                            await self.db.update_trading_signal(
                                sid,
                                {
                                    "status": "expired",
                                    "completed_at_ms": int(time.time() * 1000),
                                    "error_message": "signal expired before submission",
                                },
                            )
                        except Exception:
                            logger.debug(f"Failed to mark signal {sid} expired", exc_info=True)

                if not batch:
                    continue

                expected_prices = [
                    getattr(s, "expected_price", None)
                    for s in batch
                    if getattr(s, "expected_price", None) is not None
                ]
                if expected_prices:
                    logger.info(
                        f"Submitting HotPath batch: size={len(batch)} "
                        f"expected_price_min={min(expected_prices)} "
                        f"expected_price_max={max(expected_prices)}"
                    )
                else:
                    logger.info(
                        f"Submitting HotPath batch: size={len(batch)} expected_price=unknown"
                    )

                result = await self.hotpath_client.submit_trading_signals_batch(
                    batch, max_parallel=10
                )

                for res in result.get("results", []):
                    if res.get("accepted"):
                        signal_id = res.get("signal_id")
                        plan_id = res.get("plan_id")
                        if signal_id and plan_id:
                            self._pending_signal_plan_map[signal_id] = plan_id
                            self._pending_signal_timestamps[signal_id] = time.time()
                            logger.info(f"Signal {signal_id} accepted (batch), plan_id={plan_id}")
                    else:
                        logger.warning(
                            f"Signal batch rejection: {res.get('signal_id')} "
                            f"reason={res.get('rejection_reason')}"
                        )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Signal batch loop error: {e}")
                await asyncio.sleep(1)

    async def _hotpath_reconnect_loop(self):
        """Keep HotPath connection alive and re-establish on disconnect."""
        await asyncio.sleep(1.0)
        while not self._shutdown_event.is_set():
            try:
                if not self.hotpath_client.is_connected:
                    self.autotrader.set_hotpath_connected(False)
                    if await self.hotpath_client.connect():
                        self.autotrader.set_hotpath_connected(True)
                        logger.info("Reconnected to HotPath")
                        await self.hotpath_client.start_event_stream(["candidate", "result"])
                        if self.config.autotrader_enabled:
                            result = await self.hotpath_client.start_autotrader()
                            if result.get("autotrader_running"):
                                logger.info("HotPath AutoPipeline started after reconnect")
                            else:
                                logger.warning(
                                    f"HotPath AutoPipeline start failed after reconnect: {result}"
                                )
                    else:
                        self.autotrader.set_hotpath_connected(False)
                        logger.warning("HotPath reconnect attempt failed")
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"HotPath reconnect loop error: {e}")
                await asyncio.sleep(5.0)

    async def shutdown(self):
        """Initiate graceful shutdown."""
        await self._cleanup()

    async def _training_loop(self):
        """Periodic training loop with error recovery."""
        # Initial delay to allow data collection
        await asyncio.sleep(60)

        while not self._shutdown_event.is_set():
            try:
                # Check if we have enough data
                trade_count = await self.db.get_trade_count(
                    since=datetime.now() - timedelta(hours=24)
                )

                if trade_count < self.config.min_trades_for_training:
                    logger.info(
                        f"Insufficient trades for training: {trade_count} < "
                        f"{self.config.min_trades_for_training}"
                    )
                else:
                    logger.info(f"Running bandit training on {trade_count} trades...")

                    # Fetch training data
                    trades_df = await self.db.get_recent_trades(hours=24)

                    # Validate data quality
                    quality_report = self.data_validator.validate_dataframe(trades_df)
                    if not quality_report.passed:
                        logger.warning(f"Data quality issues: {quality_report.issues}")

                    # Train bandit model (with timeout to prevent blocking)
                    try:
                        await asyncio.wait_for(
                            self.bandit_trainer.train(trades_df),
                            timeout=120,
                        )
                    except TimeoutError:
                        logger.error("Bandit training timed out after 120s")
                        self.metrics.errors += 1
                        await asyncio.sleep(self.config.bandit_training_interval)
                        continue

                    self.metrics.training_runs += 1
                    self.metrics.last_training = datetime.now()

                    # Persist model state
                    self.artifact_store.save("bandit", self.bandit_trainer.to_state())
                    logger.info("Bandit training complete")

                # Wait for next training window
                await asyncio.sleep(self.config.bandit_training_interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.metrics.errors += 1
                logger.error(f"Training error: {e}", exc_info=True)
                await asyncio.sleep(60)  # Short retry delay on error

    async def _calibration_loop(self):
        """Periodic calibration loop with error recovery."""
        # Initial delay
        await asyncio.sleep(30)

        while not self._shutdown_event.is_set():
            try:
                # Check if we have enough observations
                obs_count = await self.db.get_fill_observation_count(
                    since=datetime.now() - timedelta(hours=6)
                )

                if obs_count < self.config.min_observations_for_calibration:
                    logger.info(
                        f"Insufficient observations for calibration: {obs_count} < "
                        f"{self.config.min_observations_for_calibration}"
                    )
                else:
                    logger.info(f"Running paper fill calibration on {obs_count} observations...")

                    # Fetch observations
                    observations = await self.db.get_fill_observations(hours=6)

                    import pandas as pd

                    obs_df = pd.DataFrame(observations)
                    obs_report = DataQualityValidator().validate_dataframe(
                        obs_df,
                        data_type="observation",
                    )
                    if not obs_report.passed:
                        logger.warning(f"Observation quality issues: {obs_report.issues}")

                    # Calibrate paper fill parameters (legacy, with timeout)
                    try:
                        await asyncio.wait_for(
                            self.calibrator.calibrate(observations),
                            timeout=60,
                        )
                    except TimeoutError:
                        logger.error("Paper fill calibration timed out after 60s")
                        self.metrics.errors += 1
                        await asyncio.sleep(self.config.calibration_interval)
                        continue

                    # Calibrate bias parameters (advanced, with timeout)
                    try:
                        bias_report = await asyncio.wait_for(
                            self.bias_calibrator.calibrate(observations),
                            timeout=60,
                        )
                    except TimeoutError:
                        logger.error("Bias calibration timed out after 60s")
                        self.metrics.errors += 1
                        await asyncio.sleep(self.config.calibration_interval)
                        continue

                    if bias_report.converged:
                        logger.info(
                            f"Bias calibration: slippage_mae="
                            f"{bias_report.slippage_mae_bps:.1f}bps, "
                            f"inclusion={bias_report.inclusion_accuracy:.2%}, "
                            f"survivorship={bias_report.coefficients.survivorship_discount:.2%}"
                        )

                    self.metrics.calibration_runs += 1
                    self.metrics.last_calibration = datetime.now()

                    # Persist calibration states
                    self.artifact_store.save("calibration", self.calibrator.to_state())
                    self.artifact_store.save("bias_calibration", self.bias_calibrator.to_state())
                    logger.info("Calibration complete")

                # Wait for next calibration window
                await asyncio.sleep(self.config.calibration_interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.metrics.errors += 1
                logger.error(f"Calibration error: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _model_sync_loop(self):
        """Periodically sync model parameters to Hot Path."""
        while not self._shutdown_event.is_set():
            try:
                # Ensure connection to Hot Path
                if not self.publisher.is_connected:
                    try:
                        await self.publisher.connect()
                    except Exception as e:
                        logger.debug(f"Could not connect to Hot Path: {e}")
                        await asyncio.sleep(self.config.model_sync_interval)
                        continue

                # Publish current parameters
                params = {
                    "bandit": self.bandit_trainer.to_params(),
                    "calibration": self.calibrator.to_params(),
                    "bias_calibration": self.bias_calibrator.to_hotpath_params(),
                    "timestamp": datetime.now().isoformat(),
                }

                try:
                    await asyncio.wait_for(
                        self.publisher.publish(params),
                        timeout=30,
                    )
                except TimeoutError:
                    logger.error("Model parameter publish timed out after 30s")
                    self.metrics.errors += 1
                    await asyncio.sleep(self.config.model_sync_interval)
                    continue

                self.metrics.model_syncs += 1
                self.metrics.last_model_sync = datetime.now()
                logger.debug("Model parameters synced to Hot Path")

                await asyncio.sleep(self.config.model_sync_interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.metrics.errors += 1
                logger.error(f"Model sync error: {e}")
                await asyncio.sleep(30)

    async def _health_check_loop(self):
        """Periodic health check and metrics logging."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # 5 minutes

                metrics = self.metrics.to_dict()
                logger.info(
                    f"Health check - Uptime: {metrics['uptime_seconds']:.0f}s, "
                    f"Training runs: {metrics['training_runs']}, "
                    f"Calibration runs: {metrics['calibration_runs']}, "
                    f"Errors: {metrics['errors']}"
                )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _cleanup_loop(self):
        """Periodic cleanup of expired resources for 14-day stability."""
        # DB cleanup runs every 6 hours, not every cleanup_interval
        _db_cleanup_counter = 0
        _DB_CLEANUP_EVERY_N_CYCLES = 72  # 72 × 300s = 6 hours

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.cleanup_interval)

                # ── Existing cleanups (keep) ──
                # Clean up old pending counterfactuals
                await self.autotrader.outcome_tracker.cleanup_old_pending()

                # Clean up expired cache entries
                await self.kv_cache.cleanup_expired()

                # Clean up completed tasks from task queue
                self.task_queue.cleanup_completed(max_age_seconds=1800)

                # Evict stale signal-plan mappings (prevents unbounded growth)
                import time

                now = time.time()
                stale = [
                    sid
                    for sid, ts in self._pending_signal_timestamps.items()
                    if now - ts > self._signal_plan_ttl
                ]
                for sid in stale:
                    self._pending_signal_plan_map.pop(sid, None)
                    self._pending_signal_timestamps.pop(sid, None)
                if stale:
                    logger.info(f"Evicted {len(stale)} stale signal-plan mappings")

                # ── NEW: DB table pruning every 6 hours ──
                _db_cleanup_counter += 1
                if _db_cleanup_counter >= _DB_CLEANUP_EVERY_N_CYCLES:
                    _db_cleanup_counter = 0
                    await self._prune_database_tables()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _prune_database_tables(self):
        """Prune old records from DB tables (run every 6 hours)."""
        import os
        from datetime import datetime, timedelta

        retention_days = int(os.getenv("DATA_RETENTION_DAYS", "30"))
        training_retention_days = int(os.getenv("TRAINING_RETENTION_DAYS", "90"))

        try:
            db = self.db
            results = {}

            cutoff_ms = int((datetime.now() - timedelta(days=retention_days)).timestamp() * 1000)
            training_cutoff_ms = int(
                (datetime.now() - timedelta(days=training_retention_days)).timestamp() * 1000
            )

            # Clean each table
            for table, col, cutoff in [
                ("scan_outcomes", "timestamp_ms", cutoff_ms),
                ("trades", "timestamp_ms", cutoff_ms),
                ("trading_signals", "timestamp_ms", cutoff_ms),
                ("fill_observations", "timestamp_ms", cutoff_ms),
                ("monte_carlo_results", "created_at_ms", cutoff_ms),
                ("regime_history", "timestamp_ms", cutoff_ms),
                ("counterfactuals", "timestamp_ms", cutoff_ms),
                ("mi_scores_history", "timestamp_ms", cutoff_ms),
                ("training_outcomes", "created_at_ms", training_cutoff_ms),
            ]:
                try:
                    async with db._lock:
                        with db._cursor() as cursor:
                            cursor.execute(f"DELETE FROM {table} WHERE {col} < ?", (cutoff,))
                            deleted = cursor.rowcount
                            if deleted > 0:
                                results[table] = deleted
                                logger.info(f"Pruned {deleted} old rows from {table}")
                except Exception as e:
                    logger.warning(f"Could not prune {table}: {e}")

            # Incremental vacuum to reclaim space
            try:
                async with db._lock:
                    with db._cursor() as cursor:
                        cursor.execute("PRAGMA incremental_vacuum(1000)")
                logger.info("Incremental vacuum complete")
            except Exception as e:
                logger.warning(f"Vacuum failed: {e}")

            if results:
                logger.info(f"DB pruning complete: {results}")

        except Exception as e:
            logger.error(f"DB pruning error: {e}", exc_info=True)

    async def _resource_metrics_loop(self):
        """Periodic logging of resource management metrics."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Log every minute

                # Memory metrics
                mem_diagnostics = self.memory_monitor.get_diagnostics()
                mem_level = mem_diagnostics.get("pressure_level", "UNKNOWN")
                mem_percent = mem_diagnostics.get("current_stats", {}).get("percent_used", 0)

                # Task queue metrics
                queue_stats = self.task_queue.stats
                queue_size = queue_stats.current_queue_size
                queue_running = queue_stats.current_running

                # Connection pool metrics
                self.connection_pool.get_all_stats()
                circuit_states = self.connection_pool.get_circuit_states()
                open_circuits = sum(1 for s in circuit_states.values() if s.value == "open")

                # Health monitor metrics
                health_report = self.health_monitor.last_report if self.health_monitor else None
                health_status = health_report.overall_status.value if health_report else "unknown"

                logger.info(
                    f"Resource metrics - Memory: {mem_percent:.1f}% ({mem_level}), "
                    f"Queue: {queue_size} pending/{queue_running} running, "
                    f"Circuits: {open_circuits} open, Health: {health_status}"
                )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Resource metrics error: {e}")

    async def _online_learning_loop(self):
        """Online learning loop - updates models from paper outcomes (Phase 4).

        Runs every 60 seconds (configurable) to:
        1. Process recent trade outcomes into the online learner
        2. Update model weights using SGD with time decay
        3. Confirm paper outcomes that match live data
        4. Trigger drift detection checks
        """
        await asyncio.sleep(10)  # Initial delay

        while not self._shutdown_event.is_set():
            try:
                update_result = await self.online_learner.update_if_ready()
                if update_result is None:
                    await asyncio.sleep(self.config.online_learning_interval)
                    continue
                if update_result.get("status") != "updated":
                    await asyncio.sleep(self.config.online_learning_interval)
                    continue

                self.metrics.online_learning_updates += 1
                self.metrics.last_online_update = datetime.now()

                drift_alerts = self.drift_detector.check_drift()
                if drift_alerts:
                    self.metrics.drift_alerts += len(drift_alerts)
                    for alert in drift_alerts:
                        if alert.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL):
                            logger.warning(
                                f"Drift detected: {alert.drift_type.value} "
                                f"severity={alert.severity.value}, "
                                f"score={alert.score:.2f}, "
                                f"action={alert.recommended_action}"
                            )

                if self.drift_detector.should_recalibrate(
                    threshold=self.config.drift_recalibration_threshold
                ):
                    logger.info("Drift threshold exceeded - triggering fast recalibration")
                    params = self.regime_calibrator.force_calibrate(
                        self._current_regime.value,
                        self._regime_confidence,
                    )
                    logger.info(
                        f"Fast recalibration complete: "
                        f"slippage_mult={params.slippage_multiplier:.2f}, "
                        f"inclusion_mult={params.inclusion_multiplier:.2f}"
                    )

                await self._publish_online_update(update_result)

                if self.metrics.online_learning_updates % 10 == 0:
                    stats = self.online_learner.get_stats()
                    logger.info(
                        f"Online learning: updates={stats['n_updates']}, "
                        f"samples={stats['cumulative_samples']}, "
                        f"buffer={stats['buffer_size']}, "
                        f"loss={stats['recent_loss']:.4f}"
                    )

                await asyncio.sleep(self.config.online_learning_interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.metrics.errors += 1
                logger.error(f"Online learning error: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def _regime_detection_loop(self):
        """Regime detection loop - detects market regime changes (Phase 4).

        Runs every 30 seconds (configurable) to:
        1. Collect current market metrics
        2. Detect regime transitions (volume shock, MEV surge, etc.)
        3. Update slippage/inclusion parameters via EMA
        4. Push regime changes to autotrader for ML context generation
        """
        await asyncio.sleep(15)  # Initial delay (staggered from online learning)

        while not self._shutdown_event.is_set():
            try:
                # Get current market metrics from HotPath or simulate
                market_metrics = await self._get_market_metrics()

                # Process metrics through regime calibrator
                params = self.regime_calibrator.observe(market_metrics)

                # Update current regime if changed
                new_regime = MarketRegime.from_str(params.regime)
                if new_regime != self._current_regime:
                    old_regime = self._current_regime
                    self._current_regime = new_regime
                    self._regime_confidence = params.regime_confidence
                    self.metrics.current_regime = new_regime.value

                    logger.info(
                        f"Regime transition: {old_regime.value} -> {new_regime.value} "
                        f"(confidence={params.regime_confidence:.2f})"
                    )

                    # Record transition for drift detection
                    self.drift_detector.record_regime_change(
                        old_regime.value,
                        new_regime.value,
                    )

                self.metrics.regime_detections += 1
                self.metrics.last_regime_detection = datetime.now()

                # Check for active transitions
                transition = self.regime_calibrator.get_active_transition()
                if transition:
                    logger.debug(
                        f"Regime transition in progress: {transition.transition_type.value}, "
                        f"blend_progress={transition.blend_progress:.2f}"
                    )

                await asyncio.sleep(self.config.regime_detection_interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.metrics.errors += 1
                logger.error(f"Regime detection error: {e}", exc_info=True)
                await asyncio.sleep(15)

    async def _distillation_loop(self):
        """Periodically retrain teacher, distill student, and push to HotPath (Phase 5).

        Runs every distillation_interval seconds (default 1 hour) to:
        1. Check if enough new training data exists
        2. Train teacher ensemble via TeacherTrainer
        3. Distill to 7-feature student via StudentDistiller
        4. Export artifact via ArtifactExporter
        5. Push to HotPath via ModelPublisher
        """
        await asyncio.sleep(120)  # Initial delay - let other loops stabilize

        teacher_trainer = TeacherTrainer(
            db=self.db,
            config=TeacherConfig(min_outcomes=self.config.distillation_min_outcomes),
        )
        student_distiller = StudentDistiller()
        artifact_exporter = ArtifactExporter()

        while not self._shutdown_event.is_set():
            try:
                # 1. Train teacher
                logger.info("Distillation: training teacher ensemble...")
                teacher_result = await teacher_trainer.train()

                if teacher_result is None:
                    logger.info("Distillation: insufficient data or training failed, skipping")
                    await asyncio.sleep(self.config.distillation_interval)
                    continue

                # 2. Distill to student
                logger.info("Distillation: distilling student model...")
                student_result = student_distiller.distill(teacher_result)

                if student_result is None:
                    logger.warning("Distillation: student quality below threshold, skipping push")
                    self.metrics.distillation_runs += 1
                    self.metrics.last_distillation = datetime.now()
                    await asyncio.sleep(self.config.distillation_interval)
                    continue

                # 3. Export artifact
                self._distillation_version += 1
                artifact = artifact_exporter.export(
                    student=student_result,
                    version=self._distillation_version,
                    model_type="profitability",
                )

                self.metrics.distillation_runs += 1
                self.metrics.last_distillation = datetime.now()

                # 4. Push to HotPath
                async with self._model_publish_lock:
                    if not self.model_publisher.is_connected:
                        await self.model_publisher.connect()

                    if self.model_publisher.is_connected:
                        push_result = await self.model_publisher.push_artifact(artifact)
                        if push_result.success:
                            self.metrics.distillation_pushes += 1
                            self.metrics.last_distillation_version = self._distillation_version
                            logger.info(
                                f"Distillation complete: v{self._distillation_version} pushed, "
                                f"correlation={student_result.metrics.get('correlation', 0):.3f}, "
                                f"agreement={student_result.metrics.get('agreement', 0):.3f}"
                            )
                        else:
                            logger.warning(f"Distillation: push failed - {push_result.error}")
                    else:
                        logger.warning("Distillation: gRPC not connected, artifact not pushed")

                await asyncio.sleep(self.config.distillation_interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.metrics.errors += 1
                logger.error(f"Distillation error: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _get_market_metrics(self) -> MarketMetrics:
        """Get current market metrics for regime detection.

        Attempts to fetch from HotPath, falls back to database aggregates.
        """
        try:
            # Try to get metrics from HotPath
            if self.hotpath_client.is_connected:
                metrics_data = await self.hotpath_client.get_market_metrics()
                if metrics_data:
                    return MarketMetrics(
                        timestamp_ms=metrics_data.get(
                            "timestamp_ms", int(datetime.now().timestamp() * 1000)
                        ),
                        volume_24h_usd=metrics_data.get("volume_24h", 0.0),
                        volatility_1h=metrics_data.get("volatility_1h", 0.0),
                        congestion_level=metrics_data.get("congestion_level", 0.5),
                        sandwich_rate=metrics_data.get("sandwich_rate", 0.0),
                    )
        except Exception as e:
            logger.debug(f"Could not get HotPath metrics: {e}")

        # Fall back to database-based metrics
        try:
            recent_trades = await self.db.get_recent_trades(hours=1)
            if recent_trades is not None and len(recent_trades) > 0:
                volume = (
                    recent_trades["volume_usd"].sum()
                    if "volume_usd" in recent_trades.columns
                    else 0.0
                )
                (
                    recent_trades["slippage_bps"].mean()
                    if "slippage_bps" in recent_trades.columns
                    else 50.0
                )
            else:
                volume = 0.0

            return MarketMetrics(
                timestamp_ms=int(datetime.now().timestamp() * 1000),
                volume_24h_usd=volume * 24,  # Extrapolate
                volatility_1h=0.0,
                congestion_level=0.5,
                sandwich_rate=0.0,
            )
        except Exception as e:
            logger.debug(f"Could not compute database metrics: {e}")

        # Return defaults
        return MarketMetrics(
            timestamp_ms=int(datetime.now().timestamp() * 1000),
            volume_24h_usd=0.0,
            volatility_1h=0.0,
            congestion_level=0.5,
            sandwich_rate=0.0,
        )

    async def run_backtest(
        self,
        start_timestamp: int,
        end_timestamp: int,
        data_source: str = "bitquery_ohlcv",
        validate_data: bool = True,
    ) -> dict:
        """Run a backtest with optional data validation."""
        logger.info(f"Running backtest from {start_timestamp} to {end_timestamp}")

        result = await self.backtest_engine.run(
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            data_source=data_source,
            validate_data=validate_data,
        )

        self.metrics.backtests_run += 1
        return result

    def get_current_params(self) -> dict[str, Any]:
        """Get current model parameters for Hot Path."""
        return {
            "bandit": self.bandit_trainer.to_params(),
            "calibration": self.calibrator.to_params(),
            "bias_calibration": self.bias_calibrator.to_hotpath_params(),
            "autotrader": self.autotrader.get_status(),
            "timestamp": datetime.now().isoformat(),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get engine runtime metrics including supervisor status."""
        metrics = self.metrics.to_dict()
        metrics["supervisor"] = self._supervisor.get_status()
        return metrics

    async def start_autotrader(self):
        """Start the AutoTrader."""
        await self.autotrader.start()

    async def stop_autotrader(self):
        """Stop the AutoTrader."""
        await self.autotrader.stop()

    def get_autotrader_status(self) -> dict[str, Any]:
        """Get AutoTrader status."""
        return self.autotrader.get_status()


async def main():
    """Main entry point with signal handling."""
    config = ColdPathConfig.from_env()
    engine = ColdPathEngine(config)

    # Setup signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(engine.shutdown())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
