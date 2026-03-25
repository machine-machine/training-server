"""Celery tasks for GPU training jobs."""

import json
import logging
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.workers.celery_app import app
from src.workers.gpu_manager import gpu_manager

logger = logging.getLogger(__name__)

MODEL_ARTIFACT_PATH = Path(os.environ.get("MODEL_ARTIFACT_PATH", "/data/models"))
TRAINING_DATA_PATH = Path(os.environ.get("TRAINING_DATA_PATH", "/data/datasets"))


def _detect_gpu() -> bool:
    """Check if CUDA GPU is available inside the container."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _parse_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _get_sync_session():
    """Create a synchronous DB session for Celery tasks."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    db_url = os.environ.get("DATABASE_URL", "").replace("+asyncpg", "")
    engine = create_engine(db_url)
    return Session(engine)


@app.task(bind=True, max_retries=1, time_limit=7200)  # 2h timeout
def run_training_job(self, job_id: str):
    """Execute a training job on GPU (falls back to CPU if unavailable)."""
    session = _get_sync_session()
    gpu_id = None
    use_gpu = False

    try:
        from src.db.models import ModelArtifact, TrainingJob, TrainingLog

        job = session.get(TrainingJob, job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        # Allocate GPU (fall back to CPU if unavailable)
        gpu_id = gpu_manager.allocate(job_id, job.gpu_id)
        use_gpu = gpu_id is not None and _detect_gpu()

        # Update job status
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        job.gpu_id = gpu_id
        job.worker_id = self.request.hostname
        session.commit()

        device_label = f"GPU {gpu_id}" if use_gpu else "CPU"
        _log(session, job_id, "INFO", f"Starting {job.job_type} training on {device_label}")

        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Dispatch to the appropriate trainer
        metrics = _dispatch_training(job.job_type, job.hyperparams or {}, job.dataset_id, session, job_id, use_gpu)

        # Determine next version before writing the artifact file
        from sqlalchemy import func, select

        max_version_stmt = (
            select(func.coalesce(func.max(ModelArtifact.version), 0))
            .where(ModelArtifact.model_type == _job_type_to_model_type(job.job_type))
        )
        next_version = (session.execute(max_version_stmt).scalar() or 0) + 1

        # Export model artifact — version is baked into the file and checksum
        artifact_path = _export_artifact(job, metrics, next_version)

        # Register artifact in DB
        artifact = ModelArtifact(
            job_id=job.id,
            model_type=_job_type_to_model_type(job.job_type),
            version=next_version,
            file_path=str(artifact_path),
            sha256_checksum=metrics.get("checksum", ""),
            metrics=metrics,
            feature_signature=metrics.get("feature_signature"),
        )
        session.add(artifact)

        # Mark job complete
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        job.metrics = metrics
        session.commit()

        _log(session, job_id, "INFO", f"Training completed. Artifact v{next_version}")

        # Push artifact to HotPath for hot-reload (after DB commit)
        _push_artifact_to_hotpath(
            _load_artifact_dict(artifact_path),
            session,
            job_id,
            next_version,
        )

    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        logger.error(traceback.format_exc())
        try:
            job = session.get(TrainingJob, job_id)
            if job:
                job.status = "failed"
                job.error_message = str(e)
                job.completed_at = datetime.now(timezone.utc)
                session.commit()
            _log(session, job_id, "ERROR", f"Training failed: {e}")
        except Exception:
            pass

    finally:
        if gpu_id is not None:
            gpu_manager.release(gpu_id)
        session.close()


def _dispatch_training(
    job_type: str,
    hyperparams: dict,
    dataset_id,
    session,
    job_id: str,
    use_gpu: bool = False,
) -> dict:
    """Route to the correct training module based on job type."""
    # Load dataset
    from src.db.models import TrainingDataset
    import pandas as pd

    dataset = session.get(TrainingDataset, str(dataset_id))
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")

    _log(session, job_id, "INFO", f"Loading dataset: {dataset.name} ({dataset.row_count} rows)")

    if dataset.file_format == "csv":
        df = pd.read_csv(dataset.file_path)
    elif dataset.file_format == "parquet":
        df = pd.read_parquet(dataset.file_path)
    elif dataset.file_format == "json":
        df = pd.read_json(dataset.file_path, lines=True)
    else:
        raise ValueError(f"Unsupported format: {dataset.file_format}")

    if job_type == "ensemble":
        return _train_ensemble(df, hyperparams, session, job_id, use_gpu)
    elif job_type == "fraud":
        return _train_fraud(df, hyperparams, session, job_id)
    elif job_type in ("rl", "bandit"):
        return _train_bandit(df, hyperparams, session, job_id)
    elif job_type == "slippage":
        return _train_slippage(df, hyperparams, session, job_id)
    elif job_type == "regime":
        return _train_regime(df, hyperparams, session, job_id)
    else:
        raise ValueError(f"Unknown job type: {job_type}")


def _train_ensemble(df, hyperparams: dict, session, job_id: str, use_gpu: bool = False) -> dict:
    """Train ensemble model (Linear + LightGBM + XGBoost + LSTM)."""
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit

    _log(session, job_id, "INFO", f"Training ensemble model ({'GPU' if use_gpu else 'CPU'})")

    label_col = hyperparams.get("label_column", "profitable")
    seed = _parse_int(hyperparams.get("seed", 42), 42)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset")

    # Keep ensemble training reproducible across smoke/prod runs when requested.
    np.random.seed(seed)

    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    y = df[label_col].values

    # Use TimeSeriesSplit to avoid temporal leakage (ML finding #2)
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, val_idx = list(tscv.split(X))[-1]
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    _log(session, job_id, "INFO", f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    # Train linear component
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(
        max_iter=1000,
        C=hyperparams.get("C", 1.0),
        random_state=seed,
    )
    lr.fit(X_train, y_train)
    lr_pred = lr.predict_proba(X_val)[:, 1]

    # Train LightGBM
    import lightgbm as lgb

    lgb_params = {
        "n_estimators": hyperparams.get("n_estimators", 500),
        "max_depth": hyperparams.get("max_depth", 6),
        "learning_rate": hyperparams.get("learning_rate", 0.01),
        "random_state": seed,
    }
    if use_gpu:
        lgb_params["device"] = "gpu"
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]

    # Train XGBoost
    import xgboost as xgb

    xgb_params = {
        "n_estimators": hyperparams.get("n_estimators", 500),
        "max_depth": hyperparams.get("max_depth", 6),
        "learning_rate": hyperparams.get("learning_rate", 0.01),
        "random_state": seed,
    }
    if use_gpu:
        xgb_params["tree_method"] = "gpu_hist"
        xgb_params["device"] = "cuda"
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]

    # Train LSTM Price Predictor (GPU-only, opt-in for robustness until feature/sequence
    # semantics are validated against real profitability data).
    lstm_pred = None
    lstm_metrics = {}
    enable_lstm = bool(hyperparams.get("enable_lstm", False))
    if "lstm_epochs" in hyperparams:
        lstm_epochs = _parse_int(hyperparams.get("lstm_epochs"), 0)
    elif enable_lstm and use_gpu:
        lstm_epochs = 50
    else:
        lstm_epochs = 0
    
    if lstm_epochs > 0 and use_gpu:
        try:
            _log(session, job_id, "INFO", f"Training LSTM price predictor ({lstm_epochs} epochs)")
            
            # Generate price sequences from numeric features
            # Use first 5 features as price-related: price, volume, volatility, momentum, order_flow
            price_cols = list(X.columns[:5]) if len(X.columns) >= 5 else list(X.columns)
            price_data = X_train[price_cols].values
            
            # Create sequences for LSTM (SEQUENCE_LENGTH=60 timesteps)
            SEQUENCE_LENGTH = 60
            sequences = []
            targets = []
            
            # Normalize price data
            price_mean = price_data.mean(axis=0)
            price_std = price_data.std(axis=0) + 1e-8
            price_normalized = (price_data - price_mean) / price_std
            
            # Create sliding window sequences
            for i in range(len(price_normalized) - SEQUENCE_LENGTH - 15):  # 15 = max horizon
                seq = price_normalized[i:i + SEQUENCE_LENGTH]
                # Target: price change at T+5, T+10, T+15
                target = [
                    price_normalized[i + SEQUENCE_LENGTH + 5, 0] if i + SEQUENCE_LENGTH + 5 < len(price_normalized) else 0,
                    price_normalized[i + SEQUENCE_LENGTH + 10, 0] if i + SEQUENCE_LENGTH + 10 < len(price_normalized) else 0,
                    price_normalized[i + SEQUENCE_LENGTH + 15, 0] if i + SEQUENCE_LENGTH + 15 < len(price_normalized) else 0,
                ]
                sequences.append(seq)
                targets.append(target)
            
            if len(sequences) > 100:
                import torch
                from src.learning.lstm_predictor import LSTMPricePredictor

                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                
                sequences_arr = np.array(sequences, dtype=np.float32)
                targets_arr = np.array(targets, dtype=np.float32)
                
                # Clip extreme values
                sequences_arr = np.clip(sequences_arr, -10.0, 10.0)
                targets_arr = np.clip(targets_arr, -10.0, 10.0)
                
                lstm_predictor = LSTMPricePredictor()
                lstm_predictor.fit(sequences_arr, targets_arr, epochs=lstm_epochs)
                
                # Generate predictions for validation set
                val_price_data = X_val[price_cols].values
                val_normalized = (val_price_data - price_mean) / price_std
                
                lstm_preds_list = []
                for i in range(len(val_normalized) - SEQUENCE_LENGTH):
                    seq = val_normalized[i:i + SEQUENCE_LENGTH].reshape(1, SEQUENCE_LENGTH, -1)
                    pred = lstm_predictor.predict(seq, val_price_data[i + SEQUENCE_LENGTH - 1, 0])
                    # Use T+5 return as confidence signal
                    lstm_preds_list.append((pred.return_t5 + 10) / 20)  # Normalize to 0-1
                
                if lstm_preds_list:
                    # Pad to match validation size
                    lstm_pred = np.array(lstm_preds_list + [0.5] * (len(y_val) - len(lstm_preds_list)))
                
                lstm_metrics = {
                    "lstm_enabled": True,
                    "lstm_epochs": lstm_epochs,
                    "lstm_sequences": len(sequences),
                    "lstm_fitted": lstm_predictor.is_fitted,
                }
                _log(session, job_id, "INFO", f"LSTM trained: {len(sequences)} sequences, fitted={lstm_predictor.is_fitted}")
            else:
                _log(session, job_id, "WARN", f"Not enough sequences for LSTM ({len(sequences)}), skipping")
                
        except Exception as e:
            _log(session, job_id, "WARN", f"LSTM training failed, continuing without: {e}")
            lstm_pred = None
    else:
        _log(
            session,
            job_id,
            "INFO",
            f"LSTM training skipped (enabled={enable_lstm}, epochs={lstm_epochs}, gpu={use_gpu})",
        )

    # Ensemble
    weights = hyperparams.get("ensemble_weights", {"linear": 0.3, "lgbm": 0.25, "xgb": 0.25, "lstm": 0.2})
    w_lr = weights.get("linear", 0.3)
    w_lgb = weights.get("lgbm", 0.25)
    w_xgb = weights.get("xgb", 0.25)
    w_lstm = weights.get("lstm", 0.2) if lstm_pred is not None else 0.0
    
    # Renormalize weights
    total_w = w_lr + w_lgb + w_xgb + w_lstm
    w_lr, w_lgb, w_xgb, w_lstm = w_lr/total_w, w_lgb/total_w, w_xgb/total_w, w_lstm/total_w
    
    ensemble_pred = w_lr * lr_pred + w_lgb * lgb_pred + w_xgb * xgb_pred
    if lstm_pred is not None:
        ensemble_pred += w_lstm * lstm_pred

    # Metrics
    from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score

    y_pred_binary = (ensemble_pred >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred_binary)),
        "precision": float(precision_score(y_val, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred_binary, zero_division=0)),
        "f1_score": float(f1_score(y_val, y_pred_binary, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_val, ensemble_pred)),
        "log_loss": float(log_loss(y_val, ensemble_pred)),
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "linear_weights": lr.coef_[0].tolist(),
        "bias": float(lr.intercept_[0]),
        "feature_names": list(X.columns),
        "feature_signature": list(X.columns),
        "seed": seed,
        "used_gpu": bool(use_gpu),
        "ensemble_config": {"linear_w": w_lr, "lgbm_w": w_lgb, "xgb_w": w_xgb, "lstm_w": w_lstm},
        **lstm_metrics,
    }

    _log(session, job_id, "INFO", f"Ensemble metrics: acc={metrics['accuracy']:.3f} auc={metrics['auc_roc']:.3f} lstm={'yes' if lstm_pred is not None else 'no'}")
    return metrics


def _train_fraud(df, hyperparams: dict, session, job_id: str) -> dict:
    """Train fraud detection model (RandomForest with MI feature selection)."""
    import numpy as np
    from src.training.fraud_model import FraudModel, TokenFeatures

    _log(session, job_id, "INFO", "Training fraud detection model")

    expected = [
        "liquidity_usd", "fdv_usd", "holder_count", "top_holder_pct",
        "mint_authority_enabled", "freeze_authority_enabled", "lp_burn_pct",
        "age_seconds", "volume_24h", "price_change_1h",
    ]
    label_col = hyperparams.get("label_column", "is_rug")

    # Build TokenFeatures list from whichever expected columns exist
    available = [c for c in expected if c in df.columns]
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset")

    tokens, labels = [], []
    for _, row in df.dropna(subset=[label_col]).iterrows():
        tokens.append(TokenFeatures(**{
            c: bool(row[c]) if c in ("mint_authority_enabled", "freeze_authority_enabled")
            else float(row.get(c, 0))
            for c in expected
        }))
        labels.append(int(row[label_col]))

    if len(tokens) < 20:
        raise ValueError(f"Need at least 20 labelled rows for fraud model, got {len(tokens)}")

    model = FraudModel(use_automl=False)
    result = model.train(tokens, labels)

    _log(session, job_id, "INFO",
         f"Fraud model trained: acc={result.accuracy:.3f} cv={result.cv_score:.3f}")

    return {
        "model_type": "fraud",
        "accuracy": float(result.accuracy),
        "cv_score": float(result.cv_score),
        "automl_used": result.automl_used,
        "feature_importance": result.feature_importance,
        "train_samples": len(tokens),
        "feature_signature": model.selected_feature_names or model.feature_names,
        "feature_names": model.selected_feature_names or model.feature_names,
        "linear_weights": [],
        "bias": 0.0,
    }


def _train_bandit(df, hyperparams: dict, session, job_id: str) -> dict:
    """Train contextual bandit on historical trade data."""
    import pandas as pd
    from src.training.bandit import BanditTrainer, ExplorationStrategy

    _log(session, job_id, "INFO", "Training contextual bandit")

    strategy_name = hyperparams.get("strategy", "ucb")
    try:
        strategy = ExplorationStrategy(strategy_name)
    except ValueError:
        strategy = ExplorationStrategy.UCB

    trainer = BanditTrainer(
        exploration_param=hyperparams.get("exploration_param", 2.0),
        strategy=strategy,
        reward_decay=hyperparams.get("reward_decay", 0.99),
    )

    # Synchronously process each trade row without a DB session
    trades_processed = 0
    total_reward = 0.0
    for _, row in df.iterrows():
        trade = row.to_dict()
        quoted_slippage = trade.get("quoted_slippage_bps", trade.get("slippage_bps", 300))
        arm_name = trainer._classify_arm(int(quoted_slippage))
        reward = trainer.calculate_reward(trade)
        trainer.update(arm_name, reward)
        trades_processed += 1
        total_reward += reward

    recommendation = trainer.get_recommendation()
    params = trainer.to_params()

    _log(session, job_id, "INFO",
         f"Bandit trained: {trades_processed} trades, "
         f"recommended={recommendation['arm']} ({recommendation['slippage_bps']}bps)")

    return {
        "model_type": "bandit",
        "train_samples": trades_processed,
        "avg_reward": float(total_reward / max(trades_processed, 1)),
        "recommended_arm": recommendation["arm"],
        "recommended_slippage_bps": recommendation["slippage_bps"],
        "confidence": float(recommendation.get("confidence", 0)),
        "arm_weights": params["arm_weights"],
        "arm_stats": params["arm_stats"],
        "feature_signature": [],
        "linear_weights": [],
        "bias": 0.0,
    }


def _train_slippage(df, hyperparams: dict, session, job_id: str) -> dict:
    """Train slippage prediction model via scipy L-BFGS-B optimisation."""
    from src.training.slippage_model import SlippageModel, SlippageObservation

    _log(session, job_id, "INFO", "Training slippage model")

    col_map = {
        "quoted": hyperparams.get("quoted_col", "quoted_slippage_bps"),
        "realized": hyperparams.get("realized_col", "realized_slippage_bps"),
        "liquidity": hyperparams.get("liquidity_col", "liquidity_usd"),
        "volume": hyperparams.get("volume_col", "volume_usd"),
        "volatility": hyperparams.get("volatility_col", "volatility"),
        "latency": hyperparams.get("latency_col", "latency_ms"),
    }

    missing = [v for v in col_map.values() if v not in df.columns]
    if missing:
        raise ValueError(f"Slippage dataset missing columns: {missing}")

    observations = [
        SlippageObservation(
            quoted_slippage_bps=float(r[col_map["quoted"]]),
            realized_slippage_bps=float(r[col_map["realized"]]),
            liquidity_usd=float(r[col_map["liquidity"]]),
            volume_usd=float(r[col_map["volume"]]),
            volatility=float(r[col_map["volatility"]]),
            latency_ms=int(r[col_map["latency"]]),
        )
        for _, r in df.dropna().iterrows()
    ]

    if len(observations) < 10:
        raise ValueError(f"Need at least 10 rows for slippage model, got {len(observations)}")

    model = SlippageModel()
    model.fit(observations)
    params = model.to_params()

    # Compute in-sample MAE
    errors = [
        abs(model.predict(o) - o.realized_slippage_bps) for o in observations
    ]
    mae = sum(errors) / len(errors)

    _log(session, job_id, "INFO", f"Slippage model trained: MAE={mae:.1f} bps, n={len(observations)}")

    return {
        "model_type": "slippage",
        "train_samples": len(observations),
        "mae_bps": float(mae),
        **params,
        "feature_signature": list(col_map.values()),
        "feature_names": list(col_map.values()),
        "linear_weights": [
            params["slippage_quote_coef"],
            params["slippage_liquidity_coef"],
            params["slippage_volatility_coef"],
            params["slippage_latency_coef"],
        ],
        "bias": float(params["slippage_base_bps"]),
    }


def _train_regime(df, hyperparams: dict, session, job_id: str) -> dict:
    """Fit regime detector on market snapshot data."""
    import time
    from src.training.regime_detector import RegimeDetector, MarketSnapshot

    _log(session, job_id, "INFO", "Training regime detector")

    # Columns → MarketSnapshot fields (best-effort mapping)
    col = lambda name, default=0.0: float(df[name].mean()) if name in df.columns else default

    snapshots = []
    for _, row in df.iterrows():
        def g(field, default=0.0):
            return float(row[field]) if field in df.columns else default

        snapshots.append(MarketSnapshot(
            timestamp_ms=int(g("timestamp_ms", time.time() * 1000)),
            total_volume_24h_sol=g("total_volume_24h_sol"),
            avg_volume_7d_sol=g("avg_volume_7d_sol"),
            new_tokens_24h=int(g("new_tokens_24h")),
            avg_new_tokens_7d=g("avg_new_tokens_7d"),
            sol_price_change_7d_pct=g("sol_price_change_7d_pct"),
            volatility_24h=g("volatility_24h"),
            avg_volatility_7d=g("avg_volatility_7d"),
            sandwich_attacks_24h=int(g("sandwich_attacks_24h")),
            avg_sandwich_attacks_7d=g("avg_sandwich_attacks_7d"),
            jito_bundle_rate=g("jito_bundle_rate"),
            pools_below_10k_usd_pct=g("pools_below_10k_usd_pct"),
            avg_slippage_500_sol_bps=int(g("avg_slippage_500_sol_bps")),
        ))

    if not snapshots:
        raise ValueError("No rows with regime snapshot data")

    detector = RegimeDetector()
    regime_counts: dict[str, int] = {}
    for snap in snapshots:
        classification = detector.classify(snap)
        regime_counts[classification.regime.value] = (
            regime_counts.get(classification.regime.value, 0) + 1
        )

    # Most frequent regime from dataset
    dominant = max(regime_counts, key=regime_counts.get)
    adjustments = detector.get_recommended_adjustments()

    _log(session, job_id, "INFO",
         f"Regime model trained: {len(snapshots)} snapshots, dominant={dominant}")

    return {
        "model_type": "regime",
        "train_samples": len(snapshots),
        "dominant_regime": dominant,
        "regime_distribution": regime_counts,
        "recommended_adjustments": adjustments,
        "feature_signature": [],
        "linear_weights": [],
        "bias": 0.0,
    }


def _export_artifact(job, metrics: dict, version: int) -> Path:
    """Export trained model to ModelArtifact JSON file.

    ``version`` must be resolved by the caller before this function is called
    so that the artifact JSON and its SHA-256 checksum are consistent with
    what gets stored in the database.
    """
    import hashlib
    import struct

    model_type = _job_type_to_model_type(job.job_type)
    artifact_dir = MODEL_ARTIFACT_PATH / model_type
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_type": model_type,
        "version": version,
        "schema_version": 1,
        "feature_signature": metrics.get("feature_names", []),
        "feature_transforms": [],
        "dataset_id": str(job.dataset_id) if job.dataset_id else "",
        "metrics": {
            k: metrics.get(k, 0)
            for k in [
                "accuracy", "precision", "recall", "f1_score",
                "auc_roc", "log_loss", "train_samples", "validation_samples",
            ]
        },
        "created_at": int(datetime.now(timezone.utc).timestamp() * 1000),
        "git_commit": "",
        "calibration": {
            "calibration_method": "platt",
            "confidence_threshold": 0.5,
            "temperature": 1.0,
            "platt_coeffs": [-1.0, 0.5],
        },
        "compatibility_notes": f"GPU-trained {model_type} model",
        "weights": {
            "student_type": "logistic",
            "linear_weights": metrics.get("linear_weights", []),
            "bias": metrics.get("bias", 0.0),
            "onnx_model": None,
        },
    }

    # Compute checksum matching Rust's compute_checksum()
    hasher = hashlib.sha256()
    hasher.update(artifact["model_type"].encode("utf-8"))
    hasher.update(struct.pack("<I", version))
    hasher.update(struct.pack("<I", artifact["schema_version"]))
    for name in artifact["feature_signature"]:
        hasher.update(name.encode("utf-8"))
    for w in artifact["weights"]["linear_weights"]:
        hasher.update(struct.pack("<d", w))
    hasher.update(struct.pack("<d", artifact["weights"]["bias"]))
    # Use confidence_threshold directly from calibration (matches Rust)
    hasher.update(struct.pack("<d", artifact["calibration"]["confidence_threshold"]))
    hasher.update(struct.pack("<d", artifact["calibration"]["temperature"]))
    hasher.update(artifact["dataset_id"].encode("utf-8"))
    hasher.update(artifact["git_commit"].encode("utf-8"))

    artifact["checksum"] = hasher.hexdigest()
    metrics["checksum"] = artifact["checksum"]

    # Write artifact to disk
    artifact_path = artifact_dir / f"artifact_{job.id}.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)

    return artifact_path


def _job_type_to_model_type(job_type: str) -> str:
    """Map job type to ModelArtifact model_type."""
    return {
        "ensemble": "profitability",
        "fraud": "fraud",
        "rl": "profitability",
        "bandit": "profitability",
        "slippage": "slippage",
        "regime": "regime",
    }.get(job_type, "profitability")


@app.task(time_limit=10)
def get_gpu_status():
    """Lightweight task: return GPU info from the worker container."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            gpus.append(
                {
                    "id": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "utilization_pct": int(parts[4]),
                }
            )
        return {
            "gpus": gpus,
            "allocated": dict(gpu_manager._allocated),
            "worker_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "worker_nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        }
    except Exception as e:
        return {
            "gpus": [],
            "error": str(e),
            "worker_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "worker_nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        }


def _log(session, job_id: str, level: str, message: str):
    """Write a structured training log entry."""
    from src.db.models import TrainingLog

    log = TrainingLog(job_id=job_id, level=level, message=message)
    session.add(log)
    session.commit()
    logger.info(f"[{job_id[:8]}] {message}")


def _load_artifact_dict(artifact_path: Path) -> dict:
    """Load artifact JSON from disk for HotPath push."""
    with open(artifact_path, "r") as f:
        return json.load(f)


def _push_artifact_to_hotpath(
    artifact_dict: dict, session, job_id: str, version: int
) -> bool:
    """Push the trained artifact to HotPath via gRPC for hot-reload.

    Args:
        artifact_dict: The artifact dictionary (will be updated with version)
        session: DB session for logging
        job_id: Job ID for logging
        version: The version number assigned to this artifact

    Returns:
        True if push succeeded or was skipped, False on error
    """
    # Check if HotPath push is enabled
    if os.environ.get("HOTPATH_PUSH_ENABLED", "true").lower() != "true":
        _log(session, job_id, "INFO", "HotPath push disabled via HOTPATH_PUSH_ENABLED")
        return True

    try:
        from src.ipc.hotpath_client import push_artifact_sync

        # Update artifact with the correct version
        artifact_dict["version"] = version

        _log(session, job_id, "INFO", f"Pushing artifact v{version} to HotPath...")

        result = push_artifact_sync(artifact_dict, timeout=30.0)

        if result.success:
            _log(
                session,
                job_id,
                "INFO",
                f"Artifact pushed to HotPath: version={result.applied_version}, "
                f"replaced_previous={result.replaced_previous}",
            )
            return True
        else:
            _log(
                session, job_id, "WARN", f"Failed to push artifact to HotPath: {result.error}"
            )
            # Don't fail the job - local artifact is still saved
            return True

    except ImportError as e:
        _log(
            session,
            job_id,
            "WARN",
            f"HotPath client not available (grpcio not installed?): {e}",
        )
        return True  # Don't fail the job

    except Exception as e:
        _log(session, job_id, "ERROR", f"Unexpected error pushing to HotPath: {e}")
        # Don't fail the job - local artifact is still saved
        return True
