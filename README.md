# 2DEXY GPU Training Server

GPU-powered ML training and model export for the HotPath trading engine.

## Services

| Service | Port | Purpose |
|---------|------|---------|
| api | 8000 | FastAPI — training jobs, model registry |
| celery-worker | — | Background GPU training tasks |
| admin | 8501 | Streamlit dashboard |
| postgres | 5432 (internal) | Training metadata |
| redis | 6379 (internal) | Celery broker + GPU lock state |

## Live URLs

| | Public (internet) | LAN (192.168.31.99) |
|-|-------------------|---------------------|
| **API** | `https://train-api.aicd.me` | `http://192.168.31.99:32779` |
| **Admin UI** | `https://train-admin.aicd.me` | `http://192.168.31.99:32780` |
| **Swagger docs** | `https://train-api.aicd.me/docs` | `http://192.168.31.99:32779/docs` |

> LAN ports (32779 / 32780) are assigned dynamically by Docker — check
> `docker ps` after a redeploy if they change.

## Authentication

All endpoints except `/health` require a Bearer token:

```
Authorization: Bearer YOUR_API_KEY
```

The key is set via the `API_KEY` environment variable in Coolify.
Generate one with:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## API Reference

### `GET /health` — server status (no auth)

```bash
curl https://train-api.aicd.me/health
```

Response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_count": 2,
  "gpus": [
    {"id": 0, "name": "NVIDIA GeForce RTX 3090", "memory_total_mb": 24576,
     "memory_used_mb": 512, "utilization_pct": 0}
  ],
  "workers_online": 1,
  "gpu_allocated": {},
  "active_jobs": 0,
  "cpu_percent": 4.2,
  "memory_percent": 18.5
}
```

---

### `POST /data/upload` — upload a training dataset

Accepts CSV, Parquet, or JSON (newline-delimited). Max 500 MB.

```bash
curl -X POST https://train-api.aicd.me/data/upload \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@trades.csv" \
  -F "name=solana-trades-feb-2026" \
  -F "description=HotPath trade outcomes with features" \
  -F "label_column=profitable"
```

Response:
```json
{
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "solana-trades-feb-2026",
  "row_count": 142857,
  "feature_count": 32,
  "checksum": "sha256:abc123..."
}
```

---

### `POST /data/feedback` — push live trade outcomes

Send HotPath trade results for online learning. Max 500 trades per batch.

```bash
curl -X POST https://train-api.aicd.me/data/feedback \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "trades": [
      {
        "trade_id": "abc123",
        "timestamp_ms": 1708531200000,
        "mint": "So11111111111111111111111111111111111111112",
        "features": [0.5, 1.2, 0.8],
        "feature_names": ["liquidity_norm", "vol_ratio", "age_score"],
        "decision": "buy",
        "confidence": 0.72,
        "amount_in_sol": 0.1,
        "amount_out_tokens": 50000,
        "pnl_sol": 0.035,
        "slippage_bps": 180,
        "execution_latency_ms": 320,
        "model_version": 3
      }
    ]
  }'
```

Response:
```json
{"accepted": 1, "rejected": 0, "total_feedback_records": 9821}
```

---

### `POST /jobs/train` — launch a training job

Dispatches an async Celery task. Returns immediately with a `job_id` to poll.

```bash
curl -X POST https://train-api.aicd.me/jobs/train \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "ensemble",
    "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
    "hyperparams": {
      "label_column": "profitable",
      "n_estimators": 500,
      "max_depth": 6,
      "learning_rate": 0.01,
      "ensemble_weights": {"linear": 0.4, "lgbm": 0.3, "xgb": 0.3}
    },
    "gpu_id": null
  }'
```

**`job_type` values:**

| Value | Model | Required dataset columns |
|-------|-------|--------------------------|
| `ensemble` | Linear + LightGBM + XGBoost | numeric features + label column |
| `fraud` | RandomForest + MI feature selection | `liquidity_usd`, `fdv_usd`, `holder_count`, `top_holder_pct`, `mint_authority_enabled`, `freeze_authority_enabled`, `lp_burn_pct`, `age_seconds`, `volume_24h`, `price_change_1h` + label (`is_rug`) |
| `bandit` | UCB / Thompson Sampling | `quoted_slippage_bps`, `slippage_bps`, `pnl_pct`, `included` |
| `slippage` | L-BFGS-B linear regression | `quoted_slippage_bps`, `realized_slippage_bps`, `liquidity_usd`, `volume_usd`, `volatility`, `latency_ms` |
| `regime` | Rule-based + scoring | `total_volume_24h_sol`, `avg_volume_7d_sol`, `new_tokens_24h`, `sol_price_change_7d_pct`, `volatility_24h`, `sandwich_attacks_24h`, `jito_bundle_rate` |

**`hyperparams` by job type:**

`ensemble`:
```json
{
  "label_column": "profitable",
  "n_estimators": 500,
  "max_depth": 6,
  "learning_rate": 0.01,
  "C": 1.0,
  "ensemble_weights": {"linear": 0.4, "lgbm": 0.3, "xgb": 0.3}
}
```

`fraud`:
```json
{"label_column": "is_rug"}
```

`bandit`:
```json
{
  "strategy": "ucb",
  "exploration_param": 2.0,
  "reward_decay": 0.99
}
```

`slippage`:
```json
{
  "quoted_col": "quoted_slippage_bps",
  "realized_col": "realized_slippage_bps",
  "liquidity_col": "liquidity_usd",
  "volume_col": "volume_usd",
  "volatility_col": "volatility",
  "latency_col": "latency_ms"
}
```

Response:
```json
{
  "job_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "status": "pending",
  "message": "Training job queued"
}
```

---

### `GET /jobs/{job_id}/status` — poll a training job

```bash
curl https://train-api.aicd.me/jobs/7c9e6679-7425-40de-944b-e07fc1f90ae7/status \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Response (running):
```json
{
  "job_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "job_type": "ensemble",
  "status": "running",
  "gpu_id": 0,
  "metrics": {},
  "error_message": null,
  "started_at": "2026-02-21T18:00:00+00:00",
  "completed_at": null,
  "elapsed_seconds": 42
}
```

Response (completed):
```json
{
  "status": "completed",
  "metrics": {
    "accuracy": 0.731,
    "auc_roc": 0.814,
    "f1_score": 0.698,
    "train_samples": 114285,
    "validation_samples": 28572
  },
  "elapsed_seconds": 387
}
```

`status` values: `pending` → `running` → `completed` | `failed`

---

### `GET /models/manifest` — list promoted models

```bash
curl https://train-api.aicd.me/models/manifest \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Response:
```json
{
  "models": [
    {
      "id": "artifact-uuid",
      "model_type": "profitability",
      "version": 3,
      "sha256": "abc123...",
      "created_at": "2026-02-21T19:30:00+00:00",
      "metrics": {"accuracy": 0.731, "auc_roc": 0.814}
    }
  ]
}
```

`model_type` values: `profitability` | `fraud` | `slippage` | `regime`

---

### `GET /models/latest/{model_type}` — download a model artifact

Returns the full artifact JSON that HotPath loads. This is the promoted version.

```bash
curl https://train-api.aicd.me/models/latest/profitability \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Response (the full artifact consumed by HotPath Rust engine):
```json
{
  "model_type": "profitability",
  "version": 3,
  "schema_version": 1,
  "feature_signature": ["vol_ratio", "liquidity_norm", ...],
  "weights": {
    "student_type": "logistic",
    "linear_weights": [0.42, -0.18, ...],
    "bias": -0.05
  },
  "calibration": {
    "calibration_method": "platt",
    "temperature": 1.0,
    "platt_coeffs": [-1.0, 0.0]
  },
  "checksum": "sha256hex..."
}
```

---

### `POST /models/promote/{artifact_id}` — promote a model to latest

Demotes the current promoted model of the same type, promotes the new one.

```bash
curl -X POST https://train-api.aicd.me/models/promote/ARTIFACT_UUID \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Response:
```json
{"promoted": true, "model_type": "profitability", "version": 4}
```

---

## Typical workflow

```bash
API=https://train-api.aicd.me
KEY=your_api_key

# 1. Upload dataset
DATASET=$(curl -s -X POST $API/data/upload \
  -H "Authorization: Bearer $KEY" \
  -F "file=@trades.csv" -F "name=feb-2026" | python3 -c "import sys,json; print(json.load(sys.stdin)['dataset_id'])")

# 2. Train
JOB=$(curl -s -X POST $API/jobs/train \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"job_type\":\"ensemble\",\"dataset_id\":\"$DATASET\"}" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# 3. Poll until done
while true; do
  STATUS=$(curl -s $API/jobs/$JOB/status -H "Authorization: Bearer $KEY" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "Status: $STATUS"
  [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]] && break
  sleep 10
done

# 4. Find the artifact and promote it
curl -s $API/models/manifest -H "Authorization: Bearer $KEY"
curl -s -X POST $API/models/promote/ARTIFACT_UUID -H "Authorization: Bearer $KEY"

# 5. HotPath can now fetch it
curl -s $API/models/latest/profitability -H "Authorization: Bearer $KEY"
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_KEY` | yes | Bearer token for all authenticated endpoints |
| `POSTGRES_PASSWORD` | yes | PostgreSQL password |
| `DATABASE_URL` | auto | Set by Coolify — `postgresql+asyncpg://...` |
| `CELERY_BROKER_URL` | auto | Set by Coolify — `redis://redis:6379/0` |
| `REDIS_URL` | auto | Set by Coolify — `redis://redis:6379/0` |
| `MODEL_ARTIFACT_PATH` | no | Where artifact JSON files are stored (default: `/data/models`) |
| `TRAINING_DATA_PATH` | no | Where uploaded datasets are stored (default: `/data/datasets`) |
| `CUDA_VISIBLE_DEVICES` | no | GPU indices to expose to workers (default: `0,1`) |

## Code Server Setup

Set `CODE_SERVER_PASSWORD` in Coolify Secrets. First time in, open the terminal and clone the repo:

```bash
cd ~/workspace
git clone https://github.com/machine-machine/trading-2dexy.git
cd trading-2dexy
git checkout 3090-training
```
