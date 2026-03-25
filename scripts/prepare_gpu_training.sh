#!/bin/bash
# =============================================================================
# 2DEXY GPU Training Preparation Script
# =============================================================================
# This script prepares everything needed to start GPU training:
# 1. Generates synthetic training data (if needed)
# 2. Uploads it to the training server
# 3. Launches a training job
# 4. Waits for completion and promotes the model
#
# Usage:
#   ./prepare_gpu_training.sh --api-key YOUR_KEY --api-base http://localhost:8000
#   ./prepare_gpu_training.sh --api-key YOUR_KEY --api-base https://train-api.aicd.me
# =============================================================================

set -e

# Default values
API_BASE="${API_BASE:-http://localhost:8000}"
API_KEY="${API_KEY:-}"
JOB_TYPE="${JOB_TYPE:-ensemble}"
ROWS="${ROWS:-5000}"
SEED="${SEED:-42}"
LABEL_COLUMN="${LABEL_COLUMN:-profitable}"
GPU_ID="${GPU_ID:-}"
POLL_INTERVAL="${POLL_INTERVAL:-3}"
TIMEOUT="${TIMEOUT:-3600}"
AUTO_PROMOTE="${AUTO_PROMOTE:-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --api-base URL       Training server base URL (default: http://localhost:8000)
    --api-key KEY        API key for authentication (required)
    --job-type TYPE      Job type: ensemble, fraud, bandit, slippage, regime (default: ensemble)
    --rows N             Number of synthetic rows to generate (default: 5000)
    --seed N             Random seed (default: 42)
    --label-column COL   Label column name (default: profitable)
    --gpu-id ID          Pin to specific GPU (optional)
    --poll-interval SEC  Polling interval in seconds (default: 3)
    --timeout SEC        Maximum wait time in seconds (default: 3600)
    --no-auto-promote    Do not auto-promote trained artifact (default: auto-promote enabled)
    --auto-promote       Auto-promote trained artifact after completion (default)
    --help               Show this help message

Examples:
    # Local training (no GPU)
    $0 --api-key mykey123 --api-base http://localhost:8000

    # Remote GPU training
    $0 --api-key mykey123 --api-base https://train-api.aicd.me --gpu-id 0

Environment variables (alternatives to flags):
    API_BASE, API_KEY, JOB_TYPE, ROWS, SEED, LABEL_COLUMN, GPU_ID, AUTO_PROMOTE

EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-base) API_BASE="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --job-type) JOB_TYPE="$2"; shift 2 ;;
        --rows) ROWS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --label-column) LABEL_COLUMN="$2"; shift 2 ;;
        --gpu-id) GPU_ID="$2"; shift 2 ;;
        --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --auto-promote) AUTO_PROMOTE="1"; shift 1 ;;
        --no-auto-promote) AUTO_PROMOTE="0"; shift 1 ;;
        --help) usage ;;
        *) log_error "Unknown option: $1"; usage ;;
    esac
done

COMPLETED_STATUS_RESPONSE=""

# Validate required parameters
if [[ -z "$API_KEY" ]]; then
    log_error "API key is required. Use --api-key or set API_KEY environment variable."
    usage
fi

# =============================================================================
# STEP 1: Check server health
# =============================================================================
log_info "Checking training server health at $API_BASE..."
HEALTH=$(curl -s -m 10 "$API_BASE/health" 2>/dev/null || echo '{"status":"unreachable"}')

if echo "$HEALTH" | grep -q '"status":"healthy"'; then
    log_success "Training server is healthy"
    GPU_AVAILABLE=$(echo "$HEALTH" | grep -o '"gpu_available":[^,}]*' | cut -d':' -f2)
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        log_success "GPU is available on the server"
    else
        log_warn "No GPU detected - training will use CPU (slower)"
    fi
else
    log_error "Training server is not reachable at $API_BASE"
    log_error "Response: $HEALTH"
    exit 1
fi

# =============================================================================
# STEP 2: Generate synthetic training data
# =============================================================================
log_info "Generating synthetic training data ($ROWS rows, seed=$SEED)..."

TMP_CSV="/tmp/dexy_training_${JOB_TYPE}_$(date +%s).csv"

# Use Python to generate the CSV (matches kickoff_training_job.py logic)
python3 << PYEOF
import csv
import random
import sys

random.seed($SEED)

feature_names = [
    "liquidity_usd", "fdv_usd", "holder_count", "top_holder_pct",
    "age_seconds", "volume_24h", "price_change_1h", "mev_sandwich_count",
    "avg_slippage_bps", "volatility",
]

with open("$TMP_CSV", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["$LABEL_COLUMN"] + feature_names)
    
    for _ in range($ROWS):
        liquidity = random.uniform(1_000, 2_000_000)
        fdv = random.uniform(10_000, 50_000_000)
        holders = random.randint(10, 50_000)
        top_pct = random.uniform(1.0, 90.0)
        age = random.randint(60, 7 * 24 * 3600)
        vol24 = random.uniform(0, 5_000_000)
        chg1h = random.uniform(-80.0, 200.0)
        mev = random.randint(0, 50)
        slip = random.uniform(10.0, 2500.0)
        vol = random.uniform(0.01, 2.5)
        
        score = (
            (liquidity / 2_000_000.0) * 0.35
            + (holders / 50_000.0) * 0.20
            + (vol24 / 5_000_000.0) * 0.15
            + (max(chg1h, -80.0) / 200.0) * 0.10
            + (1.0 - min(slip, 2500.0) / 2500.0) * 0.15
            + (1.0 - min(top_pct, 90.0) / 90.0) * 0.05
        )
        score -= min(mev / 50.0, 1.0) * 0.10
        score -= min(vol / 2.5, 1.0) * 0.05
        
        p = max(0.0, min(1.0, 0.15 + score))
        profitable = 1 if random.random() < p else 0
        
        writer.writerow([
            profitable,
            f"{liquidity:.2f}",
            f"{fdv:.2f}",
            holders,
            f"{top_pct:.4f}",
            age,
            f"{vol24:.2f}",
            f"{chg1h:.4f}",
            mev,
            f"{slip:.4f}",
            f"{vol:.6f}",
        ])

print(f"Generated $ROWS rows to $TMP_CSV")
PYEOF

if [[ ! -f "$TMP_CSV" ]]; then
    log_error "Failed to generate training data"
    exit 1
fi
log_success "Generated training data: $TMP_CSV ($(wc -l < "$TMP_CSV") lines)"

# =============================================================================
# STEP 3: Upload dataset
# =============================================================================
log_info "Uploading dataset to training server..."

UPLOAD_RESPONSE=$(curl -s -X POST "$API_BASE/data/upload" \
    -H "Authorization: Bearer $API_KEY" \
    -F "file=@$TMP_CSV" \
    -F "name=gpu-training-${JOB_TYPE}-$(date +%Y%m%d-%H%M%S)" \
    -F "description=Auto-generated synthetic dataset for ${JOB_TYPE} training" \
    -F "label_column=$LABEL_COLUMN")

DATASET_ID=$(echo "$UPLOAD_RESPONSE" | grep -o '"dataset_id":"[^"]*"' | cut -d'"' -f4)

if [[ -z "$DATASET_ID" ]]; then
    log_error "Failed to upload dataset"
    log_error "Response: $UPLOAD_RESPONSE"
    exit 1
fi
log_success "Dataset uploaded: $DATASET_ID"

# =============================================================================
# STEP 4: Launch training job
# =============================================================================
log_info "Launching ${JOB_TYPE} training job..."

TRAIN_PAYLOAD=$(cat << EOF
{
    "job_type": "$JOB_TYPE",
    "dataset_id": "$DATASET_ID",
    "hyperparams": {
        "label_column": "$LABEL_COLUMN",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.01,
        "ensemble_weights": {"linear": 0.4, "lgbm": 0.3, "xgb": 0.3}
    },
    "gpu_id": ${GPU_ID:-null}
}
EOF
)

TRAIN_RESPONSE=$(curl -s -X POST "$API_BASE/jobs/train" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "$TRAIN_PAYLOAD")

JOB_ID=$(echo "$TRAIN_RESPONSE" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)

if [[ -z "$JOB_ID" ]]; then
    log_error "Failed to launch training job"
    log_error "Response: $TRAIN_RESPONSE"
    exit 1
fi
log_success "Training job launched: $JOB_ID"

# =============================================================================
# STEP 5: Poll for completion
# =============================================================================
log_info "Waiting for training to complete (timeout: ${TIMEOUT}s)..."

START_TIME=$(date +%s)
DEADLINE=$((START_TIME + TIMEOUT))

while true; do
    CURRENT_TIME=$(date +%s)
    if [[ $CURRENT_TIME -ge $DEADLINE ]]; then
        log_error "Training timed out after ${TIMEOUT} seconds"
        exit 1
    fi
    
    STATUS_RESPONSE=$(curl -s "$API_BASE/jobs/$JOB_ID/status" \
        -H "Authorization: Bearer $API_KEY")
    
    STATUS=$(echo "$STATUS_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    case $STATUS in
        pending)
            log_info "Job pending... (${ELAPSED}s elapsed)"
            ;;
        running)
            GPU_INFO=$(echo "$STATUS_RESPONSE" | grep -o '"gpu_id":[^,}]*' | cut -d':' -f2)
            log_info "Job running on ${GPU_INFO:-CPU}... (${ELAPSED}s elapsed)"
            ;;
        completed)
            COMPLETED_STATUS_RESPONSE="$STATUS_RESPONSE"
            log_success "Training completed!"
            echo ""
            echo "=== Training Result ==="
            echo "$STATUS_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$STATUS_RESPONSE"
            echo ""
            
            # Extract metrics
            ACCURACY=$(echo "$STATUS_RESPONSE" | grep -o '"accuracy":[^,}]*' | cut -d':' -f2)
            AUC=$(echo "$STATUS_RESPONSE" | grep -o '"auc_roc":[^,}]*' | cut -d':' -f2)
            log_success "Accuracy: ${ACCURACY:-N/A}, AUC-ROC: ${AUC:-N/A}"
            ;;
        failed)
            log_error "Training failed!"
            ERROR_MSG=$(echo "$STATUS_RESPONSE" | grep -o '"error_message":"[^"]*"' | cut -d'"' -f4)
            log_error "Error: $ERROR_MSG"
            echo "$STATUS_RESPONSE" | python3 -m json.tool 2>/dev/null
            exit 1
            ;;
        *)
            log_warn "Unknown status: $STATUS"
            ;;
    esac
    
    if [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]]; then
        break
    fi
    
    sleep "$POLL_INTERVAL"
done

# =============================================================================
# STEP 6: Optional auto-promotion
# =============================================================================
if [[ "$AUTO_PROMOTE" == "1" ]]; then
    ARTIFACT_ID=$(STATUS_JSON="$COMPLETED_STATUS_RESPONSE" python3 - <<'PYEOF'
import json, os
raw = os.environ.get("STATUS_JSON", "")
try:
    data = json.loads(raw)
    print(data.get("artifact_id") or "")
except Exception:
    print("")
PYEOF
)

    if [[ -n "$ARTIFACT_ID" ]]; then
        log_info "Auto-promoting artifact: $ARTIFACT_ID"
        PROMOTE_RESPONSE=$(curl -s -X POST "$API_BASE/models/promote/$ARTIFACT_ID" \
            -H "Authorization: Bearer $API_KEY")
        if echo "$PROMOTE_RESPONSE" | grep -q '"promoted"[[:space:]]*:[[:space:]]*true'; then
            log_success "Artifact promoted successfully"
            echo "$PROMOTE_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$PROMOTE_RESPONSE"
        else
            log_warn "Auto-promotion failed or unsupported"
            echo "$PROMOTE_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$PROMOTE_RESPONSE"
        fi
    else
        log_warn "No artifact_id returned by job status; skipping auto-promotion"
    fi
else
    log_info "Auto-promotion disabled"
fi

# =============================================================================
# STEP 7: Check model manifest
# =============================================================================
log_info "Checking model manifest..."
MANIFEST=$(curl -s "$API_BASE/models/manifest" \
    -H "Authorization: Bearer $API_KEY")

echo ""
echo "=== Available Models ==="
echo "$MANIFEST" | python3 -m json.tool 2>/dev/null || echo "$MANIFEST"

# Cleanup
rm -f "$TMP_CSV"

log_success "GPU training preparation complete!"
echo ""
echo "Next steps:"
if [[ "$AUTO_PROMOTE" == "1" ]]; then
    echo "  1. Configure HotPath: export TRAINING_SERVER_URL=$API_BASE TRAINING_API_KEY=$API_KEY"
    echo "  2. Start HotPath with: cargo run --features remote-training"
else
    echo "  1. Promote the model: curl -X POST $API_BASE/models/promote/<ARTIFACT_ID> -H 'Authorization: Bearer $API_KEY'"
    echo "  2. Configure HotPath: export TRAINING_SERVER_URL=$API_BASE TRAINING_API_KEY=$API_KEY"
    echo "  3. Start HotPath with: cargo run --features remote-training"
fi
