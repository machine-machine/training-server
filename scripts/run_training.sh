#!/usr/bin/env bash
# Training automation script: upload data → submit job → monitor
# Usage: ./scripts/run_training.sh <API_URL> <API_KEY> [CSV_PATH]
set -euo pipefail

API_URL="${1:?Usage: $0 <API_URL> <API_KEY> [CSV_PATH]}"
API_KEY="${2:?Usage: $0 <API_URL> <API_KEY> [CSV_PATH]}"
CSV_PATH="${3:-/tmp/dexy_training_data.csv}"

AUTH="Authorization: Bearer ${API_KEY}"

echo "=== 1. Health Check ==="
HEALTH=$(curl -sf -H "$AUTH" "${API_URL}/health" 2>&1) || {
    echo "FAIL: API unreachable at ${API_URL}/health"
    echo "Response: $HEALTH"
    exit 1
}
echo "$HEALTH" | python3 -m json.tool
WORKERS=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get('workers_online',0))")
GPU_AVAILABLE=$(echo "$HEALTH" | python3 -c "import json,sys; print(str(json.load(sys.stdin).get('gpu_available',False)).lower())")
GPU_COUNT=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get('gpu_count',0))")
GPU_PROBE_ERROR=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get('gpu_probe_error') or '')")
echo "Workers online: $WORKERS"
echo "GPU available: $GPU_AVAILABLE (count=$GPU_COUNT)"
if [ "$GPU_AVAILABLE" != "true" ] && [ "${ALLOW_CPU_FALLBACK:-0}" != "1" ]; then
    echo "FAIL: training server reports no GPU availability."
    [ -n "$GPU_PROBE_ERROR" ] && echo "gpu_probe_error: $GPU_PROBE_ERROR"
    echo "Set ALLOW_CPU_FALLBACK=1 to intentionally run on CPU."
    exit 1
fi

echo ""
echo "=== 2. Upload Dataset ==="
UPLOAD=$(curl -sf -X POST "${API_URL}/data/upload" \
    -H "$AUTH" \
    -F "file=@${CSV_PATH}" \
    -F "name=dexy_profitability_v1" \
    -F "description=Synthetic profitability training data (5k samples, 20 features)" \
    -F "label_column=profitable")
echo "$UPLOAD" | python3 -m json.tool
DATASET_ID=$(echo "$UPLOAD" | python3 -c "import json,sys; print(json.load(sys.stdin)['dataset_id'])")
echo "Dataset ID: $DATASET_ID"

echo ""
echo "=== 3. Submit Training Job ==="
JOB=$(curl -sf -X POST "${API_URL}/jobs/train" \
    -H "$AUTH" \
    -H "Content-Type: application/json" \
    -d "{
        \"job_type\": \"ensemble\",
        \"dataset_id\": \"${DATASET_ID}\",
        \"hyperparams\": {
            \"label_column\": \"profitable\",
            \"n_estimators\": 500,
            \"max_depth\": 6,
            \"learning_rate\": 0.01,
            \"C\": 1.0,
            \"ensemble_weights\": {\"linear\": 0.4, \"lgbm\": 0.3, \"xgb\": 0.3}
        }
    }")
echo "$JOB" | python3 -m json.tool
JOB_ID=$(echo "$JOB" | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")
echo "Job ID: $JOB_ID"

echo ""
echo "=== 4. Monitor Training ==="
for i in $(seq 1 60); do
    sleep 10
    STATUS=$(curl -sf -H "$AUTH" "${API_URL}/jobs/${JOB_ID}/status" 2>&1) || {
        echo "[${i}] Failed to fetch status"
        continue
    }
    JOB_STATUS=$(echo "$STATUS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status','unknown'))")
    echo "[$(date +%H:%M:%S)] Status: $JOB_STATUS"

    if [ "$JOB_STATUS" = "completed" ]; then
        echo ""
        echo "=== TRAINING COMPLETED ==="
        echo "$STATUS" | python3 -m json.tool
        break
    elif [ "$JOB_STATUS" = "failed" ]; then
        echo ""
        echo "=== TRAINING FAILED ==="
        echo "$STATUS" | python3 -m json.tool
        exit 1
    fi
done

echo ""
echo "=== 5. Check Model Artifacts ==="
curl -sf -H "$AUTH" "${API_URL}/models/latest/profitability" 2>/dev/null | python3 -m json.tool || echo "(no artifacts yet)"

echo ""
echo "Done."
