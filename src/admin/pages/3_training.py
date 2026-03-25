"""Training — launch and monitor training jobs."""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")
API_KEY = os.environ.get("API_KEY", "")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

st.header("Training Jobs")

# Launch new job
st.subheader("Launch Training")
with st.form("train_form"):
    job_type = st.selectbox("Model type", ["ensemble", "fraud", "slippage", "regime", "bandit", "rl"])
    dataset_id = st.text_input("Dataset ID (UUID)")
    gpu_id = st.selectbox("GPU", [None, 0, 1], format_func=lambda x: "Auto" if x is None else f"GPU {x}")

    st.markdown("**Hyperparameters**")
    learning_rate = st.number_input("Learning rate", value=0.01, format="%.4f")
    n_estimators = st.number_input("Estimators", value=500, min_value=10, max_value=5000)
    max_depth = st.number_input("Max depth", value=6, min_value=1, max_value=20)
    label_column = st.text_input("Label column", value="profitable")

    submitted = st.form_submit_button("Launch Training")

    if submitted and dataset_id:
        try:
            resp = httpx.post(
                f"{API_URL}/jobs/train",
                headers=HEADERS,
                json={
                    "job_type": job_type,
                    "dataset_id": dataset_id,
                    "hyperparams": {
                        "learning_rate": learning_rate,
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "label_column": label_column,
                    },
                    "gpu_id": gpu_id,
                },
                timeout=30,
            )
            if resp.status_code == 202:
                result = resp.json()
                st.success(f"Job queued: {result['job_id']}")
            else:
                st.error(f"Failed: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# Job status checker
st.subheader("Check Job Status")
check_job_id = st.text_input("Job ID")
if st.button("Check Status") and check_job_id:
    try:
        resp = httpx.get(f"{API_URL}/jobs/{check_job_id}/status", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            status = resp.json()
            st.json(status)
        else:
            st.error(f"Not found: {resp.text}")
    except Exception as e:
        st.error(f"Error: {e}")
