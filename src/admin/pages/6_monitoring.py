"""Monitoring — training curves, GPU metrics, convergence analysis."""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")
API_KEY = os.environ.get("API_KEY", "")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

st.header("Monitoring")

# GPU real-time metrics
st.subheader("GPU Metrics")

try:
    resp = httpx.get(f"{API_URL}/health", timeout=5)
    health = resp.json()
    gpus = health.get("gpus", [])
    if gpus:
        cols = st.columns(len(gpus))
        for i, gpu in enumerate(gpus):
            with cols[i]:
                st.metric(f"GPU {gpu['id']}", f"{gpu['utilization_pct']}%", label_visibility="visible")
                vram_pct = gpu["memory_used_mb"] / gpu["memory_total_mb"] * 100 if gpu["memory_total_mb"] else 0
                st.progress(int(vram_pct), text=f"VRAM {vram_pct:.0f}%")
except Exception:
    st.warning("Cannot reach API for GPU metrics")

# Training logs viewer
st.subheader("Training Logs")
job_id_input = st.text_input("Job ID for logs")
if job_id_input:
    st.info("Log streaming will be available when connected to the training database.")
