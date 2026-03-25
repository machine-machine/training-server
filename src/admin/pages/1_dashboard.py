"""Dashboard — system status, GPU utilization, recent activity."""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")
API_KEY = os.environ.get("API_KEY", "")

st.header("Dashboard")


@st.cache_data(ttl=10)
def fetch_health():
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


health = fetch_health()

# System status
col1, col2, col3, col4 = st.columns(4)
col1.metric("Status", health.get("status", "unknown"))
col2.metric("GPUs", health.get("gpu_count", 0))
col3.metric("Active Jobs", health.get("active_jobs", 0))
col4.metric("CPU", f"{health.get('cpu_percent', 0)}%")

# GPU details
gpus = health.get("gpus", [])
if gpus:
    st.subheader("GPU Utilization")
    for gpu in gpus:
        col_a, col_b = st.columns(2)
        col_a.write(f"**GPU {gpu['id']}**: {gpu['name']}")
        vram_pct = (gpu["memory_used_mb"] / gpu["memory_total_mb"] * 100) if gpu["memory_total_mb"] else 0
        col_b.progress(int(vram_pct), text=f"VRAM: {gpu['memory_used_mb']}MB / {gpu['memory_total_mb']}MB ({vram_pct:.0f}%)")
else:
    st.warning("No GPU information available")
