"""Settings — API keys, notifications, storage management."""

import os

import streamlit as st

st.header("Settings")

# API key display
st.subheader("API Configuration")
api_key = os.environ.get("API_KEY", "not configured")
st.text_input("API Key", value=api_key[:8] + "..." if len(api_key) > 8 else api_key, disabled=True)
st.caption("API key is configured via environment variables. Update in Coolify Secrets.")

# Model polling config
st.subheader("Model Delivery")
poll_interval = st.number_input(
    "HotPath poll interval (seconds)",
    value=300,
    min_value=30,
    max_value=3600,
    help="How often the Rust HotPath checks for new promoted models.",
)

# Auto-promote
st.subheader("Auto-Promotion")
auto_promote = st.toggle("Auto-promote models that beat current metrics", value=False)
if auto_promote:
    metric_threshold = st.selectbox("Primary metric", ["auc_roc", "f1_score", "accuracy"])
    min_improvement = st.slider("Minimum improvement (%)", 0, 20, 5)

# Storage
st.subheader("Storage")
st.info("Storage usage metrics will be available when connected to the file system.")
