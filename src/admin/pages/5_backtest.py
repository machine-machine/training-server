"""Backtest — run backtests against historical data using trained models."""

import streamlit as st

st.header("Backtest")

st.info("Run backtests using trained models against historical trade data.")

# Configuration
col1, col2 = st.columns(2)
with col1:
    model_version = st.text_input("Model Artifact ID")
    initial_capital = st.number_input("Initial Capital (SOL)", value=10.0, min_value=0.1)
with col2:
    date_start = st.date_input("Start Date")
    date_end = st.date_input("End Date")

if st.button("Run Backtest"):
    if model_version:
        st.warning("Backtest execution requires wiring to the backtest engine. Configure in Settings.")
    else:
        st.error("Select a model version first.")
