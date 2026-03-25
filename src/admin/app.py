"""2DEXY Training Server — Streamlit Admin Dashboard."""

import streamlit as st

st.set_page_config(
    page_title="2DEXY ML Training",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("2DEXY ML Training Server")
st.markdown("GPU-powered model training and management for the HotPath trading engine.")
st.markdown("Use the sidebar to navigate between pages.")
