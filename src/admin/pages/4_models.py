"""Models — registry, version comparison, promotion."""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")
API_KEY = os.environ.get("API_KEY", "")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

st.header("Model Registry")


@st.cache_data(ttl=30)
def fetch_manifest():
    try:
        resp = httpx.get(f"{API_URL}/models/manifest", headers=HEADERS, timeout=10)
        return resp.json().get("models", [])
    except Exception:
        return []


models = fetch_manifest()

if models:
    for model in models:
        with st.expander(f"{model['model_type']} v{model['version']}"):
            st.write(f"**SHA256:** `{model['sha256']}`")
            st.write(f"**Created:** {model['created_at']}")
            if model.get("metrics"):
                st.json(model["metrics"])
else:
    st.info("No promoted models yet. Train a model and promote it.")

# Promote section
st.subheader("Promote Model")
artifact_id = st.text_input("Artifact ID to promote")
if st.button("Promote") and artifact_id:
    try:
        resp = httpx.post(f"{API_URL}/models/promote/{artifact_id}", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            st.success(f"Promoted: {resp.json()}")
            st.cache_data.clear()
        else:
            st.error(f"Failed: {resp.text}")
    except Exception as e:
        st.error(f"Error: {e}")
