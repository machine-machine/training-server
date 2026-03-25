"""Datasets — upload and manage training data."""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")
API_KEY = os.environ.get("API_KEY", "")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

st.header("Datasets")

# Upload section
st.subheader("Upload Dataset")
with st.form("upload_form"):
    file = st.file_uploader("Choose file", type=["csv", "parquet", "json"])
    name = st.text_input("Dataset name")
    description = st.text_area("Description (optional)")
    label_column = st.text_input("Label column name", value="profitable")
    submitted = st.form_submit_button("Upload")

    if submitted and file and name:
        try:
            resp = httpx.post(
                f"{API_URL}/data/upload",
                headers=HEADERS,
                files={"file": (file.name, file.getvalue(), file.type or "application/octet-stream")},
                data={"name": name, "description": description, "label_column": label_column},
                timeout=120,
            )
            if resp.status_code == 201:
                result = resp.json()
                st.success(f"Uploaded: {result['row_count']} rows, {result['feature_count']} features")
            else:
                st.error(f"Upload failed: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")
