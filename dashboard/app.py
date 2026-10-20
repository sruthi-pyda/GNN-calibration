"""dashboard/app.py – Streamlit dashboard for GNN calibration results.

Lane B (Saanvi) owns this file.
Phase P1-C (shell on fixture) -> P4-B (tabs 2-4) -> P6-B (tabs 5-6, demo mode).

Run: streamlit run dashboard/app.py
"""
# P1-C stub – shell implemented in Phase P1-C

import streamlit as st

st.set_page_config(page_title="GNN Calibration Dashboard", layout="wide")
st.title("GNN Calibration Dashboard")
st.info("Dashboard shell – implemented in Phase P1-C.")
st.markdown("""
**Six tabs will be added in phases P1-C through P6-B:**
1. Overview
2. Clustering explorer
3. Shift split viewer
4. Calibration
5. Cluster-wise calibration
6. Ablations
""")
