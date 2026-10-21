"""dashboard/runs.py - Cached readers for run artifacts and precomputed dashboard data.

Lane B (Saanvi) owns this file.
Phase P4-B.

Every reader returns None when a file is missing, so tabs can show a friendly message.
Nothing here trains, clusters or recomputes metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.data import results_dir


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float):
    return json.loads(Path(path).read_text(encoding="utf-8")) if mtime else None


@st.cache_data(show_spinner=False)
def _read_npy(path: str, mtime: float):
    return np.load(path) if mtime else None


def load_json(path: Path):
    return _read_json(str(path), _mtime(path))


def load_meta(run_id: str):
    return load_json(results_dir() / "runs" / run_id / "meta.json")


def load_npy(run_id: str, name: str):
    path = results_dir() / "runs" / run_id / f"{name}.npy"
    return _read_npy(str(path), _mtime(path))


def dashboard_json(name: str):
    return load_json(results_dir() / "dashboard" / name)


def conf_hist(run_id: str):
    return dashboard_json(f"conf_hist/{run_id}.json")


def true_labels(dataset: str):
    path = results_dir() / "dashboard" / "labels" / f"{dataset}.npy"
    return _read_npy(str(path), _mtime(path))


MISSING_DASHBOARD_DATA = ("not computed yet - run `python scripts/make_dashboard_data.py` "
                          "to precompute this view")


# ---------------------------------------------------------------------------
# Shared tab helpers
# ---------------------------------------------------------------------------

def pick(df: pd.DataFrame, key: str, cols=("dataset", "split", "seed")) -> dict:
    """One selectbox per column, side by side, each limited by the choices to its left."""
    chosen, sub = {}, df
    for col, slot in zip(cols, st.columns(len(cols))):
        options = sorted(sub[col].dropna().unique().tolist())
        if not options:
            return {}
        default = options.index("none") if col == "calib_method" and "none" in options else 0
        chosen[col] = slot.selectbox(col.replace("_", " ").capitalize(), options,
                                     index=default, key=f"{key}_{col}")
        sub = sub[sub[col] == chosen[col]]
    return chosen


def rows_for(df: pd.DataFrame, **match) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    for k, v in match.items():
        m &= df[k] == v
    return df[m]


def show(fig, caption: str, rows: pd.DataFrame | None = None) -> None:
    """Render a matplotlib figure with its caption and an "Underlying rows" expander."""
    st.pyplot(fig, clear_figure=True, use_container_width=False)
    plt.close(fig)
    st.caption(caption)
    underlying(rows)


def underlying(rows: pd.DataFrame | None) -> None:
    if rows is not None:
        with st.expander(f"Underlying rows ({len(rows)})"):
            st.dataframe(rows, hide_index=True, use_container_width=True)
