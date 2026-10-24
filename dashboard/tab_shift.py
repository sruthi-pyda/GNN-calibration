"""dashboard/tab_shift.py - Tab 3: Shift split viewer.

Lane B (Saanvi) owns this file.
Phase P4-B.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.runs import (
    MISSING_DASHBOARD_DATA, dashboard_json, default_index, pick, show, underlying,
)

DIAG_KEYS = ["class_tv_distance", "class_kl", "cross_region_edge_frac",
             "feature_centroid_cosine", "train_homophily", "test_homophily",
             "source_region_frac", "n_train", "n_val", "n_target", "n_train_communities"]


def _frame(splits: dict) -> pd.DataFrame:
    rows = []
    for key, v in splits.items():
        if key.startswith("_"):
            continue
        ds, kind, seed = key.split("__")
        rows.append(dict(dataset=ds, split=kind, seed=int(seed[1:]),
                         **{k: v["diagnostics"].get(k) for k in DIAG_KEYS}))
    return pd.DataFrame(rows)


def diagnostics_table(frame: pd.DataFrame, ds: str) -> None:
    sub = frame[frame["dataset"] == ds]
    table = pd.DataFrame({
        kind: [f"{sub[sub['split'] == kind][k].mean():.3f} ± {sub[sub['split'] == kind][k].std(ddof=0):.3f}"
               for k in DIAG_KEYS]
        for kind in ("random", "structural")}, index=DIAG_KEYS)
    st.dataframe(table, use_container_width=True)
    st.caption("Random vs structural split, mean ± std over seeds 0-4. A real shift shows up as "
               "a larger class TV distance and a smaller cross-region edge fraction under the "
               "structural split. Homophily and class statistics use true labels "
               "(evaluation only).")
    underlying(sub)


def class_bars(splits: dict, ds: str, kind: str, seed: int) -> None:
    v = splits.get(f"{ds}__{kind}__s{seed}")
    if v is None:
        st.info("No split data for this selection")
        return
    src, tgt = np.array(v["source_class_dist"]), np.array(v["target_class_dist"])
    x = np.arange(len(src))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(x - 0.2, src, 0.4, label="source (labelled side, excl. train sample)")
    ax.bar(x + 0.2, tgt, 0.4, label="target")
    ax.set_xticks(x)
    ax.set_xlabel("class")
    ax.set_ylabel("share of nodes")
    ax.legend(frameon=False)
    show(fig, f"Class mix on each side of the {kind} split ({ds}, seed {seed}). Different bar "
              f"heights mean the target sees a different class mix than the source - class TV "
              f"distance {v['diagnostics']['class_tv_distance']:.3f}. True labels, "
              "evaluation only.",
         pd.DataFrame(dict(cls=x, source=src.round(3), target=tgt.round(3))))


def cross_region(frame: pd.DataFrame) -> None:
    m = frame.groupby(["dataset", "split"])["cross_region_edge_frac"].agg(["mean", "std"])
    m = m.reset_index()
    datasets = sorted(m["dataset"].unique())
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, kind in enumerate(("random", "structural")):
        d = m[m["split"] == kind].set_index("dataset").reindex(datasets)
        ax.bar(x + (i - 0.5) * 0.4, d["mean"], 0.4, yerr=d["std"], capsize=3, label=kind)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20)
    ax.set_ylabel("cross-region edge fraction")
    ax.legend(frameon=False)
    show(fig, "Share of edges that connect the source side to the target side. Structural "
              "splits cut far fewer edges, so the target is a structurally separate part of "
              "the graph.", m.round(4))


def render(df: pd.DataFrame) -> None:
    splits = dashboard_json("splits.json")
    if splits is None:
        st.info(MISSING_DASHBOARD_DATA)
        return
    frame = _frame(splits)
    st.subheader("Shift diagnostics")
    names = sorted(frame["dataset"].unique())
    ds = st.selectbox("Dataset", names, index=default_index(names, "dataset"), key="shift_ds")
    diagnostics_table(frame, ds)
    st.subheader("Class distribution")
    sel = pick(frame[frame["dataset"] == ds], "shift", cols=("split", "seed"))
    if sel:
        class_bars(splits, ds, sel["split"], sel["seed"])
    st.subheader("Cross-region edges")
    cross_region(frame)
