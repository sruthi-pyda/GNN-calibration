"""dashboard/tab_clustering.py - Tab 2: Clustering explorer.

Lane B (Saanvi) owns this file.
Phase P4-B.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.runs import (
    MISSING_DASHBOARD_DATA, dashboard_json, default_index, load_meta, load_npy, pick, rows_for,
    show,
    true_labels, underlying,
)

METHOD_ORDER = ["sgc_kmeans", "feature_kmeans", "louvain", "spectral"]


def method_table(clus: dict) -> None:
    st.subheader("Clustering methods")
    names = [k for k in clus if not k.startswith("_")]
    ds = st.selectbox("Dataset", names, index=default_index(names, "dataset"),
                      key="clu_methods_ds")
    rows = []
    for m in METHOD_ORDER:
        r = clus[ds]["methods"].get(m, {})
        if "skipped" in r:
            rows.append(dict(method=m, note=f"skipped ({r['skipped']})"))
        else:
            rows.append(dict(method=m, **{k: round(v, 3) if isinstance(v, float) else v
                                          for k, v in r.items()}))
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption(f"How well each clustering method recovers the classes on {ds} (seed 42). NMI/ARI "
               "compare clusters with the true classes; modularity and silhouette need no "
               "labels. " + clus["_note"])


def gate_gauge(df: pd.DataFrame) -> None:
    st.subheader("Homophily gate")
    sel = pick(df, "gate", cols=("dataset", "split"))
    if not sel:
        return
    rows = rows_for(df, **sel)
    meta = load_meta(rows["run_id"].iloc[0])
    h_true = meta["h_true_eval_only"] if meta else None
    fig, ax = plt.subplots(figsize=(8, 1.9))
    ax.axvspan(0, 0.5, color="#D55E00", alpha=0.08)
    ax.axvspan(0.5, 1, color="#0072B2", alpha=0.08)
    ax.axvline(0.5, color="black", lw=1.5)
    ax.text(0.25, 0.85, "raw features", ha="center", transform=ax.transAxes, fontsize=11)
    ax.text(0.75, 0.85, "SGC smoothing", ha="center", transform=ax.transAxes, fontsize=11)
    marks = [("MLP estimate (used)", rows["h_hat_mlp"].mean(), "o", "#009E73"),
             ("GCN estimate", rows["h_hat_gnn"].mean(), "s", "#E69F00")]
    if h_true is not None:
        marks.append(("true (evaluation only)", h_true, "D", "black"))
    for i, (label, h, marker, colour) in enumerate(marks):
        ax.plot(h, 0.3 + 0.12 * i, marker=marker, ms=12, color=colour, label=f"{label}: {h:.2f}",
                ls="none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("edge homophily")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    decisions = ", ".join(f"{m} x{n}" for m, n in rows["homophily_gate"].value_counts().items())
    show(fig, f"The gate clusters with SGC smoothing when the estimated homophily is above 0.5. "
              f"It uses the MLP's predicted labels (mean over {len(rows)} seeds); the GCN "
              "estimate and the true value are shown for comparison only. "
              f"Gate decisions here: {decisions}.",
         rows[["run_id", "h_hat_mlp", "h_hat_gnn", "homophily_gate"]])


def hop_sweep(clus: dict) -> None:
    st.subheader("Smoothing depth")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    rows = []
    for ds in (k for k in clus if not k.startswith("_")):
        sw = pd.DataFrame(clus[ds]["hop_sweep"])
        ax.errorbar(sw["hops"], sw["nmi_mean"], yerr=sw["nmi_std"], marker="o", capsize=3,
                    label=f"{ds} (h={clus[ds]['h_true']:.2f})")
        rows += [dict(dataset=ds, **r) for r in clus[ds]["hop_sweep"]]
    ax.set_xlabel("SGC hops")
    ax.set_ylabel("NMI (mean over seeds 0-4)")
    ax.set_xticks(range(5))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    show(fig, "More smoothing hops help on homophilic graphs (lines rise) and hurt on "
              "heterophilic ones (lines fall) - the reason for the gate. h is the true edge "
              "homophily (evaluation only).", pd.DataFrame(rows).round(3))


def run_views(df: pd.DataFrame) -> None:
    st.subheader("Clusters of one run")
    sel = pick(df[df["calib_method"] == "none"], "clu_run")
    if not sel:
        return
    run = rows_for(df, calib_method="none", **sel).iloc[0]
    meta = load_meta(run["run_id"])
    if meta is None:
        st.info(f"No artifacts for {run['run_id']}")
        return
    prof = pd.DataFrame(meta["cluster_profile"]).round(3)
    low_p, low_d = prof["purity"].idxmin(), prof["avg_degree"].idxmin()

    def colour(row):
        c = "#F8D7DA" if row.name == low_p else ("#FFF3CD" if row.name == low_d else "")
        return [f"background-color: {c}; color: black" if c else ""] * len(row)

    st.dataframe(prof.style.apply(colour, axis=1).format(precision=3), hide_index=True,
                 use_container_width=True)
    st.caption(f"Clusters from the gated pipeline ({meta['cluster_method']}, gate on MLP "
               "predictions). Red: lowest-purity cluster; yellow: lowest-degree cluster. Purity "
               "and inside homophily use true labels (evaluation only).")
    underlying(rows_for(df, **sel))

    emb, clusters = load_npy(run["run_id"], "embedding2d"), load_npy(run["run_id"], "clusters")
    if emb is None or clusters is None:
        st.info("No embedding saved for this run")
        return
    by = st.radio("Colour by", ["cluster", "true class (evaluation only)"], horizontal=True,
                  key="clu_colour")
    colours = clusters
    if by != "cluster":
        colours = true_labels(sel["dataset"])
        if colours is None:
            st.info(MISSING_DASHBOARD_DATA)
            return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(emb[:, 0], emb[:, 1], c=colours, s=4, cmap="tab20", linewidths=0)
    lo, hi = np.percentile(emb, [1, 99], axis=0)
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xticks([])
    ax.set_yticks([])
    show(fig, "2-D projection of the clustering embedding, one dot per node (view limited to "
              "the central 98% so a few outliers don't squash the rest). Well-separated colour "
              "blobs mean the clusters follow the embedding's structure.")


def render(df: pd.DataFrame) -> None:
    clus = dashboard_json("clustering.json")
    if clus is None:
        st.info(MISSING_DASHBOARD_DATA)
    else:
        method_table(clus)
    gate_gauge(df)
    if clus is not None:
        hop_sweep(clus)
    run_views(df)
