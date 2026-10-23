"""dashboard/tab_clusterwise.py - Tab 5: Cluster-wise calibration (the key demo tab).

Lane B (Saanvi) owns this file.
Phase P6-B.

Reads per-run meta.json keys written by run_one: per_cluster (per-cluster ECE), cluster_profile,
per_cluster_calib (a_hat / a_tilde / T per cluster) and proxy_quality ({table, r}).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.runs import default_index, load_meta, pick, rows_for, show, underlying
from dashboard.tab_calibration import METHOD_ORDER, _is_oracle

PROXY_METHODS = {"lf_cts": "LF-CTS (a_hat)", "lf_cts_plus": "LF-CTS+ (a_tilde)"}


def _metas(runs: pd.DataFrame) -> dict:
    """calib_method -> meta.json for one (dataset, split, seed)."""
    out = {}
    for _, r in runs.iterrows():
        m = load_meta(r["run_id"])
        if m is not None:
            out[r["calib_method"]] = m
    return out


def _labels(meta: dict) -> dict:
    prof = pd.DataFrame(meta.get("cluster_profile", []))
    if prof.empty:
        return {}
    return {int(r["cluster"]): f"c{int(r['cluster'])}  p={r['purity']:.2f}  d={r['avg_degree']:.1f}"
            for _, r in prof.iterrows()}


def heatmap(metas: dict, runs: pd.DataFrame) -> None:
    st.subheader("Per-cluster ECE by method")
    methods = [m for m in METHOD_ORDER if m in metas and metas[m].get("per_cluster")]
    if not methods:
        st.info("No per-cluster metrics saved for this selection")
        return
    table = pd.DataFrame({m: pd.DataFrame(metas[m]["per_cluster"]).set_index("cluster_id")["ece"]
                          for m in methods})
    labels = _labels(metas[methods[0]])
    fig, ax = plt.subplots(figsize=(1.1 * len(methods) + 3, 0.5 * len(table) + 1.8))
    im = ax.imshow(table.to_numpy(dtype=float), cmap="Reds", aspect="auto", vmin=0)
    for (i, j), v in np.ndenumerate(table.to_numpy(dtype=float)):
        ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=10,
                color="white" if not np.isnan(v) and v > np.nanmax(table.to_numpy()) * 0.6 else "black")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([f"{m}\n(uses target labels)" if _is_oracle(m) else m for m in methods],
                       rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([labels.get(int(c), f"c{c}") for c in table.index], fontsize=10)
    fig.colorbar(im, ax=ax, label="ECE")
    show(fig, "Each row is one cluster (p = purity, d = average degree, true labels: evaluation "
              "only); each column a calibration method. Darker = worse calibrated. n/a means "
              "fewer than 20 target-eval nodes in that cluster.",
         table.round(4).reset_index())


def worst_cluster(metas: dict, runs: pd.DataFrame) -> None:
    st.subheader("Worst cluster")
    methods = [m for m in METHOD_ORDER if m in metas]
    m = st.selectbox("Method", methods, index=default_index(methods, "calib_method"),
                     key="cw_method")
    row = runs[runs["calib_method"] == m].iloc[0]
    w = int(row["worst_cluster_id"]) if pd.notna(row["worst_cluster_id"]) else -1
    prof = pd.DataFrame(metas[m].get("cluster_profile", [])).set_index("cluster")
    if w < 0 or w not in prof.index:
        st.info("No cluster has enough target-eval nodes to score")
        return
    p = prof.loc[w]
    ratio = row["cluster_ece_worst"] / row["ece"] if row["ece"] > 0 else float("nan")
    c = st.columns(4)
    c[0].metric("Worst-cluster ECE", f"{row['cluster_ece_worst']:.3f}", f"{ratio:.1f}x overall",
                delta_color="off")
    c[1].metric("Overall ECE", f"{row['ece']:.3f}")
    c[2].metric("Cluster size", int(p["size"]))
    c[3].metric("Purity", f"{p['purity']:.2f}")
    c = st.columns(4)
    c[0].metric("Conductance", f"{p['conductance']:.2f}")
    c[1].metric("Avg degree", f"{p['avg_degree']:.2f}")
    c[2].metric("Inside homophily", f"{p['inside_homophily']:.2f}")
    c[3].metric("Cluster id", w)
    st.caption(f"The average ECE hides this: cluster {w} is calibrated {ratio:.1f}x worse than the "
               f"graph as a whole under {m}. Purity, conductance and homophily use true labels "
               "(evaluation only).")
    underlying(runs[runs["calib_method"] == m])


def proxy_scatter(metas: dict) -> None:
    st.subheader("How good is the label-free accuracy estimate?")
    have = {m: metas[m]["proxy_quality"] for m in PROXY_METHODS if metas.get(m, {}).get("proxy_quality")}
    if not have:
        st.info("No proxy-quality table saved (needs lf_cts / lf_cts_plus rows from phase P4-A)")
        return
    cols = st.columns(len(have))
    for slot, (m, pq) in zip(cols, have.items()):
        slot.metric(f"Pearson r - {PROXY_METHODS[m]}", f"{pq['r']:+.2f}")
    fig, ax = plt.subplots(figsize=(6, 5))
    rows = []
    for (m, pq), marker in zip(have.items(), ("o", "s")):
        t = pd.DataFrame(pq["table"]).dropna(subset=["true_acc"])
        t = t[t["n_eval"] > 0]
        ax.scatter(t["a_hat"], t["true_acc"], s=20 + 2 * np.sqrt(t["n_eval"]) * 4, marker=marker,
                   alpha=0.75, edgecolor="black", label=f"{PROXY_METHODS[m]}  r={pq['r']:+.2f}")
        rows += [dict(method=m, **r) for r in t.to_dict("records")]
    ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("estimated accuracy (no target labels)")
    ax.set_ylabel("true accuracy (evaluation only)")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    show(fig, "One dot per cluster, sized by its number of target-eval nodes. Dots on the dashed "
              "line mean the estimate is right; above it the estimate is too low (the method "
              "will make that cluster under-confident), below it too high.", pd.DataFrame(rows))


def temperature_table(metas: dict) -> None:
    st.subheader("Per-cluster temperatures")
    frames = []
    for m, cols in (("lf_cts", {"a_hat": "a_hat", "T": "T_lf_cts", "clamped": "clamped_lf_cts"}),
                    ("lf_cts_plus", {"a_tilde": "a_tilde", "T": "T_lf_cts_plus"}),
                    ("cluster_ts_target_oracle", {"T": "T_cluster_oracle",
                                                  "used_global_T": "oracle_used_global_T"})):
        pc = metas.get(m, {}).get("per_cluster_calib")
        if pc:
            d = pd.DataFrame(pc).set_index("cluster_id")
            frames.append(d[[c for c in cols if c in d]].rename(columns=cols))
    pq = metas.get("lf_cts", {}).get("proxy_quality")
    if pq:
        frames.append(pd.DataFrame(pq["table"]).set_index("cluster_id")[["true_acc", "n_eval"]])
    if not frames:
        st.info("No per-cluster calibration saved (needs phase P4-A rows)")
        return
    table = pd.concat(frames, axis=1)
    order = [c for c in ["a_hat", "a_tilde", "true_acc", "n_eval", "T_lf_cts", "T_lf_cts_plus",
                         "T_cluster_oracle", "clamped_lf_cts", "oracle_used_global_T"] if c in table]
    st.dataframe(table[order].reset_index(), hide_index=True, use_container_width=True)
    st.caption("a_hat = label-free accuracy estimate, a_tilde = the same after the LF-CTS+ "
               "correction, true_acc = actual accuracy (evaluation only). Each method picks a "
               "temperature T per cluster: T < 1 sharpens, T > 1 softens. The cluster oracle "
               "uses target labels and is a ceiling, not a competitor.")


def render(df: pd.DataFrame) -> None:
    base = df[(df["calib_method"] == "none") & (df["ablation"].fillna("main") == "main")]
    sel = pick(base, "cw")
    if not sel:
        st.info("No uncalibrated rows to anchor this view")
        return
    runs = rows_for(df[df["ablation"].fillna("main") == "main"], **sel)
    metas = _metas(runs)
    if len(metas) <= 1:
        st.info("Only the uncalibrated model is available for this selection - calibrated rows "
                "come from phase P4-A.")
    heatmap(metas, runs)
    worst_cluster(metas, runs)
    proxy_scatter(metas)
    temperature_table(metas)
