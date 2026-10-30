"""demo/sections.py - The four demo sections (Lane B).

Each render_* draws one section from the pipeline state S up to the stage reached.
live=False is used while RUN DEMO is stepping through the stages (no widgets, so a
section can be redrawn several times in one script run); live=True after the run
adds the interactive controls. Chart keys carry the stage so redraws never clash.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from demo import charts
from demo.pipeline import METHODS, results_dir
from src.clustering import _nx_graph

NAMES = {"cora": "Cora", "citeseer": "CiteSeer", "chameleon": "Chameleon"}
TOP_COMMUNITIES = 7


@st.cache_data(show_spinner="Computing the graph layout (first time only, ~30 s)...")
def layout(ds: str) -> np.ndarray:
    """Spring layout of the whole graph, cached on disk under results/dashboard/layouts/."""
    path = results_dir() / "dashboard" / "layouts" / f"{ds}.npy"
    if path.exists():
        return np.load(path)
    from src.data import load_graph
    pos = nx.spring_layout(_nx_graph(load_graph(ds)), seed=0, iterations=50)
    arr = np.array([pos[i] for i in range(len(pos))])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def _plot(fig, key: str):
    st.plotly_chart(fig, use_container_width=True, key=key,
                    config={"displaylogo": False, "toImageButtonOptions": {"scale": 2}})


def _cluster_groups(clusters: np.ndarray):
    names = np.array([f"Cluster {c}" for c in clusters], dtype=object)
    colours = {f"Cluster {c}": charts.cluster_colour(c) for c in range(int(clusters.max()) + 1)}
    return names, colours


# ---------------------------------------------------------------------------
# 1. Graph
# ---------------------------------------------------------------------------

def render_graph(S: dict, stage: int, live: bool):
    g, sp = S["g"], S["split"]
    cols = st.columns(6)
    for col, (label, value) in zip(cols, [
            ("Nodes", len(g.y)), ("Edges", g.edge_index.shape[1] // 2), ("Classes", g.num_classes),
            ("Features", g.X.shape[1]), ("Source labels", int((sp["train"] | sp["val"]).sum())),
            ("Target nodes", int(S["target"].sum()))]):
        col.metric(label, f"{value:,}")
    mode = st.radio("Colour nodes by", ["Structural split", "Louvain communities"], horizontal=True,
                    key="graph_mode") if live else "Structural split"
    if mode == "Structural split":
        groups, colours = S["region"], charts.REGION_COLOURS
    else:
        comm = S["communities"]
        top = np.argsort(-np.bincount(comm))[:TOP_COMMUNITIES]
        groups = np.array(["Other communities"] * len(comm), dtype=object)
        for rank, c in enumerate(top):
            groups[comm == c] = f"Community {rank + 1}"
        colours = {f"Community {r + 1}": charts.SERIES[r] for r in range(len(top))}
        colours["Other communities"] = charts.LIGHT
    _plot(charts.graph_figure(layout(S["ds"]), g.edge_index, groups, colours,
                              f"{NAMES[S['ds']]} - real graph, all {len(g.y):,} nodes"), f"graph_{stage}")
    st.caption(f"Structural shift: the source region is made of whole Louvain communities "
               f"({int(S['communities'].max()) + 1} found). The {S['model'].upper()} model was trained "
               f"on source labels only; calibration is judged on the target region it never saw. "
               f"GNN predictions = saved logits of run `{S['row'].run_id}`.")


# ---------------------------------------------------------------------------
# 2. Clustering
# ---------------------------------------------------------------------------

def render_clusters(S: dict, stage: int, live: bool):
    branch = "SGC-smoothed features" if S["gate"] == "sgc_kmeans" else "raw features"
    st.markdown(f"**Homophily gate:** homophily estimated from MLP-*predicted* labels = "
                f"**{S['h_hat']:.3f}** {'>' if S['h_hat'] > 0.5 else '<='} 0.5 -> cluster on "
                f"**{branch}** (SVD 64 dims, k-means with k = {S['K']} = number of classes). "
                f"Clusters recomputed live match the saved run on {S['cluster_match']:.0%} of nodes.")
    sizes = np.bincount(S["clusters"], minlength=S["K"])
    profile = pd.DataFrame(S["meta"].get("cluster_profile", [])).set_index("cluster")
    left, right = st.columns([3, 2])
    with right:
        sel = st.selectbox("Select a cluster", [None] + list(range(S["K"])), key="cluster_sel",
                           format_func=lambda c: "All clusters" if c is None else f"Cluster {c}"
                           ) if live else None
        pc = S.get("per_cluster", pd.DataFrame(columns=["cluster", "n_target"]))
        _plot(charts.cluster_sizes(pc, sizes, sel), f"sizes_{stage}")
        if sel is not None:
            _cluster_details(S, sel, sizes, profile)
    with left:
        names, colours = _cluster_groups(S["clusters"])
        highlight = None if sel is None else f"Cluster {sel}"
        _plot(charts.graph_figure(layout(S["ds"]), S["g"].edge_index, names, colours,
                                  "Same graph, coloured by LF-CTS cluster", highlight), f"clgraph_{stage}")


def _cluster_details(S, c, sizes, profile):
    in_target = int((S["target"] & (S["clusters"] == c)).sum())
    a, b = st.columns(2)
    a.metric("Nodes", f"{sizes[c]:,}")
    b.metric("In target region", f"{in_target:,}")
    if c in profile.index:
        p = profile.loc[c]
        a.metric("Average degree", f"{p.avg_degree:.2f}")
        b.metric("Conductance", f"{p.conductance:.3f}")
    if "a_hat" in S and in_target:
        a.metric("Estimated accuracy a_hat", f"{S['a_hat'][c]:.3f}")
        b.metric("Temperature T", f"{S['T'][c]:.3f}")
    if c in profile.index:
        st.caption(f"Purity {profile.loc[c].purity:.2f} and inside-homophily "
                   f"{profile.loc[c].inside_homophily:.2f} use true labels - shown for evaluation only.")
    if not in_target:
        st.caption("This cluster has no target-region nodes, so it gets no temperature of its own.")


# ---------------------------------------------------------------------------
# 3. LF-CTS calibration
# ---------------------------------------------------------------------------

def render_calibration(S: dict, stage: int, live: bool):
    st.markdown("**One global T vs one T per cluster.** Temperature scaling divides every node's "
                "logits by a single T fitted on source validation labels. LF-CTS gives each cluster "
                "its own T, chosen with **no labels**: the T at which the cluster's mean confidence "
                f"equals its estimated accuracy a_hat ({S['proxy']} of predicted labels).")
    pc = pd.DataFrame({"cluster": range(S["K"]), "a_hat": S["a_hat"]})
    pc = pc[[bool((S["target"] & (S["clusters"] == c)).any()) for c in pc.cluster]]
    show_true = live and stage >= 6 and st.checkbox(
        "Overlay true per-cluster accuracy (evaluation only - never used by the calibrator)", key="show_true")
    left, right = st.columns(2)
    with left:
        _plot(charts.accuracy_estimates(pc, S["true_acc_eval"] if show_true else None), f"ahat_{stage}")
    if stage < 4:
        return
    pc = S["per_cluster"] if "per_cluster" in S else pc.assign(T=S["T"][pc.cluster])
    with right:
        _plot(charts.temperatures(pc, S["T_global"]), f"temps_{stage}")
    if stage < 5:
        return
    _plot(charts.confidence_shift(pc), f"conf_{stage}")
    table = pc.rename(columns={"cluster": "Cluster", "n_target": "Target nodes", "a_hat": "a_hat",
                               "T": "T (LF-CTS)", "conf_before": "Conf. before",
                               "conf_global": "Conf. global T", "conf_after": "Conf. LF-CTS"})
    st.dataframe(table.style.format(precision=3), hide_index=True, use_container_width=True)
    st.caption(f"Live check: calibrated probabilities differ from the saved run's by at most "
               f"{S['probs_diff']:.1e}; temperatures by at most {np.abs(S['T'] - S['T_saved']).max():.4f} "
               f"(the saved values are rounded to 4 decimals). Clamped temperatures (T = 0.05 or 10): "
               f"{int(S['clamped'].sum())}.")


# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------

def render_results(S: dict, stage: int, live: bool):
    summ = S["ece_summary"]
    _plot(charts.ece_bars(S["ece_live"], summ, METHODS, S["seed"]), f"ece_{stage}")
    chosen = st.multiselect("Reliability diagrams for", list(METHODS), default=list(METHODS),
                            format_func=lambda m: METHODS[m], key="rel_methods") if live else list(METHODS)
    if chosen:
        _plot(charts.reliability(S["bins"], {m: METHODS[m] for m in chosen}, S["ece_live"]), f"rel_{stage}")
    rows = pd.DataFrame({
        "Method": [METHODS[m] for m in METHODS],
        "Labels used (all_results.csv)": [S["labels_used"].get(m, "?") for m in METHODS],
        f"ECE seed {S['seed']} (live)": [S["ece_live"][m] for m in METHODS],
        f"ECE seed {S['seed']} (all_results.csv)": [S["ece_saved"].get(m, np.nan) for m in METHODS],
        "Mean ECE seeds 0-4": summ["mean"].values, "Std": summ["std"].values,
    })
    st.dataframe(rows.style.format(precision=4), hide_index=True, use_container_width=True)
    st.markdown(_verdict(S))
    st.caption(f"ECE on {S['n_eval']:,} target_eval nodes, 15 equal-width bins. True labels are "
               f"read only here, to score the result. Model accuracy on target_eval: {S['acc']:.3f}.")


def _verdict(S: dict) -> str:
    m = S["ece_summary"]["mean"]
    word = lambda a, b: "lower (better)" if a < b else "higher (worse)"
    return (f"**Over seeds 0-4 on {NAMES[S['ds']]} / {S['model'].upper()}:** LF-CTS mean ECE "
            f"{m['lf_cts']:.3f} is {word(m['lf_cts'], m['none'])} than uncalibrated ({m['none']:.3f}) and "
            f"{word(m['lf_cts'], m['ts_val'])} than global temperature scaling ({m['ts_val']:.3f}); "
            f"LF-CTS+ {m['lf_cts_plus']:.3f}. These are the project's measured results, reported as they are.")
