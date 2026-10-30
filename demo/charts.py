"""demo/charts.py - Plotly figures for the live demo (Lane B).

Every function takes plain arrays/frames and returns a go.Figure; no Streamlit here.
Colours: one fixed categorical order (validated palette), entity -> colour never
depends on rank, grey for "other" and for the uncalibrated baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY, LIGHT, INK, MUTED = "#8a8984", "#d6d5d0", "#0b0b0b", "#52514e"
METHOD_COLOURS = {"none": GREY, "ts_val": SERIES[0], "lf_cts": SERIES[1], "lf_cts_plus": SERIES[2]}
REGION_COLOURS = {"Source labels (train + val)": SERIES[0], "Source region (unlabelled)": "#86b6ef",
                  "Target region": SERIES[1]}


def _layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(color=INK, size=13),
                      legend=dict(orientation="h", y=-0.08), hovermode="closest", **kw)
    fig.update_xaxes(gridcolor="#eeeeec", zerolinecolor=LIGHT)
    fig.update_yaxes(gridcolor="#eeeeec", zerolinecolor=LIGHT)
    return fig


def cluster_colour(c: int) -> str:
    return SERIES[c % len(SERIES)]


# ---------------------------------------------------------------------------
# Graph and clusters
# ---------------------------------------------------------------------------

def graph_figure(pos: np.ndarray, edge_index: np.ndarray, groups: np.ndarray,
                 colours: dict, title: str, highlight=None, height: int = 560) -> go.Figure:
    """Nodes coloured by group (legend = group name), edges as one faint trace.
    highlight: a group to keep in colour while every other group turns light grey."""
    src, dst = edge_index[:, edge_index[0] < edge_index[1]]
    ex = np.column_stack([pos[src, 0], pos[dst, 0], np.full(len(src), np.nan)]).ravel()
    ey = np.column_stack([pos[src, 1], pos[dst, 1], np.full(len(src), np.nan)]).ravel()
    fig = go.Figure(go.Scattergl(x=ex, y=ey, mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color="rgba(120,120,115,0.15)", width=0.5)))
    for name, colour in colours.items():
        idx = np.where(groups == name)[0]
        if not len(idx):
            continue
        dim = highlight is not None and name != highlight
        fig.add_trace(go.Scattergl(
            x=pos[idx, 0], y=pos[idx, 1], mode="markers", name=str(name),
            marker=dict(size=5 if dim else 6, color=LIGHT if dim else colour,
                        line=dict(width=0.4, color="white")),
            text=[f"node {i}" for i in idx], hovertemplate="%{text}<br>" + str(name) + "<extra></extra>"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return _layout(fig, height, title=title)


def cluster_sizes(per_cluster: pd.DataFrame, sizes: np.ndarray, selected=None) -> go.Figure:
    ids = list(range(len(sizes)))
    n_target = dict(zip(per_cluster.cluster.astype(int), per_cluster.n_target.astype(int)))
    fig = go.Figure(go.Bar(
        x=[f"C{c}" for c in ids], y=sizes,
        marker_color=[cluster_colour(c) if selected in (None, c) else LIGHT for c in ids],
        customdata=[n_target.get(c, 0) for c in ids],
        hovertemplate="%{x}: %{y} nodes<br>%{customdata} in the target region<extra></extra>"))
    return _layout(fig, 300, title="Cluster sizes (all nodes)", yaxis_title="nodes", showlegend=False)


# ---------------------------------------------------------------------------
# LF-CTS
# ---------------------------------------------------------------------------

def accuracy_estimates(pc: pd.DataFrame, true_acc=None) -> go.Figure:
    """Label-free a_hat per cluster; true accuracy only if passed (evaluation only)."""
    x = [f"C{c}" for c in pc.cluster]
    fig = go.Figure(go.Bar(x=x, y=pc.a_hat, name="Estimated accuracy a_hat (label-free)",
                           marker_color=SERIES[1], hovertemplate="%{x}: a_hat %{y:.3f}<extra></extra>"))
    if true_acc is not None:
        fig.add_trace(go.Scatter(x=x, y=[true_acc[c] for c in pc.cluster], mode="markers",
                                 name="True accuracy (evaluation only, never used)",
                                 marker=dict(symbol="diamond", size=11, color=INK),
                                 hovertemplate="%{x}: true acc %{y:.3f}<extra></extra>"))
    fig.update_yaxes(range=[0, 1.05])
    return _layout(fig, 340, title="Step 3 - estimated accuracy per cluster",
                   yaxis_title="accuracy", bargap=0.35)


def temperatures(pc: pd.DataFrame, T_global: float) -> go.Figure:
    x = [f"C{c}" for c in pc.cluster]
    fig = go.Figure(go.Bar(x=x, y=pc["T"], name="LF-CTS: one T per cluster",
                           marker_color=[cluster_colour(c) for c in pc.cluster],
                           hovertemplate="%{x}: T = %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=[T_global] * len(x), mode="lines", name=f"One global T = {T_global:.3f} (TS)",
                             line=dict(color=INK, width=2, dash="dash"), hoverinfo="skip"))
    return _layout(fig, 340, title="Step 4 - per-cluster temperature vs one global temperature",
                   yaxis_title="temperature T", bargap=0.35)


def confidence_shift(pc: pd.DataFrame) -> go.Figure:
    """Mean confidence of target nodes per cluster: before, global TS, after LF-CTS."""
    x = [f"C{c}" for c in pc.cluster]
    fig = go.Figure()
    for col, name, colour in [("conf_before", "Original (uncalibrated)", GREY),
                              ("conf_global", "After one global T", SERIES[0]),
                              ("conf_after", "After LF-CTS", SERIES[1])]:
        fig.add_trace(go.Bar(x=x, y=pc[col], name=name, marker_color=colour,
                             hovertemplate="%{x}: %{y:.3f}<extra>" + name + "</extra>"))
    fig.add_trace(go.Scatter(x=x, y=pc.a_hat, mode="markers", name="Target: a_hat",
                             marker=dict(symbol="line-ew", size=26, line=dict(width=3, color=INK)),
                             hovertemplate="%{x}: a_hat %{y:.3f}<extra></extra>"))
    fig.update_yaxes(range=[0, 1.05])
    return _layout(fig, 360, title="Step 5 - mean confidence of target nodes, before and after",
                   yaxis_title="mean confidence", barmode="group", bargap=0.25, bargroupgap=0.08)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def ece_bars(live: dict, summary: pd.DataFrame, labels: dict, seed: int) -> go.Figure:
    names = [labels[m] for m in labels]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[live[m] for m in labels], name=f"This run (seed {seed}, recomputed live)",
                         marker_color=[METHOD_COLOURS[m] for m in labels],
                         hovertemplate="%{x}<br>ECE %{y:.4f}<extra>seed " + str(seed) + "</extra>"))
    fig.add_trace(go.Scatter(
        x=names, y=summary["mean"], mode="markers", name="Mean over seeds 0-4 (all_results.csv), +/- std",
        error_y=dict(type="data", array=summary["std"].fillna(0), color=INK, thickness=1.5),
        marker=dict(symbol="diamond", size=11, color=INK),
        hovertemplate="%{x}<br>mean ECE %{y:.4f}<extra></extra>"))
    return _layout(fig, 380, title="Expected calibration error on target_eval (lower is better)",
                   yaxis_title="ECE", bargap=0.4)


def reliability(bins: dict, labels: dict, ece_live: dict) -> go.Figure:
    methods = list(labels)
    fig = make_subplots(rows=1, cols=len(methods), shared_yaxes=True,
                        subplot_titles=[f"{labels[m].split(' (')[0]}<br>ECE {ece_live[m]:.3f}" for m in methods])
    for j, m in enumerate(methods, start=1):
        b = bins[m].dropna(subset=["acc"])
        mid = (b.bin_lower + b.bin_upper) / 2
        fig.add_trace(go.Bar(x=mid, y=b.acc, width=(b.bin_upper - b.bin_lower) * 0.92,
                             marker_color=METHOD_COLOURS[m], name=labels[m], showlegend=False,
                             customdata=np.column_stack([b.avg_conf, b["count"]]),
                             hovertemplate="accuracy %{y:.3f}<br>mean conf %{customdata[0]:.3f}"
                                           "<br>%{customdata[1]} nodes<extra></extra>"), 1, j)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color=INK, width=1, dash="dot")), 1, j)
        fig.update_xaxes(range=[0, 1], title_text="confidence", row=1, col=j)
    fig.update_yaxes(range=[0, 1], title_text="accuracy", row=1, col=1)
    fig = _layout(fig, 360, title="")
    fig.update_layout(margin=dict(t=70))
    return fig
