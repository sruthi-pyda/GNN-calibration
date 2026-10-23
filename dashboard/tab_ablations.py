"""dashboard/tab_ablations.py - Tab 6: Ablations.

Lane B (Saanvi) owns this file.
Phase P6-B.

Ablation rows are tagged in the `ablation` column. The knob value (k, hops, proxy variant,
cluster method) is not stored in the row, so it is read from the tag. The P5 sweep wrote
"k" (k=2), "k_14C", "k_28C", "hops_0".."hops_4" with bare "hops" (=2), "proxy"
(mlp_gnn_agreement), "cluster" (louvain) and "cluster_random"; see parse_tag / BARE_VALUES.
Takeaway sentences are computed from the rows shown, never typed in.
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.runs import default_index, show

FAMILIES = ["oracle_gate", "gate_src", "gate", "hops", "k", "proxy", "cluster", "degree"]
GATE_TAGS = {"gate": "forced wrong branch", "oracle_gate": "gate on true homophily",
             "gate_src": "gate on GCN predictions"}


# Tags as written by the P5 sweep (confirmed with Lane A): a bare family tag stands for that
# family's one non-default value, and "k_14C" / "k_28C" mean k = 14 / 28.
BARE_VALUES = {"k": "2", "hops": "2", "proxy": "mlp_gnn_agreement", "cluster": "louvain"}


def parse_tag(tag: str) -> tuple[str | None, str | None]:
    """'k_14C' -> ('k', '14'); 'hops' -> ('hops', '2'); 'gate' -> ('gate', None);
    unknown -> (None, None)."""
    for fam in FAMILIES:  # longest-prefix first: oracle_gate / gate_src before gate
        if tag == fam:
            return fam, BARE_VALUES.get(fam)
        if tag.startswith(fam) and tag[len(fam)] in "_=-":
            rest = re.sub(r"^[_=\-]+", "", tag[len(fam):])
            if fam in ("k", "hops"):
                m = re.fullmatch(r"(\d+)C?", rest)
                if m:
                    return fam, m.group(1)
            elif rest:
                return fam, rest
    return None, None


def _tagged(df: pd.DataFrame) -> pd.DataFrame:
    abl = df[df["ablation"].fillna("main") != "main"].copy()
    parsed = abl["ablation"].map(parse_tag)
    abl["family"] = parsed.map(lambda t: t[0])
    abl["value"] = parsed.map(lambda t: t[1])
    return abl


def _bars(ax, table: pd.DataFrame) -> None:
    x = np.arange(len(table.index))
    w = 0.8 / max(len(table.columns), 1)
    for i, col in enumerate(table.columns):
        ax.bar(x + (i - (len(table.columns) - 1) / 2) * w, table[col], w, label=str(col),
               edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=20)
    ax.set_ylabel("ECE")
    ax.legend(frameon=False, fontsize=10, loc="center left", bbox_to_anchor=(1.01, 0.5))


def _vs_main(table: pd.DataFrame, cols) -> str:
    parts = []
    for col in cols:
        d = (table[col] - table["main"]).dropna()
        if len(d):
            parts.append(f"{col}: {d.min():+.3f} to {d.max():+.3f} vs main "
                         f"(largest change on {d.abs().idxmax()})")
    return "; ".join(parts) if parts else "no dataset has both main and ablation rows"


def gate_section(abl, main, method):
    st.subheader("Gate")
    rows = abl[abl["family"].isin(GATE_TAGS)]
    if rows.empty:
        st.info("No gate / oracle_gate / gate_src rows yet")
        return
    m = pd.concat([main.assign(ablation="main"), rows])
    table = m.pivot_table(index="dataset", columns="ablation", values="ece", aggfunc="mean")
    cols = [c for c in ["main", *GATE_TAGS] if c in table]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    _bars(ax, table[cols].rename(columns={k: f"{k} ({v})" for k, v in GATE_TAGS.items()}))
    meaning = ", ".join(f"{k} = {v}" for k, v in GATE_TAGS.items() if k in cols)
    show(fig, f"{method} ECE with the gate as specified (main) and the gate ablations "
              f"({meaning}). Takeaway: {_vs_main(table, [c for c in cols if c != 'main'])}.",
         table[cols].round(4).reset_index())


def sweep_section(abl, main, family, label):
    st.subheader(label)
    rows = abl[abl["family"] == family]
    if rows.empty:
        st.info(f"No {family} ablation rows yet")
        return
    if rows["value"].isna().any():
        st.info(f"Knob value not recorded for {rows['value'].isna().sum()} '{family}' rows (bare "
                "tag), so they can't be placed on the axis.")
        rows = rows.dropna(subset=["value"])
        if rows.empty:
            return
    rows = rows.assign(x=rows["value"].astype(int))
    table = rows.pivot_table(index="x", columns="dataset", values="ece", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for ds in table.columns:
        ax.plot(table.index, table[ds], marker="o", label=ds)
    for ds, v in main.groupby("dataset")["ece"].mean().items():
        if ds in table.columns:
            ax.axhline(v, ls=":", lw=1, color="grey")
    ax.set_xlabel(family)
    ax.set_ylabel("ECE")
    ax.legend(frameon=False, fontsize=10, loc="center left", bbox_to_anchor=(1.01, 0.5))
    spread = (table.max() - table.min()).round(3)
    show(fig, f"ECE as {family} changes (dotted lines: the main setting). Takeaway: ECE varies by "
              f"at most {spread.max():.3f} across {family} values (largest spread on "
              f"{spread.idxmax()}).", table.round(4).reset_index())


def category_section(abl, main, family, label):
    st.subheader(label)
    rows = abl[abl["family"] == family]
    if rows.empty:
        st.info(f"No {family} ablation rows yet")
        return
    rows = rows.assign(variant=rows["value"].fillna(rows["ablation"]))
    m = pd.concat([main.assign(variant="main"), rows])
    table = m.pivot_table(index="dataset", columns="variant", values="ece", aggfunc="mean")
    cols = ["main"] + [c for c in table.columns if c != "main"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    _bars(ax, table[cols])
    show(fig, f"ECE for each {family} variant against the main setting. Takeaway: "
              f"{_vs_main(table, cols[1:])}.", table[cols].round(4).reset_index())


def degree_section(abl):
    st.subheader("Degree split")
    rows = abl[abl["family"] == "degree"]
    if rows.empty:
        st.info("No degree-split rows yet")
        return
    table = rows.pivot_table(index="dataset", columns="calib_method", values="ece", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    _bars(ax, table)
    best = table.idxmin(axis=1)
    show(fig, "ECE by method when the source is the lowest-degree half of the graph. Takeaway: "
              "lowest ECE per dataset - " + ", ".join(f"{d}: {m}" for d, m in best.items()) + ".",
         table.round(4).reset_index())


def render(df: pd.DataFrame) -> None:
    abl = _tagged(df)
    if abl.empty:
        st.info("No ablation rows yet - they come from the phase P5 sweep.")
        return
    unknown = abl[abl["family"].isna()]["ablation"].unique().tolist()
    if unknown:
        st.warning(f"Unrecognised ablation tags (not shown): {unknown}")
    c1, c2, c3 = st.columns(3)
    methods = sorted(abl["calib_method"].dropna().unique().tolist())
    method = c1.selectbox("Method", methods, index=default_index(methods, "calib_method"),
                          key="abl_method")
    splits = sorted(abl[abl["family"] != "degree"]["split"].dropna().unique().tolist()) or ["structural"]
    split = c2.selectbox("Split", splits, index=default_index(splits, "split"), key="abl_split")
    models = sorted(abl["model"].dropna().unique().tolist())
    model = c3.selectbox("Model", models, index=default_index(models, "model"), key="abl_model")
    keep = (abl["calib_method"] == method) & (abl["model"] == model)
    main = df[(df["ablation"].fillna("main") == "main") & (df["calib_method"] == method)
              & (df["split"] == split) & (df["model"] == model)]
    sub = abl[keep & ((abl["split"] == split) | (abl["family"] == "degree"))]
    gate_section(sub, main, method)
    sweep_section(sub, main, "k", "Number of clusters k")
    sweep_section(sub, main, "hops", "SGC smoothing hops")
    category_section(sub, main, "proxy", "Accuracy-proxy variant")
    category_section(sub, main, "cluster", "Clustering method")
    degree_section(abl[(abl["model"] == model) & (abl["family"] == "degree")])
