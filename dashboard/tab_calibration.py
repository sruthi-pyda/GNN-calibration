"""dashboard/tab_calibration.py - Tab 4: Calibration.

Lane B (Saanvi) owns this file.
Phase P4-B.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.runs import (
    MISSING_DASHBOARD_DATA, conf_hist, load_meta, pick, rows_for, show, underlying,
)

METHOD_ORDER = ["none", "lf_ts", "atc_ts", "stac", "lf_cts", "lf_cts_plus", "ts_val",
                "cagcn_val", "ts_target_oracle", "cluster_ts_target_oracle"]
NO_CALIBRATED = "no calibrated rows yet - phase P4-A"


def _is_oracle(method: str) -> bool:
    return method.endswith("_target_oracle")


def _reliability(ax, bins: list, title: str) -> None:
    b = pd.DataFrame(bins)
    b = b[b["count"] > 0]
    ax.bar((b["bin_lower"] + b["bin_upper"]) / 2, b["acc"], width=b["bin_upper"] - b["bin_lower"],
           edgecolor="black", alpha=0.8, label="accuracy in bin")
    ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1, label="perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_title(title, fontsize=13)


def reliability(run: pd.Series, method_run: pd.Series | None) -> None:
    st.subheader("Reliability diagram")
    scheme = st.radio("Bins", ["equal width", "equal mass"], horizontal=True, key="cal_bins")
    key = "bins_uniform" if scheme == "equal width" else "bins_mass"
    runs = [run] + ([method_run] if method_run is not None else [])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, r in zip(axes, runs):
        meta = load_meta(r["run_id"])
        _reliability(ax, meta[key], f"{r['calib_method']}  (ECE {r['ece']:.3f})")
    if method_run is None:
        axes[1].text(0.5, 0.5, NO_CALIBRATED, ha="center", va="center", fontsize=12)
        axes[1].set_axis_off()
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left", frameon=False, fontsize=11)
    show(fig, "Bars below the dashed line mean the model is over-confident in that confidence "
              "range; bars above it mean under-confident. Computed on the target eval half.",
         pd.DataFrame(runs))


def histograms(run: pd.Series, method_run: pd.Series | None) -> None:
    st.subheader("Confidence histograms")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, r in zip(axes, [run, method_run]):
        h = conf_hist(r["run_id"]) if r is not None else None
        if h is None:
            ax.text(0.5, 0.5, NO_CALIBRATED if r is None else MISSING_DASHBOARD_DATA[:40],
                    ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            continue
        e = np.array(h["edges"])
        ax.bar(e[:-1], h["counts"], width=np.diff(e), align="edge", edgecolor="black", alpha=0.8)
        ax.axvline(h["mean_conf"], color="#D55E00", lw=2, label=f"mean conf {h['mean_conf']:.2f}")
        ax.axvline(h["acc"], color="#009E73", lw=2, ls="--", label=f"accuracy {h['acc']:.2f}")
        ax.set_title(r["calib_method"], fontsize=13)
        ax.set_xlabel("max softmax confidence")
        ax.legend(frameon=False, fontsize=11)
    axes[0].set_ylabel("target eval nodes")
    show(fig, "How confident the model is on target nodes, before and after calibration. If "
              "the mean confidence (orange) sits left of the accuracy (green), the model is "
              "under-confident.")


def ece_by_method(df: pd.DataFrame, ds: str, split: str) -> None:
    st.subheader("ECE by method")
    sub = rows_for(df, dataset=ds, split=split, ablation="main")
    agg = sub.groupby("calib_method")["ece"].agg(["mean", "std", "count"])
    order = [m for m in METHOD_ORDER if m in agg.index] + \
            [m for m in agg.index if m not in METHOD_ORDER]
    agg = agg.reindex(order)
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(agg) + 2), 3.8))
    for i, (m, r) in enumerate(agg.iterrows()):
        ax.bar(i, r["mean"], yerr=r["std"] if r["count"] > 1 else None, capsize=4,
               hatch="//" if _is_oracle(m) else None, edgecolor="black",
               color="#BBBBBB" if _is_oracle(m) else "#0072B2")
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([f"{m}\n(uses target labels)" if _is_oracle(m) else m for m in agg.index],
                       rotation=20, ha="right")
    ax.set_ylabel("ECE (lower is better)")
    note = "" if len(agg) > 1 else f" Only uncalibrated rows exist so far ({NO_CALIBRATED})."
    show(fig, f"Expected calibration error on {ds} ({split} split), mean ± std over seeds. "
              f"Hatched grey bars are oracles that peek at target labels - ceilings, not "
              f"competitors.{note}", agg.reset_index().round(4))


def temperature_slider(run: pd.Series) -> None:
    st.subheader("What one temperature would do")
    meta = load_meta(run["run_id"])
    curve = meta.get("ece_vs_T") if meta else None
    if not curve:
        st.info("No ECE-vs-temperature curve saved for this run")
        return
    T, e = np.array(curve["T"]), np.array(curve["ece"])
    i = st.select_slider("Temperature T", options=list(range(len(T))),
                         value=int(np.argmin(np.abs(T - 1.0))),
                         format_func=lambda k: f"{T[k]:.2f}", key="cal_T")
    best = int(np.argmin(e))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(T, e, color="#0072B2")
    ax.plot(T[i], e[i], "o", ms=10, color="#D55E00", label=f"T={T[i]:.2f}: ECE {e[i]:.3f}")
    ax.plot(T[best], e[best], "*", ms=14, color="#009E73",
            label=f"best T={T[best]:.2f}: ECE {e[best]:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("temperature (log scale; T < 1 sharpens, T > 1 softens)")
    ax.set_ylabel("ECE")
    ax.legend(frameon=False)
    show(fig, "ECE if every logit of this run were divided by one temperature T - read from a "
              "curve saved with the run, nothing is recomputed. The best T uses target labels, "
              "so it is a reference point, not a method.",
         pd.DataFrame(dict(T=T.round(3), ece=e.round(4))))


def render(df: pd.DataFrame) -> None:
    sel = pick(df[df["calib_method"] == "none"], "cal", cols=("dataset", "split", "seed"))
    if not sel:
        st.info("No uncalibrated rows to show")
        return
    run = rows_for(df, calib_method="none", ablation="main", **sel).iloc[0]
    methods = sorted(set(rows_for(df, **sel)["calib_method"]) - {"none"})
    method_run = None
    if methods:
        m = st.selectbox("Compare with method", methods, key="cal_method")
        method_run = rows_for(df, calib_method=m, ablation="main", **sel).iloc[0]
    else:
        st.info(f"Only the uncalibrated model is available - {NO_CALIBRATED}.")
    if load_meta(run["run_id"]) is None:
        st.info(f"No artifacts for {run['run_id']}")
        return
    reliability(run, method_run)
    histograms(run, method_run)
    ece_by_method(df, sel["dataset"], sel["split"])
    temperature_slider(run)
    underlying(rows_for(df, **sel))
