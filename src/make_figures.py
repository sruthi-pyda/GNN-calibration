"""src/make_figures.py – Generate paper figures F1-F7.

Lane A (Sruthi). Phase P6-A.
Reads results/ and cache/ (y labels for NMI). Never re-runs experiments.
Output: figures/*.png (200 dpi) and figures/*.pdf.
Run: python -m src.make_figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

ROOT  = Path(__file__).parent.parent
FIGS  = ROOT / "figures"
RUNS  = ROOT / "results" / "runs"
CSV   = ROOT / "results" / "all_results.csv"
CACHE = ROOT / "cache"

FIGS.mkdir(exist_ok=True)

# Wong (2011) colour-blind-safe palette + distinct markers
WONG  = ["#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7","#000000"]
MARKS = ["o","s","^","D","v","P","X","*"]
DS_ORDER = ["cora","citeseer","pubmed","chameleon","roman_empire"]
DS_NAMES = {"cora":"Cora","citeseer":"CiteSeer","pubmed":"PubMed",
            "chameleon":"Chamln.","roman_empire":"Roman-Emp."}
M_ORDER = ["none","lf_ts","atc_ts","stac_global","lf_cts","lf_cts_plus",
           "ts_val","cagcn_val","ts_target_oracle","cluster_ts_target_oracle"]
M_NAMES = {"none":"None","lf_ts":"LF-TS","atc_ts":"ATC-TS","stac_global":"STAC",
           "lf_cts":"LF-CTS","lf_cts_plus":"LF-CTS+","ts_val":"TS-val",
           "cagcn_val":"CaGCN","ts_target_oracle":"Oracle-TS",
           "cluster_ts_target_oracle":"Cl.Oracle"}
ORACLES = {"ts_target_oracle","cluster_ts_target_oracle"}

# Projector-readable defaults
plt.rcParams.update({"font.size":13,"figure.dpi":120,"axes.grid":True,
                     "grid.alpha":0.3,"axes.spines.top":False,
                     "axes.spines.right":False})


# ── helpers ───────────────────────────────────────────────────────────────────

def _csv() -> pd.DataFrame:
    return pd.read_csv(CSV)

def _meta(run_id: str) -> dict | None:
    p = RUNS / run_id / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None

def _savefig(fig: plt.Figure, stem: str, caption: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{stem}] {caption}")

def _cache_y(ds: str) -> np.ndarray | None:
    """Load true y labels from cache/<dataset>.npz."""
    name = "chameleon_filtered" if ds == "chameleon" else ds
    p = CACHE / f"{name}.npz"
    return np.load(p)["y"] if p.exists() else None


# ── F1: ECE random vs structural ──────────────────────────────────────────────

def make_f1_ece_random_vs_structural():
    """F1: ECE random vs structural, grouped bars per dataset, seed-std error bars (C1)."""
    df = _csv()
    sel = df[(df.ablation == "main") & (df.model == "gcn") & (df.calib_method == "none")]
    grp = sel.groupby(["dataset","split"])["ece"]
    mean = grp.mean().unstack(); std = grp.std().unstack()

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(DS_ORDER)); w = 0.35
    for i, (sp, col) in enumerate([("random", WONG[1]), ("structural", WONG[5])]):
        m = [mean.loc[d, sp] if d in mean.index and sp in mean.columns else np.nan for d in DS_ORDER]
        s = [std.loc[d, sp]  if d in std.index  and sp in std.columns  else np.nan for d in DS_ORDER]
        ax.bar(x + i*w - w/2, m, w, yerr=s, capsize=4, label=sp.capitalize(),
               color=col, alpha=0.85, error_kw={"elinewidth":1.2})
    ax.set_xticks(x); ax.set_xticklabels([DS_NAMES[d] for d in DS_ORDER])
    ax.set_ylabel("ECE"); ax.legend()
    fig.tight_layout()
    _savefig(fig, "f1_ece_split",
             "F1: ECE under random vs structural shift (GCN, uncalibrated, seeds 0–4 ± std).")


# ── F2: reliability diagrams ──────────────────────────────────────────────────

def make_f2_reliability_diagrams():
    """F2: 2×2 reliability diagrams: Cora random/structural × none/lf_cts (seed 0)."""
    configs = [("random","none"),("structural","none"),
               ("random","lf_cts"),("structural","lf_cts")]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)
    for ax, (sp, m) in zip(axes.flat, configs):
        meta = _meta(f"cora__{sp}__gcn__s0__{m}__main")
        if not meta:
            ax.set_title(f"{sp}/{m} (no data)"); continue
        bdf = pd.DataFrame(meta["bins_mass"]); bdf = bdf[bdf["count"] > 0]
        mid = (bdf["bin_lower"] + bdf["bin_upper"]) / 2
        wid = bdf["bin_upper"] - bdf["bin_lower"]
        ece_val = (abs(bdf["acc"] - bdf["avg_conf"]) * bdf["count"]).sum() / bdf["count"].sum()
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect")
        ax.bar(mid, bdf["acc"], width=wid * 0.85, alpha=0.5, color=WONG[1], label="Accuracy")
        ax.plot(mid, bdf["avg_conf"], "o-", color=WONG[5], ms=5, lw=1.6, label="Confidence")
        ax.set_title(f"{sp.capitalize()} / {M_NAMES.get(m,m)}\nECE={ece_val:.3f}", fontsize=10)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    for ax in axes[1]: ax.set_xlabel("Confidence")
    for ax in axes[:, 0]: ax.set_ylabel("Accuracy")
    fig.tight_layout()
    _savefig(fig, "f2_reliability",
             "F2: Reliability diagrams – Cora GCN, seed 0, equal-mass bins, "
             "random/structural × none/LF-CTS.")


# ── F3: per-cluster ECE heatmap ───────────────────────────────────────────────

def make_f3_cluster_ece_heatmap():
    """F3: Per-cluster ECE heatmap, Cora structural GCN, clusters × methods (C2)."""
    methods = ["none","lf_ts","lf_cts","lf_cts_plus",
               "ts_val","ts_target_oracle","cluster_ts_target_oracle"]
    n_cl = 7
    data_sum = np.zeros((n_cl, len(methods))); data_cnt = np.zeros_like(data_sum)
    prof = None
    for s in range(5):
        for ji, m in enumerate(methods):
            meta = _meta(f"cora__structural__gcn__s{s}__{m}__main")
            if not meta: continue
            if prof is None and s == 0 and m == "none":
                prof = meta["cluster_profile"]
            for r in meta["per_cluster"]:
                v = r.get("ece", float("nan"))
                if v == v:  # not nan
                    data_sum[r["cluster_id"], ji] += v
                    data_cnt[r["cluster_id"], ji] += 1
    data = np.where(data_cnt > 0, data_sum / data_cnt, np.nan)

    ylabs = ([f"Cl{r['cluster']}  p={r['purity']:.2f}  d={r['avg_degree']:.1f}" for r in prof]
             if prof else [f"Cl{i}" for i in range(n_cl)])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([M_NAMES.get(m, m) for m in methods], rotation=30, ha="right")
    ax.set_yticks(range(n_cl)); ax.set_yticklabels(ylabs, fontsize=10)
    fig.colorbar(im, ax=ax, label="ECE"); ax.set_title("Cora structural GCN – per-cluster ECE")
    fig.tight_layout()
    _savefig(fig, "f3_cluster_ece_heatmap",
             "F3: Per-cluster ECE heatmap – Cora structural GCN, rows labelled by purity (p) "
             "and mean degree (d), averaged over seeds 0–4.")


# ── F4: proxy vs true accuracy scatter ───────────────────────────────────────

def make_f4_proxy_scatter():
    """F4: Proxy vs true per-cluster accuracy, all datasets, identity line, Pearson r (C3)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, (method, title) in zip(axes, [("lf_cts", "a_hat (LF-CTS)"),
                                           ("lf_cts_plus", "ã (LF-CTS+)")]):
        for ds, col, mk in zip(DS_ORDER, WONG, MARKS):
            xs, ys = [], []
            for s in range(5):
                meta = _meta(f"{ds}__structural__gcn__s{s}__{method}__main")
                if not meta or "proxy_quality" not in meta: continue
                for row in meta["proxy_quality"]["table"]:
                    ta = row.get("true_acc")
                    if ta is not None and ta == ta:  # not None/nan
                        xs.append(row["a_hat"]); ys.append(ta)
            if not xs: continue
            r = float(np.corrcoef(xs, ys)[0, 1])
            ax.scatter(xs, ys, color=col, marker=mk, s=25, alpha=0.65,
                       label=f"{DS_NAMES[ds]} r={r:.2f}")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.set_xlabel("Proxy estimate"); ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("True accuracy")
    fig.tight_layout()
    _savefig(fig, "f4_proxy_scatter",
             "F4: Proxy vs true per-cluster accuracy – identity line (dashed), "
             "all datasets structural GCN seeds 0–4, r shown per dataset.")


# ── F5: NMI and ECE vs hops ───────────────────────────────────────────────────

def make_f5_hops_nmi_ece():
    """F5: NMI and ECE vs hops (0–4), one line per dataset, gate annotation (C4)."""
    hop_map = {"hops_0": 0, "hops_1": 1, "hops": 2, "hops_3": 3, "hops_4": 4}
    df = _csv()
    sel = df[(df.ablation.isin(hop_map)) & (df.split == "structural") &
             (df.model == "gcn") & (df.calib_method == "lf_cts")].copy()
    sel["hops_val"] = sel["ablation"].map(hop_map)
    ece_df = sel.groupby(["dataset","hops_val"])["ece"].mean().unstack()

    # NMI: load stored clusters.npy vs true y
    nmi_rows = []
    for ds in DS_ORDER:
        y = _cache_y(ds)
        if y is None: continue
        for abl, hv in hop_map.items():
            nmis = []
            for s in range(5):
                cp = RUNS / f"{ds}__structural__gcn__s{s}__lf_cts__{abl}" / "clusters.npy"
                if cp.exists():
                    nmis.append(normalized_mutual_info_score(
                        y, np.load(cp), average_method="arithmetic"))
            if nmis: nmi_rows.append({"dataset": ds, "hops_val": hv, "nmi": np.mean(nmis)})
    nmi_df = (pd.DataFrame(nmi_rows).pivot(index="dataset", columns="hops_val", values="nmi")
              if nmi_rows else pd.DataFrame())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    hops_vals = sorted(hop_map.values())
    gate_cache = df[(df.ablation=="main")&(df.model=="gcn")&(df.split=="structural")&
                    (df.calib_method=="none")].groupby("dataset")["homophily_gate"].agg(
                    lambda x: x.mode()[0])
    for ds, col, mk in zip(DS_ORDER, WONG, MARKS):
        ls = "-" if gate_cache.get(ds,"?") == "sgc_kmeans" else "--"
        if ds in ece_df.index:
            ax1.plot(hops_vals, [ece_df.loc[ds, h] for h in hops_vals],
                     color=col, marker=mk, ls=ls, label=DS_NAMES[ds])
        if ds in nmi_df.index:
            ax2.plot(hops_vals, [nmi_df.loc[ds, h] for h in hops_vals],
                     color=col, marker=mk, ls=ls)
    ax1.set_ylabel("ECE"); ax1.legend(fontsize=9)
    ax1.set_title("ECE and NMI vs SGC smoothing hops (structural GCN, LF-CTS)")
    ax2.set_ylabel("NMI (clusters vs true labels)"); ax2.set_xlabel("Hops")
    ax2.set_xticks(hops_vals)
    fig.text(0.005, 0.5, "solid=sgc_kmeans gate  dashed=feature_kmeans",
             fontsize=8, va="center", rotation=90)
    fig.tight_layout(rect=[0.015, 0, 1, 1])
    _savefig(fig, "f5_hops_nmi_ece",
             "F5: ECE and NMI vs smoothing hops – structural GCN, LF-CTS; solid lines use "
             "the sgc_kmeans gate (hops relevant), dashed use feature_kmeans (hops bypassed).")


# ── F6: ECE by method ─────────────────────────────────────────────────────────

def make_f6_ece_by_method():
    """F6: ECE by method under structural shift, ordered, oracles hatched (C3, C5)."""
    df = _csv()
    sel = df[(df.ablation == "main") & (df.split == "structural") & (df.model == "gcn")]
    grp = sel.groupby(["dataset","calib_method"])["ece"]
    mean = grp.mean().unstack(); std = grp.std().unstack()
    avail = [m for m in M_ORDER if m in mean.columns]

    x = np.arange(len(avail)); w = 0.15
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (ds, col) in enumerate(zip(DS_ORDER, WONG)):
        offset = (i - (len(DS_ORDER) - 1) / 2) * w
        m = [mean.loc[ds, c] if ds in mean.index and c in mean.columns else np.nan for c in avail]
        s = [std.loc[ds, c]  if ds in std.index  and c in std.columns  else np.nan for c in avail]
        bars = ax.bar(x + offset, m, w, label=DS_NAMES[ds], color=col, alpha=0.85,
                      yerr=s, capsize=2, error_kw={"elinewidth": 1})
        for bar, c in zip(bars, avail):
            if c in ORACLES:
                bar.set_hatch("///"); bar.set_edgecolor("white")
    ax.set_xticks(x); ax.set_xticklabels([M_NAMES.get(c, c) for c in avail],
                                          rotation=30, ha="right")
    ax.set_ylabel("ECE")
    oracle_p = mpatches.Patch(facecolor="white", edgecolor="grey", hatch="///",
                               label="Uses target labels")
    handles = ([mpatches.Patch(color=WONG[i], label=DS_NAMES[ds])
                for i, ds in enumerate(DS_ORDER)] + [oracle_p])
    ax.legend(handles=handles, ncol=3, fontsize=9)
    fig.tight_layout()
    _savefig(fig, "f6_ece_by_method",
             "F6: Mean ECE by calibration method – structural shift, GCN, seeds 0–4 ± std; "
             "hatched bars use target-split labels (oracle ceiling).")


# ── F7: homophily estimates ───────────────────────────────────────────────────

def make_f7_homophily_estimates():
    """F7: h_hat_mlp vs h_hat_gnn vs true homophily per dataset (C4, gate-bias finding)."""
    df = _csv()
    sel = df[(df.ablation == "main") & (df.split == "structural") &
             (df.model == "gcn") & (df.calib_method == "none")]
    grp = sel.groupby("dataset")[["h_hat_mlp","h_hat_gnn","test_homophily"]].mean()

    x = np.arange(len(DS_ORDER)); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))
    specs = [("h_hat_mlp", "h_hat (MLP)", WONG[1]),
             ("h_hat_gnn", "h_hat (GCN)", WONG[2]),
             ("test_homophily", "True h", WONG[5])]
    for i, (col, lbl, color) in enumerate(specs):
        vals = [grp.loc[d, col] if d in grp.index else np.nan for d in DS_ORDER]
        ax.bar(x + (i - 1) * w, vals, w, label=lbl, color=color, alpha=0.85)
    ax.axhline(0.5, color="black", ls="--", lw=1.4, label="Gate threshold (0.5)")
    ax.set_xticks(x); ax.set_xticklabels([DS_NAMES[d] for d in DS_ORDER])
    ax.set_ylabel("Edge homophily"); ax.set_ylim(0, 1); ax.legend()
    fig.tight_layout()
    _savefig(fig, "f7_homophily",
             "F7: MLP-predicted, GCN-predicted, and true edge homophily per dataset – "
             "structural GCN, seeds 0–4. Dashed line = gate threshold (0.5). "
             "MLP underestimates Cora (0.45 vs true 0.81); GCN overestimates heterophilic graphs.")


# ── main ──────────────────────────────────────────────────────────────────────

_ALL = [
    make_f1_ece_random_vs_structural,
    make_f2_reliability_diagrams,
    make_f3_cluster_ece_heatmap,
    make_f4_proxy_scatter,
    make_f5_hops_nmi_ece,
    make_f6_ece_by_method,
    make_f7_homophily_estimates,
]

if __name__ == "__main__":
    ok = err = 0
    for fn in _ALL:
        try:
            fn(); ok += 1
        except Exception as e:
            print(f"[{fn.__name__}] FAILED: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)
            err += 1
    print(f"\n{ok}/{ok+err} figures written to {FIGS}/")
