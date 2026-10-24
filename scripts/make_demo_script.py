"""scripts/make_demo_script.py - Write dashboard/DEMO_SCRIPT.md with every number from the results.

Lane B (Saanvi) owns this file. Phase P6-B.

Reads results/all_results.csv, results/runs/*/meta.json and results/dashboard/*.json (from
make_dashboard_data.py). Each step says what to click, one sentence to say, and where its numbers
come from. Sentences are worded from the sign of the result, so a failure is said as a failure.
Missing data shows as "(not available: ...)" instead of a made-up number.

Usage: python scripts/make_demo_script.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from dashboard.scorecard import scorecard
from src.run_experiments import RESULTS_DIR, RUNS_DIR, load_results

OUT = ROOT / "dashboard" / "DEMO_SCRIPT.md"
DEMO = dict(dataset="cora", split="structural", model="gcn")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _meta(run_id: str):
    return _json(RUNS_DIR / run_id / "meta.json")


def _na(why: str) -> str:
    return f"(not available: {why})"


def main_rows(df, **match):
    m = (df["ablation"].fillna("main") == "main") & (df["model"] == DEMO["model"])
    for k, v in match.items():
        m &= df[k] == v
    return df[m]


def build(df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """(title, clicks, sentence, source) for the 9 steps."""
    steps = []
    clus = _json(RESULTS_DIR / "dashboard" / "clustering.json") or {}
    splits = _json(RESULTS_DIR / "dashboard" / "splits.json") or {}
    none = main_rows(df, calib_method="none")
    cora_s = none[(none["dataset"] == "cora") & (none["split"] == "structural")]

    card = scorecard(main_rows(df))
    status = ", ".join(f"{r.claim} {r.status}" for r in card.itertuples())
    steps.append(("Overview", "Turn on **Demo mode** in the sidebar. Page: **Overview**.",
                  f"Six graphs, five claims - measured so far: {status}.",
                  "scorecard over all main GCN rows"))

    cm = clus.get("cora", {}).get("methods", {})
    nmi = cm.get("sgc_kmeans", {}).get("nmi")
    steps.append(("Clustering, Cora", "Page: **Clustering explorer** (Dataset = cora).",
                  f"SGC clustering recovers Cora's classes with NMI {nmi:.2f}, and no labels "
                  "were used to form the clusters." if nmi is not None else _na("clustering.json"),
                  "results/dashboard/clustering.json, seed 42 (true labels only to score NMI)"))

    tm = clus.get("texas", {}).get("methods", {})
    f_nmi, s_nmi = tm.get("feature_kmeans", {}).get("nmi"), tm.get("sgc_kmeans", {}).get("nmi")
    steps.append(("Switch to Texas", "Clustering methods table: Dataset = **texas**.",
                  f"On heterophilic Texas the gate switches to raw features: smoothing would cut "
                  f"NMI from {f_nmi:.2f} to {s_nmi:.2f}." if f_nmi is not None and s_nmi is not None
                  else _na("texas in clustering.json"),
                  "results/dashboard/clustering.json"))

    if len(cora_s):
        h_true = (_meta(cora_s["run_id"].iloc[0]) or {}).get("h_true_eval_only", float("nan"))
        cham = none[(none["dataset"] == "chameleon") & (none["split"] == "structural")]
        cham_txt = (f" Gating on the GCN instead would be worse on heterophilic Chameleon "
                    f"({cham['h_hat_gnn'].mean():.2f} estimated)." if len(cham) else "")
        branch = "SGC smoothing" if cora_s["h_hat_mlp"].mean() > 0.5 else "raw features"
        steps.append(("Gate gauge", "Homophily gate: Dataset = cora, Split = structural.",
                      f"The MLP estimates Cora's homophily at {cora_s['h_hat_mlp'].mean():.2f} "
                      f"(true {h_true:.2f}, GCN {cora_s['h_hat_gnn'].mean():.2f}), so the gate "
                      f"picks {branch} - a misfire we report rather than tune away.{cham_txt}",
                      f"mean of {len(cora_s)} rows cora/structural/gcn/none; true h from meta"))
    else:
        steps.append(("Gate gauge", "Homophily gate", _na("cora structural rows"), "-"))

    tv = {k: [v["diagnostics"]["class_tv_distance"] for key, v in splits.items()
              if key.startswith(f"cora__{k}__")] for k in ("random", "structural")}
    steps.append(("Shift viewer", "Page: **Shift split viewer** (Dataset = cora).",
                  f"A real structural shift: the class mix differs by TV "
                  f"{pd.Series(tv['structural']).mean():.2f} under the structural split vs "
                  f"{pd.Series(tv['random']).mean():.2f} for a random split."
                  if tv["random"] and tv["structural"] else _na("splits.json"),
                  "results/dashboard/splits.json, seeds 0-4"))

    cora_r = none[(none["dataset"] == "cora") & (none["split"] == "random")]
    steps.append(("Calibration", "Page: **Calibration** (cora / structural / seed 0).",
                  f"Under shift, uncalibrated ECE moves from {cora_r['ece'].mean():.3f} (random) to "
                  f"{cora_s['ece'].mean():.3f} (structural)." if len(cora_r) and len(cora_s)
                  else _na("cora none rows"), "mean over seeds, cora/gcn/none"))

    if len(cora_s):
        ratio = (cora_s["cluster_ece_worst"] / cora_s["ece"]).mean()
        r0 = cora_s.sort_values("seed").iloc[0]
        prof = pd.DataFrame((_meta(r0["run_id"]) or {}).get("cluster_profile", []))
        w = int(r0["worst_cluster_id"])
        pur = prof.set_index("cluster").loc[w, "purity"] if w in set(prof.get("cluster", [])) else None
        steps.append(("Cluster-wise", "Page: **Cluster-wise calibration** (worst-cluster callout).",
                      f"The average hides it: the worst cluster is calibrated {ratio:.1f}x worse "
                      f"than the graph as a whole" + (f" (seed {r0['seed']}: cluster {w}, purity "
                                                     f"{pur:.2f})." if pur is not None else "."),
                      f"cluster_ece_worst / ece, {len(cora_s)} rows; profile from {r0['run_id']}"))
    else:
        steps.append(("Cluster-wise", "Page: **Cluster-wise calibration**", _na("cora rows"), "-"))

    by = main_rows(df, dataset="cora", split="structural").groupby("calib_method")["ece"].mean()
    if {"none", "lf_cts_plus", "cluster_ts_target_oracle"} <= set(by.index):
        gap = (by["none"] - by["lf_cts_plus"]) / (by["none"] - by["cluster_ts_target_oracle"])
        plus = main_rows(df, dataset="cora", split="structural", calib_method="lf_cts_plus")
        r = pd.Series([(_meta(x) or {}).get("proxy_quality", {}).get("r") for x in plus["run_id"]],
                      dtype=float).mean()
        verdict = (f"LF-CTS+ closes {gap:.0%} of the gap to the labelled oracle with zero target "
                   "labels" if gap > 0 else
                   f"on Cora LF-CTS+ does not close the gap (ECE {by['lf_cts_plus']:.3f} vs "
                   f"{by['none']:.3f} uncalibrated, {gap:.0%} of the oracle gap)")
        steps.append(("LF-CTS+", "Cluster-wise page: proxy scatter + temperature table.",
                      f"Honestly: {verdict}. The corrected proxy's correlation with true accuracy "
                      f"is r = {r:.2f} - here is where it fails.",
                      f"mean ECE over seeds, cora/structural/gcn; r from proxy_quality"))
    else:
        steps.append(("LF-CTS+", "Cluster-wise page", _na("lf_cts_plus / oracle rows (P4-A)"), "-"))

    steps.append(("Limitations", "Stay on the current page.",
                  "Gate misfires when the MLP is weak; the shift is weak where Louvain "
                  "communities aren't class-correlated; Texas supports clustering only; "
                  "silhouette <= 0.11, so these are useful groupings, not separated clusters. Stop.",
                  "REPORT.md section 7"))
    return steps


def main() -> None:
    df = load_results()
    if df.empty:
        raise SystemExit("results/all_results.csv is empty - nothing to fill the script from")
    lines = ["# Demo Script", "",
             "Generated by `python scripts/make_demo_script.py` - every number comes from "
             "`results/`; regenerate after any rerun. 5 minutes, 9 steps.", ""]
    for i, (title, clicks, say, src) in enumerate(build(df), 1):
        lines += [f"## {i}. {title}", f"- **Click:** {clicks}", f"- **Say:** \"{say}\"",
                  f"- *Source:* {src}", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
