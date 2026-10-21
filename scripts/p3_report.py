"""scripts/p3_report.py - Answer the P3 go/no-go questions (a)-(e) from the results files.

Reads only results/all_results.csv and results/runs/<run_id>/meta.json (never re-runs anything).
Prints the answers and writes results/P3_SUMMARY.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.run_experiments import RESULTS_DIR, RUNS_DIR, load_results


def _meta(run_id: str) -> dict:
    return json.loads((RUNS_DIR / run_id / "meta.json").read_text())


def _ms(s: pd.Series) -> str:
    return f"{s.mean():.3f} ± {s.std(ddof=0):.3f}"


def _section(title: str, table: pd.DataFrame, index: bool = False) -> str:
    return f"## {title}\n\n```\n{table.to_string(index=index)}\n```\n"


def _branch(h: float) -> str:
    return "sgc" if h > 0.5 else "feature"


def q_a(df):
    rows = []
    for ds, d in df.groupby("dataset"):
        r, s = d[d["split"] == "random"]["ece"], d[d["split"] == "structural"]["ece"]
        gap, noise = s.mean() - r.mean(), max(r.std(ddof=0), s.std(ddof=0))
        rows.append(dict(dataset=ds, seeds=f"{len(r)}/{len(s)}", ece_random=_ms(r),
                         ece_structural=_ms(s), gap=round(gap, 3), max_seed_std=round(noise, 3),
                         beyond_std="yes" if gap > noise else "no"))
    return _section("(a) ECE structural vs random", pd.DataFrame(rows))


def q_b(df):
    d = df.assign(ratio=df["cluster_ece_worst"] / df["ece"])
    b = d.groupby(["dataset", "split"])["ratio"].agg(["mean", "min", "max"]).round(2)
    return _section("(b) cluster_ece_worst / ece", b, index=True)


def q_c(df):
    rows = []
    for _, r in df[(df["dataset"] == "cora") & (df["split"] == "structural")].iterrows():
        m = _meta(r["run_id"])
        prof = pd.DataFrame(m["cluster_profile"]).set_index("cluster")
        pc = pd.DataFrame(m["per_cluster"]).set_index("cluster_id")
        w = int(r["worst_cluster_id"])
        p = prof.loc[w]
        rows.append(dict(seed=r["seed"], method=m["cluster_method"], worst=w,
                         worst_ece=round(r["cluster_ece_worst"], 3), overall=round(r["ece"], 3),
                         n_eval=int(pc.loc[w, "n_eval"]), size=int(p["size"]),
                         purity=round(p["purity"], 2), conductance=round(p["conductance"], 2),
                         avg_degree=round(p["avg_degree"], 2),
                         inside_h=round(p["inside_homophily"], 2),
                         lowest_purity=int(prof["purity"].idxmin()),
                         lowest_degree=int(prof["avg_degree"].idxmin())))
    return _section("(c) Cora structural: worst cluster "
                    "(profile uses true labels - evaluation only)", pd.DataFrame(rows))


def q_d(df):
    a = df.pivot_table(index="dataset", columns="split", values="acc", aggfunc="mean")
    a["drop_points"] = ((a["random"] - a["structural"]) * 100).round(1)
    a["collapse_gt_15"] = a["drop_points"] > 15
    return _section("(d) accuracy random vs structural (mean over seeds)", a.round(3), index=True)


def q_e(df):
    rows = []
    for ds, d in df.groupby("dataset"):
        h_true = _meta(d["run_id"].iloc[0])["h_true_eval_only"]
        mlp_acc = pd.Series([_meta(x)["mlp_target_eval_acc"] for x in d["run_id"]])
        rows.append(dict(
            dataset=ds, h_true=round(h_true, 3), h_hat_mlp=_ms(d["h_hat_mlp"]),
            h_hat_gnn=_ms(d["h_hat_gnn"]), mlp_acc=_ms(mlp_acc),
            gate_mlp=dict(d["homophily_gate"].value_counts()), gate_true=_branch(h_true),
            gate_gnn=dict(d["h_hat_gnn"].map(_branch).value_counts())))
    return _section("(e) homophily estimates and gate decisions (h_true is evaluation only)",
                    pd.DataFrame(rows))


def main() -> None:
    df = load_results()
    df = df[(df["calib_method"] == "none") & (df["ablation"].fillna("main") == "main")]
    head = (f"# P3 summary - uncalibrated GCN ({len(df)} rows)\n\n"
            "Every number is computed from `results/all_results.csv` rows (calib_method=none, "
            "ablation=main) and their `results/runs/<run_id>/meta.json`. Metrics are on "
            "target_eval only. Mean ± std over seeds (population std).\n")
    text = "\n".join([head, q_a(df), q_b(df), q_c(df), q_d(df), q_e(df)])
    print(text)
    (RESULTS_DIR / "P3_SUMMARY.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
