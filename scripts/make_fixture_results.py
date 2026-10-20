"""scripts/make_fixture_results.py – Write ~60 synthetic/fake rows to
results/fixture_results.csv for dashboard development.

IMPORTANT: This script writes ONLY to results/fixture_results.csv.
           It NEVER touches results/all_results.csv.
           All values are obviously synthetic (see column 'is_fixture').
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.run_experiments import RESULT_COLUMNS

# Fixture note that will appear in every row
FIXTURE_NOTE = "FIXTURE-DATA-NOT-REAL-RESULTS"

DATASETS = ["cora", "citeseer", "pubmed", "chameleon", "roman_empire"]
SPLITS = ["random", "structural"]
MODELS = ["gcn"]
SEEDS = list(range(5))
CALIBS = [
    "none", "ts_val", "lf_ts", "lf_cts", "lf_cts_plus",
    "ts_target_oracle", "cluster_ts_target_oracle",
]

rng = np.random.default_rng(0)


def _fake_row(dataset, split, model, seed, calib):
    run_id = f"{dataset}__{split}__{model}__s{seed}__{calib}__main"

    # Fake ECE: structural > random, cluster-wise > overall
    base_ece = rng.uniform(0.05, 0.20)
    if split == "structural":
        base_ece *= rng.uniform(1.3, 2.0)
    if calib in ("ts_val", "lf_cts", "lf_cts_plus"):
        base_ece *= rng.uniform(0.4, 0.8)
    if calib in ("ts_target_oracle", "cluster_ts_target_oracle"):
        base_ece *= rng.uniform(0.2, 0.5)

    cluster_ece_mean = base_ece * rng.uniform(0.8, 1.2)
    cluster_ece_worst = cluster_ece_mean * rng.uniform(1.5, 3.5)

    h_target = 0.05 if dataset in ("chameleon", "roman_empire") else 0.80
    h_hat_mlp = h_target + rng.uniform(-0.05, 0.05)
    h_hat_gnn = h_target + rng.uniform(-0.10, 0.10)

    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split,
        "model": model,
        "seed": seed,
        "calib_method": calib,
        "ablation": "main",
        "acc": float(rng.uniform(0.55, 0.85)),
        "nll": float(rng.uniform(0.5, 2.0)),
        "brier": float(rng.uniform(0.1, 0.4)),
        "ece": float(base_ece),
        "ece_massbins": float(base_ece * rng.uniform(0.9, 1.1)),
        "mce": float(base_ece * rng.uniform(1.5, 2.5)),
        "avg_conf": float(rng.uniform(0.5, 0.9)),
        "n_clusters": int(rng.integers(3, 8)),
        "cluster_ece_mean": float(cluster_ece_mean),
        "cluster_ece_worst": float(cluster_ece_worst),
        "cluster_ece_std": float(cluster_ece_mean * rng.uniform(0.2, 0.5)),
        "worst_cluster_id": int(rng.integers(0, 7)),
        "proxy_corr": float(rng.uniform(0.3, 0.85)) if "lf" in calib else float("nan"),
        "train_homophily": float(h_target + rng.uniform(-0.02, 0.02)),
        "test_homophily": float(h_target + rng.uniform(-0.05, 0.05)),
        "homophily_gate": "sgc_kmeans" if h_hat_mlp > 0.5 else "feature_kmeans",
        "temperature_summary": str(round(float(rng.uniform(0.5, 2.5)), 3)),
        "train_sec": float(rng.uniform(1.0, 60.0)),
        "timestamp": "2026-10-20T00:00:00Z",
        "labels_used": "none" if "lf" in calib or calib == "none" else "source_val",
        "gate_source": "mlp_pred",
        "h_hat_mlp": float(h_hat_mlp),
        "h_hat_gnn": float(h_hat_gnn),
        "git_sha": "fixture",
    }


def main():
    rows = []
    # ~60 rows: 5 datasets x 2 splits x 1 model x 1 seed x limited calibs
    for dataset in DATASETS:
        for split in SPLITS:
            for calib in ["none", "ts_val", "lf_cts", "lf_cts_plus", "ts_target_oracle",
                          "cluster_ts_target_oracle"]:
                rows.append(_fake_row(dataset, split, "gcn", 0, calib))

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    # Add a fixture flag column (not in RESULT_COLUMNS – dashboard can detect it)
    df["is_fixture"] = True
    df["fixture_note"] = FIXTURE_NOTE

    out_path = ROOT / "results" / "fixture_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} fixture rows to {out_path}")
    print(f"NOTE: {FIXTURE_NOTE}")


if __name__ == "__main__":
    main()
