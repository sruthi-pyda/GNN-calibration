"""src/run_experiments.py – Experiment driver, result logging, and artifacts.

Lane: joint (P0 infrastructure, both lanes fill in the run loop in P3).
"""
from __future__ import annotations

import json
import os
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
RESULTS_CSV = RESULTS_DIR / "all_results.csv"

# ---------------------------------------------------------------------------
# Frozen contract: CSV columns (do not change without joint PR)
# ---------------------------------------------------------------------------

RESULT_COLUMNS = [
    "run_id",
    "dataset",
    "split",
    "model",
    "seed",
    "calib_method",
    "ablation",
    # --- scalar metrics ---
    "acc",
    "nll",
    "brier",
    "ece",
    "ece_massbins",
    "mce",
    "avg_conf",
    # --- cluster metrics ---
    "n_clusters",
    "cluster_ece_mean",
    "cluster_ece_worst",
    "cluster_ece_std",
    "worst_cluster_id",
    # --- method-specific ---
    "proxy_corr",
    "train_homophily",
    "test_homophily",
    "homophily_gate",
    "temperature_summary",
    "train_sec",
    "timestamp",
    # --- label-use audit ---
    "labels_used",
    "gate_source",
    "h_hat_mlp",
    "h_hat_gnn",
    "git_sha",
]


def make_run_id(
    dataset: str,
    split: str,
    model: str,
    seed: int,
    calib: str,
    ablation: Optional[str] = None,
) -> str:
    abl = ablation or "main"
    return f"{dataset}__{split}__{model}__s{seed}__{calib}__{abl}"


def git_sha() -> str:
    """Return the short git SHA of HEAD, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def append_result(row: dict) -> None:
    """Validate keys, fill missing with NaN, append to results/all_results.csv.

    Does NOT write fixture data; use scripts/make_fixture_results.py for that.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Fill missing columns with NaN
    full_row = {col: row.get(col, float("nan")) for col in RESULT_COLUMNS}

    # Validate required keys
    for required in ("run_id", "dataset", "split", "model", "seed", "calib_method"):
        if full_row[required] != full_row[required]:  # NaN check
            raise ValueError(f"Required column {required!r} is missing from result row")

    df = pd.DataFrame([full_row], columns=RESULT_COLUMNS)

    if RESULTS_CSV.exists():
        df.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(RESULTS_CSV, mode="w", header=True, index=False)


def save_run_artifacts(
    run_id: str,
    logits: np.ndarray,
    probs: np.ndarray,
    clusters: np.ndarray,
    embedding2d: np.ndarray,
    meta: dict,
) -> Path:
    """Save per-run artifacts to results/runs/<run_id>/."""
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "logits.npy", logits)
    np.save(run_dir / "probs.npy", probs)
    np.save(run_dir / "clusters.npy", clusters)
    np.save(run_dir / "embedding2d.npy", embedding2d)

    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)

    return run_dir


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_results(path: Optional[Path] = None) -> pd.DataFrame:
    """Load all_results.csv into a DataFrame."""
    csv_path = path or RESULTS_CSV
    if not csv_path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# P3: run_one and CLI (stubs until P3)
# ---------------------------------------------------------------------------

def run_one(
    dataset: str,
    split_kind: str,
    model: str,
    seed: int,
    calibs: tuple = ("none",),
    ablation: Optional[str] = None,
) -> list[dict]:
    """Run one (dataset, split, model, seed) configuration with all calibrators.

    Returns a list of result rows (one per calibrator).
    Implemented in Phase P3 jointly.
    """
    raise NotImplementedError("Implemented in Phase P3")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GNN Calibration experiment runner")
    parser.add_argument("--datasets", nargs="+", default=["cora"])
    parser.add_argument("--splits", nargs="+", default=["random"])
    parser.add_argument("--models", nargs="+", default=["gcn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--calib", nargs="+", default=["none"])
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run – would run:")
        for ds in args.datasets:
            for sp in args.splits:
                for m in args.models:
                    for s in args.seeds:
                        rid = make_run_id(ds, sp, m, s, "+".join(args.calib), args.ablation)
                        print(f"  {rid}")
    else:
        raise NotImplementedError("Full runner implemented in P3")
