"""src/run_experiments.py – Experiment driver, result logging, and artifacts.

Lane: joint (P0 infrastructure, both lanes fill in the run loop in P3).
"""
from __future__ import annotations

import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data import load_graph
from src.splits import make_split
from src.train import train_model
from src.clustering import cluster_graph, cluster_profile, estimate_homophily
from src.metrics import summarise_calibration, ece_vs_T_curve, per_cluster_metrics, ece

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
# P3: logit cache helpers
# ---------------------------------------------------------------------------

_LOGIT_CACHE = ROOT / "cache" / "logits"


def _logit_cache_path(dataset: str, split_kind: str, model: str, seed: int) -> Path:
    return _LOGIT_CACHE / f"{dataset}__{split_kind}__{model}__s{seed}.npy"


def _get_logits(
    g, split: dict, dataset: str, split_kind: str, model: str, seed: int, force: bool
) -> tuple[np.ndarray, float]:
    """Return (logits, train_sec). Loads from cache unless force=True."""
    path = _logit_cache_path(dataset, split_kind, model, seed)
    if path.exists() and not force:
        return np.load(path), 0.0
    res = train_model(g, split, model, seed=seed)
    _LOGIT_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, res["logits"])
    return res["logits"], res["train_sec"]


def _softmax(logits: np.ndarray) -> np.ndarray:
    l = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(l)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def _apply_calib(
    calib: str, logits: np.ndarray, g, split: dict, cl: dict
) -> tuple[np.ndarray, str, dict]:
    """Return (probs, labels_used, temperatures).

    P3: only "none" is implemented.
    P4-A: wire in calibrate.py methods here, keyed by calib name.
    """
    if calib == "none":
        return _softmax(logits), "none", {}
    raise NotImplementedError(
        f"Calibration method {calib!r} not yet implemented (P4-A). "
        "Wire it into _apply_calib after implementing src/calibrate.py."
    )


# ---------------------------------------------------------------------------
# P3: run_one
# ---------------------------------------------------------------------------

# Datasets excluded from calibration (clustering/diagnostics only per master plan)
_CALIB_SKIP = {"texas", "cornell", "wisconsin"}


def run_one(
    dataset: str,
    split_kind: str,
    model: str,
    seed: int,
    calibs: tuple = ("none",),
    ablation: Optional[str] = None,
    force: bool = False,
) -> list[dict]:
    """Run one (dataset, split_kind, model, seed) with all calibrators.

    Steps (master plan P3):
      1. load_graph  2. make_split  3. train model + MLP for gate
      4. cluster_graph(gate_labels=MLP argmax)
      5. per calib: apply, score on target_eval, append_result + save_run_artifacts
    Returns list of result rows (one per calib).

    g.y is read in step 5 ONLY through summarise_calibration / cluster_profile
    (the designated evaluator path).  Training uses GuardedLabels internally.
    """
    if dataset.lower() in _CALIB_SKIP:
        raise ValueError(
            f"{dataset} is excluded from calibration runs (clustering/diagnostics only)."
        )

    g        = load_graph(dataset)
    split    = make_split(g, split_kind, seed=seed)
    logits, train_sec = _get_logits(g, split, dataset, split_kind, model, seed, force)

    # Gate: always MLP predictions (never g.y)
    if model == "mlp":
        mlp_logits = logits
    else:
        mlp_logits, _ = _get_logits(g, split, dataset, split_kind, "mlp", seed, force)

    gate_labels = mlp_logits.argmax(axis=1).astype(np.int64)
    h_hat_mlp   = float(estimate_homophily(g.edge_index, gate_labels))
    h_hat_gnn   = float(estimate_homophily(
        g.edge_index, logits.argmax(axis=1).astype(np.int64)
    ))

    cl   = cluster_graph(g, gate_labels, gate_source="mlp_pred", seed=seed)
    diag = split["diagnostics"]

    rows = []
    for calib in calibs:
        probs, labels_used, temperatures = _apply_calib(calib, logits, g, split, cl)

        metrics   = summarise_calibration(probs, g.y, split["target_eval"], cl["clusters"])
        curve     = ece_vs_T_curve(logits, g.y, split["target_eval"])
        pcm       = per_cluster_metrics(probs, g.y, split["target_eval"], cl["clusters"])
        _, bins_u = ece(probs, g.y, split["target_eval"], strategy="uniform")
        _, bins_q = ece(probs, g.y, split["target_eval"], strategy="quantile")
        profile   = cluster_profile(g, cl["clusters"], g.y)

        run_id = make_run_id(dataset, split_kind, model, seed, calib, ablation)
        row = {
            "run_id":      run_id,
            "dataset":     dataset,
            "split":       split_kind,
            "model":       model,
            "seed":        seed,
            "calib_method": calib,
            "ablation":    ablation or "main",
            "train_sec":   train_sec,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "labels_used": labels_used,
            "gate_source": "mlp_pred",
            "h_hat_mlp":   h_hat_mlp,
            "h_hat_gnn":   h_hat_gnn,
            "homophily_gate": float(cl["h_hat"]),
            "train_homophily": diag.get("train_homophily", float("nan")),
            "test_homophily":  diag.get("test_homophily",  float("nan")),
            "temperature_summary": json.dumps(temperatures),
            "git_sha":     git_sha(),
            # P4-A: will be set by label_free_proxy / proxy_quality
            "proxy_corr":  float("nan"),
            **metrics,
        }
        append_result(row)

        meta = {
            "split_diagnostics":   diag,
            "bins_uniform":        bins_u.to_dict("records"),
            "bins_quantile":       bins_q.to_dict("records"),
            "per_cluster_metrics": pcm.to_dict("records"),
            "cluster_profile":     profile.to_dict("records"),
            "temperatures":        temperatures,
            "ece_vs_T":  {"T": curve["T"].tolist(), "ece": curve["ece"].tolist()},
            "cluster_info": {
                "method_used": cl["method_used"],
                "h_hat":       float(cl["h_hat"]),
                "gate_source": cl["gate_source"],
            },
        }
        if dataset == "chameleon":
            meta["note"] = (
                "Chameleon random-split target ~145 eval nodes; "
                "most per-cluster ECE rows are NaN by design (< 20 eval nodes per cluster)."
            )
        save_run_artifacts(run_id, logits, probs, cl["clusters"], cl["embedding2d"], meta)
        rows.append(row)

    return rows


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
        print("Dry run - would run:")
        for ds in args.datasets:
            for sp in args.splits:
                for m in args.models:
                    for s in args.seeds:
                        for c in args.calib:
                            rid = make_run_id(ds, sp, m, s, c, args.ablation)
                            print(f"  {rid}")
    else:
        existing = load_results()
        done_ids = set(existing["run_id"].tolist()) if len(existing) else set()
        total = (len(args.datasets) * len(args.splits) * len(args.models)
                 * len(args.seeds) * len(args.calib))
        done_count = 0
        for ds in args.datasets:
            for sp in args.splits:
                for m in args.models:
                    for s in args.seeds:
                        run_calibs = [
                            c for c in args.calib
                            if args.force
                            or make_run_id(ds, sp, m, s, c, args.ablation) not in done_ids
                        ]
                        if args.resume and not run_calibs:
                            print(f"  skip (done): {make_run_id(ds, sp, m, s, '+'.join(args.calib), args.ablation)}")
                            done_count += len(args.calib)
                            continue
                        print(f"  {ds} {sp} {m} s{s} ...", flush=True)
                        try:
                            rows = run_one(
                                ds, sp, m, s,
                                calibs=tuple(run_calibs),
                                ablation=args.ablation,
                                force=args.force,
                            )
                            for row in rows:
                                done_count += 1
                                print(f"    [{done_count}/{total}] {row['run_id']}"
                                      f"  acc={row['acc']:.4f}  ece={row['ece']:.4f}"
                                      f"  h_mlp={row['h_hat_mlp']:.3f}")
                        except Exception as exc:
                            print(f"  ERROR in {ds}/{sp}/{m}/s{s}: {exc}")
                            traceback.print_exc()
        print(f"\nDone. {done_count}/{total} rows written to {RESULTS_CSV}")
