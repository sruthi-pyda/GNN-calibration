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
# P3: run_one and CLI
# ---------------------------------------------------------------------------

LOGITS_DIR = ROOT / "cache" / "logits"
FAILURES_LOG = RESULTS_DIR / "failures.log"
CLUSTERING_ONLY = ("texas", "cornell", "wisconsin")


def _softmax(logits: np.ndarray, T=1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _calib_none(logits, labels, ctx):
    return _softmax(logits), {"global": 1.0}, {}


# key -> (LABELS declaration, fn(logits, GuardedLabels, ctx) -> (probs, temps, extras))
# extras may contain: a_hat, per_cluster_calib, ridge_r2 (for LF methods)

def _calib_ts_val(logits, labels, ctx):
    from src.calibrate import ts_val
    p, t = ts_val(logits, labels, ctx["split"])
    return p, t, {}

def _calib_ts_target_oracle(logits, labels, ctx):
    from src.calibrate import ts_target_oracle
    p, t = ts_target_oracle(logits, labels, ctx["split"])
    return p, t, {}

def _calib_cluster_ts_oracle(logits, labels, ctx):
    from src.calibrate import cluster_ts_target_oracle
    p, t, pc = cluster_ts_target_oracle(logits, labels, ctx["split"], ctx["clusters"])
    return p, t, {"per_cluster_calib": pc}

def _calib_lf_ts(logits, labels, ctx):
    from src.calibrate import label_free_proxy, lf_ts
    cl = ctx["cl"]
    a = label_free_proxy(ctx["g"], logits, ctx["clusters"], cl["h_hat"], cl["embedding"])
    tmask = ctx["split"]["target_fit"] | ctx["split"]["target_eval"]
    p, t = lf_ts(logits, a, ctx["clusters"], tmask)
    return p, t, {"a_hat": a}

def _calib_lf_cts(logits, labels, ctx):
    from src.calibrate import label_free_proxy, lf_cts
    cl = ctx["cl"]
    a = label_free_proxy(ctx["g"], logits, ctx["clusters"], cl["h_hat"], cl["embedding"])
    tmask = ctx["split"]["target_fit"] | ctx["split"]["target_eval"]
    p, t, pc = lf_cts(logits, a, ctx["clusters"], tmask)
    return p, t, {"a_hat": a, "per_cluster_calib": pc}

def _calib_lf_cts_plus(logits, labels, ctx):
    from src.calibrate import lf_cts_plus
    cl = ctx["cl"]
    p, t, pc, r2 = lf_cts_plus(logits, labels, ctx["split"], ctx["clusters"],
                                 cl["h_hat"], cl["embedding"], ctx["g"])
    return p, t, {"a_hat": np.array([r["a_hat_raw"] for r in pc]),
                  "per_cluster_calib": pc, "ridge_r2": r2}

CALIBRATORS = {
    "none":                      ("none",        _calib_none),
    "ts_val":                    ("source_val",  _calib_ts_val),
    "ts_target_oracle":          ("target_fit",  _calib_ts_target_oracle),
    "cluster_ts_target_oracle":  ("target_fit",  _calib_cluster_ts_oracle),
    "lf_ts":                     ("none",        _calib_lf_ts),
    "lf_cts":                    ("none",        _calib_lf_cts),
    "lf_cts_plus":               ("source_val",  _calib_lf_cts_plus),
}


def _label_mask(split: dict, declared: str) -> np.ndarray:
    if declared == "none":
        return np.zeros_like(split["train"])
    return split["val"] if declared == "source_val" else split["target_fit"]


def _trained_logits(g, split, name, key, seed, force):
    """Train once per (dataset, split, model, seed); cache logits for later calibrators."""
    from src.train import train_model
    path = LOGITS_DIR / f"{key}__{name}.npz"
    if path.exists() and not force:
        d = np.load(path)
        return d["logits"], dict(best_epoch=int(d["best_epoch"]), train_sec=float(d["train_sec"]))
    r = train_model(g, split, name, seed=seed)
    LOGITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, logits=r["logits"], best_epoch=r["best_epoch"],
                        train_sec=r["train_sec"])
    return r["logits"], dict(best_epoch=int(r["best_epoch"]), train_sec=float(r["train_sec"]))


def run_one(
    dataset: str,
    split_kind: str,
    model: str,
    seed: int,
    calibs: tuple = ("none",),
    ablation: Optional[str] = None,
    force: bool = False,
) -> list[dict]:
    """Run one (dataset, split, model, seed) configuration with all calibrators.

    Returns a list of result rows (one per calibrator). Target labels are read only by the
    evaluator, on split["target_eval"]; calibrators get GuardedLabels from their declaration.
    """
    from src.clustering import cluster_graph, cluster_profile, estimate_homophily
    from src.data import load_graph
    from src.labels import GuardedLabels, audit_label_use
    from src.metrics import ece, ece_vs_T_curve, per_cluster_metrics, summarise_calibration
    from src.splits import make_split

    if dataset.lower() in CLUSTERING_ONLY:
        raise ValueError(f"{dataset} is used for the clustering comparison only")
    g = load_graph(dataset)
    split = make_split(g, split_kind, seed=seed)
    key = f"{dataset}__{split_kind}__s{seed}"
    mlp_logits, mlp_info = _trained_logits(g, split, "mlp", key, seed, force)
    if model == "mlp":
        logits, info = mlp_logits, mlp_info
    else:
        logits, info = _trained_logits(g, split, model, key, seed, force)

    mlp_pred, gnn_pred = mlp_logits.argmax(1), logits.argmax(1)
    h_hat_mlp = estimate_homophily(g.edge_index, mlp_pred)
    h_hat_gnn = estimate_homophily(g.edge_index, gnn_pred)
    cl = cluster_graph(g, mlp_pred, gate_source="mlp_pred", seed=seed)
    clusters, eval_mask = cl["clusters"], split["target_eval"]
    diag = split["diagnostics"]

    rows = []
    for calib in calibs:
        if calib not in CALIBRATORS:
            raise NotImplementedError(f"calibrator {calib!r} is implemented in phase P4-A")
        declared, fit = CALIBRATORS[calib]
        audit_label_use(calib, declared)
        probs, temps, extras = fit(
            logits, GuardedLabels(g.y, _label_mask(split, declared)),
            dict(g=g, split=split, clusters=clusters, cl=cl, mlp_logits=mlp_logits),
        )
        # ---- evaluator: the only place target labels are read ----
        metrics = summarise_calibration(probs, g.y, eval_mask, clusters)
        proxy_corr = float("nan")
        pq_extras: dict = {}
        if "a_hat" in extras:
            from src.calibrate import proxy_quality
            pq_df, pq_r = proxy_quality(probs, extras["a_hat"], clusters, g.y, eval_mask)
            proxy_corr = pq_r
            pq_extras = {"proxy_quality": {"table": pq_df.to_dict("records"), "r": round(float(pq_r), 4)}}
        run_id = make_run_id(dataset, split_kind, model, seed, calib, ablation)
        row = dict(
            run_id=run_id, dataset=dataset, split=split_kind, model=model, seed=seed,
            calib_method=calib, ablation=ablation or "main", **metrics,
            proxy_corr=proxy_corr, train_homophily=diag["train_homophily"], test_homophily=diag["test_homophily"],
            homophily_gate=cl["method_used"],
            temperature_summary=json.dumps({k: round(float(v), 4) for k, v in temps.items()
                                            if not k.startswith("_")}),
            train_sec=info["train_sec"], timestamp=datetime.now(timezone.utc).isoformat(),
            labels_used=declared, gate_source="mlp_pred", h_hat_mlp=h_hat_mlp,
            h_hat_gnn=h_hat_gnn, git_sha=git_sha(),
        )
        _, bins_u = ece(probs, g.y, eval_mask, strategy="uniform")
        _, bins_q = ece(probs, g.y, eval_mask, strategy="quantile")
        meta = dict(
            row=row, bins_uniform=bins_u.to_dict("records"), bins_mass=bins_q.to_dict("records"),
            per_cluster=per_cluster_metrics(probs, g.y, eval_mask, clusters).to_dict("records"),
            cluster_profile=cluster_profile(g, clusters, g.y).to_dict("records"),
            split_diagnostics=diag, class_shortfall=split["class_shortfall"],
            temperatures=temps, ece_vs_T=ece_vs_T_curve(logits, g.y, eval_mask),
            best_epoch=dict(mlp=mlp_info["best_epoch"], gnn=info["best_epoch"]),
            mlp_target_eval_acc=summarise_calibration(_softmax(mlp_logits), g.y, eval_mask,
                                                      clusters)["acc"],
            h_true_eval_only=g.meta["edge_homophily"], cluster_method=cl["method_used"],
            n_eval=int(eval_mask.sum()),
        )
        meta.update(pq_extras)
        if "per_cluster_calib" in extras:
            meta["per_cluster_calib"] = extras["per_cluster_calib"]
        if "ridge_r2" in extras:
            meta["ridge_r2"] = extras["ridge_r2"]
        append_result(row)
        save_run_artifacts(run_id, logits, probs, clusters, cl["embedding2d"], meta)
        rows.append(row)
    return rows


def _drop_rows(run_ids: set) -> None:
    df = load_results()
    if len(df):
        df[~df["run_id"].isin(run_ids)].to_csv(RESULTS_CSV, index=False)


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
    parser.add_argument("--resume", action="store_true",
                        help="skip run_ids already in all_results.csv (the default)")
    parser.add_argument("--force", action="store_true",
                        help="drop existing rows for these run_ids and retrain")
    args = parser.parse_args()

    configs = [(ds, sp, m, s) for ds in args.datasets for sp in args.splits
               for m in args.models for s in args.seeds]
    if args.dry_run:
        print("Dry run - would run:")
        for ds, sp, m, s in configs:
            print(f"  {make_run_id(ds, sp, m, s, '+'.join(args.calib), args.ablation)}")
        raise SystemExit(0)

    done = set(load_results()["run_id"])
    for i, (ds, sp, m, s) in enumerate(configs, 1):
        ids = {make_run_id(ds, sp, m, s, c, args.ablation) for c in args.calib}
        tag = f"[{i}/{len(configs)}] {ds} {sp} {m} s{s}"
        if args.force:
            _drop_rows(ids)
        elif ids <= done:
            print(f"{tag}: already done, skipping", flush=True)
            continue
        try:
            t0 = datetime.now()
            for r in run_one(ds, sp, m, s, tuple(args.calib), args.ablation, force=args.force):
                print(f"{tag} {r['calib_method']}: acc {r['acc']:.3f} ece {r['ece']:.3f} "
                      f"worst {r['cluster_ece_worst']:.3f} ({(datetime.now() - t0).seconds}s)",
                      flush=True)
        except Exception:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(FAILURES_LOG, "a") as f:
                f.write(f"{tag}\n{traceback.format_exc()}\n")
            print(f"{tag}: FAILED (see results/failures.log)", flush=True)
