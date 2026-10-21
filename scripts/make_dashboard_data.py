"""scripts/make_dashboard_data.py - Precompute the dashboard views the run artifacts lack.

Lane B (Saanvi) owns this file. Phase P4-B. Run offline; the dashboard never runs it.

Writes results/dashboard/:
  clustering.json          method table + hop sweep per dataset (true labels: evaluation only)
  splits.json              source vs target class distributions + diagnostics per split
  conf_hist/<run_id>.json  max-softmax confidence histogram over target_eval for each CSV row
  labels/<dataset>.npy     true labels for the "colour by true class" view (evaluation only)

Runs do not save their masks, so conf_hist rebuilds each split from its seed and refuses to
write a histogram unless the rebuilt target_eval reproduces the row's accuracy exactly.

Usage: python scripts/make_dashboard_data.py [--only clustering splits conf labels] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from src.clustering import (
    _embedding_for, _kmeans, cluster_graph, cluster_metrics, louvain_clusters,
    spectral_clusters,
)
from src.data import load_graph
from src.run_experiments import RESULTS_DIR, RUNS_DIR, load_results
from src.splits import make_split

OUT = RESULTS_DIR / "dashboard"
CLUSTER_DATASETS = ["cora", "citeseer", "pubmed", "texas", "chameleon", "roman_empire"]
CALIB_DATASETS = ["cora", "citeseer", "pubmed", "chameleon", "roman_empire"]
SPECTRAL_MAX_N = 5000
N_HIST_BINS = 20


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def clustering_data() -> dict:
    out = {}
    for name in CLUSTER_DATASETS:
        t0 = time.time()
        g = load_graph(name)
        y, k = g.y, g.num_classes
        methods = {}
        for m in ("sgc_kmeans", "feature_kmeans"):
            r = cluster_graph(g, y, "true_oracle", force=m, seed=42)
            methods[m] = cluster_metrics(g, r["clusters"], r["embedding"], y)
        emb = cluster_graph(g, y, "true_oracle", force="sgc_kmeans", seed=42)["embedding"]
        methods["louvain"] = cluster_metrics(g, louvain_clusters(g, 42), emb, y)
        if len(y) <= SPECTRAL_MAX_N:
            methods["spectral"] = cluster_metrics(g, spectral_clusters(g, k, 42), emb, y)
        else:
            methods["spectral"] = {"skipped": f"N={len(y)} > {SPECTRAL_MAX_N}"}
        sweep = []
        for h in range(5):
            emb_h = _embedding_for(g, h)
            nmis = [normalized_mutual_info_score(y, _kmeans(emb_h, k, s)) for s in range(5)]
            sweep.append(dict(hops=h, nmi_mean=float(np.mean(nmis)), nmi_std=float(np.std(nmis))))
        out[name] = dict(methods=methods, hop_sweep=sweep, n_nodes=len(y),
                         h_true=g.meta["edge_homophily"])
        print(f"  clustering {name}: {time.time() - t0:.0f}s", flush=True)
    out["_note"] = ("Clustering uses the TRUE-label oracle gate and true labels for NMI/ARI: "
                    "baseline reproduction, evaluation only.")
    return out


def _class_dist(y, mask, C):
    c = np.bincount(y[mask & (y >= 0)], minlength=C).astype(float)
    return (c / max(c.sum(), 1.0)).tolist()


def split_data() -> dict:
    out = {}
    for name in CALIB_DATASETS:
        g = load_graph(name)
        for kind in ("random", "structural"):
            for seed in range(5):
                s = make_split(g, kind, seed=seed)
                target = s["target_fit"] | s["target_eval"]
                source = ~target & ~s["train"]
                out[f"{name}__{kind}__s{seed}"] = dict(
                    source_class_dist=_class_dist(g.y, source, g.num_classes),
                    target_class_dist=_class_dist(g.y, target, g.num_classes),
                    diagnostics=s["diagnostics"], class_shortfall=s["class_shortfall"])
        print(f"  splits {name}: done", flush=True)
    out["_note"] = "Class distributions use TRUE labels: evaluation only."
    return out


def conf_hists(force: bool) -> int:
    df = load_results()
    written = 0
    for (name, kind, seed), rows in df.groupby(["dataset", "split", "seed"]):
        todo = [r for _, r in rows.iterrows()
                if force or not (OUT / "conf_hist" / f"{r['run_id']}.json").exists()]
        if not todo:
            continue
        g = load_graph(name)
        # labelled target_eval nodes only, as in src.metrics (CiteSeer has y == -1 nodes)
        mask = make_split(g, kind, seed=int(seed))["target_eval"] & (g.y >= 0)
        for r in todo:
            probs = np.load(RUNS_DIR / r["run_id"] / "probs.npy")
            pred, conf = probs[mask].argmax(1), probs[mask].max(1)
            acc = float((pred == g.y[mask]).mean())
            if abs(acc - r["acc"]) > 1e-6:
                raise RuntimeError(f"{r['run_id']}: rebuilt target_eval gives acc {acc:.6f}, "
                                   f"row says {r['acc']:.6f} - split not reproducible, refusing")
            counts, edges = np.histogram(conf, bins=N_HIST_BINS, range=(0.0, 1.0))
            _dump(dict(edges=edges.tolist(), counts=counts.tolist(), mean_conf=float(conf.mean()),
                       acc=acc, n_eval=int(mask.sum())),
                  OUT / "conf_hist" / f"{r['run_id']}.json")
            written += 1
    return written


def save_labels() -> None:
    for name in CLUSTER_DATASETS:
        path = OUT / "labels" / f"{name}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, load_graph(name).y)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=["clustering", "splits", "conf", "labels"])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    if "labels" in args.only:
        save_labels()
    if "conf" in args.only:
        print(f"conf_hist: wrote {conf_hists(args.force)} files", flush=True)
    if "splits" in args.only:
        _dump(split_data(), OUT / "splits.json")
    if "clustering" in args.only:
        _dump(clustering_data(), OUT / "clustering.json")
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
