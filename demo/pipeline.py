"""demo/pipeline.py - The six RUN DEMO stages as plain functions (no Streamlit).

Lane B (Saanvi). Live research demo.

Nothing is trained here. Stage 1 loads the graph, rebuilds the structural split and
reads the saved GNN logits of a P5 run (results/runs/<run_id>/). Stages 2-5 rerun
the LF-CTS steps with the project's own functions (src/clustering.py,
src/calibrate.py) on a LabelFreeGraph that has no y attribute at all, so they
cannot read target labels. True labels are read in stage 6 only, on the
target_eval mask, to score the result - exactly as in run_experiments.
Every live value is checked against the saved run it came from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibrate import _bisect_T, apply_temperature, label_free_proxy
from src.clustering import _embedding_for, _kmeans, louvain_clusters
from src.data import load_graph
from src.metrics import ece
from src.splits import make_split

ROOT = Path(__file__).parent.parent
STAGES = ["Load graph", "Create clusters", "Estimate cluster accuracy",
          "Calculate temperatures", "Calibrate predictions", "Display final results"]
DATASETS = {"Cora": "cora", "CiteSeer": "citeseer", "Chameleon": "chameleon"}
MODELS = {"SGC": "sgc", "GCN": "gcn"}
METHODS = {"none": "Uncalibrated", "ts_val": "Temperature scaling (one global T)",
           "lf_cts": "LF-CTS (one T per cluster)", "lf_cts_plus": "LF-CTS+"}
GATE_THRESHOLD = 0.5


@dataclass(frozen=True)
class LabelFreeGraph:
    """What the calibration stages may see: features and edges, no labels."""
    X: np.ndarray
    edge_index: np.ndarray
    num_classes: int


def results_dir() -> Path:
    from dashboard.data import results_dir as _rd
    return _rd()


def run_id(ds: str, model: str, seed: int, calib: str) -> str:
    return f"{ds}__structural__{model}__s{seed}__{calib}__main"


def run_path(ds, model, seed, calib) -> Path:
    return results_dir() / "runs" / run_id(ds, model, seed, calib)


def results_rows() -> pd.DataFrame:
    return pd.read_csv(results_dir() / "all_results.csv")


def available(ds: str, model: str, seed: int) -> bool:
    return all((run_path(ds, model, seed, c) / "logits.npy").exists() for c in METHODS)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_load(ds: str, model: str, seed: int) -> dict:
    """1. Graph, structural split (Louvain regions) and the saved GNN logits."""
    g = load_graph(ds)
    split = make_split(g, "structural", seed=seed)
    rows = results_rows()
    rows = rows[(rows.dataset == ds) & (rows.model == model) & (rows.split == "structural")
                & (rows.ablation == "main") & rows.calib_method.isin(list(METHODS))]
    base = run_path(ds, model, seed, "lf_cts")
    region = np.full(len(g.y), "Source region (unlabelled)", dtype=object)
    region[split["train"] | split["val"]] = "Source labels (train + val)"
    region[split["target_fit"] | split["target_eval"]] = "Target region"
    return dict(
        ds=ds, model=model, seed=seed, g=g, split=split,
        lf=LabelFreeGraph(g.X, g.edge_index, g.num_classes),
        communities=louvain_clusters(g, seed=seed), region=region,
        target=split["target_fit"] | split["target_eval"],
        logits=np.load(base / "logits.npy"),
        meta=json.loads((base / "meta.json").read_text(encoding="utf-8")),
        rows=rows, row=rows[rows.run_id == run_id(ds, model, seed, "lf_cts")].iloc[0],
        diagnostics=split.get("diagnostics", {}),
    )


def stage_cluster(S: dict) -> dict:
    """2. Homophily gate on MLP-predicted labels, then embed and k-means (k = C)."""
    h_hat = float(S["row"].h_hat_mlp)
    method = "sgc_kmeans" if h_hat > GATE_THRESHOLD else "feature_kmeans"
    emb = _embedding_for(S["lf"], 2 if method == "sgc_kmeans" else 0)
    clusters = _kmeans(emb, S["lf"].num_classes, S["seed"])
    saved = np.load(run_path(S["ds"], S["model"], S["seed"], "lf_cts") / "clusters.npy")
    return dict(h_hat=h_hat, gate=method, embedding=emb, clusters=clusters,
                K=int(clusters.max()) + 1, cluster_match=float((clusters == saved).mean()))


def stage_accuracy(S: dict) -> dict:
    """3. Label-free accuracy per cluster from neighbour agreement of predictions."""
    a_hat = label_free_proxy(S["lf"], S["logits"], S["clusters"], S["h_hat"], S["embedding"])
    saved = np.array([c["a_hat"] for c in S["meta"]["per_cluster_calib"]])
    proxy = "edge agreement" if S["h_hat"] > GATE_THRESHOLD else "k-NN agreement in the embedding"
    return dict(a_hat=a_hat, a_hat_saved=saved, proxy=proxy)


def stage_temperatures(S: dict) -> dict:
    """4. One T per cluster, bisected so mean confidence on target nodes = a_hat."""
    T = np.ones(S["K"])
    clamped = np.zeros(S["K"], dtype=bool)
    for c in range(S["K"]):
        sel = S["target"] & (S["clusters"] == c)
        if sel.any():
            T[c], clamped[c] = _bisect_T(S["logits"][sel], float(S["a_hat"][c]))
    ts_row = S["rows"][S["rows"].run_id == run_id(S["ds"], S["model"], S["seed"], "ts_val")]
    T_global = float(json.loads(ts_row.iloc[0].temperature_summary)["global"])
    saved = np.array([c["T"] for c in S["meta"]["per_cluster_calib"]])
    return dict(T=T, clamped=clamped, T_saved=saved, T_global=T_global)


def stage_calibrate(S: dict) -> dict:
    """5. Apply the temperatures. LF-CTS+ needs a ridge fit, so its probs are loaded."""
    probs = {
        "none": apply_temperature(S["logits"], 1.0),
        "ts_val": apply_temperature(S["logits"], S["T_global"]),
        "lf_cts": apply_temperature(S["logits"], S["T"][S["clusters"]]),
        "lf_cts_plus": np.load(run_path(S["ds"], S["model"], S["seed"], "lf_cts_plus")
                               / "probs.npy"),
    }
    saved = np.load(run_path(S["ds"], S["model"], S["seed"], "lf_cts") / "probs.npy")
    conf = {m: p.max(axis=1) for m, p in probs.items()}
    per_cluster = pd.DataFrame([
        dict(cluster=c, n_target=int((S["target"] & (S["clusters"] == c)).sum()),
             a_hat=float(S["a_hat"][c]), T=float(S["T"][c]),
             conf_before=float(conf["none"][S["target"] & (S["clusters"] == c)].mean()),
             conf_global=float(conf["ts_val"][S["target"] & (S["clusters"] == c)].mean()),
             conf_after=float(conf["lf_cts"][S["target"] & (S["clusters"] == c)].mean()))
        for c in range(S["K"]) if (S["target"] & (S["clusters"] == c)).any()])
    return dict(probs=probs, per_cluster=per_cluster,
                probs_diff=float(np.abs(probs["lf_cts"] - saved).max()))


def stage_results(S: dict) -> dict:
    """6. Score on target_eval with true labels (evaluation only) and compare to the CSV."""
    y = S["g"].y
    ev = S["split"]["target_eval"] & (y >= 0)
    live, bins = {}, {}
    for m, p in S["probs"].items():
        live[m], bins[m] = ece(p, y, ev)
    rows = S["rows"]
    seed_rows = rows[rows.seed == S["seed"]].set_index("calib_method")
    summary = rows.groupby("calib_method").ece.agg(["mean", "std", "count"])
    true_acc = [float((S["probs"]["none"].argmax(1) == y)[ev & (S["clusters"] == c)].mean())
                if (ev & (S["clusters"] == c)).sum() else np.nan for c in range(S["K"])]
    return dict(ece_live=live, bins=bins, ece_saved=seed_rows.ece.to_dict(),
                labels_used=seed_rows.labels_used.to_dict(),
                ece_summary=summary.reindex(list(METHODS)), true_acc_eval=true_acc,
                acc=float((S["probs"]["none"].argmax(1) == y)[ev].mean()),
                n_eval=int(ev.sum()))


STAGE_FUNCS = [None, stage_cluster, stage_accuracy, stage_temperatures,
               stage_calibrate, stage_results]


def run_stage(i: int, S: dict | None, ds: str, model: str, seed: int) -> dict:
    """Run stage i (0-based) and return the state with its outputs merged in."""
    if i == 0:
        return stage_load(ds, model, seed)
    return {**S, **STAGE_FUNCS[i](S)}


def run_all(ds: str, model: str, seed: int) -> dict:
    S = None
    for i in range(len(STAGES)):
        S = run_stage(i, S, ds, model, seed)
    return S
