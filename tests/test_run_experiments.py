"""tests/test_run_experiments.py - run_one integration on the fixture (Phase P3).

Checks:
  - one row per calibrator with every RESULT_COLUMNS key; label-use audit columns set
  - run artifacts + meta.json written
  - metrics read target labels on target_eval only (poisoning every other label changes nothing)
  - "none" runs under an all-False GuardedLabels
  - clustering-only datasets raise
  - cached logits are reused; _drop_rows removes exactly the given run_ids
"""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

import src.data
import src.splits
from src import run_experiments as rx
from src.data import make_synthetic_graph
from src.labels import GuardedLabels

METRIC_KEYS = ["acc", "nll", "brier", "ece", "ece_massbins", "mce", "avg_conf",
               "cluster_ece_mean", "cluster_ece_worst", "h_hat_mlp", "h_hat_gnn"]


@pytest.fixture(scope="module")
def graph():
    return make_synthetic_graph(n=1500, n_classes=3, homophily=0.8, seed=0)


@pytest.fixture
def sandbox(tmp_path, monkeypatch, graph):
    """Point every results/cache path at tmp_path and serve a fresh copy of the fixture."""
    monkeypatch.setattr(rx, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(rx, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(rx, "RESULTS_CSV", tmp_path / "all_results.csv")
    monkeypatch.setattr(rx, "LOGITS_DIR", tmp_path / "logits")
    monkeypatch.setattr(src.data, "load_graph", lambda name: copy.deepcopy(graph))
    return tmp_path


def test_row_has_every_column_and_artifacts(sandbox):
    rows = rx.run_one("fixture", "random", "gcn", seed=0)
    assert len(rows) == 1
    row = rows[0]
    assert set(rx.RESULT_COLUMNS) <= set(row)
    assert row["labels_used"] == "none" and row["gate_source"] == "mlp_pred"
    assert row["run_id"] == "fixture__random__gcn__s0__none__main"
    df = rx.load_results()
    assert list(df.columns) == rx.RESULT_COLUMNS and len(df) == 1
    run_dir = sandbox / "runs" / row["run_id"]
    for f in ("logits.npy", "probs.npy", "clusters.npy", "embedding2d.npy", "meta.json"):
        assert (run_dir / f).exists()
    meta = json.loads((run_dir / "meta.json").read_text())
    for k in ("bins_uniform", "bins_mass", "per_cluster", "cluster_profile",
              "split_diagnostics", "ece_vs_T", "temperatures"):
        assert k in meta
    assert len(meta["ece_vs_T"]["T"]) == 50


def test_metrics_read_target_eval_labels_only(sandbox, monkeypatch, tmp_path):
    clean = rx.run_one("fixture", "random", "gcn", seed=1)[0]

    real_make_split = src.splits.make_split

    def poisoned_split(g, kind, seed=42, **kw):
        s = real_make_split(g, kind, seed=seed, **kw)
        outside = ~(s["target_eval"] | s["train"] | s["val"])
        g.y[outside] = (g.y[outside] + 1) % g.num_classes  # in place, after the split exists
        return s

    monkeypatch.setattr(src.splits, "make_split", poisoned_split)
    monkeypatch.setattr(rx, "RESULTS_CSV", tmp_path / "poisoned.csv")
    monkeypatch.setattr(rx, "LOGITS_DIR", tmp_path / "logits_poisoned")
    poisoned = rx.run_one("fixture", "random", "gcn", seed=1)[0]
    for k in METRIC_KEYS:
        assert poisoned[k] == pytest.approx(clean[k]), k


def test_none_calibrator_needs_no_labels(graph):
    declared, fit = rx.CALIBRATORS["none"]
    assert declared == "none"
    logits = np.random.default_rng(0).normal(size=(len(graph.y), graph.num_classes))
    probs, temps = fit(logits, GuardedLabels(graph.y, np.zeros(len(graph.y), bool)), {})
    np.testing.assert_allclose(probs.sum(1), 1.0)


def test_clustering_only_dataset_raises():
    with pytest.raises(ValueError):
        rx.run_one("texas", "random", "gcn", seed=0)


def test_logits_cached_and_drop_rows(sandbox):
    a = rx.run_one("fixture", "random", "gcn", seed=2)[0]
    assert len(list((sandbox / "logits").glob("*.npz"))) == 2  # mlp + gcn
    b = rx.run_one("fixture", "random", "gcn", seed=2)[0]       # second run hits the cache
    assert a["acc"] == b["acc"] and a["train_sec"] == b["train_sec"]
    rx._drop_rows({a["run_id"]})
    assert len(rx.load_results()) == 0
