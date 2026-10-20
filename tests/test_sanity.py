"""tests/test_sanity.py – P0 sanity checks.

Tests:
  - Synthetic fixture graphs validate at h=0.8 and h=0.1
  - edge_homophily is within ±0.05 of target
  - append_result / load_results round-trip
  - GuardedLabels raises LabelLeakError outside mask, succeeds inside
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import make_synthetic_graph, validate_graph
from src.labels import GuardedLabels, LabelLeakError
from src.metrics import edge_homophily
from src.run_experiments import RESULT_COLUMNS, append_result, load_results


# ── fixture graphs ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("h_target", [0.8, 0.1])
def test_synthetic_graph_validates(h_target):
    g = make_synthetic_graph(n=600, n_classes=3, n_feat=50, homophily=h_target, seed=42)
    validate_graph(g)  # must not raise


@pytest.mark.parametrize("h_target", [0.8, 0.1])
def test_synthetic_graph_homophily(h_target):
    g = make_synthetic_graph(n=600, n_classes=3, n_feat=50, homophily=h_target, seed=42)
    h_actual = edge_homophily(g.edge_index, g.y)
    assert abs(h_actual - h_target) <= 0.05, (
        f"edge_homophily {h_actual:.3f} is not within 0.05 of target {h_target}"
    )


def test_synthetic_graph_is_fixture():
    g = make_synthetic_graph()
    assert g.meta.get("is_fixture") is True


def test_synthetic_graph_float32_features():
    g = make_synthetic_graph()
    assert g.X.dtype == np.float32


def test_synthetic_graph_int64_labels():
    g = make_synthetic_graph()
    assert g.y.dtype == np.int64


# ── GuardedLabels ────────────────────────────────────────────────────────────

def test_guarded_labels_raises_outside_mask():
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    mask = np.array([True, True, False, False, False])
    gl = GuardedLabels(y, mask)
    with pytest.raises(LabelLeakError):
        _ = gl[2]


def test_guarded_labels_raises_outside_mask_array():
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    mask = np.array([True, True, False, False, False])
    gl = GuardedLabels(y, mask)
    with pytest.raises(LabelLeakError):
        _ = gl[np.array([0, 2])]  # index 2 is outside


def test_guarded_labels_succeeds_inside_mask():
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    mask = np.array([True, True, False, False, False])
    gl = GuardedLabels(y, mask)
    result = gl[np.array([0, 1])]
    np.testing.assert_array_equal(result, [0, 1])


def test_guarded_labels_all_false_raises():
    y = np.array([0, 1, 2], dtype=np.int64)
    mask = np.array([False, False, False])
    gl = GuardedLabels(y, mask)
    with pytest.raises(LabelLeakError):
        _ = gl[0]


def test_guarded_labels_no_raw_array():
    y = np.array([0, 1, 2], dtype=np.int64)
    mask = np.ones(3, dtype=bool)
    gl = GuardedLabels(y, mask)
    with pytest.raises(LabelLeakError):
        np.array(gl)


# ── append_result / load_results round-trip ──────────────────────────────────

def test_append_load_roundtrip(tmp_path):
    """append_result then load_results should return the same row."""
    # Patch the module-level paths temporarily
    import src.run_experiments as re_mod
    orig_csv = re_mod.RESULTS_CSV
    orig_dir = re_mod.RESULTS_DIR
    orig_runs = re_mod.RUNS_DIR
    try:
        re_mod.RESULTS_DIR = tmp_path
        re_mod.RESULTS_CSV = tmp_path / "test_results.csv"
        re_mod.RUNS_DIR = tmp_path / "runs"

        row = {
            "run_id": "cora__random__gcn__s0__none__main",
            "dataset": "cora",
            "split": "random",
            "model": "gcn",
            "seed": 0,
            "calib_method": "none",
            "ablation": "main",
            "acc": 0.81,
            "ece": 0.05,
        }
        append_result(row)
        df = load_results(re_mod.RESULTS_CSV)
        assert len(df) == 1
        assert df.iloc[0]["dataset"] == "cora"
        assert df.iloc[0]["acc"] == pytest.approx(0.81)
        assert df.iloc[0]["run_id"] == "cora__random__gcn__s0__none__main"
    finally:
        re_mod.RESULTS_CSV = orig_csv
        re_mod.RESULTS_DIR = orig_dir
        re_mod.RUNS_DIR = orig_runs


def test_result_columns_complete():
    """RESULT_COLUMNS must contain exactly the contract columns."""
    required = [
        "run_id", "dataset", "split", "model", "seed", "calib_method", "ablation",
        "acc", "nll", "brier", "ece", "ece_massbins", "mce", "avg_conf",
        "n_clusters", "cluster_ece_mean", "cluster_ece_worst", "cluster_ece_std", "worst_cluster_id",
        "proxy_corr", "train_homophily", "test_homophily", "homophily_gate",
        "temperature_summary", "train_sec", "timestamp",
        "labels_used", "gate_source", "h_hat_mlp", "h_hat_gnn", "git_sha",
    ]
    for col in required:
        assert col in RESULT_COLUMNS, f"Missing column: {col}"
