"""tests/test_models_train_metrics.py – P2-A gate tests.

Gate checks:
  1. Hand-built 10-sample ECE with known answer (uniform bins)
  2. Perfectly calibrated synthetic probs → ECE < 0.02
  3. Bins returned as DataFrame, not plots
  4. GCN on Cora target_eval accuracy > 0.78
  5. Training reads labels ONLY through GuardedLabels
     (poisoned target labels must not change logits)
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    accuracy, avg_confidence, brier, ece, ece_vs_T_curve,
    mce, nll, per_cluster_metrics, summarise_calibration,
)
from src.data import load_graph, make_synthetic_graph
from src.train import _simple_split, _planetoid_split, train_model


# ── 1. Hand-built ECE with known answer ──────────────────────────────────────
#
# 10 samples, 2 classes:
#   samples 0-4: probs=[0.9, 0.1], label=0 (correct)  → conf=0.9
#   samples 5-9: probs=[0.6, 0.4], label=1 (wrong)    → conf=0.6
#
# With 15 uniform bins (edges at k/15 for k=0..15):
#   conf=0.9 → bin 13: [12/15, 13/15) = [0.8000, 0.8667)
#   conf=0.6 → bin  9: [ 9/15, 10/15) = [0.6000, 0.6667)
#
# Bin 13: 5 samples, avg_conf=0.9, acc=1.0, gap=0.1  → contribution 5/10 * 0.1 = 0.050
# Bin  9: 5 samples, avg_conf=0.6, acc=0.0, gap=0.6  → contribution 5/10 * 0.6 = 0.300
# ECE = 0.350

def _hand_built_probs():
    probs = np.zeros((10, 2), dtype=np.float32)
    probs[:5]  = [0.9, 0.1]
    probs[5:]  = [0.6, 0.4]
    y = np.zeros(10, dtype=np.int64)
    y[:5] = 0   # correct  (argmax=0 == label 0)
    y[5:] = 1   # wrong    (argmax=0 != label 1)
    mask = np.ones(10, dtype=bool)
    return probs, y, mask


def test_ece_hand_computed_value():
    probs, y, mask = _hand_built_probs()
    val, bins_df = ece(probs, y, mask, n_bins=15, strategy="uniform")
    assert abs(val - 0.350) < 1e-6, (
        f"Hand-computed ECE expected 0.350, got {val:.6f}"
    )


def test_ece_returns_dataframe():
    probs, y, mask = _hand_built_probs()
    val, bins_df = ece(probs, y, mask)
    assert isinstance(bins_df, pd.DataFrame), "bins_df must be a DataFrame, not a plot"
    for col in ["bin_lower", "bin_upper", "count", "avg_conf", "acc", "gap"]:
        assert col in bins_df.columns, f"Missing column {col!r} in bins_df"
    assert len(bins_df) == 15, f"Expected 15 rows, got {len(bins_df)}"


def test_ece_bins_populated():
    """The two non-empty bins must match the hand-computed values."""
    probs, y, mask = _hand_built_probs()
    _, bins_df = ece(probs, y, mask, n_bins=15, strategy="uniform")
    non_empty = bins_df[bins_df["count"] > 0]
    assert len(non_empty) == 2, f"Expected 2 non-empty bins, got {len(non_empty)}"
    counts = sorted(non_empty["count"].tolist())
    assert counts == [5, 5]


# ── 2. Perfectly calibrated probs → ECE < 0.02 ──────────────────────────────
#
# 1000 samples, conf=0.8, 800 correct → all in one bin, avg_conf=0.8, acc=0.8,
# gap=0, ECE=0.

def test_ece_perfectly_calibrated():
    n = 1000
    probs = np.zeros((n, 2), dtype=np.float32)
    probs[:, 0] = 0.8
    probs[:, 1] = 0.2
    y = np.zeros(n, dtype=np.int64)
    y[800:] = 1   # 800 correct (pred=0, true=0), 200 wrong (pred=0, true=1)
    mask = np.ones(n, dtype=bool)
    val, _ = ece(probs, y, mask)
    assert val < 0.02, f"Perfectly calibrated ECE should be < 0.02, got {val:.4f}"


# ── 3. Both ECE strategies return sensible values ────────────────────────────

def test_ece_uniform_vs_quantile():
    rng = np.random.default_rng(0)
    n, C = 500, 3
    probs = rng.dirichlet(np.ones(C), size=n).astype(np.float32)
    y = rng.integers(0, C, size=n).astype(np.int64)
    mask = np.ones(n, dtype=bool)
    u_val, u_df = ece(probs, y, mask, strategy="uniform")
    q_val, q_df = ece(probs, y, mask, strategy="quantile")
    assert 0 <= u_val <= 1
    assert 0 <= q_val <= 1
    assert len(u_df) == 15
    assert len(q_df) == 15


# ── 4. Scalar metric sanity checks ───────────────────────────────────────────

def test_accuracy_all_correct():
    probs = np.eye(3, dtype=np.float32)[[0, 1, 2, 0, 1, 2]]
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    mask = np.ones(6, dtype=bool)
    assert accuracy(probs, y, mask) == pytest.approx(1.0)


def test_nll_perfect():
    """NLL of a perfect predictor should be near 0."""
    probs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    probs = np.clip(probs, 1e-12, 1.0)
    y = np.array([0, 1], dtype=np.int64)
    mask = np.ones(2, dtype=bool)
    assert nll(probs, y, mask) < 1e-6


def test_brier_perfect():
    probs = np.eye(3, dtype=np.float32)[[0, 1, 2]]
    y = np.array([0, 1, 2], dtype=np.int64)
    mask = np.ones(3, dtype=bool)
    assert brier(probs, y, mask) == pytest.approx(0.0, abs=1e-6)


def test_metrics_skip_unlabeled():
    """Nodes with y == -1 must be ignored by all metrics."""
    probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]], dtype=np.float32)
    y = np.array([0, -1, 1], dtype=np.int64)
    mask = np.ones(3, dtype=bool)
    # Only nodes 0 and 2 are labeled
    acc = accuracy(probs, y, mask)
    assert acc == pytest.approx(1.0)  # both node 0 (pred=0,true=0) and node 2 (pred=1,true=1) correct


# ── 5. ece_vs_T_curve ────────────────────────────────────────────────────────

def test_ece_vs_T_curve_shape():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(100, 3)).astype(np.float32)
    y = rng.integers(0, 3, 100).astype(np.int64)
    mask = np.ones(100, dtype=bool)
    curve = ece_vs_T_curve(logits, y, mask, T_grid=50)
    assert "T" in curve and "ece" in curve
    assert len(curve["T"]) == 50
    assert len(curve["ece"]) == 50
    assert curve["T"][0] == pytest.approx(0.1, rel=0.01)
    assert curve["T"][-1] == pytest.approx(10.0, rel=0.01)


# ── 6. per_cluster_metrics NaN-row behaviour ─────────────────────────────────

def test_per_cluster_nan_under_min_nodes():
    rng = np.random.default_rng(1)
    n = 100
    probs = rng.dirichlet([1, 1, 1], size=n).astype(np.float32)
    y = rng.integers(0, 3, n).astype(np.int64)
    mask = np.ones(n, dtype=bool)
    # clusters: 0 has 5 nodes, 1 has 95 nodes
    clusters = np.ones(n, dtype=np.int64)
    clusters[:5] = 0
    df = per_cluster_metrics(probs, y, mask, clusters, min_nodes=20)
    assert len(df) == 2
    small = df[df["cluster_id"] == 0].iloc[0]
    assert np.isnan(small["ece"]), "Cluster with 5 nodes should have NaN ECE"
    big = df[df["cluster_id"] == 1].iloc[0]
    assert not np.isnan(big["ece"]), "Cluster with 95 nodes should have real ECE"


# ── 7. Canonical Planetoid split validation ───────────────────────────────────

def test_planetoid_split_cora_shape():
    """train=140, val=500, target_fit=500, target_eval=500, no node counted twice."""
    g = load_graph("cora")
    split = _planetoid_split(g, "cora", seed=42)
    assert int(split["train"].sum())       == 140, "Expected 140 train nodes"
    assert int(split["val"].sum())         == 500, "Expected 500 val nodes"
    assert int(split["target_fit"].sum())  == 500, "Expected 500 target_fit nodes"
    assert int(split["target_eval"].sum()) == 500, "Expected 500 target_eval nodes"
    total = (
        split["train"].astype(int) + split["val"].astype(int)
        + split["target_fit"].astype(int) + split["target_eval"].astype(int)
    )
    assert total.max() <= 1, "Some nodes appear in more than one mask"


def test_planetoid_split_cora_disjoint():
    """All four masks must be mutually disjoint and cover exactly the used nodes."""
    import numpy as np
    g = load_graph("cora")
    split = _planetoid_split(g, "cora", seed=42)
    for a, b in [("train", "val"), ("train", "target_fit"), ("train", "target_eval"),
                 ("val", "target_fit"), ("val", "target_eval"),
                 ("target_fit", "target_eval")]:
        overlap = int((split[a] & split[b]).sum())
        assert overlap == 0, f"Masks '{a}' and '{b}' overlap by {overlap} nodes"


def test_planetoid_split_cora_canonical_indices():
    """Train must be nodes 0–139; val nodes 140–639; target from test.index (1708–2707)."""
    import numpy as np
    g = load_graph("cora")
    split = _planetoid_split(g, "cora", seed=42)

    train_idx = np.where(split["train"])[0]
    assert train_idx.min() == 0   and train_idx.max() == 139, (
        f"Train idx range {train_idx.min()}–{train_idx.max()} ≠ 0–139"
    )

    val_idx = np.where(split["val"])[0]
    assert val_idx.min() == 140  and val_idx.max() == 639, (
        f"Val idx range {val_idx.min()}–{val_idx.max()} ≠ 140–639"
    )

    target_idx = np.where(split["target_fit"] | split["target_eval"])[0]
    assert target_idx.min() == 1708 and target_idx.max() == 2707, (
        f"Target idx range {target_idx.min()}–{target_idx.max()} ≠ 1708–2707"
    )


# ── 8. GCN on Cora > 0.78 accuracy (canonical Planetoid split) ───────────────

@pytest.mark.slow
def test_gcn_cora_accuracy():
    g = load_graph("cora")
    split = _planetoid_split(g, "cora", seed=42)
    result = train_model(g, split, "gcn", seed=42)
    logits = result["logits"]
    logits -= logits.max(axis=1, keepdims=True)
    exp_ = np.exp(logits)
    probs = exp_ / exp_.sum(axis=1, keepdims=True)
    acc = accuracy(probs, g.y, split["target_eval"])
    print(f"\n  Cora GCN target_eval accuracy: {acc:.4f}  (val_acc={result['val_acc']:.4f})")
    assert acc >= 0.78, (
        f"Cora GCN accuracy {acc:.4f} < 0.78 – DEVIATION: reporting, not tuning."
    )


# ── 9. Training reads labels ONLY through GuardedLabels ──────────────────────
#
# Poison target labels in a copy of the graph.
# If train_model reads g.y directly (bypassing GuardedLabels), the poisoned
# labels would change the forward pass and produce different logits.
# Identical logits with the same seed prove train/val labels are the only
# labels consumed.

def test_training_uses_guarded_labels():
    g = make_synthetic_graph(n=300, n_classes=3, seed=0)
    split = _simple_split(g, n_per_class=5, n_val=30, seed=0)

    r_clean = train_model(g, split, "mlp", seed=0, epochs=10)

    # Poison every target-region label
    g_poison = copy.deepcopy(g)
    target = ~split["train"] & ~split["val"]
    g_poison.y[target] = (g_poison.y[target] + 1) % g.num_classes

    r_poison = train_model(g_poison, split, "mlp", seed=0, epochs=10)

    np.testing.assert_array_almost_equal(
        r_clean["logits"], r_poison["logits"],
        decimal=5,
        err_msg="Logits differ when target labels are poisoned – "
                "train_model is reading g.y outside GuardedLabels!",
    )
