"""src/metrics.py – Calibration metrics and evaluation utilities.

Lane A (Sruthi) owns this file.
P0  : edge_homophily
P2-A: accuracy, nll, brier, ece, mce, per_cluster_metrics,
      summarise_calibration, ece_vs_T_curve
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# P0: edge homophily (used by data.py and clustering.py)
# ---------------------------------------------------------------------------

def edge_homophily(edge_index: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of edges whose endpoints share the same label.

    Edges touching a node with label == -1 are ignored.
    Computed over the full directed edge_index (both directions), which
    gives the same result as computing over undirected edges.
    """
    if edge_index.shape[1] == 0:
        return 0.0
    src, dst = edge_index[0], edge_index[1]
    valid = (labels[src] >= 0) & (labels[dst] >= 0)
    if valid.sum() == 0:
        return 0.0
    return float((labels[src[valid]] == labels[dst[valid]]).mean())


# ---------------------------------------------------------------------------
# Shared helper: resolve mask to labeled indices
# ---------------------------------------------------------------------------

def _labeled_idx(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Indices where mask is True AND y >= 0 (labeled)."""
    idx = np.where(mask)[0]
    return idx[y[idx] >= 0]


# ---------------------------------------------------------------------------
# P2-A: scalar calibration metrics
# ---------------------------------------------------------------------------

def accuracy(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Classification accuracy over labeled, masked nodes."""
    idx = _labeled_idx(y, mask)
    if len(idx) == 0:
        return float("nan")
    preds = probs[idx].argmax(axis=1)
    return float((preds == y[idx]).mean())


def nll(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Mean negative log-likelihood (cross-entropy) over labeled, masked nodes."""
    idx = _labeled_idx(y, mask)
    if len(idx) == 0:
        return float("nan")
    # Clip for numerical stability
    p_correct = np.clip(probs[idx, y[idx]], 1e-12, 1.0)
    return float(-np.log(p_correct).mean())


def brier(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Mean Brier score over labeled, masked nodes.

    Brier = (1/N) sum_i sum_c (p_ic - 1[y_i==c])^2
    """
    idx = _labeled_idx(y, mask)
    if len(idx) == 0:
        return float("nan")
    C = probs.shape[1]
    onehot = np.zeros((len(idx), C), dtype=np.float32)
    onehot[np.arange(len(idx)), y[idx]] = 1.0
    return float(np.mean(np.sum((probs[idx] - onehot) ** 2, axis=1)))


def ece(
    probs: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> tuple[float, pd.DataFrame]:
    """Expected Calibration Error.

    ECE = sum_b (|B_b|/n) |acc(B_b) - conf(B_b)|

    Parameters
    ----------
    probs    : (N, C) softmax probabilities
    y        : (N,) int64 labels, -1 = unlabelled
    mask     : (N,) bool – evaluation mask
    n_bins   : number of bins (default 15)
    strategy : "uniform" (equal-width) | "quantile" (equal-mass)

    Returns
    -------
    (ece_value, bins_df)  where bins_df has columns:
        bin_lower, bin_upper, count, avg_conf, acc, gap
    """
    idx = _labeled_idx(y, mask)
    if len(idx) == 0:
        empty = pd.DataFrame(
            columns=["bin_lower", "bin_upper", "count", "avg_conf", "acc", "gap"]
        )
        return float("nan"), empty

    conf = probs[idx].max(axis=1)
    preds = probs[idx].argmax(axis=1)
    correct = (preds == y[idx]).astype(float)
    n = len(idx)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")

    ece_val = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        # Last bin is closed on the right
        in_bin = (conf >= lo) & (conf <= hi) if i == n_bins - 1 \
                 else (conf >= lo) & (conf < hi)
        count = int(in_bin.sum())
        if count == 0:
            rows.append({
                "bin_lower": lo, "bin_upper": hi, "count": 0,
                "avg_conf": float("nan"), "acc": float("nan"), "gap": float("nan"),
            })
            continue
        avg_conf_b = float(conf[in_bin].mean())
        acc_b = float(correct[in_bin].mean())
        gap = abs(avg_conf_b - acc_b)
        ece_val += (count / n) * gap
        rows.append({
            "bin_lower": lo, "bin_upper": hi, "count": count,
            "avg_conf": avg_conf_b, "acc": acc_b, "gap": gap,
        })

    bins_df = pd.DataFrame(rows)
    return float(ece_val), bins_df


def mce(
    probs: np.ndarray, y: np.ndarray, mask: np.ndarray, n_bins: int = 15
) -> float:
    """Maximum Calibration Error: max over bins of |acc(B) - conf(B)|."""
    _, bins_df = ece(probs, y, mask, n_bins=n_bins, strategy="uniform")
    gaps = bins_df["gap"].dropna()
    return float(gaps.max()) if len(gaps) > 0 else float("nan")


def avg_confidence(probs: np.ndarray, mask: np.ndarray) -> float:
    """Mean max-softmax confidence over masked nodes."""
    if mask.sum() == 0:
        return float("nan")
    return float(probs[mask].max(axis=1).mean())


# ---------------------------------------------------------------------------
# P2-A: per-cluster and summary metrics
# ---------------------------------------------------------------------------

def per_cluster_metrics(
    probs: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    clusters: np.ndarray,
    min_nodes: int = 20,
) -> pd.DataFrame:
    """Per-cluster calibration metrics.

    Returns one row per cluster.  When a cluster has fewer than min_nodes
    eval-labeled nodes, metrics are NaN but the row is kept.

    Columns: cluster_id, n_eval, acc, ece, ece_massbins, mce, avg_conf.
    """
    cluster_ids = np.unique(clusters)
    rows = []
    for cid in cluster_ids:
        in_cluster = clusters == cid
        cmask = mask & in_cluster
        idx = _labeled_idx(y, cmask)
        n_eval = len(idx)
        if n_eval < min_nodes:
            rows.append({
                "cluster_id": int(cid), "n_eval": n_eval,
                "acc": float("nan"), "ece": float("nan"),
                "ece_massbins": float("nan"), "mce": float("nan"),
                "avg_conf": float("nan"),
            })
        else:
            ece_u, _ = ece(probs, y, cmask)
            ece_q, _ = ece(probs, y, cmask, strategy="quantile")
            rows.append({
                "cluster_id": int(cid),
                "n_eval": n_eval,
                "acc": accuracy(probs, y, cmask),
                "ece": ece_u,
                "ece_massbins": ece_q,
                "mce": mce(probs, y, cmask),
                "avg_conf": avg_confidence(probs, cmask),
            })
    return pd.DataFrame(rows)


def summarise_calibration(
    probs: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    clusters: np.ndarray,
) -> dict:
    """Compute all CSV metric keys for a (probs, y, mask, clusters) tuple.

    Returns a dict whose keys are a subset of RESULT_COLUMNS.
    """
    ece_u, _ = ece(probs, y, mask, strategy="uniform")
    ece_q, _ = ece(probs, y, mask, strategy="quantile")
    cluster_df = per_cluster_metrics(probs, y, mask, clusters)

    # Per-cluster ECE summary
    valid_ece = cluster_df["ece"].dropna()
    if len(valid_ece) > 0:
        worst_idx = valid_ece.idxmax()
        cluster_ece_worst = float(valid_ece.max())
        worst_cluster_id = int(cluster_df.loc[worst_idx, "cluster_id"])
        cluster_ece_mean = float(valid_ece.mean())
        cluster_ece_std = float(valid_ece.std())
    else:
        cluster_ece_worst = float("nan")
        worst_cluster_id = -1
        cluster_ece_mean = float("nan")
        cluster_ece_std = float("nan")

    return {
        "acc": accuracy(probs, y, mask),
        "nll": nll(probs, y, mask),
        "brier": brier(probs, y, mask),
        "ece": ece_u,
        "ece_massbins": ece_q,
        "mce": mce(probs, y, mask),
        "avg_conf": avg_confidence(probs, mask),
        "n_clusters": int(len(np.unique(clusters))),
        "cluster_ece_mean": cluster_ece_mean,
        "cluster_ece_worst": cluster_ece_worst,
        "cluster_ece_std": cluster_ece_std,
        "worst_cluster_id": worst_cluster_id,
    }


def ece_vs_T_curve(
    logits: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    T_grid: int = 50,
) -> dict:
    """ECE on a log-spaced grid of T_grid temperatures in [0.1, 10].

    Returns {"T": array(T_grid,), "ece": array(T_grid,)}.
    Used by the dashboard slider (no recomputation at render time).
    """
    T_values = np.logspace(np.log10(0.1), np.log10(10.0), T_grid)
    ece_values = np.zeros(T_grid)
    for i, T in enumerate(T_values):
        scaled = logits / T
        # Numerically stable softmax
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_ = np.exp(scaled)
        probs_T = exp_ / exp_.sum(axis=1, keepdims=True)
        ece_values[i], _ = ece(probs_T, y, mask)
    return {"T": T_values, "ece": ece_values}
