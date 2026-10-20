"""src/metrics.py – Calibration metrics and evaluation utilities.

Lane A (Sruthi) owns this file.
P0: edge_homophily only.
P2-A: accuracy, nll, brier, ece, mce, per_cluster_metrics, summarise_calibration,
      ece_vs_T_curve (implemented in Phase 2-A).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# P0: edge homophily (used by data.py and clustering.py)
# ---------------------------------------------------------------------------

def edge_homophily(edge_index: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of edges whose endpoints share the same label.

    Edges touching a node with label == -1 are ignored.

    Parameters
    ----------
    edge_index : (2, E) int64 ndarray
    labels     : (N,) int64 ndarray, -1 = unlabelled

    Returns
    -------
    float in [0, 1], or 0.0 if no valid edges exist.
    """
    if edge_index.shape[1] == 0:
        return 0.0
    src, dst = edge_index[0], edge_index[1]
    valid = (labels[src] >= 0) & (labels[dst] >= 0)
    if valid.sum() == 0:
        return 0.0
    same = labels[src[valid]] == labels[dst[valid]]
    return float(same.mean())


# ---------------------------------------------------------------------------
# P2-A stubs (implemented in Phase 2-A)
# ---------------------------------------------------------------------------

def accuracy(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    raise NotImplementedError("Implemented in Phase 2-A")


def nll(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    raise NotImplementedError("Implemented in Phase 2-A")


def brier(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    raise NotImplementedError("Implemented in Phase 2-A")


def ece(
    probs: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
):
    """Returns (value, bins_df).  strategy in {"uniform", "quantile"}."""
    raise NotImplementedError("Implemented in Phase 2-A")


def mce(probs: np.ndarray, y: np.ndarray, mask: np.ndarray, n_bins: int = 15) -> float:
    raise NotImplementedError("Implemented in Phase 2-A")


def per_cluster_metrics(
    probs: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    clusters: np.ndarray,
    min_nodes: int = 20,
):
    """Returns a DataFrame with per-cluster metrics (NaN row if < min_nodes)."""
    raise NotImplementedError("Implemented in Phase 2-A")


def summarise_calibration(
    probs: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    clusters: np.ndarray,
) -> dict:
    """Returns a dict with all CSV metric keys."""
    raise NotImplementedError("Implemented in Phase 2-A")


def ece_vs_T_curve(
    logits: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    T_grid: int = 50,
):
    """ECE on a log-spaced grid of 50 temperatures in [0.1, 10]."""
    raise NotImplementedError("Implemented in Phase 2-A")
