"""src/calibrate.py – Calibration methods: TS, LF-TS, LF-CTS, LF-CTS+, oracles.

Lane A (Sruthi) owns this file.
Phase P4-A.

Each method declares LABELS = "none" | "source_val" | "target_fit"
and accepts GuardedLabels.
"""
from __future__ import annotations

import numpy as np

LABELS = "none"  # module-level default; each function overrides


def apply_temperature(logits: np.ndarray, T) -> np.ndarray:
    """Apply temperature T (scalar or per-node array) to logits."""
    raise NotImplementedError("Implemented in Phase P4-A")


def ts_val(logits: np.ndarray, g, labels, split: dict) -> np.ndarray:
    """Temperature scaling fitted on source validation. LABELS='source_val'."""
    raise NotImplementedError("Implemented in Phase P4-A")


def ts_target_oracle(logits: np.ndarray, g, labels, split: dict) -> np.ndarray:
    """Global TS fitted on target_fit. LABELS='target_fit'. Ceiling."""
    raise NotImplementedError("Implemented in Phase P4-A")


def cluster_ts_target_oracle(
    logits: np.ndarray, g, labels, split: dict, clusters: np.ndarray
) -> np.ndarray:
    """Per-cluster TS fitted on target_fit. LABELS='target_fit'. Per-cluster ceiling."""
    raise NotImplementedError("Implemented in Phase P4-A")


def label_free_proxy(
    g, logits: np.ndarray, clusters: np.ndarray, h_hat: float,
    embedding: np.ndarray, k_nn: int = 10
) -> np.ndarray:
    """Compute label-free accuracy proxy a_hat per cluster."""
    raise NotImplementedError("Implemented in Phase P4-A")


def lf_ts(logits: np.ndarray, g, clusters: np.ndarray, h_hat: float,
          embedding: np.ndarray) -> np.ndarray:
    """One global T matched to graph-wide proxy. LABELS='none'."""
    raise NotImplementedError("Implemented in Phase P4-A")


def lf_cts(logits: np.ndarray, g, clusters: np.ndarray, h_hat: float,
           embedding: np.ndarray) -> np.ndarray:
    """Per-cluster T via bisection on a_hat[c]. LABELS='none'."""
    raise NotImplementedError("Implemented in Phase P4-A")


def lf_cts_plus(logits: np.ndarray, g, labels, split: dict,
                clusters: np.ndarray, h_hat: float, embedding: np.ndarray) -> np.ndarray:
    """Proxy-bias-corrected LF-CTS. LABELS='source_val'."""
    raise NotImplementedError("Implemented in Phase P4-A")


def proxy_quality(
    a_hat: np.ndarray, clusters: np.ndarray, y: np.ndarray, target_eval: np.ndarray
):
    """Evaluate proxy quality. Returns (DataFrame, pearson_r). EVALUATION ONLY."""
    raise NotImplementedError("Implemented in Phase P4-A")
