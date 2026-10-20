"""src/clustering.py – Homophily-gated clustering pipeline.

Lane B (Saanvi) owns this file.
Phase P1-B.

Functions
---------
estimate_homophily(edge_index, labels)
cluster_graph(g, gate_labels, gate_source, k, hops, threshold, force, seed) -> dict
feature_kmeans(g, k, seed) -> np.ndarray
louvain_clusters(g, seed) -> np.ndarray
spectral_clusters(g, k, seed) -> np.ndarray
cluster_metrics(g, clusters, embedding, eval_labels) -> dict
cluster_profile(g, clusters, eval_labels) -> pd.DataFrame
"""
from __future__ import annotations

import numpy as np


def estimate_homophily(edge_index: np.ndarray, labels: np.ndarray) -> float:
    """Compute edge homophily for true OR predicted labels (ignores label == -1)."""
    raise NotImplementedError("Implemented in Phase P1-B")


def cluster_graph(
    g,
    gate_labels: np.ndarray,
    gate_source: str,
    k: int = None,
    hops: int = 2,
    threshold: float = 0.5,
    force: str = None,
    seed: int = 42,
) -> dict:
    """Gate on h_hat, smooth, embed, cluster.

    Parameters
    ----------
    gate_labels : REQUIRED – predicted or true labels (never g.y directly).
    gate_source : one of "mlp_pred", "gnn_pred", "true_oracle".
    force       : override branch with "sgc_kmeans" or "feature_kmeans".

    Returns
    -------
    dict with keys: clusters, method_used, h_hat, gate_source, threshold,
    embedding (N,64), embedding2d (N,2).
    """
    if gate_labels is None:
        raise ValueError("gate_labels is REQUIRED – never pass None")
    if gate_source not in ("mlp_pred", "gnn_pred", "true_oracle"):
        raise ValueError(f"gate_source must be mlp_pred/gnn_pred/true_oracle, got {gate_source!r}")
    raise NotImplementedError("Implemented in Phase P1-B")


def feature_kmeans(g, k: int, seed: int = 42) -> np.ndarray:
    """KMeans on raw (row-normalised) features."""
    raise NotImplementedError("Implemented in Phase P1-B")


def louvain_clusters(g, seed: int = 42) -> np.ndarray:
    """Louvain community detection via networkx."""
    raise NotImplementedError("Implemented in Phase P1-B")


def spectral_clusters(g, k: int, seed: int = 42) -> np.ndarray:
    """Spectral clustering with precomputed affinity."""
    raise NotImplementedError("Implemented in Phase P1-B")


def cluster_metrics(
    g,
    clusters: np.ndarray,
    embedding: np.ndarray,
    eval_labels: np.ndarray = None,
) -> dict:
    """NMI, ARI, modularity, silhouette."""
    raise NotImplementedError("Implemented in Phase P1-B")


def cluster_profile(g, clusters: np.ndarray, eval_labels: np.ndarray):
    """Per-cluster DataFrame: size, majority_class, purity, conductance, avg_degree, inside_homophily."""
    raise NotImplementedError("Implemented in Phase P1-B")
