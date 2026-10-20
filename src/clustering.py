"""src/clustering.py - Homophily-gated clustering pipeline.

Lane B (Saanvi) owns this file.
Phase P1-B.

Functions
---------
estimate_homophily(edge_index, labels)
smooth_features(X, A, hops) -> np.ndarray
embed(Xs, dim, seed) -> np.ndarray
cluster_graph(g, gate_labels, gate_source, k, hops, threshold, force, seed) -> dict
feature_kmeans(g, k, seed) -> np.ndarray
louvain_clusters(g, seed) -> np.ndarray
spectral_clusters(g, k, seed) -> np.ndarray
cluster_metrics(g, clusters, embedding, eval_labels) -> dict
cluster_profile(g, clusters, eval_labels) -> pd.DataFrame

Baseline recipe (Oct 10): SGC smoothing S = D^-1/2 (A+I) D^-1/2 applied `hops` times with
row-normalisation after each, TruncatedSVD(64, random_state=42), KMeans(k, n_init=20).
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from src.data import _row_normalise, to_sparse_adj
from src.metrics import edge_homophily

log = logging.getLogger(__name__)

GATE_SOURCES = ("mlp_pred", "gnn_pred", "true_oracle")
METHODS = ("sgc_kmeans", "feature_kmeans")


def estimate_homophily(edge_index: np.ndarray, labels: np.ndarray) -> float:
    """Compute edge homophily for true OR predicted labels (ignores label == -1)."""
    return edge_homophily(edge_index, np.asarray(labels, dtype=np.int64))


def smooth_features(X: np.ndarray, A: sp.csr_matrix, hops: int) -> np.ndarray:
    """Row-normalise X, then apply S = D^-1/2 (A+I) D^-1/2 `hops` times, re-normalising each hop."""
    N = A.shape[0]
    A_hat = A + sp.eye(N, format="csr", dtype=np.float32)
    d_inv_sqrt = 1.0 / np.sqrt(np.asarray(A_hat.sum(axis=1)).ravel())
    D = sp.diags(d_inv_sqrt)
    S = (D @ A_hat @ D).tocsr()
    Xs = _row_normalise(X)
    for _ in range(hops):
        Xs = _row_normalise(np.asarray(S @ Xs))
    return Xs


def embed(Xs: np.ndarray, dim: int = 64, seed: int = 42) -> np.ndarray:
    """TruncatedSVD to min(dim, F-1, N-1) components (random_state fixed by the recipe)."""
    n_comp = min(dim, Xs.shape[1] - 1, Xs.shape[0] - 1)
    if n_comp < dim:
        log.warning("embed: using %d SVD components instead of %d", n_comp, dim)
    return TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(Xs)


def _kmeans(Z: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(Z).astype(np.int64)


def _embedding_for(g, hops: int) -> np.ndarray:
    Xs = smooth_features(g.X, to_sparse_adj(g), hops) if hops > 0 else _row_normalise(g.X)
    return embed(Xs)


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
    gate_labels : REQUIRED - predicted or true labels (never g.y directly).
    gate_source : one of "mlp_pred", "gnn_pred", "true_oracle".
    force       : override branch with "sgc_kmeans" or "feature_kmeans".

    Returns
    -------
    dict with keys: clusters, method_used, h_hat, gate_source, threshold,
    embedding (N,64), embedding2d (N,2).
    """
    if gate_labels is None:
        raise ValueError("gate_labels is REQUIRED - never pass None")
    if gate_source not in GATE_SOURCES:
        raise ValueError(f"gate_source must be mlp_pred/gnn_pred/true_oracle, got {gate_source!r}")
    if force is not None and force not in METHODS:
        raise ValueError(f"force must be one of {METHODS}, got {force!r}")

    h_hat = estimate_homophily(g.edge_index, gate_labels)
    method = force or ("sgc_kmeans" if h_hat > threshold else "feature_kmeans")
    k = k or g.num_classes
    emb = _embedding_for(g, hops if method == "sgc_kmeans" else 0)
    clusters = _kmeans(emb, k, seed)
    emb2d = PCA(n_components=2, random_state=seed).fit_transform(emb)
    return dict(
        clusters=clusters, method_used=method, h_hat=h_hat, gate_source=gate_source,
        threshold=threshold, embedding=emb, embedding2d=emb2d,
    )


def feature_kmeans(g, k: int, seed: int = 42) -> np.ndarray:
    """KMeans on the SVD(64) embedding of raw (row-normalised) features - the hops=0 path."""
    return _kmeans(_embedding_for(g, 0), k, seed)


def _nx_graph(g) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(g.X.shape[0]))
    ei = g.edge_index
    upper = ei[0] < ei[1]
    G.add_edges_from(zip(ei[0, upper].tolist(), ei[1, upper].tolist()))
    return G


def louvain_clusters(g, seed: int = 42) -> np.ndarray:
    """Louvain community detection via networkx."""
    comms = nx.community.louvain_communities(_nx_graph(g), seed=seed)
    labels = np.empty(g.X.shape[0], dtype=np.int64)
    for c, nodes in enumerate(comms):
        labels[list(nodes)] = c
    return labels


def spectral_clusters(g, k: int, seed: int = 42) -> np.ndarray:
    """Spectral clustering with precomputed affinity."""
    A = to_sparse_adj(g).astype(np.float64)
    sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=seed)
    return sc.fit_predict(A).astype(np.int64)


def cluster_metrics(
    g,
    clusters: np.ndarray,
    embedding: np.ndarray,
    eval_labels: np.ndarray = None,
) -> dict:
    """NMI, ARI (only when eval_labels given, ignoring -1), modularity, silhouette."""
    out = {"n_clusters": int(len(np.unique(clusters)))}
    if eval_labels is not None:
        m = np.asarray(eval_labels) >= 0
        out["nmi"] = float(normalized_mutual_info_score(eval_labels[m], clusters[m]))
        out["ari"] = float(adjusted_rand_score(eval_labels[m], clusters[m]))
    parts = [set(np.where(clusters == c)[0].tolist()) for c in np.unique(clusters)]
    out["modularity"] = float(nx.community.modularity(_nx_graph(g), parts))
    if out["n_clusters"] > 1 and embedding is not None:
        n = embedding.shape[0]
        sample = 10000 if n > 10000 else None
        out["silhouette"] = float(
            silhouette_score(embedding, clusters, sample_size=sample, random_state=42)
        )
    else:
        out["silhouette"] = float("nan")
    return out


def cluster_profile(g, clusters: np.ndarray, eval_labels: np.ndarray):
    """Per-cluster DataFrame: size, majority_class, purity, conductance, avg_degree, inside_homophily."""
    src, dst = g.edge_index
    deg = np.bincount(src, minlength=g.X.shape[0])
    vol_total = deg.sum()
    rows = []
    for c in np.unique(clusters):
        members = clusters == c
        lab = eval_labels[members]
        lab = lab[lab >= 0]
        if len(lab):
            counts = np.bincount(lab)
            majority, purity = int(counts.argmax()), float(counts.max() / len(lab))
        else:
            majority, purity = -1, float("nan")
        vol = deg[members].sum()
        cut = int((members[src] & ~members[dst]).sum())
        denom = min(vol, vol_total - vol)
        inside = members[src] & members[dst]
        rows.append(dict(
            cluster=int(c), size=int(members.sum()), majority_class=majority, purity=purity,
            conductance=float(cut / denom) if denom > 0 else float("nan"),
            avg_degree=float(deg[members].mean()),
            inside_homophily=edge_homophily(g.edge_index[:, inside], eval_labels),
        ))
    return pd.DataFrame(rows)


def _compare(name: str) -> None:
    from src.data import load_graph

    g = load_graph(name)
    if g.meta.get("is_fixture"):
        print("=" * 60 + "\nFIXTURE DATA - not results\n" + "=" * 60)
    print("=" * 60)
    print("TRUE-LABEL ORACLE GATE (gate_source=true_oracle) - baseline reproduction only")
    print("=" * 60)
    y, k = g.y, g.num_classes
    res = cluster_graph(g, y, "true_oracle", seed=42)
    print(f"\n{name}: h(true)={res['h_hat']:.3f} -> gate picks {res['method_used']}\n")

    rows = []
    for m in METHODS:
        r = cluster_graph(g, y, "true_oracle", force=m, seed=42)
        rows.append(dict(method=m, **cluster_metrics(g, r["clusters"], r["embedding"], y)))
    emb = cluster_graph(g, y, "true_oracle", force="sgc_kmeans")["embedding"]
    rows.append(dict(method="louvain", **cluster_metrics(g, louvain_clusters(g, 42), emb, y)))
    rows.append(dict(method="spectral", **cluster_metrics(g, spectral_clusters(g, k, 42), emb, y)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nHop sweep (NMI, mean over seeds 0-4):")
    sweep = []
    for h in range(5):
        emb_h = _embedding_for(g, h)
        nmis = [normalized_mutual_info_score(y, _kmeans(emb_h, k, s)) for s in range(5)]
        sweep.append(dict(hops=h, nmi_mean=np.mean(nmis), nmi_std=np.std(nmis)))
    print(pd.DataFrame(sweep).round(3).to_string(index=False))

    print(f"\nCluster profile ({res['method_used']}, seed 42):")
    print(cluster_profile(g, res["clusters"], y).round(3).to_string(index=False))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cora")
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()
    if args.compare:
        _compare(args.dataset)
