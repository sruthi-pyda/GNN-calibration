"""src/data.py – Graph dataclass, loaders, and synthetic graph generators.

Lane A (Sruthi) owns this file.

P0  : Graph, validate_graph, to_sparse_adj, make_synthetic_graph
P1-A: load_graph, make_csbm, _finalise, all dataset-specific loaders
"""
from __future__ import annotations

import json
import os
import pickle
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Core dataclass (P0 – frozen contract)
# ---------------------------------------------------------------------------

@dataclass
class Graph:
    name: str
    X: np.ndarray           # (N, F) float32, row-normalised
    y: np.ndarray           # (N,) int64, -1 = unlabelled
    edge_index: np.ndarray  # (2, E) int64, undirected (both dirs), no self-loops,
                            # no duplicates, sorted lex (row0 then row1)
    num_classes: int
    meta: dict = field(default_factory=dict)


def validate_graph(g: Graph) -> None:
    """Assert all Graph invariants. Raises AssertionError on violation."""
    assert isinstance(g.name, str) and g.name, "name must be a non-empty string"
    assert g.X.ndim == 2, f"X must be 2-D, got shape {g.X.shape}"
    assert g.X.dtype == np.float32, f"X must be float32, got {g.X.dtype}"
    N = g.X.shape[0]
    assert g.y.shape == (N,), f"y shape {g.y.shape} != ({N},)"
    assert g.y.dtype == np.int64, f"y must be int64, got {g.y.dtype}"
    assert g.edge_index.ndim == 2 and g.edge_index.shape[0] == 2, (
        f"edge_index must be shape (2, E), got {g.edge_index.shape}"
    )
    assert g.edge_index.dtype == np.int64, (
        f"edge_index must be int64, got {g.edge_index.dtype}"
    )
    E = g.edge_index.shape[1]
    if E > 0:
        assert int(g.edge_index.min()) >= 0 and int(g.edge_index.max()) < N, (
            "edge_index indices out of [0, N)"
        )
        src, dst = g.edge_index[0], g.edge_index[1]
        # no self-loops
        assert np.all(src != dst), "edge_index contains self-loops"
        # sorted lexicographically
        pairs = np.stack([src, dst], axis=1)
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        assert np.all(order == np.arange(E)), "edge_index is not sorted"
        # no duplicates
        assert len(np.unique(pairs, axis=0)) == E, "edge_index has duplicates"
        # undirected: every (u,v) has a matching (v,u)
        fwd = set(zip(src.tolist(), dst.tolist()))
        rev = set(zip(dst.tolist(), src.tolist()))
        assert fwd == rev, "edge_index is not symmetric (not undirected)"
    assert isinstance(g.num_classes, int) and g.num_classes >= 2, (
        "num_classes must be an int >= 2"
    )
    assert isinstance(g.meta, dict), "meta must be a dict"


def to_sparse_adj(g: Graph) -> sp.csr_matrix:
    """Return a symmetric (N x N) CSR adjacency matrix (binary, no self-loops)."""
    N = g.X.shape[0]
    E = g.edge_index.shape[1]
    data = np.ones(E, dtype=np.float32)
    A = sp.csr_matrix(
        (data, (g.edge_index[0], g.edge_index[1])), shape=(N, N)
    )
    return A


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _row_normalise(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (X / norms).astype(np.float32)


def _count_components(edge_index: np.ndarray, N: int) -> int:
    parent = list(range(N))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        union(s, d)
    return len({find(i) for i in range(N)})


def _finalise(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    edge_index: np.ndarray,
    num_classes: int,
    source_url: str = "",
) -> Graph:
    """Apply all post-processing invariants and build the Graph."""
    from src.metrics import edge_homophily

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    ei = np.asarray(edge_index, dtype=np.int64)
    N = X.shape[0]

    if ei.ndim == 1 and ei.size == 0:
        ei = np.zeros((2, 0), dtype=np.int64)

    # Make undirected: add both directions
    src, dst = ei[0], ei[1]
    both_src = np.concatenate([src, dst])
    both_dst = np.concatenate([dst, src])
    ei = np.stack([both_src, both_dst], axis=0)

    # Drop self-loops
    mask = ei[0] != ei[1]
    ei = ei[:, mask]

    # Drop out-of-range edges (safety)
    if ei.shape[1] > 0:
        valid = (ei[0] >= 0) & (ei[0] < N) & (ei[1] >= 0) & (ei[1] < N)
        ei = ei[:, valid]

    # Drop duplicates
    if ei.shape[1] > 0:
        pairs = np.unique(ei.T, axis=0)
        ei = pairs.T
    else:
        ei = np.zeros((2, 0), dtype=np.int64)

    # Sort lexicographically
    if ei.shape[1] > 0:
        order = np.lexsort((ei[1], ei[0]))
        ei = ei[:, order]

    # Row-normalise features
    X = _row_normalise(X)

    N = X.shape[0]
    degrees = np.bincount(ei[0], minlength=N) if ei.shape[1] > 0 else np.zeros(N)
    n_comp = _count_components(ei, N)
    h = edge_homophily(ei, y)

    meta = {
        "n_nodes": int(N),
        "n_edges_undirected": int(ei.shape[1] // 2),
        "n_features": int(X.shape[1]),
        "n_classes": int(num_classes),
        "n_components": int(n_comp),
        "avg_degree": float(degrees.mean()),
        "edge_homophily": float(h),
        "source_url": source_url,
    }

    return Graph(
        name=name,
        X=X,
        y=y,
        edge_index=ei.astype(np.int64),
        num_classes=int(num_classes),
        meta=meta,
    )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} -> {dest.name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    print("done")


def _cache_path(name: str, root: Path) -> tuple[Path, Path]:
    cache_dir = root / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{name}.npz", cache_dir / f"{name}_meta.json"


def _save_cache(g: Graph, root: Path) -> None:
    npz_path, meta_path = _cache_path(g.name, root)
    np.savez_compressed(npz_path, X=g.X, y=g.y, edge_index=g.edge_index)
    with open(meta_path, "w") as f:
        json.dump({"num_classes": g.num_classes, **g.meta}, f, indent=2)


def _load_cache(name: str, root: Path) -> Optional[Graph]:
    npz_path, meta_path = _cache_path(name, root)
    if not npz_path.exists() or not meta_path.exists():
        return None
    data = np.load(npz_path)
    with open(meta_path) as f:
        meta_full = json.load(f)
    num_classes = int(meta_full.pop("num_classes"))
    return Graph(
        name=name,
        X=data["X"],
        y=data["y"],
        edge_index=data["edge_index"],
        num_classes=num_classes,
        meta=meta_full,
    )


# ---------------------------------------------------------------------------
# P0: Synthetic graph generators
# ---------------------------------------------------------------------------

def make_synthetic_graph(
    n: int = 600,
    n_classes: int = 3,
    n_feat: int = 50,
    homophily: float = 0.8,
    avg_degree: float = 6.0,
    seed: int = 42,
) -> Graph:
    """Planted-partition graph with class-conditional Gaussian features.

    homophily = fraction of edges that connect same-class nodes.
    meta["is_fixture"] = True.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n)

    n_edges_target = int(n * avg_degree / 2)
    edges_src, edges_dst = [], []
    max_attempts = n_edges_target * 20

    attempts = 0
    while len(edges_src) < n_edges_target and attempts < max_attempts:
        attempts += 1
        u = int(rng.integers(0, n))
        if rng.random() < homophily:
            same = np.where(labels == labels[u])[0]
            same = same[same != u]
            if len(same) == 0:
                continue
            v = int(rng.choice(same))
        else:
            diff = np.where(labels != labels[u])[0]
            if len(diff) == 0:
                continue
            v = int(rng.choice(diff))
        edges_src.append(u)
        edges_dst.append(v)

    ei = np.array([edges_src, edges_dst], dtype=np.int64)

    # Class-conditional Gaussian features (heavily overlapping: std >> separation)
    centers = rng.normal(0, 0.5, size=(n_classes, n_feat))
    X = np.array(
        [rng.normal(centers[c], 2.0, size=n_feat) for c in labels], dtype=np.float32
    )

    g = _finalise(f"synthetic_h{homophily:.1f}", X, labels, ei, n_classes)
    g.meta["is_fixture"] = True
    g.meta["homophily_target"] = float(homophily)
    g.meta["seed"] = int(seed)
    return g


def make_csbm(
    n: int = 3000,
    n_classes: int = 3,
    homophily: float = 0.8,
    avg_degree: float = 8.0,
    feat_dim: int = 64,
    mu: float = 1.0,
    seed: int = 42,
) -> Graph:
    """Contextual Stochastic Block Model.

    meta["is_fixture"] = False, meta["synthetic"] = True.
    """
    rng = np.random.default_rng(seed)

    # Balanced classes
    sizes = [n // n_classes] * n_classes
    sizes[-1] += n - sum(sizes)
    labels = np.repeat(np.arange(n_classes), sizes)
    rng.shuffle(labels)

    # Compute p_in and p_out from homophily and avg_degree
    n_c = n // n_classes
    if homophily >= 1.0:
        p_in = min(avg_degree / (n_c - 1), 1.0)
        p_out = 0.0
    elif homophily <= 0.0:
        p_in = 0.0
        p_out = min(avg_degree / (n - n_c), 1.0)
    else:
        # h = p_in*(n_c-1) / (p_in*(n_c-1) + p_out*(n-n_c))
        # d = p_in*(n_c-1) + p_out*(n-n_c)
        ratio = homophily / (1.0 - homophily) * (n - n_c) / (n_c - 1)
        p_out = avg_degree / (ratio * (n_c - 1) + (n - n_c))
        p_in = ratio * p_out
        p_in = min(p_in, 1.0)
        p_out = min(p_out, 1.0)

    edges_src: list[int] = []
    edges_dst: list[int] = []

    for c in range(n_classes):
        nodes_c = np.where(labels == c)[0]
        nc = len(nodes_c)
        if nc < 2:
            continue
        # Intra-class (upper triangle)
        ii, jj = np.triu_indices(nc, k=1)
        keep = rng.random(len(ii)) < p_in
        edges_src.extend(nodes_c[ii[keep]].tolist())
        edges_dst.extend(nodes_c[jj[keep]].tolist())

    for c1 in range(n_classes):
        for c2 in range(c1 + 1, n_classes):
            nc1 = np.where(labels == c1)[0]
            nc2 = np.where(labels == c2)[0]
            # Sample from all n1*n2 pairs
            ii = np.repeat(np.arange(len(nc1)), len(nc2))
            jj = np.tile(np.arange(len(nc2)), len(nc1))
            keep = rng.random(len(ii)) < p_out
            edges_src.extend(nc1[ii[keep]].tolist())
            edges_dst.extend(nc2[jj[keep]].tolist())

    ei = np.array([edges_src, edges_dst], dtype=np.int64)

    # CSBM features: class means on the first n_classes dims, noise elsewhere
    centers = np.zeros((n_classes, feat_dim), dtype=np.float32)
    for c in range(min(n_classes, feat_dim)):
        centers[c, c] = mu
    X = np.array(
        [rng.normal(centers[c], 1.0, size=feat_dim) for c in labels], dtype=np.float32
    )

    g = _finalise(f"csbm_h{homophily}", X, labels, ei, n_classes)
    g.meta["is_fixture"] = False
    g.meta["synthetic"] = True
    g.meta["homophily_target"] = float(homophily)
    g.meta["seed"] = int(seed)
    return g


# ---------------------------------------------------------------------------
# P1-A: Dataset loaders
# ---------------------------------------------------------------------------

_PLANETOID_URL = (
    "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"
)
_WEBKB_URL = (
    "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data"
)
_YANDEX_URL = (
    "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data"
)

_DATASET_ALIASES = {
    "cora": "cora",
    "citeseer": "citeseer",
    "pubmed": "pubmed",
    "texas": "texas",
    "cornell": "cornell",
    "wisconsin": "wisconsin",
    "chameleon": "chameleon_filtered",
    "chameleon_filtered": "chameleon_filtered",
    "roman_empire": "roman_empire",
}


def load_graph(name: str, root: str = "data", use_cache: bool = True) -> Graph:
    """Load a graph by name, downloading raw files if needed.

    Supported names: cora, citeseer, pubmed, texas, cornell, wisconsin,
    chameleon (filtered), roman_empire.

    Downloads to <root>/<name>/ and caches to cache/<name>.npz.
    """
    root_path = ROOT / root
    canonical = _DATASET_ALIASES.get(name.lower(), name.lower())

    if use_cache:
        cached = _load_cache(canonical, ROOT)
        if cached is not None:
            return cached

    if canonical in ("cora", "citeseer", "pubmed"):
        g = _load_planetoid(canonical, root_path / canonical)
    elif canonical in ("texas", "cornell", "wisconsin"):
        g = _load_webkb(canonical, root_path / canonical)
    elif canonical == "chameleon_filtered":
        g = _load_yandex_npz(
            "chameleon_filtered",
            root_path / "chameleon_filtered",
            f"{_YANDEX_URL}/chameleon_filtered.npz",
        )
    elif canonical == "roman_empire":
        g = _load_yandex_npz(
            "roman_empire",
            root_path / "roman_empire",
            f"{_YANDEX_URL}/roman_empire.npz",
        )
    else:
        raise ValueError(f"Unknown dataset: {name!r}")

    _save_cache(g, ROOT)
    return g


def _load_planetoid(name: str, data_dir: Path) -> Graph:
    data_dir.mkdir(parents=True, exist_ok=True)
    suffixes = ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]
    for suf in suffixes:
        fname = f"ind.{name}.{suf}"
        fpath = data_dir / fname
        if not fpath.exists():
            url = f"{_PLANETOID_URL}/{fname}"
            _download(url, fpath)

    def _pkl(suf):
        with open(data_dir / f"ind.{name}.{suf}", "rb") as f:
            return pickle.load(f, encoding="latin1")

    x, y_train, tx, ty, allx, ally, graph = (
        _pkl(s) for s in ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    )

    with open(data_dir / f"ind.{name}.test.index") as f:
        test_idx = [int(line.strip()) for line in f if line.strip()]

    # --- build full feature matrix and label matrix ---
    # allx covers nodes 0 .. allx.shape[0]-1
    # test nodes have indices given by test_idx

    allx_dense = allx.toarray() if sp.issparse(allx) else np.array(allx)
    tx_dense = tx.toarray() if sp.issparse(tx) else np.array(tx)
    ally_dense = np.array(ally.toarray() if sp.issparse(ally) else ally)
    ty_dense = np.array(ty.toarray() if sp.issparse(ty) else ty)

    # For CiteSeer, some test node indices have no features (isolated nodes).
    # Extend tx/ty to cover the full range [min(test_idx), max(test_idx)].
    test_idx_sorted = sorted(test_idx)
    if name == "citeseer":
        min_ti = min(test_idx_sorted)
        max_ti = max(test_idx_sorted)
        full_len = max_ti - min_ti + 1
        tx_ext = np.zeros((full_len, allx_dense.shape[1]), dtype=np.float32)
        ty_ext = np.zeros((full_len, ally_dense.shape[1]), dtype=np.float32)
        for j, idx in enumerate(test_idx):
            tx_ext[idx - min_ti] = tx_dense[j]
            ty_ext[idx - min_ti] = ty_dense[j]
        tx_dense = tx_ext
        ty_dense = ty_ext
        test_idx_for_placement = list(range(min_ti, max_ti + 1))
    else:
        test_idx_for_placement = test_idx

    N_allx = allx_dense.shape[0]
    N = N_allx + len(test_idx_for_placement)

    X = np.zeros((N, allx_dense.shape[1]), dtype=np.float32)
    X[:N_allx] = allx_dense
    for j, idx in enumerate(test_idx_for_placement):
        X[idx] = tx_dense[j]

    labels_oh = np.zeros((N, ally_dense.shape[1]), dtype=np.float32)
    labels_oh[:N_allx] = ally_dense
    for j, idx in enumerate(test_idx_for_placement):
        labels_oh[idx] = ty_dense[j]

    y = labels_oh.argmax(axis=1).astype(np.int64)
    # Nodes with all-zero label rows (isolated citeseer nodes) get label -1
    y[labels_oh.sum(axis=1) == 0] = -1

    n_classes = int(ally_dense.shape[1])

    # Build edge index from adjacency dict
    src_list, dst_list = [], []
    for s, neighbors in graph.items():
        s = int(s)
        for d in neighbors:
            d = int(d)
            if s < N and d < N:
                src_list.append(s)
                dst_list.append(d)
    ei = (
        np.array([src_list, dst_list], dtype=np.int64)
        if src_list
        else np.zeros((2, 0), dtype=np.int64)
    )

    return _finalise(name, X, y, ei, n_classes, source_url=_PLANETOID_URL)


def _load_webkb(name: str, data_dir: Path) -> Graph:
    data_dir.mkdir(parents=True, exist_ok=True)
    base = f"{_WEBKB_URL}/{name}"
    edges_file = "out1_graph_edges.txt"
    nodes_file = "out1_node_feature_label.txt"
    for fname in [edges_file, nodes_file]:
        fpath = data_dir / fname
        if not fpath.exists():
            _download(f"{base}/{fname}", fpath)

    features, labels = [], []
    with open(data_dir / nodes_file) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            feat = [float(v) for v in parts[1].split(",")]
            label = int(parts[2])
            features.append(feat)
            labels.append(label)

    src_list, dst_list = [], []
    with open(data_dir / edges_file) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            src_list.append(int(parts[0]))
            dst_list.append(int(parts[1]))

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    ei = (
        np.array([src_list, dst_list], dtype=np.int64)
        if src_list
        else np.zeros((2, 0), dtype=np.int64)
    )
    n_classes = int(y.max()) + 1

    return _finalise(name, X, y, ei, n_classes, source_url=f"{_WEBKB_URL}/{name}")


def _load_yandex_npz(name: str, data_dir: Path, url: str) -> Graph:
    data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / f"{name}.npz"
    if not npz_path.exists():
        _download(url, npz_path)

    data = np.load(npz_path, allow_pickle=True)
    print(f"  Keys in {name}.npz: {list(data.keys())}")

    X = np.array(data["node_features"], dtype=np.float32)
    y = np.array(data["node_labels"], dtype=np.int64)
    edges = np.array(data["edges"], dtype=np.int64)  # (E, 2)

    ei = edges.T  # (2, E)
    n_classes = int(y.max()) + 1

    return _finalise(name, X, y, ei, n_classes, source_url=url)


# ---------------------------------------------------------------------------
# CLI: python -m src.data --summary
# ---------------------------------------------------------------------------

def _summary_table(names=None):
    if names is None:
        names = ["cora", "citeseer", "pubmed", "texas", "chameleon", "roman_empire"]
    rows = []
    for name in names:
        try:
            g = load_graph(name)
            rows.append({
                "Dataset": g.name,
                "Nodes": g.meta["n_nodes"],
                "Edges": g.meta["n_edges_undirected"],
                "Features": g.meta["n_features"],
                "Classes": g.num_classes,
                "Homophily": f"{g.meta['edge_homophily']:.3f}",
                "Components": g.meta["n_components"],
                "Avg Degree": f"{g.meta['avg_degree']:.2f}",
            })
        except Exception as e:
            rows.append({"Dataset": name, "Error": str(e)})
    return rows


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()

    if args.summary or True:
        names = args.datasets or None
        rows = _summary_table(names)

        # Print markdown table
        if rows and "Error" not in rows[0]:
            headers = list(rows[0].keys())
        else:
            headers = ["Dataset", "Nodes", "Edges", "Features", "Classes",
                       "Homophily", "Components", "Avg Degree"]
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        print(header_line)
        print(sep_line)
        for row in rows:
            vals = [str(row.get(h, "")) for h in headers]
            print("| " + " | ".join(vals) + " |")
