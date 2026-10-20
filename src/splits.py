"""src/splits.py - Graph split strategies and diagnostics.

Lane B (Saanvi) owns this file.
Phase P2-B.

Functions
---------
make_split(g, kind, seed, n_per_class, n_val, train_region_frac) -> dict
split_diagnostics(g, split, communities) -> dict
concat_graphs(g_source, g_target) -> Graph   (for kind="csbm_shift")

Every kind picks a source *region*; train and val are sampled only inside it and the target is
every node outside it, shuffled 50/50 into target_fit / target_eval. Region nodes not picked for
train or val are unused: they are neither source labels nor target.
"""
from __future__ import annotations

import numpy as np

from src.clustering import louvain_clusters
from src.data import _finalise
from src.metrics import edge_homophily

KINDS = ("random", "structural", "degree", "csbm_shift")
TOO_SMALL_FOR_STRUCTURAL = ("texas", "cornell", "wisconsin")


def _train_size(g, n_per_class: int) -> int:
    """Train size of the random split: n_per_class per class, capped by each class's count."""
    counts = np.bincount(g.y[g.y >= 0], minlength=g.num_classes)
    return int(np.minimum(counts, n_per_class).sum())


def _structural_region(g, rng, seed, frac):
    lo, hi = int(frac[0] * len(g.y)), int(frac[1] * len(g.y))
    comms = louvain_clusters(g, seed=seed)
    order = rng.permutation(comms.max() + 1)
    sizes = np.bincount(comms)
    region = np.zeros(len(g.y), dtype=bool)
    for c in order:
        if region.sum() >= lo:
            break
        if region.sum() + sizes[c] <= hi:
            region |= comms == c
    for cls in range(g.num_classes):
        if not np.any(region & (g.y == cls)):
            outside = ~region & (g.y == cls)
            best = np.bincount(comms[outside], minlength=len(sizes)).argmax()
            region |= comms == best
    return region, comms


def _degree_region(g, rng):
    deg = np.bincount(g.edge_index[0], minlength=len(g.y))
    tie = rng.permutation(len(g.y))
    order = np.lexsort((tie, deg))
    region = np.zeros(len(g.y), dtype=bool)
    region[order[: len(g.y) // 2]] = True
    for cls in range(g.num_classes):
        if not np.any(region & (g.y == cls)):
            cand = order[~region[order] & (g.y[order] == cls)]
            if len(cand):
                region[cand[0]] = True
    return region


def _sample_train(g, region, rng, n_per_class, target_size):
    train = np.zeros(len(g.y), dtype=bool)
    shortfall = {}
    for cls in range(g.num_classes):
        cand = np.where(region & (g.y == cls))[0]
        want = min(n_per_class, int((g.y == cls).sum()))
        take = min(want, len(cand))
        train[rng.choice(cand, size=take, replace=False)] = True
        if take < want:
            shortfall[int(cls)] = want - take
    missing = target_size - int(train.sum())
    if missing > 0:
        extra = np.where(region & ~train & (g.y >= 0))[0]
        if len(extra) < missing:
            raise ValueError(f"source region holds too few labelled nodes ({len(extra)} spare)")
        train[rng.choice(extra, size=missing, replace=False)] = True
    return train, shortfall


def make_split(
    g,
    kind: str,
    seed: int = 42,
    n_per_class: int = 20,
    n_val: int = 500,
    train_region_frac: tuple = (0.4, 0.5),
) -> dict:
    """Return bool masks: train, val, target_fit, target_eval + diagnostics.

    kind in {"random", "structural", "degree", "csbm_shift"}.

    Raises ValueError for texas/cornell/wisconsin with kind="structural".

    Hard rules:
      - every class present in train region
      - train.sum() equals random split's for same g (structural/degree)
      - target split 50/50 into target_fit and target_eval
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown split kind: {kind!r}")
    if kind == "structural" and g.name.lower() in TOO_SMALL_FOR_STRUCTURAL:
        raise ValueError(f"{g.name} is too small for a structural split (clustering only)")
    rng = np.random.default_rng(seed)
    N = len(g.y)
    comms = None

    if kind == "random":
        region = np.ones(N, dtype=bool)
    elif kind == "structural":
        region, comms = _structural_region(g, rng, seed, train_region_frac)
    elif kind == "degree":
        region = _degree_region(g, rng)
    else:
        if "n_source" not in g.meta:
            raise ValueError("csbm_shift needs a graph from concat_graphs(g_source, g_target)")
        region = np.zeros(N, dtype=bool)
        region[: g.meta["n_source"]] = True

    train, shortfall = _sample_train(g, region, rng, n_per_class, _train_size(g, n_per_class))

    val_cand = np.where(region & ~train & (g.y >= 0))[0]
    val = np.zeros(N, dtype=bool)
    val[rng.choice(val_cand, size=min(n_val, len(val_cand)), replace=False)] = True

    if kind == "random":
        target = ~train & ~val
    else:
        target = ~region
    target_idx = rng.permutation(np.where(target)[0])
    half = len(target_idx) // 2
    target_fit = np.zeros(N, dtype=bool)
    target_eval = np.zeros(N, dtype=bool)
    target_fit[target_idx[:half]] = True
    target_eval[target_idx[half:]] = True

    split = dict(train=train, val=val, target_fit=target_fit, target_eval=target_eval,
                 class_shortfall=shortfall)
    split["diagnostics"] = split_diagnostics(g, split, communities=comms)
    split["diagnostics"].update(kind=kind, seed=seed, n_val_short=max(0, n_val - int(val.sum())))
    return split


def _class_dist(y, mask, C):
    counts = np.bincount(y[mask & (y >= 0)], minlength=C).astype(float)
    return counts / max(counts.sum(), 1.0)


def split_diagnostics(g, split: dict, communities: np.ndarray = None) -> dict:
    """Compute shift diagnostics for a split.

    Source = every non-target node (the train region). Uses TRUE labels: these numbers are for
    evaluation only and must never feed a calibrator, the gate or a proxy.

    Returns: train_homophily, test_homophily, class_tv_distance, class_kl,
    feature_centroid_cosine, cross_region_edge_frac, train/target sizes, n_train_communities.
    """
    target = split["target_fit"] | split["target_eval"]
    source = ~target
    src, dst = g.edge_index
    C = g.num_classes

    def inside(mask):
        return edge_homophily(g.edge_index[:, mask[src] & mask[dst]], g.y)

    # Exclude the class-balanced train sample from the source class distribution: otherwise the
    # random split's "shift" on imbalanced graphs is just the n_per_class balancing.
    p, q = _class_dist(g.y, source & ~split["train"], C), _class_dist(g.y, target, C)
    eps = 1e-6
    p_s, q_s = (p + eps) / (p + eps).sum(), (q + eps) / (q + eps).sum()
    mu_s, mu_t = g.X[source].mean(axis=0), g.X[target].mean(axis=0)
    cos = float(mu_s @ mu_t / (np.linalg.norm(mu_s) * np.linalg.norm(mu_t) + 1e-12))
    if communities is None:
        communities = louvain_clusters(g, seed=42)
    return dict(
        train_homophily=inside(source),
        test_homophily=inside(target),
        labels_note="homophily and class stats use TRUE labels - evaluation only",
        class_tv_distance=float(0.5 * np.abs(p - q).sum()),
        class_kl=float(np.sum(q_s * np.log(q_s / p_s))),
        feature_centroid_cosine=cos,
        cross_region_edge_frac=float(np.mean(source[src] != source[dst])) if len(src) else 0.0,
        n_train=int(split["train"].sum()),
        n_val=int(split["val"].sum()),
        n_source_region=int(source.sum()),
        n_target=int(target.sum()),
        source_region_frac=float(source.mean()),
        n_train_communities=int(len(np.unique(communities[split["train"]]))),
    )


def concat_graphs(g_source, g_target):
    """Disjoint union of two graphs (source first); meta["n_source"] marks the boundary."""
    n_s = len(g_source.y)
    ei = np.concatenate([g_source.edge_index, g_target.edge_index + n_s], axis=1)
    g = _finalise(f"{g_source.name}+{g_target.name}", np.vstack([g_source.X, g_target.X]),
                  np.concatenate([g_source.y, g_target.y]), ei,
                  max(g_source.num_classes, g_target.num_classes))
    g.meta.update(n_source=n_s, synthetic=True, is_fixture=False)
    return g


def _compare(name: str) -> None:
    import pandas as pd
    from src.data import load_graph

    g = load_graph(name)
    print(f"{name}: N={len(g.y)}, components={g.meta.get('n_components')}")
    if g.meta.get("n_components", 1) > 1:
        print(f"NOTE: {name} has {g.meta['n_components']} connected components, so Louvain "
              "communities partly follow component boundaries.")
    print("Diagnostics use TRUE labels - evaluation only.\n")
    keys = ["train_homophily", "test_homophily", "class_tv_distance", "class_kl",
            "feature_centroid_cosine", "cross_region_edge_frac", "n_train", "n_val",
            "n_target", "source_region_frac", "n_train_communities"]
    rows = []
    for kind in ("random", "structural", "degree"):
        diags = [make_split(g, kind, seed=s)["diagnostics"] for s in range(5)]
        row = {"kind": kind}
        for k in keys:
            v = np.array([d[k] for d in diags], dtype=float)
            row[k] = f"{v.mean():.3f} +- {v.std():.3f}"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("kind").T.to_string())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cora")
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()
    if args.compare:
        _compare(args.dataset)
