"""src/splits.py – Graph split strategies and diagnostics.

Lane B (Saanvi) owns this file.
Phase P2-B.

Functions
---------
make_split(g, kind, seed, n_per_class, n_val, train_region_frac) -> dict
split_diagnostics(g, split) -> dict
"""
from __future__ import annotations

import numpy as np


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
    if kind not in ("random", "structural", "degree", "csbm_shift"):
        raise ValueError(f"Unknown split kind: {kind!r}")
    raise NotImplementedError("Implemented in Phase P2-B")


def split_diagnostics(g, split: dict) -> dict:
    """Compute shift diagnostics for a split.

    Returns: train_homophily, test_homophily (TRUE labels, evaluation only),
    class_tv_distance, class_kl, feature_centroid_cosine,
    cross_region_edge_frac, train/target sizes, n_train_communities.
    """
    raise NotImplementedError("Implemented in Phase P2-B")
