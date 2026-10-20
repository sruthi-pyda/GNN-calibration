"""src/models.py – GNN and MLP model definitions.

Lane A (Sruthi) owns this file.
Phase P2-A.

Models: MLP, GCN, SGC (all 2-layer, plain PyTorch).
GAT is optional (Phase P7).
"""
from __future__ import annotations


def normalized_adj(g):
    """D^{-1/2} (A+I) D^{-1/2} as a torch sparse tensor."""
    raise NotImplementedError("Implemented in Phase P2-A")


class MLP:
    """2-layer MLP: Linear(F, 64) -> ReLU -> Dropout(0.5) -> Linear(64, C)."""
    def __init__(self, in_features: int, num_classes: int, hidden: int = 64, dropout: float = 0.5):
        raise NotImplementedError("Implemented in Phase P2-A")


class GCN:
    """2-layer GCN using precomputed normalised adjacency."""
    def __init__(self, in_features: int, num_classes: int, hidden: int = 64, dropout: float = 0.5):
        raise NotImplementedError("Implemented in Phase P2-A")


class SGC:
    """SGC: precompute S^2 X once, then a single Linear layer."""
    def __init__(self, in_features: int, num_classes: int, hops: int = 2):
        raise NotImplementedError("Implemented in Phase P2-A")
