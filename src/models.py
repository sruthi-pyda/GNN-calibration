"""src/models.py – MLP, GCN, SGC model definitions (plain PyTorch).

Lane A (Sruthi) owns this file.  Phase P2-A.

All models share the same hyper-parameters (hidden=64, dropout=0.5).
GAT is optional (Phase P7).

No torch-geometric, no DGL.  Adjacency is a precomputed torch.sparse_coo_tensor.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_adj(g) -> torch.Tensor:
    """D^{-1/2}(A+I)D^{-1/2} as a coalesced sparse COO tensor.

    edge_index already stores both directions (undirected, no self-loops).
    Self-loops are added here for the +I term.
    """
    N = g.X.shape[0]
    src = torch.tensor(g.edge_index[0], dtype=torch.long)
    dst = torch.tensor(g.edge_index[1], dtype=torch.long)

    # Add self-loops
    loop = torch.arange(N, dtype=torch.long)
    src_all = torch.cat([src, loop])
    dst_all = torch.cat([dst, loop])

    # Degree (counting self-loops)
    deg = torch.bincount(src_all, minlength=N).float()
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

    vals = d_inv_sqrt[src_all] * d_inv_sqrt[dst_all]
    A_hat = torch.sparse_coo_tensor(
        torch.stack([src_all, dst_all]), vals, (N, N)
    ).coalesce()
    return A_hat


class MLP(nn.Module):
    """2-layer MLP: Linear(F,64) → ReLU → Dropout → Linear(64,C)."""

    def __init__(self, in_features: int, num_classes: int,
                 hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_hat=None) -> torch.Tensor:
        # A_hat unused; kept so train loop can call model(X, A_hat) uniformly
        return self.fc2(self.drop(F.relu(self.fc1(x))))


class GCN(nn.Module):
    """2-layer GCN: H = A_hat·X·W per layer, with ReLU + Dropout between."""

    def __init__(self, in_features: int, num_classes: int,
                 hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(torch.sparse.mm(A_hat, x)))
        h = self.drop(h)
        return self.fc2(torch.sparse.mm(A_hat, h))


class SGC(nn.Module):
    """SGC: precompute S^K X once, then a single Linear(F, C).

    Call `x_smooth = model.precompute(X, A_hat)` before training,
    then pass x_smooth as x in forward (A_hat ignored in forward).
    """

    def __init__(self, in_features: int, num_classes: int, hops: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.hops = hops

    def precompute(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            for _ in range(self.hops):
                x = torch.sparse.mm(A_hat, x)
        return x

    def forward(self, x: torch.Tensor, A_hat=None) -> torch.Tensor:
        return self.fc(x)
