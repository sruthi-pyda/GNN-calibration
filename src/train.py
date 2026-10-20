"""src/train.py – Model training loop.

Lane A (Sruthi) owns this file.
Phase P2-A.

train_model(g, split, model_name, seed, epochs, lr, wd, patience) -> dict
"""
from __future__ import annotations


def train_model(
    g,
    split: dict,
    model_name: str,
    seed: int = 42,
    epochs: int = 300,
    lr: float = 0.01,
    wd: float = 5e-4,
    patience: int = 50,
) -> dict:
    """Train a model and return logits for ALL nodes.

    Parameters
    ----------
    g           : Graph
    split       : dict from make_split
    model_name  : "mlp", "gcn", or "sgc"
    seed        : seeds torch, numpy, random

    Returns
    -------
    dict with keys: logits (N, C) numpy, best_epoch, val_acc, train_sec.

    Labels for train/val come through GuardedLabels – never g.y directly.
    """
    raise NotImplementedError("Implemented in Phase P2-A")
