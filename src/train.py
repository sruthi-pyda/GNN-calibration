"""src/train.py – Model training loop.

Lane A (Sruthi) owns this file.  Phase P2-A.

train_model(g, split, model_name, seed, ...) -> dict
  logits: (N, C) numpy for ALL nodes
  best_epoch, val_acc, train_sec

Labels for train/val are read ONLY through GuardedLabels.
g.y is never touched directly inside this file.
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.labels import GuardedLabels
from src.models import GCN, MLP, SGC, normalized_adj


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
    """Train a model and return logits for all N nodes.

    Parameters
    ----------
    g          : Graph
    split      : dict from make_split (or _simple_split for CLI/tests).
                 Must have bool-array keys "train" and "val".
    model_name : "mlp" | "gcn" | "sgc"
    seed       : used to seed torch, numpy, and random

    Returns
    -------
    dict
        logits    : (N, C) float32 numpy – raw logits for every node
        best_epoch: int
        val_acc   : float – best validation accuracy seen
        train_sec : float
    """
    # ── seed ──────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ── labels via GuardedLabels (no direct read of g.y) ──────────────────
    train_mask = split["train"]
    val_mask   = split["val"]
    allowed    = train_mask | val_mask
    gl         = GuardedLabels(g.y, allowed)

    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    # Filter to labeled nodes (y >= 0) inside each split
    train_idx = train_idx[gl[train_idx] >= 0]
    val_idx   = val_idx[gl[val_idx]   >= 0]

    y_train = torch.tensor(gl[train_idx], dtype=torch.long)
    y_val   = torch.tensor(gl[val_idx],   dtype=torch.long)
    ti      = torch.tensor(train_idx,     dtype=torch.long)
    vi      = torch.tensor(val_idx,       dtype=torch.long)

    X      = torch.tensor(g.X, dtype=torch.float32)
    A_hat  = normalized_adj(g)

    # ── build model ───────────────────────────────────────────────────────
    F_in, C = g.X.shape[1], g.num_classes
    if model_name == "mlp":
        model  = MLP(F_in, C)
        X_feed = X
    elif model_name == "gcn":
        model  = GCN(F_in, C)
        X_feed = X
    elif model_name == "sgc":
        model  = SGC(F_in, C)
        X_feed = model.precompute(X, A_hat)   # S^2 X, computed once
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # ── training loop ─────────────────────────────────────────────────────
    # Early stopping on validation *loss* (not accuracy).
    # Rationale (joint decision, Oct 2026): val-accuracy plateaus very early
    # (~epoch 17) leaving the model under-trained and ECE unstable across seeds.
    # Val-loss continues to improve after accuracy saturates, giving a more
    # fully-trained model and more stable calibration baselines for C1.
    best_val_loss = float("inf")
    best_val_acc  = 0.0          # val_acc at the best-loss epoch, for reporting
    best_weights  = None
    no_improve    = 0
    best_epoch    = 0
    t0            = time.time()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_feed, A_hat)
        loss   = F.cross_entropy(logits[ti], y_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_feed, A_hat)[vi]
            val_loss   = F.cross_entropy(val_logits, y_val).item()
            val_acc    = (val_logits.argmax(1) == y_val).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
            best_epoch    = epoch
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # ── restore best weights and get all-node logits ──────────────────────
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        all_logits = model(X_feed, A_hat).numpy()

    return {
        "logits":     all_logits,
        "best_epoch": best_epoch,
        "val_acc":    best_val_acc,
        "train_sec":  time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Private helper: canonical Planetoid split (Cora / CiteSeer / PubMed).
# Uses fixed node indices stored in the raw Planetoid pickle files.
# ---------------------------------------------------------------------------

def _planetoid_split(
    g,
    name: str,
    n_val: int = 500,
    seed: int = 42,
    data_root=None,
) -> dict:
    """Canonical Planetoid split for Cora / CiteSeer / PubMed.

    train  : the N_x nodes in ind.{name}.x  (e.g. 140 for Cora = 20/class × 7)
    val    : the next n_val nodes from allx  (e.g. nodes 140–639)
    target : nodes in ind.{name}.test.index  split 50/50 by seed into fit/eval

    Raises FileNotFoundError if the raw Planetoid files are not present.
    Run load_graph(name) first to download them.
    """
    import pickle
    from pathlib import Path as _Path

    if data_root is None:
        data_root = _Path(__file__).parent.parent / "data"
    data_dir = _Path(data_root) / name
    x_path      = data_dir / f"ind.{name}.x"
    tidx_path   = data_dir / f"ind.{name}.test.index"

    if not x_path.exists() or not tidx_path.exists():
        raise FileNotFoundError(
            f"Planetoid raw files not found in {data_dir}. "
            f"Run load_graph('{name}') first to download them."
        )

    with open(x_path, "rb") as f:
        x_pkl = pickle.load(f, encoding="latin1")
    n_train = x_pkl.shape[0]   # 140 for Cora

    with open(tidx_path) as f:
        test_idx = np.array(
            sorted(int(line.strip()) for line in f if line.strip()),
            dtype=np.int64,
        )

    N   = g.X.shape[0]
    rng = np.random.default_rng(seed)

    train_mask = np.zeros(N, dtype=bool)
    train_mask[:n_train] = True

    val_mask = np.zeros(N, dtype=bool)
    val_mask[n_train : n_train + n_val] = True

    # Split the canonical test nodes 50/50 into target_fit and target_eval
    shuffled = test_idx.copy()
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    fit_mask  = np.zeros(N, dtype=bool)
    eval_mask = np.zeros(N, dtype=bool)
    fit_mask[shuffled[:half]]  = True
    eval_mask[shuffled[half:]] = True

    return {
        "train":          train_mask,
        "val":            val_mask,
        "target_fit":     fit_mask,
        "target_eval":    eval_mask,
        "diagnostics":    {},
        "class_shortfall": {},
    }


# ---------------------------------------------------------------------------
# Private helper: minimal random split for CLI and tests.
# The real split function lives in splits.py (Lane B, P2-B).
# ---------------------------------------------------------------------------

def _simple_split(
    g,
    n_per_class: int = 20,
    n_val: int = 500,
    seed: int = 42,
) -> dict:
    """Random Planetoid-style split for CLI / unit tests only.

    train  : n_per_class labeled nodes per class (skips y==-1)
    val    : n_val labeled nodes from the non-train remainder
    target : all remaining nodes, split 50/50 into fit/eval
    """
    rng = np.random.default_rng(seed)
    N   = g.X.shape[0]
    y   = g.y

    train_mask = np.zeros(N, dtype=bool)
    for c in range(g.num_classes):
        cand = np.where(y == c)[0]
        if len(cand) == 0:
            continue
        chosen = rng.choice(cand, size=min(n_per_class, len(cand)), replace=False)
        train_mask[chosen] = True

    remaining_labeled = np.where((~train_mask) & (y >= 0))[0]
    rng.shuffle(remaining_labeled)
    val_idx = remaining_labeled[:n_val]
    val_mask = np.zeros(N, dtype=bool)
    val_mask[val_idx] = True

    target_mask = ~train_mask & ~val_mask
    target_idx  = np.where(target_mask)[0]
    rng.shuffle(target_idx)
    half = len(target_idx) // 2
    fit_mask  = np.zeros(N, dtype=bool)
    eval_mask = np.zeros(N, dtype=bool)
    fit_mask[target_idx[:half]]  = True
    eval_mask[target_idx[half:]] = True

    return {
        "train":       train_mask,
        "val":         val_mask,
        "target_fit":  fit_mask,
        "target_eval": eval_mask,
        "diagnostics": {},
        "class_shortfall": {},
    }


# ---------------------------------------------------------------------------
# CLI: python -m src.train --dataset cora --model gcn
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data import load_graph
    from src.metrics import (
        accuracy as _acc, ece as _ece, mce as _mce,
        nll as _nll, brier as _brier, avg_confidence,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--model",   default="gcn",
                        choices=["mlp", "gcn", "sgc"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--epochs",  type=int, default=300)
    args = parser.parse_args()

    print(f"Loading {args.dataset} …")
    g = load_graph(args.dataset)

    # CLI uses a random split for quick exploration.
    # The gate number (0.8140) comes from _planetoid_split; see test_gcn_cora_accuracy.
    split = _simple_split(g, seed=args.seed)
    n_train = int(split["train"].sum())
    n_val   = int(split["val"].sum())
    print(f"Split: train={n_train}  val={n_val}")

    print(f"Training {args.model.upper()} (seed={args.seed}, epochs={args.epochs}) …")
    result = train_model(g, split, args.model, seed=args.seed, epochs=args.epochs)

    logits = result["logits"]
    # Softmax
    logits_s = logits - logits.max(axis=1, keepdims=True)
    exp_ = np.exp(logits_s)
    probs = exp_ / exp_.sum(axis=1, keepdims=True)

    mask = split["target_eval"]

    acc_val  = _acc(probs, g.y, mask)
    nll_val  = _nll(probs, g.y, mask)
    brier_val = _brier(probs, g.y, mask)
    ece_u, _ = _ece(probs, g.y, mask, strategy="uniform")
    ece_q, _ = _ece(probs, g.y, mask, strategy="quantile")
    mce_val  = _mce(probs, g.y, mask)
    conf_val = avg_confidence(probs, mask)

    direction = "over-confident" if conf_val > acc_val else "under-confident"

    print(f"\n{'-'*48}")
    print(f"  Dataset        : {args.dataset}")
    print(f"  Model          : {args.model.upper()}")
    print(f"  Best epoch     : {result['best_epoch']}")
    print(f"  Val acc        : {result['val_acc']:.4f}")
    print(f"  Target-eval acc: {acc_val:.4f}")
    print(f"  NLL            : {nll_val:.4f}")
    print(f"  Brier          : {brier_val:.4f}")
    print(f"  ECE (uniform)  : {ece_u:.4f}")
    print(f"  ECE (quantile) : {ece_q:.4f}")
    print(f"  MCE            : {mce_val:.4f}")
    print(f"  Avg confidence : {conf_val:.4f}")
    print(f"  Calibration    : {direction}")
    print(f"  Train time     : {result['train_sec']:.1f}s")
    print(f"{'-'*48}")

    if acc_val < 0.78 and args.model == "gcn" and args.dataset == "cora":
        print(f"\n  WARNING: Cora GCN accuracy {acc_val:.4f} < 0.78 – reporting as deviation.")
