"""src/baselines.py – CaGCN, ATC-TS, STAC-global calibration baselines.

Lane A (Sruthi).  Phase P2-C (implemented after P4-A).

Internal signatures:
  cagcn_val(logits, g, labels, split)                       -> T (N,) per-node
  atc_ts(logits, g, labels, split)                          -> (T, threshold, acc_hat)
  stac_global(logits, g, labels, split, mlp_logits=None)    -> T float

See STAC_NOTES.md for the stac_global approximation assumptions.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ── CaGCN (Wang et al. 2021) ──────────────────────────────────────────────────

class _CaGCNTemp(torch.nn.Module):
    """2-layer GCN mapping frozen logits -> positive per-node temperature."""

    def __init__(self, C: int, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.l1 = torch.nn.Linear(C, hidden, bias=False)
        self.l2 = torch.nn.Linear(hidden, 1, bias=False)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, X, A_hat):
        h = F.relu(A_hat @ self.l1(X))
        h = self.drop(h)
        return F.softplus(A_hat @ self.l2(h)).squeeze(-1)  # (N,) > 0


def cagcn_val(
    logits: np.ndarray,
    g,
    labels,
    split: dict,
    hidden: int = 64,
    lr: float = 0.01,
    wd: float = 5e-3,
    epochs: int = 1000,
    patience: int = 50,
) -> np.ndarray:
    """CaGCN: per-node temperature from a 2-layer GCN over frozen logits.

    LABELS='source_val'.  Returns T (N,) > 0.
    """
    from src.models import normalized_adj

    val_idx = np.where(split["val"])[0]
    y_raw = np.asarray(labels[val_idx])
    keep = y_raw >= 0
    val_idx, y_val = val_idx[keep], y_raw[keep]

    X = torch.tensor(logits, dtype=torch.float32)
    A_hat = normalized_adj(g)
    vi = torch.tensor(val_idx, dtype=torch.long)
    y_t = torch.tensor(y_val, dtype=torch.long)
    L_val = torch.tensor(logits[val_idx], dtype=torch.float32)  # frozen

    model = _CaGCNTemp(logits.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_loss, best_weights, no_imp = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        T_v = model(X, A_hat)[vi].clamp(min=1e-3).unsqueeze(-1)  # (n_val, 1)
        loss = F.cross_entropy(L_val / T_v, y_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            T_v = model(X, A_hat)[vi].clamp(min=1e-3).unsqueeze(-1)
            val_loss = F.cross_entropy(L_val / T_v, y_t).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_weights:
        model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        T_all = model(X, A_hat).clamp(min=1e-3).numpy()
    return T_all  # (N,)


# ── ATC-TS (Garg et al. 2022) ─────────────────────────────────────────────────

def atc_ts(logits: np.ndarray, g, labels, split: dict) -> tuple:
    """ATC-matched temperature scaling.

    1. Find threshold t on val: fraction of val above t equals val accuracy.
    2. acc_hat = fraction of target with max-conf > t.
    3. Bisect T so mean target confidence = acc_hat.

    LABELS='source_val'.  Returns (T: float, threshold: float, acc_hat: float).
    """
    from src.calibrate import apply_temperature, _bisect_T

    val_idx = np.where(split["val"])[0]
    y_raw = np.asarray(labels[val_idx])
    keep = y_raw >= 0
    val_idx, y_val = val_idx[keep], y_raw[keep]

    probs_val = apply_temperature(logits[val_idx], 1.0)
    conf_val = probs_val.max(axis=1)
    val_acc = float((probs_val.argmax(1) == y_val).mean())

    # t is the confidence at which the top-val_acc fraction of val nodes sits
    sorted_conf = np.sort(conf_val)[::-1]
    n_above = max(1, int(round(val_acc * len(sorted_conf))))
    t = float(sorted_conf[min(n_above - 1, len(sorted_conf) - 1)])

    target_mask = split["target_fit"] | split["target_eval"]
    conf_target = apply_temperature(logits[target_mask], 1.0).max(axis=1)
    acc_hat = float(np.clip((conf_target > t).mean(), 1.0 / logits.shape[1], 0.999))

    T, _ = _bisect_T(logits[target_mask], acc_hat)
    return float(T), float(t), float(acc_hat)


# ── STAC-global (Bahi 2026 special case) ──────────────────────────────────────

def stac_global(
    logits: np.ndarray, g, labels, split: dict, mlp_logits=None,
) -> float:
    """Single global T from MLP-GNN prediction agreement (label-free).

    See STAC_NOTES.md.  LABELS='none'.  Returns T: float.
    """
    from src.calibrate import apply_temperature, _bisect_T

    target_mask = split["target_fit"] | split["target_eval"]
    if mlp_logits is not None:
        mlp_pred = mlp_logits[target_mask].argmax(1)
        gnn_pred = logits[target_mask].argmax(1)
        acc_hat = float(np.clip(
            (mlp_pred == gnn_pred).mean(), 1.0 / logits.shape[1], 0.999,
        ))
    else:
        acc_hat = float(np.clip(
            apply_temperature(logits[target_mask], 1.0).max(1).mean(),
            1.0 / logits.shape[1], 0.999,
        ))

    T, _ = _bisect_T(logits[target_mask], acc_hat)
    return float(T)
