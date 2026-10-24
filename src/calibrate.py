"""src/calibrate.py – Calibration methods: TS, LF-TS, LF-CTS, LF-CTS+, oracles.

Lane A (Sruthi) owns this file.  Phase P4-A.

LABELS='none'        – no labels used
LABELS='source_val'  – reads split["val"] only
LABELS='target_fit'  – reads split["target_fit"] only (oracle ceiling)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors


# ─── core ─────────────────────────────────────────────────────────────────────

def apply_temperature(logits: np.ndarray, T) -> np.ndarray:
    """Divide logits by T (scalar or per-node (N,)) and return float32 softmax."""
    z = logits / (float(T) if np.isscalar(T) else np.asarray(T, dtype=float)[:, None])
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def _fit_T(logits_sub: np.ndarray, y_sub: np.ndarray) -> float:
    """Minimise NLL on a (logits, y) subset; returns scalar T."""
    def obj(log_T: float) -> float:
        p = apply_temperature(logits_sub, np.exp(log_T))
        return float(-np.log(np.clip(p[np.arange(len(y_sub)), y_sub], 1e-12, 1)).mean())
    return float(np.exp(
        minimize_scalar(obj, bounds=(np.log(0.05), np.log(20.0)), method="bounded").x
    ))


def _bisect_T(logits_sub: np.ndarray, target: float,
              lo: float = 0.05, hi: float = 10.0) -> tuple:
    """Find T so mean(max-softmax) == target.  Returns (T, clamped: bool)."""
    def conf(T: float) -> float:
        return float(apply_temperature(logits_sub, T).max(axis=1).mean())
    if conf(lo) <= target:
        return lo, True
    if conf(hi) >= target:
        return hi, True
    for _ in range(60):
        mid = (lo + hi) / 2
        if conf(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, False


# ─── oracle calibrators ───────────────────────────────────────────────────────

def ts_val(logits: np.ndarray, labels, split: dict) -> tuple:
    """Global TS minimising NLL on validation.  LABELS='source_val'."""
    idx = np.where(split["val"])[0]
    y = np.asarray(labels[idx])
    keep = y >= 0
    T = _fit_T(logits[idx[keep]], y[keep])
    return apply_temperature(logits, T), {"global": round(T, 4)}


def ts_target_oracle(logits: np.ndarray, labels, split: dict) -> tuple:
    """Global TS on target_fit.  LABELS='target_fit'.  Never touches target_eval."""
    idx = np.where(split["target_fit"])[0]
    y = np.asarray(labels[idx])
    keep = y >= 0
    T = _fit_T(logits[idx[keep]], y[keep])
    return apply_temperature(logits, T), {"global": round(T, 4)}


def cluster_ts_target_oracle(
    logits: np.ndarray, labels, split: dict, clusters: np.ndarray, min_fit: int = 20,
) -> tuple:
    """Per-cluster TS on target_fit; clusters < min_fit nodes use global T.
    LABELS='target_fit'.  Returns (probs, temps, per_cluster_calib)."""
    fit_idx = np.where(split["target_fit"])[0]
    y_all = np.asarray(labels[fit_idx])
    keep = y_all >= 0
    fit_idx, y_fit = fit_idx[keep], y_all[keep]
    cl_fit = clusters[fit_idx]
    T_g = _fit_T(logits[fit_idx], y_fit)
    K = int(clusters.max()) + 1
    per_T = np.full(logits.shape[0], T_g, dtype=float)
    temps: dict = {"_global": round(T_g, 4)}
    per_cluster: list = []
    for c in range(K):
        m = cl_fit == c
        n_fit = int(m.sum())
        used_global = n_fit < min_fit
        Tc = T_g if used_global else _fit_T(logits[fit_idx[m]], y_fit[m])
        temps[str(c)] = round(Tc, 4)
        per_T[clusters == c] = Tc
        per_cluster.append({"cluster_id": c, "T": round(Tc, 4),
                             "n_fit": n_fit, "used_global_T": used_global})
    return apply_temperature(logits, per_T), temps, per_cluster


# ─── label-free proxy ─────────────────────────────────────────────────────────

def label_free_proxy(
    g, logits: np.ndarray, clusters: np.ndarray, h_hat: float,
    embedding: np.ndarray, k_nn: int = 10,
) -> np.ndarray:
    """Per-cluster label-free accuracy proxy.  LABELS='none'.

    h_hat > 0.5: intra-cluster edge-prediction agreement.
    h_hat <= 0.5: k-NN agreement in SVD embedding restricted to cluster.
    < 20 edges/pairs -> graph-wide fallback.  Returns a_hat (K,) in [1/C, 0.999].
    """
    C = logits.shape[1]
    preds = logits.argmax(axis=1)
    K = int(clusters.max()) + 1

    if h_hat > 0.5:
        src, dst = g.edge_index
        intra = clusters[src] == clusters[dst]
        s_i, d_i = src[intra], dst[intra]
        ag_all = preds[s_i] == preds[d_i]
        gw = float(ag_all.mean()) if len(ag_all) >= 20 else 1.0 / C
        a = np.array([
            float(ag_all[clusters[s_i] == c].mean())
            if (clusters[s_i] == c).sum() >= 20 else gw
            for c in range(K)
        ])
    else:
        nn = NearestNeighbors(n_neighbors=min(k_nn + 1, len(preds))).fit(embedding)
        _, nbs = nn.kneighbors(embedding)
        nbs = nbs[:, 1:]
        gw = float((preds[nbs] == preds[:, None]).mean())
        rows = []
        for c in range(K):
            ci = np.where(clusters == c)[0]
            k = min(k_nn, len(ci) - 1)
            if k < 1:
                rows.append(gw)
                continue
            _, nb_c = NearestNeighbors(n_neighbors=k + 1).fit(embedding[ci]).kneighbors(embedding[ci])
            ag = preds[ci[nb_c[:, 1:]]] == preds[ci, None]
            rows.append(float(ag.mean()) if ag.size >= 20 else gw)
        a = np.array(rows)

    return np.clip(a, 1.0 / C, 0.999)


# ─── LF calibrators ───────────────────────────────────────────────────────────

def lf_ts(
    logits: np.ndarray, a_hat: np.ndarray, clusters: np.ndarray, target_mask: np.ndarray,
) -> tuple:
    """Single T matching cluster-size-weighted mean(a_hat).  LABELS='none'."""
    sizes = np.bincount(clusters[target_mask], minlength=len(a_hat))
    gw = float(np.average(a_hat, weights=np.maximum(sizes, 1)))
    T, clamped = _bisect_T(logits[target_mask], gw)
    return apply_temperature(logits, T), {"global": round(T, 4), "_clamped": clamped}


def lf_cts(
    logits: np.ndarray, a_hat: np.ndarray, clusters: np.ndarray, target_mask: np.ndarray,
) -> tuple:
    """Per-cluster T via bisection on a_hat[c].  LABELS='none'.
    Returns (probs, temps, per_cluster_calib)."""
    K = len(a_hat)
    per_T = np.ones(logits.shape[0])
    temps: dict = {}
    per_cluster: list = []
    for c in range(K):
        ct = target_mask & (clusters == c)
        n_t = int(ct.sum())
        T_c, clamped = _bisect_T(logits[ct], float(a_hat[c])) if n_t > 0 else (1.0, False)
        per_T[clusters == c] = T_c
        temps[str(c)] = round(T_c, 4)
        per_cluster.append({"cluster_id": c, "a_hat": round(float(a_hat[c]), 4),
                             "T": round(T_c, 4), "clamped": clamped, "n_target": n_t})
    return apply_temperature(logits, per_T), temps, per_cluster


def lf_cts_plus(
    logits: np.ndarray, labels, split: dict, clusters: np.ndarray,
    h_hat: float, embedding: np.ndarray, g, k_nn: int = 10, n_pseudo: int = 200,
) -> tuple:
    """Proxy-bias-corrected LF-CTS using val labels only.  LABELS='source_val'.
    Returns (probs, temps, per_cluster_calib, ridge_r2)."""
    a_hat = label_free_proxy(g, logits, clusters, h_hat, embedding, k_nn)
    C, K = logits.shape[1], len(a_hat)
    preds = logits.argmax(axis=1)
    probs0 = apply_temperature(logits, 1.0)
    src_e, dst_e = g.edge_index
    deg = np.bincount(src_e, minlength=logits.shape[0]).astype(float)

    val_idx = np.where(split["val"])[0]
    y_raw = np.asarray(labels[val_idx])
    keep = y_raw >= 0
    val_idx, y_val = val_idx[keep], y_raw[keep]
    y_full = np.full(logits.shape[0], -1, dtype=np.int64)
    y_full[val_idx] = y_val
    by_c = {c: val_idx[clusters[val_idx] == c] for c in range(K)}
    avail = [c for c, ns in by_c.items() if len(ns) > 0]

    def _feats(ni: np.ndarray, cids: list) -> list:
        p = probs0[ni]
        ie = np.isin(src_e, ni) & np.isin(dst_e, ni)
        lh = float((preds[src_e[ie]] == preds[dst_e[ie]]).mean()) if ie.sum() >= 5 else float(h_hat)
        # log-size: pseudo-targets (~tens of nodes) and target clusters (~hundreds) are
        # on very different scales; log keeps training and prediction in the same range.
        return [float(np.mean([a_hat[c] for c in cids])), float(np.log1p(len(ni))),
                float(deg[ni].mean()), float(p.max(1).mean()),
                float(-(p * np.log(np.clip(p, 1e-12, 1))).sum(1).mean()), lh]

    rng = np.random.default_rng(42)
    Xs, ys = [], []
    for _ in range(n_pseudo):
        if len(avail) >= 2 and rng.random() > 0.5:
            c1, c2 = rng.choice(avail, 2, replace=False)
            ni = np.concatenate([by_c[int(c1)], by_c[int(c2)]])
            cids = [int(c1), int(c2)]
        else:
            c = int(rng.choice(avail))
            ni, cids = by_c[c], [c]
        yn = y_full[ni]
        ok = yn >= 0
        if ok.sum() == 0:
            continue
        Xs.append(_feats(ni, cids))
        ys.append(float((preds[ni[ok]] == yn[ok]).mean()))

    ridge, r2 = None, float("nan")
    if len(Xs) >= 5:
        X_arr, y_arr = np.array(Xs), np.array(ys)
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=min(5, len(y_arr)), scoring="r2")
        ridge.fit(X_arr, y_arr)
        r2 = float(getattr(ridge, "best_score_", float("nan")))

    target_mask = split["target_fit"] | split["target_eval"]
    a_tilde = a_hat.copy()
    if ridge is not None:
        for c in range(K):
            ni = np.where(target_mask & (clusters == c))[0]
            if len(ni):
                a_tilde[c] = float(np.clip(ridge.predict([_feats(ni, [c])])[0], 1.0 / C, 0.999))

    probs, temps, per_cluster = lf_cts(logits, a_tilde, clusters, target_mask)
    for row in per_cluster:
        c = row["cluster_id"]
        row["a_tilde"] = round(float(a_tilde[c]), 4)
        row["a_hat_raw"] = round(float(a_hat[c]), 4)
    return probs, temps, per_cluster, r2


# ─── evaluation ───────────────────────────────────────────────────────────────

def proxy_quality(
    probs: np.ndarray, a_hat: np.ndarray, clusters: np.ndarray,
    y: np.ndarray, target_eval: np.ndarray,
) -> tuple:
    """Per-cluster proxy vs true accuracy.  EVALUATION ONLY — reads y.
    Returns (DataFrame[cluster_id, a_hat, true_acc, n_eval], pearson_r)."""
    K = len(a_hat)
    rows = []
    for c in range(K):
        m = target_eval & (clusters == c)
        n = int(m.sum())
        ta = float((probs[m].argmax(1) == y[m]).mean()) if n > 0 else float("nan")
        rows.append({"cluster_id": c, "a_hat": round(float(a_hat[c]), 4),
                     "true_acc": round(ta, 4) if n > 0 else float("nan"), "n_eval": n})
    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["true_acc"])
    r = float(pearsonr(valid["a_hat"], valid["true_acc"])[0]) if len(valid) >= 2 else float("nan")
    return df, r
