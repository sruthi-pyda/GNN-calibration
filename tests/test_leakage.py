"""tests/test_leakage.py – Label-leakage guard tests for all CALIBRATORS entries.

For every calibrator registered in CALIBRATORS:
  1. Label-free methods (LABELS='none') must work with an all-False GuardedLabels
     without raising LabelLeakError.
  2. Every calibrator must produce bit-identical output when g.y is poisoned
     outside its declared allowed mask.  This catches any direct ctx["g"].y
     reads that bypass GuardedLabels.
"""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from src.data import make_synthetic_graph
from src.labels import GuardedLabels, LabelLeakError
from src.run_experiments import CALIBRATORS, _label_mask
from src.splits import make_split


# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def leak_ctx():
    """Minimal ctx for CALIBRATORS dispatch tests (n=700 matches other fixtures)."""
    g = make_synthetic_graph(n=700, n_classes=3, homophily=0.8, seed=0)
    split = make_split(g, "random", seed=0)
    rng = np.random.default_rng(0)
    N, C = g.X.shape[0], 3
    logits = rng.normal(size=(N, C)).astype(np.float32)
    mlp_logits = rng.normal(size=(N, C)).astype(np.float32)
    clusters = np.tile(np.arange(C), N // C + 1)[:N].astype(np.int64)
    embedding = rng.normal(size=(N, 16)).astype(np.float32)
    cl = {
        "h_hat": 0.8,
        "embedding": embedding,
        "method_used": "sgc_kmeans",
        "embedding2d": rng.normal(size=(N, 2)).astype(np.float32),
    }
    ctx = dict(g=g, split=split, clusters=clusters, cl=cl,
               mlp_logits=mlp_logits, proxy_variant="edge_knn")
    return g, split, logits, C, ctx


def _poison_outside(y, allowed_mask, n_classes):
    """Rotate labels by +1 for every node NOT in allowed_mask."""
    yp = y.copy()
    yp[~allowed_mask] = (yp[~allowed_mask] + 1) % n_classes
    return yp


# ── 1. baseline guard ─────────────────────────────────────────────────────────

def test_guarded_all_false_raises_on_any_index():
    """all-False GuardedLabels must raise LabelLeakError for any scalar index."""
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    mask = np.zeros(5, dtype=bool)
    gl = GuardedLabels(y, mask)
    for i in range(5):
        with pytest.raises(LabelLeakError):
            _ = gl[i]


# ── 2. label-free calibrators accept all-False guard ─────────────────────────

_LABEL_FREE = [k for k, (decl, _) in CALIBRATORS.items() if decl == "none"]


@pytest.mark.parametrize("name", _LABEL_FREE)
def test_label_free_calibrators_work_with_all_false_guard(name, leak_ctx):
    """LABELS='none' calibrators must not touch GuardedLabels (no LabelLeakError)."""
    g, split, logits, C, ctx = leak_ctx
    all_false = np.zeros(g.X.shape[0], dtype=bool)
    labels = GuardedLabels(g.y, all_false)
    _, fn = CALIBRATORS[name]
    probs, _, _ = fn(logits, labels, ctx)   # must not raise
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


# ── 3. every calibrator is invariant to g.y poison outside its declared mask ──

@pytest.mark.parametrize("name", list(CALIBRATORS))
def test_calibrator_g_y_loophole(name, leak_ctx):
    """Poisoning g.y outside the declared allowed mask must not change output.

    Calibrators must read labels only through GuardedLabels, never via ctx['g'].y
    directly.  cagcn_val uses g for normalized_adj (edge_index only); lf_cts_plus
    uses g for edge_index in _feats; all other ctx['g'] uses are structure-only.
    """
    g, split, logits, C, ctx = leak_ctx
    declared, fn = CALIBRATORS[name]
    allowed = _label_mask(split, declared)

    # Rotate all labels outside the allowed mask by +1 class.
    if allowed.any():
        yp = _poison_outside(g.y, allowed, C)
    else:  # LABELS='none': rotate every label (method must not read any)
        yp = (g.y + 1) % C

    g_poison = copy.copy(g)
    g_poison.y = yp
    ctx_poison = {**ctx, "g": g_poison}

    labels_clean  = GuardedLabels(g.y, allowed)
    labels_poison = GuardedLabels(yp,  allowed)

    # Seed torch before each call so that any non-determinism from model init
    # or dropout (e.g. cagcn_val) does not cause spurious failures — the only
    # difference between calls must be from label content, not RNG state.
    torch.manual_seed(0)
    p_c, t_c, _ = fn(logits, labels_clean,  ctx)
    torch.manual_seed(0)
    p_p, t_p, _ = fn(logits, labels_poison, ctx_poison)

    np.testing.assert_array_equal(
        p_c, p_p,
        err_msg=(
            f"calibrator '{name}' output changes when g.y is poisoned outside "
            f"its '{declared}' mask — possible direct g.y read bypassing GuardedLabels"
        ),
    )
    assert t_c == t_p, (
        f"calibrator '{name}' temps dict changes when g.y is poisoned"
    )
