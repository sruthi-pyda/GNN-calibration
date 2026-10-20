"""src/baselines.py – Published calibration baselines: CaGCN, ATC-TS, STAC.

Lane C (Smitha) owns this file, or Lane A after P4-A.
Phase P2-C.

Each method is a function:
    fit(logits, g, labels: GuardedLabels, masks) -> per-node temperature vector

and declares LABELS = "source_val" or "none".
"""
from __future__ import annotations

import numpy as np

# LABELS declarations (each function overrides)
# cagcn_val  : "source_val"
# atc_ts     : "source_val" (threshold from val, none on target)
# stac       : "none"


def cagcn_val(logits: np.ndarray, g, labels, split: dict) -> np.ndarray:
    """CaGCN (Wang et al. 2021) – 2-layer GCN over frozen logits, per-node temperature.

    LABELS = "source_val".
    """
    raise NotImplementedError("Implemented in Phase P2-C")


def atc_ts(logits: np.ndarray, g, labels, split: dict) -> np.ndarray:
    """ATC-matched TS (Garg et al. 2022) – threshold from val, acc_hat on target.

    LABELS = "source_val" for threshold; none on target.
    """
    raise NotImplementedError("Implemented in Phase P2-C")


def stac(logits: np.ndarray, g, labels, split: dict) -> np.ndarray:
    """STAC (Bahi 2026, arXiv 2607.10804).

    See STAC_NOTES.md for implementation assumptions.
    LABELS = "none".
    """
    raise NotImplementedError("Implemented in Phase P2-C")


def stac_global(logits: np.ndarray, g, labels, split: dict) -> np.ndarray:
    """Fallback: single global T without target labels (STAC special case).

    LABELS = "none".
    """
    raise NotImplementedError("Implemented in Phase P2-C")
