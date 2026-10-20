"""src/labels.py – Label access guard and leakage audit.

Lane: joint (P0). Do not modify without agreement from both lanes.
"""
from __future__ import annotations

import threading
from typing import Any

_audit_log: list[dict] = []
_audit_lock = threading.Lock()


class LabelLeakError(Exception):
    """Raised when a calibrator reads labels it is not allowed to see."""


class GuardedLabels:
    """Wraps a label array so that reads outside `allowed_mask` raise LabelLeakError.

    Parameters
    ----------
    y : array-like of int, shape (N,)
        Full label array.  -1 = unlabelled.
    allowed_mask : array-like of bool, shape (N,)
        True for indices this guard permits reading.
    """

    def __init__(self, y, allowed_mask):
        import numpy as np
        self._y = np.asarray(y, dtype=np.int64)
        self._mask = np.asarray(allowed_mask, dtype=bool)
        if self._y.shape != self._mask.shape:
            raise ValueError(
                f"y shape {self._y.shape} != allowed_mask shape {self._mask.shape}"
            )

    def __getitem__(self, idx):
        import numpy as np
        idx_arr = np.asarray(idx)
        if idx_arr.ndim == 0:
            idx_arr = idx_arr.reshape(1)
        if not np.all(self._mask[idx_arr]):
            raise LabelLeakError(
                f"Label read attempted outside allowed_mask at indices {idx_arr[~self._mask[idx_arr]]}"
            )
        return self._y[idx]

    @property
    def allowed_mask(self):
        import numpy as np
        return self._mask.copy()

    def __len__(self):
        return len(self._y)

    # Prevent accidental full-array access
    def __array__(self, dtype=None):
        raise LabelLeakError(
            "GuardedLabels cannot be converted to a numpy array directly. "
            "Access individual indices through __getitem__."
        )

    def __repr__(self):
        n_allowed = int(self._mask.sum())
        return f"GuardedLabels(N={len(self._y)}, allowed={n_allowed})"


def audit_label_use(method_name: str, declared: str) -> None:
    """Record which calibration method used which label set.

    Parameters
    ----------
    method_name : str
        The calibrator key, e.g. "lf_cts".
    declared : str
        One of "none", "source_val", "target_fit".
    """
    assert declared in ("none", "source_val", "target_fit"), (
        f"Invalid declared label usage: {declared!r}"
    )
    with _audit_lock:
        _audit_log.append({"method": method_name, "labels_used": declared})


def get_audit_log() -> list[dict]:
    """Return a copy of the audit log."""
    with _audit_lock:
        return list(_audit_log)


def clear_audit_log() -> None:
    with _audit_lock:
        _audit_log.clear()
