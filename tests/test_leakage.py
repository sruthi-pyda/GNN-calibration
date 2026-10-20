"""tests/test_leakage.py – Leakage guard tests for all label-free methods.

Every calibrator declaring LABELS="none" must not raise LabelLeakError
when called with allowed_mask all False on the target.
Implemented in Phase P4-A once calibrate.py and baselines.py are filled in.
"""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labels import GuardedLabels, LabelLeakError


def test_guarded_all_false_raises_on_any_index():
    """Baseline guard test: all-False mask raises for any index access."""
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    mask = np.zeros(5, dtype=bool)
    gl = GuardedLabels(y, mask)
    for i in range(5):
        with pytest.raises(LabelLeakError):
            _ = gl[i]


# Placeholder – filled in Phase P4-A
# def test_lf_cts_no_leakage(): ...
# def test_lf_ts_no_leakage(): ...
# def test_atc_ts_no_leakage(): ...
# def test_stac_no_leakage(): ...
