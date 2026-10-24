"""tests/test_dashboard.py - Dashboard loaders and C1-C5 scorecard rules (Phase P1-C).

Gate checks:
  - A missing results file gives None (friendly message), not an exception
  - Fixture rows are detected as fixture
  - Each scorecard rule returns pass / partial / not supported / not yet measured as documented
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from dashboard.data import SOURCES, dataset_table, is_fixture, load_results
from dashboard.scorecard import (
    NOT_SUPPORTED, NOT_YET, PARTIAL, PASS, check_c1, check_c2, check_c3, check_c4, check_c5,
    scorecard,
)


def _row(dataset="cora", split="structural", calib="none", seed=0, ece=0.1,
         worst=0.15, ablation="main"):
    return dict(dataset=dataset, split=split, calib_method=calib, seed=seed, ece=ece,
                cluster_ece_worst=worst, ablation=ablation)


def _df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def test_missing_and_empty_files_return_none(tmp_path):
    assert load_results(tmp_path / "nope.csv") is None
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert load_results(empty) is None


def test_fixture_csv_is_detected():
    df = load_results(SOURCES["fixture"])
    assert df is not None and is_fixture(df)
    assert not is_fixture(_df([_row()]).assign(git_sha="abc123"))


def test_dataset_table_has_all_graphs_and_a_source():
    t = dataset_table()
    assert len(t) == 6 and set(t["source"]) <= {"cache", "plan reference"}


# ---------------------------------------------------------------------------
# Scorecard rules
# ---------------------------------------------------------------------------

def test_c1_pass_partial_not_yet():
    seeds = range(3)
    big = [_row(split="random", seed=s, ece=0.05 + 0.001 * s) for s in seeds] + \
          [_row(split="structural", seed=s, ece=0.15 + 0.001 * s) for s in seeds]
    assert check_c1(_df(big))[0] == PASS
    one_seed = [_row(split="random", ece=0.05), _row(split="structural", ece=0.10)]
    assert check_c1(_df(one_seed))[0] == PARTIAL
    reversed_ = [_row(split="random", ece=0.10), _row(split="structural", ece=0.05)]
    assert check_c1(_df(reversed_))[0] == NOT_SUPPORTED
    assert check_c1(_df([_row(split="random")]))[0] == NOT_YET


def test_c2_thresholds():
    assert check_c2(_df([_row(ece=0.1, worst=0.25)]))[0] == PASS
    assert check_c2(_df([_row(ece=0.1, worst=0.16)]))[0] == PARTIAL
    assert check_c2(_df([_row(ece=0.1, worst=0.12)]))[0] == NOT_SUPPORTED
    assert check_c2(_df([_row(calib="ts_val")]))[0] == NOT_YET


def test_c3_gap_closed():
    def cell(lf):
        return _df([_row(calib="none", ece=0.20), _row(calib="lf_cts", ece=lf),
                    _row(calib="cluster_ts_target_oracle", ece=0.10)])
    assert check_c3(cell(0.12))[0] == PASS          # 80% closed
    assert check_c3(cell(0.18))[0] == PARTIAL       # 20% closed
    assert check_c3(cell(0.25))[0] == NOT_SUPPORTED  # worse than none
    assert check_c3(_df([_row(calib="none")]))[0] == NOT_YET


def test_c4_needs_gate_ablation():
    gated = _row(calib="lf_cts", ece=0.05)
    assert check_c4(_df([gated]))[0] == NOT_YET
    assert check_c4(_df([gated, _row(calib="lf_cts", ece=0.09, ablation="gate")]))[0] == PASS


def test_c5_ours_vs_label_free():
    rows = [_row(calib="lf_cts", ece=0.05), _row(calib="lf_ts", ece=0.08),
            _row(dataset="citeseer", calib="lf_cts", ece=0.09),
            _row(dataset="citeseer", calib="atc_ts", ece=0.07)]
    assert check_c5(_df(rows[:2]))[0] == PASS
    assert check_c5(_df(rows))[0] == PARTIAL
    assert check_c5(_df([_row(calib="lf_cts")]))[0] == NOT_YET
    # the P5 sweep names STAC "stac_global"; it must count as a label-free baseline
    stac = [_row(calib="lf_cts", ece=0.05), _row(calib="stac_global", ece=0.03)]
    assert check_c5(_df(stac))[0] == NOT_SUPPORTED


def test_scorecard_runs_on_fixture():
    card = scorecard(load_results(SOURCES["fixture"]))
    assert list(card["claim"]) == ["C1", "C2", "C3", "C4", "C5"]
    # the fixture has no gate ablation or label-free baseline rows
    assert card.set_index("claim").loc[["C4", "C5"], "status"].eq(NOT_YET).all()
