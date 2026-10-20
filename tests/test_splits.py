"""tests/test_splits.py - Shift splits and diagnostics (Phase P2-B).

Gate checks:
  - random and structural splits have identical train sizes
  - every class is present in train, on all graphs and all 5 seeds
  - on Cora, structural has higher class TV distance and lower cross-region edge fraction
  - target_fit and target_eval are disjoint and together cover the target region
  - a structural split of Texas raises
"""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import make_csbm, make_synthetic_graph
from src.splits import concat_graphs, make_split

MASKS = ("train", "val", "target_fit", "target_eval")


@pytest.fixture(scope="module")
def fixture_graph():
    return make_synthetic_graph(n=1500, n_classes=3, homophily=0.8, seed=42)


def _check_masks(g, split):
    N = len(g.y)
    for m in MASKS:
        assert split[m].dtype == bool and split[m].shape == (N,)
    target = split["target_fit"] | split["target_eval"]
    assert not np.any(split["target_fit"] & split["target_eval"])
    assert not np.any(split["train"] & split["val"])
    assert not np.any((split["train"] | split["val"]) & target)
    assert abs(int(split["target_fit"].sum()) - int(split["target_eval"].sum())) <= 1
    assert set(np.unique(g.y[split["train"]])) == set(range(g.num_classes))


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["random", "structural", "degree"])
def test_masks_valid(fixture_graph, kind):
    _check_masks(fixture_graph, make_split(fixture_graph, kind, seed=0, n_val=200))


def test_equal_train_sizes(fixture_graph):
    sizes = {k: int(make_split(fixture_graph, k, seed=1, n_val=200)["train"].sum())
             for k in ("random", "structural", "degree")}
    assert len(set(sizes.values())) == 1, sizes


def test_random_target_is_every_node_outside_train_and_val(fixture_graph):
    # Unlike the benchmark-only _planetoid_split (1,000 test nodes on Cora), the production
    # random split must leave no node unassigned.
    for seed in range(5):
        s = make_split(fixture_graph, "random", seed=seed, n_val=200)
        target = s["target_fit"] | s["target_eval"]
        np.testing.assert_array_equal(target, ~(s["train"] | s["val"]))


def test_target_is_complement_of_region(fixture_graph):
    s = make_split(fixture_graph, "structural", seed=0, n_val=200)
    d = s["diagnostics"]
    assert d["n_source_region"] + d["n_target"] == len(fixture_graph.y)
    assert 0.4 <= d["source_region_frac"] <= 0.6


def test_same_seed_same_masks(fixture_graph):
    a = make_split(fixture_graph, "structural", seed=3, n_val=200)
    b = make_split(fixture_graph, "structural", seed=3, n_val=200)
    c = make_split(fixture_graph, "structural", seed=4, n_val=200)
    for m in MASKS:
        np.testing.assert_array_equal(a[m], b[m])
    assert any(not np.array_equal(a[m], c[m]) for m in MASKS)


def test_unknown_kind_raises(fixture_graph):
    with pytest.raises(ValueError):
        make_split(fixture_graph, "temporal")


def test_csbm_shift_target_is_target_graph():
    src = make_csbm(n=600, homophily=0.8, seed=0)
    tgt = make_csbm(n=600, homophily=0.2, seed=1)
    g = concat_graphs(src, tgt)
    s = make_split(g, "csbm_shift", seed=0, n_val=100)
    _check_masks(g, s)
    target = s["target_fit"] | s["target_eval"]
    assert target[600:].all() and not target[:600].any()
    assert s["diagnostics"]["cross_region_edge_frac"] == 0.0
    with pytest.raises(ValueError):
        make_split(src, "csbm_shift")


# ---------------------------------------------------------------------------
# Real data (skipped if the download fails)
# ---------------------------------------------------------------------------

def _load(name):
    from src.data import load_graph
    try:
        return load_graph(name)
    except Exception as e:  # noqa: BLE001 - network or mirror failure
        pytest.skip(f"cannot load {name}: {e}")


def test_texas_structural_raises():
    with pytest.raises(ValueError):
        make_split(_load("texas"), "structural")


def test_cora_structural_is_a_real_shift():
    g = _load("cora")
    rnd = [make_split(g, "random", seed=s)["diagnostics"] for s in range(5)]
    st = [make_split(g, "structural", seed=s)["diagnostics"] for s in range(5)]
    mean = lambda ds, k: np.mean([d[k] for d in ds])  # noqa: E731
    assert mean(st, "class_tv_distance") > mean(rnd, "class_tv_distance")
    assert mean(st, "cross_region_edge_frac") < mean(rnd, "cross_region_edge_frac")


@pytest.mark.parametrize("name", ["cora", "citeseer", "chameleon", "roman_empire"])
def test_class_coverage_and_train_size(name):
    g = _load(name)
    for seed in range(5):
        r = make_split(g, "random", seed=seed)
        s = make_split(g, "structural", seed=seed)
        _check_masks(g, r)
        _check_masks(g, s)
        assert s["train"].sum() == r["train"].sum()
        r_target = r["target_fit"] | r["target_eval"]
        np.testing.assert_array_equal(r_target, ~(r["train"] | r["val"]))
    if name == "cora":
        assert int(r_target.sum()) == 2708 - 140 - 500
