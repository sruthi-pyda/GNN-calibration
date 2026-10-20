"""tests/test_data.py – Dataset loader tests (Phase P1-A gate).

Gate checks:
  - Cora: exactly 2708 nodes / 5278 edges / 0.81 homophily / 78 components
  - Texas: exactly 183 / 279 / 0.06 / 1 component
  - All 6 graphs pass validate_graph
  - Second load comes from cache in under 1 second
  - make_csbm homophily within 0.03 of target for h=0.8 and h=0.2
"""
import time
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_graph, make_csbm, validate_graph


# ---------------------------------------------------------------------------
# Anchor datasets – must match exactly or the edge convention is wrong
# ---------------------------------------------------------------------------

def test_cora_exact():
    g = load_graph("cora")
    assert g.meta["n_nodes"] == 2708, f"Cora nodes: {g.meta['n_nodes']} != 2708"
    assert g.meta["n_edges_undirected"] == 5278, f"Cora edges: {g.meta['n_edges_undirected']} != 5278"
    assert g.meta["n_components"] == 78, f"Cora components: {g.meta['n_components']} != 78"
    assert abs(g.meta["edge_homophily"] - 0.81) <= 0.005, (
        f"Cora homophily: {g.meta['edge_homophily']:.4f} not within 0.005 of 0.81"
    )
    assert g.num_classes == 7
    assert g.meta["n_features"] == 1433


def test_texas_exact():
    g = load_graph("texas")
    assert g.meta["n_nodes"] == 183, f"Texas nodes: {g.meta['n_nodes']} != 183"
    assert g.meta["n_edges_undirected"] == 279, f"Texas edges: {g.meta['n_edges_undirected']} != 279"
    assert g.meta["n_components"] == 1, f"Texas components: {g.meta['n_components']} != 1"
    assert abs(g.meta["edge_homophily"] - 0.06) <= 0.01, (
        f"Texas homophily: {g.meta['edge_homophily']:.4f} not within 0.01 of 0.06"
    )
    assert g.num_classes == 5
    assert g.meta["n_features"] == 1703


# ---------------------------------------------------------------------------
# Reference checks – flag >0.03 deviation but don't fail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,n_nodes,n_edges,n_classes,h_ref", [
    ("citeseer", 3327, 4552, 6, 0.74),
    ("pubmed",   19717, 44324, 3, 0.80),
])
def test_homophilic_reference(name, n_nodes, n_edges, n_classes, h_ref):
    g = load_graph(name)
    validate_graph(g)
    assert g.meta["n_nodes"] == n_nodes, f"{name} nodes: {g.meta['n_nodes']} != {n_nodes}"
    assert g.meta["n_edges_undirected"] == n_edges, (
        f"{name} edges: {g.meta['n_edges_undirected']} != {n_edges}"
    )
    assert g.num_classes == n_classes
    h_dev = abs(g.meta["edge_homophily"] - h_ref)
    if h_dev > 0.03:
        pytest.warns(UserWarning, match="homophily deviation")
    # Don't fail – just report
    print(f"\n  {name} homophily={g.meta['edge_homophily']:.3f} (ref {h_ref}, dev {h_dev:.3f})")


@pytest.mark.parametrize("name,n_nodes_approx,h_ref", [
    ("chameleon", 890, 0.24),
    ("roman_empire", 22662, 0.05),
])
def test_heterophilic_reference(name, n_nodes_approx, h_ref):
    g = load_graph(name)
    validate_graph(g)
    assert g.meta["n_nodes"] == n_nodes_approx, (
        f"{name} nodes: {g.meta['n_nodes']} != {n_nodes_approx}"
    )
    h_dev = abs(g.meta["edge_homophily"] - h_ref)
    print(f"\n  {name} homophily={g.meta['edge_homophily']:.3f} (ref {h_ref}, dev {h_dev:.3f})")


# ---------------------------------------------------------------------------
# All 6 graphs pass validate_graph
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["cora", "citeseer", "pubmed", "texas", "chameleon", "roman_empire"])
def test_all_graphs_validate(name):
    g = load_graph(name)
    validate_graph(g)  # must not raise


# ---------------------------------------------------------------------------
# Cache speed: second load in under 1 second
# ---------------------------------------------------------------------------

def test_cache_speed():
    # First load already cached; time the second
    t0 = time.time()
    g = load_graph("cora", use_cache=True)
    elapsed = time.time() - t0
    assert elapsed < 1.0, f"Cache load took {elapsed:.2f}s (should be < 1s)"


# ---------------------------------------------------------------------------
# CSBM homophily within 0.03 of target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h_target", [0.8, 0.2])
def test_csbm_homophily(h_target):
    g = make_csbm(n=3000, n_classes=3, homophily=h_target, avg_degree=8, seed=42)
    validate_graph(g)
    h_actual = g.meta["edge_homophily"]
    assert abs(h_actual - h_target) <= 0.03, (
        f"CSBM h={h_target}: got {h_actual:.3f}, deviation {abs(h_actual-h_target):.3f} > 0.03"
    )
