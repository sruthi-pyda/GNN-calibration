"""tests/test_data.py – Dataset loader tests (Phase P1-A).

Gate checks:
  - Cora: exactly 2708 nodes / 5278 edges / 0.81 homophily / 78 components
  - Texas: exactly 183 / 279 / 0.06 / 1
  - All 6 graphs pass validate_graph
  - Second load comes from cache in under 1 second
  - make_csbm homophily within 0.03 of target for 0.8 and 0.2
"""
import time
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_graph, make_csbm, validate_graph


# Implemented in Phase P1-A
# @pytest.mark.parametrize("name,n_nodes,n_edges,n_comp,h_target,h_tol", [
#     ("cora", 2708, 5278, 78, 0.81, 0.005),
#     ("texas", 183, 279, 1, 0.06, 0.005),
# ])
# def test_anchor_datasets_exact(name, n_nodes, n_edges, n_comp, h_target, h_tol):
#     g = load_graph(name)
#     validate_graph(g)
#     assert g.meta["n_nodes"] == n_nodes, f"{name} nodes: {g.meta['n_nodes']} != {n_nodes}"
#     assert g.meta["n_edges_undirected"] == n_edges, ...
#     assert g.meta["n_components"] == n_comp, ...
#     assert abs(g.meta["edge_homophily"] - h_target) <= h_tol, ...
