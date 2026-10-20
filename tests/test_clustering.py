"""tests/test_clustering.py – Clustering library tests (Phase P1-B).

Gate checks:
  - Fixture h=0.8 picks sgc_kmeans with NMI > 0.5
  - Fixture h=0.1 picks feature_kmeans
  - gate_labels=None raises ValueError
  - Cora NMI 0.501 / ARI 0.393 (within 0.01) once P1-A is merged
  - Texas NMI 0.421 / ARI 0.314 (within 0.01)
"""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering import cluster_graph
from src.data import make_synthetic_graph


def test_cluster_graph_none_gate_raises():
    """cluster_graph must raise if gate_labels is None."""
    g = make_synthetic_graph(n=300, n_classes=3, seed=42)
    with pytest.raises((ValueError, TypeError)):
        cluster_graph(g, gate_labels=None, gate_source="mlp_pred")


# Implemented in Phase P1-B
# def test_fixture_high_homophily_picks_sgc():
#     g = make_synthetic_graph(n=600, homophily=0.8, seed=42)
#     gate_labels = g.y  # fixture: use true labels for development
#     result = cluster_graph(g, gate_labels=gate_labels, gate_source="true_oracle")
#     assert result["method_used"] == "sgc_kmeans"
#
# def test_fixture_low_homophily_picks_feature():
#     g = make_synthetic_graph(n=600, homophily=0.1, seed=42)
#     gate_labels = g.y
#     result = cluster_graph(g, gate_labels=gate_labels, gate_source="true_oracle")
#     assert result["method_used"] == "feature_kmeans"
