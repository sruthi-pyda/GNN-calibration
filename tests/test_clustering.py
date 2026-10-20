"""tests/test_clustering.py - Clustering library tests (Phase P1-B).

Gate checks:
  - Fixture h=0.8 picks sgc_kmeans with NMI > 0.5
  - Fixture h=0.1 picks feature_kmeans
  - gate_labels=None raises ValueError
  - Cora NMI 0.501 / ARI 0.393 (within 0.01)
  - Texas NMI 0.421 / ARI 0.314 (within 0.01)
  - Hop sweep rises on Cora and falls on Texas
  - Cora profile shows a purity ~0.44 cluster

Fixture tests pass the fixture's true labels with gate_source="true_oracle"; that is the only
legitimate use of true labels for the gate (CLAUDE.md rule 2).
"""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import normalized_mutual_info_score

from src.clustering import (
    _embedding_for, _kmeans, cluster_graph, cluster_metrics, cluster_profile,
    estimate_homophily, louvain_clusters,
)
from src.data import _finalise, make_synthetic_graph


@pytest.fixture(scope="module")
def fixture_hi():
    return make_synthetic_graph(n=600, n_classes=3, homophily=0.8, seed=42)


@pytest.fixture(scope="module")
def fixture_lo():
    return make_synthetic_graph(n=600, n_classes=3, homophily=0.1, seed=42)


def _two_triangles():
    """Two triangles {0,1,2} and {3,4,5} joined by the single edge 2-3."""
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    ei = np.array(edges, dtype=np.int64).T
    X = np.eye(6, dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 0], dtype=np.int64)
    return _finalise("two_triangles", X, y, ei, 2)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_cluster_graph_none_gate_raises():
    """cluster_graph must raise if gate_labels is None."""
    g = make_synthetic_graph(n=300, n_classes=3, seed=42)
    with pytest.raises((ValueError, TypeError)):
        cluster_graph(g, gate_labels=None, gate_source="mlp_pred")


def test_cluster_graph_bad_gate_source_raises(fixture_hi):
    with pytest.raises(ValueError):
        cluster_graph(fixture_hi, fixture_hi.y, gate_source="labels")


# ---------------------------------------------------------------------------
# Fixture behaviour
# ---------------------------------------------------------------------------

def test_fixture_high_homophily_picks_sgc(fixture_hi):
    res = cluster_graph(fixture_hi, fixture_hi.y, gate_source="true_oracle")
    assert res["method_used"] == "sgc_kmeans"
    assert normalized_mutual_info_score(fixture_hi.y, res["clusters"]) > 0.5


def test_fixture_low_homophily_picks_feature(fixture_lo):
    res = cluster_graph(fixture_lo, fixture_lo.y, gate_source="true_oracle")
    assert res["method_used"] == "feature_kmeans"


def test_force_overrides_gate(fixture_lo):
    res = cluster_graph(fixture_lo, fixture_lo.y, "true_oracle", force="sgc_kmeans")
    assert res["method_used"] == "sgc_kmeans"


def test_output_keys_shapes_and_determinism(fixture_hi):
    a = cluster_graph(fixture_hi, fixture_hi.y, "mlp_pred", seed=3)
    b = cluster_graph(fixture_hi, fixture_hi.y, "mlp_pred", seed=3)
    n = fixture_hi.X.shape[0]
    assert set(a) == {"clusters", "method_used", "h_hat", "gate_source", "threshold",
                      "embedding", "embedding2d"}
    assert a["clusters"].shape == (n,) and a["embedding2d"].shape == (n, 2)
    assert a["embedding"].shape[0] == n and a["embedding"].shape[1] <= 64
    assert len(np.unique(a["clusters"])) == fixture_hi.num_classes
    np.testing.assert_array_equal(a["clusters"], b["clusters"])


# ---------------------------------------------------------------------------
# Hand-checked metrics
# ---------------------------------------------------------------------------

def test_estimate_homophily_ignores_unlabelled():
    ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    assert estimate_homophily(ei, np.array([0, 0, -1])) == 1.0
    assert estimate_homophily(ei, np.array([0, 1, 1])) == 0.5


def test_profile_hand_checked():
    g = _two_triangles()
    clusters = np.array([0, 0, 0, 1, 1, 1])
    prof = cluster_profile(g, clusters, g.y).set_index("cluster")
    # each triangle: vol = 2+2+3 = 7, cut = 1 -> conductance 1/7
    assert prof.loc[0, "conductance"] == pytest.approx(1 / 7)
    assert prof.loc[0, "purity"] == 1.0 and prof.loc[1, "purity"] == pytest.approx(2 / 3)
    assert prof.loc[0, "avg_degree"] == pytest.approx(7 / 3)
    # cluster 1 inside edges: 3-4 same, 4-5 diff, 3-5 diff -> 1/3
    assert prof.loc[1, "inside_homophily"] == pytest.approx(1 / 3)


def test_metrics_and_louvain_on_two_triangles():
    g = _two_triangles()
    cl = louvain_clusters(g, seed=0)
    assert len(np.unique(cl)) == 2 and cl[0] == cl[1] == cl[2] != cl[3]
    m = cluster_metrics(g, cl, g.X, eval_labels=g.y)
    assert m["n_clusters"] == 2 and 0.3 < m["modularity"] < 0.4  # 5/14 analytically


# ---------------------------------------------------------------------------
# Baseline reproduction on real data (skipped if the download fails)
# ---------------------------------------------------------------------------

def _load(name):
    from src.data import load_graph
    try:
        return load_graph(name)
    except Exception as e:  # noqa: BLE001 - network or mirror failure
        pytest.skip(f"cannot load {name}: {e}")


@pytest.mark.parametrize("name,method,nmi,ari", [
    ("cora", "sgc_kmeans", 0.501, 0.393),
    ("texas", "feature_kmeans", 0.421, 0.314),
])
def test_baseline_reproduction(name, method, nmi, ari):
    g = _load(name)
    res = cluster_graph(g, g.y, "true_oracle", seed=42)
    assert res["method_used"] == method
    m = cluster_metrics(g, res["clusters"], res["embedding"], g.y)
    assert m["nmi"] == pytest.approx(nmi, abs=0.01)
    assert m["ari"] == pytest.approx(ari, abs=0.01)


@pytest.mark.parametrize("name,rises", [("cora", True), ("texas", False)])
def test_hop_sweep_direction(name, rises):
    g = _load(name)
    nmi = [normalized_mutual_info_score(g.y, _kmeans(_embedding_for(g, h), g.num_classes, 0))
           for h in (0, 2)]
    assert (nmi[1] > nmi[0]) == rises


def test_cora_profile_has_low_purity_cluster():
    g = _load("cora")
    res = cluster_graph(g, g.y, "true_oracle", seed=42)
    prof = cluster_profile(g, res["clusters"], g.y)
    assert (prof["purity"] - 0.44).abs().min() < 0.03
