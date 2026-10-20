"""src/make_figures.py – Generate all paper figures F1-F7.

Lane A (Sruthi) owns this file.
Phase P6-A.

All figures read from results/ only. Never re-run experiments.
Output: figures/*.png (200 dpi) and figures/*.pdf.
"""
from __future__ import annotations


def make_f1_ece_random_vs_structural():
    """F1: ECE random vs structural, grouped bars per dataset, seed-std error bars (C1)."""
    raise NotImplementedError("Implemented in Phase P6-A")


def make_f2_reliability_diagrams():
    """F2: 2x2 reliability diagrams: cora random/structural x none/lf_cts."""
    raise NotImplementedError("Implemented in Phase P6-A")


def make_f3_cluster_ece_heatmap():
    """F3: Per-cluster ECE heatmap, cora structural, clusters x methods (C2)."""
    raise NotImplementedError("Implemented in Phase P6-A")


def make_f4_proxy_scatter():
    """F4: Proxy vs true per-cluster accuracy scatter, all datasets, r shown (C3)."""
    raise NotImplementedError("Implemented in Phase P6-A")


def make_f5_hops_nmi_ece():
    """F5: NMI and ECE vs hops, one line per dataset, gate threshold noted (C4)."""
    raise NotImplementedError("Implemented in Phase P6-A")


def make_f6_ece_by_method():
    """F6: ECE by method under structural shift, ordered, oracles hatched (C3, C5)."""
    raise NotImplementedError("Implemented in Phase P6-A")


def make_f7_homophily_estimates():
    """F7: h_hat_mlp vs h_hat_gnn vs true homophily per dataset (C4, gate-bias finding)."""
    raise NotImplementedError("Implemented in Phase P6-A")


if __name__ == "__main__":
    for fn in [
        make_f1_ece_random_vs_structural,
        make_f2_reliability_diagrams,
        make_f3_cluster_ece_heatmap,
        make_f4_proxy_scatter,
        make_f5_hops_nmi_ece,
        make_f6_ece_by_method,
        make_f7_homophily_estimates,
    ]:
        try:
            fn()
        except NotImplementedError:
            print(f"[skip] {fn.__name__} – not yet implemented")
