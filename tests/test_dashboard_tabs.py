"""tests/test_dashboard_tabs.py - Dashboard tabs 2-4 render from files only (Phase P4-B).

Gate checks:
  - tabs work with uncalibrated rows only, and keep working when calibration rows appear
  - missing run artifacts / precomputed data give friendly messages, not tracebacks
  - every chart has a caption and an "Underlying rows" expander
  - the fixture source still shows the FIXTURE banner
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.run_experiments import RESULT_COLUMNS

APP = str(Path(__file__).parent.parent / "dashboard" / "app.py")
N, C = 60, 3


def _bins():
    edges = np.linspace(0, 1, 16)
    return [dict(bin_lower=a, bin_upper=b, count=4, avg_conf=(a + b) / 2, acc=(a + b) / 2,
                 gap=0.0) for a, b in zip(edges[:-1], edges[1:])]


def _write_run(root: Path, calib: str, seed: int, ece: float) -> dict:
    run_id = f"cora__structural__gcn__s{seed}__{calib}__main"
    row = {c: np.nan for c in RESULT_COLUMNS}
    row.update(run_id=run_id, dataset="cora", split="structural", model="gcn", seed=seed,
               calib_method=calib, ablation="main", acc=0.7, ece=ece, cluster_ece_worst=2 * ece,
               h_hat_mlp=0.45, h_hat_gnn=0.84, homophily_gate="feature_kmeans",
               labels_used="target_fit" if "oracle" in calib else "none")
    d = root / "runs" / run_id
    d.mkdir(parents=True)
    meta = dict(bins_uniform=_bins(), bins_mass=_bins(), h_true_eval_only=0.81,
                cluster_method="feature_kmeans",
                cluster_profile=[dict(cluster=k, size=20, majority_class=k, purity=0.5 + 0.1 * k,
                                      conductance=0.3, avg_degree=3 + k, inside_homophily=0.8)
                                 for k in range(C)],
                ece_vs_T=dict(T=np.logspace(-1, 1, 50).tolist(), ece=np.linspace(0.3, 0.1, 50).tolist()))
    (d / "meta.json").write_text(json.dumps(meta))
    rng = np.random.default_rng(seed)
    np.save(d / "embedding2d.npy", rng.normal(size=(N, 2)))
    np.save(d / "clusters.npy", rng.integers(0, C, N))
    return row


def _write_dashboard_data(root: Path, run_ids) -> None:
    d = root / "dashboard"
    (d / "conf_hist").mkdir(parents=True)
    (d / "labels").mkdir()
    np.save(d / "labels" / "cora.npy", np.arange(N) % C)
    for rid in run_ids:
        (d / "conf_hist" / f"{rid}.json").write_text(json.dumps(dict(
            edges=np.linspace(0, 1, 21).tolist(), counts=[3] * 20, mean_conf=0.6, acc=0.7,
            n_eval=60)))
    sweep = [dict(hops=h, nmi_mean=0.1 * h, nmi_std=0.01) for h in range(5)]
    (d / "clustering.json").write_text(json.dumps({
        "cora": dict(methods=dict(sgc_kmeans=dict(nmi=0.5, ari=0.4), louvain=dict(nmi=0.45),
                                  spectral={"skipped": "N=1 > 0"}),
                     hop_sweep=sweep, n_nodes=N, h_true=0.81),
        "_note": "fake"}))
    diag = dict(class_tv_distance=0.5, class_kl=0.7, cross_region_edge_frac=0.06,
                feature_centroid_cosine=0.9, train_homophily=0.8, test_homophily=0.8,
                source_region_frac=0.43, n_train=140, n_val=500, n_target=1500,
                n_train_communities=16)
    (d / "splits.json").write_text(json.dumps({
        f"cora__{k}__s{s}": dict(source_class_dist=[0.3, 0.3, 0.4],
                                 target_class_dist=[0.5, 0.2, 0.3], diagnostics=diag,
                                 class_shortfall={})
        for k in ("random", "structural") for s in range(2)} | {"_note": "fake"}))


def _app(root: Path, monkeypatch, source="real") -> AppTest:
    monkeypatch.setenv("GNN_CAL_RESULTS", str(root))
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(source).run()
    return at


def _texts(at) -> str:
    return " ".join([c.value for c in at.caption] + [i.value for i in at.info]
                    + [m.value for m in at.markdown])


@pytest.fixture
def results(tmp_path):
    rows = [_write_run(tmp_path, "none", s, 0.08 + 0.01 * s) for s in range(2)]
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(tmp_path / "all_results.csv", index=False)
    return tmp_path, rows


def test_tabs_render_uncalibrated_only(results, monkeypatch):
    root, rows = results
    _write_dashboard_data(root, [r["run_id"] for r in rows])
    at = _app(root, monkeypatch)
    assert not at.exception
    text = _texts(at)
    assert "no calibrated rows yet - phase P4-A" in text
    assert "gate clusters with SGC smoothing" in text
    assert len(at.expander) >= 8  # an "Underlying rows" expander per chart/table


def test_tabs_keep_working_with_calibration_rows(results, monkeypatch):
    root, rows = results
    rows += [_write_run(root, m, s, e) for m, s, e in
             [("lf_cts", 0, 0.05), ("lf_cts", 1, 0.06), ("cluster_ts_target_oracle", 0, 0.03)]]
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(root / "all_results.csv", index=False)
    _write_dashboard_data(root, [r["run_id"] for r in rows])
    at = _app(root, monkeypatch)
    assert not at.exception
    assert "oracles that peek at target labels" in _texts(at)
    assert at.selectbox(key="cal_method").options == ["cluster_ts_target_oracle", "lf_cts"]


def test_missing_precomputed_data_is_friendly(results, monkeypatch):
    root, _ = results  # no results/dashboard/ at all
    at = _app(root, monkeypatch)
    assert not at.exception
    assert "make_dashboard_data.py" in _texts(at)


def test_fixture_source_keeps_banner(results, monkeypatch):
    root, rows = results
    pd.DataFrame(rows).assign(is_fixture=True).to_csv(root / "fixture_results.csv", index=False)
    at = _app(root, monkeypatch, source="fixture")
    assert not at.exception
    text = _texts(at)
    assert "FIXTURE DATA - not results" in text
    assert "Run-level views need real results" in text
