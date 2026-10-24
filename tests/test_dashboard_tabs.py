"""tests/test_dashboard_tabs.py - Dashboard pages 2-6 render from files only (P4-B, P6-B).

Gate checks:
  - pages work with uncalibrated rows only, and keep working when calibration rows appear
  - missing run artifacts / precomputed data give friendly messages, not tracebacks
  - every chart has a caption and an "Underlying rows" expander
  - the fixture source still shows the FIXTURE banner; demo mode removes it and preselects
    cora / structural / gcn / lf_cts_plus
  - walking every page x every selectbox option raises nothing ("random clicking")
  - export mode writes a PNG backup of every chart
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.tab_ablations import parse_tag
from src.run_experiments import RESULT_COLUMNS

APP = str(Path(__file__).parent.parent / "dashboard" / "app.py")
N, C = 60, 3
PAGES = ["Overview", "Clustering explorer", "Shift split viewer", "Calibration",
         "Cluster-wise calibration", "Ablations"]


def _bins():
    edges = np.linspace(0, 1, 16)
    return [dict(bin_lower=a, bin_upper=b, count=4, avg_conf=(a + b) / 2, acc=(a + b) / 2,
                 gap=0.0) for a, b in zip(edges[:-1], edges[1:])]


def _write_run(root: Path, calib: str, seed: int, ece: float, ablation="main", dataset="cora",
               split="structural") -> dict:
    run_id = f"{dataset}__{split}__gcn__s{seed}__{calib}__{ablation}"
    row = {c: np.nan for c in RESULT_COLUMNS}
    row.update(run_id=run_id, dataset=dataset, split=split, model="gcn", seed=seed,
               calib_method=calib, ablation=ablation, acc=0.7, ece=ece, cluster_ece_worst=2 * ece,
               worst_cluster_id=1, h_hat_mlp=0.45, h_hat_gnn=0.84,
               homophily_gate="feature_kmeans",
               labels_used="target_fit" if "oracle" in calib else "none")
    d = root / "runs" / run_id
    d.mkdir(parents=True)
    meta = dict(
        bins_uniform=_bins(), bins_mass=_bins(), h_true_eval_only=0.81,
        cluster_method="feature_kmeans",
        cluster_profile=[dict(cluster=k, size=20, majority_class=k, purity=0.5 + 0.1 * k,
                              conductance=0.3, avg_degree=3 + k, inside_homophily=0.8)
                         for k in range(C)],
        per_cluster=[dict(cluster_id=k, n_eval=25 if k else 10, ece=np.nan if k == 0 else ece * k,
                          acc=0.7) for k in range(C)],
        ece_vs_T=dict(T=np.logspace(-1, 1, 50).tolist(), ece=np.linspace(0.3, 0.1, 50).tolist()))
    if calib in ("lf_cts", "lf_cts_plus"):
        key = "a_hat" if calib == "lf_cts" else "a_tilde"
        meta["per_cluster_calib"] = [{"cluster_id": k, key: 0.5 + 0.1 * k, "T": 1.0 + 0.1 * k,
                                      "clamped": False} for k in range(C)]
        meta["proxy_quality"] = dict(r=0.6, table=[dict(cluster_id=k, a_hat=0.5 + 0.1 * k,
                                                        true_acc=0.55 + 0.1 * k, n_eval=25)
                                                   for k in range(C)])
    if calib == "cluster_ts_target_oracle":
        meta["per_cluster_calib"] = [dict(cluster_id=k, T=1.2, n_fit=25, used_global_T=k == 0)
                                     for k in range(C)]
    (d / "meta.json").write_text(json.dumps(meta))
    rng = np.random.default_rng(seed)
    np.save(d / "embedding2d.npy", rng.normal(size=(N, 2)))
    np.save(d / "clusters.npy", rng.integers(0, C, N))
    return row


def _write_dashboard_data(root: Path, run_ids) -> None:
    d = root / "dashboard"
    (d / "conf_hist").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(exist_ok=True)
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


def _save(root: Path, rows) -> None:
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(root / "all_results.csv", index=False)


def _app(root: Path, monkeypatch, source="real", page="Overview", demo=False) -> AppTest:
    monkeypatch.setenv("GNN_CAL_RESULTS", str(root))
    at = AppTest.from_file(APP, default_timeout=60).run()
    if demo:
        at.toggle(key="demo").set_value(True).run()
    else:
        at.sidebar.radio[0].set_value(source).run()
    at.radio(key="page").set_value(page).run()
    return at


def _texts(at) -> str:
    return " ".join([c.value for c in at.caption] + [i.value for i in at.info]
                    + [m.value for m in at.markdown] + [w.value for w in at.warning]
                    + [m.label for m in at.metric])


@pytest.fixture
def results(tmp_path):
    rows = [_write_run(tmp_path, "none", s, 0.08 + 0.01 * s) for s in range(2)]
    _save(tmp_path, rows)
    return tmp_path, rows


@pytest.fixture
def full_results(results):
    root, rows = results
    rows += [_write_run(root, m, s, e) for m, s, e in
             [("lf_cts", 0, 0.20), ("lf_cts", 1, 0.21), ("lf_cts_plus", 0, 0.12),
              ("lf_cts_plus", 1, 0.13), ("cluster_ts_target_oracle", 0, 0.03),
              ("cluster_ts_target_oracle", 1, 0.04)]]
    for tag, e in [("gate", 0.09), ("oracle_gate", 0.18), ("gate_src", 0.22), ("k", 0.19),
                   ("k_14C", 0.2), ("k_28C", 0.21), ("hops", 0.2), ("hops_3", 0.21),
                   ("proxy", 0.1), ("cluster", 0.07), ("cluster_random", 0.3), ("weird", 0.5)]:
        rows.append(_write_run(root, "lf_cts", 0, e, ablation=tag))
    rows.append(_write_run(root, "lf_cts", 0, 0.3, ablation="degree", split="degree"))
    # P5's degree ablation also has uncalibrated rows; "degree" sorts before "structural"
    rows.append(_write_run(root, "none", 0, 0.1, ablation="degree", split="degree"))
    _save(root, rows)
    _write_dashboard_data(root, [r["run_id"] for r in rows])
    return root, rows


def test_parse_tag_matches_p5_sweep_tags():
    # the tags Lane A's P5 sweep actually wrote (confirmed against her CSV)
    assert parse_tag("k") == ("k", "2")
    assert parse_tag("k_14C") == ("k", "14") and parse_tag("k_28C") == ("k", "28")
    assert parse_tag("hops") == ("hops", "2") and parse_tag("hops_0") == ("hops", "0")
    assert parse_tag("proxy") == ("proxy", "mlp_gnn_agreement")
    assert parse_tag("cluster") == ("cluster", "louvain")
    assert parse_tag("cluster_random") == ("cluster", "random")
    assert parse_tag("oracle_gate") == ("oracle_gate", None)
    assert parse_tag("gate_src") == ("gate_src", None) and parse_tag("gate") == ("gate", None)
    assert parse_tag("kitten") == (None, None) and parse_tag("weird") == (None, None)
    assert parse_tag("k_abc") == (None, None)


def test_pages_render_uncalibrated_only(results, monkeypatch):
    root, rows = results
    _write_dashboard_data(root, [r["run_id"] for r in rows])
    for page in PAGES:
        at = _app(root, monkeypatch, page=page)
        assert not at.exception, page
    at = _app(root, monkeypatch, page="Calibration")
    assert "no calibrated rows yet - phase P4-A" in _texts(at)
    at = _app(root, monkeypatch, page="Clustering explorer")
    assert "gate clusters with SGC smoothing" in _texts(at) and len(at.expander) >= 3
    assert "calibrated rows come from phase P4-A" in _texts(
        _app(root, monkeypatch, page="Cluster-wise calibration"))
    assert "No ablation rows yet" in _texts(_app(root, monkeypatch, page="Ablations"))


def test_clusterwise_page_with_calibration_rows(full_results, monkeypatch):
    root, _ = full_results
    at = _app(root, monkeypatch, page="Cluster-wise calibration")
    assert not at.exception
    text = _texts(at)
    assert "Pearson r - LF-CTS (a_hat)" in text and "Pearson r - LF-CTS+ (a_tilde)" in text
    assert "Worst-cluster ECE" in text and "Each row is one cluster" in text
    assert len(at.dataframe) >= 1  # per-cluster temperature table
    at = _app(root, monkeypatch, page="Calibration")
    assert at.selectbox(key="cal_method").options == ["cluster_ts_target_oracle", "lf_cts",
                                                      "lf_cts_plus"]


def test_ablations_page(full_results, monkeypatch):
    root, _ = full_results
    at = _app(root, monkeypatch, page="Ablations")
    assert not at.exception
    text = _texts(at)
    assert "Unrecognised ablation tags (not shown): ['weird']" in text
    assert "Knob value not recorded" not in text
    for phrase in ("forced wrong branch", "ECE as k changes", "ECE as hops changes",
                   "proxy variant", "cluster variant", "lowest-degree half"):
        assert phrase in text, phrase


def test_missing_precomputed_data_is_friendly(results, monkeypatch):
    root, _ = results  # no results/dashboard/ at all
    for page in PAGES[1:]:
        at = _app(root, monkeypatch, page=page)
        assert not at.exception, page
    assert "make_dashboard_data.py" in _texts(_app(root, monkeypatch, page="Shift split viewer"))


def test_fixture_source_keeps_banner(results, monkeypatch):
    root, rows = results
    pd.DataFrame(rows).assign(is_fixture=True).to_csv(root / "fixture_results.csv", index=False)
    at = _app(root, monkeypatch, source="fixture", page="Calibration")
    assert not at.exception
    text = _texts(at)
    assert "FIXTURE DATA - not results" in text
    assert "Run-level views need real results" in text


def test_demo_mode_preselects_and_hides_fixture(full_results, monkeypatch):
    root, _ = full_results
    at = _app(root, monkeypatch, page="Cluster-wise calibration", demo=True)
    assert not at.exception
    assert len(at.sidebar.radio) == 0 and len(at.sidebar.multiselect) == 0
    assert "FIXTURE DATA" not in _texts(at)
    assert at.selectbox(key="cw_dataset").value == "cora"
    assert at.selectbox(key="cw_split").value == "structural"
    assert at.selectbox(key="cw_method").value == "lf_cts_plus"


def test_random_clicking_every_page_and_option(full_results, monkeypatch):
    root, _ = full_results
    clicks = 0
    for page in PAGES:
        at = _app(root, monkeypatch, page=page)
        keys = [w.key for w in list(at.selectbox) + list(at.radio)
                if w.key and w.key != "page"]
        for key in keys:
            live = {w.key: w for w in list(at.selectbox) + list(at.radio) if w.key}
            if key not in live:  # an earlier choice removed this widget
                continue
            for opt in live[key].options[:4]:
                widget = {w.key: w for w in list(at.selectbox) + list(at.radio) if w.key}.get(key)
                if widget is None:
                    break
                widget.set_value(opt).run()
                clicks += 1
                assert not at.exception, (page, key, opt)
    assert clicks >= 20


def test_export_writes_png_per_chart(full_results, monkeypatch, tmp_path_factory):
    root, _ = full_results
    out = tmp_path_factory.mktemp("figs")
    monkeypatch.setenv("GNN_CAL_EXPORT_DIR", str(out))
    _app(root, monkeypatch, page="Cluster-wise calibration")
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert "d_clusterwise_calibration_01.png" in pngs and len(pngs) >= 2
