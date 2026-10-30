"""tests/test_demo.py - Live demo (demo/): pipeline reproduces the saved runs, app runs.

Needs the real results/ (Sruthi's P5 zip); skipped when it is not unpacked.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from demo import pipeline as P

CONFIGS = [(ds, m) for ds in P.DATASETS.values() for m in P.MODELS.values()]
needs_results = pytest.mark.skipif(not all(P.available(ds, m, 0) for ds, m in CONFIGS),
                                   reason="real results/ not unpacked")


def test_label_free_graph_has_no_labels():
    lf = P.LabelFreeGraph(np.zeros((2, 2)), np.zeros((2, 0), dtype=int), 2)
    with pytest.raises(AttributeError):
        lf.y


@needs_results
@pytest.mark.parametrize("ds,model", CONFIGS)
def test_pipeline_reproduces_saved_run(ds, model):
    S = P.run_all(ds, model, 0)
    assert S["cluster_match"] == 1.0
    assert np.abs(S["a_hat"] - S["a_hat_saved"]).max() < 1e-3
    assert S["probs_diff"] < 1e-3
    for m in P.METHODS:
        assert abs(S["ece_live"][m] - S["ece_saved"][m]) < 1e-3


@needs_results
def test_app_run_demo_and_interact():
    from streamlit.testing.v1 import AppTest
    os.environ["DEMO_PAUSE"] = "0"
    at = AppTest.from_file(str(Path(__file__).parent.parent / "demo" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 7
    at.radio(key="graph_mode").set_value("Louvain communities").run()
    at.selectbox(key="cluster_sel").set_value(1).run()
    at.selectbox(key="cluster_sel").set_value(None).run()
    at.checkbox(key="show_true").check().run()
    at.multiselect(key="rel_methods").set_value(["none", "lf_cts"]).run()
    assert not at.exception
    for ds in ["CiteSeer", "Chameleon"]:
        at.selectbox[0].set_value(ds).run()
        at.button[0].click().run()
        assert not at.exception
