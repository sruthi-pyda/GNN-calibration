"""demo/app.py - Live research demo of LF-CTS (Lane B).

Run:  python -m streamlit run demo/app.py

One page, four sections, one RUN DEMO button that steps through the six stages in
demo/pipeline.py and fills each section as its stage finishes. Everything shown is
the real data and the saved experiment runs, recomputed live - nothing is simulated.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from demo import pipeline as P
from demo.sections import render_calibration, render_clusters, render_graph, render_results

st.set_page_config(page_title="LF-CTS live demo", layout="wide")

SECTIONS = [  # (title, function, first stage that fills it, stages that redraw it)
    ("1. Graph and structural shift", render_graph, 1, {1}),
    ("2. Clustering", render_clusters, 2, {2, 3, 4}),
    ("3. LF-CTS calibration", render_calibration, 3, {3, 4, 5}),
    ("4. Results comparison", render_results, 6, {6}),
]
FLOW = ["Graph", "Structural shift", "GNN predictions", "Homophily gate", "Clustering",
        "Label-free accuracy estimation", "Cluster-wise temperature scaling", "Calibrated predictions"]
PAUSE = float(os.environ.get("DEMO_PAUSE", 0.6))  # seconds per stage, so the audience can follow


def draw(placeholders, S: dict, stage: int, live: bool, only=None):
    for i, (title, fn, first, redraw) in enumerate(SECTIONS):
        if only is not None and stage not in redraw:
            continue
        with placeholders[i].container():
            if S is not None and stage >= first:
                fn(S, stage, live)
            else:
                st.info(f"Press **RUN DEMO** - this section fills in at stage {first}: {P.STAGES[first - 1]}.")


def stage_list(done: int) -> str:
    return "  \n".join(("✅ " if i < done else "▶️ " if i == done else "⬜ ") + f"{i + 1}. {s}"
                       for i, s in enumerate(P.STAGES))


st.title("Label-free cluster-wise calibration of GNNs under structural shift")
st.markdown(" → ".join(f"**{s}**" for s in FLOW))
st.caption("Real graphs and the project's saved experiment runs (results/runs/), recomputed live. "
           "Target labels are never given to the calibration steps; they are used only to score the result.")

c1, c2, c3 = st.columns(3)
ds = P.DATASETS[c1.selectbox("Dataset", list(P.DATASETS))]
model = P.MODELS[c2.selectbox("GNN model", list(P.MODELS),
                              help="On GCN, LF-CTS does not beat the uncalibrated model - see section 4.")]
seed = c3.selectbox("Seed", [0, 1, 2, 3, 4])
key = (ds, model, seed)

if not P.available(ds, model, seed):
    st.error(f"Saved runs for {ds} / {model} / seed {seed} were not found under "
             f"{P.results_dir() / 'runs'}. Unpack the results zip into results/ first.")
    st.stop()

run = st.button("▶  RUN DEMO", type="primary", use_container_width=True)
progress = st.empty()
status = st.empty()
placeholders = []
for title, *_ in SECTIONS:
    st.divider()
    st.header(title)
    placeholders.append(st.empty())

saved = st.session_state.get("demo")
if run:
    S = None
    draw(placeholders, None, 0, live=False)
    for i, name in enumerate(P.STAGES):
        status.markdown(stage_list(i))
        progress.progress(i / len(P.STAGES), text=f"Stage {i + 1} of {len(P.STAGES)}: {name}...")
        S = P.run_stage(i, S, ds, model, seed)
        draw(placeholders, S, i + 1, live=False, only=True)
        time.sleep(PAUSE)
    st.session_state["demo"] = {"key": key, "S": S}
    st.rerun()
elif saved and saved["key"] == key:
    progress.progress(1.0, text="All six stages complete - explore the sections below.")
    status.markdown(stage_list(len(P.STAGES)))
    draw(placeholders, saved["S"], len(P.STAGES), live=True)
else:
    status.markdown(stage_list(-1))
    draw(placeholders, None, 0, live=False)
