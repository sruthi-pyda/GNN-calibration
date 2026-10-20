"""dashboard/app.py - Streamlit dashboard for GNN calibration results.

Lane B (Saanvi) owns this file.
Phase P1-C (shell on fixture) -> P4-B (tabs 2-4) -> P6-B (tabs 5-6, demo mode).

Run: streamlit run dashboard/app.py

Read-only: never trains, clusters or downloads. Everything shown comes from results/ and cache/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib as mpl
import streamlit as st

from dashboard.data import SOURCE_HINTS, SOURCES, dataset_table, is_fixture, load_results
from dashboard.scorecard import NOT_SUPPORTED, NOT_YET, PARTIAL, PASS, scorecard

FIXTURE_BANNER = "FIXTURE DATA - not results"

# Okabe-Ito: colour-blind safe and readable on a projector.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
mpl.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 12,
    "figure.dpi": 120, "axes.prop_cycle": mpl.cycler(color=PALETTE),
})

STATUS_STYLE = {PASS: ":green", PARTIAL: ":orange", NOT_SUPPORTED: ":red", NOT_YET: ":gray"}

TABS = [
    ("Overview", None),
    ("Clustering explorer", "P4-B"),
    ("Shift split viewer", "P4-B"),
    ("Calibration", "P4-B"),
    ("Cluster-wise calibration", "P6-B"),
    ("Ablations", "P6-B"),
]
FILTERS = [("dataset", "Dataset"), ("split", "Split"), ("model", "Model"),
           ("seed", "Seed"), ("calib_method", "Calibration method")]


@st.cache_data
def _load(path: str, mtime: float):
    """Cached on (path, mtime) so a regenerated CSV is picked up without a restart."""
    return load_results(Path(path))


@st.cache_data
def _datasets():
    return dataset_table()


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def sidebar():
    st.sidebar.header("Data")
    source = st.sidebar.radio("Data source", list(SOURCES), horizontal=True,
                              help="fixture = fake rows for building the UI; real = all_results.csv")
    path = SOURCES[source]
    df = _load(str(path), _mtime(path))
    if df is None:
        return source, None, None, source == "fixture"
    fixture = source == "fixture" or is_fixture(df)
    if fixture:
        st.sidebar.error(FIXTURE_BANNER)
    st.sidebar.header("Filters")
    filtered = df
    for col, label in FILTERS:
        if col not in df:
            continue
        options = sorted(df[col].dropna().unique().tolist())
        chosen = st.sidebar.multiselect(label, options, default=options)
        filtered = filtered[filtered[col].isin(chosen)]
    return source, df, filtered, fixture


def missing_source(source: str):
    st.warning(
        f"No {source} results yet: `results/{SOURCES[source].name}` "
        f"does not exist or is empty.\n\nCreate it with `{SOURCE_HINTS[source]}`, "
        "or switch the data source in the sidebar."
    )


def overview(df, filtered):
    st.subheader("Datasets")
    ds = _datasets()
    st.dataframe(ds, hide_index=True, use_container_width=True)
    st.caption("Stats come from the loader's cache when a graph has been loaded on this machine "
               "(source = cache); otherwise they are the master plan's reference values.")

    st.subheader("Claims scorecard")
    if df is None:
        st.info("not computed yet - no results file for this source")
        return
    if filtered.empty:
        st.info("No rows match the sidebar filters.")
        return
    card = scorecard(filtered)
    for _, r in card.iterrows():
        colour = STATUS_STYLE.get(r["status"], ":gray")
        st.markdown(f"**{r['claim']}** {colour}[**{r['status'].upper()}**] - {r['statement']}  \n"
                    f"<small>{r['evidence']}</small>", unsafe_allow_html=True)
    st.caption("Simple rules over the filtered rows (see dashboard/scorecard.py). "
               "Final verdicts come from results/SUMMARY.md in phase P5.")
    with st.expander(f"Underlying rows ({len(filtered)})"):
        st.dataframe(filtered, hide_index=True, use_container_width=True)


def main():
    st.set_page_config(page_title="GNN Calibration Dashboard", layout="wide")
    source, df, filtered, fixture = sidebar()
    st.title("Label-free cluster-wise calibration under structural shift")
    if fixture:
        st.markdown(
            f'<div style="background:#C00000;color:#fff;padding:0.7rem 1rem;border-radius:6px;'
            f'font-size:1.15rem;font-weight:700">{FIXTURE_BANNER}</div>'
            '<p style="margin-top:0.3rem">Every number on this page is synthetic.</p>',
            unsafe_allow_html=True,
        )
    if df is None:
        missing_source(source)

    tabs = st.tabs([name for name, _ in TABS])
    with tabs[0]:
        overview(df, filtered)
    for tab, (name, phase) in zip(tabs[1:], TABS[1:]):
        with tab:
            st.info(f"not computed yet - phase {phase}")


main()
