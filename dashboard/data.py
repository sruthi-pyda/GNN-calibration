"""dashboard/data.py - Read-only loaders for the dashboard.

Lane B (Saanvi) owns this file.
Phase P1-C.

Never trains, clusters or downloads: it only reads results/ and cache/ files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent


def results_dir() -> Path:
    """results/ by default; GNN_CAL_RESULTS points the dashboard elsewhere (used by tests)."""
    return Path(os.environ.get("GNN_CAL_RESULTS", ROOT / "results"))


def sources() -> dict:
    return {"fixture": results_dir() / "fixture_results.csv",
            "real": results_dir() / "all_results.csv"}


SOURCES = sources()

# How to create each source when it is missing (shown to the user, never run).
SOURCE_HINTS = {
    "fixture": "python scripts/make_fixture_results.py",
    "real": "python -m src.run_experiments ... (phase P3 onwards)",
}

# Master-plan reference values, shown only when the loader has not cached a graph yet.
# (cache name, display name, regime, nodes, undirected edges, classes, edge homophily, use)
PLAN_REFERENCE = [
    ("cora", "Cora", "homophilic", 2708, 5278, 7, 0.81, "anchor; calibration"),
    ("citeseer", "CiteSeer", "homophilic", 3327, 4552, 6, 0.74, "calibration"),
    ("pubmed", "PubMed", "homophilic", 19717, 44324, 3, 0.80, "calibration, scale"),
    ("texas", "Texas", "heterophilic", 183, 279, 5, 0.06, "anchor; clustering only"),
    ("chameleon_filtered", "Chameleon (filtered)", "heterophilic", 890, 8854, 5, 0.24,
     "calibration"),
    ("roman_empire", "roman-empire", "heterophilic", 22662, 32927, 18, 0.05,
     "calibration (main heterophilic)"),
]


def load_results(path: Path) -> pd.DataFrame | None:
    """Return the results CSV as a DataFrame, or None if it is missing, empty or unreadable."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if df.empty:
        return None
    if "ablation" in df:
        df["ablation"] = df["ablation"].fillna("main")
    return df


def is_fixture(df: pd.DataFrame) -> bool:
    """True if any row is marked as fixture data."""
    if "is_fixture" in df and df["is_fixture"].astype(str).str.lower().eq("true").any():
        return True
    return "git_sha" in df and df["git_sha"].astype(str).eq("fixture").any()


def dataset_table(cache_dir: Path = ROOT / "cache") -> pd.DataFrame:
    """Dataset stats from cache/<name>_meta.json when cached, else the plan reference values."""
    rows = []
    for key, display, regime, n, e, c, h, use in PLAN_REFERENCE:
        meta_path = Path(cache_dir) / f"{key}_meta.json"
        row = dict(dataset=display, regime=regime, nodes=n, edges=e, classes=c,
                   edge_homophily=h, use=use, source="plan reference")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                row.update(nodes=meta["n_nodes"], edges=meta["n_edges_undirected"],
                           classes=meta["num_classes"],
                           edge_homophily=round(meta["edge_homophily"], 3), source="cache")
            except (KeyError, ValueError):
                pass
        rows.append(row)
    return pd.DataFrame(rows)
