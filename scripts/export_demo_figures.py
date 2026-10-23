"""scripts/export_demo_figures.py - Static PNG backups of every dashboard chart on the demo path.

Lane B (Saanvi) owns this file. Phase P6-B.

Runs the real dashboard headless in demo mode (cora / structural / gcn / lf_cts_plus) and visits
every page; dashboard.runs.show() saves each chart it draws to figures/d_<page>_<nn>.png.
These are the offline fallback if the live dashboard fails during the demo.

Usage: python scripts/export_demo_figures.py [--out figures]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

PAGES = ["Overview", "Clustering explorer", "Shift split viewer", "Calibration",
         "Cluster-wise calibration", "Ablations"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "figures"))
    args = p.parse_args()
    out = Path(args.out)
    os.environ["GNN_CAL_EXPORT_DIR"] = str(out)
    before = set(out.glob("d_*.png")) if out.exists() else set()
    for old in before:
        old.unlink()
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=180).run()
    at.toggle(key="demo").set_value(True).run()
    for page in PAGES:
        at.radio(key="page").set_value(page).run()
        if at.exception:
            raise SystemExit(f"{page}: {at.exception[0].value}")
        print(f"  {page}: ok", flush=True)
    written = sorted(out.glob("d_*.png"))
    for f in written:
        print(f"  {f.name}")
    print(f"{len(written)} PNG backups in {out}")


if __name__ == "__main__":
    main()
