"""dashboard/scorecard.py - C1-C5 claim scorecard computed from a results DataFrame.

Lane B (Saanvi) owns this file.
Phase P1-C.

Each check returns (status, evidence). Status is one of PASS, PARTIAL, NOT_SUPPORTED or
NOT_YET (the rows needed to judge the claim do not exist yet).

Rules (deliberately simple; the real verdicts come from results/SUMMARY.md in P5):
  C1  calib=none, ablation=main. Per dataset with both splits: gap = mean ECE(structural)
      - mean ECE(random). With >= 2 seeds per split the gap must exceed the larger of the two
      seed stds; with 1 seed, gap > 0 counts as PARTIAL at most.
      PASS = every dataset beats its std, PARTIAL = at least one dataset has gap > 0.
  C2  calib=none, ablation=main. max(cluster_ece_worst / ece) >= 2 -> PASS, >= 1.5 -> PARTIAL.
  C3  structural, ablation=main. Per (dataset, seed) with none, lf_cts and
      cluster_ts_target_oracle: gap closed = (none - lf_cts) / (none - oracle), skipping cells
      where none <= oracle. Mean >= 0.5 -> PASS, > 0 -> PARTIAL.
  C4  Needs ablation=gate rows. Per dataset: lf_cts ECE gated (main) < forced wrong branch
      (gate). PASS = every dataset, PARTIAL = some.
  C5  structural, ablation=main. Per dataset: best of lf_cts / lf_cts_plus vs best of the
      label-free baselines lf_ts / atc_ts / stac (means over seeds). PASS = ours lower on
      every dataset, PARTIAL = some.
"""
from __future__ import annotations

import pandas as pd

PASS, PARTIAL, NOT_SUPPORTED, NOT_YET = "pass", "partial", "not supported", "not yet measured"

CLAIMS = {
    "C1": "Structural shift breaks calibration in a way random splits don't reveal",
    "C2": "Overall ECE hides per-cluster damage",
    "C3": "LF-CTS recovers most of that damage with zero target labels",
    "C4": "The homophily gate carries over from clustering to calibration",
    "C5": "LF-CTS beats existing label-free baselines",
}

OURS = ("lf_cts", "lf_cts_plus")
LABEL_FREE_BASELINES = ("lf_ts", "atc_ts", "stac")


def _rows(df, calib=None, split=None, ablation="main"):
    m = pd.Series(True, index=df.index)
    if ablation is not None:
        m &= df["ablation"].fillna("main") == ablation
    if calib is not None:
        m &= df["calib_method"].isin([calib] if isinstance(calib, str) else calib)
    if split is not None:
        m &= df["split"] == split
    return df[m]


def _share(n_hit, n_total):
    if n_total == 0 or n_hit == 0:
        return NOT_SUPPORTED
    return PASS if n_hit == n_total else PARTIAL


def check_c1(df):
    none = _rows(df, "none")
    hits, positive, total, notes = 0, 0, 0, []
    for ds, d in none.groupby("dataset"):
        s, r = d[d["split"] == "structural"]["ece"], d[d["split"] == "random"]["ece"]
        if s.empty or r.empty:
            continue
        total += 1
        gap = s.mean() - r.mean()
        positive += gap > 0
        if min(len(s), len(r)) >= 2 and gap > max(s.std(), r.std()):
            hits += 1
        notes.append(f"{ds} {gap:+.3f}")
    if total == 0:
        return NOT_YET, "needs calib=none rows for both splits"
    status = PASS if hits == total else (PARTIAL if positive else NOT_SUPPORTED)
    return status, f"structural - random ECE: {', '.join(notes)}"


def check_c2(df):
    none = _rows(df, "none")
    none = none[none["ece"] > 0]
    if none.empty or none["cluster_ece_worst"].isna().all():
        return NOT_YET, "needs calib=none rows with cluster_ece_worst"
    ratio = none["cluster_ece_worst"] / none["ece"]
    best = none.loc[ratio.idxmax()]
    top = ratio.max()
    status = PASS if top >= 2 else (PARTIAL if top >= 1.5 else NOT_SUPPORTED)
    return status, f"max worst/overall = {top:.2f}x ({best['dataset']}, {best['split']})"


def check_c3(df):
    d = _rows(df, ("none", "lf_cts", "cluster_ts_target_oracle"), split="structural")
    wide = d.pivot_table(index=["dataset", "seed"], columns="calib_method", values="ece")
    need = {"none", "lf_cts", "cluster_ts_target_oracle"}
    if not need.issubset(wide.columns):
        return NOT_YET, "needs none, lf_cts and cluster_ts_target_oracle on structural"
    wide = wide.dropna(subset=list(need))
    wide = wide[wide["none"] > wide["cluster_ts_target_oracle"]]
    if wide.empty:
        return NOT_YET, "no structural cell where the oracle improves on none"
    closed = ((wide["none"] - wide["lf_cts"])
              / (wide["none"] - wide["cluster_ts_target_oracle"])).mean()
    status = PASS if closed >= 0.5 else (PARTIAL if closed > 0 else NOT_SUPPORTED)
    return status, f"mean gap closed = {closed:.0%} over {len(wide)} cells"


def check_c4(df):
    gated = _rows(df, "lf_cts").groupby("dataset")["ece"].mean()
    wrong = _rows(df, "lf_cts", ablation="gate").groupby("dataset")["ece"].mean()
    both = gated.index.intersection(wrong.index)
    if both.empty:
        return NOT_YET, "needs ablation=gate rows (phase P5)"
    hits = int((gated[both] < wrong[both]).sum())
    return _share(hits, len(both)), f"gated beats forced-wrong on {hits}/{len(both)} datasets"


def check_c5(df):
    d = _rows(df, OURS + LABEL_FREE_BASELINES, split="structural")
    means = d.groupby(["dataset", "calib_method"])["ece"].mean().unstack()
    ours = [c for c in OURS if c in means]
    base = [c for c in LABEL_FREE_BASELINES if c in means]
    if not ours or not base:
        return NOT_YET, "needs lf_cts/lf_cts_plus and lf_ts/atc_ts/stac rows"
    cmp = pd.DataFrame({"ours": means[ours].min(axis=1), "base": means[base].min(axis=1)})
    cmp = cmp.dropna()
    if cmp.empty:
        return NOT_YET, "no dataset has both ours and a label-free baseline"
    hits = int((cmp["ours"] < cmp["base"]).sum())
    return _share(hits, len(cmp)), f"ours beats best label-free baseline on {hits}/{len(cmp)}"


def scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """One row per claim: claim, statement, status, evidence."""
    checks = dict(C1=check_c1, C2=check_c2, C3=check_c3, C4=check_c4, C5=check_c5)
    rows = []
    for claim, fn in checks.items():
        status, evidence = fn(df)
        rows.append(dict(claim=claim, statement=CLAIMS[claim], status=status, evidence=evidence))
    return pd.DataFrame(rows)
