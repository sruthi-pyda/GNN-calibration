# STAC implementation notes

STAC is described in Bahi 2026 (arXiv 2607.10804) as a label-free calibration method.
We do not have access to the full paper text for a faithful implementation.

## What we implement

`stac_global` in `src/baselines.py` implements the simplest stated special case:
a single global temperature estimated without any target labels.

**Proxy:** MLP-GNN prediction agreement on the target region.
For each target node, we check whether the MLP prediction matches the GNN prediction.
The fraction of agreeing nodes is used as a label-free accuracy estimate (acc_hat).
We then bisect T in [0.05, 10] so that mean max-softmax confidence equals acc_hat.

## Differences from the actual STAC paper

Unknown; the paper was not available during implementation.

Key assumptions we made:
1. MLP-GNN agreement is a reasonable proxy for target accuracy (untested against the paper).
2. The "single global temperature" special case is faithful to the paper's stated method.
3. The bisection step is standard and not paper-specific.

Any claim about `stac_global` performance is an approximation, not a reproduction of Bahi 2026.
The label in figures and tables should read "stac_global (approx.)" not "STAC".
