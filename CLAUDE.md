# gnn_cal: Label-Free Cluster-wise Calibration of GNNs under Structural Shift

## Read first
Master plan: GNN Calibration Project — Master Plan.md (in the parent directory).
Baseline: "Clustering Analysis for Label-Free GNN Calibration" (Oct 10, 2026).
Lanes: A = Sruthi (data, models, metrics, calibrate, figures). B = Saanvi (clustering, splits, dashboard).
No Lane C: P1-C dashboard shell -> B (after P1-B), P2-C baselines -> A (after P4-A), P4-C CSBM sweep dropped. Edit only your lane's files.

## Hard rules
1. Never pass g.y to a calibrator, the gate, or a proxy. Use GuardedLabels. tests/test_leakage.py must pass.
2. The gate uses MLP-predicted labels. True-label homophily is allowed only with ablation=oracle_gate.
3. All metrics are computed on the target_eval mask only. Oracles fit on target_fit only.
4. Never tune code to hit an expected number. If a check fails, stop and report the mismatch.
5. Fixture/synthetic data is always labelled as such in outputs and on the dashboard.
6. Every result is a CSV row + results/runs/<run_id>/. Never write a number that has no row.
7. Plain numpy/scipy/torch/sklearn/networkx. No torch-geometric, no DGL. Files under ~250 lines.
8. Seeds: default 42; sweeps use 0-4. Seed numpy, torch and random.

## Commands
```
pytest tests/ -v
python -m src.data --summary
python -m src.clustering --dataset cora --compare
python -m src.run_experiments --dry-run
streamlit run dashboard/app.py
```

## End of every phase
Stop. List files changed, tests run with real output, deviations from the plan, open questions. Do not start the next phase.
