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
python -m streamlit run demo/app.py   (live LF-CTS demo, needs results/ unpacked)
```

## Architectural notes for P3 and later

**Four masks do not always cover all N nodes — do not assume they do.**
In the random split every node outside train|val is assigned to target, so
train+val+target_fit+target_eval = N (verified by Saanvi on Cora/CiteSeer/
PubMed/Chameleon/roman-empire, seeds 0–4; Cora target = 2,068 nodes).
In the structural split, train-region nodes that were *not* selected for
train or val belong to no mask by design (train-region interior nodes that
are neither source labels nor target nodes). Typical counts: 468–611 on Cora,
~8,000 on PubMed and roman-empire, per seed.
Rule for `run_one` and all metric code: **index into the masks directly;
never compute the complement of train|val and assume it equals target.**

**Chameleon random-split target is thin.**
With kind="random" Chameleon's target region is ~290 nodes, giving ~145
eval-half nodes. Per-cluster ECE requires ≥20 eval nodes per cluster
(already in `per_cluster_metrics`), so most clusters will return NaN rows.
The plan flags this as an accepted risk; document it in the run artifacts.

## Joint decisions (deviations from the master plan)

| Date | Decision | Reason | Files affected |
|------|----------|--------|----------------|
| Oct 2026 | Early stopping uses **validation loss**, not validation accuracy | Val-accuracy saturates at epoch ~17, leaving the model under-trained (conf 0.39 vs acc 0.81 on Cora GCN). Val-loss gives a more fully-trained model and more stable ECE baselines, which is required for C1 (structural-vs-random gap > seed std). Decided before P3, on principle, not by comparing C1 outcomes. | `src/train.py` |
| Oct 2026 | **Keep the MLP-predicted homophily gate** as specified; report the Cora misfire (h_hat_mlp ≈ 0.45 vs true 0.81 → feature_kmeans) as a finding | The estimator is biased by construction when the MLP is weak (~53% accuracy on Cora), and P3 (e) measured this before any calibration results existed. Changing the gate after P4/P5 results would be redesigning the method after seeing outcomes, and would mean rerunning all 2,575 rows. The cost is measured by the planned `oracle_gate` and `gate_src` ablations instead. | none (reporting only) |

## End of every phase
Stop. List files changed, tests run with real output, deviations from the plan, open questions. Do not start the next phase.
