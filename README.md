# gnn_cal: Label-Free Cluster-wise Calibration of GNNs under Structural Shift

## Claims

| ID | Statement |
|----|-----------|
| C1 | Structural shift breaks calibration in a way random splits don't reveal |
| C2 | Overall ECE hides per-cluster damage, concentrated in low-purity and low-degree clusters |
| C3 | LF-CTS recovers most of that damage with zero target labels |
| C4 | The homophily gate carries over from clustering to calibration |
| C5 | LF-CTS beats existing label-free baselines and is competitive with labeled ones |

## Method: LF-CTS (Label-Free Cluster-wise Temperature Scaling)

```
1. Train an MLP and the GNN on the source train split. Logits for every node.
2. h_hat = edge homophily of MLP-predicted labels   # structure-free, no target labels
3. Cluster the whole graph, gated on h_hat:
      h_hat > 0.5  -> SGC 2-hop smoothing -> SVD(64) -> KMeans(k)
      h_hat <= 0.5 -> raw features         -> SVD(64) -> KMeans(k)
4. Per cluster c, label-free accuracy proxy a_hat[c]:
      h_hat > 0.5  -> intra-cluster edge agreement of GNN predictions
      h_hat <= 0.5 -> k-NN agreement in SVD embedding
      clip to [1/C, 0.999]; fallback to graph-wide value if < 20 edges/neighbours
5. Bisect T_c in [0.05, 10] so mean top-softmax confidence in c equals a_hat[c].
6. Apply T_c to every node in c.
```

## Layout and ownership

```
gnn_cal/
  src/
    data.py            Graph, loaders, CSBM                [A: Sruthi]
    labels.py          GuardedLabels + leakage audit        [joint, P0]
    clustering.py      gate, SGC, kmeans, louvain, profile  [B: Saanvi]
    splits.py          random, structural, degree, csbm     [B: Saanvi]
    models.py          MLP, GCN, SGC                        [A: Sruthi]
    train.py           train loop -> logits                 [A: Sruthi]
    metrics.py         ECE variants, NLL, Brier, MCE        [A: Sruthi]
    calibrate.py       TS, LF-TS, LF-CTS, LF-CTS+          [A: Sruthi]
    baselines.py       CaGCN, ATC-TS, STAC                  [A: Sruthi, after P4-A]
    run_experiments.py driver, logging                      [joint]
    make_figures.py    F1-F7                                [A: Sruthi]
  dashboard/app.py                                          [B: Saanvi]
  tests/
  results/all_results.csv   results/runs/
  cache/   figures/   data/
```

## How to run

```bash
# Install dependencies in a fresh venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download + verify all datasets (P1-A)
python -m src.data --summary

# Run tests
pytest tests/ -v

# Dry-run the experiment pipeline
python -m src.run_experiments --dry-run

# Launch dashboard
streamlit run dashboard/app.py
```

## Branches

- `main` – protected; merges at gates P0, P3, P5 only
- `lane-a/<phase>` – Sruthi's work
- `lane-b/<phase>` – Saanvi's work
- No Lane C: P1-C dashboard shell -> Lane B (after P1-B); P2-C baselines -> Lane A (after P4-A); P4-C CSBM sweep dropped

See CLAUDE.md for hard rules and phase prompts.
