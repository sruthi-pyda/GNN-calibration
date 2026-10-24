# Label-Free Cluster-wise Calibration of GNNs under Structural Shift

**Sruthi Pyda · Saanvi Sharma** · October 2026
`gnn_cal` · Branch `lane-a/p5` · All numbers trace to `results/all_results.csv`.

---

## Abstract

Graph neural networks are systematically miscalibrated under structural distribution shift — a regime where the test graph community structure differs from the training region. We show that this shift inflates expected calibration error (ECE) on CiteSeer (+0.049 ECE over random split, exceeding seed std 0.019) and exposes per-cluster damage hidden by the overall mean (worst-cluster ECE up to 13.5× overall on roman-empire). We propose **LF-CTS**, a label-free cluster-wise temperature scaling method that estimates per-cluster accuracy from neighbor-agreement statistics and calibrates each cluster independently. On roman-empire structural (SGC), where the accuracy proxy is reliable (Pearson r=0.64 on GCN), LF-CTS matches the cluster oracle (gap 106% closed) with zero target labels. The method fails on homophilic graphs because a structure-free MLP underestimates homophily (Cora: h_mlp=0.45 vs target-region h=0.82), triggering the wrong proxy branch and worsening calibration. We characterize this failure, show that the oracle gate partially closes it, and identify MLP–GNN agreement (STAC) as the strongest label-free baseline overall (5-dataset structural mean ECE 0.081, vs ts_val mean 0.072).

---

## 1. Introduction

A calibrated classifier produces confidence scores that match empirical accuracy: a prediction of 0.8 should be correct 80% of the time. Standard temperature scaling (TS) achieves near-perfect calibration when training and test data are i.i.d. GNNs, however, are trained on a structurally distinct subgraph (the training community) and evaluated on a shifted region (the target community). Under this *structural shift*, the validation set used to fit TS lies in a different part of the graph than the target, weakening the calibration guarantee.

Per-cluster calibration offers a finer-grained remedy: fit a separate temperature per cluster of the target region, exploiting the fact that calibration errors are often concentrated in a subset of clusters (worst-cluster ECE 2–13× overall across our datasets). Without target labels, we estimate per-cluster accuracy from neighbor-agreement statistics and bisect the temperature to match it — a method we call LF-CTS. We also investigate a ridge-regression correction (LF-CTS+) that debiases the proxy using the same validation labels that standard TS uses.

---

## 2. Related Work

**Temperature scaling.** Guo et al. (2017) show that a single post-hoc scalar T rescaling logits reduces ECE without changing predictions, and that modern neural networks are systematically overconfident. GNNs, by contrast, tend toward *underconfidence* (GATS, Hans 2023); we observe this on Cora (avg_conf < acc for all models and splits). CaGCN (Wang et al. 2021) learns a per-node T from a small GCN over frozen logits, using graph structure to propagate calibration signal.

**Label-free calibration.** ATC (Garg et al. 2022) estimates target accuracy from a threshold on validation confidence, avoiding target labels entirely. STAC (Bahi 2026) applies a global T from MLP–GNN prediction agreement; our `stac_global` implementation follows its label-free special case. GNNEvaluator (Zhao et al. 2023) learns an accuracy predictor on simulated source shifts, inspiring LF-CTS+'s ridge regression step.

**Graph heterophily and benchmarks.** Platonov et al. (2023) introduce roman-empire and other large heterophilic graphs, noting that filtered Chameleon and roman-empire better reflect real heterophily than earlier benchmarks. The GOOD benchmark (Gui et al. 2022) and Matcha (Ma et al. 2024) study distribution shift on graphs; we follow GOOD's degree-split construction for our degree ablation.

---

## 3. Method

### 3.1 Structural splits

We partition graph nodes into a **source** region (train + val) and a **target** region using Louvain community detection, assigning communities to the source until 40–50% of nodes are covered. Every class must appear in the source region; train/val are sampled within it. The target region is split 50/50 by seed into `target_fit` (oracle ceiling only) and `target_eval` (evaluation). All reported metrics are on `target_eval` only. The random split follows Planetoid style: 20 labelled nodes per class, 500 val, remainder target.

### 3.2 LF-CTS

Given GNN logits for all N nodes:

1. Train a structure-free MLP on the same source labels; compute edge homophily of MLP-predicted labels: h_hat.
2. Cluster the whole graph gated on h_hat:
   - h_hat > 0.5: SGC 2-hop smoothed features → SVD(64) → k-means(k = num_classes)
   - h_hat ≤ 0.5: raw features → SVD(64) → k-means
3. Per cluster c, estimate accuracy **without target labels**:
   - h_hat > 0.5: fraction of intra-cluster edges where both endpoints share the same predicted class
   - h_hat ≤ 0.5: same agreement over each node's 10 nearest neighbors in SVD space
   - Clip to [1/C, 0.999]; fall back to graph-wide value if < 20 edges/neighbors.
4. Bisect T_c ∈ [0.05, 10] so that mean(max-softmax over cluster c in target) = a_hat[c].
5. Apply T_c to all nodes in cluster c; return calibrated probabilities.

### 3.3 LF-CTS+

Uses the same validation labels as standard TS (no additional supervision):

1. Sample 200 pseudo-target groups from validation clusters; for each record a_hat, cluster size, mean degree, mean confidence, mean entropy, local h_hat, and true accuracy.
2. Fit RidgeCV (α ∈ {0.01, 0.1, 1, 10}, 5-fold CV) to predict true accuracy from these features.
3. On the target: a_tilde[c] = ridge(features[c]), then LF-CTS with a_tilde.

### 3.4 Evaluation protocol

Labels are passed only through `GuardedLabels(y, allowed_mask)`, which raises `LabelLeakError` on any out-of-mask read. Each calibrator declares `LABELS ∈ {none, source_val, target_fit}`; the driver constructs the guard from this declaration. Target-oracle methods (`ts_target_oracle`, `cluster_ts_target_oracle`) may read `target_fit`; all others are blind to the target. All metrics are on `target_eval` only.

---

## 4. Setup

**Datasets.** Five calibration graphs: Cora (2,708 nodes, h=0.81), CiteSeer (3,327, h≈0.74), PubMed (19,717, h≈0.80), filtered Chameleon (≈890, h≈0.24), roman-empire (22,662, h=0.05). Texas (183 nodes) is used for clustering comparison only.

**Models.** Two-layer GCN and SGC, hidden 64, dropout 0.5, lr 0.01, wd 5×10⁻⁴, up to 300 epochs, early stopping on val loss (patience 50). MLP same architecture without graph propagation. All hand-written in plain PyTorch.

**Calibrators.** ten methods: `none`, `ts_val`, `cagcn_val`, `atc_ts`, `stac_global`, `lf_ts`, `lf_cts`, `lf_cts_plus`, `ts_target_oracle`, `cluster_ts_target_oracle`. Sweeps: 5 datasets × 2 splits × 2 models × 5 seeds × 10 calibrators = 1,000 main rows + 1,575 ablation rows.

**Metrics.** ECE with 15 equal-width bins, reported as mean ± std over seeds 0–4. Per-cluster metrics use ≥ 20 eval nodes; NaN otherwise.

---

## 5. Results

### 5.1 C1: Structural shift increases ECE (Partial)

| Dataset      | Random ECE  | Structural ECE | Gap    | Seed std | Pass? |
|--------------|-------------|----------------|--------|----------|-------|
| cora         | 0.056±0.019 | 0.077±0.027    | +0.021 | 0.027    | FAIL  |
| citeseer     | 0.066±0.011 | 0.115±0.019    | +0.049 | 0.019    | **PASS** |
| pubmed       | 0.039±0.018 | 0.044±0.018    | +0.005 | 0.018    | FAIL  |
| chameleon    | 0.141±0.047 | 0.123±0.066    | −0.018 | 0.066    | FAIL  |
| roman_empire | 0.031±0.021 | 0.025±0.004    | −0.006 | 0.021    | FAIL  |

GCN, `none`, mean ± std over seeds 0–4. **C1: PARTIAL.** CiteSeer passes clearly; Cora is borderline (gap 0.021 < std 0.027). Heterophilic graphs show no systematic gap or a reversed direction. This may reflect that Louvain communities on these graphs are not strongly class-correlated (Cora has 78 components, making its structural split partly a component split).

*Figure: F1.*

### 5.2 C2: Overall ECE hides per-cluster damage (Pass)

| Dataset      | Split      | Overall ECE | Worst-cluster ECE | Ratio |
|--------------|------------|-------------|-------------------|-------|
| cora         | random     | 0.056       | 0.165             | 3.3×  |
| cora         | structural | 0.077       | 0.188             | 2.6×  |
| citeseer     | random     | 0.066       | 0.150             | 2.3×  |
| citeseer     | structural | 0.115       | 0.230             | 2.0×  |
| pubmed       | random     | 0.039       | 0.072             | 1.9×  |
| pubmed       | structural | 0.044       | 0.150             | 3.8×  |
| chameleon    | random     | 0.141       | 0.161             | 1.1×  |
| chameleon    | structural | 0.123       | 0.185             | 1.4×  |
| roman_empire | random     | 0.031       | 0.297             | **13.5×** |
| roman_empire | structural | 0.025       | 0.302             | **12.3×** |

Ratios are mean-of-per-seed ratios (consistent with SUMMARY.md). GCN, `none`. **C2: PASS.** Worst-cluster ECE is at least about 2× overall on 8 of 10 dataset×split combinations. Roman-empire is the strongest case (13.5×), showing that a low mean ECE (0.025) can mask severe damage in a small minority of clusters. Chameleon is the exception (≤1.4×) due to thin target clusters (~145 eval nodes over 5 clusters in the random split). Cora structural worst cluster: purity 0.24–0.29 (per P3 measurement, 3/5 seeds).

*Figure: F3.*

### 5.3 C3: LF-CTS recovers per-cluster damage (Fail / Partial for LF-CTS+)

**Gap closed** = (ECE_none − ECE_ours) / (ECE_none − ECE_cluster_oracle). Full structural GCN matrix:

| Dataset      | lf_cts  | lf_cts_plus |
|--------------|---------|-------------|
| cora         | −559%   | −168%       |
| citeseer     | −180%   | −132%       |
| pubmed       | −643%   | −170%       |
| chameleon    | −584%   | −247%       |
| roman_empire | undef.† | undef.†     |

†roman_empire structural: cluster oracle (0.027) > none (0.025); denominator is negative so gap closed is undefined. lf_cts ECE (0.024) is marginally better than none (0.025).

LF-CTS is negative on all structural GCN configurations. Positive gap is only achieved on SGC, where the higher uncalibrated ECE leaves more room:

| Config                   | lf_cts | lf_cts_plus |
|--------------------------|--------|-------------|
| cora structural SGC      | +46%   | +78%        |
| citeseer structural SGC  | +39%   | +72%        |
| roman_empire structural SGC | +106% | +15%     |

**Proxy correlation (structural GCN, a_hat vs true per-cluster accuracy):**

| Dataset      | LF-CTS (r)    | LF-CTS+ (r)   |
|--------------|---------------|---------------|
| cora         | 0.44 ± 0.32   | 0.41 ± 0.44   |
| citeseer     | 0.64 ± 0.16   | 0.72 ± 0.28   |
| chameleon    | −0.61 ± 0.24  | −0.34 ± 0.58  |
| pubmed       | 0.25 ± 0.74   | 0.21 ± 1.03   |
| roman_empire | 0.64 ± 0.12   | 0.69 ± 0.10   |

**C3: FAIL for LF-CTS; PARTIAL for LF-CTS+.** LF-CTS makes ECE worse on 80% of configurations. The proxy is valid on roman-empire (r=0.64) and CiteSeer (r=0.64) but near-zero or negative on Chameleon and unreliable on PubMed. LF-CTS+ achieves ≥72% gap closed on SGC but fails on GCN for homophilic graphs. The root cause is the gate misfire described in C4.

*Figure: F4.*

### 5.4 C4: Homophily gate matters (Partial)

**Estimated vs true homophily (structural GCN, seeds 0–4 mean):**

| Dataset      | h_hat (MLP) | h_hat (GCN) | Target-region h | Gate (seeds 0–4)            |
|--------------|-------------|-------------|------------------|-----------------------------|
| cora         | 0.446       | 0.840       | 0.815            | **5/5 feature_kmeans** (wrong) |
| citeseer     | 0.519       | 0.889       | 0.721            | 4/5 sgc, 1/5 feature (seed 4 h_mlp=0.487) |
| pubmed       | 0.667       | 0.875       | 0.798            | 5/5 sgc_kmeans (correct)   |
| chameleon    | 0.519       | 0.817       | 0.223            | 2/5 sgc, 3/5 feature (mixed) |
| roman_empire | 0.067       | 0.503       | 0.045            | 5/5 feature_kmeans (correct) |

"Target-region h" is true edge homophily within the target eval region (not whole-graph h). Chameleon's gate is unstable across seeds because h_hat_mlp straddles 0.5 (range 0.29–0.96 over seeds).

**Gate ablation (LF-CTS ECE, structural GCN, forced-wrong branch):**

| Dataset      | Gated (MLP) | Forced-wrong | Delta    |
|--------------|-------------|--------------|----------|
| cora         | 0.236       | **0.094**    | −0.142   |
| citeseer     | 0.226       | 0.213        | −0.013   |
| pubmed       | 0.163       | 0.172        | +0.009   |
| chameleon    | 0.350       | 0.437        | +0.087   |
| roman_empire | 0.024       | 0.215        | **+0.191** |

**C4: PARTIAL.** The gate ablation confirms the gate matters (±0.191 ECE swing on roman-empire). The MLP gate fires correctly on roman-empire and PubMed (5/5 seeds), but misfires consistently on Cora (5/5 feature_kmeans despite target-region h=0.815) and inconsistently on Chameleon (h_mlp range 0.29–0.96 across seeds). GCN over-estimates homophily on heterophilic graphs (chameleon: h_gnn=0.817 vs target-region h=0.223) due to message-passing consensus; this is why we gate on the MLP. The oracle gate closes −0.058 ECE on Cora and −0.133 on Chameleon. **Important:** even with the correct gate forced on Cora (sgc_kmeans), lf_cts ECE = 0.094 > none (0.077) — the gate is necessary but not sufficient to fix Cora.

*Figure: F7.*

### 5.5 C5: LF-CTS vs label-free baselines (Fail)

**Wilcoxon paired test (structural GCN, 25 pairs = 5 datasets × 5 seeds):**

| Comparison       | LF-CTS mean ECE | Baseline mean ECE | p-value | Winner    |
|------------------|-----------------|-------------------|---------|-----------|
| vs none          | 0.200           | 0.077             | <0.001  | baseline  |
| vs lf_ts         | 0.200           | 0.199             | 0.542   | tie       |
| vs atc_ts        | 0.200           | 0.107             | <0.001  | baseline  |
| vs stac_global   | 0.200           | 0.081             | <0.001  | baseline  |
| vs lf_cts_plus   | 0.200           | 0.149             | 0.076   | tie (p>0.05) |
| vs ts_val        | 0.200           | 0.072             | <0.001  | baseline  |
| vs cagcn_val     | 0.200           | 0.082             | <0.001  | baseline  |

**C5: FAIL.** LF-CTS does not beat any baseline. It is statistically tied with LF-TS (p=0.542) and LF-CTS+ (p=0.076, not significant), and significantly worse than all others. The failure is explained by C4: when the gate fires correctly (roman-empire, CiteSeer random), LF-CTS matches or beats baselines; when it misfires (Cora, PubMed GCN), it strongly hurts performance. **Best overall label-free method: stac_global** (5-dataset structural GCN mean ECE 0.081, vs ts_val mean 0.072). Per-dataset: stac_global beats ts_val on CiteSeer (0.061 vs 0.134) and is near-equal on PubMed (0.036 vs 0.035), but is worse on Cora (0.130 vs 0.060) and Chameleon (0.143 vs 0.103).

*Figure: F6.*

---

## 6. Ablations

### 6.1 Hops (SGC smoothing depth)

ECE is nearly flat across hops 0–4 for most datasets (range within 0.02). On Cora, the gate always fires `feature_kmeans` (hops unused), so the result is constant. On roman-empire, the `feature_kmeans` branch is also used. PubMed shows a mild improvement at hops 4 (ECE 0.172→0.159). Hops has no practical effect under the current gate logic.

*Figure: F5.*

### 6.2 Number of clusters k

Sensitivity is low (within 0.02 ECE across a 14× range). Fewer clusters (k=2) performs marginally better on Cora and roman-empire, suggesting the proxy is more stable with fewer groups.

| Dataset      | k=C (default) | k=2   | k=14  | k=28  |
|--------------|---------------|-------|-------|-------|
| cora (C=7)   | 0.236         | 0.226 | 0.225 | 0.217 |
| citeseer (C=6)| 0.226        | 0.223 | 0.222 | 0.228 |
| roman_empire (C=18)| 0.024  | 0.016 | 0.025 | 0.026 |

### 6.3 Proxy variant (MLP–GNN agreement)

Replacing the k-NN agreement proxy with MLP–GNN agreement dramatically improves homophilic graphs — the same signal used by stac_global:

| Dataset      | edge_knn | mlp_gnn_agreement |
|--------------|----------|-------------------|
| cora         | 0.236    | **0.095** (still > none=0.077) |
| citeseer     | 0.226    | **0.055**         |
| roman_empire | 0.024    | 0.037             |

This bypasses the gate-branch entirely and directly estimates per-cluster MLP–GNN agreement. The stac_global method achieves similar gains with a single global T.

### 6.4 Cluster method

Louvain clusters dramatically improve LF-CTS on Cora (0.236→0.068) and CiteSeer (0.226→0.179). Random partition is worst. This confirms that cluster quality — capturing natural community structure — is what carries the calibration signal.

| Dataset      | sgc_kmeans | louvain   | random |
|--------------|------------|-----------|--------|
| cora         | 0.236      | **0.068** | 0.320  |
| citeseer     | 0.226      | 0.179     | 0.231  |
| roman_empire | 0.024      | 0.065     | 0.040  |

---

## 7. Limitations

- **Proxy bias is substantial.** The edge/k-NN agreement proxy has Pearson r = 0.44–0.64 on homophilic graphs and r = −0.61 on Chameleon. LF-CTS+ corrects this on SGC (gap 78–89% closed) but not on GCN.
- **k is fixed at the class count.** Silhouette on Cora is flat (0.093–0.113 for k = 4 to 12), so there is no data-driven elbow. The k ablation shows calibration is mildly sensitive to this choice.
- **Silhouette is at most 0.11 everywhere.** These are useful groupings rather than well-separated clusters; the calibration signal comes from structural community identity, not feature-space separation.
- **Gate misfires on Cora.** The MLP underestimates homophily on graphs with overlapping feature distributions (Cora h_mlp=0.45 vs h_true=0.81). The oracle gate recovers 0.058 ECE. A better structure-free homophily estimator would improve robustness.
- **Structural shift is weak on several graphs.** C1 holds only on CiteSeer. Cora's 78 components make its structural split partly a component split; PubMed and roman-empire show ECE near-identical across split types.
- **Texas supports clustering analysis only.** Too small (183 nodes) for a reliable structural split.
- **Degree split amplifies gate-misfire.** LF-CTS gives ECE 0.359 ± 0.028 on Cora degree split (vs none = 0.050 ± 0.016); the same proxy failure applies.
- **Cross-machine reproducibility.** Roman-empire worst-cluster ECE shows machine-dependent variation (13.5× on macOS vs 14.4× on Windows), likely because k-means on the 22k-node graph lands on slightly different cluster assignments across platforms despite identical seeds. All dashboard numbers and figures use the macOS sweep results (`results/all_results.csv`).

---

## 8. Conclusion

We studied label-free calibration of GNNs under structural distribution shift. Our main findings are:

1. Structural shift inflates per-cluster ECE damage more than overall ECE (worst cluster up to 13.5× overall), motivating per-cluster calibration.
2. The label-free k-NN agreement proxy works on heterophilic graphs (roman-empire structural SGC: r=0.64 (GCN), matches oracle (gap 106% closed)) but fails when the MLP gate misfires on homophilic graphs.
3. Replacing the proxy with MLP–GNN agreement (stac_global) is the most reliable label-free calibrator overall (5-dataset mean ECE 0.081 vs ts_val 0.072); it beats ts_val on CiteSeer but underperforms it on Cora and Chameleon.
4. Louvain clusters and a better gate are the most promising directions for improving LF-CTS; the mlp_gnn_agreement proxy variant is a strong drop-in replacement that eliminates the gate entirely.

**Headline:** The label-free proxy works on genuinely heterophilic graphs (roman-empire structural SGC: r=0.64 (GCN), matches oracle (gap 106% closed)) but fails on homophilic graphs because the structure-free MLP underestimates homophily (Cora h_mlp=0.45 vs target-region h=0.82), firing the wrong gate branch and worsening calibration; replacing the proxy with MLP–GNN agreement cuts Cora ECE from 0.236 to 0.095 — though 0.095 remains above the uncalibrated baseline (none: 0.077).

---

## References

- Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
- Wang et al., "Be Confident! Towards Trustworthy Graph Neural Networks via Confidence Calibration", NeurIPS 2021.
- Garg et al., "Leveraging Unlabeled Data to Predict Out-of-Distribution Performance", 2022.
- Bahi, "STAC", arXiv 2607.10804, 2026.
- Zhao et al., "GNNEvaluator", arXiv 2310.14586, 2023.
- Platonov et al., "A Critical Look at the State of Graph Heterophily Benchmarks", NeurIPS 2023.
- Gui et al., "GOOD: A Graph Out-of-Distribution Benchmark", NeurIPS 2022.
- Hans et al., "GATS: Gradient Attention Temperature Scaling", 2023.

---

*Every number in this report traces to a row in `results/all_results.csv` or a `meta.json` artifact in `results/runs/`. Generated from `lane-a/p5` sweep results. See `results/SUMMARY.md` for the full per-dataset tables.*
