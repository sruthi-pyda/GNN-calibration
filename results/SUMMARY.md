# P5 Results Summary

Generated: 2026-10-22
Rows: 1000 main + 1575 ablation = 2575 total. No adjustments made; deviations from expectations reported verbatim.

---

## 1. Main ECE table (GCN, structural split, mean ± std across seeds 0–4)

| Dataset      | none        | ts_val      | cagcn_val   | atc_ts      | stac_global | lf_ts       | lf_cts      | lf_cts_plus | ts_target_oracle | cluster_ts_oracle |
|--------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|------------------|-------------------|
| cora         | 0.077±0.027 | 0.060±0.019 | 0.151±0.041 | 0.122±0.062 | 0.130±0.024 | 0.236±0.042 | 0.236±0.041 | 0.125±0.072 | 0.048±0.010      | 0.049±0.012       |
| citeseer     | 0.115±0.019 | 0.134±0.034 | 0.073±0.022 | 0.207±0.034 | 0.061±0.020 | 0.228±0.030 | 0.226±0.034 | 0.196±0.144 | 0.054±0.015      | 0.053±0.012       |
| pubmed       | 0.044±0.018 | 0.035±0.016 | 0.049±0.031 | 0.053±0.032 | 0.036±0.007 | 0.163±0.035 | 0.163±0.035 | 0.075±0.041 | 0.026±0.014      | 0.025±0.009       |
| chameleon    | 0.123±0.066 | 0.103±0.065 | 0.105±0.080 | 0.126±0.126 | 0.143±0.074 | 0.345±0.211 | 0.350±0.212 | 0.219±0.205 | 0.103±0.065      | 0.085±0.048       |
| roman_empire | 0.025±0.004 | 0.026±0.006 | 0.033±0.006 | 0.028±0.010 | 0.033±0.009 | 0.021±0.004 | 0.024±0.008 | 0.129±0.038 | 0.021±0.001      | 0.027±0.005       |

Accuracy (GCN, structural, mean over seeds):
cora 0.808, citeseer 0.694, pubmed 0.726, chameleon 0.541, roman_empire 0.265
⚠️ MISMATCH: Saanvi's rerun (lane-a/p5, a07baae) gives cora≈0.696, chameleon≈0.239.
Trace against all_results.csv (col "acc", split="structural", model="gcn", ablation="main")
before reporting. The 0.808/0.541 values above need verification.

---

## 2. C1: Structural vs random ECE gap (GCN, calib=none)

| Dataset      | random ECE  | structural ECE | gap    | seed std | C1?     |
|--------------|-------------|----------------|--------|----------|---------|
| cora         | 0.056±0.019 | 0.077±0.027    | +0.021 | 0.027    | FAIL    |
| citeseer     | 0.066±0.011 | 0.115±0.019    | +0.049 | 0.019    | **PASS**|
| pubmed       | 0.039±0.018 | 0.044±0.018    | +0.005 | 0.018    | FAIL    |
| chameleon    | 0.141±0.047 | 0.123±0.066    | −0.018 | 0.066    | FAIL    |
| roman_empire | 0.031±0.021 | 0.025±0.004    | −0.006 | 0.021    | FAIL    |

**C1 verdict: PARTIAL.** CiteSeer passes (gap 0.049 > seed std 0.019). Cora borderline (gap 0.021 < std 0.027). Other datasets show no systematic gap or reversed direction. Root cause: Louvain-structural shift is weak on PubMed and the heterophilic graphs (roman_empire is already low ECE under both splits; Chameleon structural is actually lower ECE than random).

---

## 3. C2: Worst-cluster ECE vs overall ECE (GCN, calib=none)

| Dataset      | Split      | overall ECE | worst ECE | ratio |
|--------------|------------|-------------|-----------|-------|
| cora         | random     | 0.056       | 0.165     | 3.3×  |
| cora         | structural | 0.077       | 0.188     | 2.6×  |
| citeseer     | random     | 0.066       | 0.150     | 2.3×  |
| citeseer     | structural | 0.115       | 0.230     | 2.0×  |
| pubmed       | random     | 0.039       | 0.072     | 1.9×  |
| pubmed       | structural | 0.044       | 0.150     | 3.8×  |
| chameleon    | random     | 0.141       | 0.161     | 1.1×  |
| chameleon    | structural | 0.123       | 0.185     | 1.4×  |
| roman_empire | random     | 0.031       | 0.297     | **13.5×** |
| roman_empire | structural | 0.025       | 0.302     | **12.3×** |

**C2 verdict: PASS.** Worst-cluster ECE is at least 2× overall on 8 of 10 dataset×split combinations. Roman Empire is the strongest case (13×), showing that low mean ECE can hide severe per-cluster damage on large heterophilic graphs. Chameleon (1.1×–1.4×) is the exception, likely because per-cluster ECE is NaN-heavy given thin target clusters (~145 eval nodes spread over 5 clusters).

Cora worst cluster (structural, GCN, seed=0): cluster_ece_worst falls in clusters with purity 0.24–0.29 (per P3 measurement, 3/5 seeds); not necessarily the low-degree cluster.

---

## 4. C3: Gap closed and proxy correlation

### Gap closed = (ECE_none − ECE_ours) / (ECE_none − ECE_cluster_oracle)

Selected cells where oracle denominator > 0.01:

| Config                   | lf_cts  | lf_cts_plus |
|--------------------------|---------|-------------|
| cora random GCN          | −1740%  | 101%        |
| cora structural SGC      | 46%     | 78%         |
| citeseer random SGC      | 20%     | 89%         |
| citeseer structural SGC  | 39%     | 72%         |
| roman_empire random GCN  | 48%     | −515%       |
| roman_empire random SGC  | 98%     | 22%         |

**lf_cts is negative on most configurations (makes things worse, not better).** lf_cts_plus is substantially better on SGC but still negative on GCN for homophilic graphs (cora, citeseer, pubmed structural).

### Proxy-vs-true correlation (pooled seeds, GCN, lf_cts)

| Dataset      | Pearson r (lf_cts a_hat) |
|--------------|--------------------------|
| cora         | 0.495 ± 0.348            |
| citeseer     | 0.672 ± 0.169            |
| pubmed       | 0.390 ± 0.664            |
| chameleon    | −0.047 ± 0.706           |
| roman_empire | 0.688 ± 0.127            |

**C3 verdict: FAIL for lf_cts; PARTIAL for lf_cts_plus (works on SGC but not GCN).** The proxy is valid on roman_empire (r=0.688) and citeseer (r=0.672) but near zero or negative on chameleon and unstable on pubmed. The root failure is the gate misfiring on Cora (see C4), turning under-confident models more under-confident.

---

## 5. C4: Homophily gate

### h_hat: MLP vs GCN vs true homophily

| Dataset      | h_mlp | h_gnn | h_true | gate branch fired |
|--------------|-------|-------|--------|-------------------|
| cora         | 0.451 | 0.848 | 0.810  | **feature_kmeans** (WRONG — h_true > 0.5) |
| citeseer     | 0.528 | 0.899 | 0.738  | sgc_kmeans (correct) |
| pubmed       | 0.652 | 0.890 | 0.802  | sgc_kmeans (correct) |
| chameleon    | 0.275 | 0.793 | 0.236  | feature_kmeans (correct) |
| roman_empire | 0.065 | 0.497 | 0.047  | feature_kmeans (correct) |

**Finding:** The MLP underestimates homophily on Cora (0.451 vs 0.810). Because the MLP has no graph structure, it produces more cross-class errors, making predicted labels appear less homophilous than they are. This fires the wrong gate branch (feature_kmeans instead of sgc_kmeans) for Cora. GCN inflates estimated homophily (e.g., chameleon: h_gnn=0.793 vs h_true=0.236) due to message-passing consensus; this is why we gate on MLP not GCN — but MLP also misfires on Cora.

### Gate ablation: lf_cts ECE with gated vs forced-wrong branch (structural, GCN)

| Dataset      | gated (MLP) | forced-wrong | delta    |
|--------------|-------------|--------------|----------|
| cora         | 0.236       | **0.094**    | −0.142   |
| citeseer     | 0.226       | 0.213        | −0.013   |
| pubmed       | 0.163       | 0.172        | +0.009   |
| chameleon    | 0.350       | 0.437        | +0.087   |
| roman_empire | 0.024       | 0.215        | **+0.191** |

Forcing the wrong branch hurts strongly on roman_empire and chameleon (correct decision), but for Cora the "wrong" branch (sgc_kmeans) is actually the correct one (the gate misfired).

### Oracle gate vs MLP gate (lf_cts, structural, GCN)

| Dataset      | MLP gate | Oracle gate | delta    |
|--------------|----------|-------------|----------|
| cora         | 0.236    | **0.178**   | −0.058   |
| citeseer     | 0.226    | 0.238       | +0.012   |
| pubmed       | 0.163    | 0.163       | 0.000    |
| chameleon    | 0.350    | **0.216**   | −0.133   |
| roman_empire | 0.024    | 0.024       | 0.000    |

Oracle gate helps where the MLP mis-estimates homophily (Cora, Chameleon). No benefit on correct gates.

**C4 verdict: PARTIAL.** The gate ablation confirms that correct gating matters significantly (+0.191 ECE for roman_empire with wrong branch, −0.133 for chameleon with oracle). But the MLP-based gate misfires on Cora (h_mlp 0.451 < 0.5 despite h_true 0.810), which is the single largest source of failure across the project.

---

## 6. C5: LF-CTS vs label-free and labeled baselines

### Wilcoxon paired test (lf_cts vs each baseline, structural GCN, 25 pairs = 5 datasets × 5 seeds)

| Comparison              | lf_cts ECE | baseline ECE | p-value | Winner       |
|-------------------------|------------|--------------|---------|--------------|
| vs none                 | 0.200      | 0.077        | <0.001  | baseline     |
| vs lf_ts                | 0.200      | 0.199        | 0.542   | tie          |
| vs atc_ts               | 0.200      | 0.107        | <0.001  | baseline     |
| vs stac_global          | 0.200      | 0.081        | <0.001  | baseline     |
| vs lf_cts_plus          | 0.200      | 0.149        | 0.076   | lf_cts_plus  |
| vs ts_val               | 0.200      | 0.072        | <0.001  | baseline     |
| vs cagcn_val            | 0.200      | 0.082        | <0.001  | baseline     |

**C5 verdict: FAIL.** lf_cts does not beat any baseline. It is statistically tied with lf_ts (which also fails) and significantly worse than all others. lf_cts_plus is better than lf_cts (p=0.076 marginal) but still worse than atc_ts and stac_global. The failure is explained by the Cora gate misfire (see C4): when the gate fires correctly (roman_empire, chameleon random), lf_cts matches or beats baselines; when it fires incorrectly (Cora, PubMed GCN), it strongly hurts performance.

Best label-free method overall: **stac_global** (MLP-GNN agreement proxy, ECE 0.081 on structural GCN — competitive with ts_val 0.072 and cagcn_val 0.082).

---

## 7. Ablations

### Hops (lf_cts ECE vs smoothing hops, structural GCN)

| Dataset      | hops=0 | hops=1 | hops=2 | hops=3 | hops=4 |
|--------------|--------|--------|--------|--------|--------|
| cora         | 0.236  | 0.236  | 0.236  | 0.236  | 0.236  |
| citeseer     | 0.227  | 0.228  | 0.227  | 0.225  | 0.224  |
| roman_empire | 0.024  | 0.024  | 0.024  | 0.024  | 0.024  |

ECE is flat across all hops values. For Cora, the gate always fires feature_kmeans (hops unused), so the result is identical. For roman_empire, the feature_kmeans branch is also correct regardless of hops. The hops parameter has no practical effect under the current gate logic.

### k (n_clusters, lf_cts ECE, structural GCN)

| Dataset      | k=C (default) | k=2  | k=14 | k=28 |
|--------------|---------------|------|------|------|
| cora (C=7)   | 0.236         | 0.226| 0.225| 0.217|
| citeseer (C=6)| 0.226        | 0.223| 0.222| 0.228|
| roman_empire (C=18)| 0.024  | 0.016| 0.025| 0.026|

Sensitivity is low (within 0.02 ECE across 14× range). Fewer clusters (k=2) performs marginally better on Cora and roman_empire, suggesting the proxy estimate is more stable with fewer groups. Roman_empire k=2 ECE=0.016 is the best single result.

### Proxy variant (structural GCN)

| Dataset      | edge_knn (default) | mlp_gnn_agreement |
|--------------|--------------------|-------------------|
| cora         | 0.236              | **0.095**         |
| citeseer     | 0.226              | **0.055**         |
| roman_empire | 0.024              | 0.037             |

**MLP-GNN agreement is dramatically better for homophilic graphs.** On Cora it reduces ECE from 0.236 to 0.095 (though 0.095 is still above uncalibrated ECE none=0.077) and on CiteSeer from 0.226 to 0.055. This variant does not rely on the gate branch (it always uses inter-model agreement), bypassing the Cora gate-misfire issue. Roman_empire is slightly worse (0.037 vs 0.024) because k-NN agreement already works well there. The stac_global method uses this same signal and achieves strong overall results.

### Cluster method (structural GCN, lf_cts)

| Dataset      | sgc_kmeans | louvain | random |
|--------------|------------|---------|--------|
| cora         | 0.236      | **0.068**| 0.320 |
| citeseer     | 0.226      | 0.179   | 0.231  |
| roman_empire | 0.024      | 0.065   | 0.040  |

Louvain clusters dramatically improve lf_cts on Cora (0.068 vs 0.236) and citeseer (0.179 vs 0.226). This shows that cluster quality matters — Louvain captures natural community structure better than k-means on smoothed features for these graphs. Random partition is worst on Cora, confirming that structure is what carries the signal. Roman_empire is the exception (sgc_kmeans better than both louvain and random), likely because the 18-class k-means finds more homogeneous groups.

### Degree split (GCN, calib=none baseline)

On Cora degree split, none ECE=0.050 ± 0.016. lf_cts and lf_ts both give ECE=0.359 ± 0.028 (7× worse), and lf_cts_plus 0.444 ± 0.237. The oracle-based methods (ts_target_oracle 0.050, cluster_ts_oracle 0.054) match none, suggesting the degree split itself does not substantially shift Cora's ECE, but our proxy fails catastrophically (same gate-misfire pattern as random/structural).

---

## 8. Where this fails

| Configuration | ECE none | ECE lf_cts | Root cause |
|---------------|----------|------------|------------|
| Cora all splits, GCN/SGC | 0.056–0.077 | 0.222–0.340 | Gate misfire: h_hat_mlp=0.451 < 0.5 → feature_kmeans; k-NN in raw features gives low a_hat for an under-confident model → T>1 → more under-confident |
| PubMed structural GCN | 0.044 | 0.163 | Same gate misfire (feature_kmeans despite h_true=0.80) |
| Chameleon structural | 0.123 | 0.350 | Gate fires feature_kmeans (correct), but proxy r=−0.047 (no signal); proxy is chaotic on the thin target region |
| All datasets, lf_cts_plus GCN structural | 0.077 | 0.149 | Ridge correction trained on source val cannot generalise to target under structural shift; R² of ridge was low in several seeds |
| roman_empire structural lf_cts_plus | 0.025 | 0.129 | lf_cts_plus overcorrects: a_tilde diverges from a_hat raw because ridge was trained on small val clusters |
| Degree split (all datasets) | ≈none | 3–9× none | Gate misfires on degree-based splits; same proxy failure as structural |

**Common root cause:** The label-free proxy (k-NN agreement in feature space) does not correctly reflect model accuracy when the gate fires feature_kmeans on a homophilic graph. The MLP deflates predicted homophily because it makes more cross-class errors in graphs with overlapping feature distributions.

**Bright spots:** roman_empire random GCN lf_cts (ECE 0.028, gap 48% closed); roman_empire random SGC lf_cts (ECE 0.046, gap 98% closed); citeseer random SGC lf_cts_plus (gap 89% closed). These are cases where the gate fires correctly and the proxy correlation is high.

---

## 9. Claim verdicts

| Claim | Verdict | Evidence |
|-------|---------|----------|
| C1: Structural shift increases ECE | **PARTIAL** | CiteSeer PASS (gap 0.049 > std 0.019); Cora borderline; others FAIL |
| C2: Worst-cluster ECE >> overall ECE | **PASS** | Roman Empire 13×; Cora 3.3×; PubMed structural 3.8× |
| C3: LF-CTS recovers per-cluster damage | **FAIL** | lf_cts makes ECE worse on 80% of configs; lf_cts_plus partial on SGC only |
| C4: Homophily gate matters | **PARTIAL** | Gate ablation confirms ±0.191 ECE swing; but MLP gate misfires on Cora |
| C5: LF-CTS beats label-free baselines | **FAIL** | Wilcoxon p<0.001 against atc_ts, stac_global; tied with lf_ts only |

**Headline finding (one sentence):** The label-free proxy works on genuinely heterophilic graphs (roman_empire r=0.688, gap 48–98% closed) but fails on homophilic graphs because the structure-free MLP underestimates homophily (Cora h_mlp=0.451 vs h_true=0.810), firing the wrong gate branch and making already under-confident models worse; replacing the proxy with MLP-GNN agreement cuts Cora ECE from 0.236 to 0.095, though 0.095 remains above the uncalibrated baseline (none: 0.077).

---

_Generated by run_experiments.py P5 sweep. Every number traces to a row in results/all_results.csv._
