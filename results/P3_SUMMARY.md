# P3 summary - uncalibrated GCN (50 rows)

Every number is computed from `results/all_results.csv` rows (calib_method=none, ablation=main) and their `results/runs/<run_id>/meta.json`. Metrics are on target_eval only. Mean ± std over seeds (population std).

## (a) ECE structural vs random

```
     dataset seeds    ece_random ece_structural    gap  max_seed_std beyond_std
   chameleon   5/5 0.141 ± 0.042  0.123 ± 0.059 -0.018         0.059         no
    citeseer   5/5 0.066 ± 0.010  0.115 ± 0.017  0.049         0.017        yes
        cora   5/5 0.056 ± 0.017  0.077 ± 0.024  0.021         0.024         no
      pubmed   5/5 0.039 ± 0.016  0.044 ± 0.016  0.005         0.016         no
roman_empire   5/5 0.031 ± 0.019  0.025 ± 0.004 -0.006         0.019         no
```

## (b) cluster_ece_worst / ece

```
                          mean   min    max
dataset      split                         
chameleon    random       1.15  0.91   1.38
             structural   1.43  1.01   1.72
citeseer     random       2.25  1.93   2.60
             structural   2.03  1.65   2.79
cora         random       3.25  1.97   5.73
             structural   2.65  1.73   3.66
pubmed       random       1.92  1.45   2.77
             structural   3.83  1.38   7.25
roman_empire random      14.38  3.58  27.46
             structural  12.94  9.22  15.23
```

## (c) Cora structural: worst cluster (profile uses true labels - evaluation only)

```
 seed         method  worst  worst_ece  overall  n_eval  size  purity  conductance  avg_degree  inside_h  lowest_purity  lowest_degree
    0 feature_kmeans      6      0.198    0.079      26   141    0.83         0.52        3.81      0.96              0              2
    1 feature_kmeans      1      0.186    0.058      37   102    0.29         0.89        3.83      0.77              2              6
    2 feature_kmeans      1      0.195    0.091      71   306    0.24         0.79        3.56      0.92              5              2
    3 feature_kmeans      1      0.197    0.114     106   334    0.25         0.72        3.69      0.80              3              0
    4 feature_kmeans      6      0.165    0.045      23   102    0.29         0.89        5.31      0.87              5              2
```

## (d) accuracy random vs structural (mean over seeds)

```
split         random  structural  drop_points  collapse_gt_15
dataset                                                      
chameleon      0.339       0.239         10.0           False
citeseer       0.681       0.674          0.6           False
cora           0.797       0.696         10.0           False
pubmed         0.782       0.724          5.8           False
roman_empire   0.270       0.270         -0.0           False
```

## (e) homophily estimates and gate decisions (h_true is evaluation only)

```
     dataset  h_true     h_hat_mlp     h_hat_gnn       mlp_acc                               gate_mlp gate_true                 gate_gnn
   chameleon   0.236 0.397 ± 0.209 0.805 ± 0.070 0.211 ± 0.065 {'feature_kmeans': 8, 'sgc_kmeans': 2}   feature              {'sgc': 10}
    citeseer   0.738 0.523 ± 0.029 0.894 ± 0.013 0.560 ± 0.021 {'sgc_kmeans': 8, 'feature_kmeans': 2}       sgc              {'sgc': 10}
        cora   0.810 0.448 ± 0.014 0.844 ± 0.013 0.526 ± 0.047                 {'feature_kmeans': 10}       sgc              {'sgc': 10}
      pubmed   0.802 0.659 ± 0.049 0.883 ± 0.024 0.660 ± 0.040                     {'sgc_kmeans': 10}       sgc              {'sgc': 10}
roman_empire   0.047 0.066 ± 0.003 0.500 ± 0.008 0.469 ± 0.018                 {'feature_kmeans': 10}   feature {'feature': 5, 'sgc': 5}
```
