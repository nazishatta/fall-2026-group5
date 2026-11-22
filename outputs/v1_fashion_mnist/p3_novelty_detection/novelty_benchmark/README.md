# `novelty_benchmark` (Fashion-MNIST)

**Step 3 of 3: comparison with other novelty methods**, on the same 10,500 validation and 26,032 SVHN images. Previous steps: [`../qe_novelty_detection/`](../qe_novelty_detection/README.md) and [`../u_matrix_analysis/`](../u_matrix_analysis/README.md). Which code wrote each folder: [`../run_manifest.csv`](../run_manifest.csv). Protocol: `src/v1_fashion_mnist/docs/p3_novelty_detection/NOVELTY_DETECTION_PROTOCOL.md`; results in its section 7.

| # | What it answers | Folder | File |
|---|---|---|---|
| 1 | **The protocol's statements O1 to O4 and the proposal's RQ3 and RQ4**: all six scores, intervals, paired differences, label-shuffle control | `fashion_mnist_svhn_novelty_statements_s42_v1` | `novelty_statements.json`, `tables/scores.csv`, `tables/paired_against_qe.csv`; score arrays in `scores/` (DVC) |
| 2 | QE and U-Matrix against the proposal's three baselines (Isolation Forest, One-Class SVM, autoencoder) | `fashion_mnist_svhn_five_method_comparison_s42_v1` | `comparison_metrics.json`; score arrays (DVC) |
| 3 | Max-softmax, energy and Mahalanobis; then with nearest neighbours | `fashion_mnist_svhn_required_baselines_s42_v1`, `fashion_mnist_svhn_required_baselines_knn_s42_v1` | `required_baseline_metrics.json` in each |
| 4 | Energy and nearest-neighbour score arrays for the paired comparisons | `fashion_mnist_svhn_energy_knn_score_export_s42_v1` | `score_export_metrics.json`; score arrays (DVC) |
| 5 | Paired differences: QE against max-softmax; QE against energy and nearest neighbours; QE against the U-Matrix with a random-score control | `fashion_mnist_svhn_qe_msp_paired_s42_v1`, `fashion_mnist_svhn_qe_energy_knn_paired_s42_v1`, `fashion_mnist_svhn_qe_umatrix_random_s42_v1` | `qe_msp_paired_metrics.json`, `paired_comparison_metrics.json`, `qe_umatrix_random_metrics.json` |
| 6 | Baselines with the same training budget (5 seeds); all nine methods side by side | `fashion_mnist_svhn_fair_budget_baselines_s42_46_v1`, `fashion_mnist_svhn_unified_nine_methods_s42_46_v1` | `fair_budget_metrics.json`, `unified_nine_method_metrics.json` |
| 7 | Which SVHN images QE misses | `fashion_mnist_svhn_missed_qe_figure_s42_v1` | `figure_metrics.json`, `pdf/` (DVC) |

## Why Isolation Forest and the autoencoder look odd on SVHN

Isolation Forest scores SVHN as more normal than Fashion-MNIST, and the autoencoder changes strongly with its input scaling, as on MNIST. These checks look into it.

| Check | Folder | File |
|---|---|---|
| Does QE follow the embedding length? (correlation, and AUROC among images of Fashion-MNIST-like length) | `fashion_mnist_svhn_qe_norm_dependence_s42_v1` | `qe_norm_dependence_metrics.json`, `tables/qe_auroc_by_norm_subset.csv` |
| Autoencoder trained on standardised embeddings | `fashion_mnist_svhn_ae_train_standardized_s42_v1` | `standardized_autoencoder_metrics.json` |
| Autoencoder and Isolation Forest by training size and scaling, with a length-only control (triggered diagnostic) | `fashion_mnist_svhn_autoencoder_scaling_diagnostic_s42_46_v1` | `autoencoder_scaling_metrics.json`, `tables/` |
| SVHN embedding length with the grey levels inverted (triggered diagnostic) | `fashion_mnist_svhn_inverted_embedding_norm_s42_v1` | `embedding_norm_metrics.json` |

Main result: SOM QE AUROC 0.789 against energy 0.982, max-softmax 0.884, nearest neighbours 0.864, U-Matrix 0.812, Mahalanobis 0.713. Embedding length alone separates SVHN from Fashion-MNIST almost perfectly (AUROC 0.995 for "shorter is more novel").
