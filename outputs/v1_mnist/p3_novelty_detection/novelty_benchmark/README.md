# novelty_benchmark

**Step 3 of 3 in the proposal order: comparison with other novelty methods.**
This compares QE and the U-Matrix with Isolation Forest, One-Class SVM and an autoencoder (the three baselines the proposal names). It also holds the extra baselines the reviews asked for (max-softmax, energy, Mahalanobis, kNN) and the checks that explain why some baselines look odd on SVHN. The previous steps are in [`../qe_novelty_detection/`](../qe_novelty_detection/README.md) and [`../u_matrix_analysis/`](../u_matrix_analysis/README.md). The full list of run folders is in the [Phase 3 outputs README](../README.md).

## Read in this order

The proposal asks for AUROC, AUPR and FPR at 95% TPR for every method.

| # | The proposal asks | Folder | File |
|---|---|---|---|
| 1 | Compare U-Matrix scoring with QE scoring, and the three named baselines, on the same images | `mnist_svhn_five_method_comparison_s42_initial` | `comparison_metrics.json` (QE, U-Matrix, Isolation Forest, One-Class SVM, autoencoder) |
| 2 | Is the QE vs U-Matrix gap real, and does a random score do worse? | `mnist_svhn_qe_umatrix_random_s42_v1` | `qe_umatrix_random_metrics.json` |
| 3 | Add the stronger baselines: max-softmax, energy, Mahalanobis, kNN | `mnist_svhn_required_baselines_reproduction_initial`, `mnist_svhn_required_baselines_knn_s42_v1` | `required_baseline_metrics.json` in each |
| 4 | All methods side by side, with a random-score null | `mnist_svhn_unified_nine_methods_s42_46_v1` | `unified_nine_method_metrics.json`, `README.md` |
| 5 | Baselines given the same tuning budget | `mnist_svhn_fair_budget_baselines_s42_46_v1` | `fair_budget_metrics.json` |
| 6 | Paired differences with intervals: QE vs max-softmax (-0.0174), QE vs energy and kNN | `mnist_svhn_qe_msp_reproduction_s42_v1`, `mnist_svhn_qe_energy_knn_paired_s42_v1` | `qe_msp_paired_metrics.json`, `paired_comparison_metrics.json` |
| 7 | Failure modes: which SVHN images does QE miss? | `mnist_svhn_missed_qe_figure_s42_v1` | `figure_metrics.json` (the written analysis is in `../u_matrix_analysis/`, row 4) |
| 8 | All headline numbers in one table | `mnist_svhn_consolidated_evidence_s42_v1` | `phase3_consolidated_metrics.csv` |

## Why Isolation Forest and the autoencoder look odd on SVHN

They score SVHN as more normal than MNIST. These checks look into it.

| Check | Folder | File |
|---|---|---|
| Embedding norms of MNIST and SVHN | `svhn_inverted_embedding_norm_s42_v1` | `embedding_norm_metrics.json` |
| Autoencoder with and without standardised inputs, five seeds | `mnist_svhn_autoencoder_scaling_diagnostic_s42_46_v1` | `autoencoder_scaling_metrics.json`, `tables/` |
| Autoencoder trained on standardised embeddings | `mnist_svhn_ae_train_standardized_s42_v1` | `standardized_autoencoder_metrics.json` |

The written explanation is in `src/v1_mnist/docs/p3_novelty_detection/novelty_benchmark/BASELINE_SCORE_ANALYSIS.md`.

## Other folders

- `classical_`, `autoencoder_` and `mnist_validation_reproducibility_` folders (all `*_initial`) are the first baseline runs, kept as evidence.
- `mnist_svhn_energy_knn_score_export_s42_v1` saves the energy and kNN scores that the paired comparisons read. Its `.npy` files are tracked with DVC.
- `reorg_20261102_..._v1` folders repeat steps from the new code layout. Do not cite them as new results.
