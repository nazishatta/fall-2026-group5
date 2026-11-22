# `u_matrix_analysis` (Fashion-MNIST)

**Step 2 of 3: U-Matrix and topological analysis.** The U-Matrix shows distances between neighbouring neurons; high values mark cluster borders. This folder also shows where SVHN images land on the 15x15 map. Previous step: [`../qe_novelty_detection/`](../qe_novelty_detection/README.md); next: [`../novelty_benchmark/`](../novelty_benchmark/README.md). Which code wrote each folder: [`../run_manifest.csv`](../run_manifest.csv).

| # | The proposal asks | Folder | File |
|---|---|---|---|
| 1 | Compute the U-Matrix and compare it with QE (validation only) | `fashion_mnist_qe_u_matrix_validation_s42_v1` | `qe_u_matrix_metrics.json` (mean QE 0.853, mean U-Matrix 0.698, Spearman correlation 0.383) |
| 2 | Show the U-Matrix on the grid | `fashion_mnist_umatrix_hex_grid_s42_v1`, `fashion_mnist_qe_umatrix_validation_figure_s42_v1` | `figure_metrics.json`, `pdf/` (DVC) |
| 3 | Show novelty scores on the SOM grid | `fashion_mnist_svhn_som_overlay_s42_v1` | `analysis.json`, `tables/svhn_neuron_summary.csv`, `tables/svhn_class_landing_profile.csv`, `pdf/` and `assignments/` (DVC) |
| 4 | Which SVHN images the map misses, and where | `fashion_mnist_svhn_som_overlay_s42_v1`, `fashion_mnist_svhn_proposal_gap_analysis_s42_v1` | `analysis.json` in each (25,240 of 26,032 missed at the frozen cut-off), `pdf/` (DVC) |

The U-Matrix detection score on SVHN (AUROC 0.812) and its paired difference with QE are in [`../novelty_benchmark/`](../novelty_benchmark/README.md).
