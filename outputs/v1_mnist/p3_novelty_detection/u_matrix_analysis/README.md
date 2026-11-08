# u_matrix_analysis

**Step 2 of 3 in the proposal order: U-Matrix and topological analysis.**
The U-Matrix shows distances between neighbouring SOM neurons. High values mark cluster boundaries. This folder also shows where SVHN images land on the 15x15 map. The previous step is in [`../qe_novelty_detection/`](../qe_novelty_detection/README.md) and the next is in [`../novelty_benchmark/`](../novelty_benchmark/README.md). The full list of run folders is in the [Phase 3 outputs README](../README.md).

## Read in this order

| # | The proposal asks | Folder | File |
|---|---|---|---|
| 1 | Compute the U-Matrix from the trained SOM and compare it with QE | `qe_u_matrix_validation_s42_initial` | `qe_u_matrix_validation_initial.json` (mean QE, mean U-Matrix value, correlation 0.44; validation only, no SVHN here) |
| 2 | Show the U-Matrix on the grid | `mnist_umatrix_hex_grid_s42_v1`, `mnist_qe_umatrix_validation_figure_s42_v1` | `figure_metrics.json` (the figure PDFs are tracked with DVC) |
| 3 | Show novelty scores overlaid on the SOM grid | `mnist_svhn_som_overlay_s42_v1` | `analysis.json`, `tables/svhn_neuron_summary.csv`, `tables/svhn_class_landing_profile.csv` |
| 4 | Say which SVHN images the map misses, and where | `mnist_svhn_som_overlay_s42_v1`, `mnist_svhn_proposal_gap_analysis_s42_v1` | `analysis.json` in each (996 missed of 26,032; neurons that hold the misses) |

The U-Matrix detection scores on SVHN (AUROC 0.970) and the paired QE vs U-Matrix difference are not stored here. They are in [`../novelty_benchmark/`](../novelty_benchmark/README.md), rows 1 and 2.

Large files (figure PDFs, assignment arrays) are tracked with DVC and appear here as `.dvc` pointers. Run `dvc pull` to get them.

## Ignore unless you are checking reproducibility

The `reorg_20261101_..._v1` and `reorg_20261102_..._v1/_v2` folders repeat the steps above from the new code layout. Do not cite them as new results.
