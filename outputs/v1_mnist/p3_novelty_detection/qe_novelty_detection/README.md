# qe_novelty_detection

**Step 1 of 3 in the proposal order: novelty detection with quantization error (QE).**
QE is how far an image sits from its closest SOM neuron. MNIST validation is the in-distribution set. SVHN is the out-of-distribution set. The next steps are in [`../u_matrix_analysis/`](../u_matrix_analysis/README.md) and [`../novelty_benchmark/`](../novelty_benchmark/README.md). The full list of run folders is in the [Phase 3 outputs README](../README.md).

## Read in this order

Each row is one thing the proposal asks for in this step.

| # | The proposal asks | Folder | File |
|---|---|---|---|
| 1 | Compute QE for each sample and fit a cutoff on the training QE (95th percentile) | `qe_training_calibrated_initial` | `qe_summary.json` (cutoff, how many training and validation images are flagged, flagged vs CNN errors) |
| 2 | Flag images above the cutoff and measure the SVHN detection rate | `mnist_svhn_qe_frozen_s42_initial` | `qe_ood_metrics.json` (AUROC 0.981 with interval, AUPR, FPR at 95% TPR, detection rate, false-alarm rate) |
| 3 | Check how the cutoff choice matters (90th to 99.5th percentile) | `mnist_svhn_qe_threshold_sweep_s42_v1` | `qe_threshold_sweep.json`, `tables/qe_threshold_sweep.csv` |
| 4 | Check that QE beats a random score (control) | `qe_random_control_evaluation_initial` | `qe_random_control_metrics.json` (errors captured by QE vs random, Wilson interval, p-value) |

## Ignore unless you are checking reproducibility

The `reorg_20261101_..._v1` and `reorg_20261102_..._v2` folders repeat steps 1, 2 and 4 from the new code layout. They show the code reproduces the earlier results. Do not cite them as new results.
