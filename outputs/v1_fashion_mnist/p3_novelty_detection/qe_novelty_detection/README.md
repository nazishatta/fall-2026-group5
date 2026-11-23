# `qe_novelty_detection` (Fashion-MNIST)

**Step 1 of 3: novelty detection with quantization error (QE).** QE is how far an image sits from its closest SOM neuron. Fashion-MNIST validation is the in-distribution set; SVHN test is the outside set. Next steps: [`../u_matrix_analysis/`](../u_matrix_analysis/README.md) and [`../novelty_benchmark/`](../novelty_benchmark/README.md). Which code wrote each folder: [`../run_manifest.csv`](../run_manifest.csv).

| # | The proposal asks | Folder | File |
|---|---|---|---|
| 1 | Compute QE for each sample and fix a cut-off on the training QE (95th percentile) | `fashion_mnist_qe_training_calibrated_s42_v1` | `qe_summary.json` (cut-off 1.4215, flagged images, flagged against CNN errors); score-distribution figure in `pdf/` (DVC) |
| 2 | Flag images above the cut-off and measure SVHN detection | `fashion_mnist_svhn_qe_s42_v1` | `qe_ood_metrics.json` (AUROC 0.789 with interval, AUPR, FPR at 95% TPR, detection rate 3.0%, false-alarm rate 5.1%) |
| 3 | How the cut-off choice matters (90th to 99.5th percentile) | `fashion_mnist_svhn_qe_threshold_sweep_s42_v1` | `qe_threshold_sweep.json`, `tables/qe_threshold_sweep.csv` |
| 4 | Check QE against random picks (control) | `fashion_mnist_qe_random_control_s42_v1` | `qe_random_control_metrics.json` (CNN errors among QE-flagged against randomly picked validation images) |

The frozen cut-off flags fewer SVHN images (3.0%) than Fashion-MNIST validation images (5.1%); see the Phase 3 protocol, section 7.
