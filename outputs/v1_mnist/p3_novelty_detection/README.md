# Phase 3: Novelty Detection

> `p3` in the folder name means **Phase 3**.

Flagging out-of-distribution samples with the SOM. Phase 3 novelty detection results include QE, U-Matrix, SVHN benchmark comparisons, random controls, baseline evaluations, uncertainty estimates, and PDF figures. Results are organized into separate immutable run-ID directories. Historical evidence is preserved without overwriting runs.

```
p3_novelty_detection/
├── qe_novelty_detection/          # quantization-error novelty scores
├── u_matrix_analysis/             # U-Matrix novelty scores and figures
└── novelty_benchmark/             # comparison with the other novelty methods
```

## Which step wrote which folder

Each row is one run folder. The step names are the ones `run_all.py --to novelty_detection` (or the three runners in `src/v1_mnist/component/pipeline/p3_novelty_detection/`) use; the config is in `src/v1_mnist/component/configs/`. Nothing here is renamed or moved, because result files record the run ID they were written under.

Status meanings:

- **current**: a step creates this folder from its config.
- **reorganisation run**: repeat run made while the code was reorganised (names start with `reorg_` and a date). Kept as evidence that the code reproduces the earlier results. These names describe an event, not the content; see `src/v1_mnist/docs/DEVIATIONS.md`.
- **historical**: written before the current configs; no config creates it any more.
- **not yet run**: the config exists, the folder does not. New analyses run on EC2. The three reproduction steps are skipped by `run_all.py` unless `--with-reproductions` is given, because the reorganisation runs already reproduced the originals.


### `qe_novelty_detection/` (QE novelty scores; runner `run_qe_novelty_detection.py`)

| Run folder | Step | Config | Status |
|---|---|---|---|
| `mnist_svhn_qe_dev_initial` | `qe_ood` | `qe_ood_evaluation.yaml` | reproduction, not yet run (skipped unless `--with-reproductions`) |
| `mnist_svhn_qe_frozen_s42_initial` | - | - | historical (kept as evidence); original run that step `qe_ood` repeats |
| `mnist_svhn_qe_threshold_sweep_s42_v1` | `qe_threshold_sweep` | `qe_threshold_sweep.yaml` | current (run on EC2, 2026-11-03) |
| `qe_random_control_evaluation_initial` | - | - | historical (kept as evidence); original run that step `qe_random_control` repeats |
| `qe_random_control_reproduction_initial` | `qe_random_control` | `qe_random_control.yaml` | reproduction, not yet run (skipped unless `--with-reproductions`) |
| `qe_training_calibrated_initial` | `qe_novelty` | `qe_novelty.yaml` | current |
| `reorg_20261101_qe_novelty_s42_v1` | - | `phase3_reorganization/qe_novelty.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261101_qe_ood_s42_v1` | - | `phase3_reorganization/qe_ood_evaluation.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261101_random_control_s42_v1` | - | `phase3_reorganization/qe_random_control.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261102_qe_novelty_s42_v2` | - | `phase3_reorganization/reorg_20261102/qe_novelty_rerun.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261102_qe_ood_s42_v2` | - | `phase3_reorganization/reorg_20261102/qe_ood_rerun.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261102_random_control_s42_v2` | - | `phase3_reorganization/reorg_20261102/random_control_rerun.yaml` | reorganisation run (kept as evidence) |

### `u_matrix_analysis/` (U-Matrix analysis and figures; runner `run_u_matrix_analysis.py`)

| Run folder | Step | Config | Status |
|---|---|---|---|
| `mnist_qe_umatrix_validation_figure_s42_v1` | `qe_umatrix_validation_figure` | `qe_umatrix_validation_figure.yaml` | current |
| `mnist_svhn_proposal_gap_analysis_s42_v1` | `proposal_gap_analysis` | `phase3_proposal_gap_analysis.yaml` | current |
| `mnist_svhn_som_overlay_s42_v1` | `svhn_som_overlay` | `svhn_som_overlay.yaml` | current (run on EC2, 2026-11-03) |
| `mnist_umatrix_hex_grid_s42_v1` | `u_matrix_grid_figure` | `u_matrix_grid_figure.yaml` | current |
| `qe_u_matrix_validation_reproduction_initial` | `qe_u_matrix` | `qe_u_matrix_reproduction.yaml` | reproduction, not yet run (skipped unless `--with-reproductions`) |
| `qe_u_matrix_validation_s42_initial` | - | - | historical (kept as evidence); original run that step `qe_u_matrix` repeats; later steps read it |
| `reorg_20261101_gap_analysis_s42_v1` | - | `phase3_reorganization/phase3_proposal_gap_analysis.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261101_u_matrix_s42_v1` | - | `phase3_reorganization/qe_u_matrix_reproduction.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261102_gap_analysis_s42_v2` | - | `phase3_reorganization/reorg_20261102/gap_analysis_rerun.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261102_qe_umatrix_validation_figure_s42_v1` | - | `phase3_reorganization/reorg_20261102/qe_umatrix_validation_figure.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261102_u_matrix_grid_figure_s42_v1` | - | `phase3_reorganization/reorg_20261102/u_matrix_grid_figure.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261102_u_matrix_s42_v2` | - | `phase3_reorganization/reorg_20261102/u_matrix_rerun.yaml` | reorganisation run (kept as evidence) |

### `novelty_benchmark/` (comparison with the other novelty methods; runner `run_novelty_benchmark.py`)

| Run folder | Step | Config | Status |
|---|---|---|---|
| `autoencoder_mnist_validation_s42_initial` | - | - | historical (kept as evidence); first autoencoder run, kept as evidence |
| `classical_mnist_validation_s42_initial` | - | - | historical (kept as evidence); first Isolation Forest / One-Class SVM run, kept as evidence |
| `mnist_historical_baseline_verification_s42_v1` | `historical_baseline_verification` | `historical_baseline_verification.yaml` | current |
| `mnist_svhn_ae_train_standardized_s42_v1` | `standardized_autoencoder` | `autoencoder_standardized_diagnostic.yaml` | current |
| `mnist_svhn_autoencoder_scaling_diagnostic_s42_46_v1` | `autoencoder_scaling_diagnostic` | `autoencoder_scaling_diagnostic.yaml` | current (run on EC2, 2026-11-03) |
| `mnist_svhn_consolidated_evidence_s42_v1` | - | - | historical (kept as evidence); written by `consolidate_phase3_evidence` (own command, not a `run_all` step) |
| `mnist_svhn_energy_knn_score_export_s42_v1` | `svhn_score_export` | `svhn_score_export.yaml` | current |
| `mnist_svhn_fair_budget_baselines_s42_46_v1` | `fair_budget_baselines` | `fair_budget_baselines.yaml` | current |
| `mnist_svhn_five_method_comparison_s42_initial` | `svhn_benchmark` | `novelty_benchmark.yaml` | current |
| `mnist_svhn_missed_qe_figure_s42_v1` | `missed_svhn_figure` | `missed_svhn_figure.yaml` | current |
| `mnist_svhn_qe_energy_knn_paired_s42_v1` | `qe_energy_knn_paired` | `qe_energy_knn_paired.yaml` | current |
| `mnist_svhn_qe_msp_paired_s42_v1` | - | - | historical (kept as evidence); earlier paired run; `mnist_svhn_qe_msp_reproduction_s42_v1` repeats it |
| `mnist_svhn_qe_msp_reproduction_s42_v1` | `qe_msp_paired_comparison` | `svhn_paired_comparison.yaml` | current |
| `mnist_svhn_qe_umatrix_random_s42_v1` | `qe_umatrix_random_ood` | `qe_u_matrix_random_ood.yaml` | current |
| `mnist_svhn_required_baselines_knn_s42_v1` | `required_baselines_knn` | `svhn_knn_baselines.yaml` | current |
| `mnist_svhn_required_baselines_reproduction_initial` | `required_baselines` | `required_ood_baselines.yaml` | current |
| `mnist_svhn_unified_nine_methods_s42_46_v1` | `unified_novelty_benchmark` | `unified_novelty_benchmark.yaml` | current |
| `mnist_validation_reproducibility_s42_initial` | - | - | historical (kept as evidence); early reproducibility check, kept as evidence |
| `reorg_20261102_autoencoder_standardized_diagnostic_s42_v1` | - | `phase3_reorganization/reorg_20261102/autoencoder_standardized_diagnostic.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261102_fair_budget_baselines_s42_v1` | - | `phase3_reorganization/reorg_20261102/fair_budget_baselines.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261102_historical_baseline_verification_s42_v1` | - | `phase3_reorganization/reorg_20261102/historical_baseline_verification.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261102_missed_svhn_figure_s42_v1` | - | `phase3_reorganization/reorg_20261102/missed_svhn_figure.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261102_ood_comparison_qe_s42_v1` | - | - | historical (kept as evidence); written by `run_ood_comparison` (own command, not a `run_all` step); no matching config in `configs/` |
| `reorg_20261102_qe_energy_knn_paired_s42_v1` | - | `phase3_reorganization/reorg_20261102/qe_energy_knn_paired.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261102_qe_u_matrix_random_ood_s42_v1` | - | `phase3_reorganization/reorg_20261102/qe_u_matrix_random_ood.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261102_required_ood_baselines_s42_v1` | - | `phase3_reorganization/reorg_20261102/required_ood_baselines.yaml` | reorganisation run (kept as evidence) |
| `reorg_20261102_svhn_score_export_s42_v1` | - | `phase3_reorganization/reorg_20261102/svhn_score_export.yaml` | reorganisation run (kept as evidence) |
| `svhn_inverted_embedding_norm_s42_v1` | `svhn_inverted_norm_diagnostic` | `svhn_inverted_norm_diagnostic.yaml` | current |

Large `.npy` outputs are tracked with DVC and are not committed to git; small `.csv` and `.json` files stay in git. Figures are PDF only.
