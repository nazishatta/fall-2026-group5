# Phase 3: Novelty Detection

Phase 3 evaluates whether frozen MNIST-trained representations and
Self-Organizing Map (SOM) distances can help identify unfamiliar images.
The current external out-of-distribution (OOD) evaluation uses SVHN.

The official MNIST final-test split is excluded from this evaluation.

## Historical numerical verification — 2026-11-02

Read-only EC2 reproduction verified:

- QE OOD: historical numerical metrics, bootstrap interval,
  and five recorded artifact SHA-256 hashes.
- QE/U-Matrix: historical numerical metrics and the maximum
  QE reconstruction difference (`4.860474741713583e-06`).
- QE random control: historical aggregate statistics and
  confidence interval. Original seed-specific counts
  remain unreproduced.

The exact original generating YAML revisions were not
established. Current configurations reproduce the principal
numerical findings but are not claimed to be the original
executing configurations.

See the [historical provenance audit](../../component/p3_novelty_detection/HISTORICAL_PROVENANCE_AUDIT.md)
for detailed verification evidence and limitations.

No historical experiment outputs were overwritten.
The official MNIST final-test split was not accessed.

## Components and documentation

| Component | Scientific documentation |
|---|---|
| QE novelty detection | [Preregistration](qe_novelty_detection/PREREGISTRATION_qe_novelty.md), [random-control reproduction](qe_novelty_detection/RANDOM_CONTROL_REPRODUCTION.md) |
| U-Matrix analysis | [U-Matrix reproduction](u_matrix_analysis/U_MATRIX_REPRODUCTION.md), [work report](u_matrix_analysis/MNIST_phase3_work_report.md), [historical development log](u_matrix_analysis/MNIST_phase3_historical_development_log.md) |
| Novelty benchmark | [Consolidated evidence](novelty_benchmark/CONSOLIDATED_EVIDENCE.md), [historical evidence status](novelty_benchmark/HISTORICAL_EVIDENCE_STATUS.md), [result relocation record](novelty_benchmark/RESULT_RELOCATION_RECORD.md) |

The historical development log preserves earlier file paths and
chronology. It is not an authoritative current-path inventory.

## Implementation and configuration

The implementation is organized under
`src/v1_mnist/component/p3_novelty_detection/`:

- `qe_novelty_detection/` — BMU and quantization-error novelty scoring.
- `u_matrix_analysis/` — SOM neighborhood-distance analysis.
- `novelty_benchmark/` — OOD scoring, baseline comparisons,
  metrics, and the standalone Phase 3 launcher.

The [component README](../../component/p3_novelty_detection/README.md)
contains the 22-configuration-to-result directory inventory and
identifies unmatched historical run IDs.

Configuration files are in `src/v1_mnist/component/configs/`.

## Configuration-to-result traceability

The following inventory maps the 22 identified Phase 3
configurations to their configured run IDs, output steps,
and observed result directories.

A matching directory does not independently prove which
historical configuration revision produced its contents.

| Configuration | Configured run ID | Step | Local evidence |
|---|---|---|---|
| `autoencoder_standardized_diagnostic.yaml` | `mnist_svhn_ae_train_standardized_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `fair_budget_baselines.yaml` | `mnist_svhn_fair_budget_baselines_s42_46_v1` | `novelty_benchmark` | Configured directory exists |
| `historical_baseline_verification.yaml` | `mnist_historical_baseline_verification_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `historical_svhn_inspection.yaml` | `Not applicable` | `Supporting configuration` | No standalone run expected |
| `missed_svhn_figure.yaml` | `mnist_svhn_missed_qe_figure_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `novelty_benchmark.yaml` | `mnist_svhn_five_method_comparison_s42_initial` | `novelty_benchmark` | Configured directory exists |
| `phase3_proposal_gap_analysis.yaml` | `mnist_svhn_proposal_gap_analysis_s42_v1` | `u_matrix_analysis` | Generated locally: `u_matrix_analysis/mnist_svhn_proposal_gap_analysis_s42_v1/`; `analysis.json` and `pdf.dvc` prepared for Git; PDF pushed to and verified in DVC remote |
| `phase3_shared_provenance.yaml` | `Not applicable` | `Supporting configuration` | No standalone run expected |
| `qe_energy_knn_paired.yaml` | `mnist_svhn_qe_energy_knn_paired_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `qe_novelty.yaml` | `qe_training_calibrated_initial` | `qe_novelty_detection` | Configured directory exists |
| `qe_ood_evaluation.yaml` | `mnist_svhn_qe_dev_initial` | `qe_novelty_detection` | Historical directory: `mnist_svhn_qe_frozen_s42_initial` (different run ID; provenance not fully established) |
| `qe_random_control.yaml` | `qe_random_control_reproduction_initial` | `qe_novelty_detection` | Historical directory: `qe_random_control_evaluation_initial` (different run ID; provenance not fully established) |
| `qe_u_matrix_random_ood.yaml` | `mnist_svhn_qe_umatrix_random_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `qe_u_matrix_reproduction.yaml` | `qe_u_matrix_validation_reproduction_initial` | `u_matrix_analysis` | Historical directory: `qe_u_matrix_validation_s42_initial` (different run ID; provenance not fully established) |
| `qe_umatrix_validation_figure.yaml` | `mnist_qe_umatrix_validation_figure_s42_v1` | `u_matrix_analysis` | Configured directory exists |
| `required_ood_baselines.yaml` | `mnist_svhn_required_baselines_reproduction_initial` | `novelty_benchmark` | Configured directory exists |
| `svhn_inverted_norm_diagnostic.yaml` | `svhn_inverted_embedding_norm_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `svhn_knn_baselines.yaml` | `mnist_svhn_required_baselines_knn_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `svhn_paired_comparison.yaml` | `mnist_svhn_qe_msp_reproduction_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `svhn_score_export.yaml` | `mnist_svhn_energy_knn_score_export_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `u_matrix_grid_figure.yaml` | `mnist_umatrix_hex_grid_s42_v1` | `u_matrix_analysis` | Configured directory exists |
| `unified_novelty_benchmark.yaml` | `mnist_svhn_unified_nine_methods_s42_46_v1` | `novelty_benchmark` | Configured directory exists |

For the historical run-ID mismatches and their provenance
limitations, see the component README:
`src/v1_mnist/component/p3_novelty_detection/README.md`.


## Execution and verification

The standalone launcher is:

`src/v1_mnist/component/p3_novelty_detection/novelty_benchmark/run_phase3_stage.py`

It supports `--stage`, `--config`, and `--dry-run`. The shared
`pipeline/run_all.py` also supports opt-in Phase 3 execution through
`--phase3-stage` and `--phase3-config`. The `ood_comparison` stage
continues to use the standalone launcher.

The standalone Phase 3 launcher and pipeline compatibility tests are located in
`src/v1_mnist/tests/pipeline/test_phase3_pipeline.py`.
Other Phase 3 tests are under
`src/v1_mnist/tests/p3_novelty_detection/`, with shared provenance
tests under `src/v1_mnist/tests/utils/`.

On 2026-11-02, the EC2 repository test suite returned
**252 passed, 2 warnings**. This verifies software tests, not
numerical reproduction of every saved experiment.

## Leakage control and interpretation

MNIST training data support model fitting and training-derived
calibration. MNIST validation and SVHN are used for evaluation.
Historical benchmark results are not assumed to have equal tuning
budgets or independent model-training seeds.

The original QE training-derived threshold remains frozen.
The proposal-gap analysis has been generated under
`u_matrix_analysis/mnist_svhn_proposal_gap_analysis_s42_v1/`.
Its `analysis.json` is staged for Git tracking, and its PDF
is DVC-tracked, pushed to the S3 remote, and verified.
The recorded SVHN sample-alignment limitation remains in effect.

See [DEVIATIONS.md](../DEVIATIONS.md) for protocol deviations,
historical configuration discrepancies, and the documented
metrics-JSON formatting repair.

No new experiment or result is claimed by this documentation index.



<!-- PHASE3_RUN_REGISTRY -->

## Phase 3 — Configuration and Historical Run Registry

This registry records the historical configuration-to-output mapping
before the Phase 3 source-code reorganization reruns.

**Important:** Directory existence confirms that an output directory
exists. It does not independently verify the scientific validity
of its metrics.

Historical YAML configurations, run IDs, output paths and provenance
hashes must remain unchanged.

### Historical configuration-to-result mapping

| Configuration | Configured historical run ID | Actual historical result folder | Step | Status |
|---|---|---|---|---|
| autoencoder_standardized_diagnostic.yaml | mnist_svhn_ae_train_standardized_s42_v1 | mnist_svhn_ae_train_standardized_s42_v1 | novelty_benchmark | Directory verified |
| fair_budget_baselines.yaml | mnist_svhn_fair_budget_baselines_s42_46_v1 | mnist_svhn_fair_budget_baselines_s42_46_v1 | novelty_benchmark | Directory verified |
| historical_baseline_verification.yaml | mnist_historical_baseline_verification_s42_v1 | mnist_historical_baseline_verification_s42_v1 | novelty_benchmark | Directory verified |
| missed_svhn_figure.yaml | mnist_svhn_missed_qe_figure_s42_v1 | mnist_svhn_missed_qe_figure_s42_v1 | novelty_benchmark | Directory verified |
| novelty_benchmark.yaml | mnist_svhn_five_method_comparison_s42_initial | mnist_svhn_five_method_comparison_s42_initial | novelty_benchmark | Directory verified |
| phase3_proposal_gap_analysis.yaml | mnist_svhn_proposal_gap_analysis_s42_v1 | mnist_svhn_proposal_gap_analysis_s42_v1 | u_matrix_analysis | Directory verified |
| qe_energy_knn_paired.yaml | mnist_svhn_qe_energy_knn_paired_s42_v1 | mnist_svhn_qe_energy_knn_paired_s42_v1 | novelty_benchmark | Directory verified |
| qe_novelty.yaml | qe_training_calibrated_initial | qe_training_calibrated_initial | qe_novelty_detection | Directory verified |
| qe_ood_evaluation.yaml | mnist_svhn_qe_dev_initial | mnist_svhn_qe_frozen_s42_initial | qe_novelty_detection | Historical ID mismatch |
| qe_random_control.yaml | qe_random_control_reproduction_initial | qe_random_control_evaluation_initial | qe_novelty_detection | Historical ID mismatch |
| qe_u_matrix_random_ood.yaml | mnist_svhn_qe_umatrix_random_s42_v1 | mnist_svhn_qe_umatrix_random_s42_v1 | novelty_benchmark | Directory verified |
| qe_u_matrix_reproduction.yaml | qe_u_matrix_validation_reproduction_initial | qe_u_matrix_validation_s42_initial | u_matrix_analysis | Historical ID mismatch |
| qe_umatrix_validation_figure.yaml | mnist_qe_umatrix_validation_figure_s42_v1 | mnist_qe_umatrix_validation_figure_s42_v1 | u_matrix_analysis | Directory verified |
| required_ood_baselines.yaml | mnist_svhn_required_baselines_reproduction_initial | mnist_svhn_required_baselines_reproduction_initial | novelty_benchmark | Directory verified |
| svhn_inverted_norm_diagnostic.yaml | svhn_inverted_embedding_norm_s42_v1 | svhn_inverted_embedding_norm_s42_v1 | novelty_benchmark | Directory verified |
| svhn_knn_baselines.yaml | mnist_svhn_required_baselines_knn_s42_v1 | mnist_svhn_required_baselines_knn_s42_v1 | novelty_benchmark | Directory verified |
| svhn_paired_comparison.yaml | mnist_svhn_qe_msp_reproduction_s42_v1 | mnist_svhn_qe_msp_reproduction_s42_v1 | novelty_benchmark | Directory verified |
| svhn_score_export.yaml | mnist_svhn_energy_knn_score_export_s42_v1 | mnist_svhn_energy_knn_score_export_s42_v1 | novelty_benchmark | Directory verified |
| u_matrix_grid_figure.yaml | mnist_umatrix_hex_grid_s42_v1 | mnist_umatrix_hex_grid_s42_v1 | u_matrix_analysis | Directory verified |
| unified_novelty_benchmark.yaml | mnist_svhn_unified_nine_methods_s42_46_v1 | mnist_svhn_unified_nine_methods_s42_46_v1 | novelty_benchmark | Directory verified |

### Support configurations

| Configuration | Independent run ID | Purpose |
|---|---|---|
| phase3_shared_provenance.yaml | None | Shared input provenance and SHA-256 verification |
| historical_svhn_inspection.yaml | None | Historical inspection configuration |

### Existing output folders without a direct configuration match

The following directories were present in the audited output inventory.
Their exact producing configurations require further historical provenance
inspection.

| Existing folder | Step |
|---|---|
| autoencoder_mnist_validation_s42_initial | novelty_benchmark |
| classical_mnist_validation_s42_initial | novelty_benchmark |
| mnist_svhn_consolidated_evidence_s42_v1 | novelty_benchmark |
| mnist_validation_reproducibility_s42_initial | novelty_benchmark |

These directories are retained unchanged.

### Reorganization rerun tracking

The following registry was generated from the reorganization
YAML configurations and verified local result directories.

All 20 configured runs have matching output directories
and at least one recorded evidence artifact.

| Configuration | New run ID | Step | Result directory | Evidence |
|---|---|---|---|---|
| `phase3_proposal_gap_analysis.yaml` | `reorg_20261101_gap_analysis_s42_v1` | `u_matrix_analysis` | `u_matrix_analysis/reorg_20261101_gap_analysis_s42_v1` | `analysis.json` |
| `qe_novelty.yaml` | `reorg_20261101_qe_novelty_s42_v1` | `qe_novelty_detection` | `qe_novelty_detection/reorg_20261101_qe_novelty_s42_v1` | `qe_summary.json` |
| `qe_ood_evaluation.yaml` | `reorg_20261101_qe_ood_s42_v1` | `qe_novelty_detection` | `qe_novelty_detection/reorg_20261101_qe_ood_s42_v1` | `qe_ood_metrics.json` |
| `qe_random_control.yaml` | `reorg_20261101_random_control_s42_v1` | `qe_novelty_detection` | `qe_novelty_detection/reorg_20261101_random_control_s42_v1` | `qe_random_control_metrics.json` |
| `qe_u_matrix_reproduction.yaml` | `reorg_20261101_u_matrix_s42_v1` | `u_matrix_analysis` | `u_matrix_analysis/reorg_20261101_u_matrix_s42_v1` | `qe_u_matrix_metrics.json` |
| `reorg_20261102/autoencoder_standardized_diagnostic.yaml` | `reorg_20261102_autoencoder_standardized_diagnostic_s42_v1` | `novelty_benchmark` | `novelty_benchmark/reorg_20261102_autoencoder_standardized_diagnostic_s42_v1` | `standardized_autoencoder_metrics.json` |
| `reorg_20261102/fair_budget_baselines.yaml` | `reorg_20261102_fair_budget_baselines_s42_v1` | `novelty_benchmark` | `novelty_benchmark/reorg_20261102_fair_budget_baselines_s42_v1` | `fair_budget_metrics.json` |
| `reorg_20261102/gap_analysis_rerun.yaml` | `reorg_20261102_gap_analysis_s42_v2` | `u_matrix_analysis` | `u_matrix_analysis/reorg_20261102_gap_analysis_s42_v2` | `analysis.json` |
| `reorg_20261102/historical_baseline_verification.yaml` | `reorg_20261102_historical_baseline_verification_s42_v1` | `novelty_benchmark` | `novelty_benchmark/reorg_20261102_historical_baseline_verification_s42_v1` | `verification_metrics.json` |
| `reorg_20261102/missed_svhn_figure.yaml` | `reorg_20261102_missed_svhn_figure_s42_v1` | `novelty_benchmark` | `novelty_benchmark/reorg_20261102_missed_svhn_figure_s42_v1` | `figure_metrics.json` |
| `reorg_20261102/qe_energy_knn_paired.yaml` | `reorg_20261102_qe_energy_knn_paired_s42_v1` | `novelty_benchmark` | `novelty_benchmark/reorg_20261102_qe_energy_knn_paired_s42_v1` | `paired_comparison_metrics.json` |
| `reorg_20261102/qe_novelty_rerun.yaml` | `reorg_20261102_qe_novelty_s42_v2` | `qe_novelty_detection` | `qe_novelty_detection/reorg_20261102_qe_novelty_s42_v2` | `qe_summary.json` |
| `reorg_20261102/qe_ood_rerun.yaml` | `reorg_20261102_qe_ood_s42_v2` | `qe_novelty_detection` | `qe_novelty_detection/reorg_20261102_qe_ood_s42_v2` | `qe_ood_metrics.json` |
| `reorg_20261102/qe_u_matrix_random_ood.yaml` | `reorg_20261102_qe_u_matrix_random_ood_s42_v1` | `novelty_benchmark` | `novelty_benchmark/reorg_20261102_qe_u_matrix_random_ood_s42_v1` | `qe_umatrix_random_metrics.json` |
| `reorg_20261102/qe_umatrix_validation_figure.yaml` | `reorg_20261102_qe_umatrix_validation_figure_s42_v1` | `u_matrix_analysis` | `u_matrix_analysis/reorg_20261102_qe_umatrix_validation_figure_s42_v1` | `figure_metrics.json` |
| `reorg_20261102/random_control_rerun.yaml` | `reorg_20261102_random_control_s42_v2` | `qe_novelty_detection` | `qe_novelty_detection/reorg_20261102_random_control_s42_v2` | `qe_random_control_metrics.json` |
| `reorg_20261102/required_ood_baselines.yaml` | `reorg_20261102_required_ood_baselines_s42_v1` | `novelty_benchmark` | `novelty_benchmark/reorg_20261102_required_ood_baselines_s42_v1` | `required_baseline_metrics.json` |
| `reorg_20261102/svhn_score_export.yaml` | `reorg_20261102_svhn_score_export_s42_v1` | `novelty_benchmark` | `novelty_benchmark/reorg_20261102_svhn_score_export_s42_v1` | `autoencoder_mnist_val_scores.npy.dvc`, `autoencoder_svhn_test_scores.npy.dvc`, `comparison_metrics.json`, `isolation_forest_mnist_val_scores.npy.dvc`, `isolation_forest_svhn_test_scores.npy.dvc`, `one_class_svm_mnist_val_scores.npy.dvc`, `one_class_svm_svhn_test_scores.npy.dvc`, `qe_mnist_val_scores.npy.dvc`, `qe_svhn_test_scores.npy.dvc`, `svhn_labels.npy.dvc`, `u_matrix_mnist_val_scores.npy.dvc`, `u_matrix_svhn_test_scores.npy.dvc` |
| `reorg_20261102/u_matrix_grid_figure.yaml` | `reorg_20261102_u_matrix_grid_figure_s42_v1` | `u_matrix_analysis` | `u_matrix_analysis/reorg_20261102_u_matrix_grid_figure_s42_v1` | `figure_metrics.json` |
| `reorg_20261102/u_matrix_rerun.yaml` | `reorg_20261102_u_matrix_s42_v2` | `u_matrix_analysis` | `u_matrix_analysis/reorg_20261102_u_matrix_s42_v2` | `qe_u_matrix_metrics.json` |

These checks establish configuration-to-directory and artifact
existence, not independent scientific reproduction of every result.

Historical configurations, historical run IDs, and historical
evidence remain unchanged. New runs are kept separately under
their unique reorganization IDs.

The official MNIST final-test set remains excluded.
