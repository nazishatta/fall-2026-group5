# Phase 3: Novelty Detection

Phase 3 investigates whether frozen MNIST-trained representations and SOM
distances can identify unfamiliar images. Development uses MNIST training
and validation artifacts. SVHN is the current external OOD evaluation set. Other datasets are deferred.

## Implemented components

- `qe_novelty_detection/`: BMU quantization errors, training-derived
  percentile threshold, novelty flags, and validation analysis.
- `u_matrix_analysis/`: SOM neighborhood separation and sample-level
  U-Matrix scores.
- `novelty_benchmark/`: comparison utilities and classical/autoencoder
  baselines, with AUROC, AUPR, and FPR-at-95%-TPR calculations.

Scientific runners are organized under `component/p3_novelty_detection/novelty_benchmark/`, `qe_novelty_detection/`, and `u_matrix_analysis/`. The independent stage launcher is at `component/p3_novelty_detection/novelty_benchmark/run_phase3_stage.py`.
The independent `run_phase3_stage.py` adapter accepts `--config`,
`--stage`, and `--dry-run`.

## Historical configuration-to-result traceability

This table inventories the 22 identified Phase 3 YAML files
against the local EC2 workspace as inspected on 2026-11-01.
A directory's presence does not independently establish its
scientific validity or prove which configuration revision produced it.

| Configuration | Configured run ID | Step | Local evidence |
|---|---|---|---|
| `autoencoder_standardized_diagnostic.yaml` | `mnist_svhn_ae_train_standardized_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `fair_budget_baselines.yaml` | `mnist_svhn_fair_budget_baselines_s42_46_v1` | `novelty_benchmark` | Configured directory exists |
| `historical_baseline_verification.yaml` | `mnist_historical_baseline_verification_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `historical_svhn_inspection.yaml` | `Not applicable` | `Supporting configuration` | No standalone run expected |
| `missed_svhn_figure.yaml` | `mnist_svhn_missed_qe_figure_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `novelty_benchmark.yaml` | `mnist_svhn_five_method_comparison_s42_initial` | `novelty_benchmark` | Configured directory exists |
| `phase3_proposal_gap_analysis.yaml` | `mnist_svhn_proposal_gap_analysis_s42_v1` | `u_matrix_analysis` | No matching saved directory verified |
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

### Historical traceability limitations

- The current QE OOD run ID comes from the nested
  `ood_datasets[0].run_id` entry, not a top-level field.
- QE OOD, QE random control, and QE/U-Matrix reproduction
  have historical result directories with different run IDs.
  Do not treat the current YAML as proven to be the exact
  configuration revision used for those historical runs.
- The Proposal Gap Analysis has no matching saved result
  in the inspected workspace. Its configured Phase 2
  validation BMU input is also unavailable locally.
- Historical thresholds, configurations, SHA-256 records,
  output directories, and scientific results remain unchanged.

### Additional local result directories

These directories were found but were not directly matched
to a configured run or the historical mappings above.
They are not necessarily invalid or untracked experiments.

- `novelty_benchmark/autoencoder_mnist_validation_s42_initial`
- `novelty_benchmark/classical_mnist_validation_s42_initial`
- `novelty_benchmark/mnist_svhn_consolidated_evidence_s42_v1`
- `novelty_benchmark/mnist_svhn_qe_msp_paired_s42_v1`
- `novelty_benchmark/mnist_validation_reproducibility_s42_initial`

Directory inventory is not an execution, hash-validation,
or scientific reproducibility audit. Any new execution must
use a new run ID with the required provenance preflight.

## Verification and limitations

- Historical full repository test suite on EC2: **168 passed, 2 warnings,
  0 failures**. The warnings concern existing Seaborn and NNSOM code.
- Ruff checks passed for the Phase 3 source, pipeline, and test directories.
- Four standalone Phase 3 integration tests passed.
- MNIST validation versus SVHN is the current ID/OOD comparison.
- Saved SVHN max-softmax, energy, and Mahalanobis results include
  bootstrap intervals; the SVHN kNN comparison is verified (AUROC 0.976561, 95% CI [0.974306, 0.978747]).
- Historical SVHN comparisons also include QE, U-Matrix, classical
  baselines, and a feature-reconstruction autoencoder.
- These evaluations use one frozen CNN/SOM realization;
  bootstrap intervals do not replace independent training seeds.
- Historical Isolation Forest and autoencoder results require further
  investigation. Their recorded scores must not be silently reversed
  or reselected to improve AUROC.
- Phase 3 uses the standalone `run_phase3_stage.py` launcher.
  The shared `pipeline/run_all.py` also supports opt-in Phase 3 execution
  using `--phase3-stage` and `--phase3-config`. Existing Phase 1/2
  defaults remain unchanged. `ood_comparison` uses the standalone launcher.
- The original training-derived QE threshold remains frozen. Its
  difference from the instructor's validation-selection request is
  recorded in `src/v1_mnist/docs/DEVIATIONS.md`.
- The EC2 verification does not establish cross-platform numerical
  reproducibility or multiple independent model-training seeds.
- Historical outputs are preserved; no previous run is overwritten.

### Current repository verification — 2026-11-01

The EC2 repository was verified after the Phase 3 file reorganization
and shared-pipeline integration.

| Verification | Result |
|---|---|
| Full repository test suite (`src/v1_mnist/tests`) | **260 passed** |
| Phase 3 pipeline tests (`test_novelty_pipeline.py`) | **36 passed** |
| Phase 1/2 pipeline regression tests | **39 passed** |
| Specified Phase 3 Ruff scope, including E402 | **All checks passed** |
| Git staged and unstaged whitespace checks | **Passed** |

The full repository test suite reported two warnings: a Seaborn
palette deprecation warning in Phase 1 and an NNSOM normalization
warning in Phase 2. No test failures were reported.

The 24 relocated scientific/utility implementations were compared
against their original Git versions. Their non-import implementation
code was preserved, apart from the documented provenance root-path
correction.

The standalone Phase 3 launcher exposes 16 registered stages.
Their target modules and shared-pipeline dry-run dispatch were verified.
The scientific experiments were not rerun as part of this
reorganization.

The two warnings originate from existing Phase 1/2 dependencies.
Earlier test counts in this README are historical verification
records and should not be interpreted as current results.

## SVHN dataset setup

The current OOD evaluation uses the SVHN test split at `data/ood_svhn/test_32x32.mat`.

Official Stanford download URL:

https://ufldl.stanford.edu/housenumbers/test_32x32.mat

For a fresh checkout, run the following commands from the repository root only if the dataset is missing:

    mkdir -p data/ood_svhn
    curl -fL --retry 3 https://ufldl.stanford.edu/housenumbers/test_32x32.mat -o data/ood_svhn/test_32x32.mat
    sha256sum data/ood_svhn/test_32x32.mat

Expected SHA-256:

`cdce80dfb2a2c4c6160906d0bd7c68ec5a99d7ca4831afa54f09182025b6a75b`

Verify with `sha256sum data/ood_svhn/test_32x32.mat` before running benchmarks.
Do not overwrite an existing dataset. The EC2 workspace currently uses a verified symlink to the SVHN file.

Phase 3-only pytest verification on EC2 at commit `1d30862` (2026-10-26): **135 passed**. Ruff checks also passed. This count applies only to `src/v1_mnist/tests/p3_novelty_detection` and must not be confused with historical full-repository test counts.

## Leakage control

Baseline fitting uses MNIST training embeddings. MNIST validation is
used for in-distribution evaluation, while SVHN is the current
external OOD evaluation dataset. The original training-calibrated
QE threshold is not presented as validation-selected.

Do not tune models, thresholds, or baseline hyperparameters using OOD
evaluation outcomes. The official MNIST final-test set remains outside
this Phase 3 tuning process. Preserve run IDs and SHA-256 provenance;
never overwrite historical experiment outputs.

## Test-count and benchmark provenance clarification

The earlier full-repository count of 168 passed and 2 warnings is
a historical EC2 verification result. The latest Phase 3-only suite,
executed on 2026-10-20, passed 134 tests with no failures.
These figures describe different test scopes and must not be treated
as conflicting measurements of the same test suite.

The consolidated nine-method CSV contains results from two source runs,
not a newly fitted single execution. It preserves each method's
source_run_id. QE and U-Matrix were evaluated together in the original
five-method benchmark; max-softmax, energy, Mahalanobis and kNN were
evaluated in the subsequent frozen-CNN baseline run.

The equal-fitting-budget diagnostic uses five explicit seeds, stored
under `seed_list` in its original configuration. That original
configuration is retained because its SHA-256 is recorded in the
committed experiment metrics. This is a naming compatibility exception
to the requested `seeds` key, not evidence that the experiment was
executed without a seed list.
