# Phase 3: Novelty Detection

Phase 3 evaluates whether frozen MNIST-trained representations and
Self-Organizing Map (SOM) distances can help identify unfamiliar images.
The current external out-of-distribution (OOD) evaluation uses SVHN.

The held-out MNIST test split (the project's own 10,500-image split, not the official
10,000-image MNIST test set) is not used in this evaluation. See `docs/DEVIATIONS.md`.

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

See the [historical provenance audit](qe_novelty_detection/HISTORICAL_PROVENANCE_AUDIT.md)
for detailed verification evidence and limitations.

No historical experiment outputs were overwritten.
The held-out MNIST test split was not accessed by any Phase 3 run.

## Components and documentation

| Component | Scientific documentation |
|---|---|
| QE novelty detection | [Preregistration](qe_novelty_detection/PREREGISTRATION_qe_novelty.md), [random-control reproduction](qe_novelty_detection/RANDOM_CONTROL_REPRODUCTION.md) |
| U-Matrix analysis | [U-Matrix reproduction](u_matrix_analysis/U_MATRIX_REPRODUCTION.md) |
| Novelty benchmark | [Consolidated evidence](novelty_benchmark/CONSOLIDATED_EVIDENCE.md), [why Isolation Forest and the autoencoder look odd on SVHN](novelty_benchmark/BASELINE_SCORE_ANALYSIS.md) |


## Implementation and configuration

The implementation is organized under
`src/v1_mnist/component/p3_novelty_detection/`:

- `qe_novelty_detection/` — BMU and quantization-error novelty scoring.
- `u_matrix_analysis/` — SOM neighborhood-distance analysis.
- `novelty_benchmark/` — OOD scoring, baseline comparisons,
  metrics, and the benchmark scripts.

The authoritative 22-configuration inventory is maintained
in the [configuration-to-result table](#configuration-to-result-traceability)
below. The [component README](../../component/p3_novelty_detection/README.md)
summarizes the Phase 3 implementation and runners.

Configuration files are in `src/v1_mnist/component/configs/`.

## Configuration-to-result traceability

The following inventory maps the 22 identified Phase 3
configurations to their configured run IDs, output steps,
and observed result directories.

A matching directory does not independently prove which
historical configuration revision produced its contents.

| Configuration | Configured run ID | Step | Local evidence |
|---|---|---|---|
| `autoencoder_scaling_diagnostic.yaml` | `mnist_svhn_autoencoder_scaling_diagnostic_s42_46_v1` | `novelty_benchmark` | Configured directory exists (run on EC2, 2026-11-03) |
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
| `qe_threshold_sweep.yaml` | `mnist_svhn_qe_threshold_sweep_s42_v1` | `qe_novelty_detection` | Configured directory exists (run on EC2, 2026-11-03) |
| `qe_u_matrix_random_ood.yaml` | `mnist_svhn_qe_umatrix_random_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `qe_u_matrix_reproduction.yaml` | `qe_u_matrix_validation_reproduction_initial` | `u_matrix_analysis` | Historical directory: `qe_u_matrix_validation_s42_initial` (different run ID; provenance not fully established) |
| `qe_umatrix_validation_figure.yaml` | `mnist_qe_umatrix_validation_figure_s42_v1` | `u_matrix_analysis` | Configured directory exists |
| `required_ood_baselines.yaml` | `mnist_svhn_required_baselines_reproduction_initial` | `novelty_benchmark` | Configured directory exists |
| `svhn_inverted_norm_diagnostic.yaml` | `svhn_inverted_embedding_norm_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `svhn_knn_baselines.yaml` | `mnist_svhn_required_baselines_knn_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `svhn_paired_comparison.yaml` | `mnist_svhn_qe_msp_reproduction_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `svhn_score_export.yaml` | `mnist_svhn_energy_knn_score_export_s42_v1` | `novelty_benchmark` | Configured directory exists |
| `svhn_som_overlay.yaml` | `mnist_svhn_som_overlay_s42_v1` | `u_matrix_analysis` | Configured directory exists (run on EC2, 2026-11-03) |
| `u_matrix_grid_figure.yaml` | `mnist_umatrix_hex_grid_s42_v1` | `u_matrix_analysis` | Configured directory exists |
| `unified_novelty_benchmark.yaml` | `mnist_svhn_unified_nine_methods_s42_46_v1` | `novelty_benchmark` | Configured directory exists |

The historical run-ID mismatches are identified in the
[configuration-to-result table](#configuration-to-result-traceability).
For unresolved provenance limitations, see the
[historical provenance audit](qe_novelty_detection/HISTORICAL_PROVENANCE_AUDIT.md).


## Execution and verification

The Phase 3 step runners are in
`src/v1_mnist/component/pipeline/p3_novelty_detection/`, one per
implementation folder: `run_qe_novelty_detection.py`,
`run_u_matrix_analysis.py` and `run_novelty_benchmark.py`. Each lists its
steps with `--dry-run`, runs one step with `--only STEP` (with `--config`
for a new run ID), and skips steps whose results already exist. The shared
`pipeline/run_all.py` remains unchanged from `develop` and handles the
original Phase 1/2 workflow. To compare two score files outside these steps,
run `novelty_benchmark/run_ood_comparison.py` directly.

The Phase 3 runner tests are in
`src/v1_mnist/tests/pipeline/test_phase3_runners.py`.
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

See the [authoritative configuration-to-result table](#configuration-to-result-traceability)
above. It records all 22 historical configurations, their run IDs,
step folders, and identified historical mismatches.

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

### Historical test-count reconciliation — 2026-11-02

Historical test totals refer to different execution scopes,
commits, and environments; they are retained as recorded.

| Tests passed | Recorded context |
|---|---|
| 73 | Earlier local Phase 3 subset; two PyTorch-dependent test files excluded |
| 168 | Earlier EC2 full repository verification; two warnings |
| 172 | Historical EC2 full repository verification before scope cleanup |
| 134 | Phase 3-only execution recorded for 2026-10-20 |
| 135 | Phase 3-only execution at commit `1d30862`, 2026-10-26 |
| 148 | Phase 3 PR re-review verification recorded in Issue #24 |
| 246 | Full suite from the earlier PR re-review; two warnings |
| 252 | Full suite recorded in scientific README on 2026-11-02; two warnings |
| 260 | Full suite recorded on EC2 on 2026-11-01; two warnings |

These records are historical snapshots, not conflicting
results of one identical test execution.

Some historical records lack exact environment manifests,
full commit IDs or command logs. Missing metadata are not
inferred or reconstructed.

A new current-checkout verification must report its own
command, Git revision, environment and actual pytest result.

## Historical benchmark provenance clarification

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
