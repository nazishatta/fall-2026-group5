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

Runners are in `component/pipeline/p3_novelty_detection/`.
The independent `run_phase3_stage.py` adapter accepts `--config`,
`--stage`, and `--dry-run`.

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
- The standalone adapter is not registered in shared `pipeline/run_all.py`.
  Registration is deferred to a separate PR after reviewer approval;
  see `pipeline/p3_novelty_detection/INTEGRATION_HANDOFF.md`.
- The original training-derived QE threshold remains frozen. Its
  difference from the instructor's validation-selection request is
  recorded in `src/v1_mnist/docs/DEVIATIONS.md`.
- The EC2 verification does not establish cross-platform numerical
  reproducibility or multiple independent model-training seeds.
- Historical outputs are preserved; no previous run is overwritten.

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

The latest Phase 3-only pytest run completed with **131 passed**. This is separate from the historical full-repository test count.

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
