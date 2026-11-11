# Phase 3: Novelty Detection

Phase 3 evaluates whether frozen MNIST-trained representations and
Self-Organizing Map (SOM) distances can help identify unfamiliar images.
The current external out-of-distribution (OOD) evaluation uses SVHN; Fashion-MNIST
and KMNIST are deferred until the whole pipeline runs end to end on MNIST.

The held-out MNIST test split (the project's own 10,500-image split, not the official
10,000-image MNIST test set) is not used in this evaluation. See `docs/DEVIATIONS.md`.

## Components and documentation

| Component | Scientific documentation |
|---|---|
| QE novelty detection | [Preregistration](qe_novelty_detection/PREREGISTRATION_qe_novelty.md), [random-control reproduction](qe_novelty_detection/RANDOM_CONTROL_REPRODUCTION.md), [historical provenance audit](qe_novelty_detection/HISTORICAL_PROVENANCE_AUDIT.md) |
| U-Matrix analysis | [U-Matrix reproduction](u_matrix_analysis/U_MATRIX_REPRODUCTION.md) |
| Novelty benchmark | [Consolidated evidence](novelty_benchmark/CONSOLIDATED_EVIDENCE.md), [why Isolation Forest and the autoencoder look odd on SVHN](novelty_benchmark/BASELINE_SCORE_ANALYSIS.md), [SVHN baseline diagnostic](novelty_benchmark/SVHN_BASELINE_DIAGNOSTIC.md) |
| Report | `reports/Latex_report/v1_mnist/p3_novelty_detection/novelty_detection.tex` |

## Implementation and configuration

The code is in `src/v1_mnist/component/p3_novelty_detection/`, one folder per step of
the proposal's Phase 3:

- `qe_novelty_detection/`: BMU and quantization-error (QE) novelty scoring.
- `u_matrix_analysis/`: SOM neighborhood distances and the SOM-grid figures.
- `novelty_benchmark/`: comparison with the other novelty methods, and diagnostics.

Every step has one config in `src/v1_mnist/component/configs/`. A config's `run_id`
names its results folder; a new or re-tuned run always gets a new `run_id`. The
[component README](../../component/p3_novelty_detection/README.md) lists the scripts.

## Which code wrote each result

`outputs/v1_mnist/p3_novelty_detection/run_manifest.csv` lists every Phase 3 run folder
with the step, script and config that wrote it, how that link was checked, and the docs
and report items that use it. The tables in `outputs/v1_mnist/p3_novelty_detection/README.md`
show the same rows. `src/v1_mnist/tests/p3_novelty_detection/test_run_manifest.py`
fails if a folder has no row, if a listed script or config is missing, or if a result's
recorded config SHA-256 no longer matches its config.

Three original QE results (`mnist_svhn_qe_frozen_s42_initial`,
`qe_random_control_evaluation_initial`, `qe_u_matrix_validation_s42_initial`) were
written before their configs were kept in git. They are linked to today's code through
the `reorg_20261102_` reproductions, which repeat them with the current scripts; the
random-control repeat differs from its original in six per-seed counts. Details are in
the [historical provenance audit](qe_novelty_detection/HISTORICAL_PROVENANCE_AUDIT.md).

Each figure in the Phase 3 report is the PDF written by a saved run.
`configs/phase3_report_figures.yaml` names the run for each figure,
`pipeline/p3_novelty_detection/copy_report_figures.py` copies it, and
`tests/p3_novelty_detection/test_report_figures.py` checks each report copy against the
SHA-256 its run recorded.

## Execution and verification

The step runners are in `src/v1_mnist/component/pipeline/p3_novelty_detection/`, one
per implementation folder: `run_qe_novelty_detection.py`, `run_u_matrix_analysis.py`
and `run_novelty_benchmark.py`. Each lists its steps with `--dry-run`, runs one step
with `--only STEP` (with `--config` for a new run ID), and skips steps whose results
already exist. `run_all.py --to novelty_detection` runs Phase 3 in the order its results
depend on each other (`phase3_plan.py`). To compare two score files outside these steps,
run `novelty_benchmark/run_ood_comparison.py` directly.

Tests are in `src/v1_mnist/tests/p3_novelty_detection/` (one folder per step),
`src/v1_mnist/tests/pipeline/` (runners) and `src/v1_mnist/tests/utils/` (shared code).
Run the full suite with `pytest src/v1_mnist/tests` from the repository root; every test
must pass before a merge. A passing suite verifies the code, not a numerical rerun of
every saved experiment.

## Leakage control and interpretation

MNIST training data support model fitting and training-derived calibration, including
the frozen QE threshold (95th percentile of training QE). MNIST validation and SVHN test
are used for evaluation only. SVHN has been examined, so new SVHN analyses are
exploratory. Historical benchmark results do not have equal fitting budgets or
independent CNN/SOM retraining seeds. See [DEVIATIONS.md](../DEVIATIONS.md) for protocol
deviations and historical configuration discrepancies.

## SVHN dataset setup

The OOD evaluation uses the SVHN test split at `data/ood_svhn/test_32x32.mat`
(official download: https://ufldl.stanford.edu/housenumbers/test_32x32.mat).
For a fresh checkout, run from the repository root only if the file is missing:

    mkdir -p data/ood_svhn
    curl -fL --retry 3 https://ufldl.stanford.edu/housenumbers/test_32x32.mat -o data/ood_svhn/test_32x32.mat
    sha256sum data/ood_svhn/test_32x32.mat

Expected SHA-256: `cdce80dfb2a2c4c6160906d0bd7c68ec5a99d7ca4831afa54f09182025b6a75b`
(also recorded in `configs/phase3_shared_provenance.yaml`). Do not overwrite an existing
dataset.

## Notes on two saved results

The consolidated table (`mnist_svhn_consolidated_evidence_s42_v1`) joins two source runs
rather than one new fit: QE and U-Matrix come from the five-method benchmark, and
max-softmax, energy, Mahalanobis and kNN from the later frozen-CNN baseline run. Each row
keeps its `source_run_id`.

The equal-fitting-budget diagnostic stores its five seeds under `seed_list` rather than
`seeds`. Its config is kept unchanged because the result records the config's SHA-256.
