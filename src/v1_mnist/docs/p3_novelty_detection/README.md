# Phase 3: Novelty Detection

Phase 3 evaluates whether frozen MNIST-trained representations and
Self-Organizing Map (SOM) distances can help identify unfamiliar images.
The current external out-of-distribution (OOD) evaluation uses SVHN.

The official MNIST final-test split is excluded from this evaluation.

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

On 2026-11-01, the EC2 repository test suite returned
**246 passed, 2 warnings**. This verifies software tests, not
numerical reproduction of every saved experiment.

## Leakage control and interpretation

MNIST training data support model fitting and training-derived
calibration. MNIST validation and SVHN are used for evaluation.
Historical benchmark results are not assumed to have equal tuning
budgets or independent model-training seeds.

The original QE training-derived threshold remains frozen.
The proposal-gap analysis has no verified saved result in the
inspected EC2 workspace. The recorded SVHN sample-alignment
limitation also remains in effect.

See [DEVIATIONS.md](../DEVIATIONS.md) for protocol deviations,
historical configuration discrepancies, and the documented
metrics-JSON formatting repair.

No new experiment or result is claimed by this documentation index.
