# Phase 3: Novelty Detection

Phase 3 investigates whether frozen MNIST-trained representations and SOM
distances can identify unfamiliar images. Development uses MNIST training
and validation artifacts; SVHN is used as an external OOD evaluation set.

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

- Phase 3 adapter dry runs and four adapter regression tests passed locally.
- Five targeted novelty-comparison tests passed locally.
- Ruff checks passed for the tested Phase 3 runner files.
- Full Phase 3 tests were not verified on the local Intel Mac because
  the repository-pinned PyTorch version lacked a compatible wheel.
- The adapter is not yet registered in the shared `pipeline/run_all.py`;
  see `pipeline/p3_novelty_detection/INTEGRATION_HANDOFF.md`.
- The required max-softmax, energy, and Mahalanobis comparisons and
  Fashion-MNIST evaluation must not be claimed complete until verified.
- Previously saved results are historical artifacts, not outputs from
  this documentation update.

## Leakage control

Keep model training and threshold selection separate from final OOD
evaluation. Do not retune frozen settings using SVHN test scores.
Never overwrite existing experiment runs.
