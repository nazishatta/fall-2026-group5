# Phase 3: Novelty Detection

Phase 3 investigates whether frozen MNIST-trained representations and SOM
distances can identify unfamiliar images. Development uses MNIST training
and validation artifacts. SVHN and Fashion-MNIST are separate external OOD evaluation sets.

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

- Full repository test suite on EC2: **168 passed, 2 warnings,
  0 failures**. The warnings concern existing Seaborn and NNSOM code.
- Ruff checks passed for the Phase 3 source, pipeline, and test directories.
- Four standalone Phase 3 integration tests passed.
- MNIST validation versus Fashion-MNIST evaluation was completed using
  the frozen CNN checkpoint and 84-dimensional fc2 embeddings.
- QE, max-softmax, kNN, energy, and Mahalanobis measurements have
  been independently reproduced against their saved metrics.
- These Fashion-MNIST results use one frozen CNN/SOM realization;
  bootstrap intervals do not substitute for independent training seeds.
- SVHN remains a separate historical evaluation, including classical
  baselines and a feature-reconstruction autoencoder.
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

## Leakage control

Baseline fitting uses MNIST training embeddings. MNIST validation is
used for in-distribution evaluation, while Fashion-MNIST and SVHN
are evaluated as external OOD datasets. The original training-calibrated
QE threshold is not presented as validation-selected.

Do not tune models, thresholds, or baseline hyperparameters using OOD
evaluation outcomes. The official MNIST final-test set remains outside
this Phase 3 tuning process. Preserve run IDs and SHA-256 provenance;
never overwrite historical experiment outputs.
