# Historical baseline evidence and outstanding reproduction tasks

Recorded: 2026-10-19. This is a limitations record, **not** a declaration that all producing-runner requests have been fulfilled.

| Historical artifact | Verified | Outstanding |
|---|---|---|
| Classical MNIST validation JSON | Saved mean values, preserved SHA-256 and SVHN score-array diagnostic | A dedicated runner regenerating this exact original JSON is not yet supplied; original OOD metric placeholders describe MNIST-only validation stage |
| Autoencoder MNIST validation JSON | Original train loss and mean val reconstruction error matched on EC2; saved bytes preserved | No original checkpoint and no dedicated exact historical JSON-producing runner; do not substitute new OOD metrics in old validation record |
| Reproducibility verification JSON | Historical measurements and fingerprints preserved | Original software environment manifest not captured; cannot reconstruct past versions retroactively; current pinned requirements are not proof of original package versions |
| U-Matrix JSON/PDF | Validation scores independently reproduced; reproduction runner and tests are committed | Confirm original figure paths and renderer outputs before claiming byte-identical PDF reproduction |
| QE random control JSON | Aggregate statistics independently reproduced; dedicated runner and tests committed | Historical seeded-control discrepancy documented separately |

## Leakage Control

No final MNIST test data were used for this documentation audit. No old run was overwritten; new Fashion-MNIST and SVHN baselines remain separate.

## Reviewer decision needed

Fyrooz requested runners producing the original classical/autoencoder JSONs or removal of those files. A new-run verification runner now independently reproduces all seven historical validation summary metrics, but it does not regenerate the original JSON files byte-for-byte or recover the original autoencoder checkpoint. The historical files remain unchanged, and reviewer acceptance is still required.

## Independent Historical Baseline Verification — 2026-10-19

The historical baseline verification runner was executed on EC2 using
the frozen MNIST training and validation embeddings.

- Run ID: `mnist_historical_baseline_verification_s42_v1`
- Input SHA-256 checks: passed for training and validation embeddings.
- Classical baselines: Isolation Forest and One-Class SVM mean and
  standard deviation matched the preserved historical values exactly.
- Autoencoder: initial training loss, final training loss, and mean
  validation reconstruction error matched the historical values exactly.
- Total: 7 of 7 numerical comparisons agreed, with absolute difference 0.
- Leakage control: training-fit and MNIST-validation-only evaluation;
  no final-test or OOD data was accessed by this verification.
- Regression testing: 74 Phase 3 tests passed; 172 complete MNIST
  tests passed when launched from `src/`.
- Ruff: all Phase 3 checks passed.

The verification demonstrates numerical reproducibility under the
currently committed configuration. It does not recover the original
autoencoder checkpoint or historical package-version manifest.

The original historical JSON files remain unchanged.
