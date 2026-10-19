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

Fyrooz requested runners producing the original classical/autoencoder JSONs or removal of those files. Neither has been completed in this patch. The historical records are intentionally retained with provenance pending a dedicated reproduction runner or explicit reviewer acceptance.
