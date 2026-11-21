# Protocol: guided retraining on Fashion-MNIST (Phase 4)

Part of the Fashion-MNIST protocol, one file per phase:
[Phase 1](../p1_model_training_baseline/MODEL_TRAINING_PROTOCOL.md) ·
[Phase 2](../p2_som_training_visualization/SOM_ANALYSIS_PROTOCOL.md) ·
[Phase 3](../p3_novelty_detection/NOVELTY_DETECTION_PROTOCOL.md) ·
[Phase 4](RETRAINING_PROTOCOL.md).
Changes and deviations: [`../DEVIATIONS.md`](../DEVIATIONS.md).

Written on 2026-11-17, **before any Fashion-MNIST data was loaded, any network trained or any result seen**,
and committed as one file in `5dd164c` (`docs/FASHION_MNIST_PROTOCOL.md`). Split into one file per phase on
2026-11-21 with no rule changed (see `../DEVIATIONS.md`). Sections 1 to 4 are not changed, except that the
macro-F1 guard is filled in by its rule; any other change needs a written reason in `../DEVIATIONS.md` and,
for a run already made, a new run ID.

**Rule for every phase:** every step done for MNIST is done for Fashion-MNIST, with the MNIST design copied
unchanged, unless the dataset makes that impossible.

## 1. What is fixed before anything runs

`src/v1_mnist/docs/p4_guided_retraining/RETRAINING_PROTOCOL.md` unchanged, except as stated here. Network
recipe as in the [Phase 1 protocol](../p1_model_training_baseline/MODEL_TRAINING_PROTOCOL.md), section 1;
map and hotspot rule as in the [Phase 2 protocol](../p2_som_training_visualization/SOM_ANALYSIS_PROTOCOL.md),
section 1. Seeds: retraining seeds 1-5.

## 2. Steps

- **Hard-image pools** from the Fashion-MNIST map, with the same rules: hotspot neuron (Phase 2 protocol,
  section 1), or training purity below 0.95, or QE above the 95th percentile of the training QE. The random
  and max-softmax pools have the same size and class mix.
- **Arms** (5 seeds each, 1-5, same seeds in every arm), all the MNIST arms:
  - targeted augmentation: `standard`, `augmented_som`, `augmented_random`, `augmented_msp`,
    `augmented_hotspot`, `augmented_random_hotspot`, `augmented_som_paced`, `augmented_random_paced`; 3 altered
    copies per pool image (rotation up to 10 degrees, shift up to 1 pixel, zoom 0.9-1.1, shear up to 8
    degrees, no flips), pacing as on MNIST;
  - curriculum: `standard`, `curriculum_som`, `curriculum_random`, `curriculum_msp`, with the MNIST schedule.
- **Transform check** (validation only, before any arm is compared): the Phase 1 network's accuracy on
  altered validation images must be within 2 percentage points of its accuracy on the originals. If not, the
  ranges are made smaller under a new revision before any arm runs, and that is recorded in `../DEVIATIONS.md`.
- **Macro-F1 guard:** the same rule as MNIST, 2 times the sample standard deviation of the `standard` arm's
  validation macro-F1 over the 5 seeds (the larger of the two experiments), measured before any arms are
  compared. Value: **[TBD: measured by the rule, copied here and into `retraining.yaml` before the test stage]**.
  The MNIST value (0.00157) is not reused, because it measures MNIST's seed-to-seed spread.
- **Statements**, unchanged: A1 (primary: `augmented_som` beats `augmented_random` on hotspot test error,
  interval excludes 0, guard holds), A2, A3 (non-hotspot damage at most 0.002), C1, C2. Paired bootstrap
  over test images (2,000 resamples, seed 2026).
- **SOM re-analysis** (`SOM_REANALYSIS_PROTOCOL.md`): statements R1-R5, `augmented_som` against `standard` and
  against `augmented_random`; and the secondary run on the max-softmax arm (`MSP_ARM_ADDENDUM.md`), as on MNIST.
- **Novelty after retraining** (`NOVELTY_AFTER_RETRAINING_PROTOCOL.md`): unchanged, with the outside set of the
  [Phase 3 protocol](../p3_novelty_detection/NOVELTY_DETECTION_PROTOCOL.md), section 2 (SVHN).
- The test split is opened once, by the `final_test` stage, after the macro-F1 guard is filled in.

The MNIST documents named above are in `src/v1_mnist/docs/p4_guided_retraining/`.

## 3. Leakage control

- The Fashion-MNIST test split is used twice, both fixed in advance: once at the end of Phase 1 training
  (reported only), and once in the Phase 4 `final_test` stage. No setting is chosen from either.
- The hard-image pools, the transform check and the macro-F1 guard use training or validation data only.
- The outside set is used only to compute novelty scores; nothing is fitted or chosen on it.
- Every run records the SHA-256 of its inputs; a finished run is never overwritten.
- **Declared:** this protocol was written after the MNIST results were known. It copies MNIST's design
  instead of adapting it to those results, and no Fashion-MNIST result existed when it was written.

## 4. Limitations stated before the run

- The hotspots come from the SOM, which gives the SOM-guided arms a built-in advantage on hotspot error (as
  for MNIST).

## 5. Recorded outcome

Filled in after the runs, without changing sections 1 to 4. [TBD]

## 6. References

As in the MNIST Phase 4 protocols and `reports/Latex_report/v1_mnist/Final_report/references.bib`.
