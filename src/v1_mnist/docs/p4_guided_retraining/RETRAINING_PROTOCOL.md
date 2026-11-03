# Retraining protocol (Phase 4: guided retraining)

Status: **DRAFT, not frozen.** Written 2026-11-02, before any retraining run and before the MNIST test split is used for Phase 4. It is frozen by `run_protocol_stamp.py`, which refuses while an open item (written as the marker in section 8) is left in this file or while `evaluation.macro_f1_guard` in `configs/retraining.yaml` is empty. The stamp (`retraining_protocol_stamp.txt`) stores the SHA-256 of this file and of `retraining.yaml`; the test stages refuse to run if either changes afterwards. A change after the stamp needs a written reason in section 9 and in `../DEVIATIONS.md`, and a re-run under a new revision.

## 1. Questions

1. **Curriculum comparison.** Does training on the easy images first and letting the SOM-chosen hard images in gradually give a better CNN than standard training? Does choosing the hard images with the SOM matter, compared with choosing the same number at random or by low max-softmax confidence?
2. **Targeted augmentation.** Does adding altered copies of the SOM-chosen hard images reduce the errors in the SOM's error hotspots, and does it damage the rest? Is it better than adding copies of the same number of random, or low-confidence, images?

## 2. Frozen inputs (read-only, never changed)

| Input | Where | Why |
|---|---|---|
| CNN run `lenet5_best`, its embeddings, labels, predictions, confidence | `outputs/v1_mnist/p1_model_training_baseline/cnn_baseline_training/` | The model every SOM was built from |
| Data split (70/15/15) | same folder, `embeddings/*_sample_ids.npy` | The split is rebuilt from the saved IDs (checked against the saved labels), not re-drawn from a seed |
| SOM `som_15x15_nb11_ep250_s42_v2`, its training assignments, purity table, validation hotspots | `outputs/v1_mnist/p2_som_training_visualization/` | Defines which images are hard |
| Pool run `hard_example_pools_som_15x15_nb11_ep250_s42_v2` | `outputs/v1_mnist/p4_guided_retraining/hard_example_mining/` | The pools below, with the SHA-256 of each input |

The SOM model and embeddings are checked against their recorded SHA-256 before every step.

## 3. Choosing the hard images (fixed before any retraining result)

A training image is in the **SOM pool** (4,217 images) if any of these is true:
1. its best-matching neuron is one of the 11 error hotspots (`../p2_som_training_visualization/error_pattern_analysis/HOTSPOT_RULE.md`);
2. its neuron's training purity is below 0.95;
3. its distance to its neuron is above the 95th percentile of the training distances (1.569).

The **hotspot pool** (685 images) is rule 1 alone. Controls add the same number of images:
- **random:** n random training images with the same number of images per digit as the SOM pool, redrawn for every seed; **random_hotspot** does the same for the hotspot pool;
- **msp:** the n training images with the lowest max-softmax confidence (ties by sample ID);
- **standard:** nothing added.

Only training images are selected. Labels are always the true labels.

## 4. Retraining

Every arm trains LeNet-5 from scratch with the Phase 1 recipe (Adam, StepLR, 20 epochs, batch 64; the numbers are read from `cnn_baseline.yaml`), seeds 1 to 5, the same seeds for every arm. The checkpoint of a run is its best epoch on validation accuracy. One round per arm.

**Curriculum comparison.** The easy images are all training images outside the pool. In every epoch the training set is all easy images plus a share of the hard pool: 10% in epochs 1 to 5, 20% in epochs 6 to 10, and so on (+10% every 5 epochs, as the proposal's risk note describes), so the last five epochs use 40% of the pool. Pool images are let in from the easiest to the hardest: SOM and hotspot pools by fewer satisfied rules, then closer to the neuron; msp pool by higher confidence; random pools in random order. Arms: `standard`, `curriculum_som`, `curriculum_random`, `curriculum_msp`.

**Targeted augmentation.** The training set is all 49,000 original images plus k = 3 altered copies of each pool image. Alterations are small rotations (up to 10 degrees), shifts (up to 1 pixel; first set to 2 pixels, made smaller under revision r2 because the transform check below failed with 2 pixels: validation accuracy 0.9912 original against 0.9583 altered), zoom (0.9 to 1.1) and shear (up to 8 degrees); no flips. Arms: `standard`, `augmented_som`, `augmented_random`, `augmented_msp` (each adds 12,651 images) and `augmented_hotspot`, `augmented_random_hotspot` (each adds 2,055 images). These arms use all copies from the first epoch.

**Paced augmentation.** Two further arms, `augmented_som_paced` and `augmented_random_paced`, add the same 12,651 images but bring them in gradually with the same pacing as the curriculum: the copies of 10% of the pool in epochs 1 to 5, 20% in epochs 6 to 10, and so on (easiest pool images first), so the share of hard images does not jump to its full size at the start. With 20 epochs the last five epochs use 40% of the pool's copies. The paced arms are compared with their paced control, with `standard`, and with their unpaced twin.

**Monitoring during training.** In every epoch the validation accuracy is recorded for all validation images, for the hotspot images and for the images outside the hotspots. The `summary` stage reports, per arm, the largest one-epoch fall in validation accuracy (overall and outside the hotspots, where a fall would mean the easy images are being forgotten) and how far the last epoch ends below the best one, and saves the curves as a PDF. This is reported to judge stability; it is not a success criterion.

**Transform check (validation only, before any arm is compared).** The ranges are accepted if the Phase 1 CNN's accuracy on altered validation images is within 2 percentage points of its accuracy on the originals (`run_targeted_augmentation.py --stage check`) and the contact sheet shows readable digits. If not, the ranges are made smaller in `retraining.yaml` under a new revision before any arm runs.

## 5. Measures and success criteria (fixed before the test set)

Hotspot images are the images whose neuron in the original SOM is one of the 11 hotspots. Test images are assigned to neurons once, in the final test stage.

- **Macro-F1 guard:** a guided arm's macro-F1 may be lower than its control's by at most **0.0015657542982220032**. The value is 2 times the sample standard deviation of the `standard` arm's validation macro-F1 over the 5 seeds, measured by the `summary` stage before any arms are compared (each of the two experiments has its own `standard` arm; the larger of the two values is used), and copied here and into `retraining.yaml`.
- **Uncertainty:** paired bootstrap over test images (2,000 resamples, seed 2026) of the seed-averaged difference between two arms; seed-to-seed spread is reported next to it. The pool is built once from one SOM, so it does not vary across seeds; this is a stated limitation.
- **Curriculum comparison.**
  - C1 (curriculum helps): `curriculum_som` has a lower overall test error than `standard`, the interval excludes 0, and the macro-F1 guard holds.
  - C2 (SOM guidance matters): `curriculum_som` has a lower hotspot test error than `curriculum_random`, the interval excludes 0, and the guard holds.
- **Targeted augmentation.**
  - A1 (primary, SOM guidance matters): `augmented_som` has a lower hotspot test error than `augmented_random`, the interval excludes 0, and the guard holds.
  - A2 (augmentation helps the hotspots): `augmented_som` has a lower hotspot test error than `standard`, the interval excludes 0, and the guard holds.
  - A3 (no damage elsewhere): the upper end of the interval of (`augmented_som` minus `standard`) on the non-hotspot test error is at most 0.002.
- **Also reported, not part of success:** overall error with intervals, per-digit errors, `msp` and `hotspot` arms against their controls, every arm against `standard`, the paced augmentation arms against their paced control and against their unpaced twin, and the training-stability numbers of section 4.
- **CF1 / CF2:** mean test confusion matrix of `standard` (CF1) and of each other arm (CF2) over the 5 seeds, with the CF2 minus CF1 difference, saved as PDF.
- **Null result:** if a criterion is not met, that is the result. It is reported as it is and the criteria are not changed.

## 6. Leakage control

- Pools, copies and training use training images only; all development checks (transform check, macro-F1 guard) use validation images only.
- The 11 hotspots were chosen from validation errors. This is declared. It is why the final judgement is made on the test split.
- The test split is opened once, by the `final_test` stage, after this file has no open item and its stamp matches. Each stage checks the SHA-256 of every checkpoint it evaluates.
- Finished runs are never overwritten; a changed run gets a new revision.

## 7. Decisions taken on 2026-11-02 and what each is based on

| Decision | Value | Basis |
|---|---|---|
| Pool rule | hotspot or purity < 0.95 or QE flag | The hotspot rule and the QE percentile rule already exist in this project. Combining difficulty signals follows the "difficulty measurer" idea of the curriculum-learning literature. |
| Purity cutoff | 0.95 | A round number. **Not from research.** Not tuned on results. |
| QE flag percentile | 95 | Same percentile as the novelty-detection step. |
| Curriculum schedule | 10% of the pool, +10% every 5 epochs | The proposal's risk note. With 20 epochs it ends at 40% of the pool; this is a consequence of the proposal's numbers, not a choice. Another schedule can be added in the config as a new named schedule. |
| Paced augmentation arms | same schedule as the curriculum | The proposal's risk note applies to curriculum learning and to targeted augmentation. The headline criteria A1 to A3 stay on the unpaced arms; the paced arms are reported next to them. |
| Order of unlocking | easiest first | Curriculum learning: easy to hard (Soviany et al. 2022; Wang et al. 2022). |
| k (copies per image) | 3 | A middle value. **Not from research.** |
| Transforms and ranges | small rotation, shift, zoom, shear | Label-preserving transforms are what the augmentation literature advises (Shorten and Khoshgoftaar 2019; Chen, Dobriban and Lee 2020). The numeric ranges are **conventions**, checked as in section 4. |
| Seeds | 1 to 5 | The group proposal asks for at least 5 seeds. |
| Controls with equal added images | random, msp | Without them, a gain could come from more data alone. |
| CF1 | `standard` arm, 5 seeds | Same code and recipe as the other arms. `lenet5_best` is a single seed chosen on validation. |
| Training loop | own loop on in-memory tensors, same recipe as Phase 1 | A curriculum changes the training set between epochs, which a fixed data loader does not do. |
| Damage tolerance | 0.002 on the non-hotspot error | A convention (0.2 percentage points). **Not from research.** |

Any of these can change before the freeze; none can change after.

## 8. Still open (fill before the freeze)

None

## 9. Amendments and deviations

None yet. Departures from the proposal's wording (augmentation strength, how cluster-boundary samples are approximated, the 40% end point of the schedule, a separate `standard` arm per experiment) are listed in `src/v1_mnist/docs/DEVIATIONS.md`.
