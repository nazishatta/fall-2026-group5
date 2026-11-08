# SOM re-analysis protocol (Phase 4)

Status: **fixed before any new SOM is trained.** This file and `configs/som_reanalysis.yaml` are read by the code; their SHA-256 values are recorded with every run and checked between the stages, so changing either after the first stage stops the later stages.

## 1. Questions

After retraining, the proposal asks to re-extract features from the retrained model, re-train the SOM on the new embeddings, and compare the old and new maps:

1. Did the cluster boundaries sharpen?
2. Did the error hotspots shrink?
3. How does the loop SOM analysis → targeted retraining → SOM structure look in numbers? (written up in `FEEDBACK_LOOP.md`)

## 2. Inputs (read-only)

- The retrained LeNet-5 checkpoints of the targeted augmentation run (run folder named in `configs/retraining.yaml`, section `augmentation`), arms `standard`, `augmented_som`, `augmented_random`, seeds 1 to 5. Their SHA-256 values are checked against the values recorded after training.
- The frozen split: 49,000 training and 10,500 validation images, rebuilt from the saved sample IDs. **The test split is not read.**
- The SOM recipe of the selected Phase 2 SOM (`configs/som.yaml`: 15x15, `init_neighborhood` 11, 250 epochs, 250 steps, seed 42, MinMax scaling to [-1, 1] fitted on the training embeddings).
- The reference map: the selected Phase 2 SOM (`som_15x15_nb11_ep250_s42_v2`) on the Phase 1 embeddings of `lenet5_best`. It is loaded, not retrained.

## 3. Procedure for every (arm, seed) cell

1. Extract the 84-dimensional `fc2` embeddings of the training and validation images from the checkpoint, with the same layer as Phase 1. Check: the predictions computed during the extraction must agree with the validation predictions saved after the retraining for at least 99.9% of the images; otherwise the cell stops.
2. Fit the scaler on the training embeddings only and train a new SOM with exactly the recipe above (no setting is tuned).
3. Assign the training and validation embeddings to the neurons of the new SOM.
4. Compute the measures of section 4 on the validation images (labels and the retrained CNN's predictions).

## 4. Measures (validation images)

| ID | Measure | Better |
|---|---|---|
| M1 | weighted purity: share of images whose class is the majority class of their neuron | higher |
| M2 | low-purity share: share of images in neurons with purity below 0.95 (the cut-off of the hard-image pool) | lower |
| M3 | class scatter index: average number of separate groups of neighbouring neurons per class (neurons are labelled with their majority class; neighbouring means a map distance of at most 1.1) | lower |
| M4 | topographic error, first order, in percent (NNSOM definition used in Phase 2) | lower |
| M5 | hotspot error rate: the rule of Phase 2 (top k = max(5, round(5% of the neurons)) = 11 neurons by the Wilson lower bound of the error rate) is applied to the new map; M5 is the error rate of the retrained CNN on the images in those neurons | lower |
| M6 | share of all validation errors that lie in the hotspot neurons | reported only |
| M7 | overall validation error of the retrained CNN | reported only |
| M8 | linear CKA between the embeddings of two cells (validation images) | reported only |

Quantization error is stored for every map but **not compared** between maps: it is measured in the units of the embeddings, and the embeddings of different CNNs are not on the same scale.

## 5. Statistics and success statements

For every measure the difference between two arms is formed **per seed** (the same CNN seed in both arms), giving 5 paired differences. Reported: mean, standard deviation, the 95% interval of the mean from the paired t-distribution (4 degrees of freedom), and the share of seeds in which the first arm is better (called P(A better); with 5 seeds it can only be 0, 0.2, ..., 1). The SOM seed is fixed (42), so the variation between seeds comes from the CNN; this is a stated limitation.

Pairs: `augmented_som` against `standard` (effect of retraining with guided augmentation) and `augmented_som` against `augmented_random` (effect of the SOM guidance itself).

| Statement | Rule |
|---|---|
| R1: boundaries sharpen (against `standard`) | M1 difference above 0 and its interval excludes 0 |
| R2: hotspots shrink (against `standard`) | M5 difference below 0 and its interval excludes 0 |
| R3: guidance sharpens the boundaries (against `augmented_random`) | as R1 |
| R4: guidance shrinks the hotspots (against `augmented_random`) | as R2 |
| R5: the map is not less organised (against `standard`) | the upper end of the M4 interval is at most 1.0 percentage point |

A statement that is not met is a result and is reported as it is. M2, M3 and the reference map are reported next to the statements, not used to decide them.

## 6. Leakage control

- Everything above uses training and validation images only; no test image is read.
- The scaler and the SOM are fitted on training embeddings only.
- The SOM recipe is the Phase 2 recipe; nothing is tuned on these results.
- Statements, thresholds and the arm list are fixed here before any cell is run.
- **Disclosure:** the choice of arms was made after the retraining test pass was known. They are the arms with a measurable effect of the guidance on the hotspots (criteria A1 to A3 of the retraining protocol were met). The curriculum arms can be added in the configuration as a secondary analysis; they are not in the fixed list.

## 7. Decisions and what each is based on

| Decision | Value | Basis |
|---|---|---|
| Arms | `standard`, `augmented_som`, `augmented_random` | A control with the same recipe separates the effect of retraining from the effect of the guidance. Pairing by seed follows Bouthillier et al. (2021), who recommend re-using the same seeds for the two methods. |
| Seeds | 1 to 5 | The proposal asks for at least 5. Bouthillier et al. estimate that about 29 trainings per method are needed for a reliable decision on small differences, so this analysis can only detect large effects. |
| Reference map | Phase 2 SOM on `lenet5_best` | Shows where the project started; not used for the statements because it is one seed chosen on validation. |
| M1, M2 | purity of Phase 2 | Already the project's measure of cluster structure. |
| M3, M4 | class scatter index, topographic error | Both are among the measures surveyed by Forest et al. (2020). The implementation of M3 follows the survey's verbal definition and is our own; it is not the code of that library. |
| M5 | Phase 2 hotspot rule | Same rule as in Phase 2 and in the pool construction. |
| M8 | linear CKA | Kornblith et al. (2019): a similarity index for representations of different networks. |
| Interval | paired t-interval over seeds | A convention for 5 paired values. **Not from research.** |
| M4 tolerance | 1.0 percentage point | A convention. **Not from research.** |

## 8. Limitations

- 5 seeds and one SOM seed: low power; only large effects can be seen.
- The hotspots are chosen as the worst neurons of each map, so their error rate is selected; the comparison with `standard` and `augmented_random`, which are treated the same way, is what controls for this.
- The retrained CNNs are different networks, so neurons of different maps cannot be matched one to one; only map-level measures are compared.
- The arm list was fixed after the retraining test results were known (section 6).

## 9. Amendments and deviations

None yet.
