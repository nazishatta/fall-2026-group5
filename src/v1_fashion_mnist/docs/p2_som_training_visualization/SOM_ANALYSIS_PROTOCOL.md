# Protocol: SOM training and visualization on Fashion-MNIST (Phase 2)

Part of the Fashion-MNIST protocol, one file per phase:
[Phase 1](../p1_model_training_baseline/MODEL_TRAINING_PROTOCOL.md) ·
[Phase 2](SOM_ANALYSIS_PROTOCOL.md) ·
[Phase 3](../p3_novelty_detection/NOVELTY_DETECTION_PROTOCOL.md) ·
[Phase 4](../p4_guided_retraining/RETRAINING_PROTOCOL.md).
Changes and deviations: [`../DEVIATIONS.md`](../DEVIATIONS.md).

Written on 2026-11-17, **before any Fashion-MNIST data was loaded, any network trained or any result seen**,
and committed as one file in `5dd164c` (`docs/FASHION_MNIST_PROTOCOL.md`). Split into one file per phase on
2026-11-21 with no rule changed (see `../DEVIATIONS.md`). Sections 1 to 4 are not changed; a change needs a
written reason in `../DEVIATIONS.md` and, for a run already made, a new run ID.

**Rule for every phase:** every step done for MNIST is done for Fashion-MNIST, with the MNIST design copied
unchanged, unless the dataset makes that impossible. Inputs: the Phase 1 network and embeddings
([`MODEL_TRAINING_PROTOCOL.md`](../p1_model_training_baseline/MODEL_TRAINING_PROTOCOL.md)).

## 1. What is fixed before anything runs

| Item | Fixed value | Same as MNIST? |
|---|---|---|
| SOM (headline) | NNSOM, 15x15, starting neighbourhood 11, 250 epochs, seed 42, `MinMaxScaler(-1, 1)` fitted on the training embeddings only. Fixed now, as on MNIST; the grid comparison and Optuna (section 2) check it but do not replace it | yes |
| SOM (grid comparison) | 10x10 with neighbourhood 7 and 20x20 with neighbourhood 15, other settings as above | yes |
| Hotspot rule | top k = max(5, round(5% of 225)) = 11 neurons by the Wilson lower bound of the validation error rate | yes |
| Seeds | SOM 42; SOM seeds 1-5 in the layer ablation | yes |

The headline map keeps MNIST's settings; the grid comparison and Optuna measure how good those settings are
for Fashion-MNIST.

## 2. Steps

Each MNIST document named here is followed unchanged, with Fashion-MNIST data and the settings of section 1.
All of Phase 2 uses training and validation images only.

**SOM training and grid comparison** (`SOM_GRID_SELECTION_PROTOCOL.md`, `PREREGISTRATION_grid_comparison.md`).
Train the 10x10, 15x15 and 20x20 maps; report quantization error, topographic error and purity on validation.
The headline map is 15x15 (section 1).

**Optuna tuning** (`OPTUNA_TUNING_PROTOCOL.md`). Stage 1: 20 trials per grid with the same budget (first 8
random, then TPE, seed 42), two objectives on validation (QE and first-order topographic error), one setting
picked per grid by the MNIST selection rule. Stage 2: that setting checked with 5 seeds. As on MNIST, **the
headline map stays 15x15 with neighbourhood 11** whatever Optuna finds; a better setting is reported as a
limitation of the headline map, and Phases 2 to 4 are not re-run on it.

**RQ1 (descriptive, no pass/fail).** Weighted purity on validation against the label-shuffled baseline
(1,000 shuffles, seed 20261010); class maps; overlap between classes; the interactive report.

**RQ2 (error geography)** (`ERROR_PATTERN_ROBUSTNESS_CRITERIA.md`, section 3, and `HOTSPOT_RULE.md`). Hotspots
are picked on one random half of the validation errors and judged on the other half (100 splits, seeds from
7000); the median held-out ratio must beat the label-shuffled baseline (200 shuffles, seeds from 9000) with
p < 0.05, and its 2.5th percentile must be above 1. Judged on the 15x15 map, and checked on the 10x10 and
20x20 maps as on MNIST.

**Error-pattern baselines** (`error_pattern_analysis/baselines/PLAN.md`) and the **same-budget comparison of
the error detectors** (exploratory, as on MNIST): the SOM's error cells against the MNIST set of non-SOM
error detectors, on the same validation embeddings.

**Feature-layer ablation** (`FEATURE_LAYER_ABLATION_PROTOCOL.md`): SOMs on `fc1` (120 numbers) and `fc2`
(84 numbers), SOM seeds 1-5, paired; statements L1 (primary, error-cell AUROC), L2 (weighted purity) and L3
(QE novelty AUROC) judged on fc1 minus fc2 with a 95% t-interval. For L3 the outside set is the first one
listed in the Phase 3 protocol, section 2 (SVHN, fixed on 2026-11-21); L3 is run only after that. `fc2` stays
the headline layer whatever the result.

The MNIST documents named above are in `src/v1_mnist/docs/p2_som_training_visualization/`.

## 3. Leakage control

- The scaler, the map, the hotspots and the error cells are all fitted or chosen on training or validation
  data; the test split is not read in Phase 2.
- Every run records the SHA-256 of its inputs; a finished run is never overwritten.
- **Declared:** this protocol was written after the MNIST results were known. It copies MNIST's design
  instead of adapting it to those results, and no Fashion-MNIST result existed when it was written.

## 4. Limitations stated before the run

- One CNN and one SOM in Phases 2 and 3, as for MNIST; seed-to-seed variation is measured only in the
  layer ablation (SOM seeds) and Phase 4 (CNN seeds).
- The SOM settings are not tuned for Fashion-MNIST, so the results describe the MNIST recipe on
  Fashion-MNIST, not the best possible Fashion-MNIST setup.

## 5. Recorded outcome

Filled in after the runs, without changing sections 1 to 4.

**Optuna tuning (recorded 2026-11-21).** Stage 2 confirmation, Optuna's choice minus the 3-point
setting, 5 new SOM seeds (1-5, not the search seed), mean and paired 95% t-interval on validation.
Negative is better for Optuna. Source: `outputs/v1_fashion_mnist/p2_som_training_visualization/som_optuna_tuning/som_optuna_confirm_<grid>_seeds5_v1/confirmation.json`.

| Grid | 3-point setting | Optuna setting | QE | TE1 (points) | TE1+2 (points) | Verdict |
|---|---|---|---|---|---|---|
| 10x10 | nb 7, 250 epochs | nb 7, 300 epochs | +0.020 [+0.007, +0.034] | +0.79 [+0.37, +1.21] | +0.22 [-0.33, +0.78] | 3-point setting better |
| 15x15 | nb 11, 250 epochs | nb 7, 200 epochs | -0.014 [-0.024, -0.004] | +2.36 [+1.55, +3.17] | +1.53 [+0.94, +2.11] | trade-off |
| 20x20 | nb 15, 250 epochs | nb 14, 350 epochs | -0.007 [-0.018, +0.005] | +0.12 [-0.80, +1.03] | -0.25 [-1.10, +0.60] | no clear difference |

Optuna's setting is not better than the 3-point setting on any grid. On the headline 15x15 grid it gives
a slightly lower quantization error but a clearly higher topographic error (+2.4 points first-order), so
it fits the data a little more closely while keeping neighbours less well together. The headline map stays
15x15, neighbourhood 11, 250 epochs, as section 2 requires, and this is not a limitation of the headline
map: no setting found improves both measures. No test data was used (`test_evaluated: false`).

**Feature-layer ablation (recorded 2026-11-21).** SOMs on `fc1` and `fc2`, same recipe as the headline map,
SOM seeds 1-5, paired by seed; fc1 minus fc2, mean and 95% t-interval over the 5 seeds. Source:
`outputs/v1_fashion_mnist/p2_som_training_visualization/feature_layer_ablation/feature_layer_ablation_som15x15_nb11_ep250_seeds5/tables/statements.csv`.

| Statement | fc1 minus fc2 | Seeds where fc1 is larger | Verdict | MNIST |
|---|---|---|---|---|
| L1 (primary): error-cell AUROC | -0.055 [-0.066, -0.043] | 0 of 5 | **fc2 better** | fc2 better |
| L2: weighted purity | -0.048 [-0.051, -0.045] | 0 of 5 | **fc2 better** | fc2 better |
| L3: QE novelty AUROC on SVHN | +0.087 [+0.027, +0.148] | 5 of 5 | **fc1 better** | fc1 better (+0.005) |

As on MNIST, `fc2`, the headline layer, gives maps that find the network's mistakes better and hold purer
classes, while `fc1` separates SVHN better. On Fashion-MNIST the L3 gap is much larger than on MNIST (+0.087
against +0.005), so the layer choice trades error finding against far-novelty detection. `fc2` stays the
headline layer, as fixed in section 2. The `fc2` maps' QE AUROC on SVHN ranges from 0.766 to 0.884 over the
5 seeds (Phase 3 protocol, section 7). No test data was used.

**Error-pattern baselines and the same-budget comparison (recorded 2026-11-21; exploratory, as on MNIST).**
Predicting which of the 876 validation images the network gets wrong, on the same validation embeddings.
Sources: `outputs/v1_fashion_mnist/p2_som_training_visualization/error_pattern_analysis/error_pattern_baselines/error_pattern_baselines_som_15x15/summary.csv`
and `.../error_detector_budget/error_detector_budget_som_15x15/summary.csv`.

| Method | AUROC [95% interval] | Hotspot ratio | Mistakes caught at the SOM hotspots' share of images (3.5%) | MNIST AUROC |
|---|---|---|---|---|
| SOM cell error rate | 0.811 [0.796, 0.824] | 3.9 [3.2, 4.4] | 13.7% | 0.685 |
| k-means cell error rate (10 seeds) | 0.812 [0.805, 0.821] | 4.0 [3.7, 4.5] | (15.8% at its own 4.0%) | 0.693 |
| Nearest-neighbour impurity | 0.876 [0.864, 0.888] | | 21.1% [19.2, 22.9] | 0.874 |
| Max-softmax | 0.906 [0.898, 0.914] | | 21.9% [19.8, 24.1] | 0.981 |

As on MNIST, the SOM's cells find mistakes no better than k-means cells, and the network's own confidence
finds them best. The gap is smaller on Fashion-MNIST (max-softmax minus SOM: 0.095 AUROC, against 0.296 on
MNIST), because the mistakes lie in large mixed regions of the map rather than in a few small cells. The
SOM's value in Phase 2 is the map of where classes mix and where the mistakes sit, not error detection. The
SOM hotspot analysis reproduced the Phase 2 robustness result exactly.

**SOM training (recorded 2026-11-21).** The three maps of section 1, seed 42, measured on the 10,500
validation images. Source: `outputs/v1_fashion_mnist/p2_som_training_visualization/som_nnsom_training/tables/som_grid_ablation.csv`.

| Grid | Neighbourhood | QE | TE1 (%) | TE1+2 (%) | Neurons used |
|---|---|---|---|---|---|
| 10x10 | 7 | 1.008 | 4.26 | 1.76 | 99 of 100 |
| 15x15 (headline) | 11 | 0.899 | 5.37 | 3.55 | 222 of 225 |
| 20x20 | 15 | 0.822 | 7.33 | 4.75 | 396 of 400 |

As on MNIST, bigger maps fit the images more closely (lower QE) but keep neighbours together less well
(higher topographic error), so no grid is best on every measure. The table's rank sum (10x10: 5, 15x15: 6,
20x20: 7) is for reference only; the headline map stays 15x15 as fixed in section 1. Purity for each grid
comes from the grid comparison (10 maps per grid): [TBD] until it finishes.

**RQ1: how well the map keeps classes apart (recorded 2026-11-21; descriptive, no pass/fail).** Source:
`outputs/v1_fashion_mnist/p2_som_training_visualization/cluster_analysis/som_15x15_nb11_ep250_s42/cluster_summary.json`.

| Measure | Fashion-MNIST | MNIST |
|---|---|---|
| Weighted purity on validation | 0.893 | 0.982 |
| Label-shuffled baseline, mean [95% range] (1,000 shuffles, seed 20261010) | 0.173 [0.170, 0.177] | 0.174 [0.170, 0.177] |
| p-value (smallest possible with 1,000 shuffles: 0.001) | 0.001 | 0.001 |

The map keeps the classes far better than chance: the highest of the 1,000 shuffles was 0.180. Purity is lower than on
MNIST, which fits the network's higher validation error rate (8.3% against 0.9%): more images sit in
neurons where classes mix. Purity on the training images is 0.900, so the map is not much purer on the
images it was trained on. The class maps are in `outputs/v1_fashion_mnist/p2_som_training_visualization/cluster_analysis/figures/` and the interactive
report is `outputs/v1_fashion_mnist/p2_som_training_visualization/interactive_report/interactive_report.html` (DVC).

**RQ2: where the mistakes sit (recorded 2026-11-21).** The network gets 876 of the 10,500 validation images
wrong (8.3%); its mean confidence is 0.686 on these and 0.942 on the rest. Hotspots are picked on one half
of the mistakes and judged on the other half (section 2). The held-out ratio is the hotspots' error rate
divided by the overall error rate (1 = no concentration). Sources:
`outputs/v1_fashion_mnist/p2_som_training_visualization/error_pattern_analysis/robustness/*_error_pattern_robustness.json` and
`error_pattern_robustness_verdict.json` in the same folder.

| Map | Hotspots | Held-out ratio, median [2.5th, 97.5th percentile] | Chance median (95th percentile) | p | Error rate against purity (Spearman) | Verdict | MNIST median ratio |
|---|---|---|---|---|---|---|---|
| 15x15 (headline) | 11 | 3.90 [3.23, 4.44] | 1.00 (1.16) | 0.005 | -0.95 | **passes** | 11.1 |
| 10x10 | 5 | 3.81 [3.07, 4.37] | 1.00 (1.22) | 0.005 | -0.93 | holds | 10.5 |
| 20x20 | 20 | 3.81 [3.14, 4.35] | 1.00 (1.18) | 0.005 | -0.95 | holds | 13.4 |

p = 0.005 is the smallest possible with 200 shuffles. On the headline map both pre-written criteria pass:
the mistakes are concentrated in the hotspots (median ratio above chance with p < 0.05, 2.5th percentile
above 1), and the neurons with more mistakes are the ones where classes mix (error rate falls as purity
rises, and rises with class entropy, rho = 0.96). The result holds on the 10x10 and 20x20 maps.

The ratios are much lower than on MNIST, but the two are not comparable in size: a ratio cannot exceed one
divided by the overall error rate, which is about 12 on Fashion-MNIST and about 114 on MNIST. The
Fashion-MNIST mistakes are spread over larger mixed regions of the map rather than packed into a few
cells, which also explains the error-detection results above. On the images used to pick them, the 11
hotspots hold 4.0% of the images and 19.7% of the mistakes; this overstates the concentration and is
descriptive only. Mistakes also sit a little further from their neuron than correct images (mean distance
0.994 against 0.840).

No test data was used in any of these runs (`test_used: false`, `test_evaluated: false`).

**Grid comparison:** running; recorded here when it finishes ([TBD]).

## 6. References

As in the MNIST Phase 2 protocols and `reports/Latex_report/v1_mnist/Final_report/references.bib`.
