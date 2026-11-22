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

Everything else: [TBD]

## 6. References

As in the MNIST Phase 2 protocols and `reports/Latex_report/v1_mnist/Final_report/references.bib`.
