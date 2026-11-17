# Protocol: the pipeline on Fashion-MNIST (all phases)

Written on 2026-11-17, **before any Fashion-MNIST data was loaded, any network trained or any result seen**.
It is committed before the first run; that commit's hash goes in section 11. After that, the rules
below are not changed. A change needs a written reason in section 11 and, for a run already made, a new run ID.

## 1. Why this exists

The group proposal (`proposals/Group_proposal.md`, sections 3.4 and 4) says the whole pipeline is repeated
on Fashion-MNIST. There are two reasons:

1. **More mistakes.** The MNIST network makes only 92 validation and 86 test mistakes, which makes the
   per-cell statistics weak (MNIST limitation 1 in the README).
2. **Reuse.** The professor's proposal (`proposals/Project_proposal.md`, objective 7 and deliverables)
   asks for an analysis pipeline that is "reusable across datasets/models". Running the same code on a
   second dataset, changing only the settings, is the test of that.

So the rule for this whole protocol is: **every step done for MNIST is done for Fashion-MNIST, with the MNIST
design copied unchanged, unless the dataset makes that impossible.** The results go in the same step folders
as MNIST's. Every change is listed in section 10.

## 2. What is fixed before anything runs

| Item | Fixed value | Same as MNIST? |
|---|---|---|
| Data | Fashion-MNIST, 70,000 grayscale 28x28 images, 10 classes, `torchvision.datasets.FashionMNIST` (MIT licence; Xiao et al., 2017) | different dataset |
| Classes (torchvision order 0-9) | T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot | n/a |
| Split | the original 60,000 + 10,000 images joined (sample IDs 0-59,999 and 60,000-69,999), then a 70/15/15 stratified split with seed 42: 49,000 / 10,500 / 10,500 | yes |
| Preprocessing | pixel values divided by 255; nothing else | yes |
| Network | LeNet-5 and its full training recipe from `src/v1_mnist/component/configs/` (`base.yaml`: 20 epochs, Adam, learning rate 0.001, weight decay 0.0001, StepLR every 10 epochs by 0.1, batch 64, seed 42) | yes, **not tuned for Fashion-MNIST** |
| Embeddings | `fc2` (84 numbers) for every split, with sample IDs, labels, predictions and confidence | yes |
| SOM (headline) | NNSOM, 15x15, starting neighbourhood 11, 250 epochs, seed 42, `MinMaxScaler(-1, 1)` fitted on the training embeddings only. Fixed now, as on MNIST; the grid comparison and Optuna (section 4) check it but do not replace it | yes |
| SOM (grid comparison) | 10x10 with neighbourhood 7 and 20x20 with neighbourhood 15, other settings as above | yes |
| Hotspot rule | top k = max(5, round(5% of 225)) = 11 neurons by the Wilson lower bound of the validation error rate | yes |
| Seeds | CNN 42; SOM 42; SOM seeds 1-5 in the layer ablation; retraining seeds 1-5 | yes |

The network is deliberately not tuned, and the headline map keeps MNIST's settings. The aim is to see whether
the pipeline, with MNIST's settings, still finds structure on a harder dataset; the grid comparison and Optuna
measure how good those settings are for Fashion-MNIST. Fashion-MNIST accuracy will therefore be lower than
what a tuned network reaches; that is expected and is not a weakness of the analysis.

## 3. Phase 1: model training baseline

- Train once with seed 42. Report accuracy with a 95% Wilson interval, macro-F1, per-class F1 and the
  confusion matrix.
- As for MNIST, the training script evaluates the test split **once**, at the end. Nothing (epochs,
  recipe, map, thresholds) is chosen from that number.
- No pass/fail.

## 4. Phase 2: SOM training and visualization

Each MNIST document named here is followed unchanged, with Fashion-MNIST data and the settings of section 2.
All of Phase 2 uses training and validation images only.

**SOM training and grid comparison** (`SOM_GRID_SELECTION_PROTOCOL.md`, `PREREGISTRATION_grid_comparison.md`).
Train the 10x10, 15x15 and 20x20 maps; report quantization error, topographic error and purity on validation.
The headline map is 15x15 (section 2).

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
listed in section 5, once chosen; L3 is run only after that. `fc2` stays the headline layer whatever the result.

## 5. Phase 3: novelty detection

**In-distribution reference.** The 10,500 Fashion-MNIST validation images. The Fashion-MNIST test split is
not used in Phase 3.

**Outside sets (never used for training, fitting, thresholds or choices).** Which datasets are used is
**[TBD: to be agreed with the professor and written here before any Phase 3 run]**. Candidates are SVHN
(as for MNIST), MNIST and KMNIST. Whatever is chosen, each set is loaded at 28x28 grayscale with pixels in
[0, 1], used only to compute novelty scores, and recorded here with its licence and citation. The rules
below do not depend on the choice.

**Scores** (higher means "more novel"; anything fitted is fitted on the Fashion-MNIST training embeddings
only; the functions are the MNIST Phase 3 functions, called unchanged):

| ID | Score |
|---|---|
| S1 | SOM quantization error (QE) |
| S2 | U-Matrix value of the best-matching neuron |
| S3 | max-softmax: 1 minus the largest softmax probability |
| S4 | energy: minus log-sum-exp of the logits, temperature 1 |
| S5 | Mahalanobis: smallest squared distance to the class means, one shared Ledoit-Wolf covariance |
| S6 | nearest-neighbour distance: mean distance to the 5 nearest training embeddings |

**Measures.** Per outside set and score: AUROC (outside images positive), AUPR-out and FPR at 95% TPR, with a
stratified bootstrap 95% interval for the AUROC (2,000 resamples, seed 42). Differences between S1 and each
other score use the paired bootstrap over images. **Control:** the S1 AUROC with the labels shuffled 1,000
times (seed 42); it should sit near 0.5.

**Statements** (each is met, not met, or inconclusive; a statement that is not met is reported as it is):

| ID | Statement | Met if |
|---|---|---|
| O1 (RQ3) | the SOM detects outside images | on every outside set, the S1 AUROC interval lies above 0.5 |
| **O2 (RQ4, primary)** | the SOM is competitive with max-softmax | on every outside set, the lower end of the paired interval of S1 minus S3 is above -0.01 |
| O3 (RQ4) | the SOM is competitive with energy, Mahalanobis and nearest neighbours | as O2, for S4, S5 and S6 separately |
| O4 | the U-Matrix detects outside images | on every outside set, the S2 AUROC interval lies above 0.5 |

There is one network and one map here, so no claim of "better" is made: the group proposal (RQ4) requires
at least 5 seeds for that.

**Further Phase 3 steps, as on MNIST** (`PREREGISTRATION_qe_novelty.md` and the novelty-benchmark and U-Matrix
documents): the QE flag at the 95th percentile of the training QE with its detection and false-alarm rates;
the threshold sweep; the random-image control; the classical baselines (Isolation Forest, One-Class SVM and
the autoencoder, with standardised inputs as in MNIST's final setup); the same-budget baseline comparison; the
check of how much QE depends on the embedding length; and the U-Matrix figures and outside-set overlays.

**Not repeated:** MNIST steps whose only job was to re-create or audit MNIST's own earlier runs (the historical
baseline verification, the historical SVHN inspection, the U-Matrix and random-control reproductions, the
provenance audit, the 2026-11-02 reorganisation and the evidence consolidation). Fashion-MNIST has no earlier
runs to re-check. Diagnostics that MNIST added to explain a problem it found (the autoencoder scaling checks
and the SVHN length diagnostic) are run only if the same problem appears, and that is recorded in section 11.

**Near novelty (held-out classes).** `src/v1_mnist/docs/p3_novelty_detection/held_out_class_novelty/HELD_OUT_CLASS_NOVELTY_PROTOCOL.md`
unchanged, with Fashion-MNIST classes. The five splits are the same five sets of class numbers, drawn by the
same rule (numpy `default_rng(2026)`, 4 unknown classes per split), so they are not chosen with knowledge of
the clothing classes: [0, 1, 5, 6], [2, 3, 5, 8], [1, 5, 6, 7], [2, 5, 6, 8], [0, 4, 5, 8]. One network and
one map per split, recipes as in section 2. Statements H1, H2 (primary: SOM better than max-softmax) and H3
unchanged. Training and validation images only.

## 6. Phase 4: guided retraining

`src/v1_mnist/docs/p4_guided_retraining/RETRAINING_PROTOCOL.md` unchanged, except as stated here.

- **Hard-image pools** from the Fashion-MNIST map, with the same rules: hotspot neuron (section 2), or
  training purity below 0.95, or QE above the 95th percentile of the training QE. The random and
  max-softmax pools have the same size and class mix.
- **Arms** (5 seeds each, 1-5, same seeds in every arm), all the MNIST arms:
  - targeted augmentation: `standard`, `augmented_som`, `augmented_random`, `augmented_msp`,
    `augmented_hotspot`, `augmented_random_hotspot`, `augmented_som_paced`, `augmented_random_paced`; 3 altered
    copies per pool image (rotation up to 10 degrees, shift up to 1 pixel, zoom 0.9-1.1, shear up to 8
    degrees, no flips), pacing as on MNIST;
  - curriculum: `standard`, `curriculum_som`, `curriculum_random`, `curriculum_msp`, with the MNIST schedule.
- **Transform check** (validation only, before any arm is compared): the Phase 1 network's accuracy on
  altered validation images must be within 2 percentage points of its accuracy on the originals. If not, the
  ranges are made smaller under a new revision before any arm runs, and that is recorded in section 11.
- **Macro-F1 guard:** the same rule as MNIST, 2 times the sample standard deviation of the `standard` arm's
  validation macro-F1 over the 5 seeds (the larger of the two experiments), measured before any arms are
  compared. Value: **[TBD: measured by the rule, copied here and into `retraining.yaml` before the test stage]**.
  The MNIST value (0.00157) is not reused, because it measures MNIST's seed-to-seed spread.
- **Statements**, unchanged: A1 (primary: `augmented_som` beats `augmented_random` on hotspot test error,
  interval excludes 0, guard holds), A2, A3 (non-hotspot damage at most 0.002), C1, C2. Paired bootstrap
  over test images (2,000 resamples, seed 2026).
- **SOM re-analysis** (`SOM_REANALYSIS_PROTOCOL.md`): statements R1-R5, `augmented_som` against `standard` and
  against `augmented_random`; and the secondary run on the max-softmax arm (`MSP_ARM_ADDENDUM.md`), as on MNIST.
- **Novelty after retraining** (`NOVELTY_AFTER_RETRAINING_PROTOCOL.md`): unchanged, with the outside sets of
  section 5; run only after they are chosen.
- The test split is opened once, by the `final_test` stage, after the macro-F1 guard is filled in.

## 7. Leakage control

- The Fashion-MNIST test split is used twice, both fixed in advance: once at the end of Phase 1 training
  (reported only), and once in the Phase 4 `final_test` stage. No setting is chosen from either.
- The scaler, the map, the Mahalanobis means and covariance, the nearest-neighbour reference, thresholds,
  hotspots and pools are all fitted or chosen on training or validation data.
- Outside sets are used only to compute novelty scores; nothing is fitted or chosen on them.
- The held-out-class networks never see their unknown classes, and their test split is never read.
- Every run records the SHA-256 of its inputs; a finished run is never overwritten.
- **Declared:** this protocol was written after the MNIST results were known. It copies MNIST's design
  instead of adapting it to those results, and no Fashion-MNIST result existed when it was written.

## 8. Limitations stated before the run

- One CNN and one SOM in Phases 2 and 3, as for MNIST; seed-to-seed variation is measured only in the
  layer ablation (SOM seeds) and Phase 4 (CNN seeds).
- The network and the SOM settings are not tuned for Fashion-MNIST, so the results describe the MNIST
  recipe on Fashion-MNIST, not the best possible Fashion-MNIST setup.
- The hotspots come from the SOM, which gives the SOM-guided arms a built-in advantage on hotspot error (as for MNIST).
- If more than one outside set is used, they differ from Fashion-MNIST in different ways; results are
  reported per set and not averaged.
- The held-out-class splits share classes, so they are not independent.

## 9. Where things go

- Settings: `src/v1_fashion_mnist/component/configs/` (`base.yaml` sends every result to `outputs/v1_fashion_mnist/`).
- Runner: `src/v1_fashion_mnist/component/pipeline/run_all.py`. Step code: shared, in `src/v1_mnist/component/`.
- Results: `outputs/v1_fashion_mnist/<phase>/<step>/`, the same step folders as `outputs/v1_mnist/`; large files in DVC (S3), small `.json`/`.csv` in git.
- Reports: `reports/Latex_report/v1_fashion_mnist/<phase>/`.

## 10. Changes from the MNIST design, and why

| Change | Why |
|---|---|
| Outside sets to be agreed with the professor (MNIST used SVHN only) | for Fashion-MNIST, MNIST itself could be an outside set; the choice is left to the professor |
| Macro-F1 guard re-measured | it measures the spread of the network's own results, which differs by dataset |
| MNIST's history-checking steps not repeated (section 5) | they re-created or audited earlier MNIST runs; Fashion-MNIST has none |
| U-Matrix and nearest-neighbour scores judged (O3, O4) in one run | the MNIST unified benchmark compared them in a historical run; here they are fixed in advance |

## 11. Amendments and deviations

| Date | What changed | Why | Run affected |
|---|---|---|---|
| | Protocol committed: commit [TBD: hash of the commit that adds this file] | | none |

## 12. Recorded outcome

Filled in after the runs, without changing sections 2 to 8. [TBD]

## 13. References

As in the MNIST protocols and `references.bib`; the outside sets' references are added when they are chosen.
