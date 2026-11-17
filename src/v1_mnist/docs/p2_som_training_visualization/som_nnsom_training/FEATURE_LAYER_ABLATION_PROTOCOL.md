# Protocol: feature-layer ablation (fc1 against fc2)

Status: **written 2026-11-17, before any fc1 embedding is extracted or any SOM of this step is trained.**
Commit this file and `configs/feature_layer_ablation.yaml` before running; the run records the SHA-256 of
both and stops if either changes afterwards.

Code: `component/p2_som_training_visualization/som_nnsom_training/layer_ablation/`. Runner:
`component/pipeline/p2_som_training_visualization/run_feature_layer_ablation.py`. Tests:
`tests/p2_som_training_visualization/som_nnsom_training/layer_ablation/`.

## 1. Why

The project proposal's paper outline (Phase 5, Week 13, "Experiments") asks for an ablation of the
grid size and of the **feature layer choice**. The grid-size ablation is done (`GRID_COMPARISON_RESULTS.md`,
`OPTUNA_TUNING_PROTOCOL.md`). Every result of Phases 2 to 4 uses one layer: `fc2`, the 84-number layer
just before the classifier. This step asks whether the SOM's view of the network depends on that choice.

It also tests one explanation of the project's main finding. On MNIST the SOM does not find mistakes better
than the network's own confidence (max-softmax). `fc2` feeds the classifier directly, so it may carry little
that the softmax does not. The earlier layer `fc1` (120 numbers) is one step further from the classifier.

## 2. Fixed choices

- **Network:** the Phase 1 LeNet-5 (`lenet5_best.pth`, SHA-256 from `configs/phase3_shared_provenance.yaml`).
  Nothing is retrained.
- **Layers:** `fc1` (120 numbers) and `fc2` (84 numbers), read as the output of the linear layer before the
  ReLU, exactly as Phase 1 read `fc2`.
- **Images:** the frozen training and validation images (rebuilt from the saved sample IDs) and the SVHN test
  set (Phase 3 transform). The MNIST test split is not read.
- **Reproduction check:** the `fc2` embeddings extracted here must equal the Phase 1 embeddings (largest
  absolute difference at most 1e-4) and give the same validation predictions. Otherwise the run stops.
- **SOM:** the Phase 2 recipe for both layers: 15x15 grid, starting neighbourhood 11, 250 epochs (250 steps),
  inputs scaled to [-1, 1] by a min-max scaler fitted on that layer's training embeddings only.
- **SOM seeds:** 1, 2, 3, 4, 5 for both layers (seed 42 is the headline map and is not reused). Seed *s* of
  `fc1` is paired with seed *s* of `fc2`.
- **The headline layer stays `fc2` whatever the result.** No Phase 2 to 4 result is re-run or replaced.

## 3. Measures (per layer and SOM seed)

All on validation images, except A3.

| Code | Measure | How |
|---|---|---|
| A1 | Error-cell AUROC | The Phase 2 measure: each cell's error rate is learned on one random half of the validation images (stratified by error) and scores the other half, in both directions; mean over 200 half-splits (seeds `split_seed_base + repeat` from `som.yaml`, as in `error_pattern_baselines.yaml`). |
| A2 | Weighted purity | Majority-vote purity of the validation images (Phase 2, Week 4). |
| A3 | QE novelty AUROC | QE of MNIST validation images (normal) against SVHN test images (novel). |

Also reported, not judged: hotspot error share and hotspot error rate (the one hotspot rule, 11 hotspots),
first-order topographic error, occupied neurons, and the quantization error itself (its scale depends on the
layer, so it is not compared between layers).

## 4. Statements

For each of A1 to A3, the difference `fc1 − fc2` is taken per SOM seed; its mean and 95% t-interval
(4 degrees of freedom) give one of three verdicts:

| Verdict | Rule |
|---|---|
| fc1 better | lower end of the interval above 0 |
| fc2 better | upper end of the interval below 0 |
| inconclusive | the interval contains 0 |

- **L1 (primary):** A1, error-cell AUROC.
- **L2:** A2, weighted purity.
- **L3:** A3, QE novelty AUROC against SVHN.

How the outcomes are read (fixed now):

- L1 "fc1 better": the layer choice matters for finding mistakes; reported as a finding and as a
  recommendation for the Fashion-MNIST study. The headline layer stays `fc2`.
- L1 "inconclusive": no difference between the layers can be shown with 5 SOM seeds; the Phase 2 result
  does not measurably depend on the layer.
- L1 "fc2 better": supports the choice of `fc2`.
- The same reading applies to L2 and L3 for purity and novelty detection.

## 5. Context (not judged)

Max-softmax does not depend on the SOM layer, so it is not re-measured. Its Phase 2 error AUROC (0.981) and
Phase 3 SVHN AUROC (0.9985) are quoted next to the table for context.

## 6. Limitations

- The SOM recipe (neighbourhood 11, 250 epochs) was tuned on `fc2`. Using it for `fc1` may favour `fc2`.
  This is declared now; `fc1` is not tuned separately, so both layers get the same budget.
- One network: the comparison shows how the layers of this LeNet-5 differ, not of LeNet-5 networks in general.
- Five SOM seeds can only separate fairly large differences; the interval widths are reported.
- Only 92 validation mistakes, so A1 is noisy (as in Phase 2).

## 7. Leakage control

Training embeddings only for the scalers and the SOMs; validation images for A1 and A2; SVHN test and MNIST
validation for A3. Nothing is tuned on any measure. The MNIST test split is not read.

## 8. Recorded outcome

[TBD: filled in after the run, from `tables/`; sections 1 to 7 stay unchanged.]
