# Plan: is the SOM's error-geography result special? (error pattern baselines)

Phase 2 (SOM training and visualization), step: error pattern analysis.
Written 2026-10-14, before any of this code was run. It adds a check next to the existing results and changes none of them.
Nothing in `src/` or in the existing `outputs/` folders is modified. Validation split only; the test split is never loaded.

## Questions (from the instructor review of the error pattern analysis)
1. The SOM is trained on the CNN's own embeddings, so the hotspot result may be partly circular. Does any partition of the same embeddings give it? Test: k-means with as many clusters as the SOM has neurons, and a kNN impurity score.
2. As a detector of CNN mistakes, how does the SOM cell error rate compare with the CNN's own confidence (max-softmax probability)?

## Inputs (frozen, read-only)
- LeNet-5 fc2 embeddings, labels, predictions and confidence of the training and validation splits (Phase 1 output, `cnn_baseline_training/embeddings/`). Hashes are checked against `expected_inputs` in the SOM config.
- The headline SOM (selected in the SOM config) and its best-matching cells for the validation images (`cluster_analysis/<model>/assignments/val_bmu.npy`). The model's SHA-256 is checked against the run manifest.

## Hotspot concentration (the project's rule, unchanged)
- Hotspots, held-out ratio, 100 stratified half-splits x 2 directions, 200 shuffled-error baselines: all settings and code come from the SOM config and `error_pattern_robustness.analyse`, so the SOM and k-means are judged identically.
- The SOM numbers must reproduce the project's own robustness result exactly (the run stops otherwise).
- k-means: same min-max scaling as the SOM (fitted on training embeddings), fitted on the training embeddings, 10 seeds, one cell per cluster; validation images take their nearest centroid. Reported as the median over seeds with the lowest to highest seed.

## Error detection (AUROC, average precision)
- 200 random stratified half-splits (same halves for every method). Cell scores (SOM cell, k-means cell) are an error rate fitted on one half and used on the other, both directions averaged.
- kNN impurity (k = 15): 1 - largest class share among the 15 nearest training embeddings; no label of the validation image is used.
- Max-softmax: 1 - the CNN's top class probability.
- Reported with the 2.5 to 97.5 percentile range over the repeats and the paired difference to the SOM.

## What is reported whatever the outcome
If k-means matches the SOM, the SOM's contribution is the 2-D map, not better detection. If max-softmax beats the SOM cell score, we say so.

## Amendments and deviations
- 2026-10-14, before any k-means result was seen: k-means uses `n_init: 3` per seed instead of 10 (10 seeds kept), because the machine used limits each command to a few minutes. The SOM reproduction check had already passed exactly.
- 2026-10-14, after the first results were seen (exploratory, not in the plan above): (a) the share of images and of errors inside the hotspot cells, because the hotspot ratio is errors' share divided by images' share and smaller hotspots give larger ratios; (b) bootstrap 95% intervals (2000 resamples, seed 2026) for the AUROC of kNN impurity and max-softmax. Both are now part of the pipeline and always computed, but they were not chosen in advance and are labelled exploratory in the report.
- 2026-10-15: code rewritten to follow the project rules (config file, modules split by responsibility, logging, type hints, tests, run ID without RQ or week numbers). Same settings and seeds; the results were recomputed under a new run ID (`..._r1`) and are identical to the first run to the last digit.
- 2026-10-18: re-run on the pinned environment (Python 3.13, numpy 2.4.4, scipy 1.18.0, scikit-learn 1.9.0). The SOM reproduction check was exact. Only the k-means row moved slightly (AUROC 0.693 vs 0.684, hotspot ratio 5.6 vs 5.3); the conclusions are unchanged. The run folder is `error_pattern_baselines_som_15x15` (shortened name; the first run has no revision suffix, a new run sets `run.revision`).
- Only the headline 15x15 SOM was analysed. The robustness SOMs' validation cells were not saved by the cluster analysis, so the same check on 10x10 and 20x20 needs their cells computed first.

## Limitations to carry into the report
- 92 validation errors: intervals are wide; differences below about 0.1 AUROC cannot be detected.
- The CNN checkpoint was chosen on the same validation split.
- MNIST only; the checks do not depend on the dataset and can be repeated on Fashion-MNIST and CIFAR-10.
