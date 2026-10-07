# RQ2 robustness results (MNIST)

Criteria: `RQ2_ROBUSTNESS_CRITERIA.md`, committed in `d9604e0` (2026-10-06 22:12 EDT) before any robustness run (first run 22:14 EDT).
Code: `rq2_robustness.py`, `rq2_robustness_verdict.py`. Raw outputs: `outputs/v1_mnist/som/analysis/*_rq2_robustness.json`, `rq2_robustness_verdict.json`.
Data: validation split (10,500 images, 92 CNN errors). Test split not used. Each model's SHA-256 was checked against `run_manifest.csv` before analysis.

## Verdict

| Model | Role | Claim A (hotspots) | Claim B (overlap regions) | Verdict |
|---|---|---|---|---|
| `som_15x15_nb11_ep250_s42_v2` | Headline | holds | holds | main claims established |
| `som_20x20_nb15_ep250_s42_v2` | Robustness (protocol-selected grid) | holds | holds | **RQ2 conclusion holds** |
| `som_10x10_nb7_ep250_s42_v2` | Robustness (rule's pick on surviving files) | holds | holds | **RQ2 conclusion holds** |

## Claim A: errors are concentrated in hotspots (held out)

Hotspots chosen on one half of the validation images, measured on the other; 100 splits x 2 directions. Ratio = hotspots' share of errors / their share of images (1 = no concentration). Chance baseline: 200 shuffles of the error labels.

| Grid | Hotspots (5% of neurons) | Median ratio | 2.5 to 97.5 percentile | Chance: median / 95th pct | p |
|---|---|---|---|---|---|
| 15x15 | 11 | 11.13 | 6.15 to 17.58 | 0.94 / 1.75 | 0.005 |
| 20x20 | 20 | 13.39 | 7.96 to 19.27 | 0.93 / 1.69 | 0.005 |
| 10x10 | 5 | 10.48 | 5.43 to 17.26 | 0.98 / 1.85 | 0.005 |

On every grid, about 5% of the map holds roughly 10 to 13 times more of the held-out errors than its share of images; random placement gives a ratio near 1.

## Claim B: errors occur where classes overlap

Spearman correlation over occupied validation neurons, compared with the same shuffled-label baseline.

| Grid | Occupied neurons | Error rate vs purity (chance 5th pct) | p | Error rate vs entropy (chance 95th pct) | p |
|---|---|---|---|---|---|
| 15x15 | 225 | -0.810 (0.027) | 0.005 | +0.816 (-0.027) | 0.005 |
| 20x20 | 399 | -0.766 (0.012) | 0.005 | +0.768 (-0.012) | 0.005 |
| 10x10 | 100 | -0.755 (0.110) | 0.005 | +0.778 (-0.104) | 0.005 |

Neurons that mix several digit classes have much higher CNN error rates, on every grid.

## Secondary: distance to the best-matching neuron (for novelty detection)

| Grid | Errors: mean / median | Correct: mean / median |
|---|---|---|
| 15x15 | 1.349 / 1.348 | 0.960 / 0.927 |
| 20x20 | 1.279 / 1.258 | 0.895 / 0.864 |
| 10x10 | 1.481 / 1.481 | 1.078 / 1.042 |

Misclassified images sit farther from their prototype on every grid (not part of the verdict).

## Notes for reading these numbers

- p = 0.005 is the smallest value possible with 200 shuffles (1/201): no shuffle came close to the observed values.
- With 92 errors, the ratio ranges are wide; the conclusion rests on every grid being far above chance, not on the exact ratio.
- The grids' ratios are not compared with each other here; the grid comparison is in `GRID_COMPARISON_RESULTS.md`.

## Paper sentence

> Using criteria fixed in advance (RQ2_ROBUSTNESS_CRITERIA.md), both RQ2 findings on the 15x15 headline SOM, that CNN errors concentrate in a small set of hotspot neurons and that error rates rise in neurons where classes overlap, also hold on the 20x20 SOM originally selected by the grid protocol and on a 10x10 SOM (all p = 0.005 against a shuffled-error baseline; held-out hotspot concentration 10.5 to 13.4 times the image share).
