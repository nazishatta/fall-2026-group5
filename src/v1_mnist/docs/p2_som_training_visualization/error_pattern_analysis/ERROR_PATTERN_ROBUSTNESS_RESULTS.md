# Error-pattern robustness results (MNIST)

> Renamed on 2026-10-09 to use the project proposal's term (Week 5: Error Pattern Analysis) instead of "RQ2". Only names changed: this file was `RQ2_ROBUSTNESS_RESULTS.md`, the rules file was `RQ2_ROBUSTNESS_CRITERIA.md`, the code was `rq2_robustness.py` / `rq2_robustness_verdict.py`, and the raw outputs were `*_rq2_robustness.json` / `rq2_robustness_verdict.json`. The rules file itself is left word for word as committed, so it still says "RQ2". The numbers are unchanged.
>
> The two findings are named with the proposal's terms (Objective 4: "error hotspots" and "overlap regions"). The rules file calls them **Claim A** (= hotspot concentration) and **Claim B** (= overlap regions); the raw outputs used the keys `claim_a` / `claim_b` until 2026-10-09.

**Question:** our error-pattern findings come from the 15x15 map. Do they still hold on a 20x20 map and a 10x10 map?

- Rules: `ERROR_PATTERN_ROBUSTNESS_CRITERIA.md`, committed in `d9604e0` under its old name (2026-10-06, 22:12 EDT) before any check was run (first run 22:14 EDT).
- Code: `error_pattern_robustness.py`, `error_pattern_robustness_verdict.py`. Raw outputs: `outputs/v1_mnist/p2_som_training_visualization/error_pattern_analysis/robustness/*_error_pattern_robustness.json` and `error_pattern_robustness_verdict.json`.
- Data: validation split (10,500 images, 92 CNN mistakes). The test split was not used. Each model's fingerprint was checked against `run_manifest.csv` first.

## Verdict: the error-pattern findings hold on every grid

| Model | Role | Hotspot concentration | Overlap regions | Verdict |
|---|---|---|---|---|
| `som_15x15_nb11_ep250_s42_v2` | Main model | holds | holds | main findings confirmed |
| `som_20x20_nb15_ep250_s42_v2` | Check (grid the original protocol picked) | holds | holds | **holds** |
| `som_10x10_nb7_ep250_s42_v2` | Check (grid the rule picks on the surviving files) | holds | holds | **holds** |

Each finding is compared with a "chance" version: the same numbers recomputed 200 times after shuffling which images are mistakes.

## Hotspot concentration: mistakes cluster in a few "hotspot" cells

We pick the 5% of cells with the highest error rates using one half of the validation images, then measure on the other half. This is repeated over 100 splits, in both directions.
**Ratio** = the hotspots' share of mistakes ÷ their share of images. A ratio of 1 means no clustering.

| Grid | Hotspot cells | Median ratio | 95% range | Chance: median / 95th percentile | p |
|---|---|---|---|---|---|
| 15x15 | 11 | 11.13 | 6.15 to 17.58 | 0.94 / 1.75 | 0.005 |
| 20x20 | 20 | 13.39 | 7.96 to 19.27 | 0.93 / 1.69 | 0.005 |
| 10x10 | 5 | 10.48 | 5.43 to 17.26 | 0.98 / 1.85 | 0.005 |

On every grid, about 5% of the map holds 10 to 13 times its fair share of mistakes. By chance the ratio is about 1.

## Overlap regions: mistakes happen where digits mix

For each cell, we compare its error rate with how mixed its digits are (Spearman correlation).

| Grid | Cells used | Error rate vs purity (chance limit) | p | Error rate vs mixing (chance limit) | p |
|---|---|---|---|---|---|
| 15x15 | 225 | −0.810 (0.027) | 0.005 | +0.816 (−0.027) | 0.005 |
| 20x20 | 399 | −0.766 (0.012) | 0.005 | +0.768 (−0.012) | 0.005 |
| 10x10 | 100 | −0.755 (0.110) | 0.005 | +0.778 (−0.104) | 0.005 |

On every grid, cells that mix several digits have much higher CNN error rates.

## Extra: distance to the nearest cell (for later novelty-detection work, not part of the verdict)

| Grid | Mistakes: mean / median | Correct: mean / median |
|---|---|---|
| 15x15 | 1.349 / 1.348 | 0.960 / 0.927 |
| 20x20 | 1.279 / 1.258 | 0.895 / 0.864 |
| 10x10 | 1.481 / 1.481 | 1.078 / 1.042 |

On every grid, misclassified images sit farther from their best-matching cell.

## How to read these numbers

- p = 0.005 is the smallest value possible with 200 shuffles: no shuffle came close to the real result.
- With only 92 mistakes the ranges are wide. The conclusion rests on every grid being far above chance, not on the exact ratios.
- These ratios are not meant for comparing grids with each other; that comparison is in `../som_nnsom_training/GRID_COMPARISON_RESULTS.md` (Week 3).
- Observation, not a finding: in that comparison, smaller grids tended to predict CNN errors slightly better on held-out data, likely because each cell holds more images. The differences were not statistically clear, and the error-pattern findings here were the same on all three grids.

## Sentence for the paper

> Using criteria fixed in advance (ERROR_PATTERN_ROBUSTNESS_CRITERIA.md), both error-pattern findings on the 15x15 headline SOM, that CNN errors concentrate in a small set of hotspot neurons and that error rates rise in neurons where classes overlap, also hold on the 20x20 SOM originally selected by the grid protocol and on a 10x10 SOM (all p = 0.005 against a shuffled-error baseline; held-out hotspot concentration 10.5 to 13.4 times the image share).
