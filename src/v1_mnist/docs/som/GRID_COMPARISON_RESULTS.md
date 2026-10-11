# Results: fair test of 15x15 vs 20x20 SOM grids

Run on 2026-10-06, following the rules in `PREREGISTRATION_grid_comparison.md`.
Those rules were fingerprinted in `prereg_stamp.txt` (17:20:29 UTC) before the first training run started (17:23:06 UTC). The test split was never used.

## Short answer

**Neither grid is better.**
We asked: if we know each cell's error rate, how well can we predict which images the CNN gets wrong, on images that were not used to work out those rates?

- 20x20: AUROC **0.630**
- 15x15: AUROC **0.629**
- Difference (20x20 minus 15x15): **+0.001**, 95% range **−0.100 to +0.089**

The range includes zero, so neither grid wins. The range is also too wide to say the grids are "the same" (it would need to fit inside ±0.02). The reason is that the CNN made only **92 mistakes** out of 10,500 validation images, which is too few to detect small differences.

**What this means for the project:** there is no evidence that 20x20 is better for RQ2 (finding where the CNN makes mistakes). Using 15x15, as the instructor requested, is supported by the data.

## How the test worked

1. **Equal tuning.** Each grid tried three neighbourhood sizes (the same fractions of its side length), each trained for 250 epochs. Within each grid, one setting was picked using the original protocol's ranking (QE, TE1, TE1+2). No labels were used to pick.

   | | 10x10 | 15x15 | 20x20 |
   |---|---|---|---|
   | Settings tried | 2, 5, 7 | 3, 7, 11 | 4, 9, 15 |
   | Setting chosen | **7** | **11** | **15** |

2. **Training variation.** Each grid trained 10 maps on resampled training data, using the same resamples for every grid.
3. **Scoring.** 200 times over: split the validation images into two halves. Work out each cell's error rate on one half, then use those rates to predict the mistakes in the other half (AUROC). Swap the halves and average.
4. **Decision rule, fixed in advance:**
   - the whole range is inside ±0.02: the grids are the same;
   - the whole range is above or below 0: one grid is better;
   - otherwise: inconclusive.

All 39 runs used new run IDs; none was repeated or overwritten. (The batch stopped on a time limit after map 8, so the three map-9 runs were started separately.)

## Results

**Comparisons** (main one in bold):

| Comparison | Measure | Difference | 95% range | Bigger grid better in | Verdict |
|---|---|---|---|---|---|
| **20x20 vs 15x15** | **AUROC, CNN mistakes** | **+0.001** | **−0.100 to +0.089** | 52% of repeats | **inconclusive** |
| 20x20 vs 15x15 | Average precision, CNN mistakes | +0.002 | −0.051 to +0.063 | 52% | inconclusive |
| 20x20 vs 15x15 | AUROC, low-confidence images | −0.031 | −0.071 to +0.013 | 7.5% | inconclusive |
| 20x20 vs 15x15 | Average precision, low-confidence | −0.008 | −0.088 to +0.079 | 42.5% | inconclusive |
| 15x15 vs 10x10 | AUROC, CNN mistakes | −0.065 | −0.182 to +0.054 | 14% | inconclusive |
| 15x15 vs 10x10 | AUROC, low-confidence images | −0.027 | −0.073 to +0.019 | 9.5% | inconclusive |

"Low-confidence images" are the 525 images (5%) the CNN was least sure about. They give many more examples than the 92 mistakes.

**AUROC for each grid** (average, with the 95% range over 200 repeats):

| | 10x10 | 15x15 | 20x20 |
|---|---|---|---|
| CNN mistakes (92) | 0.694 (0.582 to 0.799) | 0.629 (0.548 to 0.725) | 0.630 (0.543 to 0.707) |
| Low-confidence images (525) | 0.878 (0.844 to 0.908) | 0.850 (0.817 to 0.878) | 0.820 (0.780 to 0.857) |

**Map quality** (average ± spread over 10 maps, full validation set):

| | 10x10 | 15x15 | 20x20 |
|---|---|---|---|
| QE (NNSOM definition) | 1.210 ± 0.010 | 1.061 ± 0.005 | 0.979 ± 0.004 |
| QE per image | 1.098 ± 0.005 | 0.977 ± 0.003 | 0.910 ± 0.003 |
| TE1 (%) | 4.82 ± 0.46 | 6.56 ± 0.77 | 7.66 ± 0.51 |
| TE1+2 (%) | 2.00 ± 0.38 | 3.62 ± 0.45 | 5.04 ± 0.35 |
| Cells used | 100% | 100% | 99.6% |
| Cell purity (%) | 97.57 ± 0.17 | 98.21 ± 0.15 | 98.36 ± 0.05 |
| Neighbours with a different main digit (%) | 32.1 ± 0.8 | 20.9 ± 0.6 | 15.8 ± 0.2 |

A bigger grid gives a sharper map (closer fit, purer cells, cleaner borders) but more topological error. That is expected when there are more cells.

## What this means

1. **A sharper map is not better at finding mistakes.** 20x20 looks better on QE and purity, but it does not predict the CNN's mistakes any better on new images. An earlier figure suggested it did (75% of mistakes in its top 10% of cells, vs 52% for 15x15). That figure used the same images to choose the cells and to score them, so it rewarded luck.
2. **The data, not the method, limits the test.** With 92 mistakes, differences smaller than about 0.1 AUROC can't be detected. Training more maps would barely help; only more CNN mistakes would (for example, a harder dataset).
3. **Observation, not a finding: smaller grids leaned slightly better.** Smaller grids tended to predict CNN errors slightly better on held-out data, likely because each cell holds more images. The differences were not statistically clear, and the RQ2 findings were the same on all three grids (`RQ2_ROBUSTNESS_RESULTS.md`).
   The comparisons below were added after the results, from the same saved scores, and were **not planned in advance**:

   | Comparison | Measure | Difference | 95% range | Verdict |
   |---|---|---|---|---|
   | 20x20 vs 10x10 | AUROC, CNN mistakes | −0.064 | −0.188 to +0.050 | inconclusive |
   | 20x20 vs 10x10 | Average precision, CNN mistakes | −0.003 | −0.068 to +0.055 | inconclusive |
   | 20x20 vs 10x10 | AUROC, low-confidence images | −0.058 | −0.102 to −0.010 | 10x10 better |
   | 20x20 vs 10x10 | Average precision, low-confidence | −0.012 | −0.104 to +0.074 | inconclusive |
   | 15x15 vs 10x10 | Average precision, CNN mistakes | −0.005 | −0.070 to +0.052 | inconclusive |
   | 15x15 vs 10x10 | Average precision, low-confidence | −0.004 | −0.087 to +0.080 | inconclusive |

   Only one of these is clear, and it is against 20x20, not 15x15. With many extra comparisons, one clear result can appear by chance, so it is not used to choose a grid.

## Sentence for the paper

> Under a pre-registered, equal-budget comparison, 20x20 and 15x15 SOMs did not differ in how well their cells predicted held-out CNN errors (AUROC 0.630 vs 0.629; difference +0.001, 95% interval −0.100 to +0.089). With 92 validation errors the comparison cannot distinguish differences smaller than about 0.1 AUROC. 20x20 produced lower quantization error and purer cells, while 15x15 had lower topological error. We therefore use 15x15, as requested by the instructor, and report RQ2 robustness on 20x20.

## Note on the official models (added 2026-10-06)

The official models were later retrained through the project pipeline (`train_som.py`) under new run IDs ending in `_v2`.
- 15x15 and 10x10 match this test's runs exactly.
- 20x20 differs very slightly (pipeline val QE 0.969655 and TE1 6.2762%, vs 0.967245 and 6.3905% here). The fast trainer used in this test drifts a little on 20x20 over 250 epochs, and is exact on the other grids. The pipeline value is the reference. This does not change the result, because the rules specified the fast trainer for every grid.

## Files

- In this folder: the rules (`PREREGISTRATION_grid_comparison.md`, `prereg_stamp.txt`; unchanged, the fingerprint still matches), this report, and the chart (`figures/grid_comparison_results.svg`).
- In the working folder `Codes/mnist_codes/week_7_codes_local/grid_comparison/`: the run list (`run_manifest.csv`, all 39 runs with model fingerprints), the trained maps (`models/*.npz`), every repeat's scores (`endpoint_reps.csv`, `results.json`), and the code (`fast_som.py`, `evaluate.py`, `train_run.py`, `choose.py`, `analyze.py`, `figure.py`).
- To reproduce: set the embeddings path at the top of `evaluate.py`, then run `train_run.py tune` (runs listed in `stage1.txt`), `choose.py`, `train_run.py boot` (`stage2.txt`), `analyze.py`, `figure.py`.
