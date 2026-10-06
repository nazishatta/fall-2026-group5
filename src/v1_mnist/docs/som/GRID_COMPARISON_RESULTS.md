# Results: 15x15 vs 20x20 SOM grid comparison (pre-registered)

Run on 2026-10-06. Rules: `PREREGISTRATION_grid_comparison.md` (SHA-256 `69496ffc…73c2`, stamped 17:20:29 UTC).
The first training run started at 17:23:06 UTC, after the stamp (see `run_manifest.csv`). The test split was never loaded.

---

## 1. Answer in one paragraph

**Primary result: inconclusive. On average the two grids are the same.**
When a map's cells are used to predict which validation images the CNN gets wrong, on images the map did not use to estimate error rates, 20x20 scores AUROC 0.630 and 15x15 scores 0.629.
The pre-registered difference (20x20 minus 15x15) is **+0.001**, with a 95% interval of **[-0.100, +0.089]**.
That interval is too wide to call the grids "practically equivalent" (it would have to fit inside ±0.02), and it is centred on zero, so neither grid is better.
The cause is the small number of CNN mistakes: only **92 errors** among 10,500 validation images.

**What this means for the project:** the evidence does **not** show that 20x20 is better for the project's purpose (RQ2, locating the CNN's mistakes on the map). Using 15x15, as the instructor requested, is not contradicted by the data.

---

## 2. Setup (as pre-registered, no deviations)

| | 10x10 | 15x15 | 20x20 |
|---|---|---|---|
| Neighbourhood candidates (equal budget) | 2, 5, 7 | 3, 7, 11 | 4, 9, 15 |
| Chosen (within-grid rank sum, no labels used) | **7** | **11** | **15** |
| Epochs | 250 | 250 | 250 |
| Bootstrap maps | 10 | 10 | 10 |

Primary measure: held-out, cross-fitted AUROC for predicting CNN errors from cell error rates, over 200 repeated stratified split-halves with bootstrap resampling. Same splits and same bootstrap resamples for every grid (paired).

Operational note: the Stage 2 batch stopped on a 10-minute command time limit after resample b8. The three b9 runs were then started separately. No run was repeated or overwritten (39 unique run IDs, 39 manifest rows).

---

## 3. Primary and secondary results

| Comparison | Measure | Mean difference | 95% interval | Bigger grid better in | Verdict |
|---|---|---|---|---|---|
| **20x20 vs 15x15** | **AUROC, CNN errors (primary)** | **+0.001** | **[-0.100, +0.089]** | 52% of repeats | **inconclusive** |
| 20x20 vs 15x15 | Average precision, CNN errors | +0.002 | [-0.051, +0.063] | 52% | inconclusive |
| 20x20 vs 15x15 | AUROC, low-confidence images | -0.031 | [-0.071, +0.013] | 7.5% | inconclusive |
| 20x20 vs 15x15 | Average precision, low-confidence | -0.008 | [-0.088, +0.079] | 42.5% | inconclusive |
| 15x15 vs 10x10 | AUROC, CNN errors | -0.065 | [-0.182, +0.054] | 14% | inconclusive |
| 15x15 vs 10x10 | AUROC, low-confidence images | -0.027 | [-0.073, +0.019] | 9.5% | inconclusive |

Held-out AUROC per grid (mean, 95% range over 200 repeats):

| | 10x10 | 15x15 | 20x20 |
|---|---|---|---|
| CNN errors (92) | 0.694 [0.582, 0.799] | 0.629 [0.548, 0.725] | 0.630 [0.543, 0.707] |
| Low-confidence images (525) | 0.878 [0.844, 0.908] | 0.850 [0.817, 0.878] | 0.820 [0.780, 0.857] |

## 4. Map-quality metrics (mean ± SD over 10 bootstrap maps, full validation set)

| | 10x10 | 15x15 | 20x20 |
|---|---|---|---|
| QE (NNSOM definition) | 1.210 ± 0.010 | 1.061 ± 0.005 | 0.979 ± 0.004 |
| QE per image | 1.098 ± 0.005 | 0.977 ± 0.003 | 0.910 ± 0.003 |
| TE1 (%) | 4.82 ± 0.46 | 6.56 ± 0.77 | 7.66 ± 0.51 |
| TE1+2 (%) | 2.00 ± 0.38 | 3.62 ± 0.45 | 5.04 ± 0.35 |
| Occupancy | 1.000 | 1.000 | 0.996 |
| Class purity (%) | 97.57 ± 0.17 | 98.21 ± 0.15 | 98.36 ± 0.05 |
| Neighbour disagreement (%) | 32.1 ± 0.8 | 20.9 ± 0.6 | 15.8 ± 0.2 |

The bigger grid is a sharper map (lower QE, purer cells, cleaner borders) but has more topological error. These are the expected effects of more cells.

---

## 5. How to read this

1. **A sharper map does not mean better error localisation.** 20x20 wins on QE, purity and border sharpness, but it does not predict the CNN's mistakes any better on held-out images.
   Earlier, 20x20 looked better at concentrating errors (75% of errors in its top-10% cells vs 52% for 15x15). That figure was measured on the same images used to choose the cells, so it rewarded noise; it does not hold up on held-out data.
2. **The comparison is limited by the data, not the method.** With 92 errors, a difference smaller than about 0.1 AUROC cannot be detected. Training more maps would barely help, because most of the uncertainty comes from the small number of errors; only more CNN errors would fix it (for example, a harder dataset or a weaker CNN checkpoint).
3. **Exploratory observation (not pre-registered, do not treat as a finding):** on both targets the held-out score falls slightly as the grid gets finer (10x10 > 15x15 ≥ 20x20). With fewer images per cell, finer maps estimate cell error rates less reliably. The low-confidence measure leans this way most clearly (20x20 better in only 7.5% of repeats), but it is still inconclusive.

## 6. Suggested wording for the protocol / paper (if 15x15 stays the headline grid)

> Under a pre-registered, equal-budget comparison, 20x20 and 15x15 SOMs did not differ in how well their cells predicted held-out CNN errors (AUROC 0.630 vs 0.629; difference +0.001, 95% interval −0.100 to +0.089). With 92 validation errors the comparison cannot distinguish differences smaller than about 0.1 AUROC. 20x20 produced lower quantization error and purer cells, while 15x15 had lower topological error. We therefore use 15x15, as requested by the instructor, and report RQ2 robustness on 20x20.

---

## Files

In this folder (`src/v1_mnist/docs/som/`):

| File | Contents |
|---|---|
| `PREREGISTRATION_grid_comparison.md`, `prereg_stamp.txt` | Rules and their fingerprint, written before any run. Copied unchanged from the working folder; the SHA-256 still matches `prereg_stamp.txt`. The "Location" line inside the rules refers to that working folder. |
| `GRID_COMPARISON_RESULTS.md` | This report |
| `figures/grid_comparison_results.png`, `.svg` | Figure |

Kept in the working folder `Codes/mnist_codes/week_7_codes_local/grid_comparison/` until their place in the repo is decided:

| File | Contents |
|---|---|
| `run_manifest.csv` | All 39 runs: run ID, settings, resample hash, model SHA-256, time, map metrics |
| `models/*.npz` | Weights + scaler for every map |
| `endpoint_reps.csv`, `results.json` | Every repeat's scores, and all summary numbers in this report |
| `fast_som.py`, `evaluate.py`, `train_run.py`, `choose.py`, `analyze.py`, `figure.py`, `stage1.txt`, `stage2.txt` | Code and run lists to reproduce everything |

Reproduce: `python train_run.py tune ...` (Stage 1, see `stage1.txt`), `python choose.py`, `python train_run.py boot ...` (Stage 2, see `stage2.txt`), `python analyze.py`, `python figure.py`.
Before running, change the embeddings path at the top of `evaluate.py` to your copy.

---

## Note added 2026-10-06 (after the official models were trained)

The official models were retrained through the project pipeline (`train_som.py`) on EC2 under new run IDs (`*_v2`).
- 15x15 (nb 11) and 10x10 (nb 7) reproduce this comparison's Stage 1 runs exactly.
- 20x20 (nb 15) differs slightly: pipeline val QE 0.969655, TE1 6.2762% vs 0.967245, 6.3905% here. An independent rerun of the pipeline gave the same 0.969655, so the pipeline value is the reference. The fast re-implementation used in this comparison drifts slightly from the pipeline on the 20x20 grid (small floating-point differences that grow over 250 epochs); it is exact on 15x15 and 10x10.
- This does not change the comparison: the pre-registration specified the fast trainer for every run, and all grids were trained with it. The Stage 1 choice of nb 15 for 20x20 was made under those pre-registered rules and stands.
