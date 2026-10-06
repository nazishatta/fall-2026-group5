# Pre-registration: 15x15 vs 20x20 SOM grid comparison (with 10x10 for context)

Written: 2026-10-06, before any run described here.
Location: `Codes/mnist_codes/week_7_codes_local/grid_comparison/`
The SHA-256 of this file and the time it was written are recorded in `prereg_stamp.txt` **before** training starts. Commit both files to git before reading the results.

---

## 1. Question

For the purpose this SOM serves in the project (RQ2: where on the map does the CNN make its mistakes?), is a 20x20 map better than a 15x15 map, worse, or practically the same?

"Better" is defined as: **the map's cells predict which validation images the CNN gets wrong more accurately, on images the map never used to estimate those error rates.**

## 2. Disclosure of what was seen before this was written

Before writing this, exploratory single-seed runs (10x10, 15x15, 20x20) were inspected: QE, topological error, purity, neighbour disagreement, class maps, U-matrices, and the share of validation errors in the top-10% error-rate cells measured **on the same images used to pick the cells**.
The primary endpoint below (held-out, cross-fitted error-prediction AUROC) **has not been computed for any grid**.

## 3. Fixed inputs

- Frozen LeNet-5 fc2 embeddings (84-dim), the same files used by the original grid study:
  `train_features.npy` md5 `4d0a0f50371a1aebea82bb3dbe34d4c2`, `val_features.npy` md5 `4e7c0421ef00a691b98c1bc16e3345ab` (SHA-256 of all inputs in `prereg_stamp.txt`).
- Validation split: 10,500 images, 92 CNN errors. **The test split is not loaded or used.**
- Preprocessing: MinMax scaler to [-1, 1], fitted on the training data used for that map only.
- Trainer: batch SOM identical to the project's `train_som_with_history()` (bubble neighbourhood, linear radius decay `1 + (nb-1)(1 - step/steps)`, 90% winner retention, NNSOM 1.8.3 PCA initialisation with economy SVD). The fast re-implementation `fast_som.py` reproduces the saved project model exactly (seed 42, 15x15, nb 11: val QE 1.0412, TE1 6.18%).
- `epochs = steps = 250` for every grid.

## 4. Stage 1: equal tuning budget (labels not used)

Each grid gets the same three neighbourhood settings, as fractions of its side length (0.20, 0.47, 0.73, rounded):

| Grid | init_neighborhood candidates |
|---|---|
| 10x10 | 2, 5, 7 |
| 15x15 | 3, 7, 11 |
| 20x20 | 4, 9, 15 |

Trained on the full training set, seed 42, exact RNG mode.
Within each grid, the setting is chosen by the original protocol's equal-weight rank sum of validation QE (NNSOM definition), TE1 and TE1+2. These use validation **features only**, never labels or CNN correctness. Ties go to the smaller neighbourhood.
(Rank sums are used only within a grid, where the size bias of these metrics does not apply.)

## 5. Stage 2: training variability

For each grid's chosen setting, train **B = 10** maps on bootstrap resamples of the training set (49,000 draws with replacement).
- Resample b uses `numpy.random.default_rng(1000 + b)` to draw indices. **The same indices are used for all three grids** (paired design).
- The trainer seed for resample b is b; fast RNG mode (statistically identical to exact mode).
- Scaler refitted on each resample.

## 6. Primary endpoint and comparison

Repeat R = 200 times (rep r = 0..199, `default_rng(5000 + r)`):
1. Split the validation images into halves A and B, **stratified by CNN correct/error** (46 errors per half).
2. Within each half, draw a bootstrap resample of that half's images (captures sampling uncertainty; the halves never share an image).
3. For each grid, use bootstrap map `b = r mod 10` (the same b for every grid).
4. **A to B:** cell error rate = errors / images per cell in A*; cells with no A* images get A*'s overall error rate. Score each image in B* by its best-matching cell's rate. Compute AUROC for predicting "CNN wrong". Then **B to A** the same way. AUROC_r = mean of the two directions.
5. Delta_r = AUROC_r(20x20) - AUROC_r(15x15).

**Estimate:** mean of Delta_r. **95% interval:** 2.5th to 97.5th percentile of Delta_r. This interval mixes val-sampling, split and training variability, so it is conservative.

**Margin of practical equivalence:** delta = 0.02 AUROC.

**Decision rule (applied in this order):**
1. Interval entirely inside [-0.02, +0.02]: **practically equivalent**.
2. Interval entirely above 0: **20x20 better**.
3. Interval entirely below 0: **15x15 better**.
4. Otherwise: **inconclusive** at this sample size (92 errors).

## 7. Secondary results (reported, not used for the decision)

- Same procedure with **average precision** instead of AUROC.
- Same procedure with a **low-confidence target**: an image is positive if its CNN confidence is in the lowest 5% of the validation set (threshold 0.99574; 525 images, 81 of them errors). This uses far more positives than the 92 errors.
- The same comparison for 10x10 vs 15x15.
- Descriptive map metrics per grid (mean and SD over the 10 bootstrap maps, full validation set): NNSOM QE, per-image QE, TE1, TE1+2, occupancy, class purity, neighbour disagreement.

## 8. What will be reported regardless of outcome

All four outcomes are reportable. If the result is "equivalent" or "inconclusive", the grid used for headline results is decided by the documented instructor request, not by these numbers.
No setting, margin, endpoint or rule in this document will be changed after results are seen; any additional analysis will be labelled exploratory.
